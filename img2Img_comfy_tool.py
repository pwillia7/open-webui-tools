"""
title: Img2Img Comfy Tool
Author: Patrick Williams
author_url: https://reticulated.net
git_url: https://github.com/username/comfyui-workflow-runner.git
description: Submits an img2img node-based ComfyUI workflow over HTTP, then listens on the WebSocket only to know when generation completes. Then retrieves final images from /history and optionally reorders them.
required_open_webui_version: 0.4.0
requirements: requests, websocket-client
version: 1.3.1
license: MIT
"""

import uuid
import json
import base64
import requests
import asyncio
import websockets
import logging
from typing import Dict, Any, Optional, Callable, Awaitable
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


class Tools:
    """
    - We do not receive images over WebSocket. We only detect completion (node=None or closure).
    - Then we gather final images from /history, reorder them if requested, and emit them.
    """

    class Valves(BaseModel):
        Api_Key: Optional[str] = Field(
            None,
            description="API token for ComfyUI (if using ComfyUI-Login). If not needed, leave blank.",
        )
        Server_Address: Optional[str] = Field(
            None, description="ComfyUI address, e.g. 'localhost:8443'."
        )
        debug_mode: bool = Field(default=False, description="Extra debug logs if True.")
        workflow_file_url: Optional[str] = Field(
            default=None,
            description="URL of a JSON workflow. If not set, we use an inline default.",
        )

        # New valve for reordering images by a comma-separated list of 1-based indices
        image_order_list: str = Field(
            default="",
            description=(
                "Comma-separated 1-based indexes to reorder final images. "
                "E.g. '3,1,4,2' to reorder images in that sequence. "
                "Out-of-range indices are skipped. If empty, use original order."
            ),
        )

        # UI messages
        connecting_to_comfyui_msg: str = Field(
            default="Connecting to ComfyUI...",
            description="Shown when we first attempt WebSocket connect.",
        )
        connected_submitting_workflow_msg: str = Field(
            default="Connected! Submitting workflow...",
            description="Shown once the socket is open & we post /prompt.",
        )
        workflow_submitted_msg: str = Field(
            default="Workflow submitted. Waiting until ComfyUI forcibly closes...",
            description="Shown after the /prompt call returns OK.",
        )
        comfyui_generation_complete_msg: str = Field(
            default="ComfyUI closed the connection or signaled completion; retrieving final images...",
            description="Shown once we detect the generation is done.",
        )
        workflow_submission_failed_msg: str = Field(
            default="Workflow submission failed: {exception}",
            description="Error if we cannot submit to /prompt.",
        )
        ws_connection_error_msg: str = Field(
            default="Could not connect to ComfyUI WebSocket: {exception}",
            description="Error if the WebSocket fails to connect or drops prematurely.",
        )
        original_image_msg: str = Field(
            default="**Original Image:** ![Original]({url})",
            description="Displayed to show the user's original image in chat.",
        )
        error_fetching_image_msg: str = Field(
            default="Error fetching image: {exception}",
            description="Error if we fail to download the user-provided image.",
        )
        no_history_msg: str = Field(
            default="No history found for this prompt.",
            description="If /history has no record of the promptId.",
        )
        no_images_workflow_msg: str = Field(
            default="No images found in workflow outputs.",
            description="If the final pipeline yields zero images in /history.",
        )
        error_retrieving_images_msg: str = Field(
            default="Error retrieving images: {exception}",
            description="If /history retrieval fails.",
        )
        workflow_completed_status_msg: str = Field(
            default="Workflow completed successfully. Retrieved {image_count} image(s).",
            description="Status message at the end of generation.",
        )
        workflow_completed_return_msg: str = Field(
            default="Workflow completed. Received {image_count} image(s).",
            description="Return value after final success.",
        )

    def __init__(self):
        self.valves = self.Valves()

        # Minimal inline default if no workflow_file_url is provided
        self.workflow_template: Dict[str, Any] = {
            "workflow": {"some_node": "default or empty workflow goes here"}
        }

    def _replace_placeholders(self, data: dict, placeholders: dict) -> dict:
        """Recursively replace placeholders like %%B64IMAGE%% or %%PROMPT%% in the workflow JSON."""
        for k, v in data.items():
            if isinstance(v, str):
                for ph, rp in placeholders.items():
                    v = v.replace(ph, rp)
                data[k] = v
            elif isinstance(v, dict):
                data[k] = self._replace_placeholders(v, placeholders)
            elif isinstance(v, list):
                new_list = []
                for item in v:
                    if isinstance(item, dict):
                        new_list.append(self._replace_placeholders(item, placeholders))
                    elif isinstance(item, str):
                        for ph, rp in placeholders.items():
                            item = item.replace(ph, rp)
                        new_list.append(item)
                    else:
                        new_list.append(item)
                data[k] = new_list
        return data

    def _queue_prompt(self, workflow: dict, client_id: str) -> str:
        """
        POST the workflow to /prompt with the same client_id used in the WebSocket.
        If self.valves.Api_Key is set, we append ?token=API_KEY to the URL.
        """
        if not self.valves.Server_Address:
            raise ValueError("No ComfyUI server address set in valves.")

        base_url = f"https://{self.valves.Server_Address}/prompt"
        if self.valves.Api_Key:
            base_url += f"?token={self.valves.Api_Key}"

        body = {"prompt": workflow, "client_id": client_id}

        if self.valves.debug_mode:
            logging.debug(f"Posting workflow to: {base_url}")

        resp = requests.post(base_url, json=body, timeout=30)
        resp.raise_for_status()
        return resp.json().get("prompt_id", "")

    async def run_comfyui_img2img_workflow(
        self,
        image_url: str,
        prompt_text: str = "",
        target_node: str = "%%B64IMAGE%%",
        max_returned_images: int = 5,
        __event_emitter__: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
    ) -> str:
        """
        Steps:
          1) Download the image
          2) Possibly fetch .workflow_file_url
          3) Insert placeholders
          4) WebSocket with ?clientId=...
          5) /prompt with that client_id
          6) Wait for 'node=None' or forced close => generation done
          7) /history to gather final images, reorder them (if requested)
          8) Return
        """
        if not self.valves.Server_Address:
            raise ValueError("No server address set in valves.")

        # 1) Download the user image
        try:
            resp = requests.get(image_url, timeout=30)
            resp.raise_for_status()
            b64_image = base64.b64encode(resp.content).decode("utf-8")
        except Exception as e:
            return self.valves.error_fetching_image_msg.format(exception=e)

        # Show the original image
        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "message",
                    "data": {
                        "content": self.valves.original_image_msg.format(url=image_url)
                    },
                }
            )

        # 2) Possibly fetch remote workflow
        if self.valves.workflow_file_url:
            try:
                w = requests.get(self.valves.workflow_file_url, timeout=30)
                w.raise_for_status()
                self.workflow_template = w.json()
            except Exception as e:
                return self.valves.error_retrieving_images_msg.format(exception=e)

        if not self.workflow_template:
            return "No workflow loaded; set a .workflow_file_url or define an inline default."

        # 3) Insert placeholders (%%B64IMAGE%%, %%PROMPT%%, etc.)
        import copy

        placeholders = {target_node: b64_image}
        if prompt_text:
            placeholders["%%PROMPT%%"] = prompt_text

        wf_copy = copy.deepcopy(self.workflow_template)
        updated_workflow = self._replace_placeholders(wf_copy, placeholders)

        # 4) Build the WS URL
        client_id = str(uuid.uuid4())
        ws_url = f"wss://{self.valves.Server_Address}/ws?clientId={client_id}"
        if self.valves.Api_Key:
            ws_url += f"&token={self.valves.Api_Key}"

        if self.valves.debug_mode:
            logging.debug(f"Connecting WebSocket to: {ws_url}")

        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": self.valves.connecting_to_comfyui_msg,
                        "done": False,
                    },
                }
            )

        prompt_id = None

        # 5) Connect to the WebSocket & submit the workflow
        try:
            async with websockets.connect(ws_url) as ws:
                # Indicate we connected
                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": self.valves.connected_submitting_workflow_msg,
                                "done": False,
                            },
                        }
                    )

                # Submit the prompt via /prompt
                try:
                    prompt_id = self._queue_prompt(updated_workflow, client_id)
                    if __event_emitter__:
                        await __event_emitter__(
                            {
                                "type": "status",
                                "data": {
                                    "description": self.valves.workflow_submitted_msg,
                                    "done": False,
                                },
                            }
                        )
                except Exception as e:
                    if __event_emitter__:
                        await __event_emitter__(
                            {
                                "type": "status",
                                "data": {
                                    "description": self.valves.workflow_submission_failed_msg.format(
                                        exception=e
                                    ),
                                    "done": True,
                                },
                            }
                        )
                    return self.valves.workflow_submission_failed_msg.format(
                        exception=e
                    )

                # 6) indefinite read until node=None or forced close => generation done
                generation_done = False
                while not generation_done:
                    try:
                        raw_msg = await ws.recv()
                    except websockets.ConnectionClosed:
                        # forcibly closed => done
                        break

                    if isinstance(raw_msg, str):
                        # It's JSON
                        msg_json = json.loads(raw_msg)
                        msg_type = msg_json.get("type", "")
                        data = msg_json.get("data", {})
                        # typical ComfyUI signals "executing" with "node": None => done
                        if (
                            msg_type == "executing"
                            and data.get("prompt_id") == prompt_id
                            and data.get("node") is None
                        ):
                            # generation done => break
                            generation_done = True
                    # ignoring any binary frames

        except Exception as e:
            logging.error(f"WebSocket error: {e}")
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": self.valves.ws_connection_error_msg.format(
                                exception=e
                            ),
                            "done": True,
                        },
                    }
                )
            return self.valves.ws_connection_error_msg.format(exception=e)

        # 7) Once the server forcibly closed or we saw node=None => fetch /history
        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": self.valves.comfyui_generation_complete_msg,
                        "done": False,
                    },
                }
            )

        if not prompt_id:
            return "No prompt_id found; cannot fetch final images."

        hist_url = f"https://{self.valves.Server_Address}/history/{prompt_id}"
        if self.valves.Api_Key:
            hist_url += f"?token={self.valves.Api_Key}"

        if self.valves.debug_mode:
            logging.debug(f"Fetching final images from: {hist_url}")

        try:
            r = requests.get(hist_url, timeout=30)
            r.raise_for_status()
            history_data = r.json()
        except Exception as e:
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": self.valves.error_retrieving_images_msg.format(
                                exception=e
                            ),
                            "done": True,
                        },
                    }
                )
            return self.valves.error_retrieving_images_msg.format(exception=e)

        if prompt_id not in history_data:
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": self.valves.no_history_msg,
                            "done": True,
                        },
                    }
                )
            return self.valves.no_history_msg

        outputs = history_data[prompt_id].get("outputs", {})
        if not outputs:
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": self.valves.no_images_workflow_msg,
                            "done": True,
                        },
                    }
                )
            return self.valves.no_images_workflow_msg

        # Gather the final images
        all_images = []
        for _, node_output in outputs.items():
            node_imgs = node_output.get("images", [])
            all_images.extend(node_imgs)

        # Reorder them if user provided a list
        order_list_str = self.valves.image_order_list.strip()
        if order_list_str:
            # Parse the comma-separated 1-based indices
            new_order = []
            for part in order_list_str.split(","):
                part = part.strip()
                if part.isdigit():
                    idx = int(part) - 1  # convert to 0-based
                    if 0 <= idx < len(all_images):
                        new_order.append(all_images[idx])
            # If new_order is not empty, we replace the entire list
            if new_order:
                all_images = new_order

        # Now apply max_returned_images
        if max_returned_images > 0:
            all_images = all_images[:max_returned_images]

        # Emit the final images
        image_count = 0
        for hist_img in all_images:
            fn = hist_img["filename"]
            sf = hist_img["subfolder"]
            ft = hist_img["type"]
            final_url = f"https://{self.valves.Server_Address}/view?filename={fn}&subfolder={sf}&type={ft}"
            if self.valves.Api_Key:
                final_url += f"&token={self.valves.Api_Key}"

            image_count += 1
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "message",
                        "data": {
                            "content": f"**Enhanced Image #{image_count}:** ![Preview]({final_url})"
                        },
                    }
                )

        # final status
        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": self.valves.workflow_completed_status_msg.format(
                            image_count=image_count
                        ),
                        "done": True,
                    },
                }
            )

        return self.valves.workflow_completed_return_msg.format(image_count=image_count)
