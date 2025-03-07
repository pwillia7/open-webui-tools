"""
title: Txt2Img Comfy Tool
Author: Patrick Williams
author_url: https://reticulated.net
git_url: https://github.com/pwillia7/open-webui-tools.git
description: Submits a txt2img node-based ComfyUI workflow over HTTPS by replacing the %%PROMPT%% placeholder with the user's input. It then listens on the WebSocket until generation completes and retrieves final images from /history.
required_open_webui_version: 0.4.0
requirements: requests, websocket-client
version: 1.0.0
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
    [LLM Usage Directive]:
    - **Purpose**: This tool generates images using a node-based ComfyUI workflow.
    - **When to Use**:
         e.g. "Generate an image of a futuristic city at sunset."
    - **How to Use**:
         Call run_comfyui_workflow(prompt_text=USER_PROMPT) directly.
    - **Error Handling**:
         Returns error messages verbatim.
    """

    # All configuration now lives under Valves (admin-only settings)
    class Valves(BaseModel):
        Api_Key: Optional[str] = Field(
            None,
            description="The API token for authenticating with ComfyUI. Must be set in the Open-WebUI admin interface.",
        )
        ComfyUI_Server: Optional[str] = Field(
            "ptkwilliams.ddns.net:8443",
            description="The address of the ComfyUI server, e.g. 'localhost:8188'. Must be set in Open-WebUI.",
        )
        Workflow_URL: Optional[str] = Field(
            "https://gist.githubusercontent.com/pwillia7/9fe756338c7d35eba130c68408b705f4/raw/4a429e1ede948e02e405e3a046b2eb85546f1c0f/fluxgen",
            description="The URL where the ComfyUI workflow JSON is hosted.",
        )
        debug_mode: bool = Field(
            default=False,
            description="Enable additional debug output for troubleshooting.",
        )

    def __init__(self) -> None:
        self.valves = self.Valves()

        if not self.valves.ComfyUI_Server:
            raise ValueError(
                "ComfyUI server address is not set in Valves. Please configure it in Open-WebUI."
            )
        # Use the exact value from the valve
        self.server_address = self.valves.ComfyUI_Server

        if not self.valves.Workflow_URL:
            raise ValueError(
                "Workflow URL is not set in Valves. Please configure it in Open-WebUI."
            )
        try:
            response = requests.get(self.valves.Workflow_URL, timeout=30)
            response.raise_for_status()
            self.workflow_template: Dict[str, Any] = response.json()
        except Exception as e:
            raise ValueError(
                f"Failed to load workflow from URL {self.valves.Workflow_URL}: {e}"
            )

    def _replace_placeholders(self, data: dict, placeholders: Dict[str, str]) -> dict:
        """
        Recursively replace '%%PLACEHOLDER%%' in string fields within a dict.
        """
        for key, value in data.items():
            if isinstance(value, str):
                for ph, replacement in placeholders.items():
                    value = value.replace(ph, replacement)
                data[key] = value
            elif isinstance(value, dict):
                data[key] = self._replace_placeholders(value, placeholders)
            elif isinstance(value, list):
                data[key] = [
                    (
                        self._replace_placeholders(item, placeholders)
                        if isinstance(item, dict)
                        else (
                            self._replace_string_in_list_item(item, placeholders)
                            if isinstance(item, str)
                            else item
                        )
                    )
                    for item in value
                ]
        return data

    def _replace_string_in_list_item(
        self, text: str, placeholders: Dict[str, str]
    ) -> str:
        for ph, replacement in placeholders.items():
            text = text.replace(ph, replacement)
        return text

    def _queue_prompt(self, workflow: dict, client_id: str) -> str:
        """
        Submit the workflow to the ComfyUI API.
        The API token is sent as a Bearer token in the Authorization header.
        """
        if not self.valves.Api_Key:
            raise ValueError(
                "API token is not set in Valves. Please configure it in Open-WebUI."
            )
        url = f"https://{self.server_address}/prompt"
        headers = {"Authorization": f"Bearer {self.valves.Api_Key}"}
        body = {"prompt": workflow, "client_id": client_id}

        logging.debug(f"Submitting API request to {url}")
        logging.debug(f"Request Headers: {headers}")
        logging.debug(f"Request Body: {json.dumps(body)}")

        resp = requests.post(url, json=body, headers=headers, timeout=30)
        logging.debug(f"Response Headers: {resp.headers}")
        logging.debug(f"Response Status Code: {resp.status_code}")
        logging.debug(f"Response Body: {resp.text}")

        resp.raise_for_status()
        return resp.json()["prompt_id"]

    async def run_comfyui_workflow(
        self,
        prompt_text: str,
        __event_emitter__: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
    ) -> str:
        """
        Execute the ComfyUI workflow by:
          1) Replacing %%PROMPT%% in the workflow JSON with the user's prompt.
          2) Submitting the prompt to /prompt.
          3) Listening on the WebSocket until generation completes.
          4) Fetching generated images from /history/<prompt_id>.
          5) Emitting them as chat messages.
        """
        if not self.valves.Api_Key:
            raise ValueError(
                "API token is not set in Valves. Please configure it in Open-WebUI."
            )

        import copy

        logging.debug(f"Using user prompt: {prompt_text}")

        workflow_copy = copy.deepcopy(self.workflow_template)
        updated_workflow = self._replace_placeholders(
            workflow_copy, {"%%PROMPT%%": prompt_text}
        )
        if not isinstance(updated_workflow, dict):
            raise TypeError("Workflow must be a dict after placeholders are replaced.")

        client_id = str(uuid.uuid4())

        # Build the WebSocket URL exactly as in the working tool.
        ws_url = f"wss://{self.valves.ComfyUI_Server}/ws?clientId={client_id}"
        if self.valves.Api_Key:
            ws_url += f"&token={self.valves.Api_Key}"

        logging.debug(f"Connecting WebSocket to: {ws_url}")
        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "Connecting to ComfyUI...", "done": False},
                }
            )

        try:
            async with websockets.connect(ws_url) as ws:
                logging.debug("WebSocket connection established.")
                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": "Connected! Submitting workflow...",
                                "done": False,
                            },
                        }
                    )

                try:
                    prompt_id = self._queue_prompt(updated_workflow, client_id)
                    logging.debug(f"Workflow submitted. prompt_id={prompt_id}")
                    if __event_emitter__:
                        await __event_emitter__(
                            {
                                "type": "status",
                                "data": {
                                    "description": "Workflow submitted. Waiting for generation...",
                                    "done": False,
                                },
                            }
                        )
                except Exception as e:
                    logging.error(f"Error submitting workflow: {e}")
                    if __event_emitter__:
                        await __event_emitter__(
                            {
                                "type": "status",
                                "data": {
                                    "description": f"Workflow submission failed: {e}",
                                    "done": True,
                                },
                            }
                        )
                    return f"Error submitting workflow: {e}"

                try:
                    async for raw_msg in ws:
                        if isinstance(raw_msg, bytes):
                            continue
                        message_data = json.loads(raw_msg)
                        msg_type = message_data.get("type", "")
                        msg_info = message_data.get("data", {})
                        if (
                            msg_type == "executing"
                            and msg_info.get("prompt_id") == prompt_id
                            and msg_info.get("node") is None
                        ):
                            logging.debug(
                                "ComfyUI signaled that generation is complete."
                            )
                            break
                except Exception as e:
                    logging.error(f"Error receiving WebSocket messages: {e}")
        except Exception as e:
            logging.error(f"WebSocket connection failed: {e}")
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"Could not connect to ComfyUI WebSocket: {e}",
                            "done": True,
                        },
                    }
                )
            return f"WebSocket connection error: {e}"

        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": "Generation complete. Retrieving images...",
                        "done": False,
                    },
                }
            )

        history_url = f"https://{self.valves.ComfyUI_Server}/history/{prompt_id}?token={self.valves.Api_Key}"
        try:
            resp = requests.get(history_url, timeout=30)
            resp.raise_for_status()
            history_data = resp.json()
            logging.debug(f"History data retrieved: {history_data}")
        except Exception as e:
            logging.error(f"Error fetching /history data: {e}")
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"Error retrieving images: {e}",
                            "done": True,
                        },
                    }
                )
            return f"Error retrieving images: {e}"

        if prompt_id not in history_data:
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "No history found for this prompt.",
                            "done": True,
                        },
                    }
                )
            return "Workflow completed, but no history data was found for this prompt."

        outputs = history_data[prompt_id].get("outputs", {})
        if not outputs:
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "No images found in workflow outputs.",
                            "done": True,
                        },
                    }
                )
            return "Workflow completed, but no images were found in the outputs."

        image_count = 0
        for node_id, node_output in outputs.items():
            images_list = node_output.get("images", [])
            for img_meta in images_list:
                image_count += 1
                filename = img_meta["filename"]
                subfolder = img_meta["subfolder"]
                folder_type = img_meta["type"]
                image_url = f"https://{self.valves.ComfyUI_Server}/view?filename={filename}&subfolder={subfolder}&type={folder_type}&token={self.valves.Api_Key}"
                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "message",
                            "data": {
                                "content": f"**Generated Image #{image_count}:** ![Preview]({image_url})"
                            },
                        }
                    )

        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"Workflow completed successfully. Retrieved {image_count} image(s).",
                        "done": True,
                    },
                }
            )

        return f"Workflow completed. Received {image_count} image(s)."
