"""
title: ComfyUI Img2Img Template
Author: Patrick Williams
author_url: https://reticulated.net
git_url: https://github.com/pwillia7/open-webui-tools
description: Template for an img2img ComfyUI tool. Submits a node-based ComfyUI workflow over HTTP, listens on the WebSocket for images, and embeds them in chat.
required_open_webui_version: 0.4.0
requirements: requests, langchain-openai, langgraph, ollama, langchain_ollama, websocket-client
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
from typing import Optional
from pydantic import BaseModel, Field

# Configure logging for debugging purposes
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

class Tools:
    """
    [LLM Usage Directive]:
    - **When to Use**:
        - Use this tool whenever the user wants to enhance or modify an existing image
          (img2img workflow in ComfyUI).
        - Example triggers: "Enhance this image," "Make variations of this image," "Refine this image," etc.
    - **How to Use**:
        - Provide the user’s instructions for how to modify/enhance the image via `prompt_text`. (Optional)
        - Provide the user’s image URL in `image_url`.
        - Ensure the ComfyUI workflow includes a `base64image` input node where the image is injected.
          The workflow must have `%%B64IMAGE%%` as a placeholder in the relevant node’s input.
        - Optionally, pass a maximum number of images to return (`max_returned_images`).
    - **Behavior**:
        - Fetches the image from the provided URL and base64-encodes it.
        - Substitutes `%%B64IMAGE%%` (and optionally `%%PROMPT%%`) in the workflow JSON.
        - Sends the updated workflow to the ComfyUI server over HTTP.
        - Waits on a WebSocket for completion signals.
        - Retrieves the final images from ComfyUI’s `/history/<prompt_id>` endpoint.
        - Emits the images back into chat as Markdown image links (optionally limited to the last N).
    - **Important**:
        - Call this tool only if the user explicitly wants to perform an img2img operation (enhance or modify an existing image).
        - Do not use for simple text-to-image requests; that’s handled by a separate txt2img tool.
    """

    class Valves(BaseModel):
        """
        Configuration for API authentication.
        """
        Api_Key: Optional[str] = Field(
            None,
            description="The API token for authenticating with ComfyUI. Must be set in the Open-WebUI admin interface.",
        )

    class UserValves(BaseModel):
        """
        User-specific configurations.
        """
        debug_mode: bool = Field(
            default=False,
            description="Enable additional debug output for troubleshooting.",
        )

    def __init__(self):
        # Initialize configuration models
        self.valves = self.Valves()
        self.user_valves = self.UserValves()

        # ----------------------- User Configuration Section -----------------------
        # Users should replace the `workflow_template` with their own ComfyUI workflow JSON.
        # Ensure that the workflow includes `%%B64IMAGE%%` where the base64 image should be injected.
        # Optionally, include `%%PROMPT%%` if the workflow utilizes a text prompt.
        self.workflow_template = json.loads(
            """
            {
                "nodes": [
                    {
                        "id": "input_image",
                        "class_type": "ETN_LoadImageBase64",
                        "inputs": {
                            "image": "%%B64IMAGE%%"  # Placeholder for base64-encoded image
                        },
                        "_meta": {
                            "title": "Load Image (Base64)"
                        }
                    },
                    // Add your workflow nodes here. Ensure no comments are present in the actual JSON.
                ],
                "connections": {
                    "input_image": ["next_node_id"]  # Replace with actual node connections
                }
            }
            """
        )

        # Users should update the `server_address` to point to their ComfyUI server.
        self.server_address = "your-comfyui-server-address:port"  # e.g., "localhost:8443"

    def _replace_placeholders(self, obj, placeholders: dict):
        """
        Recursively replace placeholders in the workflow template with actual values.
        """
        if isinstance(obj, dict):
            return {
                k: self._replace_placeholders(v, placeholders) for k, v in obj.items()
            }
        elif isinstance(obj, list):
            return [self._replace_placeholders(item, placeholders) for item in obj]
        elif isinstance(obj, str):
            for ph, val in placeholders.items():
                obj = obj.replace(ph, val)
            return obj
        return obj

    def _queue_prompt(self, workflow: dict, client_id: str) -> str:
        """
        Submit the workflow to the ComfyUI API with the token included in the Authorization header.
        Returns the `prompt_id` for tracking.
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

    async def run_comfyui_img2img_workflow(
        self,
        image_url: str,
        prompt_text: Optional[str] = None,
        target_node: str = "%%B64IMAGE%%",
        max_returned_images: int = 5,
        __event_emitter__=None,
    ) -> str:
        """
        Execute the ComfyUI img2img workflow:
        1) Fetch & base64-encode the user’s image from `image_url`.
        2) Emit the original image to the chat.
        3) Replace placeholders in the workflow JSON:
           - %%B64IMAGE%% with the encoded image
           - %%PROMPT%% with the user instructions (if provided)
        4) Submit the workflow to /prompt
        5) Listen on WebSocket for generation to finish
        6) Fetch generated images from /history/<prompt_id>
        7) Emit them as messages (optionally limit to the last N).
        """
        if not self.valves.Api_Key:
            raise ValueError(
                "API token is not set in Valves. Please configure it in Open-WebUI."
            )

        # 1) Fetch the user’s image & convert to base64
        try:
            resp = requests.get(image_url, timeout=30)
            resp.raise_for_status()
            b64_image = base64.b64encode(resp.content).decode("utf-8")
            logging.debug(f"Image fetched and encoded successfully from {image_url}")
        except Exception as e:
            logging.error(f"Failed to fetch or encode image from {image_url}: {e}")
            return f"Error fetching image: {e}"

        # 2) Emit the original image to the chat
        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "message",
                    "data": {
                        "content": f"**Original Image:** ![Original]({image_url})"
                    },
                }
            )
            logging.debug("Original image emitted to chat.")

        # 3) Inject placeholders into the workflow
        placeholders = {
            "%%B64IMAGE%%": b64_image,
        }
        if prompt_text is not None:
            placeholders["%%PROMPT%%"] = prompt_text

        updated_workflow = self._replace_placeholders(
            self.workflow_template, placeholders
        )
        logging.debug("Placeholders in workflow replaced with actual values.")

        # 4) Connect to ComfyUI WebSocket
        client_id = str(uuid.uuid4())
        ws_url = f"wss://{self.server_address}/ws?clientId={client_id}&token={self.valves.Api_Key}"
        logging.debug(f"Connecting to ComfyUI WebSocket at: {ws_url}")

        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "Connecting to ComfyUI...", "done": False},
                }
            )
            logging.debug("Status emitted: Connecting to ComfyUI.")

        # Connect to WebSocket
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

                # Submit the workflow
                try:
                    prompt_id = self._queue_prompt(updated_workflow, client_id)
                    logging.debug(f"Workflow submitted. prompt_id={prompt_id}")
                    if __event_emitter__:
                        await __event_emitter__(
                            {
                                "type": "status",
                                "data": {
                                    "description": "Workflow submitted. Waiting for video generation...",
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

                # Wait for generation completion
                try:
                    async for raw_msg in ws:
                        if isinstance(raw_msg, bytes):
                            continue
                        message_data = json.loads(raw_msg)
                        msg_type = message_data.get("type", "")
                        msg_info = message_data.get("data", {})
                        if (
                            msg_type == "executing"
                            and msg_info.get("node") is None
                            and msg_info.get("prompt_id") == prompt_id
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

        # 7) Retrieve the image data from /history/<prompt_id>
        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": "ComfyUI generation complete. Retrieving images...",
                        "done": False,
                    },
                }
            )
            logging.debug("Status emitted: Generation complete, retrieving images.")

        history_url = f"https://{self.server_address}/history/{prompt_id}"
        headers = {"Authorization": f"Bearer {self.valves.Api_Key}"}

        try:
            resp = requests.get(history_url, headers=headers, timeout=30)
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
            logging.warning("No history found for the prompt ID.")
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
            logging.warning("No outputs found in the workflow history.")
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
            return "Workflow completed but no images were found in the outputs."

        # 8) Collect and emit the generated images
        all_images = []
        for node_id, node_output in outputs.items():
            images_list = node_output.get("images", [])
            all_images.extend(images_list)

        # Limit the number of returned images if specified
        if max_returned_images > 0:
            all_images = all_images[-max_returned_images:]

        image_count = 0
        for i, img_meta in enumerate(all_images, start=1):
            image_count += 1
            filename = img_meta["filename"]
            subfolder = img_meta["subfolder"]
            folder_type = img_meta["type"]

            # Construct the direct URL to each image
            final_image_url = (
                f"https://{self.server_address}/view"
                f"?filename={filename}&subfolder={subfolder}&type={folder_type}"
            )

            logging.debug(f"Emitting image {i}: {final_image_url}")

            # Emit the image as a Markdown link for inline display
            if __event_emitter__:
                try:
                    await __event_emitter__(
                        {
                            "type": "message",
                            "data": {
                                "content": f"**Enhanced Image #{i}:** ![Preview]({final_image_url})"
                            },
                        }
                    )
                    logging.debug(f"Successfully emitted image {i}.")
                except Exception as e:
                    logging.error(f"Error emitting image {i}: {e}")
                    # Optionally, decide to continue or halt. Here, we continue.

        # 9) Final status update indicating completion
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
            logging.debug(f"Final status emitted: Workflow completed with {image_count} image(s).")

        return f"Workflow completed. Received {image_count} image(s)."
