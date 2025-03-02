"""
title: ComfyUI Flux Image Maker
Author: Patrick Williams
author_url: https://reticulated.net
git_url: https://github.com/pwillia7/open-webui-tools
description: Submits a node-based ComfyUI workflow over HTTP, then listens on the WebSocket for images from SaveImageWebsocket node. Embeds them in chat.
required_open_webui_version: 0.4.0
requirements: requests, langchain-openai, langgraph, ollama, langchain_ollama, websocket-client
version: 1.0.1
license: MIT
"""

import uuid
import json
import base64
import requests
import asyncio
import websockets
import logging
from typing import Dict, Any, Optional, List, Callable, Awaitable
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


class Tools:
    """
    [LLM Usage Directive]:
    - **Purpose**: This tool generates images using ComfyUI with a node-based workflow.
    - **When to Use**:
        - Whenever the user explicitly requests an image:
          e.g. "Generate an image of X," "Create a flux image," "Make a flux image," etc.
    - **How to Use**:
        - Call run_comfyui_workflow(prompt_text=USER_PROMPT) directly.
        - Pass the user's text verbatim as prompt_text.
        - Do **not** ask for additional clarifications or parse the prompt.
    - **Error Handling**:
        - If the tool raises an error, return that error message verbatim so the user knows the failure reason.
    - **Do Not Use**:
        - For text analysis or any request not explicitly asking for an image.
    - **Example**:
        - User says: "Generate an image of a futuristic city at sunset."
          → Call this tool with prompt_text="a futuristic city at sunset."
    """

    class Valves(BaseModel):
        Api_Key: Optional[str] = Field(
            None,
            description="The API token for authenticating with ComfyUI. Must be set in the Open-WebUI admin interface.",
        )
        ComfyUI_Server: Optional[str] = Field(
            None,
            description="The address of the ComfyUI server, e.g. 'myserver.ddns.net:8443'. Must be set in Open-WebUI.",
        )

    class UserValves(BaseModel):
        debug_mode: bool = Field(
            default=False,
            description="Enable additional debug output for troubleshooting.",
        )

    def __init__(self) -> None:
        self.valves = self.Valves()
        self.user_valves = self.UserValves()

        # A placeholder workflow template with placeholders:
        # Must parse as a dict (no union) to avoid "anyOf" in JSON schema
        self.workflow_template: Dict[str, Any] = json.loads(
            """
            {
              "6": {
                "inputs": {
                  "text": "%%PROMPT%%",
                  "clip": [
                    "11",
                    0
                  ]
                },
                "class_type": "CLIPTextEncode",
                "_meta": {
                  "title": "CLIP Text Encode (Positive Prompt)"
                }
              },
              "8": {
                "inputs": {
                  "samples": [
                    "13",
                    0
                  ],
                  "vae": [
                    "10",
                    0
                  ]
                },
                "class_type": "VAEDecode",
                "_meta": {
                  "title": "VAE Decode"
                }
              },
              "10": {
                "inputs": {
                  "vae_name": "ae.safetensors"
                },
                "class_type": "VAELoader",
                "_meta": {
                  "title": "Load VAE"
                }
              },
              "11": {
                "inputs": {
                  "clip_name1": "t5xxl_fp8_e4m3fn.safetensors",
                  "clip_name2": "clip_l.safetensors",
                  "type": "flux"
                },
                "class_type": "DualCLIPLoader",
                "_meta": {
                  "title": "DualCLIPLoader"
                }
              },
              "12": {
                "inputs": {
                  "unet_name": "flux1-schnell-fp8.safetensors",
                  "weight_dtype": "default"
                },
                "class_type": "UNETLoader",
                "_meta": {
                  "title": "Load Diffusion Model"
                }
              },
              "13": {
                "inputs": {
                  "noise": [
                    "25",
                    0
                  ],
                  "guider": [
                    "22",
                    0
                  ],
                  "sampler": [
                    "16",
                    0
                  ],
                  "sigmas": [
                    "17",
                    0
                  ],
                  "latent_image": [
                    "27",
                    0
                  ]
                },
                "class_type": "SamplerCustomAdvanced",
                "_meta": {
                  "title": "SamplerCustomAdvanced"
                }
              },
              "16": {
                "inputs": {
                  "sampler_name": "euler"
                },
                "class_type": "KSamplerSelect",
                "_meta": {
                  "title": "KSamplerSelect"
                }
              },
              "17": {
                "inputs": {
                  "scheduler": "simple",
                  "steps": 5,
                  "denoise": 1,
                  "model": [
                    "30",
                    0
                  ]
                },
                "class_type": "BasicScheduler",
                "_meta": {
                  "title": "BasicScheduler"
                }
              },
              "22": {
                "inputs": {
                  "model": [
                    "30",
                    0
                  ],
                  "conditioning": [
                    "26",
                    0
                  ]
                },
                "class_type": "BasicGuider",
                "_meta": {
                  "title": "BasicGuider"
                }
              },
              "25": {
                "inputs": {
                  "noise_seed": 1119668199579776
                },
                "class_type": "RandomNoise",
                "_meta": {
                  "title": "RandomNoise"
                }
              },
              "26": {
                "inputs": {
                  "guidance": 3.5,
                  "conditioning": [
                    "6",
                    0
                  ]
                },
                "class_type": "FluxGuidance",
                "_meta": {
                  "title": "FluxGuidance"
                }
              },
              "27": {
                "inputs": {
                  "width": 1216,
                  "height": 832,
                  "batch_size": 1
                },
                "class_type": "EmptySD3LatentImage",
                "_meta": {
                  "title": "EmptySD3LatentImage"
                }
              },
              "30": {
                "inputs": {
                  "max_shift": 1.1500000000000001,
                  "base_shift": 0.5,
                  "width": 1216,
                  "height": 832,
                  "model": [
                    "12",
                    0
                  ]
                },
                "class_type": "ModelSamplingFlux",
                "_meta": {
                  "title": "ModelSamplingFlux"
                }
              },
              "41": {
                "inputs": {
                  "filename_prefix": "Fluxapi",
                  "images": [
                    "8",
                    0
                  ]
                },
                "class_type": "SaveImage",
                "_meta": {
                  "title": "Save Image"
                }
              }
            }
            """
        )

        # Use the server from valves, or fall back to a default if none set
        self.server_address = self.valves.ComfyUI_Server

    def _replace_placeholders(self, data: dict, placeholders: Dict[str, str]) -> dict:
        """
        (Private) Recursively replace '%%PLACEHOLDER%%' in string fields within a dict.
        """
        # We'll avoid a typed union here. We only publicly define dict→dict
        # so we don't produce "anyOf" in JSON schema.
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
                            # also replace placeholders in strings inside lists
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
        # Helper for strings in arrays
        for ph, replacement in placeholders.items():
            text = text.replace(ph, replacement)
        return text

    def _queue_prompt(self, workflow: dict, client_id: str) -> str:
        """
        Submit the workflow to the ComfyUI API with the token included in the Authorization header.
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
          1) Replacing the user's prompt into the workflow JSON.
          2) Submitting the prompt to /prompt.
          3) Listening on the WebSocket until generation completes.
          4) Fetching generated images from /history/<prompt_id>.
          5) Emitting them as chat messages.
        """
        if not self.valves.Api_Key:
            raise ValueError(
                "API token is not set in Valves. Please configure it in Open-WebUI."
            )

        # Make a deep copy so we don't mutate self.workflow_template
        import copy

        workflow_copy = copy.deepcopy(self.workflow_template)

        # Replace placeholders in place
        updated_workflow = self._replace_placeholders(
            workflow_copy, {"%%PROMPT%%": prompt_text}
        )

        # updated_workflow must remain a dict for ComfyUI
        if not isinstance(updated_workflow, dict):
            raise TypeError("Workflow must be a dict after placeholders are replaced.")

        # Prepare WebSocket connection
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

        # Retrieve images from /history/<prompt_id>
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

        # Emit each image
        image_count = 0
        for node_id, node_output in outputs.items():
            images_list = node_output.get("images", [])
            for img_meta in images_list:
                image_count += 1
                filename = img_meta["filename"]
                subfolder = img_meta["subfolder"]
                folder_type = img_meta["type"]

                image_url = (
                    f"https://{self.server_address}/view"
                    f"?filename={filename}&subfolder={subfolder}&type={folder_type}"
                )
                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "message",
                            "data": {
                                "content": f"**Generated Image #{image_count}:** ![Preview]({image_url})"
                            },
                        }
                    )

        # Final status
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
