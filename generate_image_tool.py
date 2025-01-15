"""
title: ComfyUI Flux Image Maker
Author: Patrick Williams
author_url: https://reticulated.net
git_url: https://github.com/pwillia7/open-webui-tools
description: Submits a node-based ComfyUI workflow over HTTP, then listens on the WebSocket for images from SaveImageWebsocket node. Embeds them in chat.
required_open_webui_version: 0.4.0
requirements: requests, langchain-openai, langgraph, ollama, langchain_ollama, websocket-client
version: 1.0.0
license: MIT
"""

import uuid
import json
import base64
import requests
import websocket
import logging
from typing import Optional
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


class Tools:
    """
    [LLM Usage Directive]:
    - **When to Use**:
        - Use this tool whenever the user explicitly asks to "generate an image,"
          "create a Flux Image," or any similar request for an image.
        - Example triggers: "Generate an image of a cat in space." / "Create a Flux image of a fantasy landscape."
    - **How to Use**:
        - Pass the user's prompt directly as the `prompt_text` parameter to `run_comfyui_workflow`.
        - Do not alter or parse the user's text beyond inserting it into the workflow placeholder `%%PROMPT%%`.
    - **Behavior**:
        - Submits a predefined ComfyUI workflow (with placeholders) to the ComfyUI server (via HTTP).
        - Listens on a WebSocket until generation completes.
        - Fetches the final images from `/history/<prompt_id>` and emits them into the chat via event emitters.
    - **Examples**:
        - "Generate an image of a futuristic city."
        - "Make a Flux Image of a mountain sunrise."
        - "Create an image of a medieval castle."
    - **Important**:
        - Only call this tool if the user wants an image.
        - If the user wants text analysis or other tasks, do **not** invoke it.
        - Provide the user’s prompt **verbatim** to `prompt_text`.

    """

    class Valves(BaseModel):
        Api_Key: Optional[str] = Field(
            None,
            description="The API token for authenticating with ComfyUI. Must be set in the Open-WebUI admin interface.",
        )

    class UserValves(BaseModel):
        debug_mode: bool = Field(
            default=False,
            description="Enable additional debug output for troubleshooting.",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.user_valves = self.UserValves()

        # Workflow template with placeholders (paste your actual JSON here)
        self.workflow_template = json.loads(
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

        # ComfyUI server configuration
        self.server_address = "ptkwilliams.ddns.net:8443"

    def _replace_placeholders(self, obj, placeholders: dict):
        """
        Replace placeholders in the workflow template with actual values.
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
        __event_emitter__=None,
    ) -> str:
        """
        Execute the ComfyUI workflow:
        1) Submit the prompt/workflow to /prompt
        2) Listen on WebSocket for the finishing message
        3) Fetch generated images from /history/<prompt_id>
        4) Emit them as chat messages using /view URLs
        """

        if not self.valves.Api_Key:
            raise ValueError(
                "API token is not set in Valves. Please configure it in Open-WebUI."
            )

        # 1) Inject the user's prompt into the workflow JSON
        updated_workflow = self._replace_placeholders(
            self.workflow_template, {"%%PROMPT%%": prompt_text}
        )

        # 2) Start connecting to the ComfyUI WebSocket
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

        try:
            ws = websocket.WebSocket()
            ws.connect(ws_url, timeout=30)
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

        # 3) Send the workflow to /prompt
        try:
            prompt_id = self._queue_prompt(updated_workflow, client_id)
            logging.debug(f"Workflow submitted. prompt_id={prompt_id}")

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "Workflow submitted. Waiting for image generation...",
                            "done": False,
                        },
                    }
                )
        except Exception as e:
            ws.close()
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

        # 4) Wait for the finishing message (type=executing, node=None, prompt_id=...) from the WebSocket
        try:
            while True:
                raw_msg = ws.recv()
                if isinstance(raw_msg, bytes):
                    # If your node sends partial image previews as binary, you could parse them here
                    continue

                message_data = json.loads(raw_msg)
                msg_type = message_data.get("type", "")
                msg_info = message_data.get("data", {})

                # Typically, ComfyUI signals completion with {"type":"executing","data":{"node":null,"prompt_id":...}}
                if (
                    msg_type == "executing"
                    and msg_info.get("node") is None
                    and msg_info.get("prompt_id") == prompt_id
                ):
                    logging.debug("ComfyUI signaled that generation is complete.")
                    break
        except Exception as e:
            logging.error(f"Error receiving WebSocket messages: {e}")

        # Close the WebSocket
        ws.close()
        logging.debug("WebSocket connection closed.")

        # 5) Retrieve the image data from /history/<prompt_id>
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
            return "Workflow completed but no images were found in the outputs."

        # 6) Loop over all outputs, build /view URLs, and emit them
        image_count = 0
        for node_id, node_output in outputs.items():
            images_list = node_output.get("images", [])
            for img_meta in images_list:
                image_count += 1
                filename = img_meta["filename"]
                subfolder = img_meta["subfolder"]
                folder_type = img_meta["type"]

                # Build direct URL to each image
                image_url = (
                    f"https://{self.server_address}/view"
                    f"?filename={filename}&subfolder={subfolder}&type={folder_type}"
                )

                # Emit as a Markdown link so that Open-WebUI shows it inline
                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "message",
                            "data": {
                                "content": f"**Generated Image #{image_count}:** ![Preview]({image_url})"
                            },
                        }
                    )

        # 7) Final status update
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
