"""
title: ComfyUI Flux Image Enhancer
Author: Patrick Williams
author_url: https://reticulated.net
git_url: https://github.com/pwillia7/open-webui-tools
description: Submits an img2img node-based ComfyUI workflow over HTTP, then listens on the WebSocket for images. Embeds them in chat.
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
        - Use this tool whenever the user wants to enhance or modify an existing image
          (img2img workflow in ComfyUI).
        - Example triggers: "Enhance this image," "Make variations of this image," "Refine this image," etc.
    - **How to Use**:
        - Provide the user’s instructions for how to modify/enhance the image via `prompt_text`. (Optional)
        - Provide the user’s image URL in `image_url`.
        - Provide the node ID or the placeholder key in `target_node` where the base64 image is injected.
          (The actual workflow must have `%%B64IMAGE%%` in that node’s input value.)
        - Optionally pass a maximum number of images to return (`max_returned_images`).
    - **Behavior**:
        - Fetches the image from the provided URL and base64-encodes it.
        - Substitutes `%%B64IMAGE%%` (and optionally `%%PROMPT%%`) in the workflow JSON.
        - Sends the updated workflow to the ComfyUI server over HTTP.
        - Waits on a WebSocket for completion signals.
        - Retrieves the final images from ComfyUI’s `/history/<prompt_id>` endpoint.
        - Emits the images back into chat as Markdown image links (optionally limited to the last N).
    - **Important**:
        - Call this tool only if the user explicitly wants to do an img2img operation (enhance or modify an existing image).
        - Do not use for simple text-to-image requests; that’s the other tool.
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

        # Paste your actual ComfyUI workflow JSON here, with placeholders:
        #   "%%PROMPT%%" for user prompt (if needed)
        #   "%%B64IMAGE%%" for the base64-encoded image
        # Ensure that this JSON does NOT contain any comments.
        self.workflow_template = json.loads(
            """
                 {
  "4": {
    "inputs": {
      "ckpt_name": "sleipnirTLHTurbo_v27TLHFP32Main.safetensors"
    },
    "class_type": "CheckpointLoaderSimple",
    "_meta": {
      "title": "Load Checkpoint"
    }
  },
  "29": {
    "inputs": {
      "control_net_name": "SDXL\\\\controlnet-tile-sdxl-1.0\\\\diffusion_pytorch_model.safetensors"
    },
    "class_type": "ControlNetLoader",
    "_meta": {
      "title": "Load ControlNet Model"
    }
  },
  "45": {
    "inputs": {
      "strength": 0.72,
      "start_percent": 0,
      "end_percent": 1,
      "positive": [
        "120",
        0
      ],
      "negative": [
        "121",
        0
      ],
      "control_net": [
        "29",
        0
      ],
      "image": [
        "105",
        0
      ]
    },
    "class_type": "ControlNetApplyAdvanced",
    "_meta": {
      "title": "Apply ControlNet (Advanced)"
    }
  },
  "69": {
    "inputs": {
      "text": [
        "85",
        0
      ],
      "text2": "A serene, gradient background transitioning from deep blue to warm orange, evoking a tranquil sunset or sunrise."
    },
    "class_type": "ShowText|pysssss",
    "_meta": {
      "title": "Show Text 🐍"
    }
  },
  "70": {
    "inputs": {
      "string": "Output a Stable Diffusion XL prompt to modernize and make the image more real/life-like while maintaining the historicity of this image. DO NOT include the medium of the object in the image in the generated prompt. The prompt should consist of relevant keywords, not full sentences. DO NOT REPEAT KEYWORDS. ONLY OUTPUT THE PROMPT. DO NOT INCLUDE THE MEDIUM OF THE SUBJECT OF THE IMAGE.",
      "strip_newlines": true
    },
    "class_type": "StringConstantMultiline",
    "_meta": {
      "title": "String Constant Multiline"
    }
  },
  "78": {
    "inputs": {
      "tile_size": 1024,
      "overlap": 64,
      "samples": [
        "115",
        0
      ],
      "vae": [
        "4",
        2
      ]
    },
    "class_type": "VAEDecodeTiled",
    "_meta": {
      "title": "VAE Decode (Tiled)"
    }
  },
  "81": {
    "inputs": {
      "tile_size": 1024,
      "fast": false,
      "color_fix": false,
      "pixels": [
        "256",
        0
      ],
      "vae": [
        "4",
        2
      ]
    },
    "class_type": "VAEEncodeTiled_TiledDiffusion",
    "_meta": {
      "title": "Tiled VAE Encode"
    }
  },
  "84": {
    "inputs": {
      "model": "OpenGVLab/InternVL2-2B"
    },
    "class_type": "InternVLModelLoader",
    "_meta": {
      "title": "InternVL Model Loader"
    }
  },
  "85": {
    "inputs": {
      "system_prompt": "You are a professional art historian and expert stable diffusion user.",
      "prompt": [
        "70",
        0
      ],
      "keep_model_loaded": true,
      "max_new_tokens": 1024,
      "do_sample": false,
      "num_beams": 1,
      "image": [
        "87",
        0
      ],
      "model": [
        "84",
        0
      ]
    },
    "class_type": "InternVLHFInference",
    "_meta": {
      "title": "InternVL HF Inference"
    }
  },
  "87": {
    "inputs": {
      "min_num": 1,
      "max_num": 1,
      "image_size": 448,
      "use_thumbnail": false,
      "image": [
        "180",
        0
      ]
    },
    "class_type": "DynamicPreprocess",
    "_meta": {
      "title": "Dynamic Preprocess"
    }
  },
  "95": {
    "inputs": {
      "rgthree_comparer": {
        "images": [
          {
            "name": "A",
            "selected": true,
            "url": "/api/view?filename=rgthree.compare._temp_jgmde_00003_.png&type=temp&subfolder=&rand=0.9458694543632062"
          },
          {
            "name": "B",
            "selected": true,
            "url": "/api/view?filename=rgthree.compare._temp_jgmde_00004_.png&type=temp&subfolder=&rand=0.22383189177408824"
          }
        ]
      },
      "image_a": [
        "273",
        0
      ],
      "image_b": [
        "78",
        0
      ]
    },
    "class_type": "Image Comparer (rgthree)",
    "_meta": {
      "title": "Image Comparer (rgthree)"
    }
  },
  "105": {
    "inputs": {
      "pyrUp_iters": 1,
      "resolution": 1024,
      "image": [
        "256",
        0
      ]
    },
    "class_type": "TilePreprocessor",
    "_meta": {
      "title": "Tile"
    }
  },
  "115": {
    "inputs": {
      "seed": [
        "240",
        0
      ],
      "tiling": 1,
      "steps": 8,
      "cfg": 2.5,
      "sampler_name": "euler_ancestral",
      "scheduler": "normal",
      "denoise": 1,
      "model": [
        "248",
        0
      ],
      "positive": [
        "45",
        0
      ],
      "negative": [
        "45",
        1
      ],
      "latent_image": [
        "81",
        0
      ]
    },
    "class_type": "Tiled KSampler",
    "_meta": {
      "title": "Tiled KSampler"
    }
  },
  "120": {
    "inputs": {
      "width": [
        "193",
        0
      ],
      "height": [
        "193",
        1
      ],
      "crop_w": 0,
      "crop_h": 0,
      "target_width": [
        "193",
        0
      ],
      "target_height": [
        "193",
        1
      ],
      "text_g": [
        "200",
        0
      ],
      "text_l": [
        "200",
        0
      ],
      "clip": [
        "4",
        1
      ]
    },
    "class_type": "CLIPTextEncodeSDXL",
    "_meta": {
      "title": "CLIPTextEncodeSDXL"
    }
  },
  "121": {
    "inputs": {
      "width": [
        "193",
        0
      ],
      "height": [
        "193",
        1
      ],
      "crop_w": 0,
      "crop_h": 0,
      "target_width": [
        "193",
        0
      ],
      "target_height": [
        "193",
        1
      ],
      "text_g": "bad anatomy, deformed, anime, manga, Blurred, blurry, poorly drawn, (crepuscular rays:1.2), dramatic lighting",
      "text_l": "bad anatomy, deformed, anime, manga, Blurred, blurry, poorly drawn, (crepuscular rays:1.2), dramatic lighting",
      "clip": [
        "4",
        1
      ]
    },
    "class_type": "CLIPTextEncodeSDXL",
    "_meta": {
      "title": "CLIPTextEncodeSDXL"
    }
  },
  "145": {
    "inputs": {
      "column_texts": "Original, Stage II; Stage I, Stage III",
      "row_texts": "",
      "font_size": 65
    },
    "class_type": "GridAnnotation",
    "_meta": {
      "title": "GridAnnotation"
    }
  },
  "147": {
    "inputs": {
      "scale": 0.3,
      "model": [
        "4",
        0
      ]
    },
    "class_type": "PerturbedAttentionGuidance",
    "_meta": {
      "title": "PerturbedAttentionGuidance"
    }
  },
  "152": {
    "inputs": {
      "guide_size": 1024,
      "guide_size_for": true,
      "max_size": 1024,
      "seed": [
        "240",
        0
      ],
      "steps": 8,
      "cfg": 2.5,
      "sampler_name": "euler_ancestral",
      "scheduler": "normal",
      "denoise": 0.26,
      "feather": 5,
      "noise_mask": false,
      "force_inpaint": false,
      "bbox_threshold": 0.5,
      "bbox_dilation": 15,
      "bbox_crop_factor": 3,
      "sam_detection_hint": "center-1",
      "sam_dilation": 0,
      "sam_threshold": 0.93,
      "sam_bbox_expansion": 0,
      "sam_mask_hint_threshold": 0.7000000000000001,
      "sam_mask_hint_use_negative": "False",
      "drop_size": 10,
      "wildcard": "",
      "cycle": 1,
      "inpaint_model": false,
      "noise_mask_feather": 20,
      "image": [
        "252",
        0
      ],
      "model": [
        "147",
        0
      ],
      "clip": [
        "4",
        1
      ],
      "vae": [
        "4",
        2
      ],
      "positive": [
        "161",
        0
      ],
      "negative": [
        "161",
        1
      ],
      "bbox_detector": [
        "159",
        0
      ],
      "sam_model_opt": [
        "160",
        0
      ]
    },
    "class_type": "FaceDetailer",
    "_meta": {
      "title": "FaceDetailer"
    }
  },
  "159": {
    "inputs": {
      "model_name": "bbox/face_yolov8n_v2.pt"
    },
    "class_type": "UltralyticsDetectorProvider",
    "_meta": {
      "title": "UltralyticsDetectorProvider"
    }
  },
  "160": {
    "inputs": {
      "model_name": "sam_vit_b_01ec64.pth",
      "device_mode": "AUTO"
    },
    "class_type": "SAMLoader",
    "_meta": {
      "title": "SAMLoader (Impact)"
    }
  },
  "161": {
    "inputs": {
      "strength": 0.5,
      "start_percent": 0.005,
      "end_percent": 0.98,
      "positive": [
        "205",
        0
      ],
      "negative": [
        "121",
        0
      ],
      "control_net": [
        "163",
        0
      ],
      "image": [
        "166",
        0
      ]
    },
    "class_type": "ControlNetApplyAdvanced",
    "_meta": {
      "title": "Apply ControlNet (CANNY)"
    }
  },
  "163": {
    "inputs": {
      "control_net_name": "controlnetxlCNXL_xinsirCnUnionPromax.safetensors"
    },
    "class_type": "ControlNetLoader",
    "_meta": {
      "title": "Load ControlNet Model"
    }
  },
  "166": {
    "inputs": {
      "enable_threshold": "true",
      "threshold_low": 0,
      "threshold_high": 0.9500000000000001,
      "images": [
        "78",
        0
      ]
    },
    "class_type": "Image Canny Filter",
    "_meta": {
      "title": "Image Canny Filter"
    }
  },
  "171": {
    "inputs": {
      "control_net_name": "controlnetxlCNXL_xinsirDepth.safetensors"
    },
    "class_type": "ControlNetLoader",
    "_meta": {
      "title": "Load ControlNet Model"
    }
  },
  "172": {
    "inputs": {
      "strength": 0.78,
      "start_percent": 0.005,
      "end_percent": 0.98,
      "positive": [
        "176",
        0
      ],
      "negative": [
        "176",
        1
      ],
      "control_net": [
        "171",
        0
      ],
      "image": [
        "211",
        0
      ]
    },
    "class_type": "ControlNetApplyAdvanced",
    "_meta": {
      "title": "Apply ControlNet (DEPTH)"
    }
  },
  "176": {
    "inputs": {
      "strength": 0.7000000000000001,
      "start_percent": 0.05,
      "end_percent": 0.9500000000000001,
      "positive": [
        "120",
        0
      ],
      "negative": [
        "121",
        0
      ],
      "control_net": [
        "163",
        0
      ],
      "image": [
        "166",
        0
      ]
    },
    "class_type": "ControlNetApplyAdvanced",
    "_meta": {
      "title": "Apply ControlNet (CANNY)"
    }
  },
  "180": {
    "inputs": {
      "upscale_model": "4xUltrasharp_4xUltrasharpV10.pt",
      "mode": "resize",
      "rescale_factor": 2,
      "resize_width": 2048,
      "resampling_method": "nearest",
      "supersample": "true",
      "rounding_modulus": 8,
      "image": [
        "308",
        0
      ]
    },
    "class_type": "CR Upscale Image",
    "_meta": {
      "title": "🔍 CR Upscale Image"
    }
  },
  "193": {
    "inputs": {
      "image": [
        "256",
        0
      ]
    },
    "class_type": "GetImageSize+",
    "_meta": {
      "title": "🔧 Get Image Size"
    }
  },
  "199": {
    "inputs": {
      "text": [
        "69",
        0
      ],
      "truncate_by": "words",
      "truncate_from": "beginning",
      "truncate_to": 55
    },
    "class_type": "Text String Truncate",
    "_meta": {
      "title": "Text String Truncate"
    }
  },
  "200": {
    "inputs": {
      "text": [
        "199",
        0
      ],
      "text2": "A serene, gradient background transitioning from deep blue to warm orange, evoking a tranquil sunset or sunrise."
    },
    "class_type": "ShowText|pysssss",
    "_meta": {
      "title": "Show Text 🐍"
    }
  },
  "205": {
    "inputs": {
      "width": [
        "193",
        0
      ],
      "height": [
        "193",
        1
      ],
      "crop_w": 0,
      "crop_h": 0,
      "target_width": [
        "193",
        0
      ],
      "target_height": [
        "193",
        1
      ],
      "text_g": [
        "241",
        0
      ],
      "text_l": [
        "241",
        0
      ],
      "clip": [
        "4",
        1
      ]
    },
    "class_type": "CLIPTextEncodeSDXL",
    "_meta": {
      "title": "CLIPTextEncodeSDXL"
    }
  },
  "211": {
    "inputs": {
      "ckpt_name": "depth_anything_v2_vitl.pth",
      "resolution": 2048,
      "image": [
        "78",
        0
      ]
    },
    "class_type": "DepthAnythingV2Preprocessor",
    "_meta": {
      "title": "Depth Anything V2 - Relative"
    }
  },
  "238": {
    "inputs": {
      "seed": [
        "240",
        0
      ],
      "tiling": 0,
      "steps": 8,
      "cfg": 3.5,
      "sampler_name": "euler_ancestral",
      "scheduler": "normal",
      "denoise": 1,
      "model": [
        "302",
        0
      ],
      "positive": [
        "172",
        0
      ],
      "negative": [
        "172",
        1
      ],
      "latent_image": [
        "115",
        0
      ]
    },
    "class_type": "Tiled KSampler",
    "_meta": {
      "title": "Tiled KSampler"
    }
  },
  "240": {
    "inputs": {
      "seed": 568159718233566
    },
    "class_type": "Seed (rgthree)",
    "_meta": {
      "title": "Seed (rgthree)"
    }
  },
  "241": {
    "inputs": {
      "delimiter": ", ",
      "clean_whitespace": "true",
      "text_a": [
        "242",
        0
      ],
      "text_b": [
        "200",
        0
      ]
    },
    "class_type": "Text Concatenate",
    "_meta": {
      "title": "Text Concatenate"
    }
  },
  "242": {
    "inputs": {
      "text": "historical face detail, correctly aged face",
      "text_b": "",
      "text_c": "",
      "text_d": ""
    },
    "class_type": "Text String",
    "_meta": {
      "title": "Text String"
    }
  },
  "248": {
    "inputs": {
      "scale": 0.45,
      "model": [
        "4",
        0
      ]
    },
    "class_type": "PerturbedAttentionGuidance",
    "_meta": {
      "title": "PerturbedAttentionGuidance"
    }
  },
  "251": {
    "inputs": {
      "tile_size": 1024,
      "overlap": 64,
      "samples": [
        "238",
        0
      ],
      "vae": [
        "4",
        2
      ]
    },
    "class_type": "VAEDecodeTiled",
    "_meta": {
      "title": "VAE Decode (Tiled)"
    }
  },
  "252": {
    "inputs": {
      "amount": 0.8,
      "image": [
        "251",
        0
      ]
    },
    "class_type": "ImageCASharpening+",
    "_meta": {
      "title": "🔧 Image Contrast Adaptive Sharpening"
    }
  },
  "256": {
    "inputs": {
      "width": 2048,
      "height": 2048,
      "upscale_method": "nearest-exact",
      "keep_proportion": true,
      "divisible_by": 0,
      "crop": "disabled",
      "image": [
        "180",
        0
      ]
    },
    "class_type": "ImageResizeKJ",
    "_meta": {
      "title": "Resize Image"
    }
  },
  "273": {
    "inputs": {
      "width": 2048,
      "height": 2048,
      "upscale_method": "nearest-exact",
      "keep_proportion": true,
      "divisible_by": 0,
      "crop": "disabled",
      "image": [
        "308",
        0
      ]
    },
    "class_type": "ImageResizeKJ",
    "_meta": {
      "title": "Resize Image"
    }
  },
  "275": {
    "inputs": {
      "seed": [
        "297",
        0
      ],
      "tiling": 0,
      "steps": 8,
      "cfg": 5,
      "sampler_name": "dpmpp_sde_gpu",
      "scheduler": "normal",
      "denoise": 1,
      "model": [
        "283",
        0
      ],
      "positive": [
        "172",
        0
      ],
      "negative": [
        "172",
        1
      ],
      "latent_image": [
        "115",
        0
      ]
    },
    "class_type": "Tiled KSampler",
    "_meta": {
      "title": "Tiled KSampler"
    }
  },
  "276": {
    "inputs": {
      "tile_size": 1024,
      "overlap": 64,
      "samples": [
        "275",
        0
      ],
      "vae": [
        "4",
        2
      ]
    },
    "class_type": "VAEDecodeTiled",
    "_meta": {
      "title": "VAE Decode (Tiled)"
    }
  },
  "277": {
    "inputs": {
      "amount": 0.8,
      "image": [
        "276",
        0
      ]
    },
    "class_type": "ImageCASharpening+",
    "_meta": {
      "title": "🔧 Image Contrast Adaptive Sharpening"
    }
  },
  "279": {
    "inputs": {
      "guide_size": 1024,
      "guide_size_for": true,
      "max_size": 1024,
      "seed": [
        "240",
        0
      ],
      "steps": 8,
      "cfg": 5,
      "sampler_name": "dpmpp_sde_gpu",
      "scheduler": "normal",
      "denoise": 0.35000000000000003,
      "feather": 5,
      "noise_mask": false,
      "force_inpaint": false,
      "bbox_threshold": 0.5,
      "bbox_dilation": 15,
      "bbox_crop_factor": 3,
      "sam_detection_hint": "center-1",
      "sam_dilation": 0,
      "sam_threshold": 0.93,
      "sam_bbox_expansion": 0,
      "sam_mask_hint_threshold": 0.7000000000000001,
      "sam_mask_hint_use_negative": "False",
      "drop_size": 10,
      "wildcard": "",
      "cycle": 1,
      "inpaint_model": false,
      "noise_mask_feather": 20,
      "image": [
        "277",
        0
      ],
      "model": [
        "283",
        0
      ],
      "clip": [
        "4",
        1
      ],
      "vae": [
        "4",
        2
      ],
      "positive": [
        "161",
        0
      ],
      "negative": [
        "161",
        1
      ],
      "bbox_detector": [
        "159",
        0
      ],
      "sam_model_opt": [
        "160",
        0
      ]
    },
    "class_type": "FaceDetailer",
    "_meta": {
      "title": "FaceDetailer"
    }
  },
  "280": {
    "inputs": {
      "filename_prefix": "stage3_tile_img2img",
      "images": [
        "279",
        0
      ]
    },
    "class_type": "SaveImage",
    "_meta": {
      "title": "Save Image"
    }
  },
  "282": {
    "inputs": {
      "filename_prefix": "stage1_tile_img2img",
      "images": [
        "78",
        0
      ]
    },
    "class_type": "SaveImage",
    "_meta": {
      "title": "Save Image"
    }
  },
  "283": {
    "inputs": {
      "scale": 0.75,
      "model": [
        "4",
        0
      ]
    },
    "class_type": "PerturbedAttentionGuidance",
    "_meta": {
      "title": "PerturbedAttentionGuidance"
    }
  },
  "287": {
    "inputs": {
      "color_space": "LAB",
      "factor": 0.6,
      "device": "gpu",
      "batch_size": 0,
      "image": [
        "152",
        0
      ],
      "reference": [
        "256",
        0
      ]
    },
    "class_type": "ImageColorMatch+",
    "_meta": {
      "title": "🔧 Image Color Match"
    }
  },
  "288": {
    "inputs": {
      "filename_prefix": "stage2_colorfix_\\\\tile_img2img",
      "images": [
        "299",
        0
      ]
    },
    "class_type": "SaveImage",
    "_meta": {
      "title": "Save Image"
    }
  },
  "297": {
    "inputs": {
      "seed": 505923945412630
    },
    "class_type": "Seed (rgthree)",
    "_meta": {
      "title": "Seed (rgthree)"
    }
  },
  "299": {
    "inputs": {
      "factor": 1.6,
      "images": [
        "301",
        0
      ]
    },
    "class_type": "Saturation",
    "_meta": {
      "title": "Saturation"
    }
  },
  "301": {
    "inputs": {
      "black_level": 0,
      "mid_level": 127.5,
      "white_level": 255,
      "image": [
        "287",
        0
      ]
    },
    "class_type": "Image Levels Adjustment",
    "_meta": {
      "title": "Image Levels Adjustment"
    }
  },
  "302": {
    "inputs": {
      "mimic_scale": 2.5,
      "threshold_percentile": 0.75,
      "model": [
        "147",
        0
      ]
    },
    "class_type": "DynamicThresholdingSimple",
    "_meta": {
      "title": "DynamicThresholdingSimple"
    }
  },
  "308": {
    "inputs": {
      "image": "%%B64IMAGE%%"
    },
    "class_type": "ETN_LoadImageBase64",
    "_meta": {
      "title": "Load Image (Base64)"
    }
  }
}
            """
        )

        # ComfyUI server config
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

        # 3) Inject placeholders
        placeholders = {
            "%%B64IMAGE%%": b64_image,
        }
        # Only add %%PROMPT%% if we actually have a prompt text
        if prompt_text is not None:
            placeholders["%%PROMPT%%"] = prompt_text

        updated_workflow = self._replace_placeholders(
            self.workflow_template, placeholders
        )

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

        try:
            ws = websocket.WebSocket()
            ws.connect(ws_url, timeout=30)
            logging.debug("WebSocket connection established.")
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "Connected! Submitting img2img workflow...",
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

        # 5) Send the workflow to /prompt
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

        # 6) Wait for the finishing message (type=executing, node=None, prompt_id=...)
        try:
            while True:
                raw_msg = ws.recv()
                if isinstance(raw_msg, bytes):
                    # If partial images or previews are sent as binary, handle here if desired
                    continue

                message_data = json.loads(raw_msg)
                msg_type = message_data.get("type", "")
                msg_info = message_data.get("data", {})

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

        # Collect all images from all output nodes
        all_images = []
        for node_id, node_output in outputs.items():
            images_list = node_output.get("images", [])
            all_images.extend(images_list)

        # If max_returned_images > 0, only keep the last N images
        if max_returned_images > 0:
            all_images = all_images[-max_returned_images:]

        # Emit the images in chat
        image_count = 0
        for i, img_meta in enumerate(all_images, start=1):
            image_count += 1
            filename = img_meta["filename"]
            subfolder = img_meta["subfolder"]
            folder_type = img_meta["type"]

            # Build direct URL to each image
            final_image_url = (
                f"https://{self.server_address}/view"
                f"?filename={filename}&subfolder={subfolder}&type={folder_type}"
            )

            # Emit as a Markdown link so that Open-WebUI shows it inline
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "message",
                        "data": {
                            "content": f"**Enhanced Image #{i}:** ![Preview]({final_image_url})"
                        },
                    }
                )

        # Final status update
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
