import importlib
import os
import hf_transfer
from loguru import logger
import sys
logger.remove()
logger.add(sys.stdout, level="WARNING")

AVAILABLE_MODELS = {
    "llava": "Llava",
    "qwen_vl": "Qwen_VL",
    "qwen_vl_chat": "qwen_vl_chat",
    "qwen2_vl_chat": "Qwen2_VL_chat",
    "fuyu": "Fuyu",
    "batch_gpt4": "BatchGPT4",
    "gpt4v": "GPT4V",
    "instructblip": "InstructBLIP",
    "minicpm_v25":"MiniCPM_V25",
    "minicpm_v": "MiniCPM_V",
    "llava_vid": "LlavaVid",
    "videoChatGPT": "VideoChatGPT",
    "llama_vid": "LLaMAVid",
    "video_llava": "VideoLLaVA",
    "xcomposer2_4KHD": "XComposer2_4KHD",
    "claude": "Claude",
    "qwen_vl_api": "Qwen_VL_API",
    "llava_sglang": "LlavaSglang",
    "idefics2": "Idefics2",
    "internvl": "InternVLChat",
    "gemini_api": "GeminiAPI",
    "reka": "Reka",
    "from_log": "FromLog",
    "mplug_owl_video": "mplug_Owl",
    "phi3v": "Phi3v",
    "xcomposer2_4KHD":"XComposer2_4KHD",
    "cogvlm_chat":"CogVLM_Chat",
    "deepseek_vl_chat":"Deepseek_VL_Chat",
    "tiny_llava": "TinyLlava",
    "tiny_llava_phi": "TinyLlava_phi",
    "llava_hf": "LlavaHf",
    "longva": "LongVA",
    "bunny_3b":"Bunny_3B",
    "vicuna_7b":"Vicuna_7b",
    "vicuna_33b":"Vicuna_33b",
    "yi":"Yi",
    "qwen":"Qwen",
    'llama':"Llama",
    "gemma":"Gemma",
    "cambrian":"Cambrian",
    "internvl2": "InternVL2",
    "mantis": "Mantis",
    "emu2": "Emu2",
    "paligemma":"Paligemma",
    "internvl2_large":"InternVL2_large",
    "MIO_sft":"MIO",
    "onevision":"onevision",
    "onevision_large":"onevision_large",
    "cogvlm2":"cogvlm2",
    "MIO_batch":"MIO_batch"
}

for model_name, model_class in AVAILABLE_MODELS.items():
    try:
        exec(f"from .{model_name} import {model_class}")
    except ImportError as e:
        # logger.warning(f"Failed to import {model_class} from {model_name}: {e}")
        pass

if os.environ.get("LMMS_EVAL_PLUGINS", None):
    # Allow specifying other packages to import models from
    for plugin in os.environ["LMMS_EVAL_PLUGINS"].split(","):
        m = importlib.import_module(f"{plugin}.models")
        for model_name, model_class in getattr(m, "AVAILABLE_MODELS").items():
            try:
                exec(f"from {plugin}.models.{model_name} import {model_class}")
            except ImportError:
                pass

import hf_transfer

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
