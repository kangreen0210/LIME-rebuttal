from io import BytesIO
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import os
import base64
from typing import List, Tuple
from tqdm import tqdm
import requests as url_requests
import time


from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval import utils

from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState

try:
    from decord import VideoReader, cpu
except ImportError:
    pass

from PIL import Image

API_TYPE = os.getenv("API_TYPE", "openai")
NUM_SECONDS_TO_SLEEP = 30
from loguru import logger as eval_logger
from openai import OpenAI
if API_TYPE == "openai":
    API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
    API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
elif API_TYPE == "azure":
    API_URL = os.getenv("AZURE_ENDPOINT", "https://api.cognitive.microsoft.com/sts/v1.0/issueToken")
    API_KEY = os.getenv("AZURE_API_KEY", "YOUR_API_KEY")
    headers = {
        "api-key": API_KEY,
        "Content-Type": "application/json",
    }


# def is_openai_v1() -> bool:
#     from importlib.metadata import version
#     from packaging.version import Version, parse
#     _version = parse(version("openai"))
#     return _version >= Version("1.0.0")
   
   
# if is_openai_v1():
#     API_URL = os.path.join(API_URL, "openai")



@register_model("gpt4v_01_batch")
class GPT4V_01_batch(lmms):
    def __init__(
        self,
        model_version: str = "gpt-4-vision-preview",
        modality: str = "image",
        max_frames_for_video: int = 10,
        timeout: int = 120,
        **kwargs,
    ) -> None:
        super().__init__()
        # Manually set a image token for GPT4V so that we can search for it
        # and split the text and image
        # Here we just use the same token as llava for convenient
        self.model_version = model_version
        self.modality = modality
        self.max_frames_for_video = max_frames_for_video
        self.image_token = "<image>"
        self.timeout = timeout


        accelerator = Accelerator()
        # assert self.batch_size_per_gpu == 1, "Llava currently does not support batched generation. See https://github.com/haotian-liu/LLaVA/issues/754. HF Llava also has this issue."
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.accelerator = accelerator
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes

        self.device = self.accelerator.device

    # Function to encode the image
    def encode_image(self, image: Image):
        output_buffer = BytesIO()
        image.save(output_buffer, format="PNG")
        byte_data = output_buffer.getvalue()
        base64_str = base64.b64encode(byte_data).decode("utf-8")
        return base64_str

    # Function to encode the video
    def encode_video(self, video_path, for_get_frames_num):
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frame_num = len(vr)
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, for_get_frames_num, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frames = vr.get_batch(frame_idx).asnumpy()

        base64_frames = []
        for frame in frames:
            img = Image.fromarray(frame)
            output_buffer = BytesIO()
            img.save(output_buffer, format="PNG")
            byte_data = output_buffer.getvalue()
            base64_str = base64.b64encode(byte_data).decode("utf-8")
            base64_frames.append(base64_str)

        return base64_frames

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def process_request(self, request, idx):
        """处理单个请求"""
        contexts, gen_kwargs, doc_to_visual, doc_id, task, split = request.args

        visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
        visuals = self.flatten(visuals)
        imgs = []  # multiple images or frames for video

        for visual in visuals:
            if self.modality == "image":
                img = self.encode_image(visual)
                imgs.append(img)
            elif self.modality == "video":
                frames = self.encode_video(visual, self.max_frames_for_video)
                imgs.extend(frames)

        payload = {"model": self.model_version, "messages": []}
        response_json = {"role": "user", "content": []}

        if self.image_token not in contexts:
            payload["messages"].append(deepcopy(response_json))
            payload["messages"][0]["content"].append({"type": "text", "text": contexts})
            for img in imgs:
                payload["messages"][0]["content"].append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}})
        else:
            contexts = contexts.split(self.image_token)
            for idx, img in enumerate(imgs):
                payload["messages"].append(deepcopy(response_json))
                payload["messages"][idx]["content"].append({"type": "text", "text": contexts[idx]})
                payload["messages"][idx]["content"].append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}})

            payload["messages"].append(deepcopy(response_json))
            payload["messages"][-1]["content"].append({"type": "text", "text": contexts[-1]})

        if "max_new_tokens" not in gen_kwargs:
            gen_kwargs["max_new_tokens"] = 1024
        if "temperature" not in gen_kwargs:
            gen_kwargs["temperature"] = 0
        if "top_p" not in gen_kwargs:
            gen_kwargs["top_p"] = None
        if "num_beams" not in gen_kwargs:
            gen_kwargs["num_beams"] = 1

        payload["max_tokens"] = gen_kwargs["max_new_tokens"]
        payload["temperature"] = gen_kwargs["temperature"]
        for attempt in range(5):
            try:
                client = OpenAI(
                    base_url=API_URL,
                    api_key=API_KEY,
                )

                response = client.chat.completions.create(
                    model=self.model_version,
                    messages=payload["messages"],
                    max_tokens=payload["max_tokens"],
                    temperature=payload["temperature"],
                )

                content = response.choices[0].message.content.strip()
                return content  # 成功时返回内容

            except Exception as e:
                eval_logger.info(f"Attempt {attempt + 1} failed with error: {str(e)}")
                if attempt < 5 - 1:  # 如果还有重试机会
                    time.sleep(NUM_SECONDS_TO_SLEEP)
                else:
                    eval_logger.error(f"All 5 attempts failed. Last error message: {str(e)}")
                    return ""  # 返回空字符串作为失败结果

    def generate_until(self, requests) -> List[str]:
        res = [None] * len(requests)  # 使用 None 初始化结果数组，长度与请求列表相同
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        NUM_WORKERS = 16
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            future_to_request = {executor.submit(lambda reg=reg, idx=idx: self.process_request(reg, idx), reg): idx for idx, reg in enumerate(requests)}

            for future in as_completed(future_to_request):
                idx = future_to_request[future]  # 获取对应请求的索引
                try:
                    content = future.result()
                except Exception as e:
                    eval_logger.error(f"Error processing request: {str(e)}")
                    content = ""
                
                res[idx] = content  # 将结果放到对应的位置
                pbar.update(1)

        pbar.close()
        return res  # 返回结果数组

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        # TODO
        assert False, "GPT4V not support"
