import torch
import logging
import copy
from tqdm import tqdm
from datetime import timedelta
from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from vllm import LLM
from vllm.sampling_params import SamplingParams
from transformers import AutoTokenizer

from accelerate import Accelerator, InitProcessGroupKwargs
from typing import List, Optional, Union,Tuple

import warnings

warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore")

eval_logger = logging.getLogger("lmms-eval")


@register_model("gemma")
class Gemma(lmms):
    """
    google/gemma-1.1-7b-it
    https://huggingface.co/google/gemma-1.1-7b-it
    """
    def __init__(
        self,
        pretrained: str = "/ML-A100/team/mm/zhangge/models/gemma-1.1-7b-it", 
        device: Optional[str] = "cuda",
        batch_size: Optional[Union[int, str]] = 1,
        device_map="auto",
        trust_remote_code: Optional[bool] = True,
        use_cache=True,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        if accelerator.num_processes > 1 and device_map == "":
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = device
            self.device_map = device_map

        self._model = LLM(pretrained, tensor_parallel_size=8, trust_remote_code=True)
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache
        self._rank = 0
        self._world_size = 1
        self.accelerator = accelerator

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model
    
    def tok_encode(self, string: str, add_special_tokens=None) -> List[int]:
        add_special_tokens = True if add_special_tokens is None else add_special_tokens
        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        return encoding
    
    def tok_decode(self, tokens):
        try:
            return self.tokenizer.decode(tokens)
        except:
            return self.tokenizer.decode([tokens])
    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        res = []
        request_batch_size = 64
        sampling_params = SamplingParams(max_tokens = 128, temperature = 0.0)
        raw_instruction_data = copy.deepcopy(requests)
        for i in tqdm(range(0, len(raw_instruction_data), request_batch_size), desc="Processing raw_instruction_data"):
            if i + request_batch_size >= len(raw_instruction_data):
                batch = raw_instruction_data[i:]
            else:
                batch = raw_instruction_data[i:i + request_batch_size]
            
            messages_list = []
            for l in batch:
                message = [
                    {"role": "user", "content": l.args[0]}
                ]
                messages_list.append(message)
            prompt_token_ids = [self.tokenizer.apply_chat_template(messages, add_generation_prompt=True) for messages in messages_list]
            outputs = self.model.generate(prompt_token_ids = prompt_token_ids, sampling_params = sampling_params)
            results = [output.outputs[0].text for output in outputs]
            for i in range(len(batch)):
                res.append(results[i])
       
        return res
        
    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []
        request_batch_size = 64
        sampling_params = SamplingParams(max_tokens = 128, temperature = 0.0)
        raw_instruction_data = copy.deepcopy(requests)
        for i in tqdm(range(0, len(raw_instruction_data), request_batch_size), desc="Processing raw_instruction_data"):
            if i + request_batch_size >= len(raw_instruction_data):
                batch = raw_instruction_data[i:]
            else:
                batch = raw_instruction_data[i:i + request_batch_size]
            
            messages_list = []
            for l in batch:
                message = [
                    {"role": "user", "content": l.args[0]}
                ]
                messages_list.append(message)
            prompt_token_ids = [self.tokenizer.apply_chat_template(messages, add_generation_prompt=True) for messages in messages_list]
            outputs = self.model.generate(prompt_token_ids = prompt_token_ids, sampling_params = sampling_params)
            results = [output.outputs[0].text for output in outputs]
            for i in range(len(batch)):
                res.append(results[i])
       
        return res