import torch
from accelerate import Accelerator, DistributedType
from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from typing import List, Optional, Tuple, Union

from loguru import logger as eval_logger

DEFAULT_IMAGE_TOKEN = '<image>'

@register_model('bunny_3b')
class Bunny_3B(lmms):
    def __init__(
        self,
        pretrained: str = "BAAI/Bunny-v1_0-3B",
        device: str = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = 'auto',
        batch_size: int = 1,
        device_map="cuda:0",
        trust_remote_code: Optional[bool] = True,
        use_cache: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"
        # use GPUs
        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
        else:
            self._device = device

        torch.set_default_device(self._device)
        # load the model
        # torch_dtype=torch.float16,
        self._model = AutoModelForCausalLM.from_pretrained(pretrained, device_map=self._device, trust_remote_code=trust_remote_code).eval()
        # self._processor = AutoProcessor.from_pretrained(pretrained, trust_remote_code=trust_remote_code).to(self.model.device)
        # self._processor._tokenizer.padding_side = "left"

        # self._tokenizer = self._processor._tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=trust_remote_code)
        self.model.eval()
        # self._eos_token_id = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=trust_remote_code)
        self._config = self._model.config
        # self.model.tie_weights()
        self.batch_size_per_gpu = int(batch_size)
        assert self.batch_size_per_gpu == 1, "batch_size_per_gpu > 1 is not supported for now."
        self.use_cache = use_cache

        # assign the model to multi-gpus
        if accelerator.num_processes > 1:
            distributed_type_list = [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED]
            assert accelerator.distributed_type in distributed_type_list, "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            eval_logger.info(f"Using single device: {self._device}")
            self.model.to(self._device)
            self._rank = 0
            self._word_size = 1
        


    @property
    def config(self):
        return self._config


    @property
    def tokenizer(self):
        return self._tokenizer


    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model


    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id


    @property
    def max_length(self):
        return self._max_length


    @property
    def batch_size(self):
        return self.batch_size_per_gpu


    @property
    def device(self):
        return self._device


    @property
    def rank(self):
        return self._rank


    @property
    def world_size(self):
        return self._world_size


    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list


    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)


    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Not implemented for Bunny_3b.")

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            visuals = self.flatten(visuals)
            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]

            # Set default values for until and max_new_tokens
            until = [self.tok_decode(self.eot_token_id)]
            # Update values from gen_kwargs if present
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str,list] but got {type(until)}")
            assert self.batch_size_per_gpu == 1, "Do not support batch_size_per_gpu > 1 for now"
            context = contexts[0]
            
            if "<image>" in contexts:
                # instruct blip does not expect the <image> tag
                contexts = contexts.replace("<image>", "")

            # Some benchmarks like MME do not contain image tokens, so we prepend them to the prompt.
            if DEFAULT_IMAGE_TOKEN not in context:
                image_tokens = [DEFAULT_IMAGE_TOKEN] * len(visuals)
                image_tokens = " ".join(image_tokens)
                context = f"{image_tokens}\n{context}"

            prompt = context
            text = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\n{prompt} ASSISTANT:"
            text_chunks = [self.tokenizer(chunk).input_ids for chunk in text.split('<image>')]

            # input_ids = torch.tensor(text_chunks[0] + [-200] + text_chunks[1], dtype=torch.long).unsqueeze(0).to(self._device, self.model.dtype)
            # image_tensor = self.model.process_images(visuals, self.config).to(self._device, self.model.dtype)
            input_ids = torch.tensor(text_chunks[0] + [-200] + text_chunks[1], dtype=torch.long).unsqueeze(0).to(device=self.model.device)
            image_tensor = self.model.process_images(visuals, self.config).to(device=self.model.device, dtype=self.model.dtype)
            # generate
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=self.use_cache,
                **gen_kwargs
            )[0]
            final_completion = self.tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip()
            res.append(final_completion)
            # for output_id, input_id in zip(output_ids, input_ids):
            #     generated_id = output_id[len(input_id) :]
            #     generated_text = self.tokenizer.decode(generated_id, skip_special_tokens=True)

            #     res.append(generated_text)
            pbar.update(1)
            
         # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)
        pbar.close()

        return res
    