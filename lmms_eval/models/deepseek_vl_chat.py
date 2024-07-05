import torch
import logging
from tqdm import tqdm
from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
# from lmms_eval.models.model_utils.qwen.qwen_generate_utils import make_context

from accelerate import Accelerator, DistributedType
from typing import List, Optional, Union, Tuple
import uuid
import os

import warnings

warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore")

eval_logger = logging.getLogger("lmms-eval")
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
    from deepseek_vl.utils.io import load_pil_images
except ImportError:
    eval_logger.debug("Deepseek-VL is not installed. Please install the environment to use this model.")


@register_model("deepseek_vl_chat")
class Deepseek_VL_Chat(lmms):
    """
    Deepseek_VL_Chat Model
    https://github.com/deepseek-ai/DeepSeek-VL
    """

    def __init__(
        self,
        pretrained: str = "deepseek-ai/deepseek-vl-7b-chat",
        device: Optional[str] = "cuda",
        batch_size: Optional[Union[int, str]] = 1,
        low_cpu_mem_usage=True,
        trust_remote_code: Optional[bool] = True,
        # use_cache=True,
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
        else:
            self._device = device

        self.vl_chat_processor = VLChatProcessor.from_pretrained(pretrained)
        self._tokenizer = self.vl_chat_processor.tokenizer

        self._model = AutoModelForCausalLM.from_pretrained(pretrained, torch_dtype=torch.bfloat16, device_map=self._device, trust_remote_code=True)

        self._config = self._model.config
        self.model.eval()
        # self.model.tie_weights() # TODO: confirm if we need this
        self.batch_size_per_gpu = int(batch_size)
        # self.use_cache = use_cache
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
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
            self.model.to(self._device)
            self._rank = 0
            self._word_size = 1

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
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

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    # should be deleted since max_new_tokens is decided by gen_kwargs not a model property
    # @property
    # def max_new_tokens(self) -> int:
    #     return 256

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

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, doc_to_target, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            # encode, pad, and truncate contexts for this batch
            if type(doc_to_target) == str:
                continuation = doc_to_target
            else:
                continuation = doc_to_target(self.task_dict[task][split][doc_id])
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)

            if "<image>" in contexts:
                # instruct blip does not expect the <image> tag
                contexts = contexts.replace("<image>", "")

            context_inputs = self.model.build_conversation_input_ids(self.tokenizer, query=context, history=[], images=visuals)  # chat mode
            inputs = self.model.build_conversation_input_ids(self.tokenizer, query=contexts + continuation, history=[], images=visuals)  # chat mode

            context_length = context_inputs['input_ids'].shape[0]-1
            context_inputs = {
                'input_ids': context_inputs['input_ids'].unsqueeze(0).to(self._device),
                'token_type_ids': context_inputs['token_type_ids'].unsqueeze(0).to(self._device),
                'attention_mask': context_inputs['attention_mask'].unsqueeze(0).to(self._device),
                'images': [[context_inputs['images'][0].to(self._device).to(torch.bfloat16)]],
            }
            inputs = {
                'input_ids': inputs['input_ids'].unsqueeze(0).to(self._device),
                'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to(self._device),
                'attention_mask': inputs['attention_mask'].unsqueeze(0).to(self._device),
                'images': [[inputs['images'][0].to(self._device).to(torch.bfloat16)]],
            }

            continuation_tokens = inputs['input_ids']
            attn_mask = torch.ones_like(continuation_tokens).to(self.model.device)
            labels = continuation_tokens.clone().to(self.model.device)
            labels[:, : context_length] = -100
            
            with torch.inference_mode():
                outputs = self.model(input_ids=continuation_tokens, labels=labels, attention_mask=attn_mask)
            loss = outputs.loss
            logits = outputs["logits"]
            greedy_tokens = logits.argmax(dim=-1)
            cont_toks = continuation_tokens[:, context_tokens.shape[1] :]
            greedy_tokens = greedy_tokens[:, context_tokens.shape[1] : continuation_tokens.shape[1]]  # [1, seq]
            max_equal = (greedy_tokens == cont_toks).all()
            res.append((float(loss.item()), bool(max_equal)))
            pbar.update(1)

        pbar.close()
        return res

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            visuals = self.flatten(visuals)
            visual_paths = []
            # save images to /tmp, name generated by hash function
            # deepseek-vl accept image path. Have to do it here....
            for visual in visuals:
                name = uuid.uuid4().hex.upper()[0:6]
                visual.save(f"/tmp/{name}.png")
                visual_paths.append(f"/tmp/{name}.png")
            assert len(visual_paths) == 1, 'the current Deepseek-VL model supports only single image'
            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]


            # Set default values for until and max_new_tokens
            until = [self.tokenizer.decode(self.eot_token_id)]

            # Update values from gen_kwargs if present
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str,list] but got {type(until)}")

            assert self.batch_size_per_gpu == 1, "Do not support batch_size_per_gpu > 1 for now"
            context = contexts[0]
            if "<image>" in context:
                # instruct blip does not expect the <image> tag
                context = context.replace("<image>", "")
            
            # Set trunction equals true here, the max length for qformer tokenizer is 512
            # if not truncate, some questions will cause size mismatch
            # The transformer implementation can't handle multi images for blip
            # Concat it into one image
            # if len(visuals) > 1:
            #     visuals = [process_images(visuals)]

            conversation = [
                {
                    "role": "User",
                    "content": "<image_placeholder>"+context,
                    "images": visual_paths,
                },
                {
                    "role": "Assistant",
                    "content": ""
                }
            ]
            # load images and prepare for inputs
            pil_images = load_pil_images(conversation)
            prepare_inputs = self.vl_chat_processor(
                conversations=conversation,
                images=pil_images,
                force_batchify=True
            ).to(self.model.device)
            # run image encoder to get the image embeddings
            inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)
            # preconfigure gen_kwargs with defaults
            # if "image_sizes" not in gen_kwargs:
            #     try:
            #         gen_kwargs["image_sizes"] = [visuals[0].size]
            #     except:
            #         gen_kwargs["image_sizes"] = None
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            # pad_token_id=tokenizer.eos_token_id
            # bos_token_id=tokenizer.bos_token_id
            # eos_token_id=tokenizer.eos_token_id
            if "pad_token_id" not in gen_kwargs:
                gen_kwargs["pad_token_id"]=self.tokenizer.eos_token_id
            if "bos_token_id" not in gen_kwargs:
                gen_kwargs["bos_token_id"]=self.tokenizer.bos_token_id
            if "eos_token_id" not in gen_kwargs: 
                gen_kwargs["eos_token_id"]=self.tokenizer.eos_token_id

            try:
                cont = self.model.language_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=prepare_inputs.attention_mask,
                    **gen_kwargs)
            except Exception as e:
                eval_logger.error(f"Error {e} in generating")
                cont = ""
            text_outputs = self.tokenizer.decode(cont[0].cpu().tolist(), skip_special_tokens=True).strip()
            res.append(text_outputs)
            print(text_outputs)
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), text_outputs)
            pbar.update(1)
        
        res = re_ords.get_original(res)

        pbar.close()
        return res
