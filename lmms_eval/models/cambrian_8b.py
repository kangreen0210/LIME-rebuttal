import torch
import os
import sys
from tqdm import tqdm
from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from PIL import Image
from datetime import timedelta
from lmms_eval.api.registry import register_model
# from lmms_eval.models.model_utils.qwen.qwen_generate_utils import make_context
# 获取当前文件的目录
# current_dir = os.path.dirname(os.path.abspath(__file__))
# # 获取同级目录的路径
# parent_dir = os.path.dirname(current_dir)

# # 将同级目录添加到模块搜索路径
# sys.path.append(parent_dir)
# sys.path.append(current_dir)
# os.chdir('/ML-A100/team/mm/zk/lmms-eval/lmms_eval/models/cambrian')
from cambrian.model.builder import load_pretrained_model
from cambrian.conversation import conv_templates, SeparatorStyle
from cambrian.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from typing import List, Optional, Union, Tuple
import uuid
import warnings
from transformers import PreTrainedTokenizer
temperature = 0
CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
IMAGE_PLACEHOLDER = "<image-placeholder>"


conv_mode = "llama_3" 

warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore")

from loguru import logger as eval_logger
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel

def process(image, question, tokenizer, image_processor, model_config):
    qs = question

    if model_config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + str(qs)

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    image_size = [image.size]
    image_tensor = process_images([image], image_processor, model_config)

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    return input_ids, image_tensor, image_size, prompt

def make_context(
    tokenizer: PreTrainedTokenizer,
    query: str,
    history: List[Tuple[str, str]] = None,
    system: str = "",
    max_window_size: int = 6144,
    chat_format: str = "chatml",
):
    if history is None:
        history = []

    if chat_format == "chatml":
        im_start, im_end = "<|im_start|>", "<|im_end|>"
        im_start_tokens = [tokenizer.im_start_id]
        im_end_tokens = [tokenizer.im_end_id]
        nl_tokens = tokenizer.encode("\n")

        def _tokenize_str(role, content):
            return f"{role}\n{content}", tokenizer.encode(role, allowed_special=set(tokenizer.IMAGE_ST)) + nl_tokens + tokenizer.encode(content, allowed_special=set(tokenizer.IMAGE_ST))

        system_text, system_tokens_part = _tokenize_str("system", system)
        system_tokens = im_start_tokens + system_tokens_part + im_end_tokens

        raw_text = ""
        context_tokens = []

        for turn_query, turn_response in reversed(history):
            query_text, query_tokens_part = _tokenize_str("user", turn_query)
            query_tokens = im_start_tokens + query_tokens_part + im_end_tokens
            if turn_response is not None:
                response_text, response_tokens_part = _tokenize_str("assistant", turn_response)
                response_tokens = im_start_tokens + response_tokens_part + im_end_tokens

                next_context_tokens = nl_tokens + query_tokens + nl_tokens + response_tokens
                prev_chat = f"\n{im_start}{query_text}{im_end}\n{im_start}{response_text}{im_end}"
            else:
                next_context_tokens = nl_tokens + query_tokens + nl_tokens
                prev_chat = f"\n{im_start}{query_text}{im_end}\n"

            current_context_size = len(system_tokens) + len(next_context_tokens) + len(context_tokens)
            if current_context_size < max_window_size:
                context_tokens = next_context_tokens + context_tokens
                raw_text = prev_chat + raw_text
            else:
                break

        context_tokens = system_tokens + context_tokens
        raw_text = f"{im_start}{system_text}{im_end}" + raw_text
        context_tokens += nl_tokens + im_start_tokens + _tokenize_str("user", query)[1] + im_end_tokens + nl_tokens + im_start_tokens + tokenizer.encode("assistant") + nl_tokens
        raw_text += f"\n{im_start}user\n{query}{im_end}\n{im_start}assistant\n"

    elif chat_format == "raw":
        raw_text = query
        context_tokens = tokenizer.encode(raw_text)
    else:
        raise NotImplementedError(f"Unknown chat format {chat_format!r}")

    return raw_text, context_tokens



@register_model("cambrian_8b")
class Cambrian_8b(lmms):
    """
    cambrian_8b model
   https://huggingface.co/nyu-visionx/cambrian-8b
    """

    def __init__(
        self,
        pretrained: str = "/ML-A100/team/mm/zhangge/models/cambrian_8b",
        # pretrained: str = "nyu-visionx/cambrian-8b",
        device: Optional[str] = "cuda",
        device_map="auto",
        batch_size: Optional[Union[int, str]] = 1,
        trust_remote_code: Optional[bool] = True,
        use_cache=True,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"


        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        if accelerator.num_processes > 1 and device_map == "":
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = device
            self.device_map = device_map

        model_name = pretrained  # Assuming `pretrained` is a path or model id
        # self._model = AutoModelForCausalLM.from_pretrained(
        #     pretrained,
        #     torch_dtype="auto",
        #     device_map=self.device_map
        # )
        # self._model = AutoModel.from_pretrained(self.pretrained, device_map=self.device_map, trust_remote_code=True)
        # self._model = AutoModelForCausalLM.from_pretrained(pretrained, device_map=self._device, trust_remote_code=trust_remote_code).eval()
        # model_path = os.path.expanduser("nyu-visionx/cambrian-8b")
        model_name = get_model_name_from_path(pretrained)
        tokenizer, model, self.image_processor, context_len = load_pretrained_model(pretrained, None, model_name,device_map=self.device_map)
        self._model = model
        self._tokenizer = tokenizer
        self.tokenizer.pad_token_id=self.tokenizer.eos_token_id
        # device_map=self.device_map
        # self._tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self._model.eval()
        # self._config = self._model.config
        # self.model.tie_weights()
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            # If you want to use DistributedType.DEEPSPEED, you have to run accelerate config before using the model
            # Also, you have to select zero stage 0 (equivalent to DDP) in order to make the prepare model works
            # I tried to set different parameters in the kwargs to let default zero 2 stage works, but it didn't work.
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
                eval_logger.info("Detected that you are using DistributedType.DEEPSPEED. Make sure you run `accelerate config` and set zero stage to 0")
            if accelerator.distributed_type == DistributedType.FSDP or accelerator.distributed_type == DistributedType.DEEPSPEED:
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
        # if accelerator.num_processes > 1:
        #     self._model = accelerator.prepare(self._model)
        # else:
        #     self.model = torch.nn.DataParallel(self.model)
        #     self.model.to(self._device)

        self.accelerator = accelerator

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
            query = []
            visual_paths = []
            for visual in visuals:
                name = uuid.uuid4().hex.upper()[0:6]
                visual.save(f"/tmp/{name}.png")
                visual_paths.append(f"/tmp/{name}.png")
                query.append({"image": f"/tmp/{name}.png"})

            # Make a copy for query to save context (text that needs to be masked)
            context_query = [_ for _ in query]
            context_query.append({"text": contexts})
            query.append({"text": contexts + continuation})

            context_query = self.tokenizer.from_list_format(context_query)
            query = self.tokenizer.from_list_format(query)

            raw_contxt_text, context_tokens = make_context(
                self.tokenizer, context_query, history=None, system="You are a helpful assistant", max_window_size=self.model.generation_config.max_window_size, chat_format=self.model.generation_config.chat_format
            )
            context_tokens = torch.tensor([context_tokens])

            raw_continuation_text, continuation_tokens = make_context(
                self.tokenizer, query, history=None, system="You are a helpful assistant", max_window_size=self.model.generation_config.max_window_size, chat_format=self.model.generation_config.chat_format
            )
            continuation_tokens = torch.tensor([continuation_tokens]).to(self.model.device)
            attn_mask = torch.ones_like(continuation_tokens).to(self.model.device)
            labels = continuation_tokens.clone().to(self.model.device)
            labels[:, : context_tokens.shape[1]] = -100
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
            # qwen accept image path. Have to do it here....
            for visual in visuals:
                name = uuid.uuid4().hex.upper()[0:6]
                visual.save(f"/ML-A100/team/mm/zk/lmms-eval/lmms_eval/tmp/{name}.png")
                visual_paths.append(f"/ML-A100/team/mm/zk/lmms-eval/lmms_eval/tmp/{name}.png")

            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]

            # Set default values for until and max_new_tokens
            # until = [self.tokenizer.decode(self.eot_token_id)]

            # # Update values from gen_kwargs if present
            # if "until" in gen_kwargs:
            #     until = gen_kwargs.pop("until")
            #     if isinstance(until, str):
            #         until = [until]
            #     elif not isinstance(until, list):
            #         raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str,list] but got {type(until)}")
            # preconfigure gen_kwargs with defaults
            if "image_sizes" not in gen_kwargs:
                try:
                    gen_kwargs["image_sizes"] = [visuals[0].size]
                except:
                    gen_kwargs["image_sizes"] = None
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            # self.tokenizer.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eod_id
            # image_path = input("image path: ")
            image = Image.open(visual_paths[0]).convert('RGB')
            # question = input("question: ")
            question=contexts[0]
            # print(question)
            input_ids, image_tensor, image_sizes, prompt = process(image, question, self.tokenizer, self.image_processor, self.model.config)
            input_ids = input_ids.to(device='cuda', non_blocking=True)
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=image_sizes,
                    do_sample=True if temperature > 0 else False,
                    temperature=gen_kwargs["temperature"],
                    num_beams=gen_kwargs["num_beams"],
                    max_new_tokens=gen_kwargs["max_new_tokens"],
                    use_cache=True)
                text_outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            # cont_toks_list = cont.tolist()
            # for cont_toks, context in zip(cont_toks_list, contexts):
            #     # discard context + left-padding toks if using causal decoder-only LMM
            #     cont_toks = cont_toks[input_ids.input_ids.shape[1] :]
            #     text_outputs = self.tokenizer.decode(cont_toks, skip_special_tokens=True).strip()
            #     for term in until:
            #         if len(term) > 0:
            #             # ignore '' separator,
            #             # for seq2seq case where self.tok_decode(self.eot_token_id) = ''
            #             text_outputs = text_outputs.split(term)[0]

                # res.append(text_outputs)

                # self.cache_hook.add_partial("generate_until", (context, gen_kwargs), text_outputs)
                # remove visuals from tmp
                # print(text_outputs)
                for visual_path in visual_paths:
                    try:
                        os.remove(visual_path)
                    except:
                        pass
                output_text=[text_outputs]
                res.extend(output_text)
                pbar.update(1)
            # reorder this group of results back to original unsorted form
        # res = re_ords.get_original(res)
        

        pbar.close()
        return res
