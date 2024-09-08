import torch
import logging
from tqdm import tqdm
from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
# from lmms_eval.models.model_utils.qwen.qwen_generate_utils import make_context
from lmms_eval.tasks.mmmu.utils_group_img import process_images
from accelerate import Accelerator, DistributedType
from typing import List, Optional, Union, Tuple
import uuid
import yaml
import sys
from PIL import Image
import re
import torchaudio
import os
sys.path.append(sys.path[0] + '/..')
sys.path.append('/xpfs/public/gezhang/zk/ViLLaMA/')
os.chdir('/xpfs/public/gezhang/zk/ViLLaMA/')
import warnings
warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore")
from modeling.dalle_utils import create_d_vae
from modeling.speechtokenizer import SpeechTokenizer
eval_logger = logging.getLogger("lmms-eval")
from transformers import AutoModelForCausalLM, AutoTokenizer,LlamaTokenizer

# try:
#     from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
#     from deepseek_vl.utils.io import load_pil_images
# except ImportError:
#     eval_logger.debug("Deepseek-VL is not installed. Please install the environment to use this model.")


@register_model("MIO_batch")
class MIO_batch(lmms):
    """
    MIO Model
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

        # self.vl_chat_processor = VLChatProcessor.from_pretrained(pretrained)
        # self._tokenizer = self.vl_chat_processor.tokenizer

        # self._model = AutoModelForCausalLM.from_pretrained(pretrained, torch_dtype=torch.bfloat16, device_map=self._device, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
                    pretrained,
                    load_in_8bit=False,
                    torch_dtype=torch.float16,
                    # device_map="auto",
                    device_map=self._device
                    ).half().eval() 
        self._config = self._model.config
        self.model.eval()
        self._tokenizer = LlamaTokenizer.from_pretrained(
            pretrained)
        self.tokenizer.eos_token_id=7
        self.inference_config=yaml.load(open('/xpfs/public/gezhang/zk/ViLLaMA/configs/Evaluate_sft.yaml', 'r'), Loader=yaml.Loader)
        # self.tokenizer.pad_token_id = (0)
        # self.tokenizer.padding_side = "left" 

        self._vae = create_d_vae(weight_path=self.inference_config["visual_tokenizer_weight_path"],
                   d_vae_type=self.inference_config["visual_tokenizer_type"],
                   variant=self.inference_config['visual_tokenizer_variant'],
                   device=self._device
                )
        self._speech_tokenizer = SpeechTokenizer.load_from_checkpoint(self.inference_config['speech_tokenizer_config_path'],
                                                      self.inference_config['speech_tokenizer_weight_path']).to(self._device).half()

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
    def speech_tokenizer(self):
        return self._speech_tokenizer
    @property
    def vae(self):
        return self._vae

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
    
    def get_mm_input(self,text_promt,images,speechs):
        flattened_image_ids = []
        for image_path in images:
            image=Image.open(image_path).convert('RGB')
            image_id=self.vae.encode_image(image_pil=image)[0]
            # image_id=vae.encode(image_id)
        # images=[Image.open(image_path).convert('RGB') for image_path in images]
        # image_ids = vae.encode(images[0])
        # image_ids= image_ids.view(-1, 14)
        
        # for image_id in image_ids:
            list=image_id.tolist()
            result_string = ''.join(f'<img{item}>' for item in list)
            flattened_image_ids.append(result_string)
        pattern_str = '\[IMG\]'
        pattern = re.compile(pattern_str)
        text_promt = pattern.sub(lambda x: flattened_image_ids.pop(0), text_promt)
        assert flattened_image_ids == [], \
            f"Assertion failed: Not all image_ids have been replaced into the context. The remaining image_ids are:({flattened_image_ids})"
        batch_wav=[]
        speech_list=[]
        if speechs==[]:pass
        else:
            for each in speechs:
                wav, sr = torchaudio.load(each)
                if wav.shape[0] > 1:
                    wav = wav[:1, ]
                if sr != self.speech_tokenizer.sample_rate:
                    wav = torchaudio.functional.resample(wav, sr, self.speech_tokenizer.sample_rate)
                    wav=wav.unsqueeze(0).to(self._device)
                # batch_wav.append(wav.unsqueeze(0).to(device))
                with torch.no_grad():
                    codes = self.speech_tokenizer.encode(wav.half()) # codes: (n_q, B, T)
                    RVQ_1 = codes[0, 0, :].tolist()
                    spch_string = ''.join(f'<spch{item}>' for item in RVQ_1)
                    speech_list.append(spch_string)
                pattern_spch='\[SPCH\]'
                pattern = re.compile(pattern_spch)
                text_promt = pattern.sub(lambda x: speech_list.pop(0), text_promt)
            assert speech_list == [], \
                f"Assertion failed: Not all speech_list have been replaced into the context. The remaining speech_list are:({speech_list})"
        return text_promt
    def get_chat_template_input(self,question_prompt):
        chat_template = \
        """<|startoftext|><|im_start|>system
        {system_message}<|im_end|>
        <|im_start|>user
        {question}<|im_end|>
        <|im_start|>assistant
        """
        system_message="You are MIO, an AI assistant capable of understanding and generating images, text, videos, and speech, selecting the appropriate modality according to the context."
        prompt=chat_template.format(system_message=system_message,question=question_prompt) 
        return prompt
    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        # res = []
        # pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        # for contexts, doc_to_target, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
        #     # encode, pad, and truncate contexts for this batch
        #     if type(doc_to_target) == str:
        #         continuation = doc_to_target
        #     else:
        #         continuation = doc_to_target(self.task_dict[task][split][doc_id])
        #     visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
        #     visuals = self.flatten(visuals)

        #     if "<image>" in contexts:
        #         # instruct blip does not expect the <image> tag
        #         contexts = contexts.replace("<image>", "")

        #     context_inputs = self.model.build_conversation_input_ids(self.tokenizer, query=context, history=[], images=visuals)  # chat mode
        #     inputs = self.model.build_conversation_input_ids(self.tokenizer, query=contexts + continuation, history=[], images=visuals)  # chat mode

        #     context_length = context_inputs['input_ids'].shape[0]-1
        #     context_inputs = {
        #         'input_ids': context_inputs['input_ids'].unsqueeze(0).to(self._device),
        #         'token_type_ids': context_inputs['token_type_ids'].unsqueeze(0).to(self._device),
        #         'attention_mask': context_inputs['attention_mask'].unsqueeze(0).to(self._device),
        #         'images': [[context_inputs['images'][0].to(self._device).to(torch.bfloat16)]],
        #     }
        #     inputs = {
        #         'input_ids': inputs['input_ids'].unsqueeze(0).to(self._device),
        #         'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to(self._device),
        #         'attention_mask': inputs['attention_mask'].unsqueeze(0).to(self._device),
        #         'images': [[inputs['images'][0].to(self._device).to(torch.bfloat16)]],
        #     }

        #     continuation_tokens = inputs['input_ids']
        #     attn_mask = torch.ones_like(continuation_tokens).to(self.model.device)
        #     labels = continuation_tokens.clone().to(self.model.device)
        #     labels[:, : context_length] = -100
            
        #     with torch.inference_mode():
        #         outputs = self.model(input_ids=continuation_tokens, labels=labels, attention_mask=attn_mask)
        #     loss = outputs.loss
        #     logits = outputs["logits"]
        #     greedy_tokens = logits.argmax(dim=-1)
        #     cont_toks = continuation_tokens[:, context_tokens.shape[1] :]
        #     greedy_tokens = greedy_tokens[:, context_tokens.shape[1] : continuation_tokens.shape[1]]  # [1, seq]
        #     max_equal = (greedy_tokens == cont_toks).all()
        #     res.append((float(loss.item()), bool(max_equal)))
        #     pbar.update(1)

        # pbar.close()
        return None

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
            # print(doc_to_visual)
            visuals_all=[]
            for doc_v in doc_to_visual:
                visual=[doc_v(self.task_dict[task][split][ids]) for ids in doc_id]
                visual = self.flatten(visual)
                visuals_all.append(visual)
            # visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            # visuals = self.flatten(visuals)
            
            # if len(visuals) > 1:
            #     visuals = [process_images(visuals)]
            # save images to /tmp, name generated by hash function
            # deepseek-vl accept image path. Have to do it here....
            visual_paths_all=[]
            for visuals in visuals_all:
                visual_paths = []
                for visual in visuals:
                    name = uuid.uuid4().hex.upper()[0:6]
                    visual.save(f"/tmp/{name}.png")
                    visual_paths.append(f"/tmp/{name}.png")
                visual_paths_all.append(visual_paths)
            # assert len(visual_paths) == 1, 'the current Deepseek-VL model supports only single image'
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

            # assert self.batch_size_per_gpu == 1, "Do not support batch_size_per_gpu > 1 for now"
            # context = contexts[0]
            query_list=[]
            for context,visual_paths in zip(contexts,visual_paths_all):
                image_token='<image>[IMG]</image>'*len(visual_paths)
                text_promt=image_token+'\n{question}'.format(question=context)
                text_promt=self.get_mm_input(text_promt,visual_paths,[])
                query=self.get_chat_template_input(text_promt)
                if not query.startswith(self.tokenizer.bos_token):
                    query = self.tokenizer.bos_token + query
            query_list.append(query)
                
            # print("input:"+query)
            inputs = self.tokenizer(query_list, padding='longest', truncation=True, max_length=1124, return_tensors='pt')
            input_ids = inputs['input_ids'].to(self._device)
            attention_mask = inputs['attention_mask'].to(self._device)
            gen_kwargs = {
                            'num_beams': 5,
                            'temperature': 0.0,
                            'top_p': 1.0,
                            'do_sample': False,
                            'repetition_penalty': 1.0,
                            'guidance_scale': 1.0,
                            'max_new_tokens': 4,
                            'length_penalty': 1.0
                        }
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 4
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1
            if "pad_token_id" not in gen_kwargs:
                gen_kwargs["pad_token_id"]=self.tokenizer.eos_token_id
            if "bos_token_id" not in gen_kwargs:
                gen_kwargs["bos_token_id"]=self.tokenizer.bos_token_id
            if "eos_token_id" not in gen_kwargs: 
                gen_kwargs["eos_token_id"]=self.tokenizer.eos_token_id
            if 'image_aspect_ratio' in gen_kwargs:
                del gen_kwargs['image_aspect_ratio']


            # try:
            cont = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **gen_kwargs
            )
            # except Exception as e:
            #     eval_logger.error(f"Error {e} in generating")
            #     cont = ""
            # print(cont.size)
            for i in range(cont.size(0)):
                # input_seq = self.tokenizer.decode(inputs['input_ids'][i], skip_special_tokens=False)
                text_output = self.tokenizer.decode(cont[i], skip_special_tokens=False)
            # text_outputs = self.tokenizer.decode(cont[0].cpu().tolist(), skip_special_tokens=True).strip()
            # text_outputs=self.tokenizer.batch_decode(cont, skip_special_tokens=False)
            # generated_tokens = cont[0][input_ids.shape[1]:]
            # text_outputs = self.tokenizer.decode(generated_tokens, skip_special_tokens=False)

                if '<|im_start|>assistant\n' in text_output:
                    text_output=text_output.split('<|im_start|>assistant\n')[1]
                    if '<|im_end|>' in text_output:
                        text_output=text_output.split("<|im_end|>")[0].strip()
            # print("output:"+text_outputs)
                # print(text_output)
                res.append(text_output)
                self.cache_hook.add_partial("generate_until", (contexts[i], gen_kwargs), text_output)
            pbar.update(self.batch_size_per_gpu*self._world_size)
        
        res = re_ords.get_original(res)

        pbar.close()
        return res
