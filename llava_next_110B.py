import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from llava.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX
from typing import Dict, Optional, Sequence, List
import transformers
import re
from collections import defaultdict
from PIL import Image
import math
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
class imageDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        return data



def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def preprocess_qwen(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, max_len=2048, system_message: str = "You are a helpful assistant.") -> Dict:
    roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}

    im_start, im_end = tokenizer.additional_special_tokens_ids
    nl_tokens = tokenizer("\n").input_ids
    _system = tokenizer("system").input_ids + nl_tokens
    _user = tokenizer("user").input_ids + nl_tokens
    _assistant = tokenizer("assistant").input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []

    source = sources
    if roles[source[0]["from"]] != roles["human"]:
        source = source[1:]

    input_id, target = [], []
    system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
    input_id += system
    target += [im_start] + [IGNORE_INDEX] * (len(system) - 3) + [im_end] + nl_tokens
    assert len(input_id) == len(target)
    for j, sentence in enumerate(source):
        role = roles[sentence["from"]]
        if has_image and sentence["value"] is not None and "<image>" in sentence["value"]:
            num_image = len(re.findall(DEFAULT_IMAGE_TOKEN, sentence["value"]))
            texts = sentence["value"].split('<image>')
            _input_id = tokenizer(role).input_ids + nl_tokens 
            for i,text in enumerate(texts):
                _input_id += tokenizer(text).input_ids 
                if i<len(texts)-1:
                    _input_id += [IMAGE_TOKEN_INDEX] + nl_tokens
            _input_id += [im_end] + nl_tokens
            assert sum([i==IMAGE_TOKEN_INDEX for i in _input_id])==num_image
        else:
            if sentence["value"] is None:
                _input_id = tokenizer(role).input_ids + nl_tokens
            else:
                _input_id = tokenizer(role).input_ids + nl_tokens + tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
        input_id += _input_id
        if role == "<|im_start|>user":
            _target = [im_start] + [IGNORE_INDEX] * (len(_input_id) - 3) + [im_end] + nl_tokens
        elif role == "<|im_start|>assistant":
            _target = [im_start] + [IGNORE_INDEX] * len(tokenizer(role).input_ids) + _input_id[len(tokenizer(role).input_ids) + 1 : -2] + [im_end] + nl_tokens
        else:
            raise NotImplementedError
        target += _target

    input_ids.append(input_id)
    targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    return input_ids

def gen_samples(root_dir):

    for task in os.listdir(root_dir):
        for sample_dir in  os.listdir(os.path.join(root_dir, task)):
            if not os.path.isdir(os.path.join(root_dir, task, sample_dir)):
                continue
            yield  root_dir, task, sample_dir

def load_jsonl(path):
    with open(path, 'r') as json_file:
        json_list = list(json_file)

    res = {}
    for json_str in json_list:
        result = json.loads(json_str)
        # print(f"result: {result}")
        # print(isinstance(result, dict))
        res.update(result)

    # print(res)
    return res

def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    root_dir=args.root_dir

    # Data
    # with open(os.path.expanduser(args.question_file)) as f:
    # #     questions = json.load(f)
    # questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    # answers_file = os.path.expanduser(args.answers_file)
    # os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    # ans_file = open(answers_file, "w")
    # samples=gen_samples(root_dir)
    tasks = os.listdir(root_dir)
    tasks.reverse()
    for task in tasks:
        image_path=[]
        # if task=='POPE' or task =='scienceqa':continue
        if task!='infovqa':continue
    # for root, task, sample_dir in tqdm(samples):
        file_path=os.path.join(root_dir,task,'question.jsonl')
        with open(file_path,'r', encoding='utf-8') as jsonl_file:
            try:
                for line in jsonl_file:
                    data=json.loads(line)
                    image_path.append(data['image'])
            except Exception as e:
                continue
        # 实例化数据集
        dataset = imageDataset(image_path)
        output_dict={}
        # 使用 DataLoader 加载数据集
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        for image_files in tqdm(dataloader, desc="Progress image description"):
            text = DEFAULT_IMAGE_TOKEN+'\nPlease provide a description of the following image, You should consider elements in the image. '
            # json_path = os.path.join(root_dir, task, sample_dir, 'qa.jsonl')

            # json_data = load_jsonl(json_path)
            # answer = json_data['answer']

            # image1 = os.path.join(root_dir, task, sample_dir, 'image1.jpg')
            # image2 = os.path.join(root_dir, task, sample_dir, 'image2.jpg')
            # image_files=[image1,image2]
            cur_prompt = args.extra_prompt + text

            args.conv_mode = "qwen_1_5"

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], text)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            image_tensors = []
            batch_input_ids=[]
            description_dict=defaultdict(str)
            for idx,image_file in enumerate(image_files):
                input_ids = preprocess_qwen([{'from': 'human','value': prompt},{'from': 'gpt','value': None}], tokenizer, has_image=True).cuda().to(model.device)
                img_num = list(input_ids.squeeze()).count(IMAGE_TOKEN_INDEX)
                image = Image.open(image_file)
                # print(image_file)
                image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values']
                image_tensors.append(image_tensor.half().cuda().to(model.device))
                # image_tensors=[image_tensor.half().cuda().to(model.device)]
            # image_tensors = torch.cat(image_tensors, dim=0)
                batch_input_ids.append(input_ids)
                stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                keywords = [stop_str]
                stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

            input_ids=torch.stack(batch_input_ids).squeeze(1)
            image_tensors=torch.stack(image_tensors)
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensors,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    # no_repeat_ngram_size=3,
                    max_new_tokens=256,
                    use_cache=True)
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            for output,image_path in zip(outputs,image_files):
                output = output.strip()
                if output.endswith(stop_str):
                    output = output[:-len(stop_str)]
                output = output.strip()
                if output.startswith('<Description>: '):
                    output=output.replace('<Description>: ','')
                # print(output)
                output_dict[image_path] = output
        save_path=os.path.join(root_dir,task,'VD.json')
        with open(save_path,'w') as json_file:
            json_data=json.dumps(output_dict)
            json_file.write(json_data)
    #         key=f'image{idx+1}'
    #         description_dict[key]=output
    # with open(json_path,'a') as jsonl_file:
    #     json_data=json.dumps(description_dict)+'\n'
    #     jsonl_file.write(json_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/ML-A100/team/mm/zk/models/llava-next-110b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--extra-prompt", type=str, default="")
    parser.add_argument("--root-dir", type=str, default="/ML-A100/team/mm/zk/lmms-eval/vlms-bench-data-selected")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--test_size", type=int, default=10000000)
    args = parser.parse_args()

    eval_model(args)