from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from PIL import Image
import requests
import torch

model_id = "/ML-A100/team/mm/zhangge/models/paligemma-3b-pt-224"
device = "cuda:0"
dtype = torch.bfloat16


model = PaliGemmaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=dtype,
    device_map=device,
    revision="bfloat16",
).eval()
processor = AutoProcessor.from_pretrained(model_id)


prompt = 'What is the difference of those two images:' 
image = Image.open('./image2.jpg').convert('RGB')

model_inputs = processor(text=[prompt], images=[image, image], return_tensors="pt").to(model.device)

print(model_inputs)

input()
input_len = model_inputs["input_ids"].shape[-1]

with torch.inference_mode():
    generation = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
    generation = generation[0][input_len:]
    decoded = processor.decode(generation, skip_special_tokens=True)
    print(decoded)