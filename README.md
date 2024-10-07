
<img src="imgs/logo.png" alt="Logo" width="50"/>

# LIME: LESS IS MORE FOR MLLM EVALUATION
## Annoucement
- [2024-09]  🍋We have open-sourced the evaluation data and corresponding evaluation code for `LIME`. The data duration pipeline for LIME will be open-sourced within two weeks, 
## Introduction
We use a gneral data process pipeline and curate a LIME, which contains 9403 samples and is refined across 10 tasks within 6 domains. We select six major tasks in the Multimodal domain and use 9 MLLMs to refine those 10 benchmarks within the corresponding domain.
<img src=imgs/task_static.png width=100% />

## How to use LIME

### 1. Installation
For quickly start using LIME, we recommend following the [lmms-eval](https://lmms-lab.github.io/)  tutorial to quickly deploy the evaluation environment.

also you can install by following steps
```bash
git clone https://anonymous.4open.science/r/LIME-49CD
cd lmms-eval
pip install -e .
```
### 2. download dataset from huggingface
coming soon

### 3.run evaluation
You can run scripts for all the subtasks included in LIME-M using the following method. 
```
accelerate launch --num_processes=8 -m lmms_eval --model internvl2 --model_args pretrained=model_path  --tasks textcaps_suit,ok_vqa_suit,coco_cap_suit,textvqa_suit,chartqa_suit,pope_suit,infovqa_suit,ai2d_suit,ocrbench_suit,scienceqa_img_suit  --batch_size 1 --log_samples --log_samples_suffix internvl2_suits --summary True --output_path output_path
```
model_path refers to the local storage path of the model, and output_path refers to the location where the final logs are stored.

## overall Leadboard
<div align="center">
<img src=imgs/main_result.png width=100% />
</div>

<!-- ## data duration pipeline
our data duration pipeline will be release in two weeks!

<div align="center">
<img src=imgs/pipeline.png width=80% />
</div> -->
