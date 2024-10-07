
<img src="imgs/logo.png" alt="Logo" width="50"/>

# LIME: LESS IS MORE FOR MLLM EVALUATION
## Annoucement
- [2024-10.01]  📰 We have released both the dataset and the  data duration pipeline!
- [2024-09]  🍋 We have open-sourced the evaluation data and corresponding evaluation code for `LIME`. The data duration pipeline for LIME will be open-sourced within two weeks.

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

## data duration pipeline
The data duration pipeline consists of three parts: (1) Using open-source models as judges, (2) A semi-automated screening process, and (3) Eliminating answer leakage.


<div align="center">
<img src=imgs/data_curation_pipeline.png width=80% />
</div>

You can reproducte the process through the following steps: 
### 1.collect models result
By running this step, you can collect all models results.
```
python data_curation_pipeline/Models_Judges.py
```

### 2.classify samples' category
now we need to classify the difficulty level of each sample.
```
todo
```

### 3.gpt double check & human double check
To mitigate these potential errors and filter out totally incorrect questions, we use gpt double.
running data_curation_pipeline/gpt_double_check.py & data_curation_pipeline/Human_double_check.ipynb


### 4. Eliminating answer leakage.
For Eliminating answer leakage, we use pure-text models for evaluation, and the other processes are similar to those mentioned above.