---
license: unknown
dataset_info:
  features:
  - name: Task
    dtype: string
  - name: QA_type
    dtype: string
  - name: question
    dtype: string
  - name: image1
    dtype: image
  - name: image2
    dtype: image
  - name: options
    dtype: string
  - name: answer
    dtype: string
  splits:
  - name: train
    num_bytes: 587125417.48
    num_examples: 1024
  download_size: 570636511
  dataset_size: 587125417.48
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---
