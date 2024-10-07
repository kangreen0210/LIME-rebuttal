import sys
import os
import json
from tqdm import tqdm
sys.path.append('.')
from lmms_eval.tasks.pope.utils import pope_process_results
from collections import defaultdict
import matplotlib.pyplot as plt
import re
from collections import Counter
task='ai2d'

task_type=task.split('_')[0]
print(task_type)
all_model_result={}
log_path='./logs'
all_task_path=[]
for entry in os.listdir(log_path):
    dir_path = os.path.join(log_path, entry)
    if 'all_task' in dir_path:
        all_task_path.append(dir_path)
    if task_type in dir_path:
        all_task_path.append(dir_path)

coco_file={}
# print(all_task_path)
for path in all_task_path:
    if 'all_task' in  path:
        pattern = r"logs/(.*?)_all_task"
    elif task_type in path:
        pattern = r"logs/(.*?)_"+task_type
        # print('yes')
    match = re.search(pattern, path)
    if match:
        model = match.group(1)
    else:
        continue
    for root, dirs, files in os.walk(path):
        for file in files:
            if task in file:
                coco_file[model]=os.path.join(path, file)
result_dic = defaultdict(dict)

for model,process_file in coco_file.items():
    if model not in ['llava1.5_7b','llava1.5_13b','llava1.6','llava_llama3_8b','xcomposer2_4khd','minicpm','instructblip','idefics2','internvl']:
        continue
    with open(process_file) as file:
        json_file=json.load(file)
        result=json_file['logs']
        for item in tqdm(result):
            doc=item['doc']
            question_id=item['doc_id']
            score=item['exact_match']
            result_dic[question_id][model]=score
with open('static_result/result_ai2d_new.json', 'w') as json_file:
    json.dump(result_dic, json_file, indent=4)