import json

import re
from collections import Counter
from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from PIL import Image
import base64
from io import BytesIO
from loguru import logger
import statistics
import re

def remove_punctuation(text):
    # 定义正则表达式模式以匹配所有标点符号
    pattern = r'[^\w\s]'
    # 使用正则表达式替换标点符号为空字符串
    return re.sub(pattern, '', text)

PROMPT   = 'You will be giving one question and two images. Please answer the question using "Yes" or "No". \
                  Please only answer the question with Yes or No.\
                  questions: {question} \
                  Your answer is '


def siwei_bench_doc_to_text(doc):
    question=PROMPT.format(question=doc['question'])
    # question = PROMPT.format(doc["question"], doc["option1"], doc["option2"], doc["option3"], doc["option4"], doc["option5"], doc["option6"])
    # pre_prompt = model_specific_prompt_kwargs["pre_prompt"]
    # post_prompt = model_specific_prompt_kwargs["post_prompt"]
    # return f"{pre_prompt}{question}{post_prompt}"
    return question


# def siwei_bench_doc_to_visual(doc):
#     return [doc["image1"].convert("RGB"),doc["image2"].convert("RGB")]
def base64_to_pil_image(base64_string):
    img_bytes = base64.b64decode(base64_string)
    
    buffered = BytesIO(img_bytes)
    
    image = Image.open(buffered)
    # image.save('temp.png')
    return image

def siwei_bench_doc_to_visual(doc):
    # prompt = construct_prompt(doc)
    # image_tokens = re.findall(r"<image \d+>", prompt)
    # # Remove <> and  swap space as _
    # image_tokens = [image_token.strip("<>").replace(" ", "_") for image_token in image_tokens]
    visual = [base64_to_pil_image(doc['image1']).convert("RGB"),base64_to_pil_image(doc['image2']).convert("RGB")]   ######################################### load image from base64 encoding
    return visual
def extract_yes_no(response):
    # 定义正则表达式模式，匹配 "yes" 或 "no"
    pattern = r'\b(yes|no)\b'
    # 使用正则表达式搜索response中的匹配项
    matches = re.findall(pattern, response, re.IGNORECASE)
    if matches:
        counter = Counter(matches)
        most_common = counter.most_common()
        max_count = most_common[0][1]
        candidates = [item for item in most_common if item[1] == max_count]
        return candidates[-1][0]
    
    # 返回匹配项列表
    return None
    


def siwei_bench_process_results(doc, results):
    response = remove_punctuation(results[0])
    pred = response.lower().strip()
    gt_ans = doc["answer"].lower().strip()
    # idx=doc["idx"]
    assert gt_ans in ["yes", "no"]
    if pred not in ["yes", "no"]:
        pred=extract_yes_no(pred)
    score = 1.0 if pred == gt_ans else 0.0
    # predict = extract_option_labels(response, [doc['options']['A'], doc['options']['B'], doc['options']['C'], doc['options']['D']])
    # if doc['answer']==predict:
    #     accuracy=1.0
    # else:
    #     accuracy=0.0
    return {"exact_match": score,"submission": {"id": doc["idx"], "predict_answer": pred, "response": response}}


# def siwei_bench_aggregate_accuracy(results):
#     total_score = 0
#     for result in results:
#         total_score += result["score"]
#     avg_score = total_score / len(results)
#     return avg_score


# def siwei_bench_aggregate_precision(results):
#     true_positives = 0
#     false_positives = 0
#     for result in results:
#         pred = result["prediction"]
#         gt = result["ground_truth"]
#         if gt == "yes" and pred == "yes":
#             true_positives += 1
#         elif gt == "no" and pred == "yes":
#             false_positives += 1
#     precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
#     return precision


# def siwei_bench_aggregate_recall(results):
#     true_positives = 0
#     false_negatives = 0
#     for result in results:
#         pred = result["prediction"]
#         gt = result["ground_truth"]
#         if gt == "yes" and pred == "yes":
#             true_positives += 1
#         elif gt == "yes" and pred == "no":
#             false_negatives += 1
#     recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
#     return recall


# def siwei_bench_aggregate_f1_score(results):
#     precision = pope_aggregate_precision(results)
#     recall = pope_aggregate_recall(results)
#     f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
#     return f1_score


# def siwei_bench_aggregate_yes_ratio(results):
#     yes_count = 0
#     no_count = 0
#     for result in results:
#         gt = result["ground_truth"]
#         if gt == "yes":
#             yes_count += 1
#         elif gt == "no":
#             no_count += 1
#     yes_ratio = yes_count / (yes_count + no_count) if (yes_count + no_count) > 0 else 0
#     return yes_ratio
