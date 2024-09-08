import json

import re
from collections import Counter
from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from PIL import Image
import base64
from io import BytesIO
from loguru import logger
import statistics

PROMPT  = 'You will be giving one question, two images , descriptions of each images, and three answers, one of them is correct. Please choose one of the three answers.\
            please only answer the question with A, B, C.\
            description of image1:{image1},description of image2:{image2},\
            questions: {question} \
            answer:  A: {A}  B: {B}  C: {C}\
            Your answer is '


def siwei_bench_doc_to_text(doc):
    question=PROMPT.format(image1=doc['image1_VD'],image2=doc['image2_VD'],question=doc['question'],A=doc['options']['A'], B=doc['options']['B'], C=doc['options']['C'])
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


def extract_option_labels(text, options=None):
    if isinstance(text, dict):
        return "error"
    pattern = r"\(([A-C])\)"
    matches = re.findall(pattern, text)

    if not matches:
        pattern = r"\b([A-C])\b"
        matches = re.findall(pattern, text)

    if matches:
        counter = Counter(matches)
        most_common = counter.most_common()
        max_count = most_common[0][1]
        candidates = [item for item in most_common if item[1] == max_count]
        return candidates[-1][0]
    else:
        if options:
            counter = Counter()
            for i, option in enumerate(options, start=1):
                label = chr(64 + i)
                option_stripped = option.strip()
                if option_stripped in text:
                    counter[label] += 1
                elif text in option:
                    counter[label] += 1
            if counter:
                most_common = counter.most_common()
                max_count = most_common[0][1]
                candidates = [item for item in most_common if item[1] == max_count]
                return candidates[-1][0]
    return None


def siwei_bench_process_results(doc, results):
    response = results[0]
    predict = extract_option_labels(response, [doc['options']['A'], doc['options']['B'], doc['options']['C']])
    if doc['answer']==predict:
        accuracy=1.0
    else:
        accuracy=0.0
    return {"exact_match": accuracy,"submission": {"id": doc["idx"], "predict_answer": predict, "response": response}}


def siwei_bench_aggregate_submissions(results, args):
    file = generate_submission_file("siwei_bench_test_for_submission.json", args)
    with open(file, "w") as f:
        json.dump(results, f, indent=4)
    logger.info(f"Results saved to {file}")
