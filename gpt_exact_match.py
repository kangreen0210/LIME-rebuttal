import json
from collections import Counter
import re
from collections import defaultdict
def extract_option_labels(text, options=None):
    if isinstance(text, dict):
        return "error"
    pattern = r"\(([A-D])\)"
    matches = re.findall(pattern, text)

    if not matches:
        pattern = r"\b([A-D])\b"
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

data_path='/ML-A100/team/mm/zk/MMIR_codebase/Mantis/gpt_result/gpt_4v.jsonl'
# save_path='/ML-A100/team/mm/zk/MMIR_codebase/Mantis/gpt_result/error_result_4o.jsonl'
task_list=['SubEvent','SimilarEvent','ShapeSimilarTo','HasProperty']
# 创建一个默认值为list的defaultdict
score_dict = defaultdict(list)
# with open(save_path,'a') as result_file:
with open(data_path,'r') as jsonl_file:
    for line in jsonl_file:
        data=json.loads(line)
        gt=data['answer']
        task=data['task']
        if data['message']==None:
            continue
        response=data['message']['data']['choices'][0]['message']['content']
        if task in task_list:
            result=extract_yes_no(response)
        else:
            result=extract_option_labels(response)
        if gt.lower()==result.lower():
            score_dict[task].append(1.0)
        else:
            score_dict[task].append(0.0)
                # data['message']=result
                # json_data=json.dumps(data,indent=2)+'\n'
                # result_file.write(json_data)


    print(score_dict)
    average_scores = {task: sum(scores) / len(scores) for task, scores in score_dict.items()}

    print("Score Dictionary:", score_dict)
    print("Average Scores:", average_scores)