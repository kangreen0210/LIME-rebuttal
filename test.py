import json
with open('/ML-A100/team/mm/zhangge/domain_data_pipeline/llm_label_data_pipeline/fasttext_seed_data/chemistry/pos/pos.jsonl') as jsonl_file:
    for line in jsonl_file:
        data=json.loads(line)
        print(data)
        break