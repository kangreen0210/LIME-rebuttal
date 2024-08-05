import json
def filter_func(task_path,threshold=1.0):
    '''
        param:
            task_path:统计结果路径
            Threshold:统计分数的阈值
        return:
            filter_list:需要double check的题目信息  
    '''
    filter_list=[]
    with open(task_path) as json_file:
        data=json.load(json_file)
        for k,model_scores in data.items():
            model_scores=list(model_scores.values())
            count = sum(1 for x in model_scores if x >= threshold)
            if count==0:filter_list.append(k)
    return filter_list

def difficult_filter_func(task_path,threshold=1.0):
    '''
        param:
            task_path:统计结果路径
            Threshold:统计分数的阈值
        return:
            easy_list:标记为easy的question
            middle_list:标记为middle的question
            hard_list:标记为hard的question
    '''
    easy_list=[]
    middle_list=[]
    hard_list=[]
    with open(task_path) as json_file:
        data=json.load(json_file)
        for k,model_scores in data.items():
            model_scores=list(model_scores.values())
            count = sum(1 for x in model_scores if x >= threshold)
            if 0<count<3:hard_list.append(k)
            elif 3<=count<5:middle_list.append(k)
            elif 5<=count<=9:easy_list.append(k)
            # elif count==9:easy_list.append(k)
    return easy_list,middle_list,hard_list


if __name__=='__main__':
    # filter_list=filter_func('/ML-A100/team/mm/zk/lmms-eval/static_result/result_ok.json',threshold=0.6)
    # print(filter_list)
    # print(f'the length of list is {len(filter_list)}')
    filter_list=filter_func('/ML-A100/team/mm/zk/lmms-eval/static_result/result_ai2d_text_only_VD.json',threshold=1.0)
    # filter_list=filter_func('/ML-A100/team/mm/zk/lmms-eval/static_result/result_nocaps_val.json',threshold=0.2)
    # print(filter_list)
    print(f'the length of list is {len(filter_list)}')
    easy_list,middle_list,hard_list=difficult_filter_func('/ML-A100/team/mm/zk/lmms-eval/static_result/result_ai2d_text_only_VD.json',threshold=1.0)
    print(f'the length of easy list is {len(easy_list)}')
    print(f'the length of middle list is {len(middle_list)}')
    print(f'the length of hard list is {len(hard_list)}')
    # print(easy_list)
    # print(easy_list,middle_list,hard_list)
