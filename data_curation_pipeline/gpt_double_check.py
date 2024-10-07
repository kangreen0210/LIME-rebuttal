from gpt import *
import json
import os
import re
def load_json(path):
    ret = []
    with open(path,'r') as f:
        ret = json.load(f)
    return ret

def get_file_names(directory):
    # 获取指定目录下的所有文件和文件夹的名称
    file_names = os.listdir(directory)
    # 过滤掉目录，只保留文件
    file_names = [f for f in file_names if os.path.isfile(os.path.join(directory, f))]
    return file_names

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

def extract_judgement(sentence):
    match = re.search(r'<Your judgement>:\s*(YES|NO)', sentence)
    if match:
        return match.group(1)
    return None

if __name__ == '__main__':
    img_path = 'image_root_path'
    id_path = './static_result/taskfile.json'
    log_path = './logs/model/result.json'
    save_path = './double_check/ai2d.json'
    id_list = filter_func(id_path,threshold=1.0)[:]
    print(len(id_list))
    log = load_json(log_path)

    images_list = get_file_names(img_path) #
    #print(images_list)
    for i in range(len(images_list)):
        images_list[i] = img_path +'/' + images_list[i]
    #print(images_list)    
    ret = []
    for num in id_list:
        for i in range(len(log['logs'])):
            if log['logs'][i]['doc']["question_id"]==num:
                question = log['logs'][i]['doc']['question']
                answer = set(log['logs'][i]['doc']["reference_strs"])
                answer = list(answer)
                answers = ''
                answers = '; '.join(answer)
                text = '''
                Instruction: Please judge whether the <Answer> is the golden answer to the <Question>. 
                If it is, please reply YES, otherwise reply NO.
                <Question>: {question}
                <Answer>: {answer}
                <Your judgement>: <YES or NO>
                '''.format(question=question,answer=answers)
                # text = '''
                # Now there is an image captioning task. 
                # Please first describe the content of the image, then compare the image content with the provided captions. 
                # If the captions are suitable as captions for the image, please answer YES; if they are not suitable, please answer NO.
                # Respond with NO if any of the captions are unsuitable. Respond with YES only if all captions are suitable. You only need to reply with either YES or NO in <Your judgement>.
                # <Captions>: {answer}
                # <Desciption>: <Content of the image>
                # <Your judgement>: <ONLY YES or NO>
                # '''.format(answer=answers)
                response = infer(text, [img_path +'/'+num+'.jpg'])
                try:
                    extraction = extract_judgement(response)
                except:
                    extraction = 'wrong'
                log['logs'][i]['remain'] = extraction
                if extraction == 'YES':
                    ret.append(log['logs'][i]['doc']['question_id'])
                #log['logs'][i]['cot_judge'] = response
                print(response)
                break
    save_json = {"double_check":ret}
    with open(save_path,'w',encoding='utf-8') as f:
        json.dump(save_json,f,ensure_ascii=False,indent=2)


        
    
