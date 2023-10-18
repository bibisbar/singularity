import json
from collections import Counter
import ijson
import os
import random




file_path = '/nfs/data2/zhang/AnetQA/qa_train/qa_train_light.json'
#file_path = '/nfs/data2/zhang/AnetQA/qa_val/qa_val.json'
save_path = '/nfs/data2/zhang/AnetQA/qa_train/answer_list_light.json'

answer_list = [] #save all data

with open(save_path, 'w') as fp:
    with open(file_path, 'r') as f:
        data = ijson.items(f, 'item')
        for item in data:
            answer_list.append(item['answer'])
    #save the data into json file
    print(len(answer_list))
    #count the number of each answer
    answer_count = Counter(answer_list)
    #only save the key of the dict
    answer_list = list(answer_count.keys())
    print(len(answer_list))
    json.dump(answer_list, fp)