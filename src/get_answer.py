import json
from collections import Counter
import ijson
import os
import random




file_path = '/nfs/data2/zhang/AnetQA/qa_train/qa_train_light.json'
file_path2 = '/nfs/data2/zhang/AnetQA/qa_val/qa_val_light.json'
file_path3 = '/nfs/data2/zhang/AnetQA/qa_val/qa_val_counter.json'
file_path4 = '/nfs/data2/zhang/AnetQA/qa_val/qa_val_temporal.json'
#file_path = '/nfs/data2/zhang/AnetQA/qa_val/qa_val.json'
save_path = '/nfs/data2/zhang/AnetQA/qa_train/answer_list_temporal.json'

answer_list = [] #save all data
answer_list2 = [] #save the data with more than 5 answers
answer_list3 = [] #save the data with more than 10 answers
answer_list4 = [] #save the data with more than 15 answers
all_answer_list = [] #save the data with more than 20 answers

with open(save_path, 'w') as fp:
    with open(file_path, 'r') as f:
        data = ijson.items(f, 'item')
        for item in data:
            answer_list.append(item['answer'])
    with open(file_path2, 'r') as f:
        data = ijson.items(f, 'item')
        for item in data:
            answer_list2.append(item['answer'])
    with open(file_path3, 'r') as f:
        data = ijson.items(f, 'item')
        for item in data:
            answer_list3.append(item['answer'])
    with open(file_path4, 'r') as f:
        data = ijson.items(f, 'item')
        for item in data:
            answer_list4.append(item['answer'])
    #save the data into json file
    print(len(answer_list))
    print(len(answer_list2))
    print(len(answer_list3))
    print(len(answer_list4))
    
    #save all answers into all_answer_list
    all_answer_list.extend(answer_list)
    all_answer_list.extend(answer_list2)
    all_answer_list.extend(answer_list3)
    all_answer_list.extend(answer_list4)
    #count the number of each answer
    answer_count1 = Counter(answer_list)
    #only save the key of the dict
    answer_list1 = list(answer_count1.keys())
    print(len(answer_list1))
    
    answer_count2 = Counter(answer_list2)
    answer_list2 = list(answer_count2.keys())
    print(len(answer_list2))
    
    answer_count3 = Counter(answer_list3)
    answer_list3 = list(answer_count3.keys())
    print(len(answer_list3))
    
    answer_count4 = Counter(answer_list4)
    answer_list4 = list(answer_count4.keys())
    print(len(answer_list4))
    
    answer_count = Counter(all_answer_list)
    answer_list = list(answer_count.keys())
    print(len(answer_list4))
    json.dump(answer_list4, fp)
    # 32050
    # 4695
    # 2000
    # 1484
    # 4567
    # 1096
    # 667
    # 4
    # 5156
