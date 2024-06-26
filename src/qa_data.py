import json
from collections import Counter
import ijson
import os
import random



# #file_path = '/nfs/data2/zhang/AnetQA/q_test/q_test.json'
# #file_path = '/nfs/data2/zhang/AnetQA/qa_train/qa_train.json'
# file_path = '/nfs/data2/zhang/AnetQA/qa_val/qa_val.json'
# save_path = '/nfs/data2/zhang/AnetQA/qa_val/qa_val_light.json'

# video_list = [] #save all data 
#  # only save the data of current video and then sample 3 questions from it
# # current_video_id = 'v_G3H3Gflf1SM'
# # #'v_G2soQTiGL10'
# length_of_each_video = []
# with open(save_path, 'w') as fp:
#     with open(file_path, 'r') as f:
#         #use ijson to read the json file one by one to avoid memory error
#         data = ijson.items(f, 'item')
#         for item in data:
#             # use random value to sample data, the probablity is 1/325
#              if random.randint(1, 325) == 1:
#                 #save the data into video_list
#                 video_list.append(item)
                
                
#     #save the data into json file
#     print(len(video_list))
#     json.dump(video_list, fp)
        
                

#file_path = '/nfs/data2/zhang/AnetQA/q_test/q_test.json'
#file_path = '/nfs/data2/zhang/AnetQA/qa_train/qa_train.json'
file_path = '/nfs/data2/zhang/AnetQA/qa_val/qa_val.json'
#save_path = '/nfs/data2/zhang/AnetQA/qa_val/qa_val_temporal.json'
save_path = '/nfs/data2/zhang/AnetQA/qa_val/qa_val_counter.json'
video_list = [] #save all data 
 # only save the data of current video and then sample 3 questions from it
# current_video_id = 'v_G3H3Gflf1SM'
# #'v_G2soQTiGL10'
length_of_each_video = []
count_actTime = 0
count_actLongerVerify = 0
count_actShorterVerify = 0
count_attrRelWhat = 0
count_attrWhat = 0
count_relWhat = 0
count_objRelExis = 0
count_actExist = 0
count_attrWhatChoose = 0
count_attrRelWhatChoose = 0
with open(save_path, 'w') as fp:
    with open(file_path, 'r') as f:
        #use ijson to read the json file one by one to avoid memory error
        data = ijson.items(f, 'item')
        #attrRelWhat  attrWhat  relWhat  objRelExis  actExist attrWhatChoose  attrRelWhatChoose
        for item in data:
            key = 'taxonomy'
            taxonomy = item[key]
            question_type = taxonomy['question_type']
            # if question_type == 'actTime':   
            #     #save the data into video_list
            #     video_list.append(item)
            #     count_actTime += 1
            # elif question_type == 'actLongerVerify':
            #     video_list.append(item)
            #     count_actLongerVerify += 1
            # elif question_type == 'actShorterVerify':
            #     video_list.append(item)
            #     count_actShorterVerify += 1
            if question_type == 'attrRelWhat':
                video_list.append(item)
                count_attrRelWhat += 1
            elif question_type == 'attrWhat':
                video_list.append(item)
                count_attrWhat += 1
            elif question_type == 'relWhat':
                video_list.append(item)
                count_relWhat += 1
            elif question_type == 'objRelExis':
                video_list.append(item)
                count_objRelExis += 1
            elif question_type == 'actExist':
                video_list.append(item)
                count_actExist += 1
            elif question_type == 'attrWhatChoose':
                video_list.append(item)
                count_attrWhatChoose += 1
            elif question_type == 'attrRelWhatChoose':
                video_list.append(item)
                count_attrRelWhatChoose += 1
    #finally only save 2000 data into json file randomly
    random.shuffle(video_list)
    video_list = video_list[:2000]
    
                
                
        
                
                
    #save the data into json file
    print(len(video_list))
    print('actTime: ', count_actTime)
    print('actLongerVerify: ', count_actLongerVerify)
    print('actShorterVerify: ', count_actShorterVerify)
    ###1484
        # actTime:  452
        # actLongerVerify:  516
        # actShorterVerify:  516
    print('attrRelWhat: ', count_attrRelWhat)
    print('attrWhat: ', count_attrWhat)
    print('relWhat: ', count_relWhat)
    print('objRelExis: ', count_objRelExis)
    print('actExist: ', count_actExist)
    print('attrWhatChoose: ', count_attrWhatChoose)
    print('attrRelWhatChoose: ', count_attrRelWhatChoose)
    json.dump(video_list, fp)
        
                
        
  
        
        

    

