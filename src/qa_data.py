import json
from collections import Counter
import ijson
import os
import random



#file_path = '/nfs/data2/zhang/AnetQA/q_test/q_test.json'
#file_path = '/nfs/data2/zhang/AnetQA/qa_train/qa_train.json'
file_path = '/nfs/data2/zhang/AnetQA/qa_val/qa_val.json'
save_path = '/nfs/data2/zhang/AnetQA/qa_val/qa_val_light.json'

video_list = [] #save all data 
 # only save the data of current video and then sample 3 questions from it
# current_video_id = 'v_G3H3Gflf1SM'
# #'v_G2soQTiGL10'
length_of_each_video = []
with open(save_path, 'w') as fp:
    with open(file_path, 'r') as f:
        #use ijson to read the json file one by one to avoid memory error
        data = ijson.items(f, 'item')
        for item in data:
            # use random value to sample data, the probablity is 1/325
             if random.randint(1, 325) == 1:
                #save the data into video_list
                video_list.append(item)
                
                
    #save the data into json file
    print(len(video_list))
    json.dump(video_list, fp)
        
                
            
        
  
        
        

    

