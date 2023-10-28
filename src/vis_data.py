import json

file_path ='/home/wiss/zhang/Jinhe/singularity/Data/anetqa/anet_ret_train_1_neg.json'

#open it and read the length
with open(file_path,'r') as load_f:
    load_dict = json.load(load_f)
    print(len(load_dict))
    