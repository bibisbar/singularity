import os

#function: to change to pt file name suffix from _ori to _mani


def change_suffix(file_path):
    file_list = os.listdir(file_path)
    for file in file_list:
        if file.endswith('_ori.pt'):
            file_name = file[:-7]
            file_name = file_name+'_mani.pt'
            os.rename(file_path+'/'+file,file_path+'/'+file_name)
        else:
            continue

file_path = '/home/wiss/zhang/Jinhe/singularity/paper_results/ret_moviegraph/moviegraph_moviegraph_train_2_Seed2/moviegraph_eval_2_Seed2_mani'
change_suffix(file_path)