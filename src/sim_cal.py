import csv
import numpy as np
import torch
import os

#this file is to calculate the similarity among postitive, negative text and video and then save the result in csv file

catogary = '/home/wiss/zhang/Jinhe/singularity/paper_results/ret_anet/anet_anet_train_3_Seed3/anet_eval_3_Seed3'
experiment = 'anet_eval_3_Seed3'
#manipulation = ['temporal_int', 'temporal_act', 'neighborhood_same_entity', 'neighborhood_diff_entity', 'counter_rel', 'counter_act', 'counter_int', 'counter_attr']
manipulation = ['temporal_contact_swap','temporal_action_swap','neighborhood_same_entity','neighborhood_different_entity','counter_spatial','counter_contact','counter_action','counter_attribute']
#'counter_int', 'counter_attr' wait for adding

result_path = '/home/wiss/zhang/nfs/video_prober/singularity/anetqa'
pos_text2video_result = {}
neg_text2video_result = {}
pos_text2neg_text_result = {}
for i in range(len(manipulation)):
    video_feat_path =catogary + '/image_feat_' + manipulation[i] + '_ori.pt'
    pos_text_feat_path = catogary + '/text_feat_' + manipulation[i] + '_ori.pt'
    neg_text_feat_path = catogary + '_mani'+'/text_feat_' + manipulation[i] + '_mani.pt'
    #load data
    video_feat = torch.load(video_feat_path,map_location=torch.device('cpu'))
    pos_text_feat = torch.load(pos_text_feat_path,map_location=torch.device('cpu'))
    neg_text_feat = torch.load(neg_text_feat_path,map_location=torch.device('cpu'))
    print('video_feat.shape', video_feat.shape)
    print('pos_text_feat.shape', pos_text_feat.shape)
    print('neg_text_feat.shape', neg_text_feat.shape)
    print('load data done')
    #calculate similarity
    pos_text2video = torch.einsum("mld,nd->mln", video_feat, pos_text_feat).mean(1)   # (N, N)
    print('pos_text2video.shape', pos_text2video.shape)
    #get the mean value and var value in diagonal
    pos_text2video_mean = pos_text2video.diag().mean()
    pos_text2video_var = pos_text2video.diag().var()
    #save the result into a dict, the key is the manipulation name,the subkey is the mean and var
    pos_text2video_result[manipulation[i]] = {'pos_text2video_mean':pos_text2video_mean, 'pos_text2video_var':pos_text2video_var}
    
    neg_text2video = torch.einsum("mld,nd->mln", video_feat, neg_text_feat).mean(1)   # (N, N)
    print('neg_text2video.shape', neg_text2video.shape)
    #get the mean value and var value in diagonal
    neg_text2video_mean = neg_text2video.diag().mean()
    neg_text2video_var = neg_text2video.diag().var()
    #save the result into a dict, the key is the manipulation name,the subkey is the mean and var
    neg_text2video_result[manipulation[i]] = {'neg_text2video_mean':neg_text2video_mean, 'neg_text2video_var':neg_text2video_var}
    
    #calculate similarity of pos_text2neg_text
    pos_text2neg_text = torch.matmul(pos_text_feat, neg_text_feat.T) # (N, N)
    print('pos_text2neg_text.shape', pos_text2neg_text.shape)
    #get the mean value and var value in diagonal
    pos_text2neg_text_mean = pos_text2neg_text.diag().mean()
    pos_text2neg_text_var = pos_text2neg_text.diag().var()
    #save the result into a dict, the key is the manipulation name,the subkey is the mean and var
    pos_text2neg_text_result[manipulation[i]] = {'pos_text2neg_text_mean':pos_text2neg_text_mean, 'pos_text2neg_text_var':pos_text2neg_text_var}
    
#save the result into a csv file
save_path = result_path + '/'+experiment
if not os.path.exists(save_path):
    os.makedirs(save_path)
with open(save_path+'/pos_text2video_result.csv', 'w') as f:
    writer = csv.writer(f)
    for k, v in pos_text2video_result.items():
        writer.writerow([k, v])
with open(save_path+ '/neg_text2video_result.csv', 'w') as f:
    writer = csv.writer(f)
    for k, v in neg_text2video_result.items():
        writer.writerow([k, v])
with open(save_path+ '/pos_text2neg_text_result.csv', 'w') as f:
    writer = csv.writer(f)
    for k, v in pos_text2neg_text_result.items():
        writer.writerow([k, v])
    
    
    
