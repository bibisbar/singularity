import csv
import numpy as np
import torch
import os

#this file is to calculate the similarity among postitive, negative text and video and then save the result in csv file

catogary = '/home/wiss/zhang/Jinhe/singularity/reb_eval/eccv_ori'
experiment = 'singularity_anet'
#manipulation = ['temporal_int', 'temporal_act', 'neighborhood_same_entity', 'neighborhood_diff_entity', 'counter_rel', 'counter_act', 'counter_int', 'counter_attr']
manipulation = ['temporal_contact_swap','temporal_action_swap','neighborhood_same_entity','neighborhood_different_entity','counter_spatial','counter_contact','counter_action','counter_attribute']
#manipulation = ['temporal_contact_swap','temporal_action_swap','neighborhood_same_entity','neighborhood_different_entity','counter_spatial']
#'counter_int', 'counter_attr' wait for adding

result_path = '/home/wiss/zhang/nfs/SPOT_similarity_feature/improvement'

for i in range(len(manipulation)):
    video_feat_path =catogary + '/image_feat_' + manipulation[i] + '_mani.pt'
    pos_text_feat_path = catogary + '/text_feat_' + manipulation[i] + '_mani.pt'
    neg_text_feat_path = '/home/wiss/zhang/Jinhe/singularity/reb_eval/eccv_mani'+'/text_feat_' + manipulation[i] + '_mani.pt'  #save the result into a csv file
    save_path = result_path + '/'+experiment+'_'+manipulation[i]
    
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
    save_path_pos_text2video = save_path + '_pos_text2video.pt'
    
    neg_text2video = torch.einsum("mld,nd->mln", video_feat, neg_text_feat).mean(1)   # (N, N)
    print('neg_text2video.shape', neg_text2video.shape)
    save_path_neg_text2video = save_path + '_neg_text2video.pt'
    #calculate similarity of pos_text2neg_text
    pos_text2neg_text = torch.matmul(pos_text_feat, neg_text_feat.T) # (N, N)
    print('pos_text2neg_text.shape', pos_text2neg_text.shape)
    save_path_pos_text2neg_text = save_path + '_pos_text2neg_text.pt'

    #save three pt
    torch.save(pos_text2video, save_path_pos_text2video)
    torch.save(neg_text2video, save_path_neg_text2video)
    torch.save(pos_text2neg_text, save_path_pos_text2neg_text)


    
    
    
