import csv
import numpy as np
import torch
import os
import matplotlib.pyplot as plt

catogary = '/home/wiss/zhang/Jinhe/singularity/paper_results/ret_moviegraph/moviegraph_moviegraph_train_1_Seed3/moviegraph_eval_1_Seed3'
experiment = 'moviegraph_eval_1_Seed3'
manipulation = ['temporal_int', 'temporal_act', 'neighborhood_same_entity', 'neighborhood_diff_entity', 'counter_rel', 'counter_act', 'counter_int', 'counter_attr']


for i in range(len(manipulation)):
    # video_feat_path =catogary + '/image_feat_' + manipulation[i] + '_ori.pt'
    # pos_text_feat_path = catogary + '/text_feat_' + manipulation[i] + '_ori.pt'
    # neg_text_feat_path = catogary + '_mani'+'/text_feat_' + manipulation[i] + '_mani.pt'
    # #load data
    # video_feat = torch.load(video_feat_path,map_location=torch.device('cpu'))
    # pos_text_feat = torch.load(pos_text_feat_path,map_location=torch.device('cpu'))
    # neg_text_feat = torch.load(neg_text_feat_path,map_location=torch.device('cpu'))
    
    # pos_text2video = torch.einsum("mld,nd->mln", video_feat, pos_text_feat).mean(1)
    # neg_text2video = torch.einsum("mld,nd->mln", video_feat, neg_text_feat).mean(1)
    
    # #save two similarity matrix into pth file
    # save_path = catogary + '/similarity'+ '/'+experiment
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    # torch.save(pos_text2video, save_path + '/pos_text2video_' + manipulation[i] + '.pt')
    # torch.save(neg_text2video, save_path + '/neg_text2video_' + manipulation[i] + '.pt')
    save_path_pos='/home/wiss/zhang/Jinhe/singularity/paper_results/ret_moviegraph/moviegraph_moviegraph_train_1_Seed3/moviegraph_eval_1_Seed3/similarity/moviegraph_eval_1_Seed3/pos_text2video_temporal_int.pt'
    save_path_neg='000'
    pos_text2video = torch.load(save_path_pos,map_location=torch.device('cpu'))
    neg_text2video = torch.load(save_path_neg,map_location=torch.device('cpu'))
    # size =600*600
    #check if pos_text2video and neg_text2video are the same size
    print('pos_text2video.shape', pos_text2video.shape)
    print('neg_text2video.shape', neg_text2video.shape)
    if pos_text2video.shape != neg_text2video.shape:
        #throw an exception
        raise Exception('pos_text2video and neg_text2video are not the same size')
    #calculate the mean and var of similarity matrix
    #first get the diagonal value
    pos_text2video_diag = pos_text2video.diag()
    neg_text2video_diag = neg_text2video.diag()
    #size 600*1
    
    #convert tensor to numpy
    pos_text2video_diag = pos_text2video_diag.numpy()
    neg_text2video_diag = neg_text2video_diag.numpy()
    
    #plot the histogram of diagonal value
    plt.hist(pos_text2video_diag, bins=100, density=True, facecolor='blue', alpha=0.5)
    plt.hist(neg_text2video_diag, bins=100, density=True, facecolor='red', alpha=0.5)
    plt.title('mg_temporal_int')
    plt.legend(['pos_text2video', 'neg_text2video'])
    plt.show()
    
    #save the histogram into png file
    save_path = catogary + '/histogram'+ '/'+experiment
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    plt.savefig(save_path + '/histogram_' + manipulation[i] + '.png')
   
