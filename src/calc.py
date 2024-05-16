import os
import csv
import numpy as np
import torch

def compute_and_save_similarity(pos_dir, neg_dir, save_dir):
    pos_text2video_result = {}
    neg_text2video_result = {}

    # Ensure save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Manipulation types
    manipulation = ['temporal_contact_swap','temporal_action_swap','neighborhood_same_entity','neighborhood_different_entity','counter_spatial','counter_contact','counter_action','counter_attribute']
    
    # Load features and calculate similarity matrices
    for manip in manipulation:
        video_feat_path = pos_dir + '/image_feat_' + manip + '_mani.pt'
        pos_text_feat_path = pos_dir + '/text_feat_' + manip + '_mani.pt'
        neg_text_feat_path = neg_dir + '/text_feat_' + manip + '_mani.pt'
        
        # Load data
        video_feat = torch.load(video_feat_path, map_location=torch.device('cpu'))
        pos_text_feat = torch.load(pos_text_feat_path, map_location=torch.device('cpu'))
        neg_text_feat = torch.load(neg_text_feat_path, map_location=torch.device('cpu'))
        
        print('video_feat.shape', video_feat.shape)
        print('pos_text_feat.shape', pos_text_feat.shape)
        print('neg_text_feat.shape', neg_text_feat.shape)
        print('load data done')

        # Calculate similarity
        pos_text2video = torch.einsum("mld,nd->mln", video_feat, pos_text_feat).mean(1)
        neg_text2video = torch.einsum("mld,nd->mln", video_feat, neg_text_feat).mean(1)

        pos_text2video_result[manip] = pos_text2video
        neg_text2video_result[manip] = neg_text2video

    # Save alignment scores
    gamma = [-0.1, -0.09, -0.08, -0.07, -0.06, -0.05, -0.04, -0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 1.0]

    pos_dir_name = os.path.basename(pos_dir.strip('/'))
    # save_path = os.path.join(save_dir, f'{pos_dir_name}_alignment_score.csv')   
    save_path = os.path.join(save_dir, f'{pos_dir_name}_alignment_improv_score.csv')   
    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model', 'manipuation_type', 'seed', 'total_num', 'mean', 'std', 'max', 'min', ] + ['gamma_' + str(i) for i in gamma])
    model = f'{pos_dir_name}_improv'  # Replace with actual model name if needed
    seed = 0  # Replace with actual seed if needed

    for manip in manipulation:
        try:
            sim_p2t = pos_text2video_result[manip]
            sim_n2t = neg_text2video_result[manip]
        except KeyError:
            print(f'Key not found for manipulation type: {manip}')
            continue
        
        alignment = sim_p2t - sim_n2t
        alignment = np.diagonal(alignment.cpu().numpy())
        
        with open(save_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([model, manip, seed, alignment.shape, np.mean(alignment), np.std(alignment), np.max(alignment), np.min(alignment), *[np.sum(alignment > g) / len(alignment) for g in gamma]])

# Example usage
pos_dir = '/home/wiss/zhang/Jinhe/singularity/eccv_reb_improv/ret_anet/anet_anet_vit_8/anet_vit_8'  
neg_dir = pos_dir + '_mani'
save_dir = '/home/wiss/zhang/Jinhe/singularity/paper_results/rebbutal'  

compute_and_save_similarity(pos_dir, neg_dir, save_dir)
