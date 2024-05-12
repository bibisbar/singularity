import json
import os


#this file is to test if the delta_p is higher than baseline


#read baseline

xclip_file_path = '/home/wiss/zhang/Yanyu/infovis/recall/recall_sin_anet_delta_reb.json'

# experiment_ori = 'anet_reb_beta_0.25_anet'
experiment_ori = 'anet_reb_beta_0.25_lr_0.0001_anet'


#read xclip baseline
with open(xclip_file_path, 'r') as f:
    xclip = json.load(f)
    
delta_p_all = {}
diff_delta_p_all = {}
count = {}
for i in range(2,16):
    count_t2v_r1 = 0
    count_t2v_r5 = 0
    count_v2t_r1 = 0
    count_v2t_r5 = 0
    ori_file = '/home/wiss/zhang/Jinhe/singularity/reb_eval/'+experiment_ori+'_'+ str(i) +'_ori/eval_res.json'
    mani_file = '/home/wiss/zhang/Jinhe/singularity/reb_eval/'+experiment_ori+'_'+str(i)+'_mani/eval_res.json'
        
    for key in xclip:
        
        t2v_r1 = xclip[key]['t2v'][0]
        t2v_r5 = xclip[key]['t2v'][1]
        v2t_r1 = xclip[key]['v2t'][0]
        v2t_r5 = xclip[key]['v2t'][1]
        
        with open(ori_file, 'r') as f:
            ori = json.load(f)
        with open(mani_file, 'r') as f:
            mani = json.load(f)
            
        if key == 'temporal_contact':
            t2v_r1_ori = ori['temporal_contact_swap/']['txt_r1']
            t2v_r1_mani = mani['temporal_contact_swap_mani/']['txt_r1']
            t2v_r5_ori = ori['temporal_contact_swap/']['txt_r5']
            t2v_r5_mani = mani['temporal_contact_swap_mani/']['txt_r5']
            
            v2t_r1_ori = ori['temporal_contact_swap/']['img_r1']
            v2t_r1_mani = mani['temporal_contact_swap_mani/']['img_r1']
            v2t_r5_ori = ori['temporal_contact_swap/']['img_r5']
            v2t_r5_mani = mani['temporal_contact_swap_mani/']['img_r5']
        elif key == 'temporal_action':
            t2v_r1_ori = ori['temporal_action_swap/']['txt_r1']
            t2v_r1_mani = mani['temporal_action_swap_mani/']['txt_r1']
            t2v_r5_ori = ori['temporal_action_swap/']['txt_r5']
            t2v_r5_mani = mani['temporal_action_swap_mani/']['txt_r5']
            
            v2t_r1_ori = ori['temporal_action_swap/']['img_r1']
            v2t_r1_mani = mani['temporal_action_swap_mani/']['img_r1']
            v2t_r5_ori = ori['temporal_action_swap/']['img_r5']
            v2t_r5_mani = mani['temporal_action_swap_mani/']['img_r5']
        elif key == 'neighborhood_same_entity':
            t2v_r1_ori = ori['neighborhood_same_entity/']['txt_r1']
            t2v_r1_mani = mani['neighborhood_same_entity_mani/']['txt_r1']
            t2v_r5_ori = ori['neighborhood_same_entity/']['txt_r5']
            t2v_r5_mani = mani['neighborhood_same_entity_mani/']['txt_r5']
            
            v2t_r1_ori = ori['neighborhood_same_entity/']['img_r1']
            v2t_r1_mani = mani['neighborhood_same_entity_mani/']['img_r1']
            v2t_r5_ori = ori['neighborhood_same_entity/']['img_r5']
            v2t_r5_mani = mani['neighborhood_same_entity_mani/']['img_r5']
        elif key == 'neighborhood_diff_entity':
            t2v_r1_ori = ori['neighborhood_diff_entity/']['txt_r1']
            t2v_r1_mani = mani['neighborhood_diff_entity_mani/']['txt_r1']
            t2v_r5_ori = ori['neighborhood_diff_entity/']['txt_r5']
            t2v_r5_mani = mani['neighborhood_diff_entity_mani/']['txt_r5']
            
            v2t_r1_ori = ori['neighborhood_diff_entity/']['img_r1']
            v2t_r1_mani = mani['neighborhood_diff_entity_mani/']['img_r1']
            v2t_r5_ori = ori['neighborhood_diff_entity/']['img_r5']
            v2t_r5_mani = mani['neighborhood_diff_entity_mani/']['img_r5']
        elif key == 'counter_spatial':
            t2v_r1_ori = ori['counter_spatial/']['txt_r1']
            t2v_r1_mani = mani['counter_spatial_mani/']['txt_r1']
            t2v_r5_ori = ori['counter_spatial/']['txt_r5']
            t2v_r5_mani = mani['counter_spatial_mani/']['txt_r5']
            
            v2t_r1_ori = ori['counter_spatial/']['img_r1']
            v2t_r1_mani = mani['counter_spatial_mani/']['img_r1']
            v2t_r5_ori = ori['counter_spatial/']['img_r5']
            v2t_r5_mani = mani['counter_spatial_mani/']['img_r5']
        elif key == 'counter_contact':
            t2v_r1_ori = ori['counter_contact/']['txt_r1']
            t2v_r1_mani = mani['counter_contact_mani/']['txt_r1']
            t2v_r5_ori = ori['counter_contact/']['txt_r5']
            t2v_r5_mani = mani['counter_contact_mani/']['txt_r5']
            
            v2t_r1_ori = ori['counter_contact/']['img_r1']
            v2t_r1_mani = mani['counter_contact_mani/']['img_r1']
            v2t_r5_ori = ori['counter_contact/']['img_r5']
            v2t_r5_mani = mani['counter_contact_mani/']['img_r5']
        elif key == 'counter_action':
            t2v_r1_ori = ori['counter_action/']['txt_r1']
            t2v_r1_mani = mani['counter_action_mani/']['txt_r1']
            t2v_r5_ori = ori['counter_action/']['txt_r5']
            t2v_r5_mani = mani['counter_action_mani/']['txt_r5']
            
            v2t_r1_ori = ori['counter_action/']['img_r1']
            v2t_r1_mani = mani['counter_action_mani/']['img_r1']
            v2t_r5_ori = ori['counter_action/']['img_r5']
            v2t_r5_mani = mani['counter_action_mani/']['img_r5']
        elif key == 'counter_attribute':
            t2v_r1_ori = ori['counter_attribute/']['txt_r1']
            t2v_r1_mani = mani['counter_attribute_mani/']['txt_r1']
            t2v_r5_ori = ori['counter_attribute/']['txt_r5']
            t2v_r5_mani = mani['counter_attribute_mani/']['txt_r5']
            
            v2t_r1_ori = ori['counter_attribute/']['img_r1']
            v2t_r1_mani = mani['counter_attribute_mani/']['img_r1']
            v2t_r5_ori = ori['counter_attribute/']['img_r5']
            v2t_r5_mani = mani['counter_attribute_mani/']['img_r5']
        else:
            print('error')
            print(key)
            
        
        #check if the value is zero, if yes, print the i and key
        if t2v_r1_ori == 0 or t2v_r1_mani == 0 or t2v_r5_ori == 0 or t2v_r5_mani == 0 or v2t_r1_ori == 0 or v2t_r1_mani == 0 or v2t_r5_ori == 0 or v2t_r5_mani == 0:
            print('zero error')
            print(i)
            print(key)
        
        
        r1_delta_t2v =  (t2v_r1_ori - t2v_r1_mani)/t2v_r1_ori
        r5_delta_t2v =  (t2v_r5_ori - t2v_r5_mani)/t2v_r5_ori
        r1_delta_v2t =  (v2t_r1_ori - v2t_r1_mani)/v2t_r1_ori
        r5_delta_v2t =  (v2t_r5_ori - v2t_r5_mani)/v2t_r5_ori
        
        
        delta_p = {}
        delta_p['t2v'] = [r1_delta_t2v, r5_delta_t2v]
        delta_p['v2t'] = [r1_delta_v2t, r5_delta_v2t]
        
        diff_delta_p = {}
        diff_delta_p['t2v'] = [r1_delta_t2v - t2v_r1, r5_delta_t2v - t2v_r5]
        diff_delta_p['v2t'] = [r1_delta_v2t - v2t_r1, r5_delta_v2t - v2t_r5]
        
        #count for all positive diff value
        
        if r1_delta_t2v - t2v_r1 > 0:
            count_t2v_r1 += 1
        if r5_delta_t2v - t2v_r5 > 0:
            count_t2v_r5 += 1
        if r1_delta_v2t - v2t_r1 > 0:
            count_v2t_r1 += 1
        if r5_delta_v2t - v2t_r5 > 0:
            count_v2t_r5 += 1
        
        delta_p_all[key] = delta_p
        diff_delta_p_all[key] = diff_delta_p
    
        
    #print out the count
    print('bin: ', i)
    print('count_t2v_r1: ', count_t2v_r1)
    print('count_t2v_r5: ', count_t2v_r5)
    print('count_v2t_r1: ', count_v2t_r1)
    print('count_v2t_r5: ', count_v2t_r5)
    print('total: ', count_t2v_r1 + count_t2v_r5 + count_v2t_r1 + count_v2t_r5)
    #save the total count over 20 and print them all with i
    
    if count_t2v_r1 + count_t2v_r5 + count_v2t_r1 + count_v2t_r5 >19:
        count[i] = [count_t2v_r1, count_t2v_r5, count_v2t_r1, count_v2t_r5]
        
    #write to file
    delta_p_file = '/home/wiss/zhang/nfs/reb_deltap/results_sin/'
    
    if not os.path.exists(delta_p_file):
        os.makedirs(delta_p_file)
    #save to file
    
    delta_p_json_path = delta_p_file + experiment_ori+'_of_bin_'+str(i)+'_delta_p.json'
    with open(delta_p_json_path, 'w') as f:
        json.dump(delta_p_all, f)
    
    diff_delta_p_file = delta_p_file + experiment_ori+'_of_bin_'+str(i)+'_diff_delta_p.json'
    with open(diff_delta_p_file, 'w') as f:
        json.dump(diff_delta_p_all, f)
            
print(count)