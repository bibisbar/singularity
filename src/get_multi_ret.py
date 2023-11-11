import json
import lmdb
import csv
import random
# create lmdb
env = lmdb.open('/home/wiss/zhang/Yanyu/AnetQA/manipulation/data_test.lmdb', map_size= 1024*1024*1024)

# write data
data = []
with env.begin(write=True) as txn:
    # open txt file to read all video ids
    with open("/home/wiss/zhang/Jinhe/singularity/src/id_for_test1185.txt","r") as load_f:
        load_dict = load_f.readlines()
        print(len(load_dict))
        for i in range(0, len(load_dict)):
            
            cur_annotation = {}
            cur_annotation['video'] = load_dict[i].strip('\n')+ '.mp4'
            video_id = load_dict[i].strip('\n')
            
            # get the annotation from the lmdb
            anno = txn.get(video_id.encode())
            pos_samples = []
            neg_samples = []
            num_pos = 0
            for mani, pos_neg in json.loads(anno).items():
                if mani == 'neighborhood_same_entity' or mani == 'neighborhood_diff_entity' or mani == 'temporal_contact_swap' or mani == 'temporal_action_swap':
                    for pos in pos_neg['positive']:
                        if pos != '':
                            pos_samples.append(pos)
                    for neg in pos_neg['negative']:
                        if neg != '':
                            neg_samples.append(neg)
                elif mani == 'counter_spatial' or mani == 'counter_contact' or mani == 'counter_action' or mani == 'counter_attribute':
                    for pos in pos_neg['positive']:
                        if pos != '':
                            pos_samples.append(pos)
                    for neg in pos_neg['negative1']:
                        if neg != '':
                            neg_samples.append(neg)
                    for neg in pos_neg['negative2']:
                        if neg != '':
                            neg_samples.append(neg)
                    for neg in pos_neg['negative3']:
                        if neg != '':
                            neg_samples.append(neg)
                    for neg in pos_neg['negative4']:
                        if neg != '':
                            neg_samples.append(neg)
                    for neg in pos_neg['negative5']:
                        if neg != '':
                            neg_samples.append(neg)
                    for neg in pos_neg['negative6']:
                        if neg != '':
                            neg_samples.append(neg)
                    for neg in pos_neg['negative7']:
                        if neg != '':
                            neg_samples.append(neg)
                    for neg in pos_neg['negative8']:
                        if neg != '':
                            neg_samples.append(neg)
                    for neg in pos_neg['negative9']:
                        if neg != '':
                            neg_samples.append(neg)
                    for neg in pos_neg['negative10']:
                        if neg != '':
                            neg_samples.append(neg)
            if len(neg_samples) > 6:
                choose_neg_samples = random.sample(neg_samples, 7)
                cur_annotation['negative1'] = 'none'
                cur_annotation['negative2'] = 'none'
                cur_annotation['negative3'] = 'none'
                cur_annotation['negative4'] = 'none'
                cur_annotation['negative5'] = 'none'
                cur_annotation['negative6'] = 'none'
                cur_annotation['negative7'] = 'none'
                
            elif len(neg_samples) > 0 and len(neg_samples) <= 6:
                choose_neg_samples = random.sample(neg_samples, 1)
                for i in range(0, len(neg_samples)):
                    cur_annotation['negative'+str(i+1)] = 'none'
                
            elif len(neg_samples) <= 0:
                #do nothing
                continue
            
            if len(pos_samples) > 3:
                choose_pos_samples = random.sample(pos_samples, 4)
                cur_annotation['caption'] = choose_pos_samples[3]
                cur_annotation['positive1'] = choose_pos_samples[0]
                cur_annotation['positive2'] = choose_pos_samples[1]
                cur_annotation['positive3'] = choose_pos_samples[2]
                cur_annotation['positive4'] = 'none'
                cur_annotation['positive5'] = 'none'
                cur_annotation['positive6'] = 'none'
                cur_annotation['positive7'] = 'none'
            elif len(pos_samples) > 0 and len(pos_samples) <= 3:
                num_pos = num_pos + 1
                cur_annotation = {}
            
            #only append the data when cur_annotation is not empty
            if cur_annotation != {}:
                data.append(cur_annotation)
print(len(data))

#random sample 600 data and write to json file
data = random.sample(data, 600)
# write data to json file
print(len(data))
with open('/home/wiss/zhang/Jinhe/singularity/Data/anetqa/anet_multi_ret_600.json', 'w') as f:
    json.dump(data, f)