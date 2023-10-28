import json
import lmdb
import csv
import random
# create lmdb
env = lmdb.open('/home/wiss/zhang/Yanyu/AnetQA/manipulation/data_train1.lmdb', map_size= 1024*1024*1024)

# write data
data = []
with env.begin(write=True) as txn:
    with open("/home/wiss/zhang/Jinhe/singularity/Data/anetqa/anet_ret_train_1.json","r") as load_f:
        load_dict = json.load(load_f)
        print(len(load_dict))
        for i in range(0, len(load_dict)):
            
            cur_annotation = {}
            #first append the original data into cur_annotation
            cur_annotation['video'] = load_dict[i]['video']
            cur_annotation['caption'] = load_dict[i]['caption']
            
            clip_id = load_dict[i]['video']
            #remove the suffix
            clip_id = clip_id.split('.')[0]
            #search metadata in lmdb
            anno = txn.get(clip_id.strip().encode()).decode()
            #save all negtion samples 
            neg_samples = []
            for mani, pos_neg in json.loads(anno).items():
                if mani == 'neighborhood_same_entity' or mani == 'neighborhood_diff_entity' or mani == 'temporal_contact_swap' or mani == 'temporal_action_swap':

                    for neg in pos_neg['negative']:
                        if neg != '':
                            neg_samples.append(neg)
                elif mani == 'counter_spatial' or mani == 'counter_contact' or mani == 'counter_action' or mani == 'counter_attribute':
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
                cur_annotation['negative1'] = choose_neg_samples[0]
                cur_annotation['negative2'] = choose_neg_samples[1]
                cur_annotation['negative3'] = choose_neg_samples[2]
                cur_annotation['negative4'] = choose_neg_samples[3]
                cur_annotation['negative5'] = choose_neg_samples[4]
                cur_annotation['negative6'] = choose_neg_samples[5]
                cur_annotation['negative7'] = choose_neg_samples[6]
                data.append(cur_annotation)
            elif len(neg_samples) > 0 and len(neg_samples) <= 6:
                choose_neg_samples = random.sample(neg_samples, 1)
                for i in range(0, len(neg_samples)):
                    cur_annotation['negative'+str(i+1)] = choose_neg_samples[0]
                data.append(cur_annotation)
            elif len(neg_samples) <= 0:
                #do nothing
                continue
    with open("/home/wiss/zhang/Jinhe/singularity/Data/anetqa/anet_ret_train_1_neg.json", "w") as f:
        print(len(data))
        json.dump(data, f)
            