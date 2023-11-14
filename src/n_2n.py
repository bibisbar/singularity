import json


with open('/home/wiss/zhang/Jinhe/singularity/Data/anetqa/anet_ret_counter_attribute.json','r') as f:
    data = json.load(f)
    #save the data into a list
    data_list = []
    for item in data:
        data_list.append(item)
    print(len(data_list))
    # open manipulation file
    with open('/home/wiss/zhang/Jinhe/singularity/Data/anetqa/anet_ret_counter_attribute_mani.json','r') as f_neg:
        data_neg = json.load(f_neg)
        #save the data into a list
        data_neg_list = []
        for item in data_neg:
            data_neg_list.append(item)
        print(len(data_neg_list))
        
        da_n_2n = []
        for i in range(0, len(data_list)):
            data_frame = {}
            data_frame['video'] = data_list[i]['video']
            data_frame['caption'] = data_list[i]['caption']
            for j in range(0, len(data_neg_list)):
                if data_list[i]['video'] == data_neg_list[j]['video']:
                    data_frame['positive1'] = data_neg_list[j]['caption']
                    break
            data_frame['positive2'] = 'none'
            data_frame['positive3'] = 'none'
            data_frame['positive4'] = 'none'
            data_frame['positive5'] = 'none'
            data_frame['positive6'] = 'none'
            data_frame['positive7'] = 'none'
            data_frame['negative1'] = 'none'
            data_frame['negative2'] = 'none'
            data_frame['negative3'] = 'none'
            data_frame['negative4'] = 'none'
            data_frame['negative5'] = 'none'
            data_frame['negative6'] = 'none'
            data_frame['negative7'] = 'none'
            da_n_2n.append(data_frame)
        print(len(da_n_2n))
        #save the data into json file
        with open('/home/wiss/zhang/Jinhe/singularity/Data/anetqa/anet_ret_counter_attribute_n_2n.json','w') as fp:
            json.dump(da_n_2n, fp)