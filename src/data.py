import json
import csv


#The annotation file is in json format, which can be loaded as a list of dictionaries. Each dictionary is {'image': path_to_video, 'caption': video_caption} for video-text dataset.

def remove_suffix(string,suffix):
    if string.endswith(suffix):
        return string[:-len(suffix)]
    return string

def csv_json_convert(csv_file_path,json_file_path):
    with open(csv_file_path, 'r') as file1:
        reader1 = csv.reader(file1,delimiter='\t')
        rows1 = list(reader1)

    data=[]
    number_of_data = 0

    for index,row in enumerate(rows1,start=0):
        
        
        
        row = row[0].split('.mp4')
        
        caption = row[1]
        
        assert len(row) == 2
        
        row_dict = {
            "video" : row[0]+'.mp4',
            "caption" : caption
        }
        data.append(row_dict)
        number_of_data += 1
    #save the data list as json file
    with open(json_file_path, 'w') as outfile:
        json.dump(data, outfile)


anet_file_name = ["temporal_contact_swap", "temporal_action_swap", "neighborhood_same_entity", "neighborhood_diff_entity", "counter_spatial", "counter_contact", "counter_action", "counter_attribute"]
moviegraph_file_name = ["temporal_int", "temporal_act", "neighborhood_same_entity", "neighborhood_diff_entity", "counter_rel", "counter_act", "counter_int", "counter_attr"]


csv_path = '/home/wiss/zhang/Jinhe/video-attr-prober/Data/MovieGraph/train2.csv'
#csv_path = '/home/wiss/zhang/Jinhe/video-attr-prober/Data/AnetQA/temporal_contact_swap.csv'
#json_path = '/home/wiss/zhang/Jinhe/singularity/anno_downstream/anet_ret_val_1.json'
json_path = '/home/wiss/zhang/Jinhe/singularity/Data/moviegraph/anet_ret_train_2.json'

# for i in range(len(moviegraph_file_name)):
#     csv_test_name = moviegraph_file_name[i]+'.csv'
#     json_test_name = 'anet_ret_'+moviegraph_file_name[i]+'.json'
#     mani_csv_test_path = moviegraph_file_name[i]+'_mani'+'.csv'
#     mani_json_test_path = 'anet_ret_'+moviegraph_file_name[i]+'_mani'+'.json'
#     csv_path = '/home/wiss/zhang/Jinhe/video-attr-prober/Data/MovieGraph/test_data_full/'+csv_test_name
#     json_path = '/home/wiss/zhang/Jinhe/singularity/Data/moviegraph/'+json_test_name
#     csv_json_convert(csv_path,json_path)
#     csv_path = '/home/wiss/zhang/Jinhe/video-attr-prober/Data/MovieGraph/test_data_full/'+mani_csv_test_path
#     json_path = '/home/wiss/zhang/Jinhe/singularity/Data/moviegraph/'+mani_json_test_path
#     csv_json_convert(csv_path,json_path)



csv_json_convert(csv_path,json_path)
