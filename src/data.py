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

csv_path = '/home/wiss/zhang/Jinhe/video-attr-prober/Data/AnetQA/train_1.csv'
#csv_path = '/home/wiss/zhang/Jinhe/video-attr-prober/Data/AnetQA/temporal_contact_swap.csv'
#json_path = '/home/wiss/zhang/Jinhe/singularity/anno_downstream/anet_ret_val_1.json'
json_path = '/home/wiss/zhang/Jinhe/singularity/anno_downstream/anet_ret_train.json'

csv_json_convert(csv_path,json_path)
