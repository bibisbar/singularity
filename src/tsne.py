
import sklearn
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch
import matplotlib.pyplot as plt
import sklearn
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch
import matplotlib.pyplot as plt


#load tensor from file and return numpy array
def load_tensor_from_file(file_path):
    tensor = torch.load(file_path,map_location=torch.device('cpu'))
    #remove 1 dimension
    tensor = tensor.squeeze()
    #tensor = tensor / tensor.norm(dim=-1, keepdim=True)
    #convert to numpy
    tensor = tensor.numpy()
    return tensor

def load_tensor_from_file_visual(file_path):
    tensor = torch.load(file_path,map_location=torch.device('cpu'))
    tensor = tensor.squeeze()
    #tensor = tensor / tensor.norm(dim=-1, keepdim=True)
    return tensor

def load_tensor_from_file_mask(file_path):
    tensor = torch.load(file_path,map_location=torch.device('cpu'))
    tensor = tensor.squeeze()
    return tensor

def _mean_pooling_for_similarity_visual(visual_output, video_mask,):
    video_mask = video_mask.view(-1, video_mask.shape[-1])
    video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
    visual_output = visual_output * video_mask_un
    video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)
    video_mask_un_sum[video_mask_un_sum == 0.] = 1.
    video_out = torch.sum(visual_output, dim=1) / video_mask_un_sum
    #video_out = video_out / video_out.norm(dim=-1, keepdim=True)
    #convert to numpy float
    video_out = video_out.numpy().astype('float')
    return video_out

# apply PCA to reduce dimensionality
def apply_pca(tensor, n_components):
    pca = PCA(n_components=n_components)
    tensor = pca.fit_transform(tensor)
    return tensor

#apply TSNE to reduce dimensionality
def apply_tsne(tensor, n_components):
    tsne = TSNE(n_components=n_components)
    tensor = tsne.fit_transform(tensor)
    return tensor

#visualize the result 
def visualize(tensor_ori,tensor_mani,label_ori,label_mani,color_ori,color_mani,save_path):
    plt.scatter(tensor_ori[:,0], tensor_ori[:,1],color=color_ori, label=label_ori)
    plt.scatter(tensor_mani[:,0], tensor_mani[:,1],color=color_mani, label=label_mani,alpha=0.5)
    plt.show()
    #save the figure
    plt.savefig(save_path)
    plt.legend()
    plt.close()


def visualize_3d(tensor_ori,tensor_mani,label_ori,label_mani,color_ori,color_mani,save_path):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    pos_text = ax.scatter(tensor_ori[:,0], tensor_ori[:,1],tensor_ori[:,2],color=color_ori, label=label_ori)
    neg_text = ax.scatter(tensor_mani[:,0], tensor_mani[:,1],tensor_mani[:,2],color=color_mani, label=label_mani,alpha=0.5)
    plt.show()
    #save the figure
    plt.savefig(save_path)
    plt.legend((pos_text, neg_text), ('pos_text', 'neg_text'))
    plt.close()


def visualize_2d_three_embedding(tensor_ori, tensor_mani, visual_tensor, label_ori, label_mani, label_visual, color_ori, color_mani, color_visual, save_path):
    postive_text= plt.scatter(tensor_ori[:,0], tensor_ori[:,1],color=color_ori, label=label_ori)
    negative_text= plt.scatter(tensor_mani[:,0], tensor_mani[:,1],color=color_mani, label=label_mani,alpha=0.5)
    visual= plt.scatter(visual_tensor[:,0], visual_tensor[:,1],color=color_visual, label=label_visual,alpha=0.5)
    plt.show()
    #save the figure
    plt.legend(handles=[postive_text, negative_text, visual], labels=['positive', 'negative', 'visual'])
    plt.savefig(save_path)
    plt.close()




# -------------------visualize manipulated visual/text 2d/3d embedding-------------------#
# loop to output plt
# ckpt_moviegraph_retrieval_looseType_new_loss_freeze6_eval_7neg_0.1
# ckpt_moviegraph_retrieval_looseType_new_loss_freeze0_eval_7neg_2
# new_loss_freeze0_eval_7neg_2
#for i in range(0,10):
    #visual_embedding_ori ='/home/wiss/zhang/Jinhe/video-attr-prober/models_retrieval/CLIP4Clip/pca_tsne/ckpt_moviegraph_retrieval_looseType_baseline_best_pt_eval/baseline_best_pt_evalbatch_visual_output_list.100'
    #visual_embedding_mani = '/home/wiss/zhang/Jinhe/video-attr-prober/models_retrieval/CLIP4Clip/pca_tsne/ckpt_moviegraph_retrieval_looseType_new_loss_freeze0_eval_7neg_2/mani_new_loss_freeze0_eval_7neg_2_'+str(i)+'batch_visual_output_list.100'
    #mask_embedding_ori = '/home/wiss/zhang/Jinhe/video-attr-prober/models_retrieval/CLIP4Clip/pca_tsne/ckpt_moviegraph_retrieval_looseType_baseline_best_pt_eval/baseline_best_pt_evalbatch_list_v.100'
    #mask_embedding_mani = '/home/wiss/zhang/Jinhe/video-attr-prober/models_retrieval/CLIP4Clip/pca_tsne/ckpt_moviegraph_retrieval_looseType_new_loss_freeze0_eval_7neg_2/mani_new_loss_freeze0_eval_7neg_2_'+str(i)+'batch_list_v.100'

    #embedding_ori ='/home/wiss/zhang/Jinhe/video-attr-prober/models_retrieval/CLIP4Clip/pca_tsne/ckpt_moviegraph_retrieval_looseType_new_loss_freeze0_eval_7neg_2/new_loss_freeze0_eval_7neg_2_'+str(i)+'batch_sequence_output_list.100'
    #embedding_mani = '/home/wiss/zhang/Jinhe/video-attr-prober/models_retrieval/CLIP4Clip/pca_tsne/ckpt_moviegraph_retrieval_looseType_new_loss_freeze0_eval_7neg_2/mani_new_loss_freeze0_eval_7neg_2_'+str(i)+'batch_sequence_output_list.100'

    
    #tensor_ori = load_tensor_from_file(embedding_ori)
    #tensor_mani = load_tensor_from_file(embedding_mani)

    #tensor_pca_ori_2d = apply_pca(tensor_ori, 2)
    #tensor_pca_mani_2d = apply_pca(tensor_mani, 2)

    #tensor_tsne_ori_2d = apply_tsne(tensor_ori, 2)
    #tensor_tsne_mani_2d = apply_tsne(tensor_mani, 2)


    # save_path_tsne_2d = '/home/wiss/zhang/Jinhe/video-attr-prober/models_retrieval/CLIP4Clip/pca_tsne/ckpt_moviegraph_retrieval_looseType_new_loss_freeze0_eval_7neg_2/tsne_2d_'+'epoch'+str(i)+'.png'
    # save_path_pca_2d = '/home/wiss/zhang/Jinhe/video-attr-prober/models_retrieval/CLIP4Clip/pca_tsne/ckpt_moviegraph_retrieval_looseType_new_loss_freeze0_eval_7neg_2/pca_2d_'+'epoch'+str(i)+'.png'
    # visualize(tensor_pca_ori_2d, tensor_pca_mani_2d, 'original', 'manipulated', 'blue', 'red', save_path_pca_2d)
    # print('viaualize tensor_pca_ori_2d'+str(i)+ 'saved!')
    # visualize(tensor_tsne_ori_2d, tensor_tsne_mani_2d, 'original', 'manipulated', 'blue', 'red', save_path_tsne_2d)
    # print('viaualize tensor_tsne_ori_2d'+str(i)+ 'saved!
    # tensor_pca_ori_3d = apply_pca(tensor_ori, 3)
    # tensor_pca_mani_3d = apply_pca(tensor_mani, 3)

    # tensor_tsne_ori_3d = apply_tsne(tensor_ori, 3)
    # tensor_tsne_mani_3d = apply_tsne(tensor_mani, 3)

    # save_path_tsne_3d = '/home/wiss/zhang/Jinhe/video-attr-prober/models_retrieval/CLIP4Clip/pca_tsne/ckpt_moviegraph_retrieval_looseType_new_loss_freeze0_eval_7neg_2/tsne_3d_'+'epoch'+str(i)+'.png'
    # save_path_pca_3d = '/home/wiss/zhang/Jinhe/video-attr-prober/models_retrieval/CLIP4Clip/pca_tsne/ckpt_moviegraph_retrieval_looseType_new_loss_freeze0_eval_7neg_2/pca_3d_'+'epoch'+str(i)+'.png'
    # visualize_3d(tensor_pca_ori_3d, tensor_pca_mani_3d, 'original', 'manipulated', 'blue', 'red', save_path_pca_3d)
    # print('viaualize tensor_pca_ori_3d'+str(i)+ 'saved!')
    # visualize_3d(tensor_tsne_ori_3d, tensor_tsne_mani_3d, 'original', 'manipulated', 'blue', 'red', save_path_tsne_3d)
    # print('viaualize tensor_tsne_ori_3d'+str(i)+ 'saved!')

    #visual_tensor_ori = load_tensor_from_file_visual(visual_embedding_ori)
    #visual_tensor_mani = load_tensor_from_file_visual(visual_embedding_mani)
    #mask_tensor_ori = load_tensor_from_file_mask(mask_embedding_ori)
    #mask_tensor_mani = load_tensor_from_file_mask(mask_embedding_mani)

    #visual_tensor_ori = _mean_pooling_for_similarity_visual(visual_tensor_ori, mask_tensor_ori)
    #visual_tensor_mani = _mean_pooling_for_similarity_visual(visual_tensor_mani, mask_tensor_mani)

    #visual_tensor_pca_ori_2d = apply_pca(visual_tensor_ori, 2)
    #visual_tensor_pca_mani_2d = apply_pca(visual_tensor_mani, 2)

    #visual_tensor_tsne_ori_2d = apply_tsne(visual_tensor_ori, 2)
    #visual_tensor_tsne_mani_2d = apply_tsne(visual_tensor_mani, 2)

    #three_embedding_save_path_tsne_2d = '/home/wiss/zhang/Jinhe/video-attr-prober/models_retrieval/CLIP4Clip/pca_tsne/ckpt_moviegraph_retrieval_looseType_new_loss_freeze0_eval_7neg_2/tsne_2d_'+'epoch'+str(i)+'.png'
    #three_embedding_save_path_pca_2d = '/home/wiss/zhang/Jinhe/video-attr-prober/models_retrieval/CLIP4Clip/pca_tsne/ckpt_moviegraph_retrieval_looseType_new_loss_freeze0_eval_7neg_2/pca_2d_'+'epoch'+str(i)+'.png'
    #visualize_2d_three_embedding(tensor_tsne_ori_2d, tensor_tsne_mani_2d, visual_tensor_tsne_mani_2d, 'positive', 'negative', 'visual', 'blue', 'red', 'green', three_embedding_save_path_tsne_2d)
    #print('viaualize visual_tensor_tsne_ori_2d'+str(i)+ 'saved!')
    #visualize_2d_three_embedding(tensor_pca_ori_2d, tensor_pca_mani_2d, visual_tensor_pca_mani_2d, 'positive', 'negative', 'visual', 'blue', 'red', 'green', three_embedding_save_path_pca_2d)
    #print('viaualize visual_tensor_pca_ori_2d'+str(i)+ 'saved!')
    # visual_save_path_tsne_2d = '/home/wiss/zhang/Jinhe/video-attr-prober/models_retrieval/CLIP4Clip/pca_tsne/ckpt_moviegraph_retrieval_looseType_new_loss_freeze0_eval_7neg_2/visual_tsne_2d_'+'epoch'+str(i)+'.png'
    # visual_save_path_pca_2d = '/home/wiss/zhang/Jinhe/video-attr-prober/models_retrieval/CLIP4Clip/pca_tsne/ckpt_moviegraph_retrieval_looseType_new_loss_freeze0_eval_7neg_2/visual_pca_2d_'+'epoch'+str(i)+'.png'
    #visualize(visual_tensor_pca_ori_2d, visual_tensor_pca_mani_2d, 'original', 'improved', 'blue', 'red', visual_save_path_pca_2d)
    #print('viaualize visual_tensor_pca_ori_2d'+str(i)+ 'saved!')
    #visualize(visual_tensor_tsne_ori_2d, visual_tensor_tsne_mani_2d, 'original', 'improved', 'blue', 'red', visual_save_path_tsne_2d)
    #print('viaualize visual_tensor_tsne_ori_2d'+str(i)+ 'saved!')
    # visual_tensor_pca_ori_3d = apply_pca(visual_tensor_ori, 3)
    # visual_tensor_pca_mani_3d = apply_pca(visual_tensor_mani, 3)

    # visual_tensor_tsne_ori_3d = apply_tsne(visual_tensor_ori, 3)
    # visual_tensor_tsne_mani_3d = apply_tsne(visual_tensor_mani, 3)

    # visual_save_path_tsne_3d = '/home/wiss/zhang/Jinhe/video-attr-prober/models_retrieval/CLIP4Clip/pca_tsne/ckpt_moviegraph_retrieval_looseType_new_loss_freeze0_eval_7neg_2/visual_tsne_3d_'+'epoch'+str(i)+'.png'
    # visual_save_path_pca_3d = '/home/wiss/zhang/Jinhe/video-attr-prober/models_retrieval/CLIP4Clip/pca_tsne/ckpt_moviegraph_retrieval_looseType_new_loss_freeze0_eval_7neg_2/visual_pca_3d_'+'epoch'+str(i)+'.png'

    # visualize_3d(visual_tensor_pca_ori_3d, visual_tensor_pca_mani_3d, 'original', 'improved', 'blue', 'red', visual_save_path_pca_3d)
    # print('viaualize visual_tensor_pca_ori_3d'+str(i)+ 'saved!')
    # visualize_3d(visual_tensor_tsne_ori_3d, visual_tensor_tsne_mani_3d, 'original', 'improved', 'blue', 'red', visual_save_path_tsne_3d)
    # print('viaualize visual_tensor_tsne_ori_3d'+str(i)+ 'saved!')

    



# -------------------visualize baseline visual/text 2d/3d embedding-------------------#  
# visual_embedding_ori ='/home/wiss/zhang/Jinhe/video-attr-prober/models_retrieval/CLIP4Clip/pca_tsne/ckpt_moviegraph_retrieval_looseType_baseline_best_pt_eval/baseline_best_pt_evalbatch_visual_output_list.100'
# visual_embedding_mani = '/home/wiss/zhang/Jinhe/video-attr-prober/models_retrieval/CLIP4Clip/pca_tsne/ckpt_moviegraph_retrieval_looseType_baseline_best_pt_eval/mani_baseline_best_pt_evalbatch_visual_output_list.100'
# mask_embedding_ori = '/home/wiss/zhang/Jinhe/video-attr-prober/models_retrieval/CLIP4Clip/pca_tsne/ckpt_moviegraph_retrieval_looseType_baseline_best_pt_eval/baseline_best_pt_evalbatch_list_v.100'
# mask_embedding_mani = '/home/wiss/zhang/Jinhe/video-attr-prober/models_retrieval/CLIP4Clip/pca_tsne/ckpt_moviegraph_retrieval_looseType_baseline_best_pt_eval/mani_baseline_best_pt_evalbatch_list_v.100'

embedding_ori ='/nfs/data2/Yanyu/pca/text_feat_counter_rel_ori.pt' #text feature
embedding_mani = '/nfs/data2/Yanyu/pca_neg/text_feat_counter_rel_mani.pt' #negative text feature
#600*512

### load image feature
image_feat_ori ='/nfs/data2/Yanyu/pca/image_feat_counter_rel_ori.pt'   #image feature
#size (600*4*512)
#load tensor from file 



image_feat_tensor = load_tensor_from_file_visual(image_feat_ori)

postive_similarity_matrix = torch.einsum("mld,nd->mln", image_feat_ori, embedding_ori).mean(1)   # (600*600
negative_similarity_matrix = torch.einsum("mld,nd->mln", image_feat_ori, embedding_mani).mean(1)


#mean pooling at 4 dimension
image_feat_tensor = image_feat_tensor.mean(1)
#remove 1 dimension
image_feat_tensor = image_feat_tensor.squeeze()
print('image_feat_tensor.shape', image_feat_tensor.shape)
#600*512



#apply PCA to reduce dimensionality
#




   
tensor_ori = load_tensor_from_file(embedding_ori)
tensor_mani = load_tensor_from_file(embedding_mani)

tensor_pca_ori_2d = apply_pca(tensor_ori, 2)
tensor_pca_mani_2d = apply_pca(tensor_mani, 2)

tensor_tsne_ori_2d = apply_tsne(tensor_ori, 2)
tensor_tsne_mani_2d = apply_tsne(tensor_mani, 2)


save_path_tsne_2d = '/home/stud/jinhe/Yanyu/sklearn_tsne/scatterplot_counter_rel.png'
save_path_pca_2d = '/home/stud/jinhe/Yanyu/sklearn_pca/scatterplot_counter_rel.png'
visualize(tensor_pca_ori_2d, tensor_pca_mani_2d, 'positive', 'negative', (63/255,113/255,128/255),(253/255,150/255,64/255), save_path_pca_2d)
print('viaualize tensor_pca_ori_2d saved!')
visualize(tensor_tsne_ori_2d, tensor_tsne_mani_2d, 'positive', 'negative', (63/255,113/255,128/255),(253/255,150/255,64/255), save_path_tsne_2d)
print('viaualize tensor_tsne_ori_2d saved!')
# tensor_pca_ori_3d = apply_pca(tensor_ori, 3)
# tensor_pca_mani_3d = apply_pca(tensor_mani, 3)

# tensor_tsne_ori_3d = apply_tsne(tensor_ori, 3)
# tensor_tsne_mani_3d = apply_tsne(tensor_mani, 3)    

# save_path_tsne_3d = '/home/wiss/zhang/Jinhe/video-attr-prober/models_retrieval/CLIP4Clip/pca_tsne/ckpt_moviegraph_retrieval_looseType_baseline_best_pt_eval/tsne_3d.png'
# save_path_pca_3d = '/home/wiss/zhang/Jinhe/video-attr-prober/models_retrieval/CLIP4Clip/pca_tsne/ckpt_moviegraph_retrieval_looseType_baseline_best_pt_eval/pca_3d.png'
# visualize_3d(tensor_pca_ori_3d, tensor_pca_mani_3d, 'original', 'manipulated', 'blue', 'red', save_path_pca_3d)
# print('viaualize tensor_pca_ori_3d saved!')
# visualize_3d(tensor_tsne_ori_3d, tensor_tsne_mani_3d, 'original', 'manipulated', 'blue', 'red', save_path_tsne_3d)
# print('viaualize tensor_tsne_ori_3d saved!')

# visual_tensor_ori = load_tensor_from_file_visual(visual_embedding_ori)
# visual_tensor_mani = load_tensor_from_file_visual(visual_embedding_mani)
# mask_tensor_ori = load_tensor_from_file_mask(mask_embedding_ori)
# mask_tensor_mani = load_tensor_from_file_mask(mask_embedding_mani)

# visual_tensor_ori = _mean_pooling_for_similarity_visual(visual_tensor_ori, mask_tensor_ori)
# visual_tensor_mani = _mean_pooling_for_similarity_visual(visual_tensor_mani, mask_tensor_mani)

# visual_tensor_pca_ori_2d = apply_pca(visual_tensor_ori, 2)
# visual_tensor_pca_mani_2d = apply_pca(visual_tensor_mani, 2)

# visual_tensor_tsne_ori_2d = apply_tsne(visual_tensor_ori, 2)
# visual_tensor_tsne_mani_2d = apply_tsne(visual_tensor_mani, 2)

# visual_save_path_tsne_2d = '/home/wiss/zhang/Jinhe/video-attr-prober/models_retrieval/CLIP4Clip/pca_tsne/ckpt_moviegraph_retrieval_looseType_baseline_best_pt_eval/visual_tsne_2d.png'
# visual_save_path_pca_2d = '/home/wiss/zhang/Jinhe/video-attr-prober/models_retrieval/CLIP4Clip/pca_tsne/ckpt_moviegraph_retrieval_looseType_baseline_best_pt_eval/visual_pca_2d.png'
# visualize(visual_tensor_pca_ori_2d, visual_tensor_pca_mani_2d, 'original', 'manipulated', 'blue', 'red', visual_save_path_pca_2d)
# print('viaualize visual_tensor_pca_ori_2d saved!')
# visualize(visual_tensor_tsne_ori_2d, visual_tensor_tsne_mani_2d, 'original', 'manipulated', 'blue', 'red', visual_save_path_tsne_2d)
# print('viaualize visual_tensor_tsne_ori_2d saved!')
# visual_tensor_pca_ori_3d = apply_pca(visual_tensor_ori, 3)
# visual_tensor_pca_mani_3d = apply_pca(visual_tensor_mani, 3)

# visual_tensor_tsne_ori_3d = apply_tsne(visual_tensor_ori, 3)
# visual_tensor_tsne_mani_3d = apply_tsne(visual_tensor_mani, 3)

# visual_save_path_tsne_3d = '/home/wiss/zhang/Jinhe/video-attr-prober/models_retrieval/CLIP4Clip/pca_tsne/ckpt_moviegraph_retrieval_looseType_baseline_best_pt_eval/visual_tsne_3d.png'
# visual_save_path_pca_3d = '/home/wiss/zhang/Jinhe/video-attr-prober/models_retrieval/CLIP4Clip/pca_tsne/ckpt_moviegraph_retrieval_looseType_baseline_best_pt_eval/visual_pca_3d.png'
# visualize_3d(visual_tensor_pca_ori_3d, visual_tensor_pca_mani_3d, 'original', 'manipulated', 'blue', 'red', visual_save_path_pca_3d)
# print('viaualize visual_tensor_pca_ori_3d saved!')
# visualize_3d(visual_tensor_tsne_ori_3d, visual_tensor_tsne_mani_3d, 'original', 'manipulated', 'blue', 'red', visual_save_path_tsne_3d)
# print('viaualize visual_tensor_tsne_ori_3d saved!')


# #visualize three embeddings
# three_embedding_save_path_tsne_2d = '/home/wiss/zhang/Jinhe/video-attr-prober/models_retrieval/CLIP4Clip/pca_tsne/three_embedding_tsne_2d.png'
# three_embedding_save_path_pca_2d = '/home/wiss/zhang/Jinhe/video-attr-prober/models_retrieval/CLIP4Clip/pca_tsne/three_embedding_pca_2d.png'

# visualize_2d_three_embedding(tensor_tsne_ori_2d, tensor_tsne_mani_2d, visual_tensor_tsne_ori_2d, 'original', 'manipulated', 'visual', 'blue', 'red', 'green', three_embedding_save_path_tsne_2d)
# visualize_2d_three_embedding(tensor_pca_ori_2d, tensor_pca_mani_2d, visual_tensor_pca_ori_2d, 'original', 'manipulated', 'visual', 'blue', 'red', 'green', three_embedding_save_path_pca_2d)
message.txt
17 KB