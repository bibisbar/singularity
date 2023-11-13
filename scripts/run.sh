
sbatch scripts/train_ret_anet.sh anet_train_1_Seed42 anet 2 local 12345 pretrained_path=/home/wiss/zhang/Jinhe/singularity/anet_ret/ft_anet_ret_singularity_temporal_17m.pth \ video_input.num_frames=4 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2 \ batch_size.video=32 \ scheduler.epoch=30 \ train_type=anet_ret_train_1.json \ seed=42

sbatch scripts/train_ret_anet.sh anet_train_2_Seed2 anet 2 local 12346 pretrained_path=/home/wiss/zhang/Jinhe/singularity/anet_ret/ft_anet_ret_singularity_temporal_17m.pth \ video_input.num_frames=4 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2 \ batch_size.video=32 \ scheduler.epoch=30 \ train_type=anet_ret_train_2.json \ seed=2

sbatch scripts/train_ret_anet.sh anet_train_3_Seed3 anet 2 local 12347 pretrained_path=/home/wiss/zhang/Jinhe/singularity/anet_ret/ft_anet_ret_singularity_temporal_17m.pth \ video_input.num_frames=4 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2 \ batch_size.video=32 \ scheduler.epoch=30 \ train_type=anet_ret_train_3.json \ seed=3


sbatch scripts/train_ret_mg.sh moviegraph_train_1_Seed42 moviegraph 2 local 12348 pretrained_path=/home/wiss/zhang/Jinhe/singularity/pt/singularity_temporal_17m.pth \ video_input.num_frames=4 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2 \ batch_size.video=32 \ scheduler.epoch=30 \ train_type=anet_ret_train_1.json \ seed=42

sbatch scripts/train_ret_mg.sh moviegraph_train_2_Seed2 moviegraph 2 local 12349 pretrained_path=/home/wiss/zhang/Jinhe/singularity/pt/singularity_temporal_17m.pth \ video_input.num_frames=4 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2 \ batch_size.video=32 \ scheduler.epoch=30 \ train_type=anet_ret_train_2.json \ seed=2


sbatch scripts/train_ret_mg.sh moviegraph_train_1_Seed3 moviegraph 2 local 12350 pretrained_path=/home/wiss/zhang/Jinhe/singularity/pt/singularity_temporal_17m.pth \ video_input.num_frames=4 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2 \ batch_size.video=32 \ scheduler.epoch=30 \ train_type=anet_ret_train_1.json \ seed=3





#eval
sbatch scripts/eval_ret.sh anet /home/wiss/zhang/Jinhe/singularity/paper_results/ret_anet/anet_anet_train_1_Seed42/ckpt_best.pth anet_eval_1_Seed42 local 1 12351 \ test_types=[temporal_contact_swap,temporal_action_swap,neighborhood_same_entity,neighborhood_diff_entity,counter_spatial,counter_contact,counter_action,counter_intibute] video_input.num_frames_test=12 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2



#ori
sbatch scripts/eval_ret.sh anet /home/wiss/zhang/Jinhe/singularity/paper_results/ret_anet/anet_anet_train_1_Seed42/ckpt_best.pth anet_eval_1_Seed42 local 1 12351 \ test_types=[temporal_contact_swap,temporal_action_swap,neighborhood_same_entity,neighborhood_diff_entity,counter_spatial,counter_contact,counter_action,counter_intibute] video_input.num_frames_test=12 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2

sbatch scripts/eval_ret.sh anet /home/wiss/zhang/Jinhe/singularity/paper_results/ret_anet/anet_anet_train_2_Seed2/ckpt_best.pth anet_eval_2_Seed2 local 1 12352 \ test_types=[temporal_contact_swap,temporal_action_swap,neighborhood_same_entity,neighborhood_diff_entity,counter_spatial,counter_contact,counter_action,counter_intibute] video_input.num_frames_test=12 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2

sbatch scripts/eval_ret.sh anet /home/wiss/zhang/Jinhe/singularity/paper_results/ret_anet/anet_anet_train_3_Seed3/ckpt_best.pth anet_eval_3_Seed3 local 1 12353 \ test_types=[temporal_contact_swap,temporal_action_swap,neighborhood_same_entity,neighborhood_diff_entity,counter_spatial,counter_contact,counter_action,counter_intibute] video_input.num_frames_test=12 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2

#mani 

sbatch scripts/eval_ret.sh anet /home/wiss/zhang/Jinhe/singularity/paper_results/ret_anet/anet_anet_train_1_Seed42/ckpt_best.pth anet_eval_1_Seed42_mani local 1 12354 \ test_types=[temporal_contact_swap_mani,temporal_action_swap_mani,neighborhood_same_entity_mani,neighborhood_diff_entity_mani,counter_spatial_mani,counter_contact_mani,counter_action_mani,counter_intibute_mani] video_input.num_frames_test=12 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2

sbatch scripts/eval_ret.sh anet /home/wiss/zhang/Jinhe/singularity/paper_results/ret_anet/anet_anet_train_2_Seed2/ckpt_best.pth anet_eval_2_Seed2_mani local 1 12355 \ test_types=[temporal_contact_swap_mani,temporal_action_swap_mani,neighborhood_same_entity_mani,neighborhood_diff_entity_mani,counter_spatial_mani,counter_contact_mani,counter_action_mani,counter_intibute_mani] video_input.num_frames_test=12 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2

sbatch scripts/eval_ret.sh anet /home/wiss/zhang/Jinhe/singularity/paper_results/ret_anet/anet_anet_train_3_Seed3/ckpt_best.pth anet_eval_3_Seed3_mani local 1 12356 \ test_types=[temporal_contact_swap_mani,temporal_action_swap_mani,neighborhood_same_entity_mani,neighborhood_diff_entity_mani,counter_spatial_mani,counter_contact_mani,counter_action_mani,counter_intibute_mani] video_input.num_frames_test=12 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2



#eval mg
sbatch scripts/eval_ret.sh moviegraph /home/wiss/zhang/Jinhe/singularity/paper_results/ret_moviegraph/moviegraph_moviegraph_train_1_Seed42/ckpt_best.pth moviegraph_eval_1_Seed42 local 1 12357 \ test_types=[temporal_int,temporal_act,neighborhood_same_entity,neighborhood_diff_entity,counter_rel,counter_act,counter_int,counter_int] video_input.num_frames_test=12 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2

sbatch scripts/eval_ret.sh moviegraph /home/wiss/zhang/Jinhe/singularity/paper_results/ret_moviegraph/moviegraph_moviegraph_train_2_Seed2/ckpt_best.pth moviegraph_eval_2_Seed2 local 1 12358 \ test_types=[temporal_int,temporal_act,neighborhood_same_entity,neighborhood_diff_entity,counter_rel,counter_act,counter_int,counter_int] video_input.num_frames_test=12 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2

sbatch scripts/eval_ret.sh moviegraph /home/wiss/zhang/Jinhe/singularity/paper_results/ret_moviegraph/moviegraph_moviegraph_train_1_Seed3/ckpt_best.pth moviegraph_eval_1_Seed3 local 1 12359 \ test_types=[temporal_int,temporal_act,neighborhood_same_entity,neighborhood_diff_entity,counter_rel,counter_act,counter_int,counter_int] video_input.num_frames_test=12 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2


#mani
sbatch scripts/eval_ret.sh moviegraph /home/wiss/zhang/Jinhe/singularity/paper_results/ret_moviegraph/moviegraph_moviegraph_train_1_Seed42/ckpt_best.pth moviegraph_eval_1_Seed42_mani local 1 12360 \ test_types=[temporal_int_mani,temporal_act_mani,neighborhood_same_entity_mani,neighborhood_diff_entity_mani,counter_rel_mani,counter_act_mani,counter_int_mani,counter_int_mani] video_input.num_frames_test=12 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2

sbatch scripts/eval_ret.sh moviegraph /home/wiss/zhang/Jinhe/singularity/paper_results/ret_moviegraph/moviegraph_moviegraph_train_2_Seed2/ckpt_best.pth moviegraph_eval_2_Seed2_mani local 1 12361 \ test_types=[temporal_int_mani,temporal_act_mani,neighborhood_same_entity_mani,neighborhood_diff_entity_mani,counter_rel_mani,counter_act_mani,counter_int_mani,counter_int_mani] video_input.num_frames_test=12 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2

sbatch scripts/eval_ret.sh moviegraph /home/wiss/zhang/Jinhe/singularity/paper_results/ret_moviegraph/moviegraph_moviegraph_train_1_Seed3/ckpt_best.pth moviegraph_eval_1_Seed3_mani local 1 12362 \ test_types=[temporal_int_mani,temporal_act_mani,neighborhood_same_entity_mani,neighborhood_diff_entity_mani,counter_rel_mani,counter_act_mani,counter_int_mani,counter_int_mani] video_input.num_frames_test=12 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2



#eval mg
sbatch scripts/eval_ret.sh moviegraph /home/wiss/zhang/Jinhe/singularity/paper_results/ret_moviegraph/moviegraph_moviegraph_train_1_Seed42/ckpt_best.pth moviegraph_eval_1_Seed42_int local 1 12357 \ test_types=[counter_int] video_input.num_frames_test=12 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2

sbatch scripts/eval_ret.sh moviegraph /home/wiss/zhang/Jinhe/singularity/paper_results/ret_moviegraph/moviegraph_moviegraph_train_2_Seed2/ckpt_best.pth moviegraph_eval_2_Seed2_int local 1 12358 \ test_types=[counter_int] video_input.num_frames_test=12 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2

sbatch scripts/eval_ret.sh moviegraph /home/wiss/zhang/Jinhe/singularity/paper_results/ret_moviegraph/moviegraph_moviegraph_train_1_Seed3/ckpt_best.pth moviegraph_eval_1_Seed3_int local 1 12359 \ test_types=[counter_int] video_input.num_frames_test=12 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2


#mani
sbatch scripts/eval_ret.sh moviegraph /home/wiss/zhang/Jinhe/singularity/paper_results/ret_moviegraph/moviegraph_moviegraph_train_1_Seed42/ckpt_best.pth moviegraph_eval_1_Seed42_mani_int local 1 12360 \ test_types=[counter_int_mani] video_input.num_frames_test=12 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2

sbatch scripts/eval_ret.sh moviegraph /home/wiss/zhang/Jinhe/singularity/paper_results/ret_moviegraph/moviegraph_moviegraph_train_2_Seed2/ckpt_best.pth moviegraph_eval_2_Seed2_mani_int local 1 12361 \ test_types=[counter_int_mani] video_input.num_frames_test=12 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2

sbatch scripts/eval_ret.sh moviegraph /home/wiss/zhang/Jinhe/singularity/paper_results/ret_moviegraph/moviegraph_moviegraph_train_1_Seed3/ckpt_best.pth moviegraph_eval_1_Seed3_mani_int local 1 12362 \ test_types=[counter_int_mani] video_input.num_frames_test=12 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2










temporal_contact_swap,temporal_action_swap,neighborhood_same_entity,neighborhood_diff_entity,counter_spatial,counter_contact,counter_action,counter_intibute

temporal_int,temporal_act,neighborhood_same_entity,neighborhood_diff_entity,counter_rel,counter_act,counter_int,counter_int,temporal_int_mani,temporal_act_mani,neighborhood_same_entity_mani,neighborhood_diff_entity_mani,counter_rel_mani,counter_act_mani,counter_int_mani,counter_int_mani




#qa task eval

sbatch scripts/eval_anet.sh anet /home/wiss/zhang/Jinhe/singularity/anet_qa/ft_anet_qa_singularity_17m.pth anetqa_show local 1 \ test_types=[val]

#qa task train
sbatch scripts/train_vqa.sh anetqa_train_qa anet 2 local 12363 pretrained_path=/home/wiss/zhang/Jinhe/singularity/anet_qa/ft_anet_qa_singularity_temporal_17m.pth \ video_input.num_frames=4 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2

/home/wiss/zhang/Jinhe/singularity/pt/singularity_temporal_17m.pth

sbatch scripts/train_vqa.sh anetqa_train_qa anet 2 local 12364 pretrained_path=/home/wiss/zhang/Jinhe/singularity/pt/singularity_temporal_17m.pth \ video_input.num_frames=4 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2

sbatch scripts/train_vqa_full.sh anetqa_train_qa_full anet 2 local 12365 pretrained_path=/home/wiss/zhang/Jinhe/singularity/pt/singularity_temporal_17m.pth \ video_input.num_frames=4 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2

sbatch scripts/train_vqa_full.sh anetqa_train_qa_full_4 anet 4 local 12366 pretrained_path=/home/wiss/zhang/Jinhe/singularity/pt/singularity_temporal_17m.pth \ video_input.num_frames=4 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2



#test model structure
sbatch scripts/test.sh anet  /home/wiss/zhang/nfs/anetqa_train_qa_full/ckpt_best.pth model_test_Qa local 1 \ test_types=[val] video_input.num_frames_test=12 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2

sbatch scripts/test_ret.sh anet  /home/wiss/zhang/Jinhe/singularity/paper_results/ret_moviegraph/moviegraph_moviegraph_train_1_Seed42/ckpt_best.pth model_test_ret local 1 22225 \ test_types=[temporal_contact_swap] video_input.num_frames_test=12 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2



#model modify
#qa model ckpt: /home/wiss/zhang/nfs/anetqa_train_qa_full/ckpt_best.pth

#ret model ckpt:  /home/wiss/zhang/Jinhe/singularity/paper_results/ret_moviegraph/moviegraph_moviegraph_train_1_Seed42/ckpt_best.pth

sbatch scripts/test_ret.sh anet  /home/wiss/zhang/nfs/anetqa_train_qa_full/ckpt_best.pth model_test_ret_model local 1 22226 \ test_types=[temporal_contact_swap] video_input.num_frames_test=12 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2

sbatch scripts/test.sh anet  /home/wiss/zhang/Jinhe/singularity/paper_results/ret_moviegraph/moviegraph_moviegraph_train_1_Seed42/ckpt_best.pth model_test_Qa_model local 1 \ test_types=[val] video_input.num_frames_test=12 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2

sbatch scripts/train_ret_anet.sh anet_neg anet 1 local 22347 pretrained_path=/home/wiss/zhang/nfs/anetqa_train_qa_full/ckpt_best.pth \ video_input.num_frames=4 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2 \ batch_size.video=2 \ scheduler.epoch=30 

/home/wiss/zhang/nfs/anet_anet_train_1_Seed42/ckpt_best.pth

sbatch scripts/train_ret_anet_neg.sh anet_neg_0.001 anet 2 local 22349 pretrained_path=/home/wiss/zhang/nfs/anet_anet_train_1_Seed42/ckpt_best.pth \ test_types=[temporal_contact_swap] video_input.num_frames=4 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2 \ scheduler.epoch=30 \ batch_size.video=32

sbatch scripts/train_ret_anet_neg.sh anet_neg_0.001_4 anet 4 local 22350 pretrained_path=/home/wiss/zhang/nfs/anet_anet_train_1_Seed42/ckpt_best.pth \ test_types=[temporal_contact_swap] video_input.num_frames=4 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2 \ scheduler.epoch=30 \ batch_size.video=32



#qa freeze
sbatch scripts/train_vqa_full.sh anetqa_train_qa_full_freeze anet 2 local 22351 pretrained_path=/home/wiss/zhang/nfs/anet_anet_train_1_Seed42/ckpt_best.pth \ video_input.num_frames=4 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2 

sbatch scripts/train_vqa_full.sh anetqa_train_qa_full_freeze_improve anet 2 local 22352 pretrained_path=/home/wiss/zhang/nfs/anet_anet_neg_0.001/ckpt_best.pth \ video_input.num_frames=4 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2

sbatch scripts/train_vqa_full.sh anetqa_train_qa_full_freeze_improve_2 anet 2 local 22352 pretrained_path=/home/wiss/zhang/nfs/anet_neg_ckpt/anet_anet_neg_0.001_4/ckpt_best.pth \ video_input.num_frames=4 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2

sbatch scripts/train_vqa_full.sh anetqa_train_qa_full_freeze_improve_3 anet 2 local 22353 pretrained_path=/home/wiss/zhang/nfs/anet_anet_neg_0.001_4/ckpt_best.pth \ video_input.num_frames=4 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2



sbatch scripts/eval_ret.sh anet /home/wiss/zhang/nfs/anet_anet_neg_1/ckpt_best.pth anet_eval_improvement local 1 22357 \ test_types=[temporal_contact_swap,temporal_action_swap,neighborhood_same_entity,neighborhood_diff_entity,counter_spatial,counter_contact,counter_action,counter_attribute] video_input.num_frames_test=12 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2

sbatch scripts/eval_ret.sh anet /home/wiss/zhang/nfs/anet_anet_neg_1/ckpt_best.pth anet_eval_improvement_mani local 1 22358 \ test_types=[temporal_contact_swap_mani,temporal_action_swap_mani,neighborhood_same_entity_mani,neighborhood_diff_entity_mani,counter_spatial_mani,counter_contact_mani,counter_action_mani,counter_attribute_mani] video_input.num_frames_test=12 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2

/home/wiss/zhang/nfs/anet_anet_neg_0.1/ckpt_best.pth

sbatch scripts/eval_ret.sh anet /home/wiss/zhang/nfs/anet_anet_neg_0.1/ckpt_best.pth anet_eval_improvement_mani local 1 22359 \ test_types=[temporal_contact_swap_mani,temporal_action_swap_mani,neighborhood_same_entity_mani,neighborhood_diff_entity_mani,counter_spatial_mani,counter_contact_mani,counter_action_mani,counter_attribute_mani] video_input.num_frames_test=12 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2

sbatch scripts/eval_ret.sh anet /home/wiss/zhang/nfs/anet_anet_neg_0.1/ckpt_best.pth anet_eval_improvement local 1 22360 \ test_types=[temporal_contact_swap,temporal_action_swap,neighborhood_same_entity,neighborhood_diff_entity,counter_spatial,counter_contact,counter_action,counter_attribute] video_input.num_frames_test=12 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2

/home/wiss/zhang/nfs/anet_anet_neg_0.001/ckpt_best.pth

sbatch scripts/eval_ret.sh anet /home/wiss/zhang/nfs/anet_anet_neg_0.001/ckpt_best.pth anet_eval_improvement_mani local 1 22361 \ test_types=[temporal_contact_swap_mani,temporal_action_swap_mani,neighborhood_same_entity_mani,neighborhood_diff_entity_mani,counter_spatial_mani,counter_contact_mani,counter_action_mani,counter_attribute_mani] video_input.num_frames_test=12 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2

sbatch scripts/eval_ret.sh anet /home/wiss/zhang/nfs/anet_anet_neg_0.001/ckpt_best.pth anet_eval_improvement local 1 22362 \ test_types=[temporal_contact_swap,temporal_action_swap,neighborhood_same_entity,neighborhood_diff_entity,counter_spatial,counter_contact,counter_action,counter_attribute] video_input.num_frames_test=12 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2


\temporal

sbatch scripts/train_ret_anet_neg.sh anet_neg_0.1_temp anet 2 local 22354 pretrained_path=/home/wiss/zhang/nfs/anet_anet_train_1_Seed42/ckpt_best.pth \ test_types=[temporal_contact_swap] video_input.num_frames=4 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2 \ scheduler.epoch=30 \ batch_size.video=32

sbatch scripts/train_ret_anet_neg.sh anet_neg_1_temp anet 1 local 22355 pretrained_path=/home/wiss/zhang/nfs/anet_anet_train_1_Seed42/ckpt_best.pth \ test_types=[temporal_contact_swap] video_input.num_frames=4 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2 \ scheduler.epoch=30 \ batch_size.video=32

sbatch scripts/train_ret_anet_neg.sh anet_neg_0.5_temp anet 2 local 22356 pretrained_path=/home/wiss/zhang/nfs/anet_anet_train_1_Seed42/ckpt_best.pth \ test_types=[temporal_contact_swap] video_input.num_frames=4 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2 \ scheduler.epoch=30 \ batch_size.video=32



sbatch scripts/eval_ret.sh anet /home/wiss/zhang/nfs/anet_neg_ckpt/anet_anet_neg_0.5_temp/ckpt_best.pth anet_eval_improvement local 1 22363 \ test_types=[temporal_contact_swap,temporal_action_swap,neighborhood_same_entity,neighborhood_diff_entity,counter_spatial,counter_contact,counter_action,counter_attribute] video_input.num_frames_test=12 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2

sbatch scripts/eval_ret.sh anet /home/wiss/zhang/nfs/anet_neg_ckpt/anet_anet_neg_0.5_temp/ckpt_best.pth anet_eval_improvement_mani local 1 22364 \ test_types=[temporal_contact_swap_mani,temporal_action_swap_mani,neighborhood_same_entity_mani,neighborhood_diff_entity_mani,counter_spatial_mani,counter_contact_mani,counter_action_mani,counter_attribute_mani] video_input.num_frames_test=12 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2


sbatch scripts/eval_ret.sh anet /home/wiss/zhang/Jinhe/singularity/neg/ret_anet/anet_anet_neg_0.1_temp/ckpt_best.pth anet_eval_improvement local 1 22365 \ test_types=[temporal_contact_swap,temporal_action_swap,neighborhood_same_entity,neighborhood_diff_entity,counter_spatial,counter_contact,counter_action,counter_attribute] video_input.num_frames_test=12 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2

sbatch scripts/eval_ret.sh anet /home/wiss/zhang/Jinhe/singularity/neg/ret_anet/anet_anet_neg_0.1_temp/ckpt_best.pth anet_eval_improvement_mani local 1 22366 \ test_types=[temporal_contact_swap_mani,temporal_action_swap_mani,neighborhood_same_entity_mani,neighborhood_diff_entity_mani,counter_spatial_mani,counter_contact_mani,counter_action_mani,counter_attribute_mani] video_input.num_frames_test=12 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2




sbatch scripts/eval_ret.sh anet /home/wiss/zhang/Jinhe/singularity/neg/ret_anet/anet_anet_neg_1_temp/ckpt_best.pth anet_eval_improvement local 1 22367 \ test_types=[temporal_contact_swap,temporal_action_swap,neighborhood_same_entity,neighborhood_diff_entity,counter_spatial,counter_contact,counter_action,counter_attribute] video_input.num_frames_test=12 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2

sbatch scripts/eval_ret.sh anet /home/wiss/zhang/Jinhe/singularity/neg/ret_anet/anet_anet_neg_1_temp/ckpt_best.pth anet_eval_improvement_mani local 1 22368 \ test_types=[temporal_contact_swap_mani,temporal_action_swap_mani,neighborhood_same_entity_mani,neighborhood_diff_entity_mani,counter_spatial_mani,counter_contact_mani,counter_action_mani,counter_attribute_mani] video_input.num_frames_test=12 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2


sbatch scripts/train_vqa_full.sh anetqa_train_anet_neg_0.5_temp anet 2 local 22410 pretrained_path=/home/wiss/zhang/nfs/anet_neg_ckpt/anet_anet_neg_0.5_temp/ckpt_best.pth \ video_input.num_frames=4 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2



sbatch scripts/train_vqa_full.sh anetqa_train_qa_full_freeze_neg_0.1_temp anet 2 local 22510 pretrained_path=/home/wiss/zhang/nfs/anet_neg_ckpt/anet_anet_neg_0.1_temp/ckpt_best.pth \ video_input.num_frames=4 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2

sbatch scripts/train_vqa_full.sh anetqa_train_qa_full_freeze_neg_0.5_temp anet 2 local 22511 pretrained_path=/home/wiss/zhang/nfs/anet_neg_ckpt/anet_anet_neg_0.5_temp/ckpt_best.pth \ video_input.num_frames=4 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2

sbatch scripts/train_vqa_full.sh anetqa_train_qa_full_freeze_neg_1_temp anet 2 local 22512 pretrained_path=/home/wiss/zhang/Jinhe/singularity/neg/ret_anet/anet_anet_neg_1_temp/ckpt_best.pth \ video_input.num_frames=4 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2


#qa freeze
sbatch scripts/train_vqa_full.sh anetqa_train_qa_full_freeze_new_subset anet 1 local 22351 pretrained_path=/home/wiss/zhang/nfs/anet_anet_train_1_Seed42/ckpt_best.pth \ video_input.num_frames=4 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2 

sbatch scripts/train_vqa_full.sh anetqa_train_qa_full_freeze_improve anet 2 local 22352 pretrained_path=/home/wiss/zhang/nfs/anet_anet_neg_0.001/ckpt_best.pth \ video_input.num_frames=4 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2

sbatch scripts/train_vqa_full.sh anetqa_train_qa_full_freeze_improve_2 anet 2 local 22352 pretrained_path=/home/wiss/zhang/nfs/anet_neg_ckpt/anet_anet_neg_0.001_4/ckpt_best.pth \ video_input.num_frames=4 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2

sbatch scripts/train_vqa_full.sh anetqa_train_qa_full_freeze_improve_3 anet 2 local 22353 pretrained_path=/home/wiss/zhang/nfs/anet_anet_neg_0.001_4/ckpt_best.pth \ video_input.num_frames=4 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2

sbatch scripts/train_vqa_full.sh anetqa_train_qa_full_freeze_neg_0.1_temp_new_subset anet 2 local 33333 pretrained_path=/home/wiss/zhang/nfs/anet_neg_ckpt/anet_anet_neg_0.1_temp/ckpt_best.pth \ video_input.num_frames=4 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2

sbatch scripts/train_vqa_full.sh anetqa_train_qa_full_freeze_neg_0.5_temp_new_subset anet 2 local 33334 pretrained_path=/home/wiss/zhang/nfs/anet_neg_ckpt/anet_anet_neg_0.5_temp/ckpt_best.pth \ video_input.num_frames=4 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2

sbatch scripts/train_vqa_full.sh anetqa_train_qa_full_freeze_neg_1_temp_new_subset anet 2 local 33335 pretrained_path=/home/wiss/zhang/Jinhe/singularity/neg/ret_anet/anet_anet_neg_1_temp/ckpt_best.pth \ video_input.num_frames=4 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2



sbatch scripts/train_ret_anet_neg.sh anet_neg_0_from_scratch anet 2 local 44443 pretrained_path=/home/wiss/zhang/Jinhe/singularity/pt/singularity_temporal_17m.pth \ test_types=[temporal_contact_swap] video_input.num_frames=4 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2 \ scheduler.epoch=10 \ batch_size.video=32 \ neg_ratio=0

sbatch scripts/train_ret_anet_neg.sh anet_neg_0.5_from_scratch anet 2 local 44444 pretrained_path=/home/wiss/zhang/Jinhe/singularity/pt/singularity_temporal_17m.pth \ test_types=[temporal_contact_swap] video_input.num_frames=4 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2 \ scheduler.epoch=10 \ batch_size.video=32 \ neg_ratio=0.5

sbatch scripts/train_ret_anet_neg.sh anet_neg_1_from_scratch anet 2 local 44445 pretrained_path=/home/wiss/zhang/Jinhe/singularity/pt/singularity_temporal_17m.pth \ test_types=[temporal_contact_swap] video_input.num_frames=4 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2 \ scheduler.epoch=10 \ batch_size.video=32 \ neg_ratio=1

sbatch scripts/train_ret_anet_neg.sh anet_neg_5_from_scratch anet 2 local 44446 pretrained_path=/home/wiss/zhang/Jinhe/singularity/pt/singularity_temporal_17m.pth \ test_types=[temporal_contact_swap] video_input.num_frames=4 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2 \ scheduler.epoch=10 \ batch_size.video=32 \ neg_ratio=5


sbatch scripts/train_vqa_full.sh anet_neg_0_from_scratch anet 2 local 44447 pretrained_path=/home/wiss/zhang/nfs/anet_neg_ckpt/anet_anet_neg_0_from_scratch/ckpt_best.pth \ video_input.num_frames=4 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2
sbatch scripts/train_vqa_full.sh anet_neg_0.5_from_scratch anet 2 local 44448 pretrained_path=/home/wiss/zhang/nfs/anet_neg_ckpt/anet_anet_neg_0.5_from_scratch/ckpt_best.pth \ video_input.num_frames=4 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2
sbatch scripts/train_vqa_full.sh anet_neg_1_from_scratch anet 2 local 44450 pretrained_path=/home/wiss/zhang/Jinhe/singularity/neg/ret_anet/anet_anet_neg_1_from_scratch/ckpt_best.pth \ video_input.num_frames=4 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2
sbatch scripts/train_vqa_full.sh anet_neg_5_from_scratch anet 2 local 44461 pretrained_path=/home/wiss/zhang/Jinhe/singularity/neg/ret_anet/anet_anet_neg_5_from_scratch/ckpt_best.pth \ video_input.num_frames=4 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2
sbatch scripts/train_vqa_full.sh anet_neg_0.5_from_scratch_temp_neg anet 2 local 44462 pretrained_path=/home/wiss/zhang/Jinhe/singularity/neg/ret_anet/anet_anet_neg_0.5_from_scratch_temp_neg/ckpt_best.pth \ video_input.num_frames=4 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2

sbatch scripts/train_ret_anet_neg.sh anet_neg_0.5_from_scratch_temp_neg anet 2 local 44449 pretrained_path=/home/wiss/zhang/Jinhe/singularity/pt/singularity_temporal_17m.pth \ test_types=[temporal_contact_swap] video_input.num_frames=4 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2 \ scheduler.epoch=10 \ batch_size.video=32 \ neg_ratio=0.5 \ temp_neg=True 

sbatch scripts/eval_anet.sh anet /home/wiss/zhang/Jinhe/singularity/qa_anet/anet_neg_0_from_scratch/ckpt_best.pth anet_eval_0_from_sc_light_answer local 1 \ video_input.num_frames=4 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2 \ answer_list=/nfs/data2/zhang/AnetQA/qa_train/answer_list_light.json

sbatch scripts/eval_anet.sh anet /home/wiss/zhang/Jinhe/singularity/qa_anet/anet_neg_0.5_from_scratch/ckpt_best.pth anet_eval_0.5_from_sc_light_answer local 1 \ video_input.num_frames=4 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2 \ answer_list=/nfs/data2/zhang/AnetQA/qa_train/answer_list_light.json



sbatch scripts/eval_anet.sh anet /home/wiss/zhang/Jinhe/singularity/qa_anet/anet_neg_1_from_scratch/ckpt_best.pth anet_eval_1_from_sc_light_answer local 1 44451 \ video_input.num_frames=4 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2 \ answer_list=/nfs/data2/zhang/AnetQA/qa_train/answer_list_light.json

sbatch scripts/eval_anet.sh anet /home/wiss/zhang/Jinhe/singularity/qa_anet/anet_neg_5_from_scratch/ckpt_best.pth anet_eval_5_from_sc_light_answer local 1 44452 \ video_input.num_frames=4 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2 \ answer_list=/nfs/data2/zhang/AnetQA/qa_train/answer_list_light.json

sbatch scripts/eval_anet.sh anet /home/wiss/zhang/Jinhe/singularity/qa_anet/anet_neg_0_from_scratch/ckpt_best.pth anet_eval_0_from_sc_full_answer local 1 44523\ video_input.num_frames=4 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2 \ answer_list=/nfs/data2/zhang/AnetQA/qa_train/answer_list.json



/home/wiss/zhang/Jinhe/singularity/neg/ret_anet/anet_anet_neg_0.5_from_scratch_temp_neg/ckpt_best.pth

sbatch scripts/train_ret_anet_neg.sh test anet 2 local 44463 pretrained_path=/home/wiss/zhang/Jinhe/singularity/neg/ret_anet/anet_anet_neg_0.5_from_scratch_temp_neg/ckpt_best.pth \ test_types=[temporal_contact_swap] video_input.num_frames=4 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2 \ scheduler.epoch=10 \ batch_size.video=32 \ neg_ratio=0.5 \ temp_neg=True 


#mil_loss
sbatch scripts/train_ret_anet_neg.sh mil_anet_neg_1_from_scratch anet 2 local 10000 pretrained_path=/home/wiss/zhang/Jinhe/singularity/pt/singularity_temporal_17m.pth \ test_types=[temporal_contact_swap] video_input.num_frames=4 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2 \ scheduler.epoch=10 \ batch_size.video=32 \ neg_ratio=1 \ temp_neg=False \ train_type=anet_ret_train_1_pos_neg.json \ wandb.enable=True
sbatch scripts/train_ret_anet_neg.sh mil_anet_neg_0_from_scratch anet 2 local 10001 pretrained_path=/home/wiss/zhang/Jinhe/singularity/pt/singularity_temporal_17m.pth \ test_types=[temporal_contact_swap] video_input.num_frames=4 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2 \ scheduler.epoch=10 \ batch_size.video=32 \ neg_ratio=0 \ temp_neg=False \ train_type=anet_ret_train_1_pos_neg.json \ wandb.enable=True

sbatch scripts/train_ret_anet_neg.sh mil_anet_neg_1_from_scratch_wo_temp_neg anet 2 local 10002 pretrained_path=/home/wiss/zhang/Jinhe/singularity/pt/singularity_temporal_17m.pth \ test_types=[temporal_contact_swap] video_input.num_frames=4 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2 \ scheduler.epoch=10 \ batch_size.video=32 \ neg_ratio=1 \ temp_neg=False \ train_type=anet_ret_train_1_pos_neg.json \ wandb.enable=True
sbatch scripts/train_ret_anet_neg.sh mil_anet_neg_0_from_scratch_wo_temp_neg anet 2 local 10003 pretrained_path=/home/wiss/zhang/Jinhe/singularity/pt/singularity_temporal_17m.pth \ test_types=[temporal_contact_swap] video_input.num_frames=4 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2 \ scheduler.epoch=10 \ batch_size.video=32 \ neg_ratio=0 \ temp_neg=False \ train_type=anet_ret_train_1_pos_neg.json \ wandb.enable=True

sbatch scripts/train_ret_anet_neg.sh mil_beta_test anet 1 local 10004 pretrained_path=/home/wiss/zhang/Jinhe/singularity/pt/singularity_temporal_17m.pth \ test_types=[temporal_contact_swap] video_input.num_frames=4 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2 \ scheduler.epoch=10 \ batch_size.video=4 \ neg_ratio=1 \ temp_neg=False \ train_type=anet_ret_train_1_pos_neg.json \ wandb.enable=True

# mil_beta
sbatch scripts/train_ret_anet_neg.sh mil_beta_0.5_7_pos_ave anet 2 local 10004 pretrained_path=/home/wiss/zhang/Jinhe/singularity/pt/singularity_temporal_17m.pth \ test_types=[temporal_contact_swap] video_input.num_frames=4 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2 \ scheduler.epoch=10 \ batch_size.video=32 \ neg_ratio=1 \ temp_neg=False \ train_type=anet_ret_train_1_pos_neg.json \ wandb.enable=True
sbatch scripts/train_ret_anet_neg.sh mil_beta_0.5_7_pos_no_ave anet 2 local 10008 pretrained_path=/home/wiss/zhang/Jinhe/singularity/pt/singularity_temporal_17m.pth \ test_types=[temporal_contact_swap] video_input.num_frames=4 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2 \ scheduler.epoch=10 \ batch_size.video=32 \ neg_ratio=1 \ temp_neg=False \ train_type=anet_ret_train_1_pos_neg.json \ wandb.enable=True
sbatch scripts/train_ret_anet_neg.sh mil_beta_0.5_3_pos_no_ave anet 2 local 10009 pretrained_path=/home/wiss/zhang/Jinhe/singularity/pt/singularity_temporal_17m.pth \ test_types=[temporal_contact_swap] video_input.num_frames=4 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2 \ scheduler.epoch=10 \ batch_size.video=32 \ neg_ratio=1 \ temp_neg=False \ train_type=anet_ret_train_1_pos_neg.json \ wandb.enable=True \ num_pos=3
sbatch scripts/train_ret_anet_neg.sh mil_beta_0.5_1_pos_no_ave anet 2 local 10010 pretrained_path=/home/wiss/zhang/Jinhe/singularity/pt/singularity_temporal_17m.pth \ test_types=[temporal_contact_swap] video_input.num_frames=4 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2 \ scheduler.epoch=10 \ batch_size.video=32 \ neg_ratio=1 \ temp_neg=False \ train_type=anet_ret_train_1_pos_neg.json \ wandb.enable=True \ num_pos=1

sbatch scripts/train_ret_anet_neg.sh mil_beta_0.5_7_pos_no_ave_5_neg anet 2 local 10011 pretrained_path=/home/wiss/zhang/Jinhe/singularity/pt/singularity_temporal_17m.pth \ test_types=[temporal_contact_swap] video_input.num_frames=4 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2 \ scheduler.epoch=10 \ batch_size.video=32 \ neg_ratio=1 \ temp_neg=False \ train_type=anet_ret_train_1_pos_neg.json \ wandb.enable=True \ num_pos=7 \ num_neg=5
sbatch scripts/train_ret_anet_neg.sh mil_beta_0.5_7_pos_no_ave_3_neg anet 2 local 10012 pretrained_path=/home/wiss/zhang/Jinhe/singularity/pt/singularity_temporal_17m.pth \ test_types=[temporal_contact_swap] video_input.num_frames=4 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2 \ scheduler.epoch=10 \ batch_size.video=32 \ neg_ratio=1 \ temp_neg=False \ train_type=anet_ret_train_1_pos_neg.json \ wandb.enable=True \ num_pos=7 \ num_neg=3
sbatch scripts/train_ret_anet_neg.sh mil_beta_0.5_7_pos_no_ave_1_neg anet 2 local 10013 pretrained_path=/home/wiss/zhang/Jinhe/singularity/pt/singularity_temporal_17m.pth \ test_types=[temporal_contact_swap] video_input.num_frames=4 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2 \ scheduler.epoch=10 \ batch_size.video=32 \ neg_ratio=1 \ temp_neg=False \ train_type=anet_ret_train_1_pos_neg.json \ wandb.enable=True \ num_pos=7 \ num_neg=1
sbatch scripts/train_ret_anet_neg.sh mil_beta_0.5_7_pos_no_ave_7_neg anet 2 local 10014 pretrained_path=/home/wiss/zhang/Jinhe/singularity/pt/singularity_temporal_17m.pth \ test_types=[temporal_contact_swap] video_input.num_frames=4 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2 \ scheduler.epoch=10 \ batch_size.video=32 \ neg_ratio=1 \ temp_neg=False \ train_type=anet_ret_train_1_pos_neg.json \ wandb.enable=True \ num_pos=7 \ num_neg=7
sbatch scripts/train_ret_anet_neg.sh mil_beta_0.5_7_pos_no_ave_0_neg anet 2 local 10022 pretrained_path=/home/wiss/zhang/Jinhe/singularity/pt/singularity_temporal_17m.pth \ test_types=[temporal_contact_swap] video_input.num_frames=4 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2 \ scheduler.epoch=10 \ batch_size.video=32 \ neg_ratio=1 \ temp_neg=False \ train_type=anet_ret_train_1_pos_neg.json \ wandb.enable=True \ num_pos=7 \ num_neg=0


sbatch scripts/train_ret_anet_neg.sh mil_beta_0.5_3_pos_no_ave_1_neg anet 2 local 10023 pretrained_path=/home/wiss/zhang/Jinhe/singularity/pt/singularity_temporal_17m.pth \ test_types=[temporal_contact_swap] video_input.num_frames=4 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2 \ scheduler.epoch=10 \ batch_size.video=32 \ neg_ratio=1 \ temp_neg=False \ train_type=anet_ret_train_1_pos_neg.json \ wandb.enable=True \ num_pos=3 \ num_neg=1
sbatch scripts/train_ret_anet_neg.sh mil_beta_0.5_3_pos_no_ave_3_neg anet 2 local 10024 pretrained_path=/home/wiss/zhang/Jinhe/singularity/pt/singularity_temporal_17m.pth \ test_types=[temporal_contact_swap] video_input.num_frames=4 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2 \ scheduler.epoch=10 \ batch_size.video=32 \ neg_ratio=1 \ temp_neg=False \ train_type=anet_ret_train_1_pos_neg.json \ wandb.enable=True \ num_pos=3 \ num_neg=3
sbatch scripts/train_ret_anet_neg.sh mil_beta_0.5_3_pos_no_ave_7_neg anet 2 local 10025 pretrained_path=/home/wiss/zhang/Jinhe/singularity/pt/singularity_temporal_17m.pth \ test_types=[temporal_contact_swap] video_input.num_frames=4 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2 \ scheduler.epoch=10 \ batch_size.video=32 \ neg_ratio=1 \ temp_neg=False \ train_type=anet_ret_train_1_pos_neg.json \ wandb.enable=True \ num_pos=3 \ num_neg=7





sbatch scripts/train_vqa_full.sh mil_anet_neg_1_from_scratch_wo_temp_neg_correction_state anet 2 local 10005 pretrained_path=/home/wiss/zhang/Jinhe/singularity/mil/ret_anet/anet_mil_anet_neg_1_from_scratch_wo_temp_neg/ckpt_best.pth \ video_input.num_frames=4 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2
sbatch scripts/train_vqa_full.sh mil_anet_neg_0_from_scratch_wo_temp_neg_correction_state anet 2 local 10006 pretrained_path=/home/wiss/zhang/Jinhe/singularity/mil/ret_anet/anet_mil_anet_neg_0_from_scratch_wo_temp_neg/ckpt_best.pth \ video_input.num_frames=4 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2
sbatch scripts/train_vqa_full.sh mil_beta_0.5_7_pos_ave_correction_state_2 anet 2 local 10007 pretrained_path=/home/wiss/zhang/Jinhe/singularity/mil/ret_anet/anet_mil_beta_0.5_7_pos_ave/ckpt_best.pth \ video_input.num_frames=4 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2

sbatch scripts/train_vqa_full.sh mil_beta_0.5_7_pos_no_ave_correction_state anet 2 local 10015 pretrained_path=/home/wiss/zhang/Jinhe/singularity/mil/ret_anet/anet_mil_beta_0.5_7_pos_no_ave/ckpt_best.pth \ video_input.num_frames=4 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2
sbatch scripts/train_vqa_full.sh mil_beta_0.5_3_pos_no_ave_correction_state anet 2 local 10016 pretrained_path=/home/wiss/zhang/Jinhe/singularity/mil/ret_anet/anet_mil_beta_0.5_3_pos_no_ave/ckpt_best.pth \ video_input.num_frames=4 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2
sbatch scripts/train_vqa_full.sh mil_beta_0.5_1_pos_no_ave_correction_state anet 2 local 10017 pretrained_path=/home/wiss/zhang/Jinhe/singularity/mil/ret_anet/anet_mil_beta_0.5_1_pos_no_ave/ckpt_best.pth \ video_input.num_frames=4 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2
sbatch scripts/train_vqa_full.sh mil_beta_0.5_7_pos_no_ave_5_neg_correction_state anet 2 local 10018 pretrained_path=/home/wiss/zhang/Jinhe/singularity/mil/ret_anet/anet_mil_beta_0.5_7_pos_no_ave_5_neg/ckpt_best.pth \ video_input.num_frames=4 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2
sbatch scripts/train_vqa_full.sh mil_beta_0.5_7_pos_no_ave_1_neg_correction_state anet 2 local 10019 pretrained_path=/home/wiss/zhang/Jinhe/singularity/mil/ret_anet/anet_mil_beta_0.5_7_pos_no_ave_1_neg/ckpt_best.pth \ video_input.num_frames=4 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2
sbatch scripts/train_vqa_full.sh mil_beta_0.5_7_pos_no_ave_3_neg_correction_state anet 2 local 10020 pretrained_path=/home/wiss/zhang/Jinhe/singularity/mil/ret_anet/anet_mil_beta_0.5_7_pos_no_ave_3_neg/ckpt_best.pth \ video_input.num_frames=4 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2
sbatch scripts/train_vqa_full.sh mil_beta_0.5_7_pos_no_ave_7_neg_correction_state anet 2 local 10021 pretrained_path=/home/wiss/zhang/Jinhe/singularity/mil/ret_anet/anet_mil_beta_0.5_7_pos_no_ave_7_neg/ckpt_best.pth \ video_input.num_frames=4 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2
sbatch scripts/train_vqa_full.sh mil_beta_0.5_7_pos_no_ave_0_neg_correction_state anet 2 local 10026 pretrained_path=/home/wiss/zhang/Jinhe/singularity/mil/ret_anet/anet_mil_beta_0.5_7_pos_no_ave_0_neg/ckpt_best.pth \ video_input.num_frames=4 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2

sbatch scripts/train_vqa_full.sh mil_beta_0.5_3_pos_no_ave_1_neg_correction_state anet 2 local 10027 pretrained_path=/home/wiss/zhang/Jinhe/singularity/mil/ret_anet/anet_mil_beta_0.5_3_pos_no_ave_1_neg/ckpt_best.pth \ video_input.num_frames=4 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2
sbatch scripts/train_vqa_full.sh mil_beta_0.5_3_pos_no_ave_3_neg_correction_state anet 2 local 10028 pretrained_path=/home/wiss/zhang/Jinhe/singularity/mil/ret_anet/anet_mil_beta_0.5_3_pos_no_ave_3_neg/ckpt_best.pth \ video_input.num_frames=4 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2
sbatch scripts/train_vqa_full.sh mil_beta_0.5_3_pos_no_ave_7_neg_correction_state anet 2 local 10029 pretrained_path=/home/wiss/zhang/Jinhe/singularity/mil/ret_anet/anet_mil_beta_0.5_3_pos_no_ave_7_neg/ckpt_best.pth \ video_input.num_frames=4 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2
