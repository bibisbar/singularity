
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

sbatch scripts/train_ret_anet.sh anet_neg anet 1 local 22347 pretrained_path=/home/wiss/zhang/nfs/anetqa_train_qa_full/ckpt_best.pth \ video_input.num_frames=4 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2 \ batch_size.video=4 \ scheduler.epoch=30 \ train_type=anet_ret_train_3.json \ seed=3
