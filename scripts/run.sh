
sbatch scripts/train_ret_anet.sh anet_train_1_Seed42 anet 2 local 12345 pretrained_path=/home/wiss/zhang/Jinhe/singularity/anet_ret/ft_anet_ret_singularity_temporal_17m.pth \ video_input.num_frames=4 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2 \ batch_size.video=32 \ scheduler.epoch=30 \ train_type=anet_ret_train_1.json \ seed=42

sbatch scripts/train_ret_anet.sh anet_train_2_Seed2 anet 2 local 12346 pretrained_path=/home/wiss/zhang/Jinhe/singularity/anet_ret/ft_anet_ret_singularity_temporal_17m.pth \ video_input.num_frames=4 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2 \ batch_size.video=32 \ scheduler.epoch=30 \ train_type=anet_ret_train_2.json \ seed=2

sbatch scripts/train_ret_anet.sh anet_train_3_Seed3 anet 2 local 12347 pretrained_path=/home/wiss/zhang/Jinhe/singularity/anet_ret/ft_anet_ret_singularity_temporal_17m.pth \ video_input.num_frames=4 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2 \ batch_size.video=32 \ scheduler.epoch=30 \ train_type=anet_ret_train_3.json \ seed=3


sbatch scripts/train_ret_mg.sh moviegraph_train_1_Seed42 moviegraph 2 local 12348 pretrained_path=/home/wiss/zhang/Jinhe/singularity/pt/singularity_temporal_17m.pth \ video_input.num_frames=4 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2 \ batch_size.video=32 \ scheduler.epoch=30 \ train_type=anet_ret_train_1.json \ seed=42

sbatch scripts/train_ret_mg.sh moviegraph_train_2_Seed2 moviegraph 2 local 12349 pretrained_path=/home/wiss/zhang/Jinhe/singularity/pt/singularity_temporal_17m.pth \ video_input.num_frames=4 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2 \ batch_size.video=32 \ scheduler.epoch=30 \ train_type=anet_ret_train_2.json \ seed=2


sbatch scripts/train_ret_mg.sh moviegraph_train_1_Seed3 moviegraph 2 local 12350 pretrained_path=/home/wiss/zhang/Jinhe/singularity/pt/singularity_temporal_17m.pth \ video_input.num_frames=4 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2 \ batch_size.video=32 \ scheduler.epoch=30 \ train_type=anet_ret_train_1.json \ seed=3





#eval
sbatch scripts/eval_ret.sh anet /home/wiss/zhang/Jinhe/singularity/paper_results/ret_anet/anet_anet_train_1_Seed42/ckpt_best.pth anet_eval_1_Seed42 local 1 12351 \ test_types=[temporal_contact_swap,temporal_action_swap,neighborhood_same_entity,neighborhood_diff_entity,counter_spatial,counter_contact,counter_action,counter_attribute] video_input.num_frames_test=12 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2



#ori
sbatch scripts/eval_ret.sh anet /home/wiss/zhang/Jinhe/singularity/paper_results/ret_anet/anet_anet_train_1_Seed42/ckpt_best.pth anet_eval_1_Seed42 local 1 12351 \ test_types=[temporal_contact_swap,temporal_action_swap,neighborhood_same_entity,neighborhood_diff_entity,counter_spatial,counter_contact,counter_action,counter_attribute] video_input.num_frames_test=12 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2

sbatch scripts/eval_ret.sh anet /home/wiss/zhang/Jinhe/singularity/paper_results/ret_anet/anet_anet_train_2_Seed2/ckpt_best.pth anet_eval_2_Seed2 local 1 12352 \ test_types=[temporal_contact_swap,temporal_action_swap,neighborhood_same_entity,neighborhood_diff_entity,counter_spatial,counter_contact,counter_action,counter_attribute] video_input.num_frames_test=12 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2

sbatch scripts/eval_ret.sh anet /home/wiss/zhang/Jinhe/singularity/paper_results/ret_anet/anet_anet_train_3_Seed3/ckpt_best.pth anet_eval_3_Seed3 local 1 12353 \ test_types=[temporal_contact_swap,temporal_action_swap,neighborhood_same_entity,neighborhood_diff_entity,counter_spatial,counter_contact,counter_action,counter_attribute] video_input.num_frames_test=12 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2

#mani 

sbatch scripts/eval_ret.sh anet /home/wiss/zhang/Jinhe/singularity/paper_results/ret_anet/anet_anet_train_1_Seed42/ckpt_best.pth anet_eval_1_Seed42_mani local 1 12354 \ test_types=[temporal_contact_swap_mani,temporal_action_swap_mani,neighborhood_same_entity_mani,neighborhood_diff_entity_mani,counter_spatial_mani,counter_contact_mani,counter_action_mani,counter_attribute_mani] video_input.num_frames_test=12 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2

sbatch scripts/eval_ret.sh anet /home/wiss/zhang/Jinhe/singularity/paper_results/ret_anet/anet_anet_train_2_Seed2/ckpt_best.pth anet_eval_2_Seed2_mani local 1 12355 \ test_types=[temporal_contact_swap_mani,temporal_action_swap_mani,neighborhood_same_entity_mani,neighborhood_diff_entity_mani,counter_spatial_mani,counter_contact_mani,counter_action_mani,counter_attribute_mani] video_input.num_frames_test=12 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2

sbatch scripts/eval_ret.sh anet /home/wiss/zhang/Jinhe/singularity/paper_results/ret_anet/anet_anet_train_3_Seed3/ckpt_best.pth anet_eval_3_Seed3_mani local 1 12356 \ test_types=[temporal_contact_swap_mani,temporal_action_swap_mani,neighborhood_same_entity_mani,neighborhood_diff_entity_mani,counter_spatial_mani,counter_contact_mani,counter_action_mani,counter_attribute_mani] video_input.num_frames_test=12 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2



#eval mg
sbatch scripts/eval_ret.sh moviegraph /home/wiss/zhang/Jinhe/singularity/paper_results/ret_moviegraph/moviegraph_moviegraph_train_1_Seed42/ckpt_best.pth moviegraph_eval_1_Seed42 local 1 12357 \ test_types=[temporal_int,temporal_act,neighborhood_same_entity,neighborhood_diff_entity,counter_rel,counter_act,counter_int,counter_attr] video_input.num_frames_test=12 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2

sbatch scripts/eval_ret.sh moviegraph /home/wiss/zhang/Jinhe/singularity/paper_results/ret_moviegraph/moviegraph_moviegraph_train_2_Seed2/ckpt_best.pth moviegraph_eval_2_Seed2 local 1 12358 \ test_types=[temporal_int,temporal_act,neighborhood_same_entity,neighborhood_diff_entity,counter_rel,counter_act,counter_int,counter_attr] video_input.num_frames_test=12 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2

sbatch scripts/eval_ret.sh moviegraph /home/wiss/zhang/Jinhe/singularity/paper_results/ret_moviegraph/moviegraph_moviegraph_train_1_Seed3/ckpt_best.pth moviegraph_eval_1_Seed3 local 1 12359 \ test_types=[temporal_int,temporal_act,neighborhood_same_entity,neighborhood_diff_entity,counter_rel,counter_act,counter_int,counter_attr] video_input.num_frames_test=12 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2


#mani
sbatch scripts/eval_ret.sh moviegraph /home/wiss/zhang/Jinhe/singularity/paper_results/ret_moviegraph/moviegraph_moviegraph_train_1_Seed42/ckpt_best.pth moviegraph_eval_1_Seed42_mani local 1 12360 \ test_types=[temporal_int_mani,temporal_act_mani,neighborhood_same_entity_mani,neighborhood_diff_entity_mani,counter_rel_mani,counter_act_mani,counter_int_mani,counter_attr_mani] video_input.num_frames_test=12 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2

sbatch scripts/eval_ret.sh moviegraph /home/wiss/zhang/Jinhe/singularity/paper_results/ret_moviegraph/moviegraph_moviegraph_train_2_Seed2/ckpt_best.pth moviegraph_eval_2_Seed2_mani local 1 12361 \ test_types=[temporal_int_mani,temporal_act_mani,neighborhood_same_entity_mani,neighborhood_diff_entity_mani,counter_rel_mani,counter_act_mani,counter_int_mani,counter_attr_mani] video_input.num_frames_test=12 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2

sbatch scripts/eval_ret.sh moviegraph /home/wiss/zhang/Jinhe/singularity/paper_results/ret_moviegraph/moviegraph_moviegraph_train_1_Seed3/ckpt_best.pth moviegraph_eval_1_Seed3_mani local 1 12362 \ test_types=[temporal_int_mani,temporal_act_mani,neighborhood_same_entity_mani,neighborhood_diff_entity_mani,counter_rel_mani,counter_act_mani,counter_int_mani,counter_attr_mani] video_input.num_frames_test=12 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2



temporal_contact_swap,temporal_action_swap,neighborhood_same_entity,neighborhood_diff_entity,counter_spatial,counter_contact,counter_action,counter_attribute

temporal_int,temporal_act,neighborhood_same_entity,neighborhood_diff_entity,counter_rel,counter_act,counter_int,counter_attr,temporal_int_mani,temporal_act_mani,neighborhood_same_entity_mani,neighborhood_diff_entity_mani,counter_rel_mani,counter_act_mani,counter_int_mani,counter_attr_mani