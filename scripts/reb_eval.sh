model_name="anet_reb_beta_0.25_lr_0.0001_anet"  # Set your model name here
port=23333  # Initial port number

for epoch in {0..15}; do
  sbatch scripts/eval_ret.sh anet /home/wiss/zhang/Jinhe/singularity/reb/ret_anet/${model_name}/ckpt_epoch_${epoch}.pth ${model_name}_${epoch}_ori local 1 $port \ test_types=[temporal_contact_swap,temporal_action_swap,neighborhood_same_entity,neighborhood_diff_entity,counter_spatial,counter_contact,counter_action,counter_attribute] video_input.num_frames_test=12 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2
  ((port++))  # Increment port number

  sbatch scripts/eval_ret.sh anet /home/wiss/zhang/Jinhe/singularity/reb/ret_anet/${model_name}/ckpt_epoch_${epoch}.pth ${model_name}_${epoch}_mani local 1 $port \ test_types=[temporal_contact_swap_mani,temporal_action_swap_mani,neighborhood_same_entity_mani,neighborhood_diff_entity_mani,counter_spatial_mani,counter_contact_mani,counter_action_mani,counter_attribute_mani] video_input.num_frames_test=12 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2

  ((port++))  # Increment port number
done
