sbatch scripts/train_ret_anet.sh 17m_single_30 anet 2 local pretrained_path=/home/wiss/zhang/Jinhe/singularity/anet_ret/ft_anet_ret_singularity_17m.pth \ batch_size.video=32 \ scheduler.epoch=30


sbatch scripts/eval_ret.sh anet /home/wiss/zhang/Jinhe/singularity/anet_ret/ft_anet_ret_singularity_17m.pth reimplemret slurm 1 \ test_types=[val1,] video_input.num_frames_test=12


sbatch scripts/train_ret_anet.sh 5m_single_30 anet 2 local pretrained_path=/home/wiss/zhang/Jinhe/singularity/anet_ret/ft_anet_ret_singularity_5m.pth \ batch_size.video=32 \ scheduler.epoch=30

sbatch scripts/train_ret_anet.sh 17m_single_30_temp anet 2 local pretrained_path=/home/wiss/zhang/Jinhe/singularity/anet_ret/ft_anet_ret_singularity_temporal_17m.pth \ video_input.num_frames=4 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2 \ batch_size.video=32 \ scheduler.epoch=30

sbatch scripts/train_ret_anet.sh 5m_single_30_temp anet 2 local pretrained_path=/home/wiss/zhang/Jinhe/singularity/anet_ret/ft_anet_ret_singularity_temporal_5m.pth \ video_input.num_frames=4 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2 \ batch_size.video=32 \ scheduler.epoch=30