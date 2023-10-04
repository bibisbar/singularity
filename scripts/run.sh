sbatch scripts/train_ret_anet.sh 17m_single_30 anet 2 local pretrained_path=/home/wiss/zhang/Jinhe/singularity/anet_ret/ft_anet_ret_singularity_17m.pth \ batch_size.video=32 \ scheduler.epoch=30



sbatch scripts/train_ret_anet.sh anet_train_1_Seed42 anet 1 local 12345 pretrained_path=/home/wiss/zhang/Jinhe/singularity/anet_ret/ft_anet_ret_singularity_temporal_17m.pth \
video_input.num_frames=4 \
add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2 \ batch_size.video=64 \ scheduler.epoch=30

sbatch scripts/train_ret_anet.sh anet_train_1_Seed42 anet 1 local 12345 pretrained_path=/home/wiss/zhang/Jinhe/singularity/anet_ret/ft_anet_ret_singularity_temporal_17m.pth \
video_input.num_frames=4 \
add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2 \ batch_size.video=64 \ scheduler.epoch=30




sbatch scripts/train_ret_anet.sh anet_train_1_Seed42 anet 2 local 12345 pretrained_path=/home/wiss/zhang/Jinhe/singularity/anet_ret/ft_anet_ret_singularity_temporal_17m.pth \ video_input.num_frames=4 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2 \ batch_size.video=32 \ scheduler.epoch=30 \ train_type=anet_ret_train_1.json \ seed=42

sbatch scripts/train_ret_anet.sh anet_train_2_Seed2 anet 2 local 12346 pretrained_path=/home/wiss/zhang/Jinhe/singularity/anet_ret/ft_anet_ret_singularity_temporal_17m.pth \ video_input.num_frames=4 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2 \ batch_size.video=32 \ scheduler.epoch=30 \ train_type=anet_ret_train_2.json \ seed=2

sbatch scripts/train_ret_anet.sh anet_train_3_Seed3 anet 2 local 12347 pretrained_path=/home/wiss/zhang/Jinhe/singularity/anet_ret/ft_anet_ret_singularity_temporal_17m.pth \ video_input.num_frames=4 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2 \ batch_size.video=32 \ scheduler.epoch=30 \ train_type=anet_ret_train_3.json \ seed=3


sbatch scripts/train_ret_mg.sh moviegraph_train_1_Seed42 moviegraph 2 local 12348 pretrained_path=/home/wiss/zhang/Jinhe/singularity/pt/singularity_temporal_17m.pth \ video_input.num_frames=4 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2 \ batch_size.video=32 \ scheduler.epoch=30 \ train_type=anet_ret_train_1.json \ seed=42

sbatch scripts/train_ret_mg.sh moviegraph_train_2_Seed2 moviegraph 2 local 12349 pretrained_path=/home/wiss/zhang/Jinhe/singularity/pt/singularity_temporal_17m.pth \ video_input.num_frames=4 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2 \ batch_size.video=32 \ scheduler.epoch=30 \ train_type=anet_ret_train_2.json \ seed=2


sbatch scripts/train_ret_mg.sh moviegraph_train_1_Seed3 moviegraph 2 local 12350 pretrained_path=/home/wiss/zhang/Jinhe/singularity/pt/singularity_temporal_17m.pth \ video_input.num_frames=4 \ add_temporal_embed=True \ temporal_vision_encoder.enable=True \ temporal_vision_encoder.num_layers=2 \ batch_size.video=32 \ scheduler.epoch=30 \ train_type=anet_ret_train_1.json \ seed=3