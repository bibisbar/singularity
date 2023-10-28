#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH --job-name=sl_qa_full_Answer

# debug info

# can add MASTER_PORT to control port for distributed training
exp_name=$1  # note we added ${corpus} prefix automatically
dataset=$2  # one of [vqa, msrvtt, anet]
exp_dir=${SL_EXP_DIR}
ngpus=$3   # number of GPUs to use
mode=$4  # [local, slurm]
MASTER_PORT=$5  # port for distributed training

export NCCL_P2P_DISABLE=1
if [[ ${dataset} != "vqa" ]] && [[ ${dataset} != "msrvtt" ]] && \
  [[ ${dataset} != "anet" ]]; then
	echo "Does not support dataset ${dataset}"
	exit 1
fi

if [[ ${mode} != "slurm" ]] && [[ ${mode} != "local" ]]; then
	echo "Got mode=${mode}, supported mode: [slurm, local]."
	exit 1
fi

output_dir=/home/wiss/zhang/Jinhe/singularity/qa_anet/${exp_name}
config_path=./configs/qa_anet_full.yaml
echo "output dir >> ${output_dir}"

### save code copy
project_dir=$PWD
if [ -d ${output_dir} ]; then
	echo "Dir ${output_dir} already exist. Exit."
	exit 1
fi

############### ======> Your training scripts [START]
if [[ ${mode} == "slurm" ]]; then
	# slurm job, started with
	# sbatch THIS_SCRIPT ... slurm ...
	master_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
	all_nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
	echo "All nodes used: ${all_nodes}"
	echo "Master node ${master_node}"
	# prepend MASTER_PORT=XXX when launching
	dist_url="tcp://$master_node:${MASTER_PORT:-40000}"  # default port 40000
	echo "dist_url: ${dist_url}"

	echo "PYTHONPATH: ${PYTHONPATH}"
	which_python=$(which python)
	echo "which python ${which_python}"
	export PYTHONPATH=${PYTHONPATH}:${which_python}
	export PYTHONPATH=${PYTHONPATH}:.
	echo "PYTHONPATH: ${PYTHONPATH}"

	srun \
	--output=${output_dir}/slurm%j.out \
	--error=${output_dir}/slurm%j.err \
	python \
	tasks/vqa.py \
  ${config_path} \
  output_dir=${output_dir} \
  wandb.project=sb_qa_${dataset} \
  wandb.enable=True \
  dist_url=${dist_url} \
  ${@:5}
elif [[ ${mode} == "local" ]]; then
  # bash THIS_SCRIPT ... local ...
  rdzv_endpoint="${HOSTNAME}:${MASTER_PORT:-40000}"
  echo "rdzv_endpoint: ${rdzv_endpoint}"

  PYTHONPATH=.:${PYTHONPATH} \
  torchrun --nnodes=1 \
  --nproc_per_node=${ngpus} \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${rdzv_endpoint} \
  tasks/vqa.py \
  ${config_path} \
  output_dir=${output_dir} \
  wandb.project=sb_qa_${dataset} \
  wandb.enable=True \
  ${@:5}
else
	echo "mode expects one of [local, slurm], got ${mode}."
fi
############### ======> Your training scripts [END]

