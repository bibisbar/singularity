#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH --job-name=sl_ret_neg




# debug info
# can add MASTER_PORT to control port for distributed training
exp_name=$1  # note we added ${corpus} prefix automatically
dataset=$2  # coco, flickr, msrvtt, ..., see complete list below
exp_dir=${SL_EXP_DIR}
ngpus=$3   # number of GPUs to use
mode=$4
MASTER_PORT=$5

if [[ ${dataset} != "coco" ]] && [[ ${dataset} != "flickr" ]] && \
  [[ ${dataset} != "msrvtt" ]] && [[ ${dataset} != "didemo" ]] && \
  [[ ${dataset} != "anet" ]] && [[ ${dataset} != "ssv2_label" ]] && \
  [[ ${dataset} != "ssv2_template" ]]; then
  echo "Does not support dataset ${dataset}"
  exit 1
fi

if [[ ${mode} != "slurm" ]] && [[ ${mode} != "local" ]]; then
  echo "Got mode=${mode}, supported mode: [slurm, local]."
  exit 1
fi

output_dir=/home/wiss/zhang/Jinhe/singularity/neg/ret_${dataset}/${dataset}_${exp_name}
config_path=./configs/ret_${dataset}.yaml
echo "output dir >> ${output_dir}"

### save code copy
project_dir=$PWD
if [ -d ${output_dir} ]; then
  echo "Dir ${output_dir} already exist. Exit."
  exit 1
fi
# mkdir -p ${output_dir}
# cd ..
# code_dir=${output_dir}/code
# project_dirname=singularity
# rsync -ar ${project_dirname} ${code_dir} --exclude='*.out'  # --exclude='.git'
# cd ${code_dir}/${project_dirname}
# echo "Copied source files to '${PWD}' and launch from this dir"
export NCCL_P2P_DISABLE=1
############### ======> Your training scripts [START]
if [[ ${mode} == "slurm" ]]; then
  nvidia-smi
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

  python \
  tasks/retrieval.py \
  ${config_path} \
  output_dir=${output_dir} \
  wandb.project=sb_ret_${dataset} \
  wandb.enable=False \
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
  tasks/retrieval.py \
  ${config_path} \
  output_dir=${output_dir} \
  wandb.project=sb_ret_${dataset} \
  wandb.enable=True \
  ${@:5}
else
  echo "mode expects one of [local, slurm], got ${mode}."
fi
############### ======> Your training scripts [END]


