#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --job-name=sl_qa_eval
#SBATCH --output=sl_qa_eval.out
#SBATCH --error=sl_qa_eval.err
# debug info

dataset=$1  # one of [vqa, msrvtt, anet]
pretrained_path=$2  # path to pth file
save_dirname=$3  # under the root dir of pretrained_path
mode=$4  # [local, slurm]
ngpus=$5  # int

output_dir=$(dirname $pretrained_path)/${save_dirname}
config_path=./configs/qa_${dataset}.yaml

if [[ ${dataset} != "vqa" ]] && [[ ${dataset} != "msrvtt" ]] && \
  [[ ${dataset} != "anet" ]]; then
  echo "Does not support dataset ${dataset}"
  exit 1
fi

if [[ ${mode} != "slurm" ]] && [[ ${mode} != "local" ]]; then
  echo "Got mode=${mode}, supported mode: [slurm, local]."
  exit 1
fi

if [ ! -f ${pretrained_path} ]; then
  echo "pretrained_path ${pretrained_path} does not exist. Exit."
  exit 1
fi

### save code copy 
# project_dir=$PWD
# if [ -d ${output_dir} ]; then
#   echo "Dir ${output_dir} already exist. Exit."
#   exit 1
# fi
# mkdir -p ${output_dir}
# cd .. 
# code_dir=${output_dir}/code
# project_dirname=singularity
# rsync -ar ${project_dirname} ${code_dir}  --exclude='*.out'  # --exclude='.git'
# cd ${code_dir}/${project_dirname}
# echo "Copied source files to '${PWD}' and launch from this dir"

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


  python \
  tasks/vqa.py \
  ${config_path} \
  output_dir=${output_dir} \
  pretrained_path=${pretrained_path} \
  evaluate=True \
  dist_url=${dist_url} \
  ${@:6}

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
  pretrained_path=${pretrained_path} \
  evaluate=True \
  ${@:6}
else
  echo "mode expects one of [local, slurm], got ${mode}."
fi
############### ======> Your training scripts [END] 


