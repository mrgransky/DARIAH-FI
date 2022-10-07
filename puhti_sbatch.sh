#!/bin/bash

# # # # # # To create the symbolic link:
# # # # # # $ cd /path/2/your_save_proj
# # # # # # $ ln -s ~/WS_Farid/OxfordRobotCar/VPR_IR/puhti_video_sbatch.sh live.sh

#SBATCH --account=project_2004072
#SBATCH -J 1feb2017_err_handling_url
#SBATCH -o LOG_%x_%N_%j.out
#SBATCH --partition=hugemem
#SBATCH --mem=660G
#SBATCH --time=2-23:59:58
#SBATCH --mail-user=farid.alijani@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=10

# # # S B A T C H --array=1-3

stars=$(printf '%*s' 90 '')
txt="SLURM JOB STARTED AT: `date`"
ch="#"
echo -e "${txt//?/$ch}\n${txt}\n${txt//?/$ch}"
echo "${stars// /*}"
echo "HOST: $SLURM_SUBMIT_HOST, CLUSTER: $SLURM_CLUSTER_NAME, WRK_DIR: $SLURM_SUBMIT_DIR"
echo "JOB: $SLURM_JOB_NAME, $SLURM_JOB_ID, $SLURM_ARRAY_JOB_ID"
echo "NUM_NODES: $SLURM_JOB_NUM_NODES, NODELIST: $SLURM_JOB_NODELIST, NODE_ID: $SLURM_NODEID"
echo "NTASKS: $SLURM_NTASKS, PROCID: $SLURM_PROCID"
echo "CPUS_ON_NODE: $SLURM_CPUS_ON_NODE, CPUS/TASK: $SLURM_CPUS_PER_TASK, MEM/CPU: $SLURM_MEM_PER_CPU"
echo "${stars// /*}"

user="`whoami`"
cluster="$SLURM_CLUSTER_NAME"
echo "Cluster: $cluster Current User: $user"

if [ $user == 'alijani' ]; then
	source activate py3_gpu
elif [ $user == 'alijanif' ]; then
	source /projappl/project_2004072/miniconda3/bin/activate py3_gpu
fi

#python nationalbiblioteket.py
python nationalbiblioteket_logs.py

done_txt="SLURM JOB ENDED AT: `date`"
echo -e "${done_txt//?/$ch}\n${done_txt}\n${done_txt//?/$ch}"
echo "${stars// /*}"