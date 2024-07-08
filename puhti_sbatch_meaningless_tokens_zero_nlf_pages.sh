#!/bin/bash

#SBATCH --account=project_2004072
#SBATCH --job-name=meaningless_tokens_zero_nlf_page_longrun
#SBATCH --output=/scratch/project_2004072/Nationalbiblioteket/trash/NLF_logs/%x_%N_%j_%A.out
#SBATCH --mail-user=farid.alijani@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --partition=longrun
#SBATCH --time=14-00:00:00

user="`whoami`"
stars=$(printf '%*s' 100 '')
txt="$user began Slurm job: `date`"
ch="#"

echo -e "${txt//?/$ch}\n${txt}\n${txt//?/$ch}"
echo "${stars// /*}"
echo "CPUS/NODE: $SLURM_JOB_CPUS_PER_NODE, MEM/NODE(--mem): $SLURM_MEM_PER_NODE"
echo "HOST: $SLURM_SUBMIT_HOST @ $SLURM_JOB_ACCOUNT, Partition: $SLURM_JOB_PARTITION"
echo "JOBname: $SLURM_JOB_NAME, ID: $SLURM_JOB_ID, WRK_DIR: $SLURM_SUBMIT_DIR"
echo "nNODES: $SLURM_NNODES, NODELIST: $SLURM_JOB_NODELIST, NODE_ID: $SLURM_NODEID"
echo "nTASKS: $SLURM_NTASKS, TASKS/NODE: $SLURM_TASKS_PER_NODE, nPROCS: $SLURM_NPROCS"
echo "CPUS_ON_NODE: $SLURM_CPUS_ON_NODE, CPUS/TASK: $SLURM_CPUS_PER_TASK, MEM/CPU: $SLURM_MEM_PER_CPU"
echo "nTASKS/CORE: $SLURM_NTASKS_PER_CORE, nTASKS/NODE: $SLURM_NTASKS_PER_NODE"
echo "THREADS/CORE: $SLURM_THREADS_PER_CORE"
echo "${stars// /*}"

echo "$SLURM_SUBMIT_HOST conda env from tykky module..."
# vocab_fpath="/scratch/project_2004072/Nationalbiblioteket/dataframes_x732/concatinated_732_SPMs_lm_stanza_spMtx_x_8702764_BoWs.json" # big BoWs
vocab_fpath="/scratch/project_2004072/Nationalbiblioteket/dataframes_x732/concatinated_732_SPMs_lm_stanza_shrinked_spMtx_x_6567120_BoWs.json" # small BoWs

python -u parallel_rest_api.py --vbfpath $vocab_fpath

done_txt="$user finished Slurm job: `date`"
echo -e "${done_txt//?/$ch}\n${done_txt}\n${done_txt//?/$ch}"
echo "${stars// /*}"