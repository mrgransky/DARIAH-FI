#!/bin/bash

#SBATCH --account=project_2004072
#SBATCH --job-name=spMtx_x182_concat
#SBATCH --output=/scratch/project_2004072/Nationalbiblioteket/trash/NLF_logs/%x_%N_%j.out
#SBATCH --mail-user=farid.alijani@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=373G
#SBATCH --partition=gpu
#SBATCH --time=02-23:00:00
#SBATCH --gres=gpu:v100:1

user="`whoami`"
stars=$(printf '%*s' 100 '')
txt="$user began Slurm job: `date`"
ch="#"
echo -e "${txt//?/$ch}\n${txt}\n${txt//?/$ch}"
# echo "${stars// /*}"
echo "CPUS/NODE: $SLURM_JOB_CPUS_PER_NODE, MEM/NODE(--mem): $SLURM_MEM_PER_NODE"
echo "HOST: $SLURM_SUBMIT_HOST @ $SLURM_JOB_ACCOUNT, CLUSTER: $SLURM_CLUSTER_NAME, Partition: $SLURM_JOB_PARTITION"
echo "JOBname: $SLURM_JOB_NAME, ID: $SLURM_JOB_ID, WRK_DIR: $SLURM_SUBMIT_DIR"
echo "nNODES: $SLURM_NNODES, NODELIST: $SLURM_JOB_NODELIST, NODE_ID: $SLURM_NODEID"
echo "nTASKS: $SLURM_NTASKS, TASKS/NODE: $SLURM_TASKS_PER_NODE, nPROCS: $SLURM_NPROCS"
echo "CPUS_ON_NODE: $SLURM_CPUS_ON_NODE, CPUS/TASK: $SLURM_CPUS_PER_TASK, MEM/CPU: $SLURM_MEM_PER_CPU"
echo "nTASKS/CORE: $SLURM_NTASKS_PER_CORE, nTASKS/NODE: $SLURM_NTASKS_PER_NODE"
echo "THREADS/CORE: $SLURM_THREADS_PER_CORE"
echo "${stars// /*}"
echo "<> Using $SLURM_CLUSTER_NAME conda env from tykky module..."
dfsDIR="/scratch/project_2004072/Nationalbiblioteket/dataframes_x182" ########## must be adjusted! ##########

for qu in 'TAMPEREEN TEHDAS' 'Juha Sipilä Sahalahti' 'Tampereen seudun työväenopisto' 'Helsingin pörssi ja suomen pankki' 'torvisoittokunta' 'Tampereen Työväen Teatteri' 'Suomen pankki lainat ja talletukset' 'Global Warming' 'Economical Crisis in Finland' 'Helsingin Kaupunginteatteri' 'Suomalainen Kirjakauppa' 'kantakirjasonni' 'Senaatti-kiinteistöt ja Helsingin kaupunki' 'finska skolor på åland' 'Helsingfors stadsteater' 'Åbo Akademi i Vasa' 'Stockholms universitet' 'Jakobstads svenska församling' 'Ålands kulturhistoriska museum'
# for qu in 'Tampereen seudun työväenopisto' 
do
  echo "Query: $qu"
  python -u concat_dfs.py --dfsPath $dfsDIR --lmMethod 'stanza' --qphrase "$qu"
done

done_txt="$user finished Slurm job: `date`"
echo -e "${done_txt//?/$ch}\n${done_txt}\n${done_txt//?/$ch}"
# echo "${stars// /*}"