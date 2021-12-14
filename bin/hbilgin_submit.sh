#!/bin/bash
#
# CompecTA (c) 2018
#
#
# TODO:
#   - Set name of the job below changing  value.
#   - Set the requested number of nodes (servers) with --nodes parameter.
#   - Set the requested number of tasks (cpu cores) with --ntasks parameter. (Total accross all nodes)
#   - Select the partition (queue) you want to run the job in:
#     - shorter : For jobs that have maximum run time of 120 mins. Has higher priority.
#     - short   : For jobs that have maximum run time of 1 days. Has higher priority.
#     - mid     : For jobs that have maximum run time of 3 days. Lower priority than short.
#     - long    : For jobs that have maximum run time of 7 days. Lower priority than long.
#     - longer  : For testing purposes, queue has 15 days limit but only 2 nodes.
#     - cuda    : For CUDA jobs. Solver that can utilize CUDA acceleration can use this queue. 15 days limit.
#   - Set the required time limit for the job with --time parameter.
#     - Acceptable time formats include "minutes", "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds"
#   - Put this script and all the input file under the same directory.
#   - Set the required parameters, input/output file names below.
#   - If you do not want mail please remove the line that has --mail-type and --mail-user. If you do want to get notification emails, set your email address.
#   - Put this script and all the input file under the same directory.
#   - Submit this file using:
#      sbatch slurm_submit.sh
#
# -= Resources =-
#
#SBATCH --job-name=Keras
#SBATCH --account=users
#SBATCH --partition=cuda
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --output=%j-keras.out
#SBATCH --mem=50gb
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --mail-type=ALL
##SBATCH --mail-user=bilginhalil@gmail.com

INPUT_FILE="$@"

######### DON'T DELETE BELOW THIS LINE ########################################
source /etc/profile.d/zzz_cta.sh
echo "source /etc/profile.d/zzz_cta.sh"
######### DON'T DELETE ABOW THIS LINE #########################################

# MODULES LOAD...
echo "CompecTA Pulsar..."
module load singularity/pulsar

echo ""
echo "============================== ENVIRONMENT VARIABLES ==============================="
env
echo "===================================================================================="
echo ""
echo ""

echo "======================================================================================"
echo "Setting stack size to unlimited..."
ulimit -s unlimited
ulimit -l unlimited
ulimit -a
echo "======================================================================================"

PULSAR=/cta/capps/singularity/pulsar

# For dta_pred
export PATH=$HOME/.local/bin:$PATH
source ~/.bashrc
module load cuda/9.0

conda activate dta_pred
echo "Running Tensorflow command..."
echo "===================================================================================="
# GPU
#singularity exec --nv $PULSAR/hbilgin-2.2.4.py3.simg python $INPUT_FILE
#module load cuda/9.0
nvidia-smi
 /cta/users/hbilgin/.conda/envs/dta_pred/bin/python $INPUT_FILE --mongodb="192.168.12.1:80:DTA_PRED"
RET=$?

echo ""
echo "===================================================================================="
echo "Solver exited with return code: $RET"
exit $RET
