#!/bin/bash
#SBATCH -n 1                    # Number of cores
#SBATCH -N 1                    # Ensure that all cores are on one machine
#SBATCH -t 1-00:00              # Runtime in D-HH:MM
#SBATCH -p gpu                  # Partition to submit to
#SBATCH -w roberto                  # Partition to submit to
#SBATCH --gres=gpu:1            # Number of gpus
#SBATCH -w roberto                  # Partition to submit to
#SBATCH --mem=100               # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o log/hostname_%j.out      # File to which STDOUT will be written
#SBATCH -e log/hostname_%j.err      # File to which STDERR will be written
#SBATCH --mail-type=ALL         # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=xiangwew@cs.cmu.edu # Email to which notifications will be sent


#nvidia-docker run --rm --ipc=host -e CUDA_VISIBLE_DEVICES=`echo $CUDA_VISIBLE_DEVICES` -v /data/datasets:/data/datasets  -v /home/$USER:/home/$USER  wxw/deep_learning:v6 sh /home/wangxiangwei/Program/DarkVO/SFM_DEEPVO/train_cycle.sh -a 4
nvidia-docker run --rm --ipc=host -e CUDA_VISIBLE_DEVICES=2 -v /data/datasets:/data/datasets  -v /home/$USER:/home/$USER  xiangwei/pytorch:cu80 sh /home/wangxiangwei/Program/DataMaker/run_cmu.sh -a 4



