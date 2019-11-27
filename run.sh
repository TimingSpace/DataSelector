#!/bin/bash
cd /home/wangxiangwei/Program/DataMaker/
for seq in 09 10
do
    echo $seq
    find /data/datasets/xiangwei/dataset/sequences/$seq/image_2 | sort > kitti_image_$seq.txt
    python3 /home/wangxiangwei/Program/DataMaker/data_maker_1_3.py kitti_image_$seq.txt  $seq
done

