#!/bin/bash
cd /home/wangxiangwei/Program/DataMaker/
for seq in 00 01 02 03 04 05 06 07 08 09 10
do
    echo $seq
    python motion_update.py motion_$seq.txt motion_2_$seq.txt
done

