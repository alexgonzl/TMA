#!/bin/bash

id=Mi

n_tasks=30
n_cores=2
#dat_path=/Data2_SSD2T/Data/PreProcessed/
dat_path=/mnt/Data3_SSD2T/Data/Pre_Processed/

for ((jj=1; jj<=n_tasks; jj++)); do
#for jj in 3; do
   ((i=i%n_cores)); ((i++==0)) && wait
   python sort_task_manager.py -t "$jj" -a "$id" -d "$dat_path"
done
