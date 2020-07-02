#!/bin/bash

id='Li'

n_tasks=63
n_cores=10
dat_path=/Data_SSD2T/Data/PreProcessed/

#for ((jj=50; jj<=n_tasks; jj++)); do
for jj in 3; do
   #((i=i%n_cores)); ((i++==0)) && wait -n
   python sort_task_manager.py -t "$jj" -a "$id" -d "$dat_path"
done
