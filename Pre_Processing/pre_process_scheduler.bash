#!/bin/bash

id=Ca
n_tasks=61
#path=/Data_SSD2T/Data/PreProcessed/
#path=/Data2_SSD2T/Data/PreProcessed/
path=/mnt/Data3_SSD2T/Data/Pre_Processed/
n_cores=10

for ((jj=1; jj<=n_tasks; jj++)); do
#for jj in 14; do
  ((i=i%n_cores)); ((i++==0)) && wait
  python pre_process_task_manager.py -t "$jj" -a "$id" -d "$path" &
  echo task "$jj" completed
done
echo ""

