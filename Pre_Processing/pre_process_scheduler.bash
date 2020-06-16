#!/bin/bash

id='Li'
n_tasks=66
n_cores=10

#for ((jj=1; jj<=n_tasks; jj++)); do
for jj in 31 50 66 ; do
   ((i=i%n_cores)); ((i++==0)) && wait -n
   python pre_process_task_manager.py -t "$jj" -a "$id" &
done