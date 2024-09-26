#!/bin/bash

# set condition
n=1000
batch_size=20
range_end=$((n-1))

# execute python scripts
counter=0
for IDX in $(seq 0 $range_end); do

  # process counter
  counter=$((counter+1))

  # python call
  (
  echo "Starting job #$((IDX+1)) of ${n} jobs."
  srun --ntasks=1 --nodes=1 --mem=4G --time=4:00:00 --gpus-per-task=1 --job-name="parallel_sampling" \
  --output="out/par_$counter.out" --error="err/par_$counter.err" --partition="gpu" --exclusive -c 1 \
  python synthetic_probs_gen_par.py $"t_4" $"balanced4" "$IDX"
  ) &

  # batch control
  if [[ $(jobs -r -p | wc -l) -ge $batch_size ]]; then
        wait
  fi

done


wait

echo "All jobs finished."

