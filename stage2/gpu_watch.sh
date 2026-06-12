#!/bin/bash
# fires (exits 0) when any 4090 (1,2,4,5) drops below 1500MB used
while true; do
  for g in 1 2 4 5; do
    m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g 2>/dev/null | tr -d ' ')
    [ -n "$m" ] && [ "$m" -lt 1500 ] 2>/dev/null && echo "FREE_GPU:$g" && exit 0
  done
  sleep 280
done
