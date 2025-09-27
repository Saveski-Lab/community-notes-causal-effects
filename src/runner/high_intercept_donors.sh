#!/bin/bash
# lower priority below ssh (default 0 → +10)
renice +10 $$ >/dev/null

# cap VM size to 60% RAM
total_mem_kb=$(awk '/MemTotal/ {print $2}' /proc/meminfo)
ulimit -v $(( total_mem_kb*6/10 ))


configs=(
  "controls_above_0.25"
  "controls_above_0.3"
  "controls_above_0.35"
)

for config in "${configs[@]}"; do
  until  python src/pipeline/c_find_controls.py --config src/config/"$config".json; do
    code=$?
    if (( code > 128 )); then
      sig=$(( code - 128 ))
      name=$(kill -l $sig 2>/dev/null || echo "SIG?$sig")
      echo "c_find_controls.py was killed by signal $sig ($name)" >&2
    else
      echo "c_find_controls.py exited with code $code" >&2
    fi
    echo "retrying in 1s…" >&2
    sleep 1
  done


  until Rscript src/pipeline/f_calc_weights.R --config src/config/"$config".json; do
    code=$?
    if (( code > 128 )); then
      sig=$(( code - 128 ))
      name=$(kill -l $sig 2>/dev/null || echo "SIG?$sig")
      echo "f_calc_weights.R was killed by signal $sig ($name)" >&2
    else
      echo "f_calc_weights.R exited with code $code" >&2
    fi
    echo "retrying in 1s…" >&2
    sleep 1
  done


  python src/pipeline/h_treatment_effects.py --config src/config/"$config".json || exit 0
  python src/analysis/gather_cate_data.py --config src/config/"$config".json || true
  python src/analysis/plot_final_figures.py --config src/config/"$config".json || true
done
