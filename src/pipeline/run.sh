#!/bin/bash


python src/pipeline/a_preprocess.py --config src/config/with_root_and_non_root_rts_prod.json
python src/pipeline/b_merge.py --config src/config/with_root_and_non_root_rts_prod.json

configs=(
  "with_root_and_non_root_rts_prod"
  "1_hr_placebo_prod"
)

for config in "${configs[@]}"
do
    python src/pipeline/c_find_controls.py --config src/config/$config.json
    R --slave --no-save --no-restore --no-site-file --no-environ --vanilla -f src/pipeline/f_calc_weights.R --args --config src/config/$config.json
    python src/pipeline/h_treatment_effects.py --config src/config/$config.json
    python src/analysis/plot_treatment_effects.py --config src/config/$config.json
    python src/analysis/plot_tree_size_vs_shape.py --config src/config/$config.json
    python src/analysis/gather_cate_data.py --config src/config/$config.json
    python src/analysis/plot_final_figures.py --config src/config/$config.json
done
