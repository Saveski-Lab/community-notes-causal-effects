# pipeline
Scripts for getting from `cn_effect_input` to `cn_effect_output`.

Files within the `pipeline` directory are ordered alphabetically (e.g., code in 
`a_preprocess.py` should be run before code in `b_merge.py`). These files use data either from the `cn_effect_input`
directory, or from the `cn_effect_intermediate` directory. They will output data either to 
the `cn_effect_intermediate` directory or to the `cn_effect_output` directory.
