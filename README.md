# Community notes moderate how users engage with and diffuse false information online
Replication code for the paper by Isaac Slaughter, Axel Peytavin, Johan Ugander, and Martin Saveski. 


## Repository structure
Python and R scripts developed for this project are available in the `src` directory. The code is organized as follows:
* `src/pipeline` contains scripts for estimating treatment effects with synthetic controls.
* `src/analysis` contains scripts for summarizing these effects and analyzing heterogeneity within them. 

The following scripts contain code for calculating the statistics and artifacts used in the paper:
* `src/analysis/paper_stats.py` and `src/analysis/paper_stats.html`
* `src/analysis/plot_final_figures.py`
* `src/analysis/volatile_tweet_labeling_irr.R`

The following directories contain code for characterizing the posts and notes for usage in heterogeneity analysis:
* `src/analysis/llm_labeling` contains files for labeling posts and notes based on partisanship and other attributes.
* `src/analysis/llm_labeling` contains files for labeling posts and notes based on readability.

Other noteworthy files and locations:
* `src/pipeline/run.sh` contains a bash script for running the pipeline.
* `src/config` contains json files with configuration parameters for the pipeline.
* `src/references` contains links to external repository used in this project.
* `src/analysis/volatile_tweet_labeling_round_*`: Contains potato logs for labeling posts as anomalous.
* `src/analysis/find_volatile_tweet_cutoff.html` contains code for finding the optimal cutoff for identifying anomalous posts.


##  Data
Input data for this project comes from the publicly-released Community Notes datasets as well as from the X API. The authors are not currently redistributing this data.
There are three primary directories used by the project's code: `cn_effect_input`, `cn_effect_intermediate_prod`, `cn_effect_output`.