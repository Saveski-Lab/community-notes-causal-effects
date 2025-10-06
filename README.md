# Community notes reduce engagement with and diffusion of false information online
Replication code for the paper:

Isaac Slaughter, Axel Peytavin, Johan Ugander, and Martin Saveski (2025) "[Community notes reduce engagement with and diffusion of false information online,](https://www.pnas.org/doi/10.1073/pnas.2503413122)" PNAS.

## Quick Start
1. Download data [here](https://doi.org/10.7910/DVN/K0RQTM) and place it in the `data` directory.
2. Install python and R dependencies listed in `requirements.txt` and `renv.lock`.
3. Run the pipeline using one of the scripts in `src/runner`. For example, for the main analysis in the paper, run `bash src/runner/main.sh`.
4. Run `python src/analysis/plot_final_figures.py` to generate the figures in the paper, or `marimo edit src/analysis/paper_stats.py` to generate the statistics in the paper.

##  Data
Replication data for this project is available for download [here](https://doi.org/10.7910/DVN/K0RQTM).

## Repository structure
Source code for this project is available in the `src` directory. The code is organized as follows:
* `src/pipeline` contains scripts for estimating treatment effects with synthetic controls.
* `src/analysis` contains scripts for summarizing these effects and analyzing heterogeneity within them. 
* `src/config` contains JSON files for running the pipeline under the various configurations we consider in the paper.
* `src/runner` contains shell scripts for running the pipeline.

The following scripts contain code for calculating the statistics and artifacts used in the paper:
* `src/analysis/paper_stats.py`
* `src/analysis/plot_final_figures.py`
* `src/analysis/gather_cate_data.py`


## Notes
* The code was designed to run on single high-memory machine. It took multiple days to run the main pipeline, and will take longer for the runs discussed in the supplementary information.
* Each script in the `pipeline` folder has a parameter for specifying the number of threads/workers/processes to use, which you can modify for your setting. 
* There are some sources of stochasticity in the pipeline—for example there are non-unique solutions to some of the quadratic programs that are solved when constructing synthetic controls. This can lead to small differences in results between runs. (We observed a maximum relative change of 2% when rerunning the pipeline.) Please feel free to reach out to the corresponding author with any questions.

## Citation
If you use this code in your research, please cite the following paper:
```
@article{
    doi:10.1073/pnas.2503413122,
    author = {Isaac Slaughter and Axel Peytavin and Johan Ugander and Martin Saveski},
    title = {Community notes reduce engagement with and diffusion of false information online},
    journal = {Proceedings of the National Academy of Sciences},
    volume = {122},
    number = {38},
    pages = {e2503413122},
    year = {2025},
    doi = {10.1073/pnas.2503413122},
}
```
