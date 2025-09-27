import json
import sys
import socket
import shutil
import os
import re
import argparse
import traceback
from pathlib import Path
from copy import deepcopy
from itertools import chain
from functools import reduce

import pandas as pd
import numpy as np
from tqdm import tqdm, trange
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.lines as mlines
from matplotlib import gridspec
from matplotlib import rcParams
from scipy.stats import gmean, norm

matplotlib.use('Agg')  # non-GUI backend
plt.ioff()

# Add current directory to path for imports
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.utils import ConfigError, read_weights, data_root
from src.pipeline.c_find_controls import (
    metric_parents,
    tweet_level_datasets,
    read_trt_and_ctrl,
)
from src.analysis.plotting_utils import (
    plot_individual_tweet,
    treatment_and_control_scatter_binned,
    plot_bins,
    plot_overall,
    SUBFIGURE_LETTER_SIZE,
    MetricFormatter,
    load_tes_for_metrics,
    get_trt_and_control_ids,
    get_metadata,
    colors,
)

# Set the fonts
rcParams.update(
    {
        "font.size": 6,
        "font.family": "sans-serif",
        "legend.fontsize": 6,
        "legend.title_fontsize": 6,
        "axes.labelsize": 7.2,
        "axes.titlesize": 7.2,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
    }
)




def plot_figure_1(
    treatment_tids,
    control_tweets,
    treatment_tweets,
    weights,
    output_dir,
    config,
    te,
):


    # Create new imgs
    for metric in ["impressions", "calculated_retweets"]:
        # Delete the output directory if it already exists

        rt_dir = output_dir / "individual_tweets" / metric
        if rt_dir.exists():
            shutil.rmtree(rt_dir)
        os.makedirs(rt_dir, exist_ok=True)

        for tid in tqdm(
          np.random.choice(treatment_tids[metric], size=1, replace=False).tolist()
        ):
            try:
                plot_individual_tweet(
                    tid,
                    metric,
                    control_tweets,
                    treatment_tweets,
                    weights,
                    output_dir,
                    config,
                    te,
                    save_to_disk=True,
                    weight_filter=0.00001,
                )
            except (ValueError, KeyError) as e:
                print(f"Error processing tweet {tid}: {e}")

def plot_figure_2(te, config, output_dir, diffs, include_observed_plots=True, include_matching_plots=True,
                  metrics_to_plot="engagement", treatment_variable="bias_adjusted_treatment_effect", time_limit=None,
                  highlight_placebo=False):
    if metrics_to_plot == "engagement":
        metrics = ["impressions", "calculated_replies", "likes", "calculated_retweets"]
    elif metrics_to_plot == "structural":
        metrics = ["calculated_retweets", "rt_cascade_width", "rt_cascade_depth","rt_cascade_wiener_index"]
    else:
        raise ValueError("metrics_to_plot must be either 'structural' or 'engagement'")

    assert treatment_variable in [
        "bias_adjusted_treatment_effect",
        "bias_adjusted_growth",
        "bias_adjusted_relative_treatment_effect"
    ]


    # Convert 17.8 cm to inches for figure size (1 inch = 2.54 cm)
    width_in_inches = 17.8 / 2.54
    height_in_inches = width_in_inches / 3 * 2

    # Drop height depending on number of rows
    total_num_rows = include_observed_plots + 1 + include_matching_plots
    height_in_inches = height_in_inches / 3 * total_num_rows

    # Create subplots
    fig, axs = plt.subplots(
        total_num_rows,
        len(metrics),
        figsize=(width_in_inches, height_in_inches),
        sharex=False,
    )

    # If only one row, give axs two dimensionsk
    if total_num_rows == 1:
        axs = axs.reshape(1, -1)

    # Calculate x-axis limits based on configuration
    train_backdate = pd.to_timedelta(config["train_backdate"]).total_seconds() / 3600
    if time_limit:
        xmax = pd.to_timedelta(time_limit).total_seconds() / 3600
        xmin = -1.85
    else:
        xmax = (
                pd.to_timedelta(config["post_break_min_time"]).total_seconds() / 3600
                - train_backdate
        )
        xmin = -train_backdate - 1

    xlims = [xmin,  xmax]

    # Get an ordered list of letters to use for subfigures
    ncols = len(metrics)
    letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]
    letters_to_use = []
    for c in range(ncols):
        for r in range(total_num_rows):
            letters_to_use.append(
                letters[
                    (r * ncols)
                    + c
                    - (1 if (include_matching_plots and r == (total_num_rows - 1)) else 0) # Skip first plot in row 3, since it just has the legend
                ]
            )
    letter_counter = 0
    row_counter = 0


    # Iterate over the metrics and plot the data
    for i, metric in enumerate(metrics):
        data_to_plot = te[metric][te[metric]["bias_adjusted_treatment_effect"].notna()]
        if time_limit:
            data_to_plot = data_to_plot[data_to_plot["note_0_time_since_first_crh"] <= pd.to_timedelta(time_limit)]

        if include_observed_plots:
            # Plot absolute effect
            plot_overall(
                data_to_plot,
                metric,
                output_dir,
                "black",
                ylim=None,
                xlim=xlims,
                save=False,
                y_var="treatment",
                ax=axs[row_counter, i],
                include_y_axis_label=(i == 0),
                use_basic_axis_labels=(i == 0),
                include_ci=False,
            )
            # Plot absolute effect
            plot_overall(
                data_to_plot,
                metric,
                output_dir,
                color=colors["control"],
                ylim=None,
                xlim=xlims,
                save=False,
                y_var="bias_adjusted_control",
                ax=axs[row_counter, i],
                include_y_axis_label=(i == 0),
                use_basic_axis_labels=(i == 0),
                include_ci=False
            )

            axs[row_counter, i].text(
                -0.05,
                1.21,
                letters_to_use[letter_counter],
                transform=axs[row_counter, i].transAxes,
                fontweight="bold",
                va="top",
                ha="left",
                size=SUBFIGURE_LETTER_SIZE,
            )
            letter_counter += 1
            row_counter += 1

        # Plot absolute effect
        plot_overall(
            data_to_plot,
            metric,
            output_dir,
            color=colors["te"],
            ylim=None,
            xlim=xlims,
            save=False,
            y_var=treatment_variable,
            ax=axs[row_counter, i],
            include_y_axis_label=(i == 0),
            use_basic_axis_labels=(i == 0),
            include_matching_window=not highlight_placebo
        )

        axs[row_counter, i].text(
            -0.05,
            1.21,
            letters_to_use[letter_counter],
            transform=axs[row_counter, i].transAxes,
            fontweight="bold",
            va="top",
            ha="left",
            size=SUBFIGURE_LETTER_SIZE,
        )
        letter_counter += 1
        row_counter += 1

        if include_matching_plots:
            if metric == metrics[0]:
                axs[row_counter, i].axis("off")

                sc_proxy = mlines.Line2D(
                    [],
                    [],
                    color=colors["control"],
                    alpha=1,
                    label="Synthetic Control",
                    linestyle="--",
                )
                trt_proxy = mlines.Line2D(
                    [],
                    [],
                    color=colors["treated"],
                    alpha=1,
                    label="Treatment",
                    linestyle="solid",
                )
                ate_proxy = mlines.Line2D(
                    [],
                    [],
                    color=colors["te"],
                    alpha=1,
                    label="Treatment Effect",
                    linestyle="--",
                )
                axs[row_counter, i].legend(
                    handles=[sc_proxy, trt_proxy, ate_proxy],
                    loc="center",
                    frameon=False,
                    labelcolor=colors["legend"]
                )
                letter_counter +=1
            else:
                treatment_and_control_scatter_binned(
                    diffs,
                    metrics[0],
                    metric,
                    output_dir,
                    ax=axs[row_counter, i],
                    offset=True
                )

                axs[row_counter, i].text(
                    -0.05,
                    1.21,
                    letters_to_use[letter_counter],
                    transform=axs[row_counter, i].transAxes,
                    fontweight="bold",
                    va="top",
                    ha="left",
                    size=SUBFIGURE_LETTER_SIZE,
                )
                letter_counter += 1
        row_counter = 0


    if highlight_placebo:
        for ax in axs.flatten():
            ax.axvspan(-2, 0, color="gray", alpha=0.25, zorder=-1000)
            ax.axvspan(-1, 0, color="#F9DCDC", alpha=1, zorder=-1001)

    # Add column titles based on the metric
    for i, metric in enumerate(
        metrics
    ):
        axs[0, i].set_title(
            metric.replace("calculated_", "")
            .replace("rt_cascade_", "")
            .replace("width", "Max Breadth")
            .replace("depth", "Max Depth")
            .replace("wiener_index", "Structural Virality")
            .replace("impressions", "views")
            .replace("tweet", "post")
            .replace("reposts", "cascade_size" if metrics_to_plot == "structural" else "reposts")
            .replace("_", " ")
            .title()
        )

    # Adjust layout and save the figure
    fig.tight_layout(rect=[0, 0.075, 1, 1], pad=1)


    # Add legend below figure if we haven't previously
    if not include_matching_plots:
        sc_proxy = mlines.Line2D(
            [],
            [],
            color=colors["control"],
            alpha=1,
            label="Synthetic Control",
            linestyle="--",
        )
        trt_proxy = mlines.Line2D(
            [],
            [],
            color=colors["treated"],
            alpha=1,
            label="Treatment",
            linestyle="solid",
        )
        ate_proxy = mlines.Line2D(
            [],
            [],
            color=colors["te"],
            alpha=1,
            label="Treatment Effect",
            linestyle="--",
        )
        axs[total_num_rows - 1, 1].legend(
            handles=[sc_proxy, trt_proxy, ate_proxy],
            loc="center",
            bbox_to_anchor=(1.15, -0.6),
            ncol=3,
            frameon=False,
            labelcolor=colors["legend"],
        )


    suffix = "" if include_matching_plots else "_no_row_3"
    suffix += f"_{metrics_to_plot}"

    suffix += "" if treatment_variable == "bias_adjusted_treatment_effect" else f"_{treatment_variable}"

    if highlight_placebo:
        suffix += "_highlight_placebo"

    if time_limit:
        suffix += f"_time_limit_{time_limit}"

    fig.savefig(output_dir / f"figure_2{suffix}.pdf")



def plot_permutations(te, config, output_dir, drop_trt_posts=False):
    shuffled_te_config = deepcopy(config)
    del shuffled_te_config["intermediate_dir"]
    shuffled_te_config["shuffle"] = True
    metrics = config["target_metrics"]
    metrics = sorted(metrics)

    shuffled_dir = "../final_submission_anon_tids_permuted/cn_effect_intermediate_prod"

    shuffled_te = load_tes_for_metrics(metrics, shuffled_te_config, shuffled_dir)


    metrics_to_plot = "engagement"
    metrics = ["impressions", "calculated_replies", "likes", "calculated_retweets"]
    treatment_variable = "bias_adjusted_treatment_effect"

    # Convert 17.8 cm to inches for figure size (1 inch = 2.54 cm)
    width_in_inches = 17.8 / 2.54
    height_in_inches = width_in_inches / 3 * 2

    # Drop height depending on number of rows
    total_num_rows = 1
    height_in_inches = height_in_inches / 3 * total_num_rows

    # Create subplots
    fig, axs = plt.subplots(
        total_num_rows,
        len(metrics),
        figsize=(width_in_inches, height_in_inches),
        sharex=False,
    )

    # If only one row, give axs two dimensionsk
    if total_num_rows == 1:
        axs = axs.reshape(1, -1)

    # Calculate x-axis limits based on configuration
    train_backdate = pd.to_timedelta(config["train_backdate"]).total_seconds() / 3600
    xmax = (
            pd.to_timedelta(config["post_break_min_time"]).total_seconds() / 3600
            - train_backdate
    )
    xmin = -train_backdate - 1

    xlims = [xmin,  xmax]

    # Get an ordered list of letters to use for subfigures
    ncols = len(metrics)
    letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]
    letters_to_use = []
    for c in range(ncols):
        for r in range(total_num_rows):
            letters_to_use.append(
                letters[
                    (r * ncols)
                    + c
                ]
            )


    B = 200
    main_tids = list(reduce(set.union, [set(df["tweet_id"].unique().tolist()) for df in te.values()]))
    shuffled_tids = list(reduce(set.union, [set(df["tweet_id"].unique().tolist()) for df in shuffled_te.values()]))
    if drop_trt_posts:
        shuffled_tids = [id for id in shuffled_tids if id not in main_tids]
    for b in trange(B):
        rng = np.random.default_rng(hash((b, 7911)) % (2 ** 32))
        permuted_tids = rng.choice(shuffled_tids, size=len(main_tids), replace=False)

        for i, metric in enumerate(metrics):
            # Make a DataFrame to explode
            replicated_df = pd.DataFrame({"tweet_id": permuted_tids})
            data_to_plot = replicated_df.merge(
                shuffled_te[metric][shuffled_te[metric]["bias_adjusted_treatment_effect"].notna()],
                on="tweet_id",
                how="left"
            )

            # Plot absolute effect
            plot_overall(
                data_to_plot,
                metric,
                output_dir,
                color="gray",
                alpha = 0.15,
                ylim=None,
                xlim=xlims,
                save=False,
                y_var=treatment_variable,
                ax=axs[0, i],
                include_y_axis_label=(i == 0),
                use_basic_axis_labels=(i == 0),
                include_matching_window=True,
                include_ci=False,
                linestyle="-",
                linewidth=0.2,
            )

    letter_counter = 0
    row_counter = 0
    # Iterate over the metrics and plot the data
    for i, metric in enumerate(metrics):
        data_to_plot = shuffled_te[metric][
            shuffled_te[metric]["bias_adjusted_treatment_effect"].notna()
            & shuffled_te[metric]["tweet_id"].isin(te[metric]["tweet_id"])
        ]



        # Plot absolute effect
        plot_overall(
            data_to_plot,
            metric,
            output_dir,
            color=colors["te"],
            ylim=None,
            xlim=xlims,
            save=False,
            y_var=treatment_variable,
            ax=axs[row_counter, i],
            include_y_axis_label=(i == 0),
            use_basic_axis_labels=(i == 0),
            include_matching_window=True,
            include_ci=False
        )

        axs[row_counter, i].text(
            -0.05,
            1.21,
            letters_to_use[letter_counter],
            transform=axs[row_counter, i].transAxes,
            fontweight="bold",
            va="top",
            ha="left",
            size=SUBFIGURE_LETTER_SIZE,
        )
        letter_counter += 1
        row_counter = 0

    # Add column titles based on the metric
    for i, metric in enumerate(
        metrics
    ):
        axs[0, i].set_title(
            metric.replace("calculated_", "")
            .replace("rt_cascade_", "")
            .replace("width", "Max Breadth")
            .replace("depth", "Max Depth")
            .replace("wiener_index", "Structural Virality")
            .replace("num_nodes_root_tweet", "Root Reposts")
            .replace("num_nodes_non_root_tweet", "Deep Reposts")
            .replace("impressions", "views")
            .replace("tweet", "post")
            .replace("reposts", "cascade_size" if metrics_to_plot == "structural" else "reposts")
            .replace("_", " ")
            .title()
        )

    # Adjust layout and save the figure
    fig.tight_layout(rect=[0, 0.075, 1, 1], pad=1)


    # Add legend below figure
    ate_proxy = mlines.Line2D(
        [],
        [],
        color=colors["te"],
        alpha=1,
        label="Treatment Effect",
        linestyle="--",
    )
    replicate_proxy = mlines.Line2D(
        [],
        [],
        color="gray",
        alpha=0.5,
        label="Shuffled Treatment Effect",
        linestyle="-",
    )
    axs[total_num_rows - 1, 1].legend(
        handles=[ate_proxy, replicate_proxy],
        loc="center",
        bbox_to_anchor=(1.15, -0.6),
        ncol=3,
        frameon=False,
        labelcolor=colors["legend"],
    )

    save_nm = "permuted_dropped_trt.pdf" if drop_trt_posts else "permuted.pdf"

    fig.savefig(output_dir / save_nm)


def plot_multiple_configs(te, config, output_dir, alternate_config_names):
    metrics_to_plot = "engagement"
    metrics = ["impressions", "calculated_replies", "likes", "calculated_retweets"]
    treatment_variable = "bias_adjusted_treatment_effect"

    _config_dirs = {
        "controls_above_0.25": "../final_submission_anon_tids_limited/cn_effect_intermediate_prod",
        "controls_above_0.3": "../final_submission_anon_tids_limited/cn_effect_intermediate_prod",
        "controls_above_0.35": "../final_submission_anon_tids_limited/cn_effect_intermediate_prod",
        "main_embd_20": "../final_submission_anon_tids_embd/cn_effect_intermediate_prod",
        "main_embd_58": "../final_submission_anon_tids_embd_58/cn_effect_intermediate_prod",
    }
    alt_te  = {}
    for cfg_nm in alternate_config_names:
        alt_config = deepcopy(config)
        del alt_config["intermediate_dir"]
        if "controls_above_" in cfg_nm:
            threshold = "0." + cfg_nm.split(".")[1]
            alt_config["control_list"] = f"src/config/posts_above_{threshold}.csv"
        elif "embd" in cfg_nm:
            with open(f"src/config/{cfg_nm}.json", "r") as f:
                alt_config["matching_metrics"] = json.load(f)["matching_metrics"]
        alt_te[cfg_nm] = load_tes_for_metrics(metrics, alt_config, _config_dirs[cfg_nm])


    # Convert 17.8 cm to inches for figure size (1 inch = 2.54 cm)
    width_in_inches = 17.8 / 2.54
    height_in_inches = width_in_inches / 3 * 2

    # Drop height depending on number of rows
    total_num_rows = 1
    height_in_inches = height_in_inches / 3 * total_num_rows

    # Create subplots
    fig, axs = plt.subplots(
        total_num_rows,
        len(metrics),
        figsize=(width_in_inches, height_in_inches),
        sharex=False,
    )

    # If only one row, give axs two dimensionsk
    if total_num_rows == 1:
        axs = axs.reshape(1, -1)

    # Calculate x-axis limits based on configuration
    train_backdate = pd.to_timedelta(config["train_backdate"]).total_seconds() / 3600
    xmax = (
            pd.to_timedelta(config["post_break_min_time"]).total_seconds() / 3600
            - train_backdate
    )
    xmin = -train_backdate - 1

    xlims = [xmin,  xmax]

    # Get an ordered list of letters to use for subfigures
    letters_to_use = ["A", "B", "C", "D"]

    handles = []

    linestyles = ["dotted", "dashed", "dashdot"]
    for j, (cfg_nm, cfg_te) in enumerate(alt_te.items()):
        # Count duplicates

        alpha = 1 - 0.15 * j

        handles.append(
            mlines.Line2D(
                [],
                [],
                color="gray",
                alpha=alpha,
                label=(
                   f"Note Score >  0." + cfg_nm.split(".")[1]
                    if "controls_above_" in cfg_nm
                    else f"+{cfg_nm[-2:]}-Dimensional Text Embeddings"
                ),
                linestyle=linestyles[j],
            )
        )

        for i, metric in enumerate(metrics):
            # Make a DataFrame to explode
            data_to_plot = cfg_te[metric][cfg_te[metric]["bias_adjusted_treatment_effect"].notna()]
            print(cfg_nm, cfg_te, len(data_to_plot))

            # Plot absolute effect
            plot_overall(
                data_to_plot,
                metric,
                output_dir,
                color="gray",
                alpha = alpha,
                ylim=None,
                xlim=xlims,
                save=False,
                y_var=treatment_variable,
                ax=axs[0, i],
                include_y_axis_label=(i == 0),
                use_basic_axis_labels=(i == 0),
                include_matching_window=True,
                include_ci=False,
                linestyle=linestyles[j]
            )

    letter_counter = 0
    row_counter = 0
    # Iterate over the metrics and plot the data
    for i, metric in enumerate(metrics):
        data_to_plot = te[metric][te[metric]["bias_adjusted_treatment_effect"].notna()]

        # Plot absolute effect
        plot_overall(
            data_to_plot,
            metric,
            output_dir,
            color=colors["te"],
            ylim=None,
            xlim=xlims,
            save=False,
            y_var=treatment_variable,
            ax=axs[row_counter, i],
            include_y_axis_label=(i == 0),
            use_basic_axis_labels=(i == 0),
            include_matching_window=True,
            include_ci=True,
            linestyle="solid"
        )

        axs[row_counter, i].text(
            -0.05,
            1.21,
            letters_to_use[letter_counter],
            transform=axs[row_counter, i].transAxes,
            fontweight="bold",
            va="top",
            ha="left",
            size=SUBFIGURE_LETTER_SIZE,
        )
        letter_counter += 1
        row_counter = 0

    # Add column titles based on the metric
    for i, metric in enumerate(
        metrics
    ):
        axs[0, i].set_title(
            metric.replace("calculated_", "")
            .replace("rt_cascade_", "")
            .replace("width", "Max Breadth")
            .replace("depth", "Max Depth")
            .replace("wiener_index", "Structural Virality")
            .replace("num_nodes_root_tweet", "Root Reposts")
            .replace("num_nodes_non_root_tweet", "Deep Reposts")
            .replace("impressions", "views")
            .replace("tweet", "post")
            .replace("reposts", "cascade_size" if metrics_to_plot == "structural" else "reposts")
            .replace("_", " ")
            .title()
        )
    # Add legend below figure
    handles.append(
        mlines.Line2D(
            [],
            [],
            color=colors["te"],
            alpha=1,
            label=(
                "Unchanged"
            ),
            linestyle="solid",
        )
    )

    # Adjust layout and save the figure
    fig.tight_layout(rect=[0, 0.075, 1, 1], pad=1)

    axs[total_num_rows - 1, 1].legend(
        title=(
            "Requirement for Inclusion in Donor Pool"
            if "controls_above_" in cfg_nm
            else "Metrics Used to Construct Synthetic Controls"
        ),
        handles=handles[::-1],
        loc="center",
        bbox_to_anchor=(1.15, -0.6),
        ncol=4,
        frameon=False,
        labelcolor=colors["legend"],
    )

    save_name = (
                "restricted_donors.pdf"
                if "controls_above_" in cfg_nm
                else "text_matched.pdf"
    )
    fig.savefig(output_dir /  save_name)


def plot_figure_4(
        te_with_metadata, config, output_dir, y_var="bias_adjusted_treatment_effect", include_a=True,
        center_at_50=False, metric="calculated_retweets", only_plot_top_bins_in_B=False
):
    # Convert 17.8 cm to inches for figure size (1 inch = 2.54 cm)
    width_in_inches = 17.8 / 2.54
    height_in_inches = width_in_inches / 9 * 5 / 2

    # Create figure and subplots
    fig = plt.figure(figsize=(width_in_inches, height_in_inches))
    # Create a main GridSpec with 2 rows, 1 column
    main_gs = gridspec.GridSpec(1, 1)

    # Create a sub-GridSpec for the top row only, with extra space for legends
    top_gs = gridspec.GridSpecFromSubplotSpec(1, 3 if include_a else 1, subplot_spec=main_gs[0], width_ratios=(1, 0.1, 3) if include_a else (1,))

    # Create axes
    if include_a:
        ax1 = fig.add_subplot(top_gs[0])
        ax2 = fig.add_subplot(top_gs[2])
    else:
        ax2 = fig.add_subplot(top_gs[0])


    te_with_metadata[metric]["Increase"] = (te_with_metadata[metric]["bias_adjusted_treatment_effect"] > 0).astype(float) * 100

    train_backdate = pd.to_timedelta(config["train_backdate"]).total_seconds() / 3600
    post_break_time = (
            pd.to_timedelta(config["post_break_min_time"]).total_seconds() / 3600
    )

    xlims = [-train_backdate - 1, post_break_time - train_backdate]
    if include_a:
        plot_bins(
            te_with_metadata[metric][
                te_with_metadata[metric][
                    y_var
                ].notna()
            ],
            metric,
            output_dir,
            [colors["bin_te_3"], colors["bin_te_3.3"], colors["bin_te_3.6"], colors["bin_te_4"]][::-1],
            [(1, 1), (3, 1, 1, 1), (4, 1), (6, 0.5)],
            ylim=None,
            xlim=xlims,
            save=False,
            binning_variable="hours_to_slap_bin",
            y_var=y_var,
            ax=ax1,
            reverse_legend=False
        )


    # Get data after 48h
    data = te_with_metadata[metric].copy()
    data = data[data["note_0_time_since_first_crh"] == pd.to_timedelta("48h")]


    # Find unique values of the binning variables for us to iterate over
    time_bins = data[data["hours_to_slap_bin"].notna()]["hours_to_slap_bin"].unique()
    bins_are_formatted = isinstance(time_bins[0], str) and any(
        ["–" in b or ">" in b or "_" in b for b in time_bins]
    )
    if bins_are_formatted:
        def sort_key(b):
            first_num = float(re.sub(r"([–_].*|[>,<≤])", "", b))
            starts_with_lt = b.strip().startswith(("≤", "<"))
            return (first_num, 0 if starts_with_lt else 1)
        time_bins = sorted(time_bins, key=sort_key)
    else:
        time_bins = sorted(time_bins)

    size_bins = data[data[f"pre_break_{metric}_bin"].notna()][f"pre_break_{metric}_bin"].unique()
    bins_are_formatted = isinstance(size_bins[0], str) and any(
        [b.startswith(">") or b.startswith("≤") for b in size_bins]
    )
    if bins_are_formatted:
        def sort_key(b):
            first_num = float(re.sub(r"([–_].*|[>,<≤])", "", b))
            starts_with_lt = b.strip().startswith(("≤", "<"))
            return (first_num, 0 if starts_with_lt else 1)
        size_bins = sorted(size_bins, key=sort_key)
    else:
        size_bins = sorted(size_bins)


    def get_ci(data, col, tweet_or_note, col_name_for_printing, order):
        data = data[data[y_var].notna() & data[col].notna() & (data[col].str.lower() != "nan")].copy()

        points_for_col = []

        # Calculate bin means/counts
        bin_means = data.groupby(col)[y_var].mean()
        ns = data.groupby(col).size()

        # Calculate standard error
        if y_var == "Increase":
            # Convert to proportions
            prop_means = bin_means / 100

            # Get SE for wald
            ses = np.sqrt(prop_means * (1 - prop_means) / ns)

            # Confidence intervals (rescale to 0-100)
            bin_lower_bounds = (prop_means - norm.ppf(0.975) * ses) * 100
            bin_upper_bounds = (prop_means + norm.ppf(0.975) * ses) * 100

        else:
            stds = data.groupby(col)[y_var].std()
            ses = stds / np.sqrt(ns)

            bin_lower_bounds = bin_means - norm.ppf(0.975) * ses
            bin_upper_bounds = bin_means + norm.ppf(0.975) * ses

        bin_names = bin_means.index

        for mean, lower, upper, value in zip(
                bin_means, bin_lower_bounds, bin_upper_bounds, bin_names
        ):
            points_for_col.append(
                (
                    mean,
                    lower,
                    upper,
                    tweet_or_note,
                    col_name_for_printing,
                    value.replace("_", " ").title(),
                )
            )

        # Sort to correct order
        points_for_col = sorted(points_for_col, key=lambda x: order.index(x[5]))

        return points_for_col

    points_to_plot = []

    # Get means/CI for each value of patisan_lean
    data["partisan_lean"] = np.where(
        data["partisan_lean"] == "none", "Non-Political", data["partisan_lean"].str.title()
    )
    data["partisan_lean"] = np.where(
        data["partisan_lean"] == "Unknown", "Ambiguous", data["partisan_lean"].str.title()
    )

    if not include_a:
        points_to_plot += get_ci(data,
                                 "hours_to_slap_bin", "Tweet", r"Post" "\nAge When Note\nAttached (Hours)",
                                 order=time_bins)

        points_to_plot += get_ci(data,
                                 f"pre_break_{metric}_bin", "Tweet", r"Post" "\nReposts\nPrior to Note",
                                 order=size_bins)

    if only_plot_top_bins_in_B:
        # Filter to only tweets in the top two bins for each variable
        time_bins = time_bins[:2]
        size_bins = size_bins[-2:]
        data = data[data["hours_to_slap_bin"].isin(time_bins)]
        data = data[data[f"pre_break_{metric}_bin"].isin(size_bins)]

    points_to_plot += get_ci(data,
                             "partisan_lean", "Tweet", r"Post" "\nPartisan Lean",
                             order=["Left", "Right", "Center", "Ambiguous", "Non-Political"])


    # Get means/CI for each value of tweet_media
    data["tweet_media"] = data["tweet_media"].replace(
        {
            "Multiple Photos or Videos": "Multi Image/Video",
            "Single Photo": "One Image",
            "Single Video": "One Video",
        }
    )
    points_to_plot += get_ci(
        data,
        "tweet_media",
        "Tweet",
        r"Post" "\nMedia Type",
        order=["Text Only", "One Image", "One Video", "Multi Image/Video"],
    )

    # Get means/CI for each value of misleading_type
    # Identify columns
    data = data.rename(columns = {
        "tweet_rated_misleading_missing_important_context": "tweet_rated_misleading_missing_context",
        "tweet_rated_misleading_unverified_claim_as_fact": "tweet_rated_misleading_unverified_claim",
        "tweet_rated_misleading_outdated_information": "tweet_rated_misleading_outdated_info",
        "tweet_rated_misleading_manipulated_media": "tweet_rated_misleading_altered_media"
    })
    misleading_cols = [
        "tweet_rated_misleading_factual_error",
        "tweet_rated_misleading_missing_context",
        "tweet_rated_misleading_unverified_claim",
        "tweet_rated_misleading_outdated_info",
        "tweet_rated_misleading_satire",
        "tweet_rated_misleading_altered_media",
    ]
    for c in misleading_cols:
        if "other" in c:
            continue
        mean = data[data[c] > 0.5][y_var].mean()
        sd = data[data[c] > 0.5][y_var].std()
        n = data[data[c] > 0.5].shape[0]
        se = sd / np.sqrt(n)
        lower = mean - se * norm.ppf(0.975)
        upper = mean + se * norm.ppf(0.975)
        points_to_plot.append(
            (
                mean,
                lower,
                upper,
                "Tweet",
                r"Post" "\nAccuracy Concern",
                c.replace("tweet_rated_misleading_", "").replace("satire", "misleading_satire").replace("_", " ").title(),
            )
        )

    data["note_text_sentence_count_bin"] = np.where(
        data["note_text_sentence_count"] == 1,  "1",
    np.where(
        data["note_text_sentence_count"] > 3, "> 3",
    np.where(
        data["note_text_sentence_count"].notna(),
        data["note_text_sentence_count"].fillna(-1).astype(int).astype(str),
        np.nan
    )))


    points_to_plot += get_ci(data,"note_text_flesch_kincaid_grade_bin", "Note", r"Note" "\nGrade Level", order=[ "≤ 5", "5–8", "8–10", "> 10"])

    points_to_plot += get_ci(data,"note_text_sentence_count_bin", "Note", r"Note" "\nSentence Count", order=["1", "2", "3", "> 3"])

    # Extract the data
    categories = [x[3] for x in points_to_plot]
    subcategories = [x[4] for x in points_to_plot]
    values = [x[5] for x in points_to_plot]

    # Create a unique list of categories and subcategories while maintaining order
    unique_categories = list(dict.fromkeys(categories))
    unique_subcategories = list(dict.fromkeys(subcategories))

    # Create a mapping for x positions
    x_pos = {}
    current_x = 0
    prev_category = None
    prev_subcategory = None
    # Plot the points
    for i, (mean, lower, upper, category, subcategory, label) in enumerate(
            points_to_plot
    ):
        current_x += 1
        x_pos[label] = current_x
        if prev_subcategory != subcategory:
            prev_subcategory = subcategory
            x_pos[subcategory] = current_x
        if prev_category != category:
            prev_category = category
            x_pos[category] = current_x

        _CUTOFF = 50 if y_var in ["Increase"] else 0
        is_significant = lower > _CUTOFF or upper < _CUTOFF

        x = x_pos[label]
        ax2.errorbar(
            x,
            mean,
            yerr=([mean - lower], [upper - mean]),
            color=colors["bin_te_2.5"] if is_significant else "darkgray",
            markersize=4,
            marker="o",
            linestyle="",
            mew=0.15,
            mec="white",
        )

    # Add light shading for every other label
    for i, label in enumerate(values):
        ymin=ax2.get_ylim()[0]
        ymax=ax2.get_ylim()[1]
        if i % 2 == 0:
            ax2.axvspan(i + 0.5, i + 1.5,
                        color="lightgray", alpha=0.2)

    # Add text labels for categories
    minor_ticks = []
    for category in unique_categories[1:]:
        x = x_pos[category] - 0.5
        minor_ticks += [x]
        ax2.axvline(x=x, color="black", linestyle="-", linewidth=1)

    # Add lines between subcategories
    for subcategory in unique_subcategories[1:]:
        x = x_pos[subcategory] - 0.5
        minor_ticks += [x]
        ax2.axvline(x=x, color="gray", linestyle="-", linewidth=1, alpha=0.5)

    plt.axhline(0, c="gray", alpha=0.4, zorder=-1000, linewidth=1)


    # Set x-axis labels
    ax2.set_xticks([x + 1 for x, _ in enumerate(values)], minor=False)
    ax2.set_xticks(minor_ticks, minor=True)
    ax2.set_xticklabels(values, rotation=45, ha="right")

    # Make minor ticks longer/wider
    ax2.tick_params(axis='x', which='minor', length=8, width=1, color=(0.5, 0.5, 0.5, 0.5))

    plt.draw()
    minor_tick_lines = ax2.xaxis.get_minorticklines()
    minor_tick_lines[2].set_color('red')

    # Set formatter for ax2
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(MetricFormatter("%" if y_var in ["Increase"] else "")))

    # Set xlim
    ax2.set_xlim(0.5, len(values) + 0.5)

    if center_at_50:
        y_min = 0
        y_max = 68
    else:
        if include_a:
            y_min_1, y_max_1 = ax1.get_ylim()
        else:
            y_min_1, y_max_1 = np.inf, -np.inf
        y_min_2, y_max_2 = ax2.get_ylim()
        y_min = min(y_min_1, y_min_2)
        y_max = max(y_max_1, y_max_2)

    if include_a:
        ax1.set_ylim(y_min, y_max)
    ax2.set_ylim(y_min, y_max)
    yrange = y_max - y_min

    # Add text labels for subcategories
    for subcategory in unique_subcategories:
        ax2.text(
            x_pos[subcategory] - 0.4,
            y_max + yrange * 0.05,
            subcategory,
            fontsize= 6.5,
            va="baseline"
        )

    if center_at_50:
        plt.axhline(50, c="lightgray", alpha=1, zorder=-1000, linewidth=1)

    # Set axis labels
    if y_var == "bias_adjusted_treatment_effect":
        ax2.set_ylabel("Avg. Treatment\nEffect After 48 Hours" + ("\nfor High Impact Notes" if only_plot_top_bins_in_B else ""))
    else:
        ax2.set_ylabel("Pct. Of Posts w/\nPositive Treatment Effect")

    if include_a:
        axs = [ax1, ax2]
        for idx, label in enumerate(["A", "B"]):
            axs[idx].text(
                -0.01 if idx == 2 else -0.05,
                1.29,
                label,
                transform=axs[idx].transAxes,
                fontweight="bold",
                va="top",
                ha="left",
                size=12,
            )


    fig.tight_layout()

    fig.savefig(output_dir / f"figure_4{'_high_impact' if only_plot_top_bins_in_B else '_all_posts'}_{y_var}_{metric}.pdf")


def plot_all_metrics_by_pre_treatment_visibility(te_with_metadata, config, output_dir, y_var="bias_adjusted_treatment_effect"):
    # Convert 17.8 cm to inches for figure size (1 inch = 2.54 cm)
    width_in_inches = 17.8 / 2.54
    height_in_inches = width_in_inches / 9 * 5 / 2

    # Create figure and subplots
    fig = plt.figure(figsize=(width_in_inches, height_in_inches))
    # Create a main GridSpec with 2 rows, 1 column
    main_gs = gridspec.GridSpec(1, 1)

    # Create a sub-GridSpec for the top row only, with extra space for legends
    _NUM_PLOTS = 4
    top_gs = gridspec.GridSpecFromSubplotSpec(1, _NUM_PLOTS * 2, subplot_spec=main_gs[0],
                                              width_ratios = [1, 0.1] * _NUM_PLOTS)

    # Create axes
    axs = [fig.add_subplot(top_gs[i * 2]) for i in range(_NUM_PLOTS)]


    for metric in te_with_metadata.keys():
        te_with_metadata[metric]["Increase"] = (te_with_metadata[metric]["bias_adjusted_treatment_effect"] > 0).astype(float) * 100


    train_backdate = pd.to_timedelta(config["train_backdate"]).total_seconds() / 3600
    post_break_time = (
        pd.to_timedelta(config["post_break_min_time"]).total_seconds() / 3600
    )
    xlims = [-train_backdate - 1, post_break_time - train_backdate]

    for i, metric in enumerate(
        [
            "impressions",
            "calculated_replies",
            "likes",
            "calculated_retweets",
        ]
    ):

        plot_bins(
            te_with_metadata[metric][
                te_with_metadata[metric][
                    y_var
                ].notna()
            ],
            metric,
            output_dir,
            [colors["bin_te_3"], colors["bin_te_3.3"], colors["bin_te_3.6"], colors["bin_te_4"]][::-1],
            [(1, 1), (3, 1, 1, 1), (4, 1), (6, 0.5)],
            ylim=None,
            xlim=xlims,
            save=False,
            binning_variable="hours_to_slap_bin",
            y_var=y_var,
            ax=axs[i],
            reverse_legend=True,
            include_y_axis_label=(i == 0),
            include_legend=(i == 1),
        )


    fig.tight_layout()

    fig.savefig(output_dir / f"figure_s5_{y_var}.pdf")

if __name__ == "__main__":
    # Read config file
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to config file")
    args = parser.parse_args()
    config_path = args.config
    with open(config_path, "r") as fin:
        config = json.loads(fin.read())

    # Make sure config has everything we need
    necessary_config = [
        "time_freq",
        "dev",
        "volatile_tweet_filtering",
        "train_backdate",
        "pre_break_min_time",
        "pre_break_max_time",
        "post_break_min_time",
        "matching_metrics",
        "replace_calculated_when_missing",
        "missing_metric_action",
        "include_bias_correction",
        "pre_break_bias_correction_time",
        "bias_correction_model",
        "bias_correction_missing_actions",
        "target_metrics",
        "restrict_donor_pool",
        "restricted_pool_size",
        "sample_bias_correction_controls",
        "lambda",
        "shuffle",
    ]
    optional_config = [
        "control_list",
    ]
    for c in necessary_config:
        if c not in config.keys():
            raise ConfigError(
                f"Please specify config value '{c}' in config file '{config_path}.'"
            )
    for c in optional_config:
        if c not in config.keys():
            config[c] = None

    # Drop unneeded config values
    config = {c: config[c] for c in chain(necessary_config, optional_config) if c in config.keys()}

    # Convert lambda from list
    if len(config["lambda"]) == 1:
        config["lambda"] = config["lambda"][0]
    else:
        raise ConfigError("Unsure how to handle multiple lambda values in same config.")

    # Fill in the dev value based on whether this is a local run
    if config["dev"] == "DEVICE_DEFAULT":
        config["dev"] = socket.gethostname() == "is-is28m16x"

    config["tweet_metrics"] = [
        m
        for m in config["matching_metrics"]
        if metric_parents[m] in tweet_level_datasets
    ]
    if config["replace_calculated_when_missing"]:
        config["backup_tweet_metrics"] = [
            m.replace("calculated_", "")
            for m in config["matching_metrics"]
            if "calculated_" in m
        ]
    else:
        config["backup_tweet_metrics"] = []

    config["author_metrics"] = [
        m for m in config["matching_metrics"] if metric_parents[m] == "author"
    ]
    if len(config["tweet_metrics"]) + len(config["author_metrics"]) != len(
        config["matching_metrics"]
    ):
        unknown_metrics = [
            m for m in config["metrics"] if m not in (metric_parents.keys())
        ]
        raise ConfigError(
            f"Unknown metric(s) in config file: {unknown_metrics}. Please define these metrics as "
            f"either tweet or author metrics."
        )

    config["intermediate_dir"] = "cn_effect_intermediate" + (
        "_dev" if config["dev"] else "_prod"
    )
    output_dir = (
        data_root
        / "cn_effect_output"
        / "paper_figures"
        / Path(config_path).name.replace(".json", "")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    weights = read_weights(config["intermediate_dir"], config)

    te_config = deepcopy(config)
    del te_config["intermediate_dir"]
    metrics = config["target_metrics"]
    metrics = sorted(metrics)
    te = load_tes_for_metrics(metrics, te_config, config["intermediate_dir"])


    # Get trt and control tweets
    trt_and_ctrl_config = deepcopy(config)
    trt_and_ctr_config_keys = [
        "time_freq",
        "dev",
        "volatile_tweet_filtering",
        "train_backdate",
        "pre_break_min_time",
        "backup_tweet_metrics",
        "tweet_metrics",
        "author_metrics",
        "pre_break_max_time",
        "post_break_min_time",
        "matching_metrics",
        "replace_calculated_when_missing",
        "missing_metric_action",
        "shuffle"
    ]
    for c in list(trt_and_ctrl_config.keys()):
        if c not in trt_and_ctr_config_keys:
            del trt_and_ctrl_config[c]
    treatment_tids, control_tids = get_trt_and_control_ids(metrics, te, weights)
    control_tweets, treatment_tweets = read_trt_and_ctrl(
        config["intermediate_dir"], trt_and_ctrl_config, log_name=None,
    )


    tweet_metadata = get_metadata()

    te_with_metadata = {
        metric: pd.merge(te[metric], tweet_metadata, on="tweet_id", how="left")
        for metric in metrics
    }

    # Calculate differences
    firsts = {
        metric: te_with_metadata[metric][
            te_with_metadata[metric]["note_0_time_since_first_crh"]
            == -pd.to_timedelta(config["train_backdate"])
        ][
            [
                "tweet_id",
                "hours_to_slap_bin",
                "control",
                "treatment",
                "unadjusted_treatment_effect",
                "bias_adjusted_treatment_effect",
                "bias_adjusted_control"
            ]
        ]
        .set_index(["tweet_id", "hours_to_slap_bin"])
        .sort_index()
        for metric in config["target_metrics"]
    }
    firsts = pd.concat(firsts, axis=1)
    firsts.columns = (
        firsts.columns.get_level_values(1) + "_" + firsts.columns.get_level_values(0)
    )

    lasts = {
        metric: te_with_metadata[metric][
            te_with_metadata[metric]["note_0_time_since_first_crh"]
            == (
                pd.to_timedelta(config["post_break_min_time"])
                - pd.to_timedelta(config["train_backdate"])
                - pd.to_timedelta(config["time_freq"])
            )
        ][
            [
                "tweet_id",
                "hours_to_slap_bin",
                "control",
                "treatment",
                "unadjusted_treatment_effect",
                "bias_adjusted_treatment_effect",
                "bias_adjusted_control"
            ]
        ]
        .set_index(["tweet_id", "hours_to_slap_bin"])
        .sort_index()
        for metric in config["target_metrics"]
    }
    lasts = pd.concat(lasts, axis=1)
    lasts.columns = (
        lasts.columns.get_level_values(1) + "_" + lasts.columns.get_level_values(0)
    )

    diffs = lasts - firsts

    firsts = firsts.reset_index()
    lasts = lasts.reset_index()
    diffs = diffs.reset_index()

    for metric in ["calculated_retweets", "calculated_replies", "impressions", "likes"]:

        for trt_or_ctrl in ["treatment", "control"]:

            diffs[f"{trt_or_ctrl}_{metric}"] = np.where(
                diffs[f"{trt_or_ctrl}_{metric}"] >= 0,
                diffs[f"{trt_or_ctrl}_{metric}"],
                np.nan,
            )

        pooled = pd.concat([diffs[f"treatment_{metric}"], diffs[f"bias_adjusted_control_{metric}"]])

        logged_min = np.log10(
            min(
                diffs[f"treatment_{metric}"]
                .abs()[
                    (diffs[f"treatment_{metric}"] != 0)
                    & diffs[f"treatment_{metric}"].notna()
                    & (~np.isinf(diffs[f"treatment_{metric}"]))
                ]
                .min(),
                diffs[f"bias_adjusted_control_{metric}"]
                .abs()[
                    (diffs[f"bias_adjusted_control_{metric}"] != 0)
                    & diffs[f"bias_adjusted_control_{metric}"].notna()
                    & (~np.isinf(diffs[f"bias_adjusted_control_{metric}"]))
                ]
                .min(),
            )
        )
        logged_max = np.log10(
            max(
                diffs[f"treatment_{metric}"]
                .abs()[
                    (diffs[f"treatment_{metric}"] != 0)
                    & diffs[f"treatment_{metric}"].notna()
                    & (~np.isinf(diffs[f"treatment_{metric}"]))
                ]
                .max(),
                diffs[f"bias_adjusted_control_{metric}"]
                .abs()[
                    (diffs[f"bias_adjusted_control_{metric}"] != 0)
                    & diffs[f"bias_adjusted_control_{metric}"].notna()
                    & (~np.isinf(diffs[f"bias_adjusted_control_{metric}"]))
                ]
                .max(),
            )
        )

        APPROX_N_BINS = 13
        logged_cuts = np.arange(
            logged_min, logged_max + 1, (logged_max - logged_min) / APPROX_N_BINS
        )

        # Include both positive and negative, as some values are negative
        unlogged_cuts = (
            [0] + [10**i for i in logged_cuts] + [-(10**i) for i in logged_cuts]
        )

        unlogged_cuts = sorted(unlogged_cuts)

        for trt_or_ctrl in ["treatment", "control"]:
            diffs[f"{trt_or_ctrl}_{metric}_q"] = pd.cut(
                diffs[f"{trt_or_ctrl}_{metric}"], bins=unlogged_cuts
            )


        trt_obs_per_bin = diffs[f"treatment_{metric}_q"].value_counts()
        trt_bins = trt_obs_per_bin.index[trt_obs_per_bin > 5]

        ctrl_obs_per_bin = diffs[f"control_{metric}_q"].value_counts()
        ctrl_bins = ctrl_obs_per_bin.index[trt_obs_per_bin > 5]

        bins_to_use = [b for b in trt_bins if b in ctrl_bins]

        diffs[f"treatment_{metric}_q"] = np.where(
            diffs[f"treatment_{metric}_q"].isin(bins_to_use),
            diffs[f"treatment_{metric}_q"],
            np.nan,
        )
        diffs[f"control_{metric}_q"] = np.where(
            diffs[f"control_{metric}_q"].isin(bins_to_use),
            diffs[f"control_{metric}_q"],
            np.nan,
        )

        def interval_formatter_factory(column_min, column_max):
            def interval_formatter(value):
                if pd.isna(value):
                    return np.nan
                elif np.isclose(value.right, column_max):
                    return f"> {int(value.left):,}"
                elif np.isclose(value.left, column_min) or value.left < 0:
                    return f"≤ {int(value.right):,}"

                return f"{int(value.left):,}–{int(value.right):,}"

            return interval_formatter

        # Apply formatting
        diffs[f"treatment_{metric}_q_labels"] = diffs[f"treatment_{metric}_q"].map(interval_formatter_factory(diffs[f"treatment_{metric}"].min(), diffs[f"treatment_{metric}"].max()))
        diffs[f"control_{metric}_q_labels"] = diffs[f"control_{metric}_q"].map(interval_formatter_factory(diffs[f"control_{metric}"].min(), diffs[f"control_{metric}"].max() ))

        # Ensure the formatted strings are ordered based on the original intervals
        diffs[f"treatment_{metric}_q_labels"] = pd.Categorical(diffs[f"treatment_{metric}_q_labels"],
                                           categories=sorted(diffs[f"treatment_{metric}_q_labels"].dropna().unique(),
                                                             key=lambda x: float(re.sub(r"(–.*|[>,<≤])", "", x))),
                                           ordered=True)
        diffs[f"control_{metric}_q_labels"] = pd.Categorical(diffs[f"control_{metric}_q_labels"],
                                           categories=sorted(diffs[f"control_{metric}_q_labels"].dropna().unique(),
                                                             key=lambda x: float(re.sub(r"(–.*|[>,<≤])", "", x))),
                                           ordered=True)

        diffs[f"treatment_{metric}_q"] = diffs[f"treatment_{metric}_q"].apply(
            lambda x: gmean([x.left, x.right]) if pd.notna(x) else np.nan
        )
        diffs[f"control_{metric}_q"] = diffs[f"control_{metric}_q"].apply(
            lambda x: gmean([x.left, x.right]) if pd.notna(x) else np.nan
        )


    def safe_call(fn, *args, label=None, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            print(f"Exception in {label or fn.__name__}: {e}")
            traceback.print_exc()
            return None

    safe_call(plot_figure_2, te, config, output_dir, diffs, include_observed_plots=False, include_matching_plots=False, time_limit="5h", highlight_placebo=False, label="plot_figure_2 placebo=False")
    safe_call(plot_figure_2, te, config, output_dir, diffs, include_observed_plots=False, include_matching_plots=False, time_limit="5h", highlight_placebo=True, label="plot_figure_2 placebo=True")
    safe_call(plot_figure_2, te, config, output_dir, diffs, include_observed_plots=True, include_matching_plots=False, label="plot_figure_2 with matching")
    safe_call(plot_figure_2, te, config, output_dir, diffs, include_matching_plots=True, metrics_to_plot="structural", label="plot_figure_2 structural")

    safe_call(plot_figure_4, te_with_metadata, config, output_dir, y_var="Increase", metric="calculated_retweets", include_a=False, center_at_50=True, label=f"plot_figure_4 increase calculated_replies")

    for m in ["calculated_retweets", "calculated_replies", "impressions", "likes"]:
        safe_call(plot_figure_4, te_with_metadata, config, output_dir, metric=m, center_at_50=False, label=f"plot_figure_4 {m}")

    safe_call(plot_multiple_configs, te, config, output_dir, ["main_embd_58", "main_embd_20"], label="embd")
    safe_call(plot_multiple_configs, te, config, output_dir, ["controls_above_0.35", "controls_above_0.3", "controls_above_0.25"], label="limited_donor")
    safe_call(plot_permutations, te, config, output_dir, label="permuted")

    safe_call(plot_figure_1, treatment_tids, control_tweets, treatment_tweets, weights, output_dir, config, te, label="plot_figure_1")



