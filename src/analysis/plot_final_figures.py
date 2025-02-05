import json
import sys
import socket
import shutil
import os
import argparse
from pathlib import Path
from copy import deepcopy
from matplotlib import gridspec
import matplotlib.ticker as ticker

import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import rcParams
from scipy.stats import gmean, norm

# Add current directory to path for imports
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.utils import ConfigError, read_weights, read_control_configs, local_data_root
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
)

from src.analysis.plot_treatment_effects import (
    load_tes_for_metrics,
    get_trt_and_control_ids,
    get_metadata,
    get_artifact_dir,
)
from src.analysis.colors import colors

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

        if metric == "calculated_retweets":
            definetly_plot = [
                "1641823145605709825",
                "1658362065193451521",
                "1654220529329397760",
                "1669118692100063233",
                "1659491718721335297",
                "1653749253720666115",
                "1660265063595270145",
                "1657822137518792705",
                "1587730689008570368",
                "1656679086763266050",
                "1658875945443545090",
                "1639988583863042050",
                "1661794404678635521",
                "1645484675589103620",
                "1666430157379317761",
                "1651121963643838468",
                "1661894568445640706",
                "1662309067765735425",
                "1662266731451269120",
                "1664292351995461633",
            ]
        elif metric == "impressions":
            definetly_plot = [
                "1667440111741444097",
                "1659541496893153280",
                "1658800531127648262",
                "1655232038155300864",
                "1646149996478181376"
            ]
        else:
            definetly_plot = []

        for tid in tqdm(
          definetly_plot + np.random.choice(treatment_tids[metric], size=4, replace=False).tolist()
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
    elif metrics_to_plot == "root_and_non_root_rts":
        metrics = ["rt_cascade_num_nodes_non_root_tweet", "rt_cascade_num_nodes_root_tweet"]
    else:
        raise ValueError("metrics_to_plot must be either 'structural' or 'engagement' or 'root_and_non_root_rts'")

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
        xmin = -1.25
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
            ax.axvspan(-100, -1, color="gray", alpha=0.35, zorder=1000)
            ax.axvspan(0, 100, color="gray", alpha=0.35, zorder=1000)

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
            bbox_to_anchor=(1.15, -0.72),
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


    te_with_metadata[metric]["Increase"] = (te_with_metadata[metric]["bias_adjusted_treatment_effect"] > 0).astype(int) * 100

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
            reverse_legend=True
        )


    # Get data after 48h
    data = te_with_metadata[metric].copy()
    data = data[data["note_0_time_since_first_crh"] == pd.to_timedelta("48h")]


    # Find unique values of the binning variables for us to iterate over
    time_bins = data[data["hours_to_slap_bin"].notna()]["hours_to_slap_bin"].unique()
    bins_start_with_paren = isinstance(time_bins[0], str) and any(
        [b.startswith("(") or b.startswith("[") for b in time_bins]
    )
    if bins_start_with_paren:
        first_num_in_bin = [int(b[1:].split(" ")[0].replace(",", "")) for b in time_bins]
        time_bins = [b for _, b in sorted(zip(first_num_in_bin, time_bins))]
    else:
        time_bins = sorted(time_bins)

    size_bins = data[data[f"pre_break_{metric}_bin"].notna()][f"pre_break_{metric}_bin"].unique()
    bins_start_with_paren = isinstance(size_bins[0], str) and any(
        [b.startswith("(") or b.startswith("[") for b in size_bins]
    )
    if bins_start_with_paren:
        first_num_in_bin = [int(b[1:].split(" ")[0].replace(",", "")) for b in size_bins]
        size_bins = [b for _, b in sorted(zip(first_num_in_bin, size_bins))]
    else:
        size_bins = sorted(size_bins)

    # Replace "(x — y]" with "> x" for the last bins
    new_final_size_bin = "> " + size_bins[-1].split("—")[0].replace("(", "")
    new_final_time_bin = "> " + time_bins[-1].split("—")[0].replace("(", "")

    data["hours_to_slap_bin"] = data["hours_to_slap_bin"].replace(
        {time_bins[-1]: new_final_time_bin}
    )
    data[f"pre_break_{metric}_bin"] = data[f"pre_break_{metric}_bin"].replace(
        {size_bins[-1]: new_final_size_bin}
    )

    time_bins[-1] = new_final_time_bin
    size_bins[-1] = new_final_size_bin

    def get_ci(data, col, tweet_or_note, col_name_for_printing, order):
        data = data[data[y_var].notna()].copy()

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
            "Multiple Photos or Videos": "Multi Img./Vid.",
            "Single Photo": "One Img.",
            "Single Video": "One Vid.",
        }
    )
    points_to_plot += get_ci(
        data,
        "tweet_media",
        "Tweet",
        r"Post" "\nMedia Type",
        order=["Text Only", "One Img.", "One Vid.", "Multi Img./Vid."],
    )

    # Get means/CI for each value of misleading_type
    # Identify columns
    data =data.rename(columns = {
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

    data["note_text_flesch_kincaid_grade_bin"] = data["note_text_flesch_kincaid_grade_bin"].replace(
        {"[-3 — 5]": "≤5",
         "(10 — 147]": ">10"})

    data["note_text_sentence_count_bin"] = np.where(
        data["note_text_sentence_count"] > 3, ">3",
        data["note_text_sentence_count"].round().astype(int).astype(str)
    )

    print(data["note_text_sentence_count_bin"].unique())
    print(data["note_text_flesch_kincaid_grade_bin"].unique())

    points_to_plot += get_ci(data,"note_text_flesch_kincaid_grade_bin", "Note", r"Note" "\nGrade Level", order=[ "≤5", "(5 — 8]", "(8 — 10]", ">10"])

    points_to_plot += get_ci(data,"note_text_sentence_count_bin", "Note", r"Note" "\nSentence Count", order=["1", "2", "3", ">3"])

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
        x = x_pos[label]
        ax2.errorbar(
            x,
            mean,
            yerr=([mean - lower], [upper - mean]),
            fmt="o",
            color=colors["bin_te_2.5"],
            markersize=4,
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
        te_with_metadata[metric]["Increase"] = (te_with_metadata[metric]["bias_adjusted_treatment_effect"] > 0).astype(int) * 100


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
        "use_backup_tweets",
        "use_bookmark_tweets",
        "volatile_tweet_filtering",
        "max_date",
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
    ]
    for c in necessary_config:
        if c not in config.keys():
            raise ConfigError(
                f"Please specify config value '{c}' in config file '{config_path}.'"
            )

    # Convert lambda from list
    if len(config["lambda"]) == 1:
        config["lambda"] = config["lambda"][0]
    else:
        raise ConfigError("Unsure how to handle multiple lambda values in same config.")

    # Drop unneeded config values
    config = {c: config[c] for c in necessary_config}

    # Fill in the dev value based on whether this is a local run
    if config["dev"] == "DEVICE_DEFAULT":
        config["dev"] = socket.gethostname() == "is-is28m16x"

    # Convert max date to datetime
    config["max_date"] = pd.to_datetime(config["max_date"], utc=True)

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
        local_data_root
        / "cn_effect_output"
        / "paper_figures"
        / Path(config_path).name.replace(".json", "")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    config["artifact_dir"] = get_artifact_dir(config)

    weights = read_weights(config["intermediate_dir"], config)

    te_config = deepcopy(config)
    del te_config["intermediate_dir"]
    del te_config["artifact_dir"]
    metrics = config["target_metrics"]
    metrics += [m + "_pct_change" for m in metrics]
    metrics = sorted(metrics)
    te = load_tes_for_metrics(metrics, te_config, config["intermediate_dir"])

    # Get trt and control tweets
    treatment_tids, control_tids = get_trt_and_control_ids(metrics, te, weights)
    control_tweets, treatment_tweets = read_trt_and_ctrl(
        config["intermediate_dir"], config, logger=None
    )

    control_configs = read_control_configs(config["intermediate_dir"], config)

    tweet_metadata = get_metadata(
        treatment_tids,
        treatment_tweets,
        control_configs,
        config["intermediate_dir"],
        config["artifact_dir"],
        config,
        metrics=te.keys(),
    )

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

    ds = diffs
    for metric in ["calculated_retweets", "calculated_replies", "impressions", "likes"]:

        for trt_or_ctrl in ["treatment", "control"]:

            ds[f"{trt_or_ctrl}_{metric}"] = np.where(
                ds[f"{trt_or_ctrl}_{metric}"] >= 0,
                ds[f"{trt_or_ctrl}_{metric}"],
                np.nan,
            )

        pooled = pd.concat([ds[f"treatment_{metric}"], ds[f"bias_adjusted_control_{metric}"]])

        logged_min = np.log10(
            min(
                ds[f"treatment_{metric}"]
                .abs()[
                    (ds[f"treatment_{metric}"] != 0)
                    & ds[f"treatment_{metric}"].notna()
                    & (~np.isinf(ds[f"treatment_{metric}"]))
                ]
                .min(),
                ds[f"bias_adjusted_control_{metric}"]
                .abs()[
                    (ds[f"bias_adjusted_control_{metric}"] != 0)
                    & ds[f"bias_adjusted_control_{metric}"].notna()
                    & (~np.isinf(ds[f"bias_adjusted_control_{metric}"]))
                ]
                .min(),
            )
        )
        logged_max = np.log10(
            max(
                ds[f"treatment_{metric}"]
                .abs()[
                    (ds[f"treatment_{metric}"] != 0)
                    & ds[f"treatment_{metric}"].notna()
                    & (~np.isinf(ds[f"treatment_{metric}"]))
                ]
                .max(),
                ds[f"bias_adjusted_control_{metric}"]
                .abs()[
                    (ds[f"bias_adjusted_control_{metric}"] != 0)
                    & ds[f"bias_adjusted_control_{metric}"].notna()
                    & (~np.isinf(ds[f"bias_adjusted_control_{metric}"]))
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
            ds[f"{trt_or_ctrl}_{metric}_q"] = pd.cut(
                ds[f"{trt_or_ctrl}_{metric}"], bins=unlogged_cuts
            )


        trt_obs_per_bin = ds[f"treatment_{metric}_q"].value_counts()
        trt_bins = trt_obs_per_bin.index[trt_obs_per_bin > 5]

        ctrl_obs_per_bin = ds[f"control_{metric}_q"].value_counts()
        ctrl_bins = ctrl_obs_per_bin.index[trt_obs_per_bin > 5]

        bins_to_use = [b for b in trt_bins if b in ctrl_bins]

        ds[f"treatment_{metric}_q"] = np.where(
            ds[f"treatment_{metric}_q"].isin(bins_to_use),
            ds[f"treatment_{metric}_q"],
            np.nan,
        )
        ds[f"control_{metric}_q"] = np.where(
            ds[f"control_{metric}_q"].isin(bins_to_use),
            ds[f"control_{metric}_q"],
            np.nan,
        )

        def format_interval(interval):
            if pd.isna(interval):
                return np.nan
            return f"({round(interval.left):,}—{round(interval.right):,}]"


        # Apply formatting
        ds[f"treatment_{metric}_q_labels"] = ds[f"treatment_{metric}_q"].apply(format_interval)
        ds[f"control_{metric}_q_labels"] = ds[f"control_{metric}_q"].apply(format_interval)

        # Ensure the formatted strings are ordered based on the original intervals
        ds[f"treatment_{metric}_q__labels"] = pd.Categorical(ds[f"treatment_{metric}_q_labels"],
                                           categories=sorted(ds[f"treatment_{metric}_q_labels"].dropna().unique(),
                                                             key=lambda x: float(x.split('—')[0][1:].replace(',', ''))),
                                           ordered=True)
        ds[f"control_{metric}_q_labels"] = pd.Categorical(ds[f"control_{metric}_q_labels"],
                                           categories=sorted(ds[f"control_{metric}_q_labels"].dropna().unique(),
                                                             key=lambda x: float(x.split('—')[0][1:].replace(',', ''))),
                                           ordered=True)

        ds[f"treatment_{metric}_q"] = ds[f"treatment_{metric}_q"].apply(
            lambda x: gmean([x.left, x.right]) if pd.notna(x) else np.nan
        )
        ds[f"control_{metric}_q"] = ds[f"control_{metric}_q"].apply(
            lambda x: gmean([x.left, x.right]) if pd.notna(x) else np.nan
        )



    ## Use for generating placebo test figure in appendix
    plot_figure_2(te, config, output_dir, diffs, include_observed_plots=False, include_matching_plots=False,
                  time_limit="0.5h", highlight_placebo=True)
    plot_figure_2(te, config, output_dir, diffs, include_observed_plots=False, include_matching_plots=False,
                  time_limit="3h", highlight_placebo=False)
    plot_figure_2(te, config, output_dir, diffs, include_observed_plots=False, include_matching_plots=False,
                  time_limit="6h", highlight_placebo=False)
    plot_figure_2(te, config, output_dir, diffs, include_observed_plots=False, include_matching_plots=False,
                  time_limit="5h", highlight_placebo=False)
    plot_figure_2(te, config, output_dir, diffs, include_observed_plots=False, include_matching_plots=False,
                  time_limit="6h", highlight_placebo=False)
    plot_figure_2(te, config, output_dir, diffs, include_observed_plots=False, include_matching_plots=False,
                  time_limit="7h", highlight_placebo=False)
    plot_figure_2(te, config, output_dir, diffs, include_observed_plots=False, include_matching_plots=False,
                  time_limit="8h", highlight_placebo=False)
    plot_figure_2(te, config, output_dir, diffs, include_observed_plots=False, include_matching_plots=False,
                  time_limit="9h", highlight_placebo=False)
    plot_figure_2(te, config, output_dir, diffs, include_observed_plots=False, include_matching_plots=False,
                  time_limit="10h", highlight_placebo=False)
    plot_figure_2(te, config, output_dir, diffs, include_observed_plots=False, include_matching_plots=False,
                  time_limit="11h", highlight_placebo=False)
    plot_figure_2(te, config, output_dir, diffs, include_observed_plots=False, include_matching_plots=False,
                  time_limit="12h", highlight_placebo=False)


    try:
        plot_figure_1(
            treatment_tids,
            control_tweets,
            treatment_tweets,
            weights,
            output_dir,
            config,
            te,
        )
    except Exception as e:
        pass
    try:
        plot_all_metrics_by_pre_treatment_visibility(te_with_metadata, config, output_dir,
                                                     y_var="bias_adjusted_treatment_effect")
        for _METRIC in ["calculated_retweets", "calculated_replies", "impressions", "likes"]:
            plot_figure_4(te_with_metadata, config, output_dir, y_var="Increase", metric=_METRIC, include_a=False, center_at_50=True)
            plot_figure_4(te_with_metadata, config, output_dir, y_var="Increase",  metric=_METRIC, include_a=False, center_at_50=True, only_plot_top_bins_in_B=True)
            plot_figure_4(te_with_metadata, config, output_dir, metric=_METRIC, center_at_50=False)
            plot_figure_4(te_with_metadata, config, output_dir, metric=_METRIC, only_plot_top_bins_in_B=True)
    except Exception as e:
        # Ignore exception, for now
        pass

    plot_figure_2(te, config, output_dir, diffs, include_observed_plots=True, include_matching_plots=False)

    plot_figure_2(te, config, output_dir, diffs, include_matching_plots=True)
    plot_figure_2(te, config, output_dir, diffs, include_matching_plots=False,
                  treatment_variable="bias_adjusted_relative_treatment_effect")
    plot_figure_2(te, config, output_dir, diffs, include_matching_plots=False,
                  treatment_variable="bias_adjusted_growth")

    plot_figure_2(te, config, output_dir, diffs, include_matching_plots=True, metrics_to_plot="structural")

    plot_figure_2(te, config, output_dir, diffs, include_matching_plots=False, metrics_to_plot="root_and_non_root_rts")
    plot_figure_2(te, config, output_dir, diffs, include_matching_plots=False, metrics_to_plot="root_and_non_root_rts",
                  treatment_variable="bias_adjusted_relative_treatment_effect")
    plot_figure_2(te, config, output_dir, diffs, include_matching_plots=False, metrics_to_plot="root_and_non_root_rts",
                  treatment_variable="bias_adjusted_growth")





