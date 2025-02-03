import os
import re

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt, ticker as ticker
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib import gridspec
from scipy.stats import norm

from src.analysis.colors import colors as analysis_colors

USE_CUSTOM_YLIMS = True
SUBFIGURE_LETTER_SIZE = 12
INDIVIDUAL_LINE_WIDTH = 0.75
AVERAGE_LINE_WIDTH = 1.5


class MetricFormatter:
    def __init__(self, suffix=""):
        self.prefixes = {0: "", 3: "k", 6: "M", 9: "B", 12: "T"}
        self.suffix = suffix

    def __call__(self, x, p):
        """
        Format number with metric prefix and appropriate precision

        Parameters:
        x (float): Number to format
        p (int): Precision (unused but required by matplotlib)

        Returns:
        str: Formatted string
        """
        # Handle 0 and negative numbers
        if x == 0:
            return "0"

        is_negative = x < 0
        x = abs(x)

        # Handle small decimals (< 0.001)
        if x < 0.001:
            return f"{'-' if is_negative else ''}{x}"

        # Handle numbers between 0 and 1
        if x < 1:
            rounded = round(x, 3)
            if rounded == 0:  # If rounding would make it 0, keep original
                return f"{'-' if is_negative else ''}{x}"
            formatted = f"{rounded:.3f}".rstrip("0").rstrip(".")
            return f"{'-' if is_negative else ''}{formatted}"

        # Find the appropriate prefix
        exp = max(k for k in self.prefixes.keys() if x >= 10**k)
        prefix = self.prefixes[exp]

        # Scale the number
        scaled = x / (10**exp) if exp > 0 else x

        # Determine precision based on number size
        if scaled >= 100:
            precision = 0
        elif scaled >= 10:
            precision = 1
        else:
            precision = 2

        # Format the number with appropriate precision
        formatted = f"{scaled:.{precision}f}"

        # Remove trailing zeros after decimal point
        if "." in formatted:
            formatted = formatted.rstrip("0").rstrip(".")

        # Add prefix and handle negative numbers
        result = f"{'-' if is_negative else ''}{formatted}{prefix}{self.suffix}"

        return result


# Create formatter instance
metric_formatter = ticker.FuncFormatter(MetricFormatter())


def plot_individual_tweet(
    tid,
    metric,
    control_tweets,
    treatment_tweets,
    weights,
    output_dir,
    config,
    te,
    save_to_disk=True,
    n_to_plot=None,
    weight_filter=0.01,
    add_subplot_identifiers=True,
):

    # Convert 17.8 cm to inches for figure size (1 inch = 2.54 cm)
    width_in_inches = 17.8 / 2.54 / 2
    height_in_inches = width_in_inches / 9 * 2 * 7 / 10

    assert (n_to_plot is None) or (weight_filter is None)
    assert not ((n_to_plot is None) and (weight_filter is None))

    data_collection_start_time = (
        treatment_tweets[tid][treatment_tweets[tid][metric].notna()][
            "note_0_time_since_first_crh"
        ]
        .dt.total_seconds()
        .min()
        / 3600
    )

    slap_time = 0

    xlim = (data_collection_start_time, 48.1)

    ######### Format data
    treatment = treatment_tweets[tid].copy()[
        ["time_since_publication", "note_0_time_since_first_crh", metric]
    ]
    treatment["hours_since_publication"] = (
        treatment["time_since_publication"].dt.total_seconds() / 3600
    )
    treatment["note_0_hours_since_first_crh"] = (
        treatment["note_0_time_since_first_crh"].dt.total_seconds() / 3600
    )
    treatment[["tweet_id"]] = "treatment"

    control_ids = [col for col in weights[tid].columns if col.isdigit()]

    controls_df_long = pd.concat(
        [
            control_tweets[cid][["time_since_publication", "tweet_id", metric]]
            for cid in control_ids
        ]
    )
    controls_df_long["hours_since_publication"] = (
        controls_df_long["time_since_publication"].dt.total_seconds() / 3600
    )

    controls_df_wide = controls_df_long.pivot(
        index="time_since_publication", columns="tweet_id", values=metric
    )[control_ids]

    control = (
        pd.DataFrame(
            controls_df_wide[control_ids].to_numpy()
            * weights[tid][control_ids].to_numpy(),
            index=controls_df_wide.index,
            columns=control_ids,
        )
        .sum(axis=1)
        .reset_index()
    )
    control.columns = ["time_since_publication", metric]
    control["hours_since_publication"] = (
        control["time_since_publication"].dt.total_seconds() / 3600
    )
    # Merge to time since CRH
    control = control.merge(
        treatment[
            ["hours_since_publication", "note_0_hours_since_first_crh", metric]
        ].drop_duplicates(),
        on="hours_since_publication",
        suffixes=("", "_treatment"),
    )
    controls_df_long = controls_df_long.merge(
        treatment[
            ["hours_since_publication", "note_0_hours_since_first_crh"]
        ].drop_duplicates(),
        on="hours_since_publication",
    )

    if n_to_plot:
        controls_to_plot = (
            weights[tid][control_ids].transpose().sort_values(0).index[-n_to_plot:]
        )
    else:
        controls_to_plot = (
            weights[tid][control_ids]
            .transpose()[(weights[tid][control_ids] > weight_filter).values.flatten()]
            .index.tolist()
        )

    ######### Create a 2-column by 1-row subplot grid without shared y-axis
    # Create a main GridSpec with 2 rows, 1 column
    fig = plt.figure(figsize=(width_in_inches, height_in_inches))

    main_gs = gridspec.GridSpec(1, 1)

    # Create a sub-GridSpec for the top row only, with extra space for legends
    top_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=main_gs[0], width_ratios=(1, 1))

    # Create axes
    axs = [
        fig.add_subplot(top_gs[0]),  # Top left
        fig.add_subplot(top_gs[1])  # Bottom, full width
    ]


    ######### First plot: Full treatment vs. control plot, with donors
    sns.lineplot(
        x="note_0_hours_since_first_crh",
        y=metric,
        data=treatment,
        color=analysis_colors["treated"],
        alpha=1,  # Full opacity
        label="Treatment Post",
        zorder=999,
        ax=axs[0],
        linestyle="solid",
        linewidth=INDIVIDUAL_LINE_WIDTH,
    )

    sns.lineplot(
        x="note_0_hours_since_first_crh",
        y=metric,
        data=control,
        color=analysis_colors["control"],
        alpha=1,  # Full opacity
        label="Synthetic Control",
        zorder=1000,
        linestyle="--",
        ax=axs[0],  # Second subplot (middle)
        linewidth=INDIVIDUAL_LINE_WIDTH,
    )

    for cid in controls_to_plot:
        sns.lineplot(
            x="note_0_hours_since_first_crh",
            y=metric,
            data=controls_df_long[controls_df_long["tweet_id"] == cid],
            color=analysis_colors["control"],
            alpha=weights[tid][cid].iloc[0]
            / weights[tid][control_ids].max().max()
            * 0.3,
            zorder=-1001,
            linestyle="solid",
            ax=axs[0],  # Second subplot (middle)
            linewidth=INDIVIDUAL_LINE_WIDTH,
        )

    # Set x and y limits based on the treatment data
    y_limit_treatment = (
            1.1
            * treatment[
                (treatment["note_0_hours_since_first_crh"] <= xlim[1])
                & (treatment["note_0_hours_since_first_crh"] >= xlim[0])
                ][metric].max()
    )

    axs[0].set_xlim(*xlim)
    y_limit_control = 1.1 * max(
        control[control["note_0_hours_since_first_crh"] <= xlim[1]][metric].max(),
        treatment[treatment["note_0_hours_since_first_crh"] <= xlim[1]][metric].max(),
    )
    shared_ylim = max(y_limit_treatment, y_limit_control)
    axs[0].set_ylim(0, shared_ylim)

    del y_limit_treatment, y_limit_control, shared_ylim

    axs[0].axhline(0, c="lightgray", alpha=1, zorder=-1000)
    axs[0].axvspan(
        xlim[0],
        slap_time,
        color="gray",
        alpha=0.25,
        zorder=-2000,
    )
    axs[0].axvline(
        slap_time,
        c=analysis_colors["other_2"],
        alpha=1,
        zorder=-1000,
        linestyle=":",
        linewidth=INDIVIDUAL_LINE_WIDTH,
    )

    ######### Second plot: Treatment effect
    # third_plot_data = te[metric][te[metric]["tweet_id"] == tid].copy()
    # third_plot_data = third_plot_data.merge(
    #     treatment[["hours_since_publication", "note_0_time_since_first_crh"]]
    # )

    third_plot_data = control.copy()
    third_plot_data["unadjusted_treatment_effect"] = (
        third_plot_data[f"{metric}_treatment"] - third_plot_data[metric]
    )

    sns.lineplot(
        x="note_0_hours_since_first_crh",
        y="unadjusted_treatment_effect",
        data=third_plot_data,
        color=analysis_colors["te"],
        alpha=1,  # Full opacity
        zorder=1000,
        ax=axs[1],  # Second subplot
        linestyle="--",
        linewidth=INDIVIDUAL_LINE_WIDTH,
    )

    # Add line and shaded area
    axs[1].axhline(0, c="lightgray", alpha=1, zorder=-1000)
    axs[1].axvspan(
        xlim[0],
        slap_time,
        color="gray",
        alpha=0.25,
        zorder=-1000,
    )
    axs[1].axvline(
        slap_time,
        c=analysis_colors["other_2"],
        alpha=1,
        zorder=-999,
        linestyle=":",
        linewidth=INDIVIDUAL_LINE_WIDTH
    )

    # Set xlim
    axs[1].set_xlim(*xlim)

    # Set ylim so that plot appears to show difference from slap
    zero_point = treatment[treatment["note_0_hours_since_first_crh"] >= slap_time][
        metric
    ].iloc[0]
    upper_lim = axs[0].get_ylim()[1] - zero_point
    lower_lim = axs[0].get_ylim()[0] - zero_point
    axs[1].set_ylim(lower_lim, upper_lim)

    del zero_point, upper_lim, lower_lim


    ######### Set Y-axis labels
    axs[0].set_ylabel("Individual Treatment\nand Synthetic Control", fontsize=6.4)
    axs[1].set_ylabel("Individual\nTreatment Effect")

    for ax in axs:
        ######### Set titles
        ax.set_title(
            metric.replace("calculated_", "")
            .replace("impressions", "views")
            .replace("tweet", "post")
            .title()
        )

        ######### Set X-axis labels
        ax.set_xlabel("Hours After Note Attached", color=analysis_colors["hours"], fontsize=6.4)

        ######### Remove individual legends from all subplots
        ax.legend().remove()

    # Create first plot legend
    i_treated_proxy = mlines.Line2D(
        [], [], color="black", linestyle="-", label="Views for Individual\nTreatment Post", linewidth=INDIVIDUAL_LINE_WIDTH,
    )
    matching_period_proxy = mpatches.Patch(
        color="gray", alpha=0.25, label="Matching Period"
    )

    attached_proxy = mlines.Line2D(
        [],
        [],
        color=analysis_colors["other_2"],
        linestyle=":",
        transform=axs[0].get_yaxis_transform(),  # Makes the line vertical
        label="Note Attached",
        linewidth=INDIVIDUAL_LINE_WIDTH
    )

    donor_proxy = mlines.Line2D(
        [],
        [],
        color=analysis_colors["control"],
        alpha=0.3,
        label="Donor Post",
        linestyle="solid",
        linewidth=INDIVIDUAL_LINE_WIDTH
    )
    i_sc_proxy = mlines.Line2D(
        [],
        [],
        color=analysis_colors["control"],
        alpha=1,
        label="Views for Individual\nSynthetic Control",
        linestyle="--",
        linewidth=INDIVIDUAL_LINE_WIDTH
    )

    blank_proxy = mlines.Line2D([], [], color="white", label="")
    ite_proxy = mlines.Line2D(
        [],
        [],
        color=analysis_colors["te"],
        label="Estimated Individual\nTreatment Effect",
        linestyle="--",
        linewidth=INDIVIDUAL_LINE_WIDTH
    )

    axs[1].legend(
        handles=[
            i_treated_proxy,
            i_sc_proxy,
            blank_proxy,


            matching_period_proxy,
            attached_proxy,
            donor_proxy,

            # blank_proxy,
            ite_proxy,

        ],
        loc="upper center",
        bbox_to_anchor=(-0.45, -0.6),
        ncol=3,
        frameon=False,
        columnspacing=1,
        fontsize=6,
        labelcolor=analysis_colors["legend"],
    )

    # Format y-axis ticks
    for ax in axs:
        ax.yaxis.set_major_formatter(metric_formatter)

    if add_subplot_identifiers:
        ######### Add "A", "B", "C", "D" to the subplots
        for idx, label in enumerate(["A", "B"]):
            axs[idx].text(
                -0.12,
                1.29,
                label,
                transform=axs[idx].transAxes,
                fontweight="bold",
                va="top",
                ha="left",
                size=12,
            )

    fig.subplots_adjust(bottom=0, top=1.1, wspace=0.53)  # Adjust as needed for space

    ######### Format ticks
    for ax in axs:
        ax.xaxis.set_major_locator(plt.MultipleLocator(12))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(3))

        # Make sure there are at least five major ticks
        # Get the y range
        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]

        # Find the largest number like 1000 or 500 that results in more than 5 tickets
        tick_size = None


        for exp in range(100, -100, -1):
            if y_range / (10 ** exp) >= 4 and y_range / (10 ** exp) < 7:
                tick_size = int(10 ** exp)
            elif y_range / (10 ** exp / 2) >= 4 and y_range / (10 ** exp / 2) < 7:
                tick_size = int(10 ** exp / 2)
            elif y_range / (10 ** (exp - 1) * 2) >= 4 and y_range / (10 ** (exp - 1) * 2) < 7:
                tick_size = int(10 ** (exp - 1) * 2)
            elif y_range / (10 ** (exp - 1) * 3) >= 4 and y_range / (10 ** (exp - 1) * 3) < 7:
                tick_size = int(10 ** (exp - 1) * 3)
            elif y_range / (10 ** (exp - 1) * 4) >= 4 and y_range / (10 ** (exp - 1) * 4) < 7:
                tick_size = int(10 ** (exp - 1) * 4)

            if tick_size:
                # Set the major ticks
                ax.yaxis.set_major_locator(plt.MultipleLocator(tick_size))

                # Remove minor ticks
                ax.yaxis.set_minor_locator(plt.NullLocator())
                break

        if not tick_size:
            print(f"Could not find a good tick size for {metric} with range {y_range}")



    ######### Adjust layout and save or show the plot
    # fig.tight_layout(rect=[0, -0.1, 0, 0])  # Adjust rect to fit legends without overlap

    if save_to_disk:
        fig.savefig(
            output_dir / "individual_tweets" / metric / f"{tid}_trt_and_control.pdf",
            bbox_inches="tight",
        )
    else:
        plt.show()

    plt.close(fig)


def plot_overall(
    data_to_plot,
    metric,
    output_dir,
    color,
    ylim,
    xlim,
    y_var="bias_adjusted_treatment_effect",
    save=False,
    ax=None,  # Optional axes
    include_y_axis_label=True,
    use_basic_axis_labels=False,
    include_ci=True,
    reverse_legend=False,
    stat="mean",
    include_matching_window=True,
):
    data_to_plot = data_to_plot.copy()

    # Calculate RTE on the fly
    if y_var == "bias_adjusted_relative_treatment_effect":
        data_to_plot[y_var] = data_to_plot["bias_adjusted_treatment_effect"] / data_to_plot["bias_adjusted_control"] * 100
    elif y_var == "bias_adjusted_growth":
        first = data_to_plot[data_to_plot["note_0_hours_since_first_crh"] == 0][["tweet_id", "treatment", "bias_adjusted_control"]]
        data_to_plot = data_to_plot.merge(first, on="tweet_id", suffixes=("", "_at_0"))

        bcc_growth = data_to_plot["bias_adjusted_control"] - data_to_plot["bias_adjusted_control_at_0"]
        t_growth = data_to_plot["treatment"] - data_to_plot["treatment_at_0"]

        data_to_plot[y_var] = (
                                       t_growth - bcc_growth
                               ) / bcc_growth * 100

    if y_var in ["bias_adjusted_relative_treatment_effect", "bias_adjusted_growth"]:
        # Clip top/bottom 1% of RTEs
        bad_tids = []
        for t in data_to_plot["note_0_hours_since_first_crh"].unique():
            t_obs = data_to_plot[
                data_to_plot["note_0_hours_since_first_crh"] == t
                ]

            p1 = t_obs[y_var].quantile(0.005) # Calc 1st pctile
            p99 = t_obs[y_var].quantile(0.995)  # Calc 99th pctile
            bad_tids += t_obs[
                (t_obs[y_var] < p1) | (t_obs[y_var] > p99)
            ]["tweet_id"].to_list()  # Find TIDs outside of 1-99th pctiles

        bad_tids = list(set(bad_tids))  # Remove duplicates
        data_to_plot = data_to_plot[
            ~data_to_plot["tweet_id"].isin(bad_tids)
        ]
        print(f"Removed {len(bad_tids):,}/{data_to_plot['tweet_id'].nunique()} outliers for metric {metric} and yvar {y_var}")

    # If no axes are passed in, create a new figure and axes
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure  # Get the figure from the axes

    if include_ci and stat == "mean":
        errorbar = ("se", norm.ppf(0.975))
    elif include_ci and stat == "median":
        errorbar = ("ci", 0.95)
    else:
        errorbar = None

    sns.lineplot(
        data=data_to_plot,
        x="note_0_hours_since_first_crh",
        y=y_var,
        errorbar=errorbar,
        c=color,
        estimator=stat,
        ax=ax,  # Plot on the passed axes
        linestyle="-" if y_var == "treatment" else "--",
        linewidth=AVERAGE_LINE_WIDTH,
    )

    # Set plot limits
    if USE_CUSTOM_YLIMS and not ylim is None:
        ax.set_ylim(*ylim)
    ax.set_xlim(*xlim)

    # Lines at x and y = 0
    # Check if plot already has an axvspan
    if not any([line.get_alpha() == 0.25 for line in ax.lines]):

        ax.axhline(0, c="lightgray", alpha=1, zorder=-1000)
        if include_matching_window:
            ax.axvline(0, c=analysis_colors["other_2"], alpha=1, zorder=-1000, linestyle=":",
                       linewidth=INDIVIDUAL_LINE_WIDTH)
            ax.axvspan(
                -100,
                0,
                color="gray",
                alpha=0.25,
                zorder=-1000,
            )

    # Label axes
    ax.set_xlabel("Hours After Note Attached", color=analysis_colors["hours"])
    if include_y_axis_label:
        ylab = (
            metric.replace("impression", "view")
            .replace("rt_cascade_", "")
            .replace("calculated_", "")
            .replace("width", "max_breadth")
            .replace("tweet", "post")
            .replace("reposts", "size")
            .replace("wiener_index", "structural_virality")
            .replace("_", " ")
            .title()
        )

        # Change x_pct_change to "Percent Change in x"
        if "Pct Change" in ylab:
            ylab = re.sub(r"(\w+) Pct Change", r"Percent Change in \1", ylab)

        if use_basic_axis_labels:
            if "Percent Change" in ylab:
                ylab = "Percent Change"
            elif "Rate" in ylab or "Per View" in ylab:
                ylab = "Change per View"
            elif y_var in ["bias_adjusted_treatment_effect"]:
                ylab = "Average Treatment Effect"
            elif y_var in ["bias_adjusted_relative_treatment_effect"]:
                ylab = "Average Relative Treatment Effect"
            elif y_var in ["bias_adjusted_growth"]:
                ylab = "Average Percent Change in Growth"
            else:
                ylab = "Average Treatment\nand Synthetic Control"

        ax.set_ylabel(ylabel=ylab)
    else:
        ax.set_ylabel("")

    # Set major ticks
    if abs(xlim[1] - xlim[0]) >= 24:
        ax.xaxis.set_major_locator(plt.MultipleLocator(12))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(3))
    elif abs(xlim[1] - xlim[0]) <= 5:
        ax.xaxis.set_major_locator(plt.MultipleLocator(1))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(0.25))
    elif abs(xlim[1] - xlim[0]) <= 2:
        ax.xaxis.set_major_locator(plt.MultipleLocator(0.25))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(0.125))
    else:
        ax.xaxis.set_major_locator(plt.MultipleLocator(2))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(1))

    # Format y-axis
    # Format y-axis ticks with commas
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(MetricFormatter("%" if y_var in ["bias_adjusted_relative_treatment_effect", "bias_adjusted_growth"] else ""))
    )

    # Reverse order of legend
    if reverse_legend:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1])

    fig.tight_layout()
    save_name = "reach" if metric == "impressions" else metric

    if save:
        os.makedirs(output_dir / y_var, exist_ok=True)
        fig.savefig(output_dir / y_var / f"{save_name}.pdf")
    else:
        fig.show()

    if ax is None:  # Only close if we created a new figure
        plt.close(fig)


def treatment_and_control_scatter_binned(
    diffs, metric_1, metric_2, output_dir, ax=None, offset=False
):
    if ax is None:
        savefig = True
        plt.clf()
        fig, ax = plt.subplots()
    else:
        savefig = False

    positive = diffs.copy()


    xticks = sorted(
        x
        for x in positive[f"control_{metric_1}_q"].unique().astype(float)
        if not pd.isna(x)
    )
    xtick_labels = (
        positive[f"control_{metric_1}_q_labels"].cat.categories.astype(str).tolist()
    )
    # Replace "X — Y" with "X+" for last bin
    xtick_labels[-1] = "> " + xtick_labels[-1].split("—")[0].replace("(", "")


    if offset:

        def apply_log_offset(df, offset=1.3):
            """
            Apply a log scale offset to control and treatment columns.

            Args:
                df (pd.DataFrame): DataFrame containing the columns.
                metric_1 (str): Metric name used to identify columns.
                offset (float): The offset to apply on the log scale.

            Returns:
                pd.DataFrame: DataFrame with updated control and treatment columns.
            """
            control_col = f"control_{metric_1}_q"
            treatment_col = f"treatment_{metric_1}_q"

            # Apply log-offset transformation
            df[control_col] = df[control_col].astype(float) / offset
            df[treatment_col] = df[treatment_col].astype(float) * offset

            return df

        # Apply log-scale offsets
        positive = apply_log_offset(positive, offset=1.3)


    # Plotting the scatter plots
    sns.lineplot(
        data=positive,
        x=f"treatment_{metric_1}_q",
        y=f"treatment_{metric_2}",
        color=analysis_colors["treated"],
        marker="o",
        label="Treatment",
        errorbar=("se", norm.ppf(0.975)),
        estimator="mean",
        markersize=4,
        linewidth=0.15,
        ax=ax,
        linestyle="",
        err_style="bars",
        alpha=0.75 if offset else 0.5,
        #  error bars behind markers
        err_kws={"zorder": -1000},
    )

    sns.lineplot(
        data=positive,
        x=f"control_{metric_1}_q",
        y=f"control_{metric_2}",
        color=analysis_colors["control"],
        marker="o",
        label="Control",
        errorbar=("se", norm.ppf(0.975)),
        estimator="mean",
        markersize=4,
        linewidth=0.15,
        ax=ax,
        linestyle="",
        err_style="bars",
        alpha=0.75 if offset else 0.5,
        #  error bars behind markers
        err_kws={"zorder": -1000},
    )

    # Setting labels
    ax.set_xlabel(
        "$\Delta$ "
        + (
            metric_1.replace("impression", "view")
            .replace("rt_cascade_", "")
            .replace("calculated_", "")
            .replace("width", "max_breadth")
            .replace("depth", "max_depth")
            .replace("tweet", "post")
            .replace("reposts", "cascade_size")
            .replace("wiener_index", "structural_virality")
            .replace("_", " ")
            .title()
        )
        # + " in 48h After Note"
    )
    ax.set_ylabel(
        "$\Delta$ "
        + (
            metric_2.replace("impression", "view")
            .replace("rt_cascade_", "")
            .replace("calculated_", "")
            .replace("width", "max_breadth")
            .replace("depth", "max_depth")
            .replace("tweet", "post")
            .replace("reposts", "cascade_size")
            .replace("wiener_index", "structural_virality")
            .replace("_", " ")
            .title()
        )
        # + " in 48h After Note"
    )

    # Apply log scale to x axis after setting ticks
    ax.set_xscale("log")

    # Set major ticks (every 2nd)
    EVERY_OTHER = 1
    ax.xaxis.set_major_locator(plt.FixedLocator(xticks[::EVERY_OTHER]))

    # Set minor ticks (the rest)
    minor_ticks = [x for i, x in enumerate(xticks) if i % EVERY_OTHER != 0]
    ax.xaxis.set_minor_locator(plt.FixedLocator(minor_ticks))

    # Clear and set major tick labels only
    ax.xaxis.set_major_formatter(plt.NullFormatter())  # Clear major labels
    ax.xaxis.set_minor_formatter(plt.NullFormatter())  # Clear minor labels
    ax.set_xticklabels(
        xtick_labels[::EVERY_OTHER],
    )

    # Tilt x-axis labels
    ax.tick_params(axis="x", rotation=45)

    # Right-align the x-tick labels
    for label in ax.get_xticklabels():
        label.set_ha('right')

    # Format axis ticks
    ax.yaxis.set_major_formatter(metric_formatter)

    def log_midpoints(tickmarks):
        """
        Find the midpoints on a logarithmic scale between consecutive tickmarks.

        Args:
            tickmarks (list): List of tickmark values on a log scale.

        Returns:
            list: List of midpoints in log scale between consecutive tickmarks.
        """
        tickmarks = np.array(tickmarks)  # Convert to numpy array for easier operations
        log_ticks = np.log10(tickmarks)  # Convert to log10 scale
        midpoints = 10 ** ((log_ticks[:-1] + log_ticks[1:]) / 2)  # Compute midpoints
        return midpoints.tolist()

    # Calculate midpoints between ticks using geometric mean
    midpoints = log_midpoints(xticks)

    # Create bands using midpoints
    for i in range(0, len(midpoints) - 1, 2):  # Step by 2 for every other band
        ax.axvspan(midpoints[i], midpoints[i + 1], color="gray", alpha=0.1, zorder=-1)

    plt.tight_layout()

    # Remove legend
    ax.get_legend().remove()

    # Save the figure
    if savefig:
        plt.savefig(
            output_dir / f"{metric_1}_vs_{metric_2}.pdf".replace("calculated_", "")
        )


def plot_bins(
    data_to_plot,
    metric,
    output_dir,
    colors,
    dashes,
    ylim,
    xlim,
    y_var="bias_adjusted_treatment_effect",
    save=False,
    binning_variable="hours_to_slap_bin",
    ax=None,
    build=False,
    reverse_legend=False,
    include_legend=True,
    include_y_axis_label=True,
):

    assert not (build and not save), "Cannot build plots 1-by-1 without saving"

    # Find unique values of the binning variable for us to iterate over
    bins = data_to_plot[data_to_plot[binning_variable].notna()][
        binning_variable
    ].unique()
    bins_start_with_paren = isinstance(bins[0], str) and any(
        [b.startswith("(") or b.startswith("[") for b in bins]
    )
    if bins_start_with_paren:
        first_num_in_bin = [int(b[1:].split(" ")[0].replace(",", "")) for b in bins]
        bins = [b for _, b in sorted(zip(first_num_in_bin, bins))]
    else:
        bins = sorted(bins)

    # Find out if we're plotting everything at once, or building up the plot 1-by-1
    if build:
        num_plots = len(bins)
    else:
        num_plots = 1

    for i in range(num_plots):
        if build:
            data_for_build = data_to_plot[
                data_to_plot[binning_variable].isin(bins[: i + 1])
            ]
        else:
            data_for_build = data_to_plot

        # Set the color palette
        sns.set_palette(colors)

        if ax is None:
            fig, plotting_ax = plt.subplots(figsize=(6, 3))
        else:
            plotting_ax = ax

        # Plot all data at once for current bins
        sns.lineplot(
            ax=plotting_ax,
            data=data_for_build,
            x="note_0_hours_since_first_crh",
            y=y_var,
            hue=binning_variable,
            style=binning_variable,
            dashes=dashes,
            errorbar=("se", norm.ppf(0.975)),
        )

        # Set plot limits
        if USE_CUSTOM_YLIMS and not ylim is None:
            plotting_ax.set_ylim(*ylim)
        plotting_ax.set_xlim(*xlim)

        # Lines at x and y = 0
        plotting_ax.axhline(0, c="lightgray", alpha=1, zorder=-1000)
        plotting_ax.axvline(
            0, c=analysis_colors["other_2"], alpha=1, zorder=-1000, linestyle=":",
            linewidth=INDIVIDUAL_LINE_WIDTH
        )
        plotting_ax.axvspan(
            -100,
            0,
            color="gray",
            alpha=0.25,
            zorder=-1000,
        )

        # Label axes
        plotting_ax.set_xlabel("Hours After Note Attached", color=analysis_colors["hours"])
        if include_y_axis_label:
            if y_var == "Increase":
                plotting_ax.set_ylabel(ylabel="Pct. of Posts in Quartile\nW/ Positive Treatment Effect")
            else:
                plotting_ax.set_ylabel(ylabel="Avg. Treatment Effect\nWithin Quartile")
        else:
            plotting_ax.set_ylabel("")

        # Set major ticks
        plotting_ax.xaxis.set_major_locator(plt.MultipleLocator(12))
        plotting_ax.xaxis.set_minor_locator(plt.MultipleLocator(3))

        # Format y-axis
        plotting_ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(MetricFormatter("%" if y_var in ["bias_adjusted_relative_treatment_effect", "bias_adjusted_growth"] else ""))
        )

        # Put legend outside plot
        if binning_variable == "hours_to_slap_bin":
            legend_title = "Age of Post When Note Attached (Hours)"
        elif binning_variable == "pre_break_calculated_retweets_bin":
            legend_title = "Reposts Prior to Note Attached"
        else:
            legend_title = (
                binning_variable.replace("_bin", "")
                .replace("_short_", "")
                .replace("_", " ")
                .title()
            )
        # Place legend below plot
        if include_legend:
            plotting_ax.legend(
                loc="center",
                bbox_to_anchor=(0.5, -0.65),
                title=legend_title,
                frameon=False,
                ncol=2,
            )

            # Reverse order of legend
            if reverse_legend:
                handles, labels = ax.get_legend_handles_labels()
                legend = ax.legend(handles[::-1], labels[::-1], loc="center", title=legend_title, bbox_to_anchor=(0.5, -0.66), frameon=False,ncol=2,
                                   # labelcolor=analysis_colors["legend"],
                                   )
                # legend.get_title().set_color(analysis_colors["legend"])

        else:
            plotting_ax.get_legend().remove()

        # Add title
        plotting_ax.set_title(
            metric.replace("calculated_", "")
            .replace("impressions", "views")
            .replace("tweet", "post")
            .title()
        )

        plt.tight_layout()


        if save:
            metric_save_name = "reach" if metric == "impressions" else metric
            os.makedirs(
                output_dir / y_var / f"{metric_save_name}_bins",
                exist_ok=True,
            )

            # Get the file path, which changes based on whether or not we're building the plot out
            if build:
                save_name = (
                    output_dir
                    / y_var
                    / f"{metric_save_name}_bins"
                    / f"{binning_variable}_{i}.pdf"
                )
            else:
                save_name = (
                    output_dir
                    / y_var
                    / f"{metric_save_name}_bins"
                    / f"{binning_variable}.pdf"
                )

            plt.savefig(save_name)


def plot_percentiles(
    data_to_plot,
    metric,
    ylim,
    xlim,
    output_dir,
    ax=None,
    include_y_axis_label=True,
    save=True,
    y_var="bias_adjusted_treatment_effect",
):
    percentiles_to_plot = [
        0.5,
        0.25,
        0.75,
        0.1,
        0.9,
    ]
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    # Reverse sort
    percentiles_to_plot = sorted(percentiles_to_plot)[::-1]
    for percentile in percentiles_to_plot:
        sns.lineplot(
            data=data_to_plot,
            x="note_0_hours_since_first_crh",
            y=y_var,
            errorbar=("ci", 0.95),
            estimator=lambda x: np.percentile(x, percentile * 100),
            # label=f"{int(percentile*100)}th Percentile",
            ax=ax,
            linestyle="--",
        )

    # Set plot limits
    if USE_CUSTOM_YLIMS and not ylim is None:
        plt.ylim(*ylim)
    plt.xlim(*xlim)

    # Lines at x and y = 0
    plt.axhline(0, c="lightgray", alpha=1, zorder=-1000)
    plt.axvline(0, c="lightgray", alpha=1, zorder=-1000)

    # Label axes
    ax.set_xlabel("Hours After Note Attached", color=analysis_colors["hours"])
    if include_y_axis_label:
        ax.set_ylabel(f"Absolute Change")
    else:
        ax.set_ylabel("")

    # Set major ticks to 5hr and minor ticks to 1hr
    if ax is not None:
        ax = plt.gca()
        ax.xaxis.set_major_locator(plt.MultipleLocator(8))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(1))

    # Format y-axis
    if "pct_change" in metric:
        ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, pos: f"{int(np.round(x * 100))}%")
        )
    elif (
        "depth" in metric
        or "density" in metric
        or "transitivity" in metric
        or "wiener_index" in metric
    ):
        ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, pos: f"{round(x, 4):}")
        )
    elif metric == "like_through_rate" or "per_impression" in metric:
        ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(
                lambda x, pos: (
                    f"{int(np.round(x * 10000))}/10k" if not np.isclose(x, 0) else "0"
                )
            )
        )
    elif metric == "likes" or metric == "impressions":
        ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(
                lambda x, pos: (
                    f"{int(np.round(x / 1000))}k" if not np.isclose(x, 0) else "0"
                )
            )
        )
    else:
        ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(
                lambda x, pos: f"{int(np.round(x)):,}" if not np.isclose(x, 0) else "0"
            )
        )

    if save:
        save_name = (
            "reach_percentiles" if metric == "impressions" else f"{metric}_percentiles"
        )
        os.makedirs(output_dir / y_var / f"{metric}_percentiles", exist_ok=True)
        plt.savefig(output_dir / y_var / f"{metric}_percentiles" / f"{save_name}.pdf")
