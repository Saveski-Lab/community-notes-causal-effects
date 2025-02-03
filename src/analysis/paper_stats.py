import marimo

__generated_with = "0.9.14"
app = marimo.App()


@app.cell
def __(mo):
    mo.md("# Calculate Statistics Used in Paper")
    return


@app.cell
def __(__file__):
    import json
    import gzip
    from collections import defaultdict
    from dataclasses import dataclass

    import pandas as pd
    import numpy as np
    import re
    import sys
    from pathlib import Path
    import marimo as mo
    from tqdm import tqdm
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from matplotlib import rcParams
    import matplotlib.patches as mpatches

    # Import gaussian and bootstrap CIs from scipy
    from scipy.stats import norm
    from scipy.stats import sem
    from scipy.stats import t
    from scipy.stats import ttest_ind
    from scipy.stats import bootstrap

    sys.path.append(str(Path(__file__).resolve().parents[2]))

    from src.analysis.plotting_utils import MetricFormatter, INDIVIDUAL_LINE_WIDTH

    from src.analysis.colors import colors as analysis_colors
    return (
        INDIVIDUAL_LINE_WIDTH,
        MetricFormatter,
        Path,
        analysis_colors,
        bootstrap,
        dataclass,
        defaultdict,
        gzip,
        json,
        mo,
        mpatches,
        norm,
        np,
        pd,
        plt,
        rcParams,
        re,
        sem,
        sns,
        sys,
        t,
        ticker,
        tqdm,
        ttest_ind,
    )


@app.cell
def __(mo):
    mo.md("""### Load Data/Set Params""")
    return


@app.cell
def __(relative_change_table):
    relative_change_table()
    return


@app.cell
def __(Path, json, np, pd, rcParams):
    CONFIG_TO_USE = "with_root_and_non_root_rts_prod.json"

    config_dir = Path("src/config")
    with open(config_dir / CONFIG_TO_USE) as _f:
        config = json.load(_f)

    BIG_METRICS = [
        "calculated_retweets",
        "likes",
        "calculated_replies",
        "impressions",
        "rt_cascade_num_nodes_root_tweet",
        "rt_cascade_num_nodes_non_root_tweet",
        "rt_cascade_width",
    ]

    SMALL_METRICS = [
        "rt_cascade_depth",
        "rt_cascade_wiener_index",
    ]

    METRICS = BIG_METRICS + SMALL_METRICS
    CONFIDENCE = 0.95

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


    # Load main TE
    te = (
        pd.read_pickle(
            "cn_effect_output/treatment_effects/with_root_and_non_root_rts_prod/final_treatment_effects.pkl.gz"
        )
        .sort_values(["tweet_id", "note_0_hours_since_first_crh"])
        .reset_index(drop=True)
    )

    te["note_text_flesch_kincaid_grade_bin"] = te[
        "note_text_flesch_kincaid_grade_bin"
    ].replace({"[-3 — 5]": "≤5", "(10 — 147]": ">10"})

    te["note_text_flesch_kincaid_grade_bin"] = te[
        "note_text_flesch_kincaid_grade_bin"
    ].replace({"[-3 — 5]": "≤5", "(10 — 147]": ">10"})

    te["note_text_sentence_count_bin"] = np.where(
        te["note_text_sentence_count"] > 3,
        "4+",
        te["note_text_sentence_count"].round().astype(int).astype(str),
    )


    for _metric in METRICS:
        te[f"rte_{_metric}"] = te[f"te_{_metric}"] / te[f"bcc_{_metric}"]


    first = te[te["note_0_hours_since_first_crh"] == 0]
    final = te[te["note_0_hours_since_first_crh"] == 48]

    # Hot and fast
    hf_first = first[
        (
            first["pre_break_calculated_retweets"]
            > first["pre_break_calculated_retweets"].median()
        )
        & (first["time_to_slap"] < first["time_to_slap"].median())
    ]

    hf_final = final[
        (
            final["pre_break_calculated_retweets"]
            > final["pre_break_calculated_retweets"].median()
        )
        & (final["time_to_slap"] < final["time_to_slap"].median())
    ]


    nt_pairs = pd.read_csv(
        "cn_effect_intermediate_prod/a_preprocess2024-07-03_09-17-57_808447/analyzed_notes_and_tweets.csv",
        dtype={"tweet_id": str, "note_id": str},
    )

    ratings = pd.read_csv(
        "cn_effect_intermediate_prod/pre_slap_ratings.csv",
        dtype={"tweet_id": str},
    )

    bad_tids_to_plot = [
        "1644491893169070080",
        "1665140803092762629",
        "1644321203295748096",
        "1664244363340161026",
    ]

    bad_tid_metrics_data = pd.read_parquet(
        "cn_effect_intermediate_prod/a_metrics.parquet",
        filters=[("tweet_id", "in", bad_tids_to_plot)],
    )


    bad_tid_metrics_data["Hour Since Post Created"] = (
        bad_tid_metrics_data["pulled_at"] - bad_tid_metrics_data["created_at"]
    ).dt.total_seconds() / 3600


    bad_tid_xmaxes = {
        "1644321203295748096": 100,
        "1644491893169070080": 75,
        "1665140803092762629": 75,
        "1664244363340161026": 55,
    }

    # Load placebo TE
    placebo_te = pd.read_csv(
        "cn_effect_output/treatment_effects/1_hr_placebo_prod/te_at_slap.csv"
    )
    return (
        BIG_METRICS,
        CONFIDENCE,
        CONFIG_TO_USE,
        METRICS,
        SMALL_METRICS,
        bad_tid_metrics_data,
        bad_tid_xmaxes,
        bad_tids_to_plot,
        config,
        config_dir,
        final,
        first,
        hf_final,
        hf_first,
        nt_pairs,
        placebo_te,
        ratings,
        te,
    )


@app.cell
def __(mo):
    mo.md("""## Factors Associated with Effects on Reposts""")
    return


@app.cell
def __(binning_table, final, first):
    binning_table(first, final, "hours_to_slap_bin", "likes")
    return


@app.cell
def __(binning_table, final, first):
    binning_table(first, final, "hours_to_slap_bin", "calculated_retweets")
    return


@app.cell
def __(binning_table, final, first):
    binning_table(first, final, "partisan_lean", "calculated_retweets")
    return


@app.cell
def __(binning_table, final, first):
    binning_table(first, final, "tweet_media", "calculated_retweets")
    return


@app.cell
def __(binning_table, final, first):
    binning_table(
        first, final, "note_text_sentence_count_bin", "calculated_retweets"
    )
    return


@app.cell
def __(binning_table, final, first):
    binning_table(
        first,
        final,
        [
            "tweet_text_flesch_kincaid_grade_bin",
            "note_text_flesch_kincaid_grade_bin",
        ],
        "calculated_retweets",
    )
    return


@app.cell
def __(binning_table, final, first):
    binning_table(
        first,
        final,
        "note_text_flesch_kincaid_grade_bin",
        "calculated_retweets",
    )
    return


@app.cell
def __(binning_table, final, first):
    binning_table(
        first,
        final,
        ["note_text_flesch_kincaid_grade_bin", "note_text_sentence_count_bin"],
        "calculated_retweets",
    )
    return


@app.cell
def __(distribution_table, final):
    _filtered_te = final[final["partisan_lean"] == "right"]
    distribution_table(_filtered_te, "te")
    return


@app.cell
def __(mo):
    mo.md("""#SI""")
    return


@app.cell
def __(mo):
    mo.md("""##1. Defining Treatment Status""")
    return


@app.cell
def __(nt_pairs, ratings, te):
    # Average number of noted posts per tweet
    print(
        "Average number of noted posts per tweet",
        nt_pairs.groupby(["tweet_id"])["note_id"].nunique().mean(),
    )

    # Average number of noted posts per tweet, for treated tweets
    print(
        "Average number of noted posts per tweet, for treated tweets",
        nt_pairs[nt_pairs["tweet_id"].isin(te["tweet_id"].unique())]
        .groupby(["tweet_id"])["note_id"]
        .nunique()
        .mean(),
    )

    # Avg number of ratings for slapped tweets
    print(
        "Avg number of ratings for slapped tweets",
        ratings[ratings["tweet_id"].isin(te["tweet_id"].unique())][
            "total_number_of_ratings"
        ].mean(),
    )
    return


@app.cell
def __(ratings, te):
    helpful_notes = ratings[
        ratings["tweet_id"].isin(te["tweet_id"].unique())
    ].copy()
    helpful_notes["note_text"] = helpful_notes["note_full_text"].str.split(
        "\n\n-------------------\n"
    )
    helpful_notes = helpful_notes.explode("note_text")
    helpful_notes["time_crh"] = (
        helpful_notes["note_text"].str.split(" hours").str[0].astype(float)
    )
    helpful_notes["completley_helpful"] = helpful_notes["time_crh"] == 48.25

    note_counts = helpful_notes.groupby("tweet_id")["time_crh"].count()

    completley_helpful_counts = helpful_notes.groupby("tweet_id")[
        "completley_helpful"
    ].sum()

    print(
        "Number of posts with more than 1 helpful note in 48h after 1st note attached:",
        (note_counts > 1).sum(),
        "\nPct",
        (note_counts > 1).mean() * 100,
    )
    print("Num helpful notes", len(helpful_notes))
    return completley_helpful_counts, helpful_notes, note_counts


@app.cell
def __(pd, total_views_while_crh, total_views_while_not_crh):
    print(
        "pct of all views for treated posts that occur while posts are helpful",
        pd.Series(total_views_while_crh).sum()
        / (
            pd.Series(total_views_while_not_crh).sum()
            + pd.Series(total_views_while_crh).sum()
        ),
    )
    return


@app.cell
def __(any_crh_time, pd, total_views_while_crh, total_views_while_not_crh):
    print(
        "Same pct but just for treated posts that go CRH->NCRH",
        pd.Series(total_views_while_crh)[pd.Series(any_crh_time) != 48.25].sum()
        / (
            pd.Series(total_views_while_not_crh)[
                pd.Series(any_crh_time) != 48.25
            ].sum()
            + pd.Series(total_views_while_crh)[
                pd.Series(any_crh_time) != 48.25
            ].sum()
        ),
    )
    return


@app.cell
def __(crh_to_ncrh, pd):
    print(
        "pct of posts that ever go CRH->NCRH:", 1 - pd.Series(crh_to_ncrh).mean()
    )
    return


@app.cell
def __(pd, time_of_crh_to_ncrh):
    print(
        "median time where CRH-> NCRH transition occurs:",
        pd.Series(time_of_crh_to_ncrh).median(),
    )
    return


@app.cell
def __(any_crh_time, crh_to_ncrh, np):
    print(
        f"""Of the {len(crh_to_ncrh):,} notes shown in the treatment period, {np.sum(crh_to_ncrh):,} were rated helpful at one point and unhelpful at a later point ({round(np.mean(crh_to_ncrh)*100,1)}\%). At the post level, {(np.array(any_crh_time) == 48.25).sum():,} posts had at least one note attached for the entire treatment period ({round((np.array(any_crh_time) == 48.25).mean()*100,1)}\%)."""
    )
    return


@app.cell
def __(pd, te, tqdm):
    # Calc data for cells above
    any_crh_time = []
    crh_to_ncrh = []
    total_views_while_crh = []
    total_views_while_not_crh = []
    time_of_crh_to_ncrh = []
    for _tid in tqdm(te["tweet_id"].unique()):
        te_data = pd.read_parquet(
            f"cn_effect_intermediate_prod/b_merged/{_tid}.parquet"
        )

        # Calc num views since last ts
        te_data["new_views"] = te_data["impressions"].diff()

        # Filter to 48h after crh
        te_data = te_data[
            (te_data["note_0_time_since_first_crh"] >= pd.Timedelta(0))
            & (te_data["note_0_time_since_first_crh"] <= pd.Timedelta("48h"))
        ]

        # Find if any notes were CRH
        _is_crh = (
            te_data[
                [
                    c
                    for c in te_data.columns
                    if "twitter" in c
                    and (te_data[c] == "CURRENTLY_RATED_HELPFUL").any()
                ]
            ]
            == "CURRENTLY_RATED_HELPFUL"
        )

        # Calc views while crh and not crh
        _new_views = te_data["new_views"][1:].to_numpy()
        _was_crh_last_ts = _is_crh.iloc[:-1].sum(axis=1).to_numpy(dtype=bool)
        _views_while_crh = (_new_views * _was_crh_last_ts).sum()
        _views_while_not_crh = (_new_views * (~_was_crh_last_ts)).sum()

        total_views_while_crh.append(_views_while_crh)
        total_views_while_not_crh.append(_views_while_not_crh)

        any_crh_time.append(0.25 * _is_crh.any(axis=1).sum())

        _any_crh = _is_crh.any(axis=1)
        _made_crh_transition = _any_crh.astype(int).diff() == -1

        crh_to_ncrh += [_made_crh_transition.any()]
        time_of_crh_to_ncrh += te_data["note_0_time_since_first_crh"][
            _made_crh_transition
        ].tolist()
    return (
        any_crh_time,
        crh_to_ncrh,
        te_data,
        time_of_crh_to_ncrh,
        total_views_while_crh,
        total_views_while_not_crh,
    )


@app.cell
def __(mo):
    mo.md("""## 2. Positive Treatment Effects""")
    return


@app.cell
def __(binning_table, final, first):
    binning_table(first, final, "hours_to_slap_bin", "calculated_retweets")
    return


@app.cell
def __(height_in_inches, mpatches, np, pd, plt, sns, te, width_in_inches):
    _TIMEPOINT = 48
    _ESTIMAND = "te"
    plt.clf()


    _fig, _axs = plt.subplots(
        1,
        4,
        figsize=(width_in_inches, height_in_inches),
        sharex=False,
        sharey=False,
    )

    _filtered_te = te[te["note_0_hours_since_first_crh"] == _TIMEPOINT].copy()

    _descriptive_stats_by_direction_of_effect = {}

    for _i, _METRIC in enumerate(
        ["impressions", "calculated_replies", "likes", "calculated_retweets"]
    ):
        _mt_data = _filtered_te.copy()

        _mt_data = _mt_data[_mt_data[f"{_ESTIMAND}_{_METRIC}"].notna()]

        _metric_name = (
            _METRIC.replace("impression", "view")
            .replace("rt_cascade_", "")
            .replace("calculated_", "")
            .replace("width", "max_breadth")
            .replace("tweet", "post")
            .replace("wiener_index", "structural_virality")
            .replace("_", " ")
            .title()
        )

        _mt_data[f"Magnitude of Effect"] = te[f"{_ESTIMAND}_{_METRIC}"].abs()

        _mt_data[f"Direction of Effect"] = np.where(
            _mt_data[f"{_ESTIMAND}_{_METRIC}"] > 0,
            f"Increase in {_metric_name}",
            f"Decrease in {_metric_name}",
        )

        _descriptive_stats_for_metric = {
            "Percent Positive": (_mt_data[f"{_ESTIMAND}_{_METRIC}"] > 0).mean()
            * 100,
            "Positive Mean:": _mt_data[_mt_data[f"{_ESTIMAND}_{_METRIC}"] > 0][
                f"{_ESTIMAND}_{_METRIC}"
            ].mean(),
            "Negative Mean": _mt_data[_mt_data[f"{_ESTIMAND}_{_METRIC}"] < 0][
                f"{_ESTIMAND}_{_METRIC}"
            ]
            .abs()
            .mean(),
        }

        for pctile in [0.5, 0.75, 0.9, 0.95, 0.99]:
            _descriptive_stats_for_metric[f"Positive {pctile*100:.0f} pctile"] = (
                _mt_data[
                    _mt_data[f"{_ESTIMAND}_{_METRIC}"] > 0
                ][f"{_ESTIMAND}_{_METRIC}"].quantile(pctile)
            )
            _descriptive_stats_for_metric[f"Negative {pctile*100:.0f} pctile"] = (
                _mt_data[_mt_data[f"{_ESTIMAND}_{_METRIC}"] < 0][
                    f"{_ESTIMAND}_{_METRIC}"
                ]
                .abs()
                .quantile(pctile)
            )

            _descriptive_stats_for_metric[f"{pctile*100:.0f} pctile neg/pos"] = (
                _descriptive_stats_for_metric[f"Negative {pctile*100:.0f} pctile"]
                / _descriptive_stats_for_metric[
                    f"Positive {pctile*100:.0f} pctile"
                ]
            )

        pctile_ratios = []
        for pctile in range(50, 100):
            pctile_ratios.append(
                _mt_data[_mt_data[f"{_ESTIMAND}_{_METRIC}"] < 0][
                    f"{_ESTIMAND}_{_METRIC}"
                ]
                .abs()
                .quantile(pctile / 100)
                / _mt_data[_mt_data[f"{_ESTIMAND}_{_METRIC}"] > 0][
                    f"{_ESTIMAND}_{_METRIC}"
                ].quantile(pctile / 100)
            )

        _descriptive_stats_for_metric["max neg/pos"] = np.max(pctile_ratios)
        _descriptive_stats_for_metric["min neg/pos"] = np.min(pctile_ratios)

        _descriptive_stats_by_direction_of_effect[_METRIC] = pd.Series(
            _descriptive_stats_for_metric
        )

        sns.histplot(
            _mt_data,
            x=f"Magnitude of Effect",
            log_scale=True,
            hue=f"Direction of Effect",
            hue_order=[
                f"Increase in {_metric_name}",
                f"Decrease in {_metric_name}",
            ],
            bins=15,
            ax=_axs[_i],
            legend=False,
        )

        # Add title
        _axs[_i].set_title(_metric_name)

    descriptive_stats_by_direction_of_effect = pd.DataFrame(
        _descriptive_stats_by_direction_of_effect
    )

    negative_proxy = mpatches.Patch(
        facecolor="tab:orange",
        alpha=0.5,
        label="Decrease in Metric\nDue to Note Attachment",
        edgecolor="black",
    )

    positive_proxy = mpatches.Patch(
        facecolor="tab:blue",
        alpha=0.5,
        label="Increase in Metric\nDue to Note Attachment",
        edgecolor="black",
    )

    _fig.tight_layout(rect=[0, 0.1, 1, 1])

    _axs[1].legend(
        handles=[
            positive_proxy,
            negative_proxy,
        ],
        loc="lower center",
        bbox_to_anchor=(1.22, -0.6),
        ncol=2,
        frameon=False,
        columnspacing=1,
    )


    _fig.savefig(
        f"/Users/is28/Desktop/TE_Distributions/figure_s2.pdf",
    )
    plt.close()

    descriptive_stats_by_direction_of_effect
    return (
        descriptive_stats_by_direction_of_effect,
        negative_proxy,
        pctile,
        pctile_ratios,
        positive_proxy,
    )


@app.cell
def __(mo):
    mo.md("""## 3. Heterogeneity Based on Post Popularity""")
    return


@app.cell
def __(binning_table, final, first):
    binning_table(
        first, final, "pre_break_calculated_retweets_bin", "calculated_retweets"
    )
    return


@app.cell
def __(mo):
    mo.md("""## 4. Classifying Post Partisanship""")
    return


@app.cell
def __(final):
    # Counts of num posts to receive each label
    labeled_total = final["partisan_lean"].notna().sum()
    final["partisan_lean"].value_counts()
    print(f"labeled total: {labeled_total}")
    return (labeled_total,)


@app.cell
def __(final, labeled_total):
    print(final["tweet_language"].value_counts())
    non_english_total = 721 + 597 + 262
    print(f"num non-english: {non_english_total}")
    no_text_total = final["tweet_full_text_x"].isna().sum()
    print(f"num w/o tweet text: {no_text_total}")
    print(f"total: {no_text_total + non_english_total + labeled_total}")
    return no_text_total, non_english_total


@app.cell
def __(mo):
    mo.md("""## 5. Placebo-in-Time Test""")
    return


@app.cell
def __(gaussian_ci, placebo_te):
    for _metric in [
        "impressions",
        "calculated_replies",
        "likes",
        "calculated_retweets",
    ]:
        print(
            f"{_metric} treatment effect from placebo test: {gaussian_ci(placebo_te[f'{_metric}_bias_adjusted_treatment_effect'])}"
        )

    num_te_dropped = 6753 - len(placebo_te)
    print(
        f"number of te dropped due to moving matching period up 1 hr: {num_te_dropped}"
    )
    return (num_te_dropped,)


@app.cell
def __(gaussian_ci, te):
    data_at_1_hr = te[te["note_0_hours_since_first_crh"] == 1]

    for _metric in [
        "impressions",
        "calculated_replies",
        "likes",
        "calculated_retweets",
    ]:
        print(
            f"{_metric} treatment effect from real data: {gaussian_ci(data_at_1_hr[f'te_{_metric}'])}"
        )
    return (data_at_1_hr,)


@app.cell
def __(mo):
    mo.md("""## 6. Anomalous Post Removal""")
    return


@app.cell
def __(
    INDIVIDUAL_LINE_WIDTH,
    MetricFormatter,
    Path,
    analysis_colors,
    bad_tid_metrics_data,
    bad_tid_xmaxes,
    bad_tids_to_plot,
    plt,
    sns,
    ticker,
):
    # Convert 17.8 cm to inches for figure size (1 inch = 2.54 cm)
    width_in_inches = 17.8 / 2.54
    height_in_inches = width_in_inches / 9 * 2 * 7 / 5

    fig, axs = plt.subplots(
        1,
        4,
        figsize=(width_in_inches, height_in_inches),
        sharex=False,
        sharey=False,
    )


    for i, tid in enumerate(bad_tids_to_plot):
        _data = bad_tid_metrics_data[
            (bad_tid_metrics_data["tweet_id"] == tid)
            & (
                bad_tid_metrics_data["Hour Since Post Created"]
                < bad_tid_xmaxes[tid]
            )
        ]

        sns.lineplot(
            data=_data,
            x="Hour Since Post Created",
            y="likes",
            ax=axs[i],
            color="black",
            linewidth=INDIVIDUAL_LINE_WIDTH,
        )

        # Format Y-Axis
        axs[i].yaxis.set_major_formatter(ticker.FuncFormatter(MetricFormatter()))

        # Remove y-axis label for all but left most plot
        if i == 0:
            axs[i].set_ylabel("Likes")
        else:
            axs[i].set_ylabel("")

        # CHange to 12 hr ticks
        axs[i].xaxis.set_major_locator(plt.MultipleLocator(24))
        axs[i].xaxis.set_minor_locator(plt.MultipleLocator(6))

        axs[i].set_xlabel(
            "Hours After Post Created", color=analysis_colors["hours"]
        )


    for i, label in enumerate(["A", "B", "C", "D"]):
        axs[i].text(
            -0.12,
            1.1,
            label,
            transform=axs[i].transAxes,
            fontweight="bold",
            va="top",
            ha="left",
            size=12,
        )

    output_dir = Path(
        "cn_effect_output/paper_figures/with_root_and_non_root_rts_prod"
    )
    fig.tight_layout()
    fig.savefig(output_dir / "figure_s1.pdf")
    return (
        axs,
        fig,
        height_in_inches,
        i,
        label,
        output_dir,
        tid,
        width_in_inches,
    )


@app.cell
def __(binning_table, final, first):
    binning_table(
        first, final, "pre_break_calculated_retweets_bin", "calculated_retweets"
    )
    return


@app.cell
def __(binning_table, final, first):
    binning_table(first, final, "hours_to_slap_bin", "calculated_retweets")
    return


@app.cell
def __(mo):
    mo.md("""## 7. Posts With Missing Metrics""")
    return


@app.cell
def __(Path, config):
    from src.pipeline.g_compare_weights import read_weights, read_control_configs
    control_configs = read_control_configs(
        Path("cn_effect_intermediate_prod"),
        config,
    )
    return control_configs, read_control_configs, read_weights


@app.cell
def __(METRICS, Path, control_configs, defaultdict, json, pd, te, tqdm):
    # Read metadata from step C
    with open(
        "cn_effect_intermediate_prod/c_find_controls/c_find_control_metadata.json"
    ) as f:
        step_c_metadata = json.load(f)

    # Get complete list of posts that were dropped from at least one metric )
    treated_used_tids = set(step_c_metadata["used_tweet_ids"])

    #Make sure we actually estimated te and could solve convex problem
    treated_used_tids = treated_used_tids.intersection(te["tweet_id"].unique())

    # Get all treatment and control ids
    all_used_tids = set(pd.read_csv(
        Path("cn_effect_intermediate_prod") /
        "a_preprocess2024-09-20_18-33-30_872437" /
        "analyzed_notes_and_tweets.csv",
        dtype=str
    )["tweet_id"].unique())

    # Find IDs that were dropped
    failed_treated_tids = set(
        step_c_metadata["tids_without_pre_break_data"] 
        + step_c_metadata["tids_without_post_break_data"]
        + step_c_metadata["tweets_without_controls"]
        + step_c_metadata["not_enough_metrics"]
        + [t for t in step_c_metadata["used_tweet_ids"] if t not in treated_used_tids]
    )


    all_used_tids = all_used_tids - failed_treated_tids

    control_tids = all_used_tids - treated_used_tids

    treatment_missing_metrics_by_tid = defaultdict(list)
    treatment_missing_metrics_by_metric = defaultdict(list)

    for _metric in METRICS + ["author_n_followers"]:
        # Make sure metric is one we specifically count
        if f"tweets_using_{_metric}" not in step_c_metadata.keys():
            print(_metric, "not used")
            continue

        used_for_metric = step_c_metadata[f"tweets_using_{_metric}"]
        missing_tids = treated_used_tids - set(used_for_metric)
        for _tid in missing_tids:
            treatment_missing_metrics_by_tid[_tid] += [_metric]
            treatment_missing_metrics_by_metric[_metric] += [_tid]


    control_present_tids_by_tid = defaultdict(set)
    control_present_tids_by_metric = defaultdict(set)

    for _tid in tqdm(list(treated_used_tids)):
        assert len(control_configs[_tid]) == 1
        _tid_metadata = control_configs[_tid][0]

        for _metric in _tid_metadata["metrics_present_for_tweet"]:
            _control_tids_used = _tid_metadata["control_tweet_ids"]
            control_present_tids_by_metric[_metric] = \
                control_present_tids_by_metric[_metric].union(
                    _control_tids_used 
                )
            for _control_tid in _control_tids_used:
                control_present_tids_by_tid[_control_tid] = \
                    control_present_tids_by_tid[_control_tid].union([_metric])

    all_metrics = set(METRICS + ["author_n_followers"])

    control_missing_metrics_by_tid = {
        _tid: all_metrics.difference(_metrics)
        for _tid, _metrics in control_present_tids_by_tid.items()
    }

    control_missing_metrics_by_metric = {
        m: control_tids.difference(_tids)
        for m, _tids in control_present_tids_by_metric.items()
    }
    return (
        all_metrics,
        all_used_tids,
        control_missing_metrics_by_metric,
        control_missing_metrics_by_tid,
        control_present_tids_by_metric,
        control_present_tids_by_tid,
        control_tids,
        f,
        failed_treated_tids,
        missing_tids,
        step_c_metadata,
        treated_used_tids,
        treatment_missing_metrics_by_metric,
        treatment_missing_metrics_by_tid,
        used_for_metric,
    )


@app.cell
def __(dfs):
    dfs.transpose()
    return


@app.cell
def __(
    control_fully_present,
    control_missing_metrics_by_tid,
    control_never_present,
    control_partially_present,
    control_tids,
    pd,
    treated_fully_present,
    treated_never_present,
    treated_partially_present,
    treated_used_tids,
    treatment_missing_metrics_by_tid,
):
    post_subsets = [
        ("treated", 
         (
             len(treated_used_tids),
             treatment_missing_metrics_by_tid, 
          treated_fully_present,
          treated_partially_present, 
          treated_never_present)
        ),
        ("control", 
         (
             len(control_tids),
             control_missing_metrics_by_tid, 
          control_fully_present, 
          control_partially_present, 
          control_never_present)
        ),
    ]

    dfs = []
    pcts_dfs = []
    for post_subset_name, \
        (total_subset_size, missing_metrics_by_tid, fully_present, partially_present, never_present) \
    in post_subsets:

        subset_df = pd.concat([
                pd.Series(
                    {
                        metric: len(tids) + total_subset_size - len(missing_metrics_by_tid)
                        for metric, tids in fully_present.items()
                    }, 
                    name=(post_subset_name,"Full")
                    ),
                pd.Series(
                    {
                        metric: len(tids)
                        for metric, tids in partially_present.items()
                    }, 
                    name=(post_subset_name,"Partial")
                    ),
                pd.Series(
                    {
                        metric: len(tids)
                        for metric, tids in never_present.items()
                    }, 
                    name=(post_subset_name,"Never")
                    ),
        ], axis=1)

        subset_df[(post_subset_name,"Total")] = subset_df.fillna(0).sum(axis=1)

        # percents_df = subset_df.copy() / subset_df[(post_subset_name,"Total")]

        dfs.append(subset_df)
        # pcts_dfs.append(percents_df)

    dfs = pd.concat(dfs, axis=1)
    # pcts_dfs = pd.concat(pcts_dfs, axis=1)
    return (
        dfs,
        fully_present,
        missing_metrics_by_tid,
        never_present,
        partially_present,
        pcts_dfs,
        post_subset_name,
        post_subsets,
        subset_df,
        total_subset_size,
    )


@app.cell
def __(
    METRICS,
    control_missing_metrics_by_tid,
    defaultdict,
    read_and_filter_tid,
    tqdm,
    treatment_missing_metrics_by_tid,
):
    def check_tids_for_data_availability(missing_metrics_by_tid):

        # Check which metrics were present, present but dropped, and never returned.
        never_present = defaultdict(list)
        partially_present = defaultdict(list)
        fully_present = defaultdict(list)

        # Read all of them
        for _tid, _missing_metrics in tqdm(missing_metrics_by_tid.items()):
            pre, post, pre_and_post = read_and_filter_tid(_tid)

            # Check if they have each metric column:
            for _metric in METRICS + ["author_n_followers"]:
                _metric_was_ever_present = (
                    _metric in pre_and_post.columns 
                    and pre_and_post[_metric].notna().any()
                )
                if not _metric_was_ever_present:
                    # Save the fact that metric was never present
                    never_present[_metric].append(_tid)

                else:    
                    if _metric in _missing_metrics:
                        # Save the fact that the metric was available at one point
                        partially_present[_metric].append(_tid)
                    else: 
                        # save the fact that the metric was fully available
                        fully_present[_metric].append(_tid)

        return fully_present, partially_present, never_present

    (
        control_fully_present, 
        control_partially_present, 
        control_never_present
    ) = check_tids_for_data_availability(control_missing_metrics_by_tid)

    (
        treated_fully_present, 
        treated_partially_present,
        treated_never_present
    ) = check_tids_for_data_availability(treatment_missing_metrics_by_tid)
    return (
        check_tids_for_data_availability,
        control_fully_present,
        control_never_present,
        control_partially_present,
        treated_fully_present,
        treated_never_present,
        treated_partially_present,
    )


@app.cell
def __(METRICS, Path, pd):
    def read_and_filter_tid(_tid):
        _tid_data = pd.read_parquet(
            Path("cn_effect_intermediate_prod") / "b_merged" / f"{_tid}.parquet"
        )

        is_control = (
            "note_0_time_since_first_crh" not in _tid_data.columns 
            or _tid_data["note_0_time_since_first_crh"].isna().all()
        )
        if is_control:
            return None, None, _tid_data
        else:
            # Filter to 12 hours prior and 48 hours after
            _tid_data = _tid_data[
                (_tid_data["note_0_time_since_first_crh"] <= pd.to_timedelta("48h"))
                & (_tid_data["note_0_time_since_first_crh"] >= pd.to_timedelta("-12h"))
            ]
            _tid_data = _tid_data[
                ["tweet_id", "note_0_time_since_first_crh"]
                + [m for m in METRICS if m in _tid_data.columns]
            ]

            # Define groups of metrics that tend to be missing together
            _METRICS_GROUPS = {
                "api": ["replies", "retweets", "likes", "impressions"],
                "structural": ["rt_cascade_width", "rt_cascade_depth", "rt_cascade_wiener_index"]
            }

            for _metric_group in ["api", "structural"]:

                # Create accumulators
                _tid_data[f"all_{_metric_group}_present"] = True
                _tid_data[f"any_{_metric_group}_present"] = False
                group_entirely_missing = True

                # Iterate through metrics in groupos
                for _metric in _METRICS_GROUPS[_metric_group]:
                    if _metric not in _tid_data.columns:
                        continue
                    else:
                        group_entirely_missing = False
                        _tid_data[f"all_{_metric_group}_present"] = (
                            _tid_data[f"all_{_metric_group}_present"] & _tid_data[_metric].notna()
                        )
                        _tid_data[f"any_{_metric_group}_present"] = (
                            _tid_data[f"any_{_metric_group}_present"] | _tid_data[_metric].notna()
                        )
                if group_entirely_missing:
                    _tid_data[f"all_{_metric_group}_present"] = False

                # Are there any rows with one metric (out of the group) but not others?
                _tid_data[f"partially_missing_{_metric_group}"] = (
                    _tid_data[f"any_{_metric_group}_present"] 
                    & ~_tid_data[f"all_{_metric_group}_present"]
                )


            _pre_slap = _tid_data[
                _tid_data["note_0_time_since_first_crh"] < pd.Timedelta(0)
            ]

            _post_slap = _tid_data[
                _tid_data["note_0_time_since_first_crh"] >= pd.Timedelta(0)
            ]

            return _pre_slap, _post_slap, _tid_data
    return (read_and_filter_tid,)


@app.cell
def __():
    # Of the X treated posts that we consider, x are not matched on the full dataset.
    return


@app.cell
def __(mo):
    mo.md("""# Random Other Stats""")
    return


@app.cell
def __(mo):
    mo.md("""## Treatment Effects""")
    return


@app.cell
def __(final, first):
    treatment_cols = [c for c in final.columns if c.startswith("t_")]
    control_cols = [c for c in final.columns if c.startswith("bcc_")]
    treatment_growth = (
        final[treatment_cols].reset_index() - first[treatment_cols].reset_index()
    )
    control_growth = (
        final[control_cols].reset_index() - first[control_cols].reset_index()
    )

    treatment_growth = treatment_growth.rename(
        columns=lambda x: x.removeprefix("t_")
    )
    control_growth = control_growth.rename(
        columns=lambda x: x.removeprefix("bcc_")
    )

    percent_change_in_growth = (treatment_growth - control_growth) / control_growth
    percent_change_in_growth
    return (
        control_cols,
        control_growth,
        percent_change_in_growth,
        treatment_cols,
        treatment_growth,
    )


@app.cell
def __(mo):
    mo.md("""## Relative Treatment Effects""")
    return


@app.cell
def __(final, first):
    final.sort_values("rte_calculated_retweets", ascending=True)[
        ["tweet_id"] + [c for c in final.columns if "calculated_retweets" in c]
    ].merge(
        first[
            [
                "tweet_id",
                "t_calculated_retweets",
                "c_calculated_retweets",
                "bcc_calculated_retweets",
            ]
        ],
        on="tweet_id",
        suffixes=("_final", "_first"),
    )
    return


@app.cell
def __(final):
    final.groupby("tweet_media")["te_calculated_retweets"].agg(
        ["mean", "count"]
    ).sort_index()
    return


@app.cell
def __(mo):
    mo.md("""# Helper Functions""")
    return


@app.cell
def __(BIG_METRICS, CONFIDENCE, METRICS, final, first, gaussian_ci, pd):
    def relative_change_table():
        summaries = []
        for metric in METRICS:
            mean_te, lower_te, upper_te = gaussian_ci(
                final[f"te_{metric}"], CONFIDENCE
            )
            mean_t, _, _ = gaussian_ci(final[f"t_{metric}"], CONFIDENCE)

            mean_bcc = mean_t - mean_te
            upper_bcc = mean_t - lower_te
            lower_bcc = mean_t - upper_te

            mean_t_at_0, _, _ = gaussian_ci(first[f"t_{metric}"], CONFIDENCE)

            mean_bcc_at_0, _, _ = gaussian_ci(first[f"bcc_{metric}"], CONFIDENCE)

            mean_te_pct_change = (mean_te) / (mean_t - mean_te)
            upper_te_pct_change = (upper_te) / (mean_t - upper_te)
            lower_te_pct_change = (lower_te) / (mean_t - lower_te)

            total_te = mean_te * len(final)
            lower_total_te = lower_te * len(final)
            upper_total_te = upper_te * len(final)

            mean_bcc_growth = mean_bcc - mean_bcc_at_0
            mean_t_growth = mean_t - mean_t_at_0

            pct_change_in_growth = (
                mean_t_growth - mean_bcc_growth
            ) / mean_bcc_growth

            if metric in BIG_METRICS:
                summaries.append(
                    {
                        "metric": metric,
                        "mean_t": f"{mean_t:.3}",
                        "mean_bcc": f"{mean_bcc:,.0f} ({int(CONFIDENCE * 100)}\% CI: [{lower_bcc:,.0f}, {upper_bcc:,.0f}])",
                        "mean_te": f"{mean_te:,.0f} ({int(CONFIDENCE * 100)}\% CI: [{lower_te:,.0f}, {upper_te:,.0f}])",
                        "mean_te_pct_change": f"{mean_te_pct_change*100:.1f}\% ({CONFIDENCE * 100:.0f}\% CI: [{lower_te_pct_change * 100:.1f}\%, {upper_te_pct_change * 100:.1f}\%])",
                        "total_te": f"{total_te:,.0f} ({int(CONFIDENCE * 100)}\% CI: [{lower_total_te:,.0f}, {upper_total_te:,.0f}])",
                        "pct_change_in_growth": f"{pct_change_in_growth*100:.1f}\%",
                    }
                )
            else:
                summaries.append(
                    {
                        "metric": metric,
                        "mean_t": f"{mean_t:,}",
                        "mean_bcc": f"{mean_bcc:,.5f} ({int(CONFIDENCE * 100)}\% CI: [{lower_bcc:,.5f}, {upper_bcc:,.5f}])",
                        "mean_te": f"{mean_te:,.5f} ({int(CONFIDENCE * 100)}\% CI: [{lower_te:,.5f}, {upper_te:,.5f}])",
                        "mean_te_pct_change": f"{mean_te_pct_change*100:.1f}\% ({CONFIDENCE * 100:.0f}\% CI: [{lower_te_pct_change * 100:.1f}\%, {upper_te_pct_change * 100:.1f}\%])",
                        "total_te": f"{total_te:,.5f} ({int(CONFIDENCE * 100)}\% CI: [{lower_total_te:,.5f}, {upper_total_te:,.5f}])",
                        "pct_change_in_growth": f"{pct_change_in_growth*100:.1f}\%",
                    }
                )
        return pd.DataFrame(summaries)



    return (relative_change_table,)


@app.cell
def __(norm, sem):
    def gaussian_ci(data, confidence=0.95):
        mean = data.dropna().mean()
        std_err = sem(data.dropna())
        z = norm.ppf(1 - (1 - confidence) / 2)
        ci = z * std_err

        lower = mean - ci
        upper = mean + ci
        return mean, lower, upper
    return (gaussian_ci,)


@app.cell
def __(BIG_METRICS, METRICS, final, pd):
    def distribution_table(data, rte_or_te="rte"):
        summaries = []
        for metric in METRICS:
            gtz = data[f"te_{metric}"] > 0
            gt25 = data[f"rte_{metric}"] > 0.25

            iqr = data[f"{rte_or_te}_{metric}"].quantile(0.75) - data[
                f"{rte_or_te}_{metric}"
            ].quantile(0.25) * (100 if rte_or_te == "rte" else 1)

            percentiles = [
                0.1,
                0.5,
                1,
                5,
                25,
                50,
                75,
                95,
                99,
                99.5,
                99.9,
            ]

            percentile_observations = {
                p: data[f"{rte_or_te}_{metric}"].quantile(p / 100)
                * (100 if rte_or_te == "rte" else 1)
                for p in percentiles
            }

            result = {
                "metric": metric,
                "pct greater than 0": f"{gtz.mean():.1%}",
                "pct greater than 25% increase": f"{gt25.mean():.1%}",
            }
            if metric in BIG_METRICS:
                result["iqr"] = f"{iqr:,.0f}"
                result.update(
                    {
                        f"percentile_{p}": f"{v:,.0f}"
                        for p, v in percentile_observations.items()
                    }
                )
            else:
                result["iqr"] = f"{iqr:,.4f}"
                result.update(
                    {
                        f"percentile_{p}": f"{v:,.4f}"
                        for p, v in percentile_observations.items()
                    }
                )
            summaries.append(result)

        return pd.DataFrame(summaries)


    distribution_table(final, "te")
    return (distribution_table,)


@app.cell
def __(BIG_METRICS, CONFIDENCE, gaussian_ci, np, pd):
    def flexible_round(number):
        if abs(number) >= 100:
            return round(number)
        elif abs(number) >= 10:
            return round(number, 1)
        else:
            return round(number, 2)

    def binning_table(first, final, binning_var, target_var):
        summaries = []

        for (bin, bin_data), (bin2, bin_data_at_0) in zip(
            final.groupby(binning_var), first.groupby(binning_var)
        ):
            bin_data = bin_data[bin_data[f"te_{target_var}"].notna()]

            bin_data_at_0 = bin_data_at_0[
                bin_data_at_0[f"te_{target_var}"].notna()
            ]

            pct_positive = (bin_data[f"te_{target_var}"] > 0).mean() * 100

            if pd.isnull(bin):
                continue

            assert bin == bin2

            if len(bin_data) == 0 or len(bin_data_at_0) == 0:
                continue

            mean_te, lower_te, upper_te = gaussian_ci(
                bin_data[f"te_{target_var}"], CONFIDENCE
            )
            mean_t, _, _ = gaussian_ci(bin_data[f"t_{target_var}"], CONFIDENCE)
            mean_t_at_0, _, _ = gaussian_ci(
                bin_data_at_0[f"t_{target_var}"], CONFIDENCE
            )

            std_estimated = np.std(bin_data[f"te_{target_var}"], ddof=1)

            mean_bcc_at_0, _, _ = gaussian_ci(
                bin_data_at_0[f"bcc_{target_var}"], CONFIDENCE
            )

            mean_bcc = mean_t - mean_te
            upper_bcc = mean_t - lower_te
            lower_bcc = mean_t - upper_te

            mean_bcc_growth = mean_bcc - mean_bcc_at_0
            mean_t_growth = mean_t - mean_t_at_0

            pct_change_in_growth = (
                mean_t_growth - mean_bcc_growth
            ) / mean_bcc_growth

            treatment_cols = [c for c in final.columns if c.startswith("t_")]
            control_cols = [c for c in final.columns if c.startswith("bcc_")]
            treatment_growth = (
                final[treatment_cols].reset_index()
                - first[treatment_cols].reset_index()
            )
            control_growth = (
                final[control_cols].reset_index()
                - first[control_cols].reset_index()
            )

            treatment_growth = treatment_growth.rename(
                columns=lambda x: x.removeprefix("t_")
            )
            control_growth = control_growth.rename(
                columns=lambda x: x.removeprefix("bcc_")
            )

            percent_change_in_growth = (
                treatment_growth - control_growth
            ) / control_growth

            mean_te_pct_change = (mean_te) / (mean_t - mean_te)
            upper_te_pct_change = (upper_te) / (mean_t - upper_te)
            lower_te_pct_change = (lower_te) / (mean_t - lower_te)

            total_te = mean_te * len(bin_data)
            lower_total_te = lower_te * len(bin_data)
            upper_total_te = upper_te * len(bin_data)

            if target_var in BIG_METRICS:
                summaries.append(
                    {
                        "binning_var": binning_var,
                        "target_var": target_var,
                        "bin": bin,
                        "n": f"{len(bin_data):,}",
                        "mean_t": f"{flexible_round(mean_t)}",
                        "mean_bcc": f"{flexible_round(mean_bcc) } ({int(CONFIDENCE * 100)}\% CI: [{flexible_round(lower_bcc) }, {flexible_round(upper_bcc) }])",
                        "mean_te": f"{flexible_round(mean_te) } ({int(CONFIDENCE * 100)}\% CI: [{flexible_round(lower_te) }, {flexible_round(upper_te) }])",
                        "mean_te_pct_change": f"{mean_te_pct_change*100:.3g}\% CI: [{lower_te_pct_change*100:.3g}\%, {upper_te_pct_change*100:.3g}\%])",
                        "total_te": f"{flexible_round(total_te)} ({int(CONFIDENCE * 100)}\% CI: [{flexible_round(lower_total_te) }, {flexible_round(upper_total_te) }])",
                        "pct_change_in_growth": f"{pct_change_in_growth*100:.3g}\%",
                        "pct_positive": f"{pct_positive*100:.3g}\%",
                        "coefficient_of_variation": f"{flexible_round(std_estimated / mean_te)}",
                    }
                )
            else:
                summaries.append(
                    {
                        "binning_var": binning_var,
                        "target_var": target_var,
                        "bin": bin,
                        "n": f"{len(bin_data):,}",
                        "mean_t": f"{flexible_round(mean_t):,}",
                        "mean_bcc": f"{mean_bcc:,.5f} ({int(CONFIDENCE * 100)}\% CI: [{lower_bcc:,.5f}, {upper_bcc:,.5f}])",
                        "mean_te": f"{flexible_round(mean_te):,.5f} ({int(CONFIDENCE * 100)}\% CI: [{lower_te:,.5f}, {flexible_round(upper_te):,.5f}])",
                        "mean_te_pct_change": f"{mean_te_pct_change*100:.1f}\%",
                        "total_te": f"{flexible_round(total_te)} ({int(CONFIDENCE * 100)}\% CI: [{lower_total_te:,.5f}, {upper_total_te:,.5f}])",
                        "pct_change_in_growth": f"{pct_change_in_growth*100:.1f}\%",
                        "pct_positive": f"{pct_positive:.3g}\%",
                        "coefficient_of_variation": f"{std_estimated / mean_te:.3g}",
                    }
            )
        return pd.DataFrame(summaries)
    return binning_table, flexible_round


if __name__ == "__main__":
    app.run()
