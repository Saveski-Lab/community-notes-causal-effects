

import marimo

__generated_with = "0.13.1"
app = marimo.App()


@app.cell
def _(mo):
    mo.md("""# Main Paper""")
    return


@app.cell
def _(__file__):
    import json
    from collections import defaultdict

    import pandas as pd
    import numpy as np
    import sys
    from pathlib import Path
    import marimo as mo
    from tqdm import tqdm
    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    import matplotlib.patches as mpatches

    # Import gaussian and bootstrap CIs from scipy
    from scipy.stats import norm
    from scipy.stats import sem

    sys.path.append(str(Path(__file__).resolve().parents[2]))

    from src.analysis.plotting_utils import MetricFormatter, INDIVIDUAL_LINE_WIDTH
    from src.analysis.plotting_utils import colors as analysis_colors
    from src.utils import read_control_configs
    return (
        Path,
        defaultdict,
        json,
        mo,
        mpatches,
        norm,
        np,
        pd,
        plt,
        rcParams,
        sem,
        sns,
        tqdm,
    )


@app.cell
def _(mo):
    mo.md("""## Load datasets and set parameters""")
    return


@app.cell
def _(Path):
    data_root = Path("/data/cn_effect/public_data")
    return (data_root,)


@app.cell
def _(Path, data_root, json, np, pd, rcParams):
    CONFIG_TO_USE = "main.json"

    config_dir = Path("src/config")
    with open(config_dir / CONFIG_TO_USE) as _f:
        config = json.load(_f)

    intermediate_dir = "cn_effect_intermediate" + (
        "_dev" if config["dev"] else "_prod"
    )

    BIG_METRICS = [
        "calculated_retweets",
        "likes",
        "calculated_replies",
        "impressions",
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
    te = pd.read_parquet(
        data_root
        / "cn_effect_output/treatment_effects/main/final_treatment_effects.parquet"
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
        te["note_text_sentence_count"].round().astype(str),
    )

    for _metric in METRICS:
        te[f"rte_{_metric}"] = te[f"te_{_metric}"] / te[f"bcc_{_metric}"]

    first = te[te["note_0_hours_since_first_crh"] == 0]
    final = te[te["note_0_hours_since_first_crh"] == 48]

    nt_pairs = pd.read_parquet(data_root / "note_metadata.parquet")

    ratings = pd.read_parquet(data_root / "tweet_metadata.parquet")
    return (
        BIG_METRICS,
        CONFIDENCE,
        METRICS,
        final,
        first,
        nt_pairs,
        ratings,
        te,
    )


@app.cell
def _(data_root, pd):
    # # Load placebo TE and shuffled TE
    placebo_te = pd.read_parquet(data_root / "cn_effect_output/treatment_effects/placebo/final_treatment_effects.parquet")

    shuffled_te = pd.read_parquet(data_root / "cn_effect_output/treatment_effects/main_shuffled/final_treatment_effects.parquet")
    return placebo_te, shuffled_te


@app.cell
def _(mo):
    mo.md("""## Main Treatment Effects""")
    return


@app.cell
def _(main_effects_table):
    main_effects_table()
    return


@app.cell
def _(mo):
    mo.md("""## Factors Associated with Effects on Reposts""")
    return


@app.cell
def _(binning_table, final, first):
    binning_table(first, final, "hours_to_slap_bin", "calculated_retweets")
    return


@app.cell
def _(binning_table, final, first):
    binning_table(first, final, "hours_to_slap_bin", "likes")
    return


@app.cell
def _(binning_table, final, first):
    binning_table(first, final, "hours_to_slap_bin", "calculated_replies")
    return


@app.cell
def _(binning_table, final, first):
    binning_table(first, final, "hours_to_slap_bin", "impressions")
    return


@app.cell
def _(binning_table, final, first):
    binning_table(first, final, "partisan_lean", "calculated_retweets")
    return


@app.cell
def _(binning_table, final, first):
    binning_table(first, final, "tweet_media", "calculated_retweets")
    return


@app.cell
def _(binning_table, final, first):
    binning_table(
        first, final, "note_text_sentence_count_bin", "calculated_retweets"
    )
    return


@app.cell
def _(binning_table, final, first):
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
def _(binning_table, final, first):
    binning_table(
        first,
        final,
        "note_text_flesch_kincaid_grade_bin",
        "calculated_retweets",
    )
    return


@app.cell
def _(binning_table, final, first):
    binning_table(
        first,
        final,
        ["note_text_flesch_kincaid_grade_bin", "note_text_sentence_count_bin"],
        "calculated_retweets",
    )
    return


@app.cell
def _(distribution_table, final):
    _filtered_te = final[final["partisan_lean"] == "right"]
    distribution_table(_filtered_te, "te")
    return


@app.cell
def _(mo):
    mo.md("""#SI""")
    return


@app.cell
def _(mo):
    mo.md("""## Heterogeneity Based on Post Popularity""")
    return


@app.cell
def _(binning_table, final, first):
    binning_table(
        first, final, "pre_break_calculated_retweets_bin", "calculated_retweets"
    )
    return


@app.cell
def _(mo):
    mo.md("""## Distribution of Treatment Effects""")
    return


@app.cell
def _(binning_table, final, first):
    binning_table(first, final, "hours_to_slap_bin", "calculated_retweets")
    return


@app.cell
def _(Path, final, mpatches, np, pd, plt, sns, te):
    _ESTIMAND = "te"
    plt.clf()

    width_in_inches = 17.8 / 2.54
    height_in_inches = width_in_inches / 4

    _fig, _axs = plt.subplots(
        1,
        4,
        figsize=(width_in_inches, height_in_inches),
        sharex=False,
        sharey=False,
    )

    _descriptive_stats_by_direction_of_effect = {}

    for _i, _METRIC in enumerate(
        ["impressions", "calculated_replies", "likes", "calculated_retweets"]
    ):
        _mt_data = final.copy()

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
        label="Decrease in Metric",
        edgecolor="black",
    )

    positive_proxy = mpatches.Patch(
        facecolor="tab:blue",
        alpha=0.5,
        label="Increase in Metric",
        edgecolor="black",
    )

    _fig.tight_layout(rect=[0, 0.1, 1, 1])

    _axs[1].legend(
        handles=[
            positive_proxy,
            negative_proxy,
        ],
        loc="lower center",
        bbox_to_anchor=(1.22, -0.78),
        ncol=2,
        frameon=False,
        columnspacing=1,
        title="Effect of Note Attachment",
    )

    _figure_root = Path("cn_effect_output/paper_figures/main/")
    Path(_figure_root).mkdir(parents=True, exist_ok=True)
    _fig.savefig(_figure_root / "positive_te.pdf")
    plt.close()

    descriptive_stats_by_direction_of_effect
    return


@app.cell
def _(mo):
    mo.md("""## Placebo-in-Time Test""")
    return


@app.cell
def _(gaussian_ci, placebo_te):
    _data_at_1_hr = placebo_te[placebo_te["note_0_hours_since_first_crh"] == 0]

    for _metric in [
        "impressions",
        "calculated_replies",
        "likes",
        "calculated_retweets",
    ]:
        print(
            f"{_metric} treatment effect from placebo test: {gaussian_ci(_data_at_1_hr[f'te_{_metric}'])}"
        )

    num_te_dropped = 6757 - len(_data_at_1_hr)
    print(
        f"number of te dropped due to moving matching period up 1 hr: {num_te_dropped}"
    )
    return


@app.cell
def _(gaussian_ci, te):
    data_at_1_hr = te[te["note_0_hours_since_first_crh"] == 1]

    for _metric in [
        "impressions",
        "calculated_replies",
        "likes",
        "calculated_retweets",
    ]:
        print(
            f"{_metric} treatment effect from observed data: {gaussian_ci(data_at_1_hr[f'te_{_metric}'])}"
        )
    return


@app.cell
def _(mo):
    mo.md("""## Permutation Test""")
    return


@app.cell
def _(defaultdict, np, shuffled_te, te):
    from tqdm import trange

    B = 100000 - 1

    _metrics = [
        "impressions",
        "calculated_retweets",
        "likes",
        "calculated_replies",
    ]
    _final_te = te[te["note_0_hours_since_first_crh"] == 48]
    _final_shuffled_te = shuffled_te[
        shuffled_te["note_0_hours_since_first_crh"] == 48
    ]

    main_tids = _final_te["tweet_id"].unique()
    shuffled_tids = _final_shuffled_te["tweet_id"].unique()

    bootstrapped_ates = defaultdict(list)
    for b in trange(B):
        rng = np.random.default_rng(hash((b * 1000, 7911)) % (2**32))
        _permuted_tids = rng.choice(
            list(shuffled_tids), size=len(main_tids), replace=False
        )
        _permuted_tids = _final_shuffled_te[
            _final_shuffled_te["tweet_id"].isin(_permuted_tids)
        ]
        for _metric in _metrics:
            bootstrapped_ates[_metric].append(
                _permuted_tids[f"te_{_metric}"].mean()
            )

    for _metric in _metrics:
        bootstrapped_ates[_metric] = np.array(bootstrapped_ates[_metric])

        observed_ate = _final_te[f"te_{_metric}"].mean()

        print(
            _metric,
            "$p$=",
            (1 + (bootstrapped_ates[_metric] < observed_ate).sum()) / (B + 1),
        )
    return


@app.cell
def _(mo):
    mo.md("""## Matching on Post Text""")
    return


@app.cell
def _(data_root):
    data_root
    return


@app.cell
def _(data_root, pd):
    text_te_means = []
    _metrics = [
        "likes",
        "calculated_replies",
        "calculated_retweets",
        "impressions",
    ]
    for _embd_dim in [20, 58]:
        _alt_te = pd.read_parquet(
            data_root
            / f"cn_effect_output/treatment_effects/main_embd_{_embd_dim}/final_treatment_effects.parquet"
        )
        _final_alt_te = _alt_te[
            _alt_te["note_0_hours_since_first_crh"] == 48
        ].copy()

        _final_alt_te["embd_dim"] = _embd_dim

        _final_alt_te = _final_alt_te[[f"te_{m}" for m in _metrics] + ["embd_dim"]]

        text_te_means.append(_final_alt_te.mean())

    text_te_means = pd.concat(text_te_means, axis=1)
    text_te_means
    return


@app.cell
def _(mo):
    mo.md("""## Restricted Donor Pool""")
    return


@app.cell
def _(data_root, pd):
    alt_te_means = []
    _metrics = [
        "likes",
        "calculated_replies",
        "calculated_retweets",
        "impressions",
    ]
    for _cutoff in [0.35, 0.3, 0.25]:
        _alt_te = pd.read_parquet(
            data_root
            / f"cn_effect_output/treatment_effects/controls_above_{_cutoff}/final_treatment_effects.parquet"
        )

        _final_alt_te = _alt_te[
            _alt_te["note_0_hours_since_first_crh"] == 48
        ].copy()

        _final_alt_te["cutoff"] = _cutoff

        _final_alt_te = _final_alt_te[[f"te_{m}" for m in _metrics] + ["cutoff"]]

        alt_te_means.append(_final_alt_te.mean())

    alt_te_means = pd.concat(alt_te_means, axis=1)
    alt_te_means
    return


@app.cell
def _(mo):
    mo.md("""## Defining Treatment Status""")
    return


@app.cell
def _(final, nt_pairs):
    helpful_nt_pairs = nt_pairs[nt_pairs["ever_crh_in_48h_post_slap"]].merge(
        final["tweet_id"], how="inner"
    )

    total_treated = helpful_nt_pairs["tweet_id"].nunique()
    total_notes = helpful_nt_pairs["note_id"].nunique()
    n_per_t = helpful_nt_pairs["tweet_id"].value_counts()
    more_than_one_n = len(n_per_t[n_per_t > 1])

    print(
        f"Of the {total_treated:,} treated posts, {more_than_one_n} had more than one note found helpful in the 48 hours after initial note attachment ({(more_than_one_n/total_treated) * 100:0.1f}\%)"
    )

    print(
        f"Considering all treated posts, there were a total of {total_notes:,} notes potentially shown with the {total_treated:,} treated posts in the 48 hours following first note attachment"
    )
    return


@app.cell
def _(nt_pairs, ratings, te):
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
def _(pd, total_views_while_crh, total_views_while_not_crh):
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
def _(any_crh_time, pd, total_views_while_crh, total_views_while_not_crh):
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
def _(crh_to_ncrh, pd):
    print(
        "pct of posts that ever go CRH->NCRH:", 1 - pd.Series(crh_to_ncrh).mean()
    )
    return


@app.cell
def _(pd, time_of_crh_to_ncrh):
    print(
        "median time where CRH-> NCRH transition occurs:",
        pd.Series(time_of_crh_to_ncrh).median(),
    )
    return


@app.cell
def _(any_crh_time, crh_to_ncrh, np):
    print(
        f"""Of the {len(crh_to_ncrh):,} posts that received notes in the treatment period, {np.sum(crh_to_ncrh):,} were rated helpful at one point and unhelpful at a later point ({round(np.mean(crh_to_ncrh)*100,1)}\%). At the post level, {(np.array(any_crh_time) == 48.25).sum():,} posts had at least one note attached for the entire treatment period ({round((np.array(any_crh_time) == 48.25).mean()*100,1)}\%)."""
    )
    return


@app.cell
def _(data_root, pd, te, tqdm):
    # Calc data for cells above
    any_crh_time = []
    crh_to_ncrh = []
    total_views_while_crh = []
    total_views_while_not_crh = []
    time_of_crh_to_ncrh = []
    for _tid in tqdm(te["tweet_id"].unique()):
        te_data = pd.read_parquet(
            data_root
            / f"cn_effect_intermediate_prod/b_merged/{_tid}.parquet"
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
        time_of_crh_to_ncrh,
        total_views_while_crh,
        total_views_while_not_crh,
    )


@app.cell
def _(mo):
    mo.md("""## Classifying Post Partisanship""")
    return


@app.cell
def _(final):
    # Counts of num posts to receive each label
    labeled_total = final["partisan_lean"].notna().sum()
    final["partisan_lean"].value_counts()
    print(f"labeled total: {labeled_total}")

    print(final["partisan_lean"].value_counts())
    return


@app.cell
def _(final):
    print(final["tweet_language"].value_counts())
    non_english_total = 721 + 597 + 262
    print(f"num non-english: {non_english_total}")
    return

@app.cell
def _(binning_table, final, first):
    binning_table(
        first, final, "pre_break_calculated_retweets_bin", "calculated_retweets"
    )
    return

@app.cell
def _(binning_table, final, first):
    binning_table(first, final, "hours_to_slap_bin", "calculated_retweets")
    return


@app.cell
def _(mo):
    mo.md("""# Helper Functions""")
    return


@app.cell
def _(BIG_METRICS, CONFIDENCE, METRICS, final, first, gaussian_ci, pd):
    def main_effects_table():
        """Return a LaTeX-ready summary of treatment_effects."""
        rows = []
        ci_pct = int(CONFIDENCE * 100)

        for metric in METRICS:
            non_na_first = first[first[f"te_{metric}"].notna()].copy()
            non_na_final = final[final[f"te_{metric}"].notna()].copy()

            pct_positive = (non_na_final[f"te_{metric}"] > 0).mean()

            mean_te, lower_te, upper_te = gaussian_ci(
                non_na_final[f"te_{metric}"], CONFIDENCE
            )
            mean_t, _, _ = gaussian_ci(non_na_final[f"t_{metric}"], CONFIDENCE)
            mean_bcc = mean_t - mean_te
            lower_bcc = mean_t - upper_te
            upper_bcc = mean_t - lower_te

            mean_t0, _, _ = gaussian_ci(non_na_first[f"t_{metric}"], CONFIDENCE)
            mean_bcc0, _, _ = gaussian_ci(
                non_na_first[f"bcc_{metric}"], CONFIDENCE
            )

            pct_te = mean_te / (mean_t - mean_te)
            lower_pct_te = lower_te / (mean_t - lower_te)
            upper_pct_te = upper_te / (mean_t - upper_te)

            bcc_growth = mean_bcc - mean_bcc0
            t_growth = mean_t - mean_t0
            pct_growth = (t_growth - bcc_growth) / bcc_growth

            fmt = ",.0f" if metric in BIG_METRICS else ",.2f"

            rows.append(
                {
                    "metric": metric,
                    "Mean Treatment Value @48 Hours": f"${mean_t:{fmt}}$",
                    "Mean Control @48 Hours": f"${mean_bcc:{fmt}}$ ({ci_pct}\\% CI: [${lower_bcc:{fmt}}$, ${upper_bcc:{fmt}}$])",
                    "Mean Treatment Effect": f"${mean_te:{fmt}}$ ({ci_pct}\\% CI: [${lower_te:{fmt}}$, ${upper_te:{fmt}}$])",
                    "Percent Reduction Over Life": f"${pct_te*100:.1f}\\%$",
                    "Percent Reduction After CRH": f"${pct_growth*100:.1f}\\%$",
                    "Percent Positive TE": f"${pct_positive*100:.1f}\\%$",
                }
            )

        df = pd.DataFrame(rows)
        return df
    return (main_effects_table,)


@app.cell
def _(norm, sem):
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
def _(BIG_METRICS, METRICS, final, pd):
    def distribution_table(data, rte_or_te="rte"):
        summaries = []
        for metric in METRICS:
            non_na = data.copy()
            non_na = non_na[non_na[f"{rte_or_te}_{metric}"].notna()]

            gtz = non_na[f"{rte_or_te}_{metric}"] > 0
            gt25 = non_na[f"{rte_or_te}_{metric}"] > 0.25

            iqr = non_na[f"{rte_or_te}_{metric}"].quantile(0.75) - non_na[
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
                p: non_na[f"{rte_or_te}_{metric}"].quantile(p / 100)
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
def _(BIG_METRICS, CONFIDENCE, gaussian_ci, np, pd):
    def binning_table(
        first, final, binning_var: str, target_var: str
    ) -> pd.DataFrame:
        """
        Summarise treatment / control values, effects and percentage changes
        for each bin of *binning_var*.  Output is copy-and-paste ready for LaTeX.
        """
        rows, ci_pct = [], int(CONFIDENCE * 100)

        # iterate over matched bins in `first` and `final`
        for (b, df_final), (_, df_first) in zip(
            final.groupby(binning_var), first.groupby(binning_var)
        ):
            assert b == _
            if pd.isnull(b) or pd.isna(b) or (str(b) == "nan"):
                continue

            df_final = df_final[df_final[f"te_{target_var}"].notna()].copy()
            df_first = df_first[df_first[f"te_{target_var}"].notna()].copy()
            if df_final.empty or df_first.empty:
                continue

            n = len(df_final)

            # point & CI estimates
            mean_te, lower_te, upper_te = gaussian_ci(
                df_final[f"te_{target_var}"], CONFIDENCE
            )
            mean_t, _, _ = gaussian_ci(df_final[f"t_{target_var}"], CONFIDENCE)
            mean_t0, _, _ = gaussian_ci(df_first[f"t_{target_var}"], CONFIDENCE)

            mean_bcc = mean_t - mean_te
            lower_bcc, upper_bcc = mean_t - upper_te, mean_t - lower_te
            mean_bcc0, _, _ = gaussian_ci(
                df_first[f"bcc_{target_var}"], CONFIDENCE
            )

            # derived stats
            pct_te = mean_te / (mean_t - mean_te)
            bcc_growth = mean_bcc - mean_bcc0
            t_growth = mean_t - mean_t0
            pct_growth = (t_growth - bcc_growth) / bcc_growth
            pct_positive = (df_final[f"te_{target_var}"] > 0).mean()

            std_est = np.std(df_final[f"te_{target_var}"], ddof=1)
            cv = std_est / mean_te

            fmt = ",.0f" if target_var in BIG_METRICS else ",.5f"

            rows.append(
                {
                    "Bin": b,
                    "Obs.": f"{n:,}",
                    "Mean Treatment @48 h": f"${mean_t:{fmt}}$",
                    "Mean Control @48 h": f"${mean_bcc:{fmt}}$ ({ci_pct}\\% CI: [${lower_bcc:{fmt}}$, ${upper_bcc:{fmt}}$])",
                    "Mean Treatment Effect": f"${mean_te:{fmt}}$ ({ci_pct}\\% CI: [${lower_te:{fmt}}$, ${upper_te:{fmt}}$])",
                    "Percent Red. Over Life": f"${pct_te*100:.1f}\\%$",
                    "Percent Red. After CRH": f"${pct_growth*100:.1f}\\%$",
                    "Percent Pos. TE": f"${pct_positive*100:.1f}\\%$",
                    "Coeff. of Var.": f"${cv:.3g}$",
                }
            )

        return pd.DataFrame(rows)
    return (binning_table,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
