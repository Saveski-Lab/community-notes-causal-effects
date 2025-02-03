import json
import os
import sys
from pathlib import Path

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add current directory to path for imports
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.pipeline.a_preprocess import local_data_root

PERMISSIVE_PCT_CUTOFF = 0.01
PERMISSIVE_ABS_CUTOFF = 10

CONSERVATIVE_PCT_MIN = 0.005
CONSERVATIVE_PCT_MAX = 0.03
CONSERVATIVE_ABS_MIN = 20
CONSERVATIVE_ABS_MAX = 100


def plot_tweet(tid, metrics_data, biggest_change, save_dir, metrics=["likes"]):
    metrics_for_tid = metrics_data[metrics_data["tweet_id"] == tid]

    num_metrics = len(metrics)
    fig, axes = plt.subplots(
        2, num_metrics, figsize=(6 * num_metrics, 8), sharex=False, sharey=False
    )

    if num_metrics == 1:
        axes = [[axes[0]], [axes[1]]]

    for i, metric in enumerate(metrics):
        if metric not in metrics_for_tid.columns:
            print(f"Metric '{metric}' not found in DataFrame columns")
            continue

        # Plot full history
        sns.lineplot(
            data=metrics_for_tid[metrics_for_tid[metric].notna()],
            x="pulled_at",
            y=metric,
            hue="source",
            alpha=0.5,
            style="source",
            ax=axes[0][i],
        )
        axes[0][i].set_title(f"{tid} - {metric} (Full History)")
        axes[0][i].set_xlabel("pulled_at")
        axes[0][i].set_ylabel(metric)
        axes[0][i].tick_params(axis="x", labelrotation=30)  # Rotate y-axis ticks
        for label in axes[0][i].get_xticklabels():
            label.set_ha("right")

        # Zoomed in plot
        biggest_change_time = metrics_for_tid[
            metrics_for_tid[f"{metric}_pct_change"].abs()
            == metrics_for_tid[f"{metric}_pct_change"].abs().max()
        ]["pulled_at"]
        if len(biggest_change_time) == 0:
            continue
        else:
            biggest_change_time = biggest_change_time.iloc[0]
        one_day_prior = biggest_change_time + pd.Timedelta(days=-1)
        one_day_post = biggest_change_time + pd.Timedelta(days=1)
        sns.lineplot(
            data=metrics_for_tid[
                (metrics_for_tid[metric].notna())
                & (metrics_for_tid["pulled_at"] >= one_day_prior)
                & (metrics_for_tid["pulled_at"] <= one_day_post)
            ],
            x="pulled_at",
            y=metric,
            hue="source",
            alpha=0.5,
            style="source",
            ax=axes[1][i],
        )
        axes[1][i].set_title(f"{tid} - {metric} (2 Days Around Big Drop)")
        axes[1][i].set_xlabel("pulled_at")
        axes[1][i].set_ylabel(metric)
        axes[1][i].tick_params(axis="x", labelrotation=20)  # Rotate y-axis ticks
        for label in axes[1][i].get_xticklabels():
            label.set_ha("right")

    plt.tight_layout()
    # plt.show()
    plt.savefig(save_dir / f"{tid}.jpg")

    with open(save_dir / f"tids.jsonl", "a") as f:
        json.dump(
            {
                **{
                    "tid": tid,
                    "text": f"<img src='imgs/{tid}.jpg' width='1200' height='450'>",
                },
                **{
                    f"{metric}_biggest_abs_drop": biggest_change.loc[
                        tid, (f"{metric}_diff", "min")
                    ]
                    for metric in metrics
                },
                **{
                    f"{metric}_biggest_abs_rise": biggest_change.loc[
                        tid, (f"{metric}_diff", "max")
                    ]
                    for metric in metrics
                },
                **{
                    f"{metric}_biggest_pct_drop": biggest_change.loc[
                        tid, (f"{metric}_pct_change", "min")
                    ]
                    for metric in metrics
                },
                **{
                    f"{metric}_biggest_pct_rise": biggest_change.loc[
                        tid, (f"{metric}_pct_change", "max")
                    ]
                    for metric in metrics
                },
                **{
                    f"{metric}_biggest_abs_drop_per_minute": biggest_change.loc[
                        tid, (f"{metric}_diff_per_minute", "min")
                    ]
                    for metric in metrics
                },
                **{
                    f"{metric}_biggest_abs_rise_per_minute": biggest_change.loc[
                        tid, (f"{metric}_diff_per_minute", "max")
                    ]
                    for metric in metrics
                },
                **{
                    f"{metric}_biggest_pct_drop_per_minute": biggest_change.loc[
                        tid, (f"{metric}_pct_change_per_minute", "min")
                    ]
                    for metric in metrics
                },
                **{
                    f"{metric}_biggest_pct_rise_per_minute": biggest_change.loc[
                        tid, (f"{metric}_pct_change_per_minute", "max")
                    ]
                    for metric in metrics
                },
            },
            f,
        )
        f.write("\n")

    plt.close()


if __name__ == "__main__":
    # Read in raw data
    raw_metrics = pd.read_parquet(
        local_data_root / "cn_effect_intermediate_prod" / "a_raw_metrics.parquet"
    ).sort_values(["tweet_id", "pulled_at"])

    # Calculate heuristic metrics for each tweet, for each metric
    for metric in ["likes", "retweets", "replies", "impressions"]:
        # Get time change in minutes
        raw_metrics["time_diff"] = raw_metrics.groupby(["tweet_id"])[
            "pulled_at"
        ].diff() / pd.Timedelta(minutes=1)

        # Calculate pct change from one timestamp to the next
        raw_metrics[f"{metric}_pct_change"] = raw_metrics.groupby(["tweet_id"])[
            metric
        ].pct_change(fill_method=None)

        # Calculate difference from one timestamp to the next
        raw_metrics[f"{metric}_diff"] = raw_metrics.groupby(["tweet_id"])[metric].diff()

        # Normalize by time
        raw_metrics[f"{metric}_pct_change_per_minute"] = (
            raw_metrics[f"{metric}_pct_change"] / raw_metrics["time_diff"]
        )
        raw_metrics[f"{metric}_diff_per_minute"] = (
            raw_metrics[f"{metric}_diff"] / raw_metrics["time_diff"]
        )

        # Create boolean for whether both cutoffs have been met
        raw_metrics[f"permissive_volatile_{metric}"] = (
            raw_metrics[f"{metric}_pct_change"].abs() > PERMISSIVE_PCT_CUTOFF
        ) & (raw_metrics[f"{metric}_diff"].abs() > PERMISSIVE_ABS_CUTOFF)

        raw_metrics[f"conservative_rise_threshold_met_{metric}"] = (
            (raw_metrics[f"{metric}_pct_change"] >= CONSERVATIVE_PCT_MIN)
            & (raw_metrics[f"{metric}_pct_change"] < CONSERVATIVE_PCT_MAX)
            & (raw_metrics[f"{metric}_diff"] >= CONSERVATIVE_ABS_MIN)
            & (raw_metrics[f"{metric}_diff"] < CONSERVATIVE_ABS_MAX)
        )

        raw_metrics[f"conservative_drop_threshold_met_{metric}"] = (
            (-raw_metrics[f"{metric}_pct_change"] >= CONSERVATIVE_PCT_MIN)
            & (-raw_metrics[f"{metric}_pct_change"] < CONSERVATIVE_PCT_MAX)
            & (-raw_metrics[f"{metric}_diff"] >= CONSERVATIVE_ABS_MIN)
            & (-raw_metrics[f"{metric}_diff"] < CONSERVATIVE_ABS_MAX)
        )

    # Calculate the biggest drop for each tweet
    biggest_change = raw_metrics.groupby("tweet_id").agg(
        {
            **{
                k: ["min", "max"]
                for k in [
                    "likes_pct_change",
                    "likes_diff",
                    "impressions_pct_change",
                    "impressions_diff",
                    "retweets_pct_change",
                    "retweets_diff",
                    "replies_pct_change",
                    "replies_diff",
                    "likes_pct_change_per_minute",
                    "likes_diff_per_minute",
                    "impressions_pct_change_per_minute",
                    "impressions_diff_per_minute",
                    "retweets_pct_change_per_minute",
                    "retweets_diff_per_minute",
                    "replies_pct_change_per_minute",
                    "replies_diff_per_minute",
                ]
            },
            **{
                k: "any"
                for k in [
                    "permissive_volatile_likes",
                    "permissive_volatile_impressions",
                    "permissive_volatile_retweets",
                    "permissive_volatile_replies",
                    "conservative_drop_threshold_met_likes",
                    "conservative_drop_threshold_met_impressions",
                    "conservative_drop_threshold_met_retweets",
                    "conservative_drop_threshold_met_replies",
                    "conservative_rise_threshold_met_likes",
                    "conservative_rise_threshold_met_impressions",
                    "conservative_rise_threshold_met_retweets",
                    "conservative_rise_threshold_met_replies",
                ]
            },
        }
    )

    save_dir = (
        local_data_root
        / "cn_effect_intermediate_prod"
        / "volatile_tweet_plots"
        / "round_1"
    )
    os.makedirs(save_dir, exist_ok=True)
    biggest_change.to_csv(save_dir / "change_sizes.csv")

    # Filter to only volatile tweets
    permissive_subset = biggest_change[
        biggest_change[
            [
                ("permissive_volatile_likes", "any"),
                ("permissive_volatile_impressions", "any"),
                ("permissive_volatile_retweets", "any"),
                ("permissive_volatile_replies", "any"),
            ]
        ].any(axis=1)
    ]

    # Filter to only observations from these tweets
    permissive_raw_metrics = raw_metrics[
        raw_metrics["tweet_id"].isin(permissive_subset.index)
    ]

    print(f"Found {len(permissive_subset):,} volatile tweets")

    np.random.seed(6981)
    for tid in tqdm(np.random.choice(permissive_subset.index, 300, replace=False)):
        plot_tweet(
            tid,
            permissive_raw_metrics,
            permissive_subset,
            save_dir,
            ["likes", "impressions", "retweets", "replies"],
        )

    save_dir = (
        local_data_root
        / "cn_effect_intermediate_prod"
        / "volatile_tweet_plots"
        / "round_2"
    )
    os.makedirs(save_dir, exist_ok=True)

    for metric in ["likes", "retweets", "replies", "impressions"]:
        biggest_change[f"conservative_volatile_{metric}"] = (
            biggest_change[f"conservative_rise_threshold_met_{metric}"]
            & biggest_change[f"conservative_drop_threshold_met_{metric}"]
        )

    conservative_subset = biggest_change[
        biggest_change[
            [
                "conservative_volatile_likes",
            ]
        ].any(axis=1)
    ]

    # Filter to only observations from these tweets
    conservative_raw_metrics = raw_metrics[
        raw_metrics["tweet_id"].isin(conservative_subset.index)
    ]

    np.random.seed(7166)
    for tid in tqdm(np.random.choice(conservative_subset.index, 300, replace=False)):
        plot_tweet(
            tid,
            conservative_raw_metrics,
            conservative_subset,
            save_dir,
            ["likes", "impressions", "retweets", "replies"],
        )
