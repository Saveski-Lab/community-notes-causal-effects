import marimo

__generated_with = "0.9.14"
app = marimo.App(width="medium")


@app.cell
def __():
    import sys
    import os
    from pathlib import Path

    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    sys.path.append(str(Path(os.path.abspath(".")).resolve().parents[2]))
    return Path, np, os, pd, plt, sns, sys, tqdm


@app.cell
def __(pd):
    # Read annotations and metadata from disk and merge
    tweet_metadata = pd.read_json(
        "src/analysis/volatile_tweet_labeling_round_1/imgs/tids.jsonl",
        lines=True,
        dtype={"tid": str},
    )
    annotations = pd.read_csv(
        "src/analysis/volatile_tweet_labeling_round_1/annotation_output/volatile-tweet-labeling/annotated_instances.csv",
        dtype={"instance_id": str},
    )
    annotations = tweet_metadata.merge(
        annotations, left_on="text", right_on="displayed_text"
    )


    # Fill in NAs
    for c in annotations.columns:
        if ":::" in c:
            annotations[c] = ~pd.isna(annotations[c])

    # Drop annotations that were reviewed twice
    annotations = annotations.drop_duplicates(
        subset=[c for c in annotations.columns if c != "user"]
    )

    for _metric in ["likes", "retweets", "replies", "impressions"]:
        annotations[f"is_volatile_{_metric}"] = (
            (annotations["is_volatile:::Volatile"])
            & annotations[f"is_volatile_metric:::{_metric.title()}"]
        )

        # Format numbers for plotting
        annotations[f"{_metric}_biggest_pct_drop"] = (
            -annotations[f"likes_biggest_pct_drop"] * 100
        )
        annotations[f"{_metric}_biggest_pct_rise"] = (
            annotations[f"{_metric}_biggest_pct_rise"] * 100
        )
        # Format numbers for plotting
        annotations[f"{_metric}_biggest_pct_drop_per_minute"] = (
            -annotations[f"{_metric}_biggest_pct_drop_per_minute"] * 100
        )
        annotations[f"{_metric}_biggest_pct_rise_per_minute"] = (
            annotations[f"{_metric}_biggest_pct_rise_per_minute"] * 100
        )
        annotations[f"{_metric}_biggest_abs_drop"] = -annotations[
            f"{_metric}_biggest_abs_drop"
        ]
        annotations[f"{_metric}_biggest_abs_drop_per_minute"] = -annotations[
            f"{_metric}_biggest_abs_drop_per_minute"
        ]


    # Drop tweets where drops are related to dataset mismatches
    annotations = annotations[~annotations["is_volatile:::Source Mismatch"]]

    annotations["is_volatile"] = (
        annotations["is_volatile_likes"]
        | annotations["is_volatile_impressions"]
        | annotations["is_volatile_replies"]
        | annotations["is_volatile_retweets"]
    )

    # Get the biggest across all metrics
    for _criterion in [
        "biggest_abs_drop",
        "biggest_abs_rise",
        "biggest_pct_drop",
        "biggest_pct_rise",
        "biggest_abs_drop_per_minute",
        "biggest_abs_rise_per_minute",
        "biggest_pct_drop_per_minute",
        "biggest_pct_rise_per_minute",
    ]:
        annotations[_criterion] = annotations[
            [
                f"{_metric}_{_criterion}"
                for _metric in ["likes", "impressions", "retweets", "replies"]
            ]
        ].max(axis=1)

    del tweet_metadata, c, _metric
    return annotations, c, tweet_metadata


@app.cell
def __(annotations, pd):
    # Read annotations and metadata from disk and merge
    tweet_metadata2 = pd.read_json(
        "src/analysis/volatile_tweet_labeling_round_2/imgs/tids.jsonl",
        lines=True,
        dtype={"tid": str},
    )
    annotations2 = pd.read_csv(
        "src/analysis/volatile_tweet_labeling_round_2/annotation_output/volatile-tweet-labeling/annotated_instances.csv",
        dtype={"instance_id": str},
    )
    annotations2 = tweet_metadata2.merge(
        annotations2, left_on="text", right_on="displayed_text"
    )

    # Fill in NAs
    for col in annotations2.columns:
        if ":::" in col:
            annotations2[col] = ~pd.isna(annotations2[col])

    # Drop annotations that were reviewed twice
    annotations2 = annotations2.drop_duplicates(
        subset=[c for c in annotations2.columns if c != "user"]
    )

    # Filter out source mismatches, which we will deal with by dropping sources and which are not relevant here
    annotations2 = annotations2[~annotations2["is_volatile:::Source Mismatch"]]

    # Pull out volatile column
    annotations2["is_volatile"] = annotations2["is_volatile:::Volatile"]


    for _metric in ["likes", "retweets", "replies", "impressions"]:
        # Format numbers for plotting
        annotations2[f"{_metric}_biggest_pct_drop"] = (
            -annotations2[f"likes_biggest_pct_drop"] * 100
        )
        annotations2[f"{_metric}_biggest_pct_rise"] = (
            annotations2[f"{_metric}_biggest_pct_rise"] * 100
        )
        # Format numbers for plotting
        annotations2[f"{_metric}_biggest_pct_drop_per_minute"] = (
            -annotations2[f"{_metric}_biggest_pct_drop_per_minute"] * 100
        )
        annotations2[f"{_metric}_biggest_pct_rise_per_minute"] = (
            annotations2[f"{_metric}_biggest_pct_rise_per_minute"] * 100
        )
        annotations2[f"{_metric}_biggest_abs_drop"] = -annotations2[
            f"{_metric}_biggest_abs_drop"
        ]
        annotations2[f"{_metric}_biggest_abs_drop_per_minute"] = -annotations2[
            f"{_metric}_biggest_abs_drop_per_minute"
        ]

    # Get the biggest across all metrics
    for _criterion in [
        "biggest_abs_drop",
        "biggest_abs_rise",
        "biggest_pct_drop",
        "biggest_pct_rise",
        "biggest_abs_drop_per_minute",
        "biggest_abs_rise_per_minute",
        "biggest_pct_drop_per_minute",
        "biggest_pct_rise_per_minute",
    ]:
        annotations2[_criterion] = annotations2[
            [
                f"{_metric}_{_criterion}"
                for _metric in ["likes", "impressions", "retweets", "replies"]
            ]
        ].max(axis=1)

    del tweet_metadata2, col

    unioned = pd.concat([annotations, annotations2], ignore_index=True)
    return annotations2, col, tweet_metadata2, unioned


@app.cell
def __(annotations, plt, sns):
    # Plot drops in first round of annotations
    # They seem to be a decent separator
    sns.scatterplot(
        annotations,
        x="biggest_pct_drop",
        y="biggest_abs_drop",
        hue="is_volatile",
    )

    # # Log scale
    plt.xscale("log")
    plt.yscale("log")

    plt.show()
    return


@app.cell
def __(annotations, plt, sns):
    # Plot rises in first round of annotations
    # They don't seem to help as much with separation
    sns.scatterplot(
        annotations,
        x="likes_biggest_pct_rise",
        y="likes_biggest_abs_rise",
        hue="is_volatile_likes",
    )

    plt.xscale("log")
    plt.yscale("log")
    plt.show()
    return


@app.cell
def __(plt, sns, unioned):
    # Plot drops in second round of annotations
    # They still seem to be a decent separator
    sns.scatterplot(
        unioned,
        x="biggest_pct_drop",
        y="biggest_abs_drop",
        hue="is_volatile",
    )

    plt.xscale("log")
    plt.yscale("log")
    plt.show()
    return


@app.cell
def __(plt, sns, unioned):
    # Plot drops in second round of annotations
    # They still seem to be a decent separator
    sns.scatterplot(
        unioned,
        x="biggest_abs_drop",
        y="biggest_pct_drop",
        hue="is_volatile",
    )

    plt.xscale("log")
    plt.yscale("log")
    plt.show()
    return


@app.cell
def __(plt, sns, unioned):
    # Plot drops in second round of annotations
    # They still seem to be a decent separator
    sns.scatterplot(
        unioned,
        x="likes_biggest_pct_drop",
        y="likes_biggest_abs_drop_per_minute",
        hue="is_volatile_likes",
    )

    plt.xscale("log")
    plt.yscale("log")
    plt.show()
    return


@app.cell
def __():
    def measure_f1(_preds, _true_labels, print_results=True):
        # Calculate performance metrics
        _TP = ((_preds == 1) & (_true_labels == 1)).sum()  # True Positives
        _TN = ((_preds == 0) & (_true_labels == 0)).sum()  # True Negatives
        _FP = ((_preds == 1) & (_true_labels == 0)).sum()  # False Positives
        _FN = ((_preds == 0) & (_true_labels == 1)).sum()  # False Negatives

        # Accuracy
        _accuracy = (_TP + _TN) / (_TP + _TN + _FP + _FN)

        # Precision, Recall, F1-Score
        _precision = _TP / (_TP + _FP) if (_TP + _FP) > 0 else 0
        _recall = _TP / (_TP + _FN) if (_TP + _FN) > 0 else 0
        _f1_score = (
            2 * (_precision * _recall) / (_precision + _recall)
            if (_precision + _recall) > 0
            else 0
        )

        if print_results:
            # Print performance metrics
            print(f"False Positives: {_FP}")
            print(f"False Negatives: {_FN}")
            print(f"True Positives: {_TP}")
            print(f"True Negatives: {_TN}")
            print(f"Accuracy: {_accuracy:.4f}")
            print(f"Precision: {_precision:.4f}")
            print(f"Recall: {_recall:.4f}")
            print(f"F1 Score: {_f1_score:.4f}")
        else:
            return _f1_score
    return (measure_f1,)


@app.cell
def __():
    # Get unique values for each variable to iterate over
    values_to_try = {
        "abs": [
            0,
            1,
            2,
            5,
            10,
            25,
            50,
            100,
            150,
            250,
            300,
            400,
            500,
            1000,
            1500,
            2000,
        ],
        "pct": [0, 0.05, 0.1, 0.5, 1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50],
    }
    abs_values = values_to_try["abs"]
    pct_values = values_to_try["pct"]

    # Total number of iterations for the progress bar
    total_iterations = len(abs_values) * len(pct_values)
    return abs_values, pct_values, total_iterations, values_to_try


@app.cell
def __(abs_values, np, pct_values, total_iterations, tqdm, unioned):
    # Initialize best cutoffs and minimum FP/FN
    best_cutoffs_fp = None
    _min_fp = np.inf
    _min_fn = np.inf


    with tqdm(total=total_iterations, desc="Processing Cutoffs") as _pbar:
        # Iterate over possible cutoff values
        for _rise_cutoff in abs_values:
            for _pct_rise_cutoff in pct_values:
                # Classify based on the current cutoffs
                _predicted = (
                    (unioned["biggest_abs_rise"] > _rise_cutoff)
                    & (unioned["biggest_abs_drop"] > _rise_cutoff)
                    & (unioned["biggest_pct_rise"] > _pct_rise_cutoff)
                    & (unioned["biggest_pct_drop"] > _pct_rise_cutoff)
                ).astype(int)

                # Calculate FP and FN
                _fp = ((_predicted == 1) & (unioned["is_volatile"] == 0)).sum()
                _fn = ((_predicted == 0) & (unioned["is_volatile"] == 1)).sum()

                if best_cutoffs_fp is not None:
                    _is_more_conservative = (
                        _rise_cutoff >= best_cutoffs_fp[0]
                        and _rise_cutoff >= best_cutoffs_fp[1]
                        and _pct_rise_cutoff >= best_cutoffs_fp[2]
                        and _pct_rise_cutoff >= best_cutoffs_fp[3]
                    )
                else:
                    _is_more_conservative = True

                # Update best cutoffs if necessary
                if (
                    _fp < _min_fp
                    or (_fp == _min_fp and _fn < _min_fn)
                    or (
                        _fp == _min_fp and _fn == _min_fn and _is_more_conservative
                    )
                ):
                    _min_fp = _fp
                    _min_fn = _fn
                    best_cutoffs_fp = (
                        _rise_cutoff,
                        _rise_cutoff,
                        _pct_rise_cutoff,
                        _pct_rise_cutoff,
                    )

                # Update the progress bar after each innermost loop iteration
                _pbar.update(1)
    return (best_cutoffs_fp,)


@app.cell
def __(best_cutoffs_fp, measure_f1, unioned):
    print(best_cutoffs_fp)

    # Use the optimal cutoffs to classify
    _preds = (
        (unioned["biggest_abs_rise"] > best_cutoffs_fp[0])
        & (unioned["biggest_abs_drop"] > best_cutoffs_fp[1])
        & (unioned["biggest_pct_rise"] > best_cutoffs_fp[2])
        & (unioned["biggest_pct_drop"] > best_cutoffs_fp[3])
    ).astype(int)

    # Ground truth
    _true_labels = unioned["is_volatile"]

    measure_f1(_preds, _true_labels, print_results=True)
    return


if __name__ == "__main__":
    app.run()
