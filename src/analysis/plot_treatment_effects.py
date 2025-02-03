# Imports
import os
import gzip
import json
import sys
import socket
import argparse
from pathlib import Path
from copy import deepcopy
from collections import defaultdict

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import matplotlib.ticker as ticker
from pyarrow.lib import ArrowInvalid

# Add current directory to path for imports
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.utils import read_from_table, informative_merge, ConfigError, remove_urls, read_weights, read_control_configs
from src.pipeline.a_preprocess import local_data_root, shared_data_root
from src.pipeline.c_find_controls import (
    metric_parents,
    tweet_level_datasets,
    read_trt_and_ctrl,
)
from src.analysis.plotting_utils import (
    plot_individual_tweet,
    plot_overall,
    plot_bins,
    plot_percentiles,
)
from src.analysis.lexicon_based_labeling.readability import measure_readability
from src.analysis.lexicon_based_labeling.lexicon_labeling import (
    load_all_lexicons,
    process_tokens,
)

# Add headline-parser to path, which has a hyphen so can't be imported w/ relative imports from the root dir
sys.path.append(
    str(Path(__file__).resolve().parents[2] / "src" / "references" / "headline-parser")
)
from headline_parser import parse_headline

########################################################################################################################
# Set plotting params
USE_CUSTOM_YLIMS = False
individual_tes_to_plot = 0
plt.rc("font", size=14)  # General font size
plt.rc("axes", titlesize=18, labelsize=16)  # Title and axis labels
plt.rc("xtick", labelsize=14)  # X-axis tick labels
plt.rc("ytick", labelsize=14)  # Y-axis tick labels
plt.rc("legend", fontsize=6)  # Legend font size

########################################################################################################################
# Ylimits for plots

metric_ylims = {
    "likes": [-9000, 1000],
    "likes_pct_change": [-2, 0.5],
    "impressions": [-725000, 480000],
    "impressions_pct_change": [-1, 1],
    "calculated_retweets": [-1100, 200],
    "calculated_retweets_pct_change": [-1, 1],
    "calculated_replies": [-700, 250],
    "calculated_replies_pct_change": [-1, 0.5],
    "like_through_rate": [-0.0035, 0.001],
    "like_through_rate_pct_change": [-0.25, 0.25],
    "retweets": [-1100, 200],
    "retweets_pct_change": [-2, 1],
    "replies": [-700, 250],
    "replies_pct_change": [-0.5, 0.5],
    "quotes": [-150, 50],
    "quotes_pct_change": [-1, 1],
    "rt_graph_num_nodes": [-1200, 40],
    "rt_graph_num_nodes_pct_change": [-2, 4],
    "rt_graph_density": [-0.0005, 0.0005],
    "rt_graph_density_pct_change": [-5, 5],
    "rt_cascade_width": [-700, 500],
    "rt_cascade_width_pct_change": [-2, 4],
    "rt_cascade_depth": [-4, 1],
    "rt_cascade_depth_pct_change": [-0.3, 0.3],
    "rt_cascade_wiener_index": [-1, 0.1],
    "rt_cascade_wiener_index_pct_change": [-0.3, 0.3],
    "rt_graph_transitivity": [-0.01, 0.01],
    "rt_graph_transitivity_pct_change": [-5, 5],
    "reply_graph_num_nodes": [-800, 300],
    "reply_graph_num_nodes_pct_change": [-1, 0.5],
    "reply_graph_density": [-0.05, 0.05],
    "reply_graph_density_pct_change": [-5, 5],
    "reply_tree_width": [-550, 250],
    "reply_tree_width_pct_change": [-1, 0.5],
    "reply_tree_depth": [-2.2, 6],
    "reply_tree_depth_pct_change": [-0.5, 0.5],
    "reply_tree_wiener_index": [-0.75, 0.75],
    "reply_tree_wiener_index_pct_change": [-0.3, 0.3],
    "reply_graph_transitivity": [-0.01, 0.01],
    "reply_graph_transitivity_pct_change": [-5, 5],
    "rt_cascade_num_nodes_non_root_tweet": [-210, 10],
    "rt_cascade_num_nodes_non_root_tweet_pct_change": [-0.48, 0.04],
    "rt_cascade_num_nodes_root_tweet": [-210, 10],
    "rt_cascade_num_nodes_root_tweet_pct_change": [-0.48, 0.04],
}

########################################################################################################################


def load_tes_for_metrics(metrics, te_config, intermediate_dir):
    # Read in treatment effects
    te = {}

    te_config = deepcopy(te_config)

    te_config["matching_metrics"] = ",".join(te_config["matching_metrics"])
    te_config["bias_correction_missing_actions"] = ",".join(
        te_config["bias_correction_missing_actions"]
    )

    # Create the config file to use for filtering to the right weights
    unneeded_config = [
        "target_metrics",
        "backup_tweet_metrics",
        "tweet_metrics",
        "author_metrics",
        "restrict_donor_pool",
        "sample_bias_correction_controls",
    ]

    for k in unneeded_config:
        if k in te_config.keys():
            del te_config[k]

    for metric in metrics:
        te_path = (
            local_data_root / intermediate_dir / "h_treatment_effects" / f"{metric}"
        )
        if not os.path.exists(te_path):
            continue
        all_tes = os.listdir(te_path)

        # Filter out files that have been previously noted as corrupted
        all_tes = [f for f in all_tes if not "_corrupted" in f]

        all_te_df = []
        corrupted = []
        for te_fn in tqdm(all_tes, desc=f"Reading TE for metric {metric}", smoothing=0):
            try:
                all_te_df.append(read_from_table(te_path / te_fn, config=te_config))
            except ArrowInvalid as e:
                print(f"Error reading {te_fn}: {e}")
                os.rename(
                    te_path / te_fn,
                    te_path / te_fn.replace(".parquet", "_corrupted.parquet"),
                )
                corrupted.append(te_fn)

        if corrupted:
            print(f"Corrupted files for metric {metric}: {corrupted}")

        te[metric] = pd.concat(all_te_df)
        te[metric]["note_0_time_since_first_crh"] = pd.to_timedelta(
            te[metric]["note_0_time_since_first_crh"]
        )
        te[metric]["note_0_hours_since_first_crh"] = (
            te[metric]["note_0_time_since_first_crh"].dt.total_seconds() / 3600
        )

        # Calculate bias adjusted control
        te[metric]["bias_adjusted_control"] = te[metric]["treatment"] - te[metric]["bias_adjusted_treatment_effect"]

    return te


def get_trt_and_control_ids(metrics, te, weights):
    treatment_tids = {}
    control_tids = defaultdict(list)
    for metric in metrics:
        if metric not in te.keys():
            continue
        treatment_tids[metric] = te[metric]["tweet_id"].unique().tolist()
        for tid in tqdm(weights.keys(), desc=f"Getting control TIDs for {metric}"):
            control_tids[metric] += (
                weights[tid]
                .columns[[c.isdigit() and pd.notna(c) for c in weights[tid].columns]]
                .to_list()
            )
        control_tids[metric] = np.unique(control_tids[metric]).tolist()

    return treatment_tids, control_tids


def get_artifact_dir(config):
    # TODO: Make this dynamic
    artifact_dir = (
        "a_preprocess2024-09-20_18-33-38_330330"
        if config["dev"]
        else "a_preprocess2024-09-20_18-33-30_872437"
    )
    return artifact_dir


def get_metadata(
    treatment_tids,
    treatment_tweets,
    control_configs,
    intermediate_dir,
    artifact_dir,
    config,
    metrics,
    logger=None,
):
    # Read tweet creation times
    tweet_creation_times = pd.read_csv(
        local_data_root
        / intermediate_dir
        / artifact_dir
        / "a_tweet_creation_times.csv",
        dtype={"tweet_id": str},
    )[["tweet_id", "created_at"]].drop_duplicates()

    # Read in slap times
    slap_times = pd.read_csv(
        local_data_root / intermediate_dir / artifact_dir / "a_slap_times.csv",
        dtype={"tweet_id": str},
    )

    # Read in note creation times
    note_creation_times = pd.read_csv(
        local_data_root / intermediate_dir / artifact_dir / "a_note_creation_times.csv",
        dtype={"tweet_id": str, "note_id": str},
    )

    # Convert datetimes
    tweet_creation_times["created_at"] = pd.to_datetime(
        tweet_creation_times["created_at"], utc=True, format="mixed"
    )
    note_creation_times["note_created_at"] = pd.to_datetime(
        note_creation_times["note_created_at"], utc=True, format="mixed"
    )
    slap_times["first_crh"] = pd.to_datetime(
        slap_times["first_crh"], utc=True, format="mixed"
    )

    # Find first note creation time for a tweet
    first_note_times = (
        note_creation_times.groupby("tweet_id")["note_created_at"].min().reset_index()
    )

    # Calculate time between creation and slap
    tweet_metadata = informative_merge(
        slap_times,
        tweet_creation_times,
        "Slap times",
        "Tweet creation times",
        on="tweet_id",
        how="outer",
        logger=logger,
    )

    # Calculate time between creation and first note being written
    tweet_metadata = informative_merge(
        tweet_metadata,
        first_note_times,
        "Slap times + tweet creation times",
        "First note times",
        on="tweet_id",
        how="outer",
        logger=logger,
    )

    # Calculate time between creation and first note creation or first slap
    tweet_metadata["time_to_slap"] = (
        tweet_metadata["first_crh"] - tweet_metadata["created_at"]
    )
    tweet_metadata["time_to_note"] = (
        tweet_metadata["note_created_at"] - tweet_metadata["created_at"]
    )

    # Convert to hours
    tweet_metadata["hours_to_slap"] = (
        tweet_metadata["time_to_slap"].dt.total_seconds() / 3600
    )
    tweet_metadata["hours_to_note"] = (
        tweet_metadata["time_to_note"].dt.total_seconds() / 3600
    )

    # Calculate likes/impressions/retweets/etc right before slap

    # Create list for pre-break metrics
    pre_break_metrics = []

    # Create list for all treatment tids
    all_treatment_tids = []
    for metric in treatment_tids.keys():
        # Get all treatment tids
        all_treatment_tids = np.unique(
            all_treatment_tids + treatment_tids[metric]
        ).tolist()

    # Iterate over all treatment tids
    for tid in tqdm(all_treatment_tids, desc="Reading pre-break metrics"):

        # Get the metrics for the tweet
        metrics_for_tweet = treatment_tweets[tid].copy()

        # Filter to only metrics that occured during the matching period
        metrics_for_tweet = metrics_for_tweet[
            metrics_for_tweet["note_0_time_since_first_crh"]
            < -pd.to_timedelta(config["train_backdate"])
        ]

        # Sort by time since publication
        metrics_for_tweet = metrics_for_tweet.sort_values("time_since_publication")

        # Get the last metric before the break
        pre_break_ts = metrics_for_tweet.iloc[-1]

        # Create a dictionary for the tweet data
        tweet_data = {
            "tweet_id": [tid],
        }

        # Add the pre-break metrics to the dictionary
        for metric in metrics:
            if metric in pre_break_ts:
                tweet_data[f"pre_break_{metric}"] = pre_break_ts[metric]

        # Append the tweet data to the list
        pre_break_metrics.append(pd.DataFrame(tweet_data))

    # Concatenate the pre-break metrics
    pre_break_metrics = pd.concat(pre_break_metrics)
    tweet_metadata = informative_merge(
        tweet_metadata,
        pre_break_metrics,
        "Time of first slap/first note/tweet creation",
        "Metrics prior to slap",
        on="tweet_id",
        how="outer",
        logger=logger,
    )

    # Get the number of timestamps that were used for matching for each treatment tweet
    matching_metadata = []
    matching_metrics = set()
    for tid, cconfig in control_configs.items():
        matching_metadata_for_tweet = {"tweet_id": [tid]}
        for matching_metric, matching_ts in cconfig[0]["matching_timestamps"]:
            matching_metrics = matching_metrics.union({matching_metric})
            matching_metadata_for_tweet[
                f"number_of_{matching_metric}_ts_matched_on"
            ] = len(matching_ts)
        matching_metadata.append(pd.DataFrame(matching_metadata_for_tweet))
    matching_metadata = pd.concat(matching_metadata, ignore_index=True)

    # Merge with matching metadata
    tweet_metadata = informative_merge(
        tweet_metadata,
        matching_metadata,
        "Times + pre-break metrics",
        "Number of TS matched on",
        on="tweet_id",
        how="outer",
    )

    # Add bert-topics
    with gzip.open(
        shared_data_root / "hte_analysis" / "topic-models" / "k_20_output.json.gz", "rb"
    ) as f:
        k_20_output = json.load(f)
    topics = pd.DataFrame(k_20_output["tweet_topic"])
    topics.columns = ["tweet_id", "topic"]
    topics["topic"] = topics["topic"].astype(str).map(k_20_output["topic_labels"])

    tweet_metadata = informative_merge(
        tweet_metadata,
        topics,
        "Times + pre-break metrics + number of TS matched on",
        "Topics",
        on="tweet_id",
        how="left",
        logger=logger,
    )

    # Add author data
    author_data_path = local_data_root / "cn_effect_input" / "author_metrics.csv"
    author_data = pd.read_csv(
        author_data_path, dtype={"author_id": str, "tweet_id": str}
    ).drop(columns=["document.id", "..JSON", "created_at"])
    author_data["author_registration_date"] = pd.to_datetime(
        author_data["author_registration_date"], utc=True
    )
    tweet_metadata = informative_merge(
        tweet_metadata,
        author_data,
        "Times + pre-break metrics + number of TS matched on + topics",
        "Author data",
        on="tweet_id",
        how="left",
        logger=logger,
    )

    tweet_metadata["author_account_age"] = (
        tweet_metadata["created_at"] - tweet_metadata["author_registration_date"]
    ).dt.days

    other_covars = pd.read_json(
        shared_data_root / "hte_analysis" / "web_tweets_objs_parsed_min_time.json.gz",
        lines=True,
        dtype={"tweet_id": str},
    )

    # tweet_entites_media contains strictly less info than tweet_ext_entites_media
    other_covars = other_covars.drop(columns="tweet_entites_media")

    # If there's a url card title, there's a url card
    other_covars["has_url_card"] = other_covars["url_card_title"].notna()

    # if there's a url card but no media, we'll call it a photo
    # (if there's a url card but there is media, keep the media, as the media is what's shown)
    other_covars["tweet_ext_entites_media"] = np.where(
        other_covars["has_url_card"] & (other_covars["tweet_ext_entites_media"] == ""),
        "photo",
        other_covars["tweet_ext_entites_media"],
    )

    # Separate Text Only from One Image from One Video from
    other_covars["tweet_ext_entites_media"] = other_covars[
        "tweet_ext_entites_media"
    ].map(
        lambda x: {
            "": "Text Only",
            "photo": "Single Photo",
            "video": "Single Video",
            "animated_gif": "Single Video",
        }.get(x, "Multiple Photos or Videos")
    )

    # Simplify Tweet Language
    other_covars["tweet_language"] = other_covars["tweet_lang"].apply(
        lambda x: {
            "en": "English",
            "es": "Spanish",
            "ja": "Japanese",
        }.get(x, "Other")
    )

    tweet_readability_path = (
        local_data_root / intermediate_dir / "tweet_readability.parquet"
    )
    tweet_lexical_analysis_path = (
        local_data_root / intermediate_dir / "tweet_lexical_analysis.parquet"
    )
    if not os.path.exists(tweet_readability_path):
        # NB: removing some punctuation at the beginning of the tweet, and remove urls
        stripped_tweets = [t.lstrip(".,:& ") for t in other_covars["tweet_full_text"]]
        stripped_tweets = [remove_urls(t) for t in stripped_tweets]

        # Measure readability
        tweet_readability = pd.DataFrame(
            [
                measure_readability(stripped_tweets[i])
                for i in tqdm(
                    range(len(other_covars)),
                    desc="Measure tweet readability",
                    smoothing=0,
                )
            ]
        )

        # Tokenize
        tokenized_tweets = [
            parse_headline(
                remove_urls(other_covars["tweet_full_text"].iloc[i])
            )._.clf_token_texts
            for i in tqdm(
                range(len(other_covars)), desc="Tokenizing tweets", smoothing=0
            )
        ]

        # Measure valence, arousal, and dominance, etc. with lexicons
        lexicons, valued_lexicons, fields = load_all_lexicons()
        tweet_lexical_analysis = pd.DataFrame(
            [
                process_tokens(tokenized_tweets[i], lexicons, valued_lexicons, fields)
                for i in tqdm(
                    range(len(other_covars)),
                    desc="Calculating text stats with lexicons for tweets",
                    smoothing=0,
                )
            ]
        )

        # Clarify that these columns are for the tweet text
        tweet_readability = tweet_readability.add_prefix("tweet_text_")
        tweet_lexical_analysis = tweet_lexical_analysis.add_prefix("tweet_text_")

        tweet_readability.to_parquet(tweet_readability_path)
        tweet_lexical_analysis.to_parquet(tweet_lexical_analysis_path)
    else:
        tweet_readability = pd.read_parquet(tweet_readability_path)
        tweet_lexical_analysis = pd.read_parquet(tweet_lexical_analysis_path)

    # Merge to other covars
    other_covars = pd.concat(
        [other_covars, tweet_readability, tweet_lexical_analysis], axis=1
    )

    # Clean up
    other_covars = other_covars.rename(
        columns={"tweet_ext_entites_media": "tweet_media"}
    ).drop(
        columns=[
            c
            for c in other_covars.columns
            if c.startswith("url_card_")
            or c.startswith("qt_")
            or c.startswith("batch_")
            or c
            in [
                "user_verified",
                "tweet_lang",
                "tweet_full_text_note_text",
                "tweet_created_at",
                "tweet_user_id",
                "tweet_conversation_id",
                "user_id",
                "user_screen_name",
                "user_full_name",
                "user_created_at",
                "user_description",
                "user_location",
            ]
            or (c.startswith("user_") and c.endswith("_count"))
        ]
    )


    tweet_metadata = informative_merge(
        tweet_metadata,
        other_covars,
        "All other tweet metadata",
        "Tweet language + user verification status + whether tweet contains media",
        on="tweet_id",
        how="left",
        logger=logger,
    )
    # Merge to ratings data
    ratings_data = pd.read_csv(
        local_data_root / intermediate_dir / "pre_slap_ratings.csv",
        dtype={"tweet_id": str},
    )

    # Identify columns related to helpfulness ratings
    helpfulness_cols = [
        c
        for c in ratings_data.columns
        if c.startswith("total_helpful_") and c != "total_helpful_ratings"
    ]

    # Determine the dominant helpfulness type for each tweet
    ratings_data["dominant_helpfulness_type"] = ratings_data[helpfulness_cols].columns[
        ratings_data[helpfulness_cols].values.argmax(axis=1)
    ]

    # Clean up the dominant helpfulness type column
    ratings_data["dominant_helpfulness_type"] = (
        ratings_data["dominant_helpfulness_type"]
        .str.replace("total_helpful_(.*)_ratings", r"\1", regex=True)
        .str.replace("_", " ")
        .str.title()
    )

    tweet_metadata = informative_merge(
        tweet_metadata,
        ratings_data,
        "All other tweet metadata",
        "Note ratings of tweet + user ratings of note + text stats for note",
        on="tweet_id",
        how="left",
        logger=logger,
    )

    time_spent_crh = pd.read_csv(
        local_data_root
        / intermediate_dir
        / get_artifact_dir({"dev": config["dev"]})
        / "total_crh_times_post_slap.csv",
        dtype={"tweet_id": str, "total_crh_times_post_slap": float},
    )

    tweet_metadata = informative_merge(
        tweet_metadata,
        time_spent_crh,
        "All other tweet metadata",
        "Amount of time that tweet spent as CRH in 48h post slap",
        on="tweet_id",
        how="left",
        logger=logger,
    )

    # Load in LLM labels of tweets
    tweet_ratings = pd.read_csv(
        local_data_root
        / "cn_effect_input"
        / "all_english_tweets_claude-3-5-sonnet-v2@20241022_responses.csv",
        dtype=str,
    ).sample(frac=1, replace=False, random_state=155)

    # For each tweet, get the most common rating
    tweet_ratings = (
        tweet_ratings.groupby("tweet_id")
        .agg(lambda x: x.mode()[0])
        .reset_index()
        .rename(columns={"topic": "llm_topic"})
    )

    # Merge to tweet metadata
    tweet_metadata = informative_merge(
        tweet_metadata,
        tweet_ratings,
        "All other tweet metadata",
        "LLM ratings of tweets",
        on="tweet_id",
        how="left",
    )

    # Read note ratings
    note_ratings = pd.read_csv(
        local_data_root
        / "cn_effect_input"
        / "all_english_notes_claude-3-5-sonnet-v2@20241022_responses.csv",
        dtype=str,
    ).sample(frac=1, replace=False, random_state=155)

    # For each note, get the most common rating
    note_ratings = (
        note_ratings.groupby(["tweet_id", "note_full_text"])
        .agg(lambda x: x.mode()[0])
        .reset_index()
    )

    # Load how much time each note spent crh
    note_crh_times = pd.read_csv(
        local_data_root / "cn_effect_input" / "all_english_notes.csv",
        dtype={"tweet_id": str},
    )[["tweet_id", "note_full_text", "note_hours_crh"]]

    # Merge to amount of CRH times shown
    note_ratings = note_ratings.merge(note_crh_times, on=["tweet_id", "note_full_text"])

    # For each tweet ID, find the row with the highest CRH times shown
    note_ratings = note_ratings[
        note_ratings["note_hours_crh"]
        == note_ratings.groupby("tweet_id")["note_hours_crh"].transform("max")
    ]

    # Get a random note for each tweet
    note_ratings = note_ratings.groupby("tweet_id").head(1)

    # Merge to tweet metadata
    tweet_metadata = informative_merge(
        tweet_metadata,
        note_ratings,
        "All other tweet metadata",
        "LLM labels of note",
        on="tweet_id",
        how="left",
        logger=logger,
    )

    # Filter to only treatment tweets before calculating percentiles
    tweet_metadata = tweet_metadata[
        (tweet_metadata["total_crh_times_post_slap"] != 0)
        & tweet_metadata["total_crh_times_post_slap"].notna()
    ]

    # Bin variables
    columns_to_bin = (
        [
            "hours_to_slap",
            "hours_to_note",
            "author_n_followers",
            "author_account_age",
            "total_crh_times_post_slap",
        ]
        + [f"pre_break_{m}" for m in metrics]
        + [f"number_of_{m}_ts_matched_on" for m in metrics]
        + [c for c in tweet_metadata.columns if "percent_rated_" in c]
        + [c for c in tweet_metadata.columns if "total_" in c]
        + [c for c in tweet_metadata.columns if "tweet_rated_" in c]
        + [c for c in tweet_metadata.columns if c.startswith("tweet_text_")]
        + [c for c in tweet_metadata.columns if c.startswith("note_text_")]
    )

    for column in columns_to_bin:
        if column in tweet_metadata.columns:
            if "percent_rated_" in column:
                tweet_metadata[f"{column}_bin"] = pd.cut(
                    tweet_metadata[column],
                    bins=[0, 25, 50, 75, 100],
                    include_lowest=True,
                    right=True,
                    labels=["[0 — 25%]", "(25 — 50%]", "(50 — 75%]", "(75 — 100%]"],
                )
            elif column == "total_crh_times_post_slap":
                tweet_metadata[f"{column}_bin"] = pd.cut(
                    tweet_metadata[column],
                    bins=[0, 12, 24, 36, 49],
                    include_lowest=True,
                    right=True,
                    labels=["[0 — 12h]", "(12 — 24h]", "(24 — 36h]", "(36 — 48h]"],
                )
            elif "tweet_rated_" in column:
                true_value = (
                    column.replace("tweet_rated_not_misleading_", "")
                    .replace("tweet_rated_misleading_", "")
                    .replace("tweet_rated_", "")
                    .replace("_", " ")
                    .title()
                )
                false_value = (
                    "Not " + true_value
                    if "missing_important_context" in column
                    or "clearly_satire" in column
                    or "outdated_but_not" in column
                    or "factually_correct" in column
                    or "satire" in column
                    else "No " + true_value
                )
                tweet_metadata[f"{column}_bin"] = np.where(
                    tweet_metadata[column].isna(),
                    np.nan,
                    np.where(tweet_metadata[column] > 0.5, true_value, false_value),
                )
            else:
                # Define columns that should use Formatter 1
                formatter_1_columns = [
                    "like_through_rate",
                    "rt_graph_density",
                    "reply_graph_density",
                    "reply_tree_depth",
                    "reply_tree_wiener_index",
                    "rt_cascade_depth",
                    "rt_cascade_wiener_index",
                    "pct_change",
                ]

                # Define the formatters
                def interval_formatter_factory(column_name, column_min):
                    def interval_formatter(value):
                        if pd.isna(value):
                            return np.nan
                        if value.closed == "left":
                            left_bracket = "["
                            right_bracket = ")"
                        elif value.closed == "right" and not value.left < column_min:
                            left_bracket = "("
                            right_bracket = "]"
                        elif value.closed == "right":
                            left_bracket = "["
                            right_bracket = "]"
                        elif value.closed == "both":
                            left_bracket = "["
                            right_bracket = "]"
                        else:
                            left_bracket = "("
                            right_bracket = ")"
                        if np.any([c in column_name for c in formatter_1_columns]):
                            return f"{left_bracket}{round(value.left, 4) if value.left > column_min else round(column_min, 4):} — {round(value.right, 4):}{right_bracket}"
                        else:
                            return f"{left_bracket}{int(value.left):,} — {int(value.right):,}{right_bracket}"

                    return interval_formatter

                # Apply qcut and format labels based on column name
                tweet_metadata[f"{column}_bin"] = pd.qcut(
                    tweet_metadata[column], q=4, duplicates="drop"
                ).map(interval_formatter_factory(column, tweet_metadata[column].min()))

    tweet_metadata["tweet_creation_month"] = tweet_metadata["created_at"].dt.strftime(
        "%Y-%m"
    )
    tweet_metadata["note_creation_month"] = tweet_metadata[
        "note_created_at"
    ].dt.strftime("%Y-%m")

    return tweet_metadata


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

    intermediate_dir = "cn_effect_intermediate" + ("_dev" if config["dev"] else "_prod")
    output_dir = (
        local_data_root
        / "cn_effect_output"
        / "treatment_effects"
        / Path(config_path).name.replace(".json", "")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = get_artifact_dir(config)

    # Read in treatment effects
    metrics = config["target_metrics"]
    metrics += [m + "_pct_change" for m in metrics]
    metrics = sorted(metrics)

    all_metrics_have_data_aggregated = all(
        [
            os.path.exists(
                output_dir / f"{metric}_bias_adjusted_treatment_effect.parquet"
            )
            for metric in metrics
        ]
    )



    if not all_metrics_have_data_aggregated:
        te = load_tes_for_metrics(metrics, config, intermediate_dir)

        # Read in Weights
        weights = read_weights(intermediate_dir, config)

        # Get treatment and control ids
        treatment_tids, control_tids = get_trt_and_control_ids(metrics, te, weights)

        # Join back to tweet metadata
        control_tweets, treatment_tweets = read_trt_and_ctrl(
            intermediate_dir, config, logger=None
        )
        control_configs = read_control_configs(intermediate_dir, config)

        tweet_metadata = get_metadata(
            treatment_tids,
            treatment_tweets,
            control_configs,
            intermediate_dir,
            artifact_dir,
            config,
            metrics,
        )
        te_with_metadata = {
            metric: pd.merge(te[metric], tweet_metadata, on="tweet_id", how="left")
            for metric in te.keys()
        }
    else:
        te_with_metadata = {
            metric: pd.read_parquet(
                output_dir / f"{metric}_bias_adjusted_treatment_effect.parquet"
            )
            for metric in metrics
        }

    train_backdate = pd.to_timedelta(config["train_backdate"]).total_seconds() / 3600
    post_break_time = (
        pd.to_timedelta(config["post_break_min_time"]).total_seconds() / 3600
    )

    # plot
    xlims = [-train_backdate - 1, post_break_time - train_backdate]

    for metric in tqdm(te_with_metadata.keys(), desc="Plotting treatment effects"):

        for y_var in [
            "bias_adjusted_treatment_effect",
            # "unadjusted_treatment_effect"
        ]:
            data_to_plot = te_with_metadata[metric][
                te_with_metadata[metric][y_var].notna()
            ]
            data_to_plot.to_csv(output_dir / f"{metric}_{y_var}.csv", index=False)

            # plot_percentiles(data_to_plot, metric, metric_ylims[metric] if metric in metric_ylims.keys() else None,
            #                  xlims, output_dir, include_y_axis_label=True, y_var=y_var)
            plot_overall(
                data_to_plot,
                metric,
                output_dir,
                "#20A187",
                metric_ylims[metric] if metric in metric_ylims.keys() else None,
                xlims,
                save=True,
                y_var=y_var,
            )

            raw_metric = metric.replace("_pct_change", "")

            if raw_metric in ["calculated_retweets"]:
                binning_variables = [
                    v
                    for v in np.unique(
                        [
                            "note_text_flesch_kincaid_grade_bin",
                            "tweet_text_flesch_kincaid_grade_bin",
                            "tweet_text_controversial_bin",
                            "note_text_sentence_count_bin",
                            "hours_to_slap_bin",
                            "hours_to_note_bin",
                            f"pre_break_{raw_metric}_bin",
                            f"number_of_{raw_metric}_ts_matched_on_bin",
                            "pre_break_likes_bin",
                            "pre_break_impressions_bin",
                            "pre_break_calculated_retweets_bin",
                            "pre_break_calculated_replies_bin",
                            "user_blue_verified",
                            "author_account_age_bin",
                            "author_n_followers_bin",
                            "tweet_creation_month",
                            "note_creation_month",
                            "tweet_media",
                            "tweet_language",
                            "total_crh_times_post_slap_bin",
                            "topic",
                            # "percent_rated_helpful_clear_bin",
                            # "percent_rated_helpful_good_sources_bin",
                            # "percent_rated_helpful_addresses_claim_bin",
                            # "percent_rated_helpful_important_context_bin",
                            # "percent_rated_helpful_unbiased_language_bin",
                            # "percent_rated_not_helpful_bin",
                            # "percent_rated_helpful_bin",
                            # "percent_rated_somewhat_helpful_bin",
                            # "tweet_rated_misleading_factual_error_bin",
                            # "tweet_rated_misleading_manipulated_media_bin",
                            # "tweet_rated_misleading_outdated_information_bin",
                            # "tweet_rated_misleading_missing_important_context_bin",
                            # "tweet_rated_misleading_unverified_claim_as_fact_bin",
                            # "tweet_rated_misleading_satire_bin",
                        ]
                    )
                    if v in data_to_plot.columns
                ]

                for binning_variable in tqdm(
                    binning_variables,
                    desc=f"Plotting binned {y_var} for metric {metric}",
                ):
                    plot_bins(
                        data_to_plot,
                        metric,
                        output_dir,
                        "viridis",
                        [(6, 2, 1, 2), (2, 1, 1, 1), (5, 2), ""][::-1],
                        None,
                        xlims,
                        save=True,
                        y_var=y_var,
                        binning_variable=binning_variable,
                        build=binning_variable
                        in ["hours_to_slap_bin", "pre_break_calculated_retweets_bin"],
                    )

        try:
            placebo_te = te_with_metadata[metric][
                te_with_metadata[metric]["note_0_time_since_first_crh"]
                < pd.to_timedelta(0)
            ]
            last_placebo = placebo_te[
                placebo_te["note_0_time_since_first_crh"]
                == placebo_te["note_0_time_since_first_crh"].max()
            ]
            most_negatively_biased_tids = (
                last_placebo.sort_values("bias_adjusted_treatment_effect")
                .head(individual_tes_to_plot)["tweet_id"]
                .to_list()
            )

            most_positively_biased_tids = (
                last_placebo.sort_values("bias_adjusted_treatment_effect")
                .tail(individual_tes_to_plot)["tweet_id"]
                .to_list()
            )

            random_tes = np.random.choice(
                treatment_tids[metric], individual_tes_to_plot, replace=False
            ).tolist()

            for tid, file_prefix in tqdm(
                zip(
                    most_negatively_biased_tids
                    + most_positively_biased_tids
                    + random_tes,
                    ["most_negatively_biased"] * individual_tes_to_plot
                    + ["most_positively_biased"] * individual_tes_to_plot
                    + ["random"] * individual_tes_to_plot,
                ),
                desc=f"Plotting individual tweets for metric {metric}",
                total=3 * individual_tes_to_plot,
            ):
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
                    weight_filter=0.05,
                )

        except Exception as e:
            print(f"Error plotting individual tweets for {metric}: {e}")
            continue
