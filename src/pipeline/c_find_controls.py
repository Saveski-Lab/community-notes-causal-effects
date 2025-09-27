import json
import os
import sys
import argparse
import socket
import uuid
import random
from collections import defaultdict
from pathlib import Path
from copy import deepcopy
from joblib import Parallel, delayed, dump, load
from itertools import chain

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add current directory to path for imports
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.utils import (
    get_logger,
    read_control_configs,
    check_run_type,
    clear_and_write,
    ConfigError,
    read_from_table,
    data_root,
)

# This script takes in the merged data. For each treatment tweet, it creates a json that
# contains the treatment tweet and all control tweets that have enough data to be used in the
# synthetic control.

########################################################################################################

NUM_THREADS = 6

author_path = data_root / "tweet_metadata.parquet"

tweet_seeds_path = data_root / "__tweet_seeds.json"
TWEET_SEEDS = json.load(open(tweet_seeds_path))

########################################################################################################

metric_parents = {
    "replies": "metrics",
    "retweets": "metrics",
    "likes": "metrics",
    "impressions": "metrics",
    "calculated_replies": "calculated_replies",
    "calculated_retweets": "calculated_retweets",
    "author_n_followers": "author",
    "rt_cascade_width": "rt_cascade",
    "rt_cascade_depth": "rt_cascade",
    "rt_cascade_wiener_index": "rt_cascade"
}
for embd_dim in [20, 58]:
    for i in range(embd_dim):
        metric_parents[f"PCA_{embd_dim}_embd_{i}"] = "author"

tweet_level_datasets = [
    "metrics",
    "calculated_retweets",
    "calculated_replies",
    "rt_graph",
    "reply_graph",
    "reply_tree",
    "rt_cascade",
]


########################################################################################################


def get_prediction_df(
    treatment_id, tweet_data, metric_of_interest, control_df, metric, log_name=None
):
    """
    For a given treatment tweet, get a dataframe of control tweets with enough data
    so that we can use them to create a synthetic control.
    """
    prediction_logger = get_logger(
        dir=data_root / intermediate_dir,
        log_name=log_name,
        enqueue=True,
    )

    # Make sure that metric is present for this tweet, which is not always true for
    # graph-based metrics
    if metric_of_interest not in tweet_data.columns:
        return pd.DataFrame()

    # Filter to only columns of interest
    treatment_df = tweet_data[
        [
            "time_since_publication",
            metric_of_interest,
            "note_0_time_since_first_crh",
        ]
    ]
    treatment_df.set_index("time_since_publication", inplace=True)
    treatment_df = treatment_df.rename(columns=lambda x: f"treatment_{x}")

    # Filter treatment df to not include history that goes back further than our maximum pre-break time
    treatment_df = treatment_df[
        treatment_df["treatment_note_0_time_since_first_crh"]
        >= -(
            pd.Timedelta(config["train_backdate"])
            + pd.Timedelta(config["pre_break_max_time"])
        )
    ]

    # Find which observations occurred between data collection
    # and time_after_break for the treatment note (which is how long we'll predict for)
    required_data_period = treatment_df["treatment_note_0_time_since_first_crh"] <= (
        -pd.Timedelta(config["train_backdate"])
        + pd.Timedelta(config["post_break_min_time"])
    )

    # Filter down to only these observations
    treatment_df = treatment_df[required_data_period]

    # Filter out NAs
    treatment_df = treatment_df[treatment_df[f"treatment_{metric_of_interest}"].notna()]

    if not (treatment_df.index.diff()[1:] == pd.Timedelta(config["time_freq"])).all():
        prediction_logger.warning(
            f"Time series for tweet {treatment_id} metric {metric_of_interest} is not "
            f"continuous, and missing some timestamps"
        )

    # Get relevant timestamps in treatment df
    control_df = (
        control_df[control_df.index.isin(treatment_df.index)]
        .reset_index()
        .pivot(index="time_since_publication", columns="tweet_id", values=metric)
        .rename(columns=lambda x: f"control_{metric}_{x}")
    )

    # Merge trt/control
    complete_df = treatment_df.merge(
        control_df, left_index=True, right_index=True, how="left"
    )

    # Find control notes that have NAs in this time period
    includes_na = complete_df.isna().any(axis=0)

    # Drop those control notes
    cols_to_drop = complete_df.columns[includes_na]
    complete_df = complete_df.drop(columns=cols_to_drop)

    return complete_df


def save_matching_data(tweet_id, tweet_matching_data, config):
    for k in config.keys():
        tweet_matching_data[k] = config[k]

    for k in tweet_matching_data.keys():
        if type(tweet_matching_data[k]) is pd.Timestamp:
            tweet_matching_data[k] = tweet_matching_data[k].strftime(
                "%Y-%m-%dT%H:%M:%S.%fZ"
            )

    str_config = deepcopy(config)

    for k in str_config.keys():
        if type(str_config[k]) is pd.Timestamp:
            str_config[k] = str_config[k].strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    clear_and_write([tweet_matching_data], save_dir / f"{tweet_id}.json.gz", str_config)


def required_number_of_ts(config):
    # Find how many timestamps we're supposed to have after the break
    required = {}
    required["post_break"] = len(
        pd.timedelta_range(
            -pd.Timedelta(config["train_backdate"]),
            (
                pd.Timedelta(config["post_break_min_time"])
                - pd.Timedelta(config["train_backdate"])
            ),
            freq=config["time_freq"],
        )
    )

    # Find out how many observations there are supposed to be pre-break
    required["pre_break"] = (
        len(
            pd.timedelta_range(
                pd.to_timedelta(config["train_backdate"]),
                pd.Timedelta(config["pre_break_min_time"])
                + pd.to_timedelta(config["train_backdate"]),
                freq=config["time_freq"],
            )
        )
        - 1
    )

    return required


def enough_treatment_data(
    tweet_data: pd.DataFrame, config: dict
) -> tuple[bool, bool, bool]:
    """
    Check that we have enough data post break for this tweet
    Start by finding all post-break data

    Arguments
    tweet_data: DataFrame of tweet data
    config: config dictionary

    Returns
    correct_num_post_break_ts: Tuple[bool]
    """
    # Check that we have enough data post break for this tweet
    # Start by finding all post-break data
    post_break_metrics = tweet_data[
        (
            (
                tweet_data["note_0_time_since_first_crh"]
                <= pd.Timedelta(config["post_break_min_time"])
                - pd.Timedelta(config["train_backdate"])
            )
            & (
                tweet_data["note_0_time_since_first_crh"]
                >= -pd.Timedelta(config["train_backdate"])
            )
        )
    ]

    # Find out how many pre-break observations there are
    pre_break_metrics = tweet_data[
        (
            (
                tweet_data["note_0_time_since_first_crh"]
                >= -(
                    pd.Timedelta(config["pre_break_min_time"])
                    + pd.to_timedelta(config["train_backdate"])
                )
            )
            & (
                tweet_data["note_0_time_since_first_crh"]
                < -pd.Timedelta(config["train_backdate"])
            )
        )
    ]

    # Validate that we have the correct number of observations pre break, and that
    # we have metrics data for at least one of the tweet level variables we are interested in
    correct_num_post_break_ts = (
        len(post_break_metrics) == required_number_of_ts(config)["post_break"]
    )
    correct_num_pre_break_ts = (
        len(pre_break_metrics) == required_number_of_ts(config)["pre_break"]
    )
    at_least_one_metric = False
    for metric in config["tweet_metrics"] + config["backup_tweet_metrics"]:
        at_least_one_metric = (
            metric in post_break_metrics.columns
            and pre_break_metrics[metric].notna().all()
            and post_break_metrics[metric].notna().all()
        )
        if at_least_one_metric:
            break

    return correct_num_post_break_ts, correct_num_pre_break_ts, at_least_one_metric


def find_controls_for_tweet(
    tweet_id,
    tweet_data,
    config,
    author_df,
    mmap_file,
    save_dir,
):
    # Check again that this tweet has not been completed, in case another process got here first
    str_config = deepcopy(config)

    for k in str_config.keys():
        if type(str_config[k]) is pd.Timestamp:
            str_config[k] = str_config[k].strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    if os.path.exists(save_dir / f"{tweet_id}.json.gz") and (
        len(read_from_table(save_dir / f"{tweet_id}.json.gz", str_config)) > 0
    ):
        return {}

    control_tweets_df = load(mmap_file, mmap_mode="r")

    # Filter out the treatment tweet itself from the control tweets df
    # in case we are permuting the treatment assignment
    control_tweets_df = control_tweets_df[control_tweets_df["tweet_id"] != tweet_id]

    metadata = defaultdict(list)

    correct_num_post_break_ts, correct_num_pre_break_ts, at_least_one_metric = (
        enough_treatment_data(tweet_data, config)
    )

    if not (correct_num_pre_break_ts and at_least_one_metric):
        metadata["tids_without_pre_break_data"].append(tweet_id)

    if not (correct_num_post_break_ts and at_least_one_metric):
        metadata["tids_without_post_break_data"].append(tweet_id)

    # Stop processing tweet, if we don't have enough pre or post break
    if (
        tweet_id
        in metadata["tids_without_pre_break_data"]
        + metadata["tids_without_post_break_data"]
    ):
        save_matching_data(
            tweet_id,
            {
                "treatment_tweet_id": tweet_id,
                "use_tweet": False,
                "control_tweet_ids": [],
                "matching_timestamps": [],
                "metrics_present_for_tweet": [],
            },
            config,
        )
        return metadata

    # Get the treatment tweet
    matching_metrics = []
    matching_timestamps = []
    control_dfs = {}
    for metric in config["tweet_metrics"] + config["backup_tweet_metrics"]:
        control_dfs[metric] = get_prediction_df(
            tweet_id,
            tweet_data=tweet_data,
            metric_of_interest=metric,
            control_df=control_tweets_df,
            metric=metric,
            log_name=log_name,
        )
        # Make sure we have data for this metric
        if len(control_dfs[metric]) == 0:
            continue

        # See how many timestamps are there are prior to break
        pre_break_control = control_dfs[metric][
            control_dfs[metric]["treatment_note_0_time_since_first_crh"]
            < -pd.Timedelta(config["train_backdate"])
        ]

        # how many there are after break
        post_break_control = control_dfs[metric][
            control_dfs[metric]["treatment_note_0_time_since_first_crh"]
            >= -pd.Timedelta(config["train_backdate"])
        ]
        # If we have enough data, add to the list of metrics to match on
        if (len(pre_break_control) >= required_number_of_ts(config)["pre_break"]) and (
            len(post_break_control) >= required_number_of_ts(config)["post_break"]
        ):
            matching_metrics.append(metric)
            matching_timestamps.append(
                (
                    metric,
                    pre_break_control.index.total_seconds().astype(int).to_list(),
                )
            )

    # Drop the non-calculated retweets/replies, if we have been asked to do so by config file
    # and if the calculated versions are present
    if config["replace_calculated_when_missing"]:
        for metric in config["backup_tweet_metrics"]:
            if f"calculated_{metric}" in matching_metrics:
                matching_metrics = [m for m in matching_metrics if m != metric]
                matching_timestamps = [
                    (m, ts) for (m, ts) in matching_timestamps if m != metric
                ]

    # Find control ids for each metric
    control_ids = {}
    for metric in matching_metrics:
        control_ids[metric] = {
            c.replace("control_" + metric + "_", "")
            for c in control_dfs[metric].columns
            if metric + "_" in c
        }

    # Check if we have at least one time series metric to match on
    if len(matching_metrics) == 0:
        metadata["not_enough_metrics"].append(tweet_id)
        save_matching_data(
            tweet_id,
            {
                "treatment_tweet_id": tweet_id,
                "use_tweet": False,
                "control_tweet_ids": [],
                "matching_timestamps": [],
                "metrics_present_for_tweet": [],
            },
            config,
        )
        return metadata

    # Find intersection of control ids
    control_ids = list(set.intersection(*control_ids.values()))

    # Now, make a df for the author metrics

    # Make sure the author metrics are present for treatment and tweet
    if tweet_id not in author_df.index:
        metadata["missing_author_data"].append(tweet_id)
    else:
        # Find which control IDs are present in the author metrics df
        author_metrics_ids = [c for c in control_ids if c in author_df.index]

        # Get control tweets that have the required author metrics
        if len(author_metrics_ids) < len(control_ids):
            control_ids = author_metrics_ids

        for metric in config["author_metrics"]:
            # Log that we will be matching on this metric for this tweet
            matching_metrics.append(metric)

            # Make a df for this metric
            control_dfs[metric] = author_df[[metric]].loc[
                [tweet_id] + author_metrics_ids
            ]

            # Rename columns
            control_dfs[metric].index.name = None

            # Transpose
            control_dfs[metric] = control_dfs[metric].T

            # Rename columns
            control_dfs[metric].columns = [f"treatment_{metric}"] + [
                f"{metric}_{c}" for c in author_metrics_ids
            ]

            # Add in columns that are in other metrics dfs, but that don't make sense here
            control_dfs[metric]["treatment_note_0_time_since_first_crh"] = np.NaN

    # Filter out tweets without control IDs
    if len(control_ids) == 0:
        metadata["tweets_without_controls"].append(tweet_id)
        save_matching_data(
            tweet_id,
            {
                "treatment_tweet_id": tweet_id,
                "use_tweet": False,
                "control_tweet_ids": [],
                "matching_timestamps": [],
                "metrics_present_for_tweet": [],
            },
            config,
        )
        return metadata

    save_matching_data(
        tweet_id,
        {
            "treatment_tweet_id": tweet_id,
            "use_tweet": True,
            "control_tweet_ids": control_ids,
            "matching_timestamps": matching_timestamps,
            "metrics_present_for_tweet": matching_metrics,
        },
        config,
    )

    metadata["used_tweet_ids"].append(tweet_id)

    for metric in matching_metrics:
        metadata[f"tweets_using_{metric}"].append(tweet_id)

    return metadata


def read_trt_and_ctrl(intermediate_dir, config, log_name=None, sample_size=None):
    reading_logger = get_logger(
        dir=data_root / intermediate_dir,
        log_name=log_name,
        enqueue=True,
    )

    # Load in datasets for each note
    merged_tweet_fnames = os.listdir(data_root / intermediate_dir / "b_merged")

    merged_tweet_fnames = [
        fname
        for fname in merged_tweet_fnames
        if fname.endswith(".parquet")
    ]

    # sort by seed
    merged_tweet_ids = sorted(
        merged_tweet_fnames,
        key=lambda x: TWEET_SEEDS[x.replace(".parquet", "")]
    )

    # sample if needed
    rng = np.random.default_rng(42)

    merged_tweet_ids = (
        rng.choice(merged_tweet_ids, len(merged_tweet_ids), replace=False)
        if sample_size is None
        else rng.choice(merged_tweet_ids, sample_size, replace=False)
    )

    merged_tweets = {}

    for file in tqdm(merged_tweet_ids, desc="Loading merged tweets", smoothing=0):
        if not file.endswith(".parquet"):
            continue
        tid = file.replace(".parquet", "")
        # Load parquet
        merged_tweets[tid] = read_from_table(
            data_root / intermediate_dir / "b_merged" / file,
            config={
                "dev": config["dev"],
                "time_freq": config["time_freq"],
                "volatile_tweet_filtering": config["volatile_tweet_filtering"],
            },
        )

    # Split control from treatment tweets
    control_tweets = {
        k: v.copy()
        for k, v in merged_tweets.items()
        if v["note_0_time_since_first_crh"].isna().all() and len(v) > 0
    }
    treatment_tweets = {
        k: v.copy()
        for k, v in merged_tweets.items()
        if not v["note_0_time_since_first_crh"].isna().all()
    }
    if log_name is not None:
        reading_logger.info(f"Found {len(control_tweets):,} control tweets.")
        reading_logger.info(f"Found {len(treatment_tweets):,} treatment tweets.")

    # Find the CRH times for treatment tweets
    crh_times = [
        tweet_df[
            tweet_df["note_0_time_since_first_crh"]
            == (-pd.Timedelta(config["train_backdate"]))
        ]["time_since_publication"]
        .dropna()
        .iloc[0]
        for tweet_df in treatment_tweets.values()
    ]

    # Drop rows that are only present in note history, which can't be used for matching
    tweets_to_drop = defaultdict(list)
    for tweet_id in chain(control_tweets.keys(), treatment_tweets.keys()):
        correct_df = control_tweets if tweet_id in control_tweets else treatment_tweets
        only_present_in_note_history = (
            correct_df[tweet_id]["present_in_note_history"]
            & (~correct_df[tweet_id]["present_in_metrics"])
            & (~correct_df[tweet_id]["present_in_calculated_retweets"])
            & (~correct_df[tweet_id]["present_in_calculated_replies"])
            & (~correct_df[tweet_id]["present_in_reply_tree"])
            & (~correct_df[tweet_id]["present_in_rt_cascade"])
        )
        correct_df[tweet_id] = correct_df[tweet_id][~only_present_in_note_history]

        if len(correct_df[tweet_id]) == 0:
            tweets_to_drop[
                "control" if tweet_id in control_tweets else "treatment"
            ].append(tweet_id)

    # Drop tweets that have no data
    for tweet_type in tweets_to_drop.keys():
        for tweet_id in tweets_to_drop[tweet_type]:
            if tweet_type == "control":
                control_tweets.pop(tweet_id)
            else:
                treatment_tweets.pop(tweet_id)

        # Log dropped tweets
        reading_logger.info(
            f"Dropping {len(tweets_to_drop[tweet_type]):,} {tweet_type} "
            f"tweets that are only present in "
            f"note history: {tweets_to_drop[tweet_type]}."
        )

    if config["shuffle"]:
        trt_time_not_found = []
        shuffled_treatment_times_path = (
            data_root
            / intermediate_dir
            / "c_find_controls"
            / "shuffled_treatment_times.parquet"
        )

        output_config = deepcopy(config)
        for cname, cval in config.items():
            if cname in ["tweet_metrics", "author_metrics", "backup_tweet_metrics"]:
                del output_config[cname]
                continue
            elif type(cval) is list:
                output_config[cname] = ",".join(cval)

        if shuffled_treatment_times_path.exists():
            reading_logger.info(
                f"Found existing shuffled treatment times file at {shuffled_treatment_times_path}. "
            )
            shuffled_treatment_times = read_from_table(
                shuffled_treatment_times_path,
                output_config,
            ).set_index("tweet_id").to_dict(
                orient="index"
            )
            shuffled_treatment_times = {
                k: pd.to_datetime(v["note_0_first_crh_time"])
                for k, v in shuffled_treatment_times.items()
            }
        else:
            shuffled_treatment_times = {}

        # For control posts, randomly sample a treatment time
        # from the empirical distribution of treatment times
        for ctweet in tqdm(
            control_tweets.keys(),
            desc="Randomly assigning treatment times to control tweets",
            smoothing=0,
        ):
            # Calculate the timestamp of publication
            publication_timestamp = (
                (
                        control_tweets[ctweet].loc[:, "timestamp"]
                        - control_tweets[ctweet].loc[:, "time_since_publication"]
                )
                .dropna()
                .iloc[0]
            )


            # Randomly order the treatment times
            if ctweet in shuffled_treatment_times:
                crh_times_for_ctrl_tweet = [
                    shuffled_treatment_times[ctweet] - publication_timestamp
                ]
            else:
                crh_times_for_ctrl_tweet = [
                    t
                    for t in crh_times
                    if t in control_tweets[ctweet]["time_since_publication"].values
                ]
                crh_times_for_ctrl_tweet.sort()
                rng = np.random.default_rng(TWEET_SEEDS[ctweet])
                crh_times_for_ctrl_tweet = rng.choice(
                    crh_times_for_ctrl_tweet, len(crh_times_for_ctrl_tweet), replace=False
                )

            # Try treatment times until we get one that the control post has data for
            # We need to do this because the treatment times are not guaranteed to be in the control
            # post data
            treatment_time_found = False
            tried_treatment_times = []
            for i, treatment_time in enumerate(crh_times_for_ctrl_tweet):
                # If we have already tried this treatment time, skip it (there
                # are duplicates, but we want to sample from the empirical
                # distribution)
                if treatment_time in tried_treatment_times:
                    continue
                # delete columns
                control_tweets[ctweet].drop(
                    columns=["note_0_time_since_first_crh", "note_0_first_crh_time"],
                    inplace=True,
                )
                control_tweets[ctweet].loc[:, "note_0_time_since_first_crh"] = (
                    control_tweets[ctweet]["time_since_publication"] - treatment_time
                )
                treatment_timestamp = publication_timestamp + treatment_time
                control_tweets[ctweet].loc[
                    :, "note_0_first_crh_time"
                ] = treatment_timestamp
                if all(enough_treatment_data(control_tweets[ctweet], config)):
                    treatment_time_found = True
                    break

            if not treatment_time_found:
                assert (ctweet not in shuffled_treatment_times) or pd.isna(shuffled_treatment_times[ctweet]), f"Control tweet {ctweet} already has a treatment time assigned, but was now found to not be valid"
                control_tweets[ctweet].loc[:, "note_0_time_since_first_crh"] = np.NaN
                control_tweets[ctweet].loc[:, "note_0_first_crh_time"] = np.NaN
                trt_time_not_found.append(ctweet)

            shuffled_treatment_times[ctweet] = control_tweets[ctweet][
                "note_0_first_crh_time"
            ].iloc[0]

        # Log treatment times that were not found
        if len(trt_time_not_found) > 0:
            reading_logger.info(
                f"Treatment times not found for {len(trt_time_not_found):,} control tweets: {trt_time_not_found}"
            )

        shuffled_treatment_times = pd.Series(shuffled_treatment_times).to_frame(name="note_0_first_crh_time").reset_index()
        shuffled_treatment_times.rename(columns={"index": "tweet_id"}, inplace=True)
        for cname, cval in config.items():
            if cname in ["tweet_metrics", "author_metrics", "backup_tweet_metrics"]:
                continue
            shuffled_treatment_times[cname] = output_config[cname]

        clear_and_write(
            shuffled_treatment_times,
            shuffled_treatment_times_path,
            output_config,
        )

        tweets_needing_tes = list(chain(treatment_tweets.items(), control_tweets.items()))
        all_eligible_donors = list(chain(treatment_tweets.items(), control_tweets.items()))
    else:
        tweets_needing_tes = list(treatment_tweets.items())
        all_eligible_donors = list(control_tweets.items())

    # Remove control tweets where we couldn't find a treatment time
    # Also convert back to a dict
    tweets_needing_tes = {
        tweet_id: tweet_data.copy()
        for tweet_id, tweet_data in tweets_needing_tes
        if not tweet_data["note_0_time_since_first_crh"].isna().all()
    }

    all_eligible_donors = {
        tweet_id: tweet_data.copy() for tweet_id, tweet_data in all_eligible_donors
    }

    return all_eligible_donors, tweets_needing_tes


if __name__ == "__main__":
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
        "shuffle",
    ]
    for c in necessary_config:
        if c not in config.keys():
            raise ConfigError(
                f"Please specify config value '{c}' in config file '{config_path}.'"
            )

    # Drop unneeded config values
    config = {c: config[c] for c in necessary_config}

    # Fill in the default dev value, if we've been asked to
    if config["dev"] == "DEVICE_DEFAULT":
        config["dev"] = socket.gethostname() == "is-is28m16x"

    # Check that the missing metric action is valid
    possible_missing_metric_actions = ["drop_tweet", "impute", "drop_metric"]
    if config["missing_metric_action"] not in possible_missing_metric_actions:
        raise ConfigError(
            f"Invalid 'missing_metric_action' in config file '{config_path}'. "
            f"Please specify one of {possible_missing_metric_actions}."
        )

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

    # Get directories
    intermediate_dir = "cn_effect_intermediate" + ("_dev" if config["dev"] else "_prod")
    save_dir = data_root / intermediate_dir / "c_find_controls/"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Get logger
    logger, log_name = get_logger(
        data_root / intermediate_dir, enqueue=True, return_logger_name=True
    )

    # Check that we're running the type of run we'd expect on this machine
    check_run_type(config["dev"], logger)

    # Log config path and config
    logger.info(f"Config path: {config_path}")
    logger.info(f"Config: {config}")

    # Get author metrics
    author_df = pd.read_parquet(author_path).set_index("tweet_id")
    author_df = author_df[author_df["author_n_followers"].notna()]
    author_df = author_df[["author_n_followers"]]

    control_tweets, treatment_tweets = read_trt_and_ctrl(
        intermediate_dir, config, log_name=log_name, sample_size=None
    )

    # Drop unneeded columns
    for tweet_id in chain(control_tweets.keys(), treatment_tweets.keys()):
        correct_df = (
            control_tweets if tweet_id in control_tweets.keys() else treatment_tweets
        )
        correct_df[tweet_id].drop(
            columns=[
                c
                for c in correct_df[tweet_id].columns
                if c
                not in config["tweet_metrics"]
                + config["backup_tweet_metrics"]
                + ["tweet_id", "time_since_publication", "note_0_time_since_first_crh"]
            ],
            inplace=True,
        )

    control_tweets_df = pd.concat(
        [
            v.drop(columns=["note_0_time_since_first_crh"])
            for v in control_tweets.values()
        ],
    ).reset_index(drop=True)

    # Set index, which we'll use later
    control_tweets_df = control_tweets_df.set_index("time_since_publication")

    # Remove any tweets that have already been calculated
    str_config = deepcopy(config)
    for k in str_config.keys():
        if type(str_config[k]) is pd.Timestamp:
            str_config[k] = str_config[k].strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    previously_calculated = []
    for tweet_id, _ in treatment_tweets.items():
        if os.path.exists(save_dir / f"{tweet_id}.json.gz") and (
            len(read_from_table(save_dir / f"{tweet_id}.json.gz", str_config)) > 0
        ):
            previously_calculated.append(tweet_id)

    config_copy = deepcopy(config)

    tasks = [
        (tid, td)
        for tid, td in treatment_tweets.items()
        if tid not in previously_calculated
    ]

    # randomize the order of the tasks
    random.shuffle(tasks)

    # Save control tweets to disk
    fname = f"control_tweets{uuid.uuid4()}.joblib"
    (data_root / intermediate_dir / "tmp").mkdir(exist_ok=True, parents=True)
    fp = data_root / intermediate_dir / "tmp" / fname
    dump(control_tweets_df, fp, compress=0)

    with Parallel(
        n_jobs=NUM_THREADS,
        backend="loky",
        prefer="processes",
        max_nbytes=None,
        batch_size="auto",
    ) as par:
        metadatas = par(
            delayed(find_controls_for_tweet)(
                tid,
                tweet_data=td,
                config=config_copy,
                author_df=author_df,
                mmap_file=fp,
                save_dir=save_dir,
            )
            for tid, td in tqdm(tasks, desc="Finding controls for tweets", smoothing=0.0, unit=" tweet")
        )

    metadata = defaultdict(list)
    for m in metadatas:
        for k, v in m.items():
            metadata[k] += v
    metadata["previously_calculated"] = previously_calculated

    logger.info(
        f"There were {len(metadata['used_tweet_ids']):,} treatment tweets that had enough history to be used. "
    )
    logger.info(
        f"There were {len(metadata['tweets_without_controls']):,} treatment tweets that were removed as no valid "
        f"controls could be found."
    )
    logger.info(
        f"There were {len(metadata['tids_without_post_break_data']):,} tweets that did not have the minimum amount of "
        f"metrics data post-break to be used ({config['post_break_min_time']})."
    )
    logger.info(
        f"There were {len(metadata['tids_without_pre_break_data']):,} tweets that did not have the minimum amount of "
        f"metrics data pre-break to be used ({config['pre_break_min_time']}). (The requirement is that there is "
        f"at least one metric for which the treatment tweet has this much data. Metrics that do not have the minimum "
        f"history requirement will not receive a TE for these tweets)"
    )
    logger.info(
        f"There were {len(metadata['not_enough_metrics']):,} tweets that did not have any time series metrics to "
        f"match on."
    )
    logger.info(
        f"There were {len(metadata['previously_calculated']):,} tweets that have already been calculated."
    )

    # Now load metadata about which metrics were used for each tweet from disk
    # Load metadata from disk
    all_control_configs = {
        k: v
        for k, v in read_control_configs(intermediate_dir, config).items()
        if (len(v) > 0) and v[0]["use_tweet"]
    }

    # Make sure tweets_needing_tes and all_control_configs match
    tweets_ran =  list(treatment_tweets.keys())
    if not all([tweet_id in all_control_configs.keys() for tweet_id in tweets_ran]):
        logger.error(
            f"Some controls configs were supposed to be run but were not on disk: {len(set(tweets_ran) - set(all_control_configs.keys()))} in total."
        )

    if not all([tweet_id in tweets_ran for tweet_id in all_control_configs.keys()]):
        logger.error(
            f"Some control configs were on disk but were not supposed to be run: {len(set(all_control_configs.keys()) - set(tweets_ran))} in total."
        )

    logger.info(
        f"There were {len(all_control_configs):,} tweets that had controls found for them."
    )
    for metric in (
        config["tweet_metrics"]
        + config["backup_tweet_metrics"]
        + config["author_metrics"]
    ):
        metadata[f"tweets_using_{metric}"] = []

    for k, v in all_control_configs.items():
        assert len(v) == 1
        v = v[0]
        for m in v["metrics_present_for_tweet"]:
            metadata[f"tweets_using_{m}"].append(k)

    for metric in (
        config["tweet_metrics"]
        + config["backup_tweet_metrics"]
        + config["author_metrics"]
    ):
        metric_col = f"tweets_using_{metric}"
        logger.info(
            f"There were {len(metadata[metric_col]):,} tweets that will be matched on {metric}."
        )

    # Calculate standard deviations for metrics and timestamps
    all_sds = []
    for metric in config["tweet_metrics"] + config["author_metrics"]:
        if not metadata[f"tweets_using_{metric}"]:
            continue

        series = []

        for tweet_id in metadata[f"tweets_using_{metric}"]:
            df = treatment_tweets if tweet_id in treatment_tweets else control_tweets
            if metric in config["tweet_metrics"]:
                s = df[tweet_id].set_index(  # frame for one tweet
                    "note_0_time_since_first_crh"
                )[metric]
                s.index = s.index.total_seconds()
                series.append(s)
            else:
                series.append(
                    pd.Series(author_df.loc[tweet_id, metric], index=[np.nan])
                )

        # If we are filling in missing calculated metrics with non-calculated metrics, we need to
        # add these to the observations
        if config["replace_calculated_when_missing"]:
            api_metric = metric.replace("calculated_", "")
            if api_metric in config["backup_tweet_metrics"]:
                for tweet_id in metadata[f"tweets_using_{api_metric}"]:
                    df = (
                        treatment_tweets
                        if tweet_id in treatment_tweets
                        else control_tweets
                    )
                    s = (
                        df[tweet_id].set_index(  # frame for one tweet
                            "note_0_time_since_first_crh"
                        )[api_metric]
                    ).rename(metric)
                    s.index = s.index.total_seconds()
                    series.append(s)

        combined = pd.concat(series)
        sd = combined.groupby(level=0, dropna=False).std(ddof=1)

        all_sds.append(sd.to_frame("sd").reset_index().assign(metric=metric))

        # Log that we completed
        logger.info(
            f"Calculated standard deviations for {len(all_sds[-1]):,} timestamps for {metric}. "
        )

        # Add metric name to index
        all_sds[-1]["metric"] = metric

    # Concatenate into a single series
    all_sds = pd.concat(all_sds, axis=0, ignore_index=True).drop(columns="index")

    # Drop any NAs, which can occur if there aren't any treatment tweets with observations of
    # a metric at a time
    all_sds = all_sds[all_sds["sd"].notna()].reset_index(drop=True)

    # Embeddings were already standardized, so we overwrite the standard deviation
    # here to be 1. This results in them not being scaled again during step f.
    for metric in config["author_metrics"]:
        if (metric in all_sds["metric"].values) and ("embd" in metric.lower()):
            all_sds.loc[all_sds["metric"] == metric, "sd"] = 1.0

    # Save config info
    output_config = deepcopy(config)
    for cname, cval in config.items():
        if cname in ["tweet_metrics", "author_metrics", "backup_tweet_metrics"]:
            del output_config[cname]
            continue
        elif type(cval) is list:
            output_config[cname] = ",".join(cval)

        all_sds[cname] = output_config[cname]

    # Save to disk
    clear_and_write(
        all_sds,
        save_dir / "metric_sds.parquet",
        output_config,
    )

    logger.info(
        f"Metric standard deviations have been written to {save_dir / 'metric_sds.parquet'}."
    )

    control_tweet_ids = pd.DataFrame(
        {
            "tweet_id": [k for k in control_tweets.keys()],
            "tweet_type": ["control"] * len(control_tweets),
        }
    )
    treatment_tweet_ids = pd.DataFrame(
        {
            "tweet_id": [k for k in all_control_configs.keys()],
            "tweet_type": ["treatment"] * len(all_control_configs),
        }
    )

    tweet_ids = pd.concat([control_tweet_ids, treatment_tweet_ids], axis=0)

    # Write treatment and control tweet IDs to disk
    output_config = deepcopy(config)
    for cname, cval in config.items():
        if cname in ["tweet_metrics", "author_metrics", "backup_tweet_metrics"]:
            del output_config[cname]
            continue
        elif type(cval) is list:
            output_config[cname] = ",".join(cval)

        tweet_ids[cname] = output_config[cname]

    clear_and_write(
        tweet_ids,
        save_dir / "tweet_ids.csv",
        output_config,
    )
