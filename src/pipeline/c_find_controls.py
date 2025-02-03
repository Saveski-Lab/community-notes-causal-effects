import json
import os
import re
import sys
import argparse
import socket
from collections import defaultdict
from pathlib import Path
from copy import deepcopy
from joblib import Parallel, delayed

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add current directory to path for imports
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.pipeline.a_preprocess import local_data_root, shared_data_root
from src.utils import (
    save_environment,
    get_logger,
    check_run_type,
    clear_and_write,
    ConfigError,
    read_from_table,
)

# This script takes in the merged data. For each treatment tweet, it creates a json that
# contains the treatment tweet and all control tweets that have enough data to be used in the
# synthetic control.

########################################################################################################

NUM_THREADS = 30  # TODO: Change to production value

author_path = shared_data_root / "data" / "_dfs" / "web_tweets_author_info_df.json.gz"

########################################################################################################

metric_parents = {
    "replies": "metrics",
    "replies_per_impression": "metrics",
    "retweets": "metrics",
    "retweets_per_impression": "metrics",
    "quotes": "metrics",
    "likes": "metrics",
    "like_through_rate": "metrics",
    "impressions": "metrics",
    "calculated_replies": "calculated_replies",
    "calculated_replies_per_impression": "calculated_replies",
    "calculated_retweets": "calculated_retweets",
    "calculated_retweets_per_impression": "calculated_retweets",
    "author_n_followers": "author",
    "author_n_friends": "author",
    "author_n_statuses": "author",
    "rt_graph_num_nodes": "rt_graph",
    "rt_graph_density": "rt_graph",
    "rt_graph_transitivity": "rt_graph",
    "reply_graph_num_nodes": "reply_graph",
    "reply_graph_density": "reply_graph",
    "reply_graph_transitivity": "reply_graph",
    "reply_tree_width": "reply_tree",
    "reply_tree_depth": "reply_tree",
    "reply_tree_wiener_index": "reply_tree",
    "rt_cascade_width": "rt_cascade",
    "rt_cascade_depth": "rt_cascade",
    "rt_cascade_wiener_index": "rt_cascade",
    "rt_cascade_num_nodes_root_tweet": "rt_cascade",
    "rt_cascade_num_nodes": "rt_cascade",
    "rt_cascade_num_nodes_non_root_tweet": "rt_cascade",
}

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


def get_prediction_df(treatment_id, tweet_data, metric_of_interest, control_df, metric):
    """
    For a given treatment tweet, get a dataframe of control tweets with enough data
    so that we can use them to create a synthetic control.
    """

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
            f"present_in_{metric_parents[metric_of_interest]}",
            "any_crh",
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

    # Filter treatment df to make sure that this was included in the metrics we're interested in
    treatment_df = treatment_df[
        treatment_df[f"treatment_present_in_{metric_parents[metric_of_interest]}"]
    ]

    # Filter out NAs, which can occur if the source data has NAs (e.g. impressions that were not recorded)
    if treatment_df[f"treatment_{metric_of_interest}"].isna().any():
        logger.warning(
            f"Treatment tweet {treatment_id} has NA in metric {metric_of_interest}"
        )
    treatment_df = treatment_df[treatment_df[f"treatment_{metric_of_interest}"].notna()]

    if not (treatment_df.index.diff()[1:] == pd.Timedelta(config["time_freq"])).all():
        logger.warning(
            f"Time series for tweet {treatment_id} metric {metric_of_interest} is not "
            f"continuous, and missing some timestamps"
        )

    # Get relevant timestamps in control_df
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


def find_controls_for_tweet(
    tweet_id,
    tweet_data,
    logger,
    config,
    control_tweets_df,
    author_df,
):

    metadata = defaultdict(list)

    # Get all note IDs
    note_id_cols = [
        col for col in tweet_data.columns if re.match("note_\d+_note_id", col)
    ]
    all_note_ids = []
    for nidc in note_id_cols:
        all_note_ids.append(tweet_data[pd.notna(tweet_data[nidc])][nidc].iloc[0])

    logger.info(f"Processing tweet {tweet_id}, Note IDs: {all_note_ids}.")

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

    # Find how many timestamps we're supposed to have after the break
    number_of_post_break_timestamps = len(
        pd.timedelta_range(
            -pd.Timedelta(config["train_backdate"]),
            (
                pd.Timedelta(config["post_break_min_time"])
                - pd.Timedelta(config["train_backdate"])
            ),
            freq=config["time_freq"],
        )
    )

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

    # Find out how many observations there are supposed to be pre-break
    number_of_pre_break_timestamps = (
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

    # Validate that we have the correct number of observations pre break, and that
    # we have metrics data for at least one of the tweet level variables we are interested in
    correct_num_post_break_ts = len(post_break_metrics) == number_of_post_break_timestamps
    correct_num_pre_break_ts = len(pre_break_metrics) == number_of_pre_break_timestamps
    at_least_one_metric = False
    for metric in config["tweet_metrics"] + config["backup_tweet_metrics"]:
        at_least_one_metric = at_least_one_metric or (
            pre_break_metrics[f"present_in_{metric_parents[metric]}"].all()
            and post_break_metrics[f"present_in_{metric_parents[metric]}"].all()
            and pre_break_metrics[metric].notna().all()
            and post_break_metrics[metric].notna().all()
        )
        if at_least_one_metric:
            break

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
        if (len(pre_break_control) >= number_of_pre_break_timestamps) and (
            len(post_break_control) >= number_of_post_break_timestamps
        ):
            matching_metrics.append(metric)
            matching_timestamps.append(
                (
                    metric,
                    pre_break_control.index.total_seconds().astype(int).to_list(),
                )
            )
            if pd.isna(matching_timestamps[-1][-1]).any():
                logger.warning(
                    f"Issue processing tweet {tweet_id}: NA matching time for metric {metric}"
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
        metadata["not_enough_metrics"] = tweet_id
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
            control_dfs[metric]["treatment_any_crh"] = np.NaN

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


def read_trt_and_ctrl(intermediate_dir, config, logger=None, sample_size=None):
    # Load in datasets for each note
    merged_tweet_ids = os.listdir(local_data_root / intermediate_dir / "b_merged")
    merged_tweets = {}
    merged_tweet_ids = (
        merged_tweet_ids
        if sample_size is None
        else np.random.choice(merged_tweet_ids, sample_size, replace=False)
    )
    for file in tqdm(merged_tweet_ids, desc="Loading merged tweets", smoothing=0):
        if not file.endswith(".parquet"):
            continue
        tid = file.replace(".parquet", "")
        # Load parquet
        merged_tweets[tid] = read_from_table(
            local_data_root / intermediate_dir / "b_merged" / file,
            config={
                "dev": config["dev"],
                "time_freq": config["time_freq"],
                "max_date": config["max_date"],
                "use_backup_tweets": config["use_backup_tweets"],
                "use_bookmark_tweets": config["use_bookmark_tweets"],
                "volatile_tweet_filtering": config["volatile_tweet_filtering"],
            },
        )
        # Calculated view-normalized rt/replies
        merged_tweets[tid]["replies_per_impression"] = merged_tweets[tid]["replies"] / merged_tweets[tid]["impressions"]
        merged_tweets[tid]["retweets_per_impression"] = merged_tweets[tid]["replies"] / merged_tweets[tid]["impressions"]
        merged_tweets[tid]["calculated_replies_per_impression"] = merged_tweets[tid]["calculated_replies"] / merged_tweets[tid]["impressions"]
        merged_tweets[tid]["calculated_retweets_per_impression"] = merged_tweets[tid]["calculated_retweets"] / merged_tweets[tid]["impressions"]
        if (
                "rt_cascade_num_nodes_root_tweet" in merged_tweets[tid].columns
                and "rt_cascade_num_nodes" in merged_tweets[tid].columns
        ):
            merged_tweets[tid]["rt_cascade_num_nodes_non_root_tweet"] = (
                    merged_tweets[tid]["rt_cascade_num_nodes"] -
                    merged_tweets[tid]["rt_cascade_num_nodes_root_tweet"]
            )
        if config["replace_calculated_when_missing"]:
            impression_normalized_backups = np.unique(
                [
                    f"{m}_per_impression"
                    for m in config["backup_tweet_metrics"]
                    if "_per_impression" not in m
                ]
                + [m for m in config["backup_tweet_metrics"] if "_per_impression" in m]
            )
            for metric in impression_normalized_backups:
                merged_tweets[tid][f"calculated_{metric}"] = np.where(
                    pd.isna(merged_tweets[tid][f"calculated_{metric}"]),
                    merged_tweets[tid][metric],
                    merged_tweets[tid][f"calculated_{metric}"],
                )

    # Split control from treatment tweets
    control_tweets = {
        k: v
        for k, v in merged_tweets.items()
        if v["note_0_time_since_first_crh"].isna().all() and len(v) > 0
    }
    if logger:
        logger.info(f"Found {len(control_tweets):,} control tweets.")
    treatment_tweets = {
        k: v
        for k, v in merged_tweets.items()
        if not v["note_0_time_since_first_crh"].isna().all()
    }
    if logger:
        logger.info(f"Found {len(treatment_tweets):,} treatment tweets.")

    return control_tweets, treatment_tweets


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
    ]
    for c in necessary_config:
        if c not in config.keys():
            raise ConfigError(
                f"Please specify config value '{c}' in config file '{config_path}.'"
            )

    # Drop unneeded config values
    config = {c: config[c] for c in necessary_config}

    # Fill in the dev value based on whether this is a local run
    if config["dev"] == "DEVICE_DEFAULT":
        config["dev"] = socket.gethostname() == "is-is28m16x"

    # Convert max date to datetime
    config["max_date"] = pd.to_datetime(config["max_date"], utc=True)

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
    save_dir = local_data_root / intermediate_dir / "c_find_controls/"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Get logger
    logger = get_logger(local_data_root / intermediate_dir)

    # Check that we're running the type of run we'd expect on this machine
    check_run_type(config["dev"], logger)

    # Log config path and config
    logger.info(f"Config path: {config_path}")
    logger.info(f"Config: {config}")

    # Save the environment
    save_environment("src/pipeline/c_find_controls.yml", logger)

    # Get author metrics
    author_df = pd.read_json(
        author_path, dtype={"tweet_id": str}, lines=True
    ).set_index("tweet_id")

    control_tweets, treatment_tweets = read_trt_and_ctrl(
        intermediate_dir, config, logger, sample_size=None
    )

    # Save tweet IDs out to csv
    tweet_ids = pd.DataFrame(
        {
            "tweet_id": [k for k in treatment_tweets.keys()]
            + [k for k in control_tweets.keys()],
            "tweet_type": (["treatment"] * len(treatment_tweets))
            + (["control"] * len(control_tweets)),
        }
    )
    tweet_ids.to_csv(save_dir / "trt_and_ctrl_ids.csv", index=False)

    # Find out if, at each time stamp, any note was CRH for the TRT tweets
    for ttweet in treatment_tweets.keys():
        status_cols = [
            col
            for col in treatment_tweets[ttweet].columns
            if re.match("note_\d+_twitter_status", col)
        ]
        treatment_tweets[ttweet]["any_crh"] = (
            treatment_tweets[ttweet][status_cols].fillna("NA")
            == "CURRENTLY_RATED_HELPFUL"
        ).any(axis=1)

    # Drop extra note data from control tweets
    for ctweet in control_tweets.keys():
        note_cols = [col for col in control_tweets[ctweet].columns if "note_" in col]
        control_tweets[ctweet].drop(
            columns=note_cols
            + [
                "most_recent_source",
            ]
            + [
                m
                for m in metric_parents.keys()
                if (m in control_tweets[ctweet].columns)
                and (m not in config["tweet_metrics"] + config["backup_tweet_metrics"])
            ],
            inplace=True,
        )

    # Join control tweets into a big df
    control_tweets_df = pd.concat(control_tweets.values()).reset_index(drop=True)

    # Set index, which we'll use later
    control_tweets_df = control_tweets_df.set_index("time_since_publication")

    # Save memory
    del control_tweets

    # Create objects to log metadata into
    tweet_metrics_to_match_on = defaultdict(list)

    # execute the function in parallel
    metadatas = Parallel(n_jobs=NUM_THREADS, backend="threading")(
        delayed(find_controls_for_tweet)(
            tweet_id,
            tweet_data=tweet_data,
            logger=logger,
            config=deepcopy(config),
            control_tweets_df=control_tweets_df,
            author_df=author_df,
        )
        for tweet_id, tweet_data in tqdm(
            [i for i in treatment_tweets.items()],
            desc="Finding controls for treatment tweets",
            smoothing=0,
        )
    )

    metadata = defaultdict(list)
    for m in metadatas:
        for k, v in m.items():
            metadata[k] += v

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
    for metric in (
        config["tweet_metrics"]
        + config["backup_tweet_metrics"]
        + config["author_metrics"]
    ):
        metric_col = f"tweets_using_{metric}"
        logger.info(
            f"There were {len(metadata[metric_col]):,} tweets that will be matched on {metric}."
        )
    logger.info("Metadata has been written to c_find_control_metadata.json.")

    with open(save_dir / "c_find_control_metadata.json", "w") as f:
        metadata = {
            k: v if type(v) != list else [str(x) for x in v]
            for k, v in metadata.items()
        }
        json.dump(metadata, f)

    # Calculate standard deviations for metrics and timestamps
    all_sds = []
    for metric in config["tweet_metrics"] + config["author_metrics"]:
        if len(metadata[f"tweets_using_{metric}"]) == 0:
            continue

        # Iterate through treatment tweets, getting observations
        # for the relevant metric
        treatment_observations = []
        for tweet_id in metadata[f"tweets_using_{metric}"]:
            if metric in config["tweet_metrics"]:
                treatment_observations.append(
                    treatment_tweets[tweet_id][["note_0_time_since_first_crh", metric]]
                    .set_index("note_0_time_since_first_crh")
                    .rename(columns={metric: tweet_id})
                )

                # Convert to seconds
                treatment_observations[-1].index = treatment_observations[
                    -1
                ].index.total_seconds()

            elif metric in config["author_metrics"]:
                treatment_observations.append(
                    pd.DataFrame(
                        {
                            "note_0_time_since_first_crh": [pd.to_timedelta(np.nan)],
                            tweet_id: [author_df.loc[tweet_id, metric]],
                        }
                    ).set_index("note_0_time_since_first_crh")
                )

        # If we are filling in missing calculated metrics with non-calculated metrics, we need to
        # add these to the observations
        if config["replace_calculated_when_missing"]:
            if metric.replace("calculated_", "") in config["backup_tweet_metrics"]:
                for tweet_id in metadata[
                    f"tweets_using_{metric.replace('calculated_', '')}"
                ]:
                    treatment_observations.append(
                        treatment_tweets[tweet_id][
                            [
                                "note_0_time_since_first_crh",
                                metric.replace("calculated_", ""),
                            ]
                        ]
                        .set_index("note_0_time_since_first_crh")
                        .rename(columns={metric.replace("calculated_", ""): tweet_id})
                    )

                    treatment_observations[-1].index = treatment_observations[
                        -1
                    ].index.total_seconds()

        # Concatenate observations from different treatment tweets
        treatment_observations = pd.concat(treatment_observations, axis=1)

        # Calculate sample standard deviation
        all_sds.append(
            treatment_observations.std(axis=1, skipna=True, ddof=1)
            .to_frame(name="sd")
            .reset_index()
        )

        # Add metric name to index
        all_sds[-1]["metric"] = metric

    # Concatenate into a single series
    all_sds = pd.concat(all_sds, axis=0, ignore_index=True)

    # Drop any NAs, which can occur if there aren't any treatment tweets with observations of
    # a metric at a time
    all_sds = all_sds[all_sds["sd"].notna()].copy()

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

