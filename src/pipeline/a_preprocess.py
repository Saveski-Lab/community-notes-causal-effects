import gzip
import os
import sys
import re
import socket
from glob import glob
import warnings
import dask.dataframe as dd
from pathlib import Path
from collections import defaultdict

import argparse
import numpy as np
import json
import pandas as pd
from tqdm.auto import tqdm

# Add current directory to path for imports
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.utils import (
    save_environment,
    get_logger,
    check_run_type,
    clear_and_write,
    ConfigError,
    json_gzip_reader,
)

########################################################################################################################
# Paths, which will need to be set in your environment

local_data_root = Path(os.getenv("BW_LOCAL_DATA_ROOT"))

input_data_dir = local_data_root / "cn_effect_input"

shared_data_root = Path(os.getenv("BW_SHARED_DATA_ROOT"))

full_pipeline_db_export_dir = (
    shared_data_root / "database" / "db-export-apr-1-2024" / "birdwatch"
)

calculated_metrics_dir = shared_data_root / "data" / "_dfs"

note_history_dir = shared_data_root / "public-releases-csvs" / "downloads"

structural_metrics_dir = shared_data_root / "analysis" / "structural_metrics_v7"

partial_structural_metrics_dir = (
    shared_data_root / "analysis" / "structural_metrics_v7_partial"
)

########################################################################################################################
structural_metrics_keys = {
    "metrics_reply_tree_over_time": "reply_tree",
    "metrics_retweet_cascade_over_time": "rt_cascade",
    "metrics_network_replies_over_time": "reply_graph",
    "metrics_network_retweets_over_time": "rt_graph",
}

structural_metrics_columns = [
    "num_nodes_root_tweet",
    "width",
    "density",
    "wiener_index",
    "windowed_wiener_index",
    "fraction_two_way_connected_pairs",
    "fraction_one_way_connected_pairs",
    "windowed_depth_of_new_nodes",
    "public_metrics_n_nodes",
    "transitivity",
    "num_nodes",
    "depth",
    "num_connected_components",
    "num_edges",
    "fraction_nodes_in_largest_component",
    "fraction_non_connected_pairs",
]

########################################################################################################################
# Function for interpolating with the maximum value since the last NA


def max_ffill(series):
    # Fill in missing values with the maximum value since the last NA. Used to make sure that we are capturing all CRH
    # values
    max_value = None
    no_values_since_last_na = True
    last_non_na_value = None
    result = []
    for value in series:
        # Check if we have a missing value
        if pd.isna(value):
            # If the last status was also missing, then we should fill with the last recorded status, which is presumed
            # to still be in effect
            if no_values_since_last_na:
                result.append(last_non_na_value)
            # If the last value was present, then we should fill with the maximum status since the last NA
            else:
                result.append(max_value)
            # We have a new NA, so reset the max value
            max_value = None
            no_values_since_last_na = True
        else:
            max_value = value if max_value is None else max(max_value, value)
            result.append(value)
            last_non_na_value = value
            no_values_since_last_na = False
    return pd.Series(result)


########################################################################################################################
# class for loading engagement metrics


def get_timezone(tweet_store_path):
    if tweet_store_path in [
        "public.api_tweets/1",
        "public.web_tweets/1",
        "public.bookmark_tweets/1",
        "1_v1_csvs_misc/ec2-bw1-axel-api-csvs/all_sorted.csv",
        "1_v1_csvs_misc/ec2-bw1-martin/all_sorted.csv",
        "3_psql_db_v2_bw2/birdwatched_tweets_api_data.csv",
        "3_psql_db_v2_bw2/web_birdwatched_tweets.csv",
        "4_sqlite_csvs_bw2/api_tweets.csv",
        "4_sqlite_csvs_bw2/web_tweets.csv",
        "database-fs-martin/api_tweets*.csv.gz",
        "database-fs-martin/api_tweets_2023-04*.csv.gz",
        "database-fs-martin/api_tweets_2023-05*.csv.gz",
        "database-fs-martin/api_tweets_2023-06*.csv.gz",
        "database-fs-martin/api_tweets_2023*.csv.gz",
    ]:
        return "UTC"
    elif tweet_store_path in [
        "1_v1_csvs_misc/local-axel/all_sorted.csv",
        "1_v1_csvs_misc/local-martin/all_sorted.csv",
    ]:
        return "America/Los_Angeles"


########################################################################################################################
# Class for reading creation times from raw tweet objects


class CreationTimeRetriever(object):
    def __init__(self, dev):
        self.load_api_tweet_objects()
        self.load_tweet_objects_v1_api_05_05_2023()
        self.load_web_tweets_objects()
        self.merge()
        self.convert_creation_times()
        if dev:
            self.creation_times = self.creation_times.loc[
                self.creation_times["tweet_id"].isin(tweet_ids)
            ]

    def load_api_tweet_objects(self):
        # Read api_tweet_objects.json
        self.api_tweet_objects = pd.read_json(
            local_data_root
            / "cn_effect_input"
            / "bw-tweets"
            / "api_tweet_objects.json",
        )

        # Transpose, and get relevant columns
        self.api_tweet_objects = self.api_tweet_objects.T[["id", "created_at"]]
        self.api_tweet_objects["creation_time_source"] = "api_tweet_objects"

    def load_tweet_objects_v1_api_05_05_2023(self):
        # Read tweet_objects_v1_api_05_05_2023.json.gz
        tweet_objs = defaultdict(list)

        # Open file
        with gzip.open(
            local_data_root
            / "cn_effect_input"
            / "bw-tweets"
            / "tweet_objects_v1_api_05_05_2023.json.gz",
            "r",
        ) as fin:
            # Each line in file contains a separate json, so iterate through
            for line in fin:
                # Load json
                tweet_json = json.loads(line)

                # Extract id and creation time
                tweet_objs["id"].append(tweet_json[0])
                tweet_objs["created_at"].append(tweet_json[1]["result"]["created_at"])

        # Convert to df
        self.tweet_objects_v1_api_05_05_2023 = pd.DataFrame(tweet_objs)
        self.tweet_objects_v1_api_05_05_2023["creation_time_source"] = (
            "tweet_objects_v1_api_05_05_2023"
        )

    def load_web_tweets_objects(self):
        # Read web_tweets_objects.json.gz
        with gzip.open(
            local_data_root
            / "cn_effect_input"
            / "bw-tweets"
            / "web_tweets_objects.json.gz",
            "r",
        ) as fin:
            loaded = json.loads(fin.read())

        all_items = list()
        no_result = list()
        no_creation_info = list()

        # Iterate through web tweet objects
        for tid, item in loaded.items():
            # Get tweet_results from object
            results = item["items"][0]["item"]["itemContent"]["tweet_results"]
            # Filter out web tweets without results
            if "result" not in results.keys():
                no_result.append(tid)
                continue
            else:
                results = results["result"]

            # Some web tweets have a "tweet" level that needs to be descended, so do so if necessary
            if "tweet" in results.keys():
                results = results["tweet"]

            # Creation times are part of legacy dict, which not all tweets have
            if "legacy" in results.keys():
                created_at = results["legacy"]["created_at"]
            else:
                no_creation_info.append(tid)
                continue

            all_items.append(
                {
                    "id": tid,
                    "created_at": created_at,
                }
            )

        self.web_tweet_objects = pd.DataFrame(all_items)
        self.web_tweet_objects["creation_time_source"] = "web_tweet_objects"

    def merge(self):
        # Merge the dataframes
        self.creation_times = pd.concat(
            [
                self.api_tweet_objects,
                self.tweet_objects_v1_api_05_05_2023,
                self.web_tweet_objects,
            ]
        )

        self.creation_times = self.creation_times.rename(
            columns={"id": "tweet_id"}
        ).reset_index(drop=True)

    def convert_creation_times(self):
        first_time_format = self.creation_times[
            self.creation_times["created_at"].str.contains(" ")
        ].copy()
        first_time_format["created_at"] = pd.to_datetime(
            first_time_format["created_at"],
            format="%a %b %d %H:%M:%S %z %Y",
            utc=True,
        )

        second_time_format = self.creation_times[
            ~self.creation_times["created_at"].str.contains(" ")
        ].copy()
        second_time_format["created_at"] = pd.to_datetime(
            second_time_format["created_at"],
            format="%Y-%m-%dT%H:%M:%S.%fZ",
            utc=True,
        )

        self.creation_times = pd.concat([first_time_format, second_time_format])

        self.creation_times = self.creation_times.drop_duplicates(
            subset=["tweet_id", "created_at"]
        ).reset_index(drop=True)


class TweetSnapshotProcessor:

    def __init__(
        self, time_freq, dev, use_backup_tweets, use_bookmark_tweets, volatile_tweet_filtering, max_date
    ):
        """
        Load engagement, retweets, etc. from the Twitter API.
        """
        # Save config to object
        self.time_freq = time_freq
        self.dev = dev
        self.use_backup_tweets = use_backup_tweets
        self.use_bookmark_tweets = use_bookmark_tweets
        self.volatile_tweet_filtering = volatile_tweet_filtering
        self.max_date = max_date

        self.tweet_stores = [
            (full_pipeline_db_export_dir, "public.api_tweets/1"),
            (shared_data_root, "database-fs-martin/api_tweets_2023*.csv.gz"),
        ]
        if self.use_backup_tweets:
            self.tweet_stores.extend(
                [
                    (full_pipeline_db_export_dir, "public.web_tweets/1"),
                    (input_data_dir, "1_v1_csvs_misc/local-axel/all_sorted.csv"),
                    (
                        input_data_dir,
                        "1_v1_csvs_misc/local-martin/all_sorted.csv",
                    ),
                    (
                        input_data_dir,
                        "1_v1_csvs_misc/ec2-bw1-axel-api-csvs/all_sorted.csv",
                    ),
                    (
                        input_data_dir,
                        "1_v1_csvs_misc/ec2-bw1-martin/all_sorted.csv",
                    ),
                    (
                        input_data_dir,
                        "3_psql_db_v2_bw2/birdwatched_tweets_api_data.csv",
                    ),
                    (
                        input_data_dir,
                        "3_psql_db_v2_bw2/web_birdwatched_tweets.csv",
                    ),
                    (input_data_dir, "4_sqlite_csvs_bw2/api_tweets.csv"),
                    (input_data_dir, "4_sqlite_csvs_bw2/web_tweets.csv"),
                ]
            )

        if self.use_bookmark_tweets:
            self.tweet_stores.append(
                (full_pipeline_db_export_dir, "public.bookmark_tweets/1")
            )

        self.current_state = "__init__"

    def load(self):
        if self.current_state != "__init__":
            raise ValueError(
                f"TweetSnapshotProcessor cannot load from state '{self.current_state}'."
            )

        # Read metrics from disk
        self.tweets = {}
        for tweet_store_dir, tweet_store_name in self.tweet_stores:

            # Are we reading a parquet or a csv?
            is_csv = tweet_store_name.endswith(".csv") or tweet_store_name.endswith(
                ".csv.gz"
            )
            reader = dd.read_csv if is_csv else dd.read_parquet

            columns = [
                "tweet_id",
                "pulled_at",
                "created_at",
                "impressions",
                "retweets",
                "likes",
                "quotes",
                "replies",
            ]

            self.tweets[tweet_store_name] = reader(
                tweet_store_dir / tweet_store_name,
                dtype={
                    "tweet_id": str,
                    "pulled_at": str,
                    "created_at": str,
                    # Include these columns as they are contained in some files,
                    # and if we don't specify they raise an error
                    "author_id": object,
                    "id": object,
                },
                low_memory=False,
                assume_missing=True,
                blocksize=None if is_csv else "default",
            )

            # Rename columns (remove the n_ prefix, e.g. "n_likes" -> "likes")
            self.tweets[tweet_store_name] = self.tweets[tweet_store_name].rename(
                columns=lambda x: re.sub(r"^n_", "", x)
            )

            # Remove missing columns
            for col in columns:
                if col not in self.tweets[tweet_store_name].columns:
                    logger.warning(f"Column {col} not found in {tweet_store_name}.")
                    columns.remove(col)

            # Select right columns
            self.tweets[tweet_store_name] = self.tweets[tweet_store_name][columns]

            # Add source column
            self.tweets[tweet_store_name]["source"] = tweet_store_name

            # Filter to only the development tweets, if this is a dev run
            if self.dev:
                self.tweets[tweet_store_name] = self.tweets[tweet_store_name].loc[
                    self.tweets[tweet_store_name]["tweet_id"].isin(tweet_ids)
                ]

            # Drop any potential duplicates
            self.tweets[tweet_store_name] = self.tweets[
                tweet_store_name
            ].drop_duplicates()

            # Perform loading
            logger.info(f"Reading {tweet_store_name}.")
            self.tweets[tweet_store_name] = self.tweets[tweet_store_name].compute()

            # Convert to UTC timestamps
            timezone = get_timezone(tweet_store_name)
            time_cols = ["pulled_at"]
            if "created_at" in self.tweets[tweet_store_name].columns:
                time_cols.append("created_at")
            for time_column in time_cols:
                self.tweets[tweet_store_name][time_column] = (
                    pd.to_datetime(
                        self.tweets[tweet_store_name][time_column],
                        utc=False,
                        format="mixed",
                    )
                    .dt.tz_localize(timezone)
                    .dt.tz_convert("UTC")
                )

        # Join all types of tweets
        self.metrics = pd.concat(list(self.tweets.values()), axis=0)

        self.current_state = "load"

    def filter_and_log_stats(self, slap_times):
        if self.current_state != "load":
            raise ValueError(
                f"TweetSnapshotProcessor cannot filter_and_log_stats from state '{self.current_state}'."
            )

        self.slap_times = slap_times

        logger.info(
            f"Snapshot data contains {len(self.metrics):,} observations."
            f"There are currently {len(self.metrics['tweet_id'].unique()):,} unique tweets."
        )

        if self.max_date is not None:
            # Filter to tweet snapshots that happened on or before our max date
            self.metrics = self.metrics[
                self.metrics["pulled_at"] <= self.max_date
            ].reset_index(drop=True)
            logger.info(
                f"Filtered to snapshots that occurred on or before max_date {self.max_date}. Data now contains "
                f"{len(self.metrics):,} observations."
                f"There are now {len(self.metrics['tweet_id'].unique()):,} unique tweets."
            )

        # Get sources for each tweet
        for _, tweet_store_name in self.tweet_stores:
            self.metrics[f"in_{tweet_store_name}"] = (
                self.metrics["source"] == tweet_store_name
            )

        # Get sources for each tweet
        sources = (
            self.metrics.groupby(["tweet_id"])[
                [f"in_{tweet_name}" for _, tweet_name in self.tweet_stores]
            ]
            .max()
            .reset_index()
        )

        # Get lengths of datasets
        sizes_by_source = {
            tweet_name: self.tweets[tweet_name].shape[0]
            for _, tweet_name in self.tweet_stores
        }

        # Write out source for each tweet, in case we need it for later
        sources.to_csv(
            local_data_root / intermediate_dir / artifact_dir / "a_tweet_sources.csv",
            index=False,
        )

        contingency_table = sources.groupby(
            [f"in_{tweet_name}" for _, tweet_name in self.tweet_stores]
        ).size()
        contingency_table_path = (
            local_data_root
            / intermediate_dir
            / artifact_dir
            / "a_tweet_sources_contingency_table.csv"
        )
        contingency_table.to_csv(contingency_table_path)

        logger.info(
            f"Wrote out sources for each tweet. There were {len(sources):,} unique tweets. Of these, "
            + ", ".join(
                [
                    f"{len(sources[sources[f'in_{tweet_name}']]):,} were in the {tweet_name} store"
                    for _, tweet_name in self.tweet_stores
                ]
            )
            + f". The contingency table has been saved to {contingency_table_path}."
        )

        # Drop source dummy columns
        self.metrics = self.metrics.drop(
            columns=[f"in_{tweet_name}" for _, tweet_name in self.tweet_stores]
        )

        # Get When first/last timestamp occurred for tweet sources
        tweet_counts_by_source = self.metrics.groupby(["tweet_id", "source"]).agg(
            {
                "pulled_at": ["min", "max", "nunique"],
                "created_at": "first",
            }
        )

        # Condense column names from multiindex
        tweet_counts_by_source.columns = [
            "_".join(col).strip() for col in tweet_counts_by_source.columns.values
        ]

        # Calculate hours of data
        tweet_counts_by_source["hours_of_data"] = (
            tweet_counts_by_source["pulled_at_max"]
            - tweet_counts_by_source["pulled_at_min"]
        ).dt.total_seconds() / 3600
        tweet_counts_by_source["average_minutes_between_ts"] = (
            tweet_counts_by_source["hours_of_data"]
            / tweet_counts_by_source["pulled_at_nunique"]
            * 60
        )
        tweet_counts_by_source = tweet_counts_by_source.reset_index()
        tweet_counts_by_source = tweet_counts_by_source.pivot(
            index="tweet_id",
            columns="source",
            values=[
                "pulled_at_max",
                "pulled_at_min",
                "pulled_at_nunique",
                "created_at_first",
                "hours_of_data",
                "average_minutes_between_ts",
            ],
        )

        # Condense column names from multiindex again
        tweet_counts_by_source.columns = [
            "_".join(col).strip() for col in tweet_counts_by_source.columns.values
        ]

        # Get totals across all sources
        tweet_count_totals = self.metrics.groupby(["tweet_id"]).agg(
            {
                "pulled_at": ["min", "max", "nunique"],
                "created_at": "first",
            }
        )

        # Condense column names from multiindex
        tweet_count_totals.columns = [
            "_".join(col).strip() + "_total"
            for col in tweet_count_totals.columns.values
        ]

        # Calculate total hours of data
        tweet_count_totals["hours_of_data_total"] = (
            tweet_count_totals["pulled_at_max_total"]
            - tweet_count_totals["pulled_at_min_total"]
        ).dt.total_seconds() / 3600
        tweet_count_totals["average_minutes_between_ts_total"] = (
            tweet_count_totals["hours_of_data_total"]
            / tweet_count_totals["pulled_at_nunique_total"]
            * 60
        )
        tweet_count_totals = tweet_count_totals.reset_index()

        # Merge totals with source data
        tweet_counts_by_source = tweet_counts_by_source.merge(
            tweet_count_totals, on="tweet_id", how="left"
        )

        # Write out tweet counts by source
        tweet_counts_by_source.to_csv(
            local_data_root
            / intermediate_dir
            / artifact_dir
            / "a_snapshot_counts_by_source.csv",
            index=False,
        )

        # How many tweets were collected from each source on each day
        self.metrics["pulled_day"] = self.metrics["pulled_at"].dt.date
        day_counts = self.metrics[["source", "pulled_day"]].value_counts().reset_index()
        self.metrics = self.metrics.drop(columns=["pulled_day"])

        # Write out day counts
        day_counts.to_csv(
            local_data_root
            / intermediate_dir
            / artifact_dir
            / "a_snapshots_by_day.csv",
            index=False,
        )

        # What is the min/max time between timestamps for each tweet?
        sorted = self.metrics.sort_values(["tweet_id", "pulled_at"])
        time_between_all_sources = (
            sorted.set_index("tweet_id")
            .groupby(level=0)["pulled_at"]
            .diff()
            .dt.total_seconds()
            / 60
        ).reset_index()
        time_between_all_sources["source"] = "all_sources"
        time_between_by_source = (
            sorted.set_index(["tweet_id", "source"])
            .groupby(level=[0, 1])["pulled_at"]
            .diff()
            .dt.total_seconds()
            / 60
        ).reset_index()
        time_between_by_source = pd.concat(
            [time_between_by_source, time_between_all_sources],
            axis=0,
        )
        time_between_by_tweet = (
            time_between_by_source.groupby(["tweet_id", "source"])["pulled_at"]
            .agg(["min", "max"])
            .reset_index(drop=False)
        )
        time_between_by_tweet.columns = [
            "tweet_id",
            "source",
            "min_minutes_between_ts",
            "max_minutes_between_ts",
        ]

        # Write out time between timestamps
        time_between_by_tweet.to_csv(
            local_data_root / intermediate_dir / artifact_dir / "a_time_between_ts.csv",
            index=False,
        )

        # Merge with slap times
        with_slap_times = self.metrics.merge(
            self.slap_times, on="tweet_id", how="inner"
        )

        # How much data occurred before/after the slap?
        with_slap_times["pre_slap"] = (
            with_slap_times["pulled_at"] < with_slap_times["first_crh"]
        )

        # For each tweet/source, how much data do we have pre/post
        pre_post_counts_by_source = (
            with_slap_times.groupby(["tweet_id", "source", "pre_slap"])["pulled_at"]
            .count()
            .to_frame(name="count")
            .reset_index()
        )

        # For each tweet, how much data do we have pre/post
        pre_post_counts_total = (
            pre_post_counts_by_source.groupby(["tweet_id", "pre_slap"])["count"]
            .sum()
            .to_frame(name="count")
            .reset_index()
        )
        pre_post_counts_total["source"] = "all_sources"

        # Write out to csv
        pre_post_counts = pd.concat([pre_post_counts_by_source, pre_post_counts_total])
        pre_post_counts.to_csv(
            local_data_root
            / intermediate_dir
            / artifact_dir
            / "a_pre_slap_snapshot_counts.csv",
            index=False,
        )

        # How soon before/after slap do we have data?
        with_slap_times["time_since_slap"] = (
            with_slap_times["pulled_at"] - with_slap_times["first_crh"]
        )
        with_slap_times["hours_since_slap"] = (
            with_slap_times["time_since_slap"].dt.total_seconds() // 3600
        )
        with_slap_times["days_since_slap"] = with_slap_times["time_since_slap"].dt.days

        # For each tweet/source, how much data do we have at each hour since slap
        hour_counts_by_source = (
            with_slap_times.groupby(
                ["tweet_id", "source", "days_since_slap", "hours_since_slap"]
            )["pulled_at"]
            .count()
            .to_frame(name="count")
            .reset_index()
        )

        # For each tweet, how much data do we have at each hour since slap
        hour_counts_total = (
            hour_counts_by_source.groupby(
                ["tweet_id", "days_since_slap", "hours_since_slap"]
            )["count"]
            .sum()
            .reset_index()
        )
        hour_counts_total["source"] = "all_sources"

        # Write out to csv
        hour_counts = pd.concat([hour_counts_by_source, hour_counts_total])
        hour_counts.to_csv(
            local_data_root
            / intermediate_dir
            / artifact_dir
            / "a_snapshot_hours_since_slap_counts.csv",
            index=False,
        )

        # Make sure there are no tweets with multiple creation times
        if (self.metrics.groupby("tweet_id")["created_at"].nunique() > 1).any():
            logger.error("There are tweets with multiple creation times.")

        # Log dataset size
        logger.info(
            f"Loaded {len(self.metrics):,} observations in metrics dataset. "
            f"Data contains {len(self.metrics.columns)} columns. Of the observations in the metrics dataset, "
            + ", ".join(
                [
                    f"{sizes_by_source[tweet_name]:,} are from the {tweet_name} store"
                    for _, tweet_name in self.tweet_stores
                ]
            )
            + f"."
        )

        # Drop duplicates across sources
        pre_drop_size = len(self.metrics)
        self.metrics.sort_values(["tweet_id", "pulled_at"], inplace=True)
        self.metrics.drop_duplicates(
            subset=[
                "tweet_id",
                "pulled_at",
                "impressions",
                "retweets",
                "likes",
                "quotes",
                "replies",
            ],
            inplace=True
        )

        post_drop_size = len(self.metrics)
        post_drop_sizes_by_source = {
            tweet_name: self.metrics[self.metrics["source"] == tweet_name].shape[0]
            for _, tweet_name in self.tweet_stores
        }
        if pre_drop_size == post_drop_size:
            logger.info(
                "No observations were found to be duplicated between sources in the metrics dataset."
            )
        else:
            logger.info(
                f"Dropped {pre_drop_size - post_drop_size:,} observations due to duplication between sources "
                f"in the metrics data. Data now contains {post_drop_size:,} observations. "
                + ", ".join(
                    [
                        f"{post_drop_sizes_by_source[tweet_name] - sizes_by_source[tweet_name]:,} were dropped"
                        f" from the {tweet_name} store"
                        for _, tweet_name in self.tweet_stores
                    ]
                )
                + f"."
            )

        # Save config to data
        self.metrics["time_freq"] = self.time_freq
        self.metrics["dev"] = self.dev
        self.metrics["use_backup_tweets"] = self.use_backup_tweets
        self.metrics["use_bookmark_tweets"] = self.use_bookmark_tweets
        self.metrics["volatile_tweet_filtering"] = self.volatile_tweet_filtering
        self.metrics["max_date"] = self.max_date

        # Write out raw metrics, before data is interpolated or filtered further
        clear_and_write(
            self.metrics,
            local_data_root / intermediate_dir / "a_raw_metrics.parquet",
            config=config,
        )


        self.metrics = self.metrics.drop(
            columns=[
                "time_freq", "dev", "use_backup_tweets", "use_bookmark_tweets", "volatile_tweet_filtering",  "max_date"
            ]
        )

        self.filter_volatile_tweets()

        # Find creation time for each tweet
        creation_times_from_metrics = (
            self.metrics[["tweet_id", "created_at", "source"]].drop_duplicates()
        ).rename(columns={"source": "creation_time_source"})

        # Merge to creation times from tweet objects
        self.creation_times = (
            pd.concat(
                [
                    creation_times_from_metrics,
                    CreationTimeRetriever(dev=self.dev).creation_times,
                ]
            )
            .drop_duplicates(subset=["tweet_id", "created_at"])
            .reset_index(drop=True)
        )

        # Find tweets that have creation times
        non_na_creation_times = self.creation_times[
            self.creation_times["created_at"].notna()
        ]

        # Check for missing creation times (and make sure they don't have a creation time from another source)
        na_creation_times = self.creation_times[
            self.creation_times["created_at"].isna()
            & (~self.creation_times["tweet_id"].isin(non_na_creation_times["tweet_id"]))
        ]

        # Only use non-nas
        self.creation_times = non_na_creation_times

        if len(na_creation_times) > 0:
            logger.warning(
                f"Found {na_creation_times['tweet_id'].nunique():,} tweets with missing creation times out of "
                f"{len(self.creation_times):,} ({len(na_creation_times) / len(self.creation_times) *100:0.1f}%) "
                f"in metrics dataset. "
                f"Tweets with missing creation times: {na_creation_times['tweet_id'].to_list()}."
            )
            self.metrics = self.metrics[
                ~self.metrics["tweet_id"].isin(na_creation_times["tweet_id"])
            ]
            logger.warning(
                f"Filtered out tweets with missing creation times. Metrics data now contains {len(self.metrics):,} "
                f"observations from {len(self.creation_times):,} unique tweets. "
            )

        # Find tweets with multiple creation times
        duplicates = self.creation_times[
            self.creation_times["tweet_id"].duplicated(keep=False)
        ]
        # Log them, if found
        if len(self.creation_times) != self.creation_times["tweet_id"].nunique():
            logger.error(
                f"The following {duplicates['tweet_id'].nunique()} tweets "
                f"have more than one listed creation time. These have been written out to "
                f"a_multiple_creation_times.csv, and the minimum creation time has been selected."
            )
            duplicates.to_csv(
                local_data_root
                / intermediate_dir
                / artifact_dir
                / "a_multiple_creation_times.csv",
                index=False,
            )
            self.creation_times = (
                self.creation_times.groupby("tweet_id")["created_at"]
                .min()
                .reset_index()
            )

        # Save to disk
        self.creation_times.to_csv(
            local_data_root
            / intermediate_dir
            / artifact_dir
            / "a_tweet_creation_times.csv",
            index=False,
        )

        # Drop source col, which we no longer need
        self.creation_times = self.creation_times.drop(columns="creation_time_source")

        # Add any missing creation times
        self.metrics = self.metrics.drop(columns=["created_at"]).merge(
            self.creation_times, on="tweet_id", how="left"
        )

        self.current_state = "filter_and_log_stats"

    def filter_volatile_tweets(self):
        if self.volatile_tweet_filtering == "none":
            logger.info(f"Volatile tweets not filtered in this run.")
        elif self.volatile_tweet_filtering == "min_max_filter_25_3":
            for metric in ["likes", "retweets", "replies", "impressions"]:
                # Calculate pct change from one timestamp to the next
                self.metrics[f"pct_change_{metric}"] = self.metrics.groupby(["tweet_id"])[
                    metric
                ].pct_change()

                # Calculate difference from one timestamp to the next
                self.metrics[f"diff_{metric}"] = self.metrics.groupby(["tweet_id"])[metric].diff()

            # Get the min/max across all four metric
            self.metrics["min_diff"] = self.metrics[[f"diff_{metric}" for metric in ["likes", "retweets", "replies", "impressions"]]].min(axis=1)
            self.metrics["max_diff"] = self.metrics[[f"diff_{metric}" for metric in ["likes", "retweets", "replies", "impressions"]]].max(axis=1)
            self.metrics["min_pct_change"] = self.metrics[[f"pct_change_{metric}" for metric in ["likes", "retweets", "replies", "impressions"]]].min(axis=1)
            self.metrics["max_pct_change"] = self.metrics[[f"pct_change_{metric}" for metric in ["likes", "retweets", "replies", "impressions"]]].max(axis=1)

            biggest_change = self.metrics.groupby("tweet_id").agg(
                {
                    "min_diff": "min",
                    "max_diff": "max",
                    "min_pct_change": "min",
                    "max_pct_change": "max"
                }
            )

            # Rename columns
            biggest_change.columns = [
                "biggest_abs_drop",
                "biggest_abs_rise",
                "biggest_pct_drop",
                "biggest_pct_rise",
            ]

            # Define filtering criteria
            ABSOLUTE_CUTOFF = 25
            PCT_CUTOFF = 0.03

            # Apply filtering criteria
            bad_tids = biggest_change[
                (biggest_change["biggest_abs_drop"] < -ABSOLUTE_CUTOFF)
                & (biggest_change["biggest_abs_rise"] > ABSOLUTE_CUTOFF)
                & (biggest_change["biggest_pct_drop"] < -PCT_CUTOFF)
                & (biggest_change["biggest_pct_rise"] > PCT_CUTOFF)
            ].index

            # Filter out bad tids
            self.metrics = self.metrics[~self.metrics["tweet_id"].isin(bad_tids)]

            logger.info(f"There were {len(bad_tids):,} tweets that were filtered out due to large changes in "
                        f"engagement metrics (out of {len(biggest_change):,} total tweets at this point, so "
                        f"{len(bad_tids) / len(biggest_change) * 100:0.1f}% were filtered). ")


            # Drop extra columns
            self.metrics = self.metrics.drop(
                columns=[f"pct_change_{metric}" for metric in ["likes", "retweets", "replies", "impressions"]]
                        + [f"diff_{metric}" for metric in ["likes", "retweets", "replies", "impressions"]]
                        + ["min_diff", "max_diff", "min_pct_change", "max_pct_change"]
            )

    def interpolate(self):

        if self.current_state != "filter_and_log_stats":
            raise ValueError(
                f"TweetSnapshotProcessor cannot interpolate from state '{self.current_state}'."
            )

        # Get unique tweet ids before interpolation, so later
        # we can check if any tweet ids were dropped during interpolation
        pre_interpolation_tids = set(self.metrics["tweet_id"].unique())

        # Log how long between first and last data collection for each tweet
        time_periods = (
            self.metrics.groupby("tweet_id")["pulled_at"]
            .agg(["min", "max", "count"])
            .reset_index()
        )
        time_periods["time_between"] = time_periods["max"] - time_periods["min"]

        # Replace impressions values of 0 with NaN; Impressions cannot be 0, but due to pipeline issues
        # were sometimes recorded as such
        zero_impressions_idx = self.metrics["impressions"] == 0
        num_0_impressions = sum(zero_impressions_idx)
        tweets_with_zero_impressions = self.metrics[zero_impressions_idx][
            "tweet_id"
        ].nunique()
        sources_of_zero_impressions = self.metrics[zero_impressions_idx][
            "source"
        ].value_counts()
        self.metrics["impressions"] = self.metrics["impressions"].replace(0, np.nan)
        logger.info(
            f"Replaced {num_0_impressions:,} impressions values of 0 with NaN. These come from "
            f"{tweets_with_zero_impressions:,} unique tweets. The sources of these values are: "
            + ", ".join(
                [
                    f"{source}: {sources_of_zero_impressions[source]:,}"
                    for source in sources_of_zero_impressions.index
                ]
            )
        )

        # Round times to the nearest 15 min, and interpolate inbetween
        self.metrics = (
            self.metrics.groupby("tweet_id")
            .apply(
                _interpolate,
                include_groups=False,
                freq=self.time_freq,
                time_col="pulled_at",
                interpolation={
                    "time": [
                        "impressions",
                        "retweets",
                        "likes",
                        "quotes",
                        "replies",
                    ],
                    "ffill": ["source"],
                },
                stable_cols=["created_at"],
            )
            .rename(columns={"source": "most_recent_source"})
        )
        self.metrics.index = self.metrics.index.get_level_values(0)
        self.metrics = self.metrics.reset_index(drop=False)

        # Log data size
        logger.info(
            f"Interpolated metrics. Data now contains {len(self.metrics):,} observations."
            f"There are {len(self.metrics['tweet_id'].unique()):,} unique tweets."
        )

        # Log if any tweet ids were dropped during interpolation
        post_interpolation_tids = set(self.metrics["tweet_id"].unique())
        if pre_interpolation_tids != post_interpolation_tids:
            time_periods["dropped"] = time_periods["tweet_id"].isin(
                pre_interpolation_tids - post_interpolation_tids
            )
            logger.warning(
                f"Interpolation of metrics resulted in a change in the tweet ids in the dataset. For the tweets that "
                f"were dropped, the longest time between the first and last data collection was: "
                f"{time_periods[time_periods['dropped']]['time_between'].max()}, and the most number of "
                f"observations collected was: {time_periods[time_periods['dropped']]['count'].max()}. For "
                f"the tweets that were kept, the shortest time between the first and last data collection was: "
                f"{time_periods[~time_periods['dropped']]['time_between'].min()}, and the least number of "
                f"observations collected was: {time_periods[~time_periods['dropped']]['count'].min()}. The dropped "
                f"tweet ids were: {pre_interpolation_tids - post_interpolation_tids}."
            )
            time_periods[time_periods["dropped"]].to_csv(
                local_data_root
                / intermediate_dir
                / artifact_dir
                / "a_tweet_ids_dropped_in_interpolation.csv",
                index=False,
            )

        # Save config to data
        self.metrics["time_freq"] = self.time_freq
        self.metrics["dev"] = self.dev
        self.metrics["use_backup_tweets"] = self.use_backup_tweets
        self.metrics["use_bookmark_tweets"] = self.use_bookmark_tweets
        self.metrics["volatile_tweet_filtering"] = self.volatile_tweet_filtering
        self.metrics["max_date"] = self.max_date

        self.current_state = "interpolate"

    def save(self):
        """
        Save metrics to disk
        """
        if self.current_state != "interpolate":
            raise ValueError(
                f"TweetSnapshotProcessor cannot clear_and_write from state '{self.current_state}'."
            )

        output_path = local_data_root / intermediate_dir / "a_metrics.parquet"
        clear_and_write(
            self.metrics,
            output_path,
            {
                "time_freq": self.time_freq,
                "dev": self.dev,
                "use_backup_tweets": self.use_backup_tweets,
                "use_bookmark_tweets": self.use_bookmark_tweets,
                "volatile_tweet_filtering": self.volatile_tweet_filtering,
                "max_date": self.max_date,
            },
        )
        self.current_state = "clear_and_write"


########################################################################################################################
# class for loading note scores over time


class NoteHistoryPreprocessor:
    """
    Load complete note history.
    """

    def __init__(
        self, time_freq, dev, use_backup_tweets, use_bookmark_tweets, volatile_tweet_filtering, max_date
    ):
        """
        Load the history of each note, including timestamps for when the note was created, locked, etc.
        """
        # Save config to object
        self.time_freq = time_freq
        self.dev = dev
        self.use_backup_tweets = use_backup_tweets
        self.use_bookmark_tweets = use_bookmark_tweets
        self.volatile_tweet_filtering = volatile_tweet_filtering
        self.max_date = max_date

        self.current_state = "__init__"

    def load(self):
        if self.current_state != "__init__":
            raise ValueError(
                f"NoteHistoryPreprocessor cannot load from state '{self.current_state}'."
            )

        # Some files appear empty, so we filter those out here
        note_history_files = [
            f
            for f in glob(str(note_history_dir / "20??-??-??/noteStatusHistory*.tsv"))
            if os.path.getsize(f) > 0
        ]

        # Define columns with note status information
        status_columns = [
            "noteId",
            "createdAtMillis",
            "timestampMillisOfFirstNonNMRStatus",
            "firstNonNMRStatus",
            "timestampMillisOfLatestNonNMRStatus",
            "mostRecentNonNMRStatus",
            "timestampMillisOfStatusLock",
            "lockedStatus",
            "timestampMillisOfCurrentStatus",
            "currentStatus",
        ]

        # Read note history files
        nhf_dfs = []
        for nhf in tqdm(
            note_history_files, desc="Reading note history files", smoothing=0
        ):
            df = pd.read_csv(
                nhf,
                sep="\t",
                low_memory=False,
                dtype={c: str for c in status_columns},
                usecols=status_columns,
            )
            df["file"] = Path(nhf).parent.name
            nhf_dfs.append(df)

        # Merge note history files
        df_h = (
            pd.concat(nhf_dfs, ignore_index=True)
            .sort_values(["noteId", "timestampMillisOfCurrentStatus", "file"])
            .drop_duplicates(subset=status_columns, keep="first")
            .reset_index(drop=True)
        )

        # Save all files that notes were present in
        df_h[["noteId", "file"]].rename(
            columns={"noteId": "note_id"}
        ).drop_duplicates().to_csv(
            local_data_root / intermediate_dir / artifact_dir / "a_note_files.csv",
            index=False,
        )

        # Get creation times for all notes
        creation_events = df_h[["noteId", "createdAtMillis", "file"]].drop_duplicates(
            subset=["noteId", "createdAtMillis"]
        )

        na_creation_events = creation_events[
            creation_events["createdAtMillis"].isna().any()
            | creation_events["createdAtMillis"].isin(["-1", ""])
        ]

        # Find if there are any NaNs in the creation time
        if len(na_creation_events) > 0:
            logger.warning(
                f"Found {len(na_creation_events)} NaN note creation times. These "
                f"are being filtered out. "
                f"They were written out to a_na_creation_events.csv."
            )
            creation_events = creation_events[
                ~creation_events["createdAtMillis"].isna()
            ]
            na_creation_events.to_csv(
                local_data_root
                / intermediate_dir
                / artifact_dir
                / "a_na_creation_events.csv",
                index=False,
            )

        # Convert timestamp to datetime and add status column
        creation_events["timestamp"] = pd.to_datetime(
            creation_events["createdAtMillis"].astype(np.int64), unit="ms", utc=True
        )
        creation_events["status"] = "CREATED"

        # Find any notes with more than one creation time
        creation_event_dupes = creation_events[
            creation_events["noteId"].duplicated(keep=False)
        ]

        # If there are any, log this issue
        if len(creation_event_dupes) > 0:
            logger.error(
                f"Found {creation_event_dupes['noteId'].nunique()} noteIds with multiple creation_events. "
                f"Duplicates have been written to duplicated_creation_events.csv"
            )
            creation_event_dupes.to_csv(
                local_data_root
                / intermediate_dir
                / artifact_dir
                / "a_duplicated_creation_events.csv",
                index=False,
            )

        # Get first non-nmr status events
        first_non_nmr_events = df_h[
            (~df_h["timestampMillisOfFirstNonNMRStatus"].isin(["-1", ""]))
            & df_h["timestampMillisOfFirstNonNMRStatus"].notna()
        ][
            [
                "noteId",
                "timestampMillisOfFirstNonNMRStatus",
                "firstNonNMRStatus",
                "file",
            ]
        ].drop_duplicates(
            subset=["noteId", "timestampMillisOfFirstNonNMRStatus", "firstNonNMRStatus"]
        )

        # Get any notes with NaN status, and filter them out of the dataset
        na_statuses = first_non_nmr_events[
            first_non_nmr_events["firstNonNMRStatus"].isna()
            | first_non_nmr_events["firstNonNMRStatus"].isin(["-1", ""])
        ]
        if len(na_statuses) > 0:
            logger.warning(
                f"Found {len(na_statuses)} first non-NMR status events with NaN status. These are being filtered out. "
                f"They were written out to a_na_first_non_nmr_statuses.csv."
            )
            first_non_nmr_events = first_non_nmr_events[
                (~first_non_nmr_events["firstNonNMRStatus"].isna())
                & (~first_non_nmr_events["firstNonNMRStatus"].isin(["-1", ""]))
            ]
            na_statuses.to_csv(
                local_data_root
                / intermediate_dir
                / artifact_dir
                / "a_na_first_non_nmr_statuses.csv",
                index=False,
            )

        # Get timestamp as dt and add status column
        first_non_nmr_events["timestamp"] = pd.to_datetime(
            first_non_nmr_events["timestampMillisOfFirstNonNMRStatus"].astype(np.int64),
            unit="ms",
            utc=True,
        )
        first_non_nmr_events["status"] = first_non_nmr_events["firstNonNMRStatus"]

        # Log any notes with more than one first non-nmr status event
        first_non_nmr_dupes = first_non_nmr_events[
            first_non_nmr_events["noteId"].duplicated(keep=False)
        ]
        if len(first_non_nmr_dupes) > 0:
            logger.error(
                f"Found {first_non_nmr_dupes['noteId'].nunique()} noteIds with multiple first non-nmr events."
                f" Duplicates have been written to a_duplicated_first_non_nmr.csv"
            )
            first_non_nmr_dupes.to_csv(
                local_data_root
                / intermediate_dir
                / artifact_dir
                / "a_duplicated_first_non_nmr.csv",
                index=False,
            )

        # Get latest non-nmr status events
        latest_non_nmr_events = df_h[
            (~df_h["timestampMillisOfLatestNonNMRStatus"].isin(["-1", ""]))
            & df_h["timestampMillisOfLatestNonNMRStatus"].notna()
        ][
            [
                "noteId",
                "timestampMillisOfLatestNonNMRStatus",
                "mostRecentNonNMRStatus",
                "file",
            ]
        ].drop_duplicates(
            subset=[
                "noteId",
                "timestampMillisOfLatestNonNMRStatus",
                "mostRecentNonNMRStatus",
            ]
        )

        # Get any notes with NaN status, and filter them out of the dataset
        na_statuses = latest_non_nmr_events[
            latest_non_nmr_events["mostRecentNonNMRStatus"].isna()
            | latest_non_nmr_events["mostRecentNonNMRStatus"].isin(["-1", ""])
        ]
        if len(na_statuses) > 0:
            logger.warning(
                f"Found {len(na_statuses)} most recent non-NMR status events with NaN status. These are being filtered out. "
                f"They were written out to a_na_latest_non_nmr_statuses.csv."
            )
            latest_non_nmr_events = latest_non_nmr_events[
                (~latest_non_nmr_events["mostRecentNonNMRStatus"].isna())
                & (~latest_non_nmr_events["mostRecentNonNMRStatus"].isin("-1", ""))
            ]
            na_statuses.to_csv(
                local_data_root
                / intermediate_dir
                / artifact_dir
                / "a_na_latest_non_nmr_statuses.csv",
                index=False,
            )

        # Get timestamp as dt and add status column
        latest_non_nmr_events["timestamp"] = pd.to_datetime(
            latest_non_nmr_events["timestampMillisOfLatestNonNMRStatus"].astype(
                np.int64
            ),
            unit="ms",
            utc=True,
        )
        latest_non_nmr_events["status"] = latest_non_nmr_events[
            "mostRecentNonNMRStatus"
        ]

        # Log any notes with more than one latest non-nmr status event
        latest_non_nmr_dupes = latest_non_nmr_events[
            latest_non_nmr_events["noteId"].duplicated(keep=False)
        ]
        if len(latest_non_nmr_dupes) > 0:
            logger.error(
                f"Found {latest_non_nmr_dupes['noteId'].nunique()} noteIds with multiple latest non-nmr "
                f"statuses. Duplicates have been written to a_duplicated_latest_non_nmr.csv"
            )
            latest_non_nmr_dupes.to_csv("a_duplicated_latest_non_nmr.csv", index=False)

        # Get locked statuses
        locked_statuses = df_h[
            (~df_h["timestampMillisOfStatusLock"].isin(["-1", ""]))
            & df_h["timestampMillisOfStatusLock"].notna()
        ][
            ["noteId", "timestampMillisOfStatusLock", "lockedStatus", "file"]
        ].drop_duplicates(
            subset=["noteId", "timestampMillisOfStatusLock", "lockedStatus"]
        )

        # Get any notes with NaN status, and filter them out of the dataset
        na_statuses = locked_statuses[
            locked_statuses["lockedStatus"].isna()
            | locked_statuses["lockedStatus"].isin(["-1", ""])
        ]
        if len(na_statuses) > 0:
            logger.warning(
                f"Found {len(na_statuses)} locked status events with NaN status. These are being filtered out. "
                f"They were written out to a_na_locked_statuses.csv."
            )
            locked_statuses = locked_statuses[
                (~locked_statuses["lockedStatus"].isna())
                & (~locked_statuses["lockedStatus"].isin(["-1", ""]))
            ]
            na_statuses.to_csv(
                local_data_root
                / intermediate_dir
                / artifact_dir
                / "a_na_locked_statuses.csv",
                index=False,
            )

        # Get timestamp as dt and add status column
        locked_statuses["timestamp"] = pd.to_datetime(
            locked_statuses["timestampMillisOfStatusLock"].astype(np.int64),
            unit="ms",
            utc=True,
        )
        locked_statuses["status"] = locked_statuses["lockedStatus"]

        # Log any notes with more than one locked status event
        locked_status_dupes = locked_statuses[
            locked_statuses["noteId"].duplicated(keep=False)
        ]
        if len(locked_status_dupes) > 0:
            logger.error(
                f"Found {locked_status_dupes['noteId'].nunique()} noteIds with multiple creation_events. Duplicates have been written to duplicated_locked_statuses.csv"
            )
            locked_status_dupes.to_csv("duplicated_locked_statuses.csv", index=False)

        # Because locked statuses are valid in perpetuity, we need to forward fill them up to the current date
        present_day_locked_statuses = locked_statuses.copy()

        present_day_locked_statuses["max_timestamp"] = (
            self.max_date
            if self.max_date is not None
            else pd.to_datetime("now", utc=True)
        )
        present_day_locked_statuses = present_day_locked_statuses[
            present_day_locked_statuses["timestamp"]
            < present_day_locked_statuses["max_timestamp"]
        ]
        present_day_locked_statuses["timestamp"] = present_day_locked_statuses[
            "max_timestamp"
        ]
        present_day_locked_statuses = present_day_locked_statuses.drop(
            columns=["max_timestamp"]
        )

        # Concatenate locked statuses with present day locked statuses
        locked_statuses = pd.concat(
            [locked_statuses, present_day_locked_statuses]
        ).reset_index(drop=True)

        # Get current statuses
        current_statuses = df_h[
            (~df_h["timestampMillisOfCurrentStatus"].isin(["-1", ""]))
            & df_h["timestampMillisOfCurrentStatus"].notna()
        ][
            [
                "noteId",
                "timestampMillisOfCurrentStatus",
                "currentStatus",
                "timestampMillisOfStatusLock",
                "lockedStatus",
                "createdAtMillis",
                "file",
            ]
        ].drop_duplicates(
            subset=["noteId", "timestampMillisOfCurrentStatus", "currentStatus"]
        )

        # Get any notes with NaN status, and filter them out of the dataset
        na_statuses = current_statuses[
            current_statuses["currentStatus"].isna()
            | current_statuses["currentStatus"].isin(["-1", ""])
        ]
        if len(na_statuses) > 0:
            logger.warning(
                f"Found {len(na_statuses):,} current status events with NaN status. These are being filtered out. "
                f"They were written out to a_na_current_statuses.csv."
            )
            current_statuses = current_statuses[
                (~current_statuses["currentStatus"].isna())
                & (~current_statuses["currentStatus"].isin(["-1", ""]))
            ].copy()
            na_statuses.to_csv(
                local_data_root
                / intermediate_dir
                / artifact_dir
                / "a_na_current_statuses.csv",
                index=False,
            )

        # Convert current status timestamp to datetime
        current_statuses["timestampMillisOfCurrentStatus"] = pd.to_datetime(
            current_statuses["timestampMillisOfCurrentStatus"].astype(np.int64),
            unit="ms",
            utc=True,
        )

        # Get current statuses that have a lock time
        locked = current_statuses[
            (~current_statuses["timestampMillisOfStatusLock"].isin(["-1", ""]))
            & current_statuses["timestampMillisOfStatusLock"].notna()
        ].copy()
        # Convert lock time/creation time to dt
        locked["timestampMillisOfStatusLock"] = pd.to_datetime(
            locked["timestampMillisOfStatusLock"].astype(np.int64), unit="ms", utc=True
        )
        locked["createdAtMillis"] = pd.to_datetime(
            locked["createdAtMillis"].astype(np.int64), unit="ms", utc=True
        )

        # Get unlocked statuses
        unlocked = current_statuses[
            current_statuses["timestampMillisOfStatusLock"].isin(["-1", ""])
            | current_statuses["timestampMillisOfStatusLock"].isna()
        ].copy()

        # Get pre and post lock statuses
        pre_lock = locked[
            (
                locked["timestampMillisOfCurrentStatus"]
                <= (locked["timestampMillisOfStatusLock"] + pd.Timedelta("48 hours"))
            )
            & (
                locked["timestampMillisOfCurrentStatus"]
                <= (locked["createdAtMillis"] + pd.Timedelta("16 days"))
            )
        ]
        post_lock = locked[
            (
                locked["timestampMillisOfCurrentStatus"]
                > (locked["timestampMillisOfStatusLock"] + pd.Timedelta("48 hours"))
            )
            | (
                locked["timestampMillisOfCurrentStatus"]
                > (locked["createdAtMillis"] + pd.Timedelta("16 days"))
            )
        ]

        post_lock_changes = post_lock[
            (post_lock["currentStatus"] != post_lock["lockedStatus"])
        ]
        if len(post_lock_changes) > 0:
            logger.error(
                f"Found {len(post_lock_changes):,} cases with statuses that were changed after being locked. "
                f"These cases have been written to a_post_lock_changes.csv. The locked status will be used in these "
                f"cases."
            )
            post_lock_changes.to_csv(
                local_data_root
                / intermediate_dir
                / artifact_dir
                / "a_post_lock_changes.csv",
                index=False,
            )

        # Get all current statuses that occured pre lock
        current_statuses = pd.concat([unlocked, pre_lock], ignore_index=True)[
            ["noteId", "timestampMillisOfCurrentStatus", "currentStatus", "file"]
        ].drop_duplicates(
            subset=["noteId", "timestampMillisOfCurrentStatus", "currentStatus"]
        )

        # Define timestamp and status cols
        current_statuses["timestamp"] = current_statuses[
            "timestampMillisOfCurrentStatus"
        ]
        current_statuses["status"] = current_statuses["currentStatus"]

        # Get current status dupes
        current_status_dupes = current_statuses[
            current_statuses[["noteId", "file"]].duplicated(keep=False)
        ]
        if len(current_status_dupes) > 0:
            logger.error(
                f"Found noteIds with multiple current statuses in a single file. Duplicates have been "
                f"written to a_duplicated_current_statuses.csv"
            )

            current_status_dupes.to_csv(
                local_data_root
                / intermediate_dir
                / artifact_dir
                / "a_duplicated_current_statuses.csv",
                index=False,
            )

        # Concatenate all status events
        self.note_history = (
            pd.concat(
                [
                    creation_events,
                    first_non_nmr_events,
                    latest_non_nmr_events,
                    locked_statuses,
                    current_statuses,
                ],
                ignore_index=True,
            )[["noteId", "timestamp", "status", "file"]]
            .sort_values(["noteId", "timestamp"], ascending=True)
            .reset_index(drop=True)
        ).rename(columns={"noteId": "note_id"})

        # Log size of dataset
        logger.info(
            f"Created note history dataframe with {len(self.note_history):,} observations."
            f"There are {len(self.note_history['note_id'].unique()):,} unique notes."
        )

        # Log the paths that the data was read from
        logger.info(f"Data was read from the following paths: {note_history_files}.")

        # Join to tweet ids
        prev_note_history_len = len(self.note_history)
        prev_note_ids = (
            self.note_history.groupby("note_id")["timestamp"]
            .agg(["min", "max", "count"])
            .reset_index()
        )

        # Some files appear empty, so we filter those out here
        note_tweet_files = [
            f
            for f in glob(str(note_history_dir / "20??-??-??/notes-*.tsv"))
            if os.path.getsize(f) > 0
        ]

        logger.info(f"Reading note/tweet pairs from files: {note_tweet_files}.")

        # Read note/tweet pairs

        note_tweet_pairs = pd.DataFrame(columns=["note_id", "tweet_id"])

        for ntf in note_tweet_files:
            note_tweet_pairs = (
                pd.concat(
                    [
                        note_tweet_pairs,
                        pd.read_csv(
                            ntf,
                            sep="\t",
                            dtype={"noteId": str, "tweetId": str},
                            usecols=["noteId", "tweetId"],
                            low_memory=False,
                            # index_col=False,
                        ).rename(columns={"noteId": "note_id", "tweetId": "tweet_id"}),
                    ]
                )
                .drop_duplicates()
                .reset_index(drop=True)
            )

        # Log first/last timestamp for notes that did not make merge
        dropped_note_ids = prev_note_ids[
            ~prev_note_ids["note_id"].isin(note_tweet_pairs["note_id"])
        ]
        if len(dropped_note_ids) > 0:
            logger.warning(
                f"There were {len(dropped_note_ids):,} notes that did not have a corresponding tweet id."
            )

        # Log size of note/tweet pairs
        messager = (
            logger.info
            if len(note_tweet_pairs) - note_tweet_pairs["note_id"].nunique() == 0
            else logger.warning
        )
        messager(
            f"Read note-tweet pairs from notes-*.tsv files. There were {len(note_tweet_pairs):,} pairs, "
            f"corresponding to {note_tweet_pairs['note_id'].nunique():,} notes and "
            f"{note_tweet_pairs['tweet_id'].nunique():,} tweets. "
            f"This means that there were "
            f"{len(note_tweet_pairs) - note_tweet_pairs['note_id'].nunique():,} notes with multiple tweet ids."
        )

        # Save note tweet pairs
        note_tweet_pairs.to_csv(
            local_data_root
            / intermediate_dir
            / artifact_dir
            / "a_note_tweet_pairs.csv",
            index=False,
        )

        self.note_history = self.note_history.merge(note_tweet_pairs, on="note_id")

        logger.info(
            "Merged note history with note-tweet pairs. Data now contains "
            f"{len(self.note_history):,} observations. This means that "
            f"{prev_note_history_len - len(self.note_history):,} observations were dropped "
            f"for not having a corresponding tweet id. "
            f"There are {len(self.note_history['note_id'].unique()):,} unique notes now, "
            f"which correspond to {len(self.note_history['tweet_id'].unique()):,} unique tweets."
        )

        # Filter to tweet Ids we're interested in, if this is dev version
        if self.dev:
            self.note_history = self.note_history[
                self.note_history["tweet_id"].isin(tweet_ids)
            ]

            logger.info(
                f"Filtered note history to only include tweets we're interested in. Data now contains "
                f"{len(self.note_history):,} observations. There are now {len(self.note_history['note_id'].unique()):,} unique notes "
                f"and {len(self.note_history['tweet_id'].unique()):,} unique tweets."
            )

        self.note_history = self.note_history.sort_values(
            ["tweet_id", "note_id", "timestamp"]
        )

        if self.max_date is not None:
            # Filter to note history events that happened on or before our max date
            self.note_history = self.note_history[
                self.note_history["timestamp"] <= self.max_date
            ].reset_index(drop=True)
            logger.info(
                f"Filtered to note history that occurred on or before max_date {self.max_date}. Data now contains "
                f"{len(self.note_history):,} observations."
                f"There are {len(self.note_history['note_id'].unique()):,} unique notes now, "
                f"which correspond to {len(self.note_history['tweet_id'].unique()):,} unique tweets."
            )

        # Save config to data
        self.note_history["time_freq"] = self.time_freq
        self.note_history["dev"] = self.dev
        self.note_history["use_backup_tweets"] = self.use_backup_tweets
        self.note_history["use_bookmark_tweets"] = self.use_bookmark_tweets
        self.note_history["volatile_tweet_filtering"] = self.volatile_tweet_filtering
        self.note_history["max_date"] = self.max_date

        self.note_history = self.note_history.drop(columns=["file"])

        # Write out raw note history, before data is interpolated or filtered further
        clear_and_write(
            self.note_history,
            local_data_root / intermediate_dir / "a_raw_note_history.parquet",
            config=config,
        )

        self.note_history = self.note_history.drop(
            columns=[
                "time_freq", "dev", "use_backup_tweets", "use_bookmark_tweets", "volatile_tweet_filtering", "max_date"
            ]
        )

        self.current_state = "load"

    def log_slap_times(self):
        if self.current_state != "load":
            raise ValueError(
                f"NoteHistoryPreprocessor cannot log_slap_times from state '{self.current_state}'."
            )

        self.note_history = self.note_history.drop_duplicates()

        logger.info(
            f"Dropped duplicates in note history. Data now contains "
            f"{len(self.note_history):,} observations. There are now {len(self.note_history['note_id'].unique()):,} unique notes "
            f"and {len(self.note_history['tweet_id'].unique()):,} unique tweets."
        )

        self.note_history = self.note_history[self.note_history["status"].notna()]

        logger.info(
            f"Filtered out NA statuses in note history. Data now contains "
            f"{len(self.note_history):,} observations. There are now {len(self.note_history['note_id'].unique()):,} unique notes "
            f"and {len(self.note_history['tweet_id'].unique()):,} unique tweets."
        )

        # Write out actual slap times, in case we need them for analysis later
        self.slap_times = (
            self.note_history[self.note_history["status"] == "CURRENTLY_RATED_HELPFUL"]
            .sort_values(["tweet_id", "timestamp"])
            .groupby("tweet_id")["timestamp"]
            .first()
            .reset_index()
        ).rename(columns={"timestamp": "first_crh"})

        self.slap_times.to_csv(
            local_data_root / intermediate_dir / artifact_dir / "a_slap_times.csv",
            index=False,
        )

        # Find creation time for each tweet, in case we need them for later
        self.note_creation_times = (
            self.note_history[self.note_history["status"] == "CREATED"]
            .sort_values(["tweet_id", "note_id", "timestamp"])
            .groupby(["tweet_id", "note_id"])["timestamp"]
            .first()
            .reset_index()
        ).rename(columns={"timestamp": "note_created_at"})

        self.note_creation_times.to_csv(
            local_data_root
            / intermediate_dir
            / artifact_dir
            / "a_note_creation_times.csv",
            index=False,
        )

        # Record the shortest time between two timestamps for any note
        shortest_time = (
            self.note_history.sort_values(
                ["tweet_id", "note_id", "timestamp"], ascending=True
            )
            .groupby(["tweet_id", "note_id"])["timestamp"]
            .diff()
            .min()
        )

        logger.info(
            "In the note history dataframe, the shortest time between two timestamps for any note is "
            f"{shortest_time}. If this is less than {self.time_freq}, it means it's possible for a note to be CRH then "
            f"NCRH in our window, which would result in simple forward filling missing the CRH status."
        )

        # Get duplicated snapshots
        duplicated_notes = self.note_history[
            self.note_history.duplicated(
                subset=["tweet_id", "note_id", "timestamp"], keep=False
            )
        ]

        # Log duplicated snapshots
        if len(duplicated_notes) > 0:
            logger.warning(
                f"Found {len(duplicated_notes):,} duplicated snapshots in the note history dataset. "
                f"These are the first 5 duplicated snapshots:\n{duplicated_notes.head()}. They have been written out "
                f"to {local_data_root / intermediate_dir / 'a_duplicated_snapshots.csv'}."
            )
            duplicated_notes.to_csv(
                local_data_root
                / intermediate_dir
                / artifact_dir
                / "a_duplicated_snapshots.csv",
                index=False,
            )

        self.current_state = "log_slap_times"

    def filter(self, creation_times):
        if self.current_state != "log_slap_times":
            raise ValueError(
                f"NoteHistoryPreprocessor cannot filter from state '{self.current_state}'."
            )

        self.creation_times = creation_times

        # Filter out notes without creation times
        pre_filter_len = len(self.note_history)
        self.note_history = self.note_history[
            self.note_history["tweet_id"].isin(self.creation_times["tweet_id"].values)
        ].reset_index(drop=True)
        if len(self.note_history) != pre_filter_len:
            logger.info(
                f"Filtered note history to only include tweets that have a creation time. Data now contains "
                f"{len(self.note_history):,} observations. There are now {len(self.note_history['note_id'].unique()):,} unique notes "
                f"and {len(self.note_history['tweet_id'].unique()):,} unique tweets."
            )

        # Get size of dataset, so we can check if it changes in merge
        pre_merge_len = len(self.note_history)

        # Merge to creation time
        self.note_history = self.note_history.merge(
            self.creation_times, on="tweet_id", how="left"
        )

        self.note_history = self.note_history.merge(
            self.note_creation_times[["note_id", "note_created_at"]],
            on="note_id",
            how="left",
        )

        # Get size of dataset after merge
        post_merge_len = len(self.note_history)

        # Raise error if merge changed the length of the dataset
        if pre_merge_len != post_merge_len:
            logger.error(
                "Merge of note history to creation times resulted in a change in the length of the dataset."
            )

        self.current_state = "filter"

    def interpolate(self):
        if self.current_state != "filter":
            raise ValueError(
                f"NoteHistoryPreprocessor cannot interpolate from state '{self.current_state}'."
            )

        # We treat a note as CRH if it ever had a CRH status in the period that is being interpolated
        status_codes = {
            "CURRENTLY_RATED_HELPFUL": 3,
            "CURRENTLY_RATED_NOT_HELPFUL": 2,
            "NEEDS_MORE_RATINGS": 1,
            "CREATED": 0,
        }
        self.note_history["numeric_status"] = self.note_history["status"].map(
            status_codes
        )

        # Log how long between first and last note history event for each note
        pre_interpolation_nids = set(self.note_history["note_id"].unique())
        time_periods = (
            self.note_history.groupby(["tweet_id", "note_id"])["timestamp"]
            .agg(["min", "max", "count"])
            .reset_index()
        )
        time_periods["time_between"] = time_periods["max"] - time_periods["min"]

        # Interpolate in between notes
        self.note_history = self.note_history.groupby(["tweet_id", "note_id"]).apply(
            _interpolate,
            include_groups=False,
            freq=self.time_freq,
            time_col="timestamp",
            interpolation={
                "ffill": ["status"],
                "max_ffill": ["numeric_status"],
            },
            stable_cols=["note_created_at", "created_at"],
        )

        self.note_history["tweet_id"] = self.note_history.index.get_level_values(0)
        self.note_history["note_id"] = self.note_history.index.get_level_values(1)
        self.note_history = self.note_history.reset_index(drop=True)
        logger.info(
            f"Interpolated note history. Data now contains {len(self.note_history):,} observations."
            f"There are {len(self.note_history['note_id'].unique()):,} unique notes. There are "
            f"{len(self.note_history['tweet_id'].unique()):,} unique tweets."
        )

        # Log if any notes ids were dropped during interpolation
        post_interpolation_nids = set(self.note_history["note_id"].unique())
        if pre_interpolation_nids != post_interpolation_nids:
            time_periods["dropped"] = time_periods["note_id"].isin(
                pre_interpolation_nids - post_interpolation_nids
            )
            logger.warning(
                f"Interpolation of note history resulted in a change in the note ids in the dataset. For the notes that "
                f"were dropped, the longest time between the first and last data collection was: "
                f"{time_periods[time_periods['dropped']]['time_between'].max()}, and the most number of "
                f"observations collected was: {time_periods[time_periods['dropped']]['count'].max()}. For "
                f"the notes that were kept, the shortest time between the first and last data collection was: "
                f"{time_periods[~time_periods['dropped']]['time_between'].min()}, and the least number of "
                f"observations collected was: {time_periods[~time_periods['dropped']]['count'].min()}. The dropped "
                f"note ids were: {pre_interpolation_nids - post_interpolation_nids}."
            )
            time_periods[time_periods["dropped"]].to_csv(
                local_data_root
                / intermediate_dir
                / artifact_dir
                / "a_note_ids_dropped_in_interpolation.csv",
                index=False,
            )

        self.note_history["max_status"] = self.note_history["numeric_status"].map(
            {v: k for k, v in status_codes.items()}
        )

        if (self.note_history["status"] != self.note_history["max_status"]).any():
            logger.warning(
                "Forward filling with max status resulted in different status than"
                "simple forward filling. This should not happen if there is only ever 1 "
                f"status per {self.time_freq} period."
            )

        # Use max status to fill in status
        self.note_history["status"] = self.note_history["max_status"]
        self.note_history = self.note_history.drop(
            columns=["max_status", "numeric_status"]
        )

        # Add config info
        self.note_history["time_freq"] = self.time_freq
        self.note_history["dev"] = self.dev
        self.note_history["use_backup_tweets"] = self.use_backup_tweets
        self.note_history["use_bookmark_tweets"] = self.use_bookmark_tweets
        self.note_history["volatile_tweet_filtering"] = self.volatile_tweet_filtering
        self.note_history["max_date"] = self.max_date

        self.current_state = "interpolate"

    def save(self):
        """
        Save note_history to disk
        """
        if self.current_state != "interpolate":
            raise ValueError(
                f"NoteHistoryPreprocessor cannot clear_and_write from state '{self.current_state}'."
            )

        output_path = local_data_root / intermediate_dir / "a_note_history.parquet"
        clear_and_write(
            self.note_history,
            output_path,
            {
                "time_freq": self.time_freq,
                "dev": self.dev,
                "use_backup_tweets": self.use_backup_tweets,
                "use_bookmark_tweets": self.use_bookmark_tweets,
                "volatile_tweet_filtering": self.volatile_tweet_filtering,
                "max_date": self.max_date,
            },
        )

        self.current_state = "clear_and_write"


########################################################################################################################


class StructuralMetricsProcessor(object):
    """
    This class turns the structural metrics data, stored as json, into parquet tables that can be read
    more easily later in the pipeline.
    """

    def __init__(self, dev, time_freq, use_backup_tweets, use_bookmark_tweets, volatile_tweet_filtering, max_date):
        # Save config data to object
        self.dev = dev
        self.time_freq = time_freq
        self.use_backup_tweets = use_backup_tweets
        self.use_bookmark_tweets= use_bookmark_tweets
        self.volatile_tweet_filtering=volatile_tweet_filtering
        self.max_date = max_date
        self.keys = structural_metrics_keys

        self.data_columns = structural_metrics_columns
        self.current_state = "__init__"

    def load(self, creation_times):
        # Load all structural metrics files
        if self.current_state != "__init__":
            raise ValueError(
                f"StructuralMetricsProcessor cannot load from state '{self.current_state}'."
            )
        self.creation_times = creation_times

        full_matches = set(os.listdir(structural_metrics_dir))
        partial_matches = set(os.listdir(partial_structural_metrics_dir))
        structural_metrics_fps = list(full_matches.union(partial_matches))
        counter = defaultdict(lambda: 0)

        for fp in tqdm(
            structural_metrics_fps,
            smoothing=0,
            desc="Loading structural metrics",
        ):
            tweet_id = fp.replace(".json.gz", "")
            if self.dev:
                if tweet_id not in tweet_ids.values:
                    continue
            if fp in full_matches:
                structural_metrics = json_gzip_reader(structural_metrics_dir / fp)
                assert fp not in partial_matches
                counter["full_matches"] += 1
            else:
                structural_metrics = json_gzip_reader(
                    partial_structural_metrics_dir / fp
                )
                assert fp not in full_matches
                counter["partial_matches"] += 1

            # Unload dictionaries
            structural_metrics = self._unload_structural_metrics_dict(
                structural_metrics
            )

            for ds_new_name in self.keys.values():
                if ds_new_name in structural_metrics.keys():
                    counter[ds_new_name] += 1
                    save_dir = local_data_root / intermediate_dir / f"a_{ds_new_name}"
                    save_dir.mkdir(exist_ok=True, parents=True)
                    structural_metrics[ds_new_name]["time_freq"] = self.time_freq
                    structural_metrics[ds_new_name]["dev"] = self.dev
                    structural_metrics[ds_new_name]["use_backup_tweets"] = self.use_backup_tweets
                    structural_metrics[ds_new_name]["max_date"] = self.max_date
                    structural_metrics[ds_new_name]["use_bookmark_tweets"] = self.use_bookmark_tweets
                    structural_metrics[ds_new_name]["volatile_tweet_filtering"] = self.volatile_tweet_filtering
                    clear_and_write(
                        structural_metrics[ds_new_name],
                        save_dir / f"{tweet_id}.parquet",
                        {
                            "time_freq": self.time_freq,
                            "dev": self.dev,
                            "use_backup_tweets": self.use_backup_tweets,
                            "max_date": self.max_date,
                            "use_bookmark_tweets": self.use_bookmark_tweets,
                            "volatile_tweet_filtering": self.volatile_tweet_filtering,
                        },
                        logger=None,
                    )


        logger.info(
            f"There were {counter['full_matches']:,} tweets in {structural_metrics_dir}."
        )
        logger.info(
            f"There were {counter['partial_matches']:,} tweets in {partial_structural_metrics_dir}."
        )
        for key in self.keys.values():
            logger.info(f"There were {counter[key]:,} tweets with {key} metrics.")

        self.current_state = "load"

    def _unload_structural_metrics_dict(self, structural_metrics_dict):
        results_dict = {}
        for ds_original_name, ds_new_name in self.keys.items():
            if ds_original_name in structural_metrics_dict.keys():
                results_dict[ds_new_name] = pd.DataFrame(
                    structural_metrics_dict[ds_original_name]
                )
                results_dict[ds_new_name]["tweet_id"] = str(
                    structural_metrics_dict["conversation_id"]
                )

            # If there are not observations, skip this key
            if len(results_dict[ds_new_name]) == 0:
                del results_dict[ds_new_name]
                continue

            # Convert to time
            results_dict[ds_new_name]["time"] = pd.to_datetime(
                results_dict[ds_new_name]["time"], utc=True, format="%Y-%m-%dT%H:%M:%S"
            )

            # Join to creation times
            results_dict[ds_new_name] = results_dict[ds_new_name].merge(
                self.creation_times, on="tweet_id", how="left"
            )

            # Interpolate
            results_dict[ds_new_name] = self._interpolate(results_dict[ds_new_name])

            # Drop unneeded columns
            results_dict[ds_new_name] = results_dict[ds_new_name][
                [
                    "tweet_id",
                    "time",
                    "created_at",
                ]
                + [
                    c
                    for c in self.data_columns
                    if c in results_dict[ds_new_name].columns
                ]
            ]

            # Rename data columns
            results_dict[ds_new_name] = results_dict[ds_new_name].rename(
                columns={
                    c: f"{ds_new_name}_{c}"
                    for c in self.data_columns
                    if c in results_dict[ds_new_name].columns
                }
            )

            if self.max_date is not None:
                # Filter to data that happened on or before our max date
                results_dict[ds_new_name] = results_dict[ds_new_name][
                    results_dict[ds_new_name]["time"] <= self.max_date
                ].reset_index(drop=True)

        return results_dict

    def _interpolate(self, tweet_df):
        # Perform interpolation
        tweet_df = _interpolate(
            tweet_df,
            freq=self.time_freq,
            time_col="time",
            interpolation={
                "ffill": [c for c in self.data_columns if c in tweet_df.columns]
            },
            stable_cols=[
                "created_at",
                "tweet_id",
            ],
        )

        return tweet_df


########################################################################################################################


class CalculatedMetricsProcessor:
    def __init__(self, time_freq, dev, use_backup_tweets, use_bookmark_tweets, volatile_tweet_filtering, max_date):
        # save config to object
        self.time_freq = time_freq
        self.dev = dev
        self.use_backup_tweets = use_backup_tweets
        self.use_bookmark_tweets = use_bookmark_tweets
        self.volatile_tweet_filtering = volatile_tweet_filtering
        self.max_date = max_date

        # Create dictionary to store rts/replies in
        self.metrics = {}

        self.current_state = "__init__"

    def load_data_collection_times(self, creation_times, slap_times):
        if self.current_state != "__init__":
            raise ValueError(
                f"CalculatedMetricsProcessor cannot load from state '{self.current_state}'."
            )

        # Save creation times and slap times to object
        self.creation_times = creation_times
        self.slap_times = slap_times

        # Load files with timestamps for when replies were collected
        reply_ts_path = (
            calculated_metrics_dir / "conversations_file_creations_df.json.gz"
        )
        reply_collection_times = pd.read_json(
            reply_ts_path, dtype={"root_tweet_id": str}, lines=True
        ).rename(columns={"root_tweet_id": "root_id"})

        # Convert to dt
        reply_collection_times["file_created_at"] = pd.to_datetime(
            reply_collection_times["file_created_at"],
            utc=True,
            format="%Y-%m-%dT%H:%M:%S.%fZ",
        )

        # Load files with timestamps for when rts were collected
        retweet_ts_path = calculated_metrics_dir / "retweets_file_creations_df.json.gz"
        retweet_collection_times = pd.read_json(
            retweet_ts_path, dtype={"root_tweet_id": str}, lines=True
        ).rename(columns={"root_tweet_id": "root_id"})

        # Find final data collection times for both RTs/Replies
        self.final_data_collection_times = {
            "replies": (
                reply_collection_times.groupby("root_id")["file_created_at"]
                .max()
                .reset_index()
            ).rename(columns={"file_created_at": "final_data_collection"}),
            "retweets": (
                retweet_collection_times.groupby("root_id")["file_created_at"]
                .max()
                .reset_index()
            ).rename(columns={"file_created_at": "final_data_collection"}),
        }

        # Save variable names/paths of replies/retweets to dictionaries
        self.paths = {
            "replies": calculated_metrics_dir / "conversations_df.json.gz",
            "retweets": calculated_metrics_dir / "retweets_df.json.gz",
        }
        self.root_tweet_cols = {
            "replies": "conversation_id",
            "retweets": "retweeted_tweet_id",
        }
        self.leaf_tweet_cols = {"replies": "tweet_id", "retweets": "rt_tweet_id"}
        self.creation_time_cols = {
            "replies": "tweet_created_at",
            "retweets": "rt_created_at",
        }

        # Save current state
        self.current_state = "load_data_collection_times"

    def load_metrics(self):
        # Perform loading for both RTs and replies
        for metric_name in ["replies", "retweets"]:
            if self.current_state != "load_data_collection_times":
                raise ValueError(
                    f"CalculatedMetricsProcessor cannot load from state '{self.current_state}'."
                )

            # Get objects to store tweet data in
            all_tweet_observations = defaultdict(list)

            # Create variable to store last conversation id, so we can tell when we've moved on to a new one
            last_conversation_id = None

            # Open and read metrics file
            with gzip.open(self.paths[metric_name], "rb") as f:
                # Iterate through each line of the file
                for line in tqdm(f, smoothing=0, desc=f"Processing {metric_name}"):

                    # Load the line as dict
                    complete_data = json.loads(line)

                    conversation_id = complete_data[self.root_tweet_cols[metric_name]]

                    # Extract root tweet, leaf tweet, tweet creation time, and file creation time
                    all_tweet_observations[conversation_id].append(
                        {
                            "response_id": complete_data[
                                self.leaf_tweet_cols[metric_name]
                            ],
                            "timestamp": complete_data[
                                self.creation_time_cols[metric_name]
                            ],
                            "file_created_at": complete_data["file_created_at"],
                        }
                    )

                    # If the conversation id has changed, consider flushing the data
                    # (i.e. if this is a dev run, delete the read data if it isn't in the dev set)
                    if (last_conversation_id is not None) and (
                        last_conversation_id != conversation_id
                    ):
                        if self.dev and (last_conversation_id not in tweet_ids.values):
                            del all_tweet_observations[last_conversation_id]

                    # Save previous conversation id
                    last_conversation_id = conversation_id

                # Consider deleting the last conversation if this is a dev run
                if self.dev and (conversation_id not in tweet_ids.values):
                    del all_tweet_observations[conversation_id]

            # Create metadata to track data size throughout processing
            metadata = defaultdict(lambda: 0)
            metadata["missing_creation_time_ids"] = []

            valid_tweets = list(all_tweet_observations.keys())
            for conversation_id in valid_tweets:
                # Log that we've found another tweet
                metadata["unique_tweets"] += 1

                tweet_df = pd.DataFrame(all_tweet_observations[conversation_id])
                tweet_df["tweet_id"] = conversation_id

                # Process the conversation
                metadata, tweet_df = self._process_metric(
                    metric_name,
                    tweet_df,
                    metadata,
                    self.final_data_collection_times[metric_name],
                    conversation_id,
                )

                if tweet_df is None:
                    del all_tweet_observations[conversation_id]

                # Add the conversation to the list of all conversations
                all_tweet_observations[conversation_id] = tweet_df

            # Log data
            logger.info(
                f"There were {metadata['missing_creation_time']:,} "
                f"tweets missing creation times."
                + (
                    f" These were removed. The missing tweet ids were: {metadata['missing_creation_time_ids']}."
                    if metadata["missing_creation_time"] > 0
                    else ""
                )
            )
            logger.info(
                f"Processed {metadata['original_count']:,} {metric_name} from {metadata['unique_tweets']:,} "
                f"unique tweets."
            )
            logger.info(
                f"{metric_name} data contained {metadata['post_deduplication']:,} records after deduplication (by "
                f"{self.root_tweet_cols[metric_name]} and {self.leaf_tweet_cols[metric_name]})."
            )
            logger.info(
                f"There were {metadata['post_creation_time_observations']:,} observations in {metric_name} data"
                f"after merging to creation times."
            )
            logger.info(
                f"There were {metadata['post_interpolation_observations']:,} observations in {metric_name} data "
                f"after forward filling to every {self.time_freq}."
            )
            logger.info(
                f"There were {metadata['post_date_filter_observations']:,} observations in {metric_name} data after "
                f"filtering to only include observations that occurred on or before the max date."
            )

            # Join data together
            self.metrics[metric_name] = pd.concat(
                all_tweet_observations.values()
            ).reset_index(drop=True)

            # Add config info
            self.metrics[metric_name]["time_freq"] = self.time_freq
            self.metrics[metric_name]["dev"] = self.dev
            self.metrics[metric_name]["use_backup_tweets"] = self.use_backup_tweets
            self.metrics[metric_name]["use_bookmark_tweets"] = self.use_bookmark_tweets
            self.metrics[metric_name]["volatile_tweet_filtering"] = self.volatile_tweet_filtering
            self.metrics[metric_name]["max_date"] = self.max_date
            self.metrics[metric_name] = self.metrics[metric_name].drop(
                columns="file_created_at"
            )

        self.current_state = "load_metrics"

    def save(self):
        """
        Save calculated metrics to disk
        """
        if self.current_state != "load_metrics":
            raise ValueError(
                f"CalculatedMetricsProcessor cannot save from state '{self.current_state}'."
            )

        for metric_name in ["replies", "retweets"]:
            # Get path
            output_path = (
                local_data_root
                / intermediate_dir
                / f"a_calculated_{metric_name}.parquet"
            )

            # Save
            clear_and_write(
                self.metrics[metric_name],
                output_path,
                {
                    "time_freq": self.time_freq,
                    "dev": self.dev,
                    "use_backup_tweets": self.use_backup_tweets,
                    "use_bookmark_tweets": self.use_bookmark_tweets,
                    "volatile_tweet_filtering": self.volatile_tweet_filtering,
                    "max_date": self.max_date,
                },
            )

        self.current_state = "save"

    def _process_metric(
        self,
        metric_name,
        tweet_df,
        metadata,
        final_data_collection_times,
        conversation_id,
    ):
        # Make sure we have creation times for this tweet
        if conversation_id not in self.creation_times["tweet_id"].values:
            metadata["missing_creation_time"] += 1
            metadata["missing_creation_time_ids"].append(conversation_id)
            return metadata, None

        # Count the number of replies or retweets we've found
        metadata["original_count"] += len(tweet_df)

        # Convert creation time to datetime
        tweet_df["timestamp"] = pd.to_datetime(
            tweet_df["timestamp"],
            utc=True,
            format="%Y-%m-%dT%H:%M:%S.%fZ",
        )

        # Convert file creation time to datetime
        tweet_df["file_created_at"] = pd.to_datetime(
            tweet_df["file_created_at"],
            utc=True,
            format="%Y-%m-%dT%H:%M:%S.%fZ",
        )

        # Find final data collection time for this conversation
        try:
            final_collection_time = max(
                final_data_collection_times.set_index("root_id").loc[
                    conversation_id, "final_data_collection"
                ],
                tweet_df["file_created_at"].max(),
            )
        except KeyError:
            logger.error(
                f"Tweet {conversation_id} not found "
                f"in {metric_name}_file_creations_df.json.gz. "
                f"Maximum file creation time time from {metric_name}_df.json.gz data will be used."
            )
            final_collection_time = tweet_df["file_created_at"].max()

        # Make sure the final collection time occured after all the replies/retweets
        if (final_collection_time < tweet_df["timestamp"]).any():
            logger.error(
                f"Conversation {conversation_id} has {metric_name} that were collected after the final"
                f" data collection time. Maximum file creation time time from {metric_name}_df.json.gz "
                f"data will be used."
            )

        # Drop any duplicated replies/retweets
        tweet_df = tweet_df.drop_duplicates(subset=["response_id", "tweet_id"])

        metadata["post_deduplication"] += len(tweet_df)

        # Drop reply/retweet ID column, now that we've dropped dupes
        tweet_df = tweet_df.drop(columns=["response_id"])

        # Sort by reply/retweet time
        tweet_df = tweet_df.sort_values(["timestamp"], ascending=True)

        # Calculate the number of replies/retweets at each time point
        tweet_df[f"calculated_{metric_name}"] = 1
        tweet_df[f"calculated_{metric_name}"] = tweet_df[
            f"calculated_{metric_name}"
        ].cumsum()

        # Merge to creation times
        tweet_df = tweet_df.merge(
            self.creation_times,
            how="left",
        )

        # Log how many observations we have after merging to creation times and before interpolation
        metadata["post_creation_time_observations"] += len(tweet_df)

        # Find when tweet was created
        tweet_creation_time = tweet_df["created_at"].iloc[0]

        # Add a first observation, where there are 0 replies/retweets,
        # and a final observation (for the last time data was collected)
        # where the replies/retweets will not have changed since the last observation
        tweet_df = pd.concat(
            [
                # Observation at creation time
                pd.DataFrame(
                    {
                        "tweet_id": [conversation_id],
                        "timestamp": [tweet_creation_time],
                        f"calculated_{metric_name}": [0],
                        "created_at": [tweet_creation_time],
                    }
                ),
                tweet_df,
                # Observation at final collection time
                pd.DataFrame(
                    {
                        "tweet_id": [conversation_id],
                        "timestamp": [final_collection_time],
                        "created_at": [tweet_creation_time],
                    }
                ),
            ]
        )

        # Forward fill the number of replies/retweets until the final collection time
        tweet_df = tweet_df.sort_values(["timestamp"], ascending=True)
        tweet_df[f"calculated_{metric_name}"] = (
            tweet_df[f"calculated_{metric_name}"].ffill().astype(int)
        )

        # Perform interpolation
        tweet_df = _interpolate(
            tweet_df,
            freq=self.time_freq,
            time_col="timestamp",
            interpolation={"ffill": [f"calculated_{metric_name}"]},
            stable_cols=[
                "created_at",
                "tweet_id",
            ],
        )

        # Convert to int
        tweet_df[f"calculated_{metric_name}"] = tweet_df[
            f"calculated_{metric_name}"
        ].astype(int)

        # Log how many observations we have after interpolation
        metadata["post_interpolation_observations"] += len(tweet_df)

        if self.max_date is not None:
            # Filter to replies/retweets that happened on or before our max date
            tweet_df = tweet_df[tweet_df["timestamp"] <= self.max_date].reset_index(
                drop=True
            )

        # Log how many observations we have after filtering to only include
        # observations that occurred on or before the max date
        metadata["post_date_filter_observations"] += len(tweet_df)

        return metadata, tweet_df


########################################################################################################################
def _interpolate(
    df,
    freq,
    time_col="pulled_at",
    interpolation={"time": ["impressions"]},
    stable_cols=("publication", "publication_ts"),
    time_col_unit=None,
):
    # Convert to datetime
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], unit=time_col_unit)

    # Find creation_time/max time of data collection, and ts for each entry between min/max, at the desired freq
    creation_time = df["created_at"].iloc[0]
    min_ts = df[time_col].min()
    max_ts = df[time_col].max()

    if pd.isna([creation_time, min_ts, max_ts]).any():
        raise ValueError(
            f"Missing creation time ({creation_time}), min time ({min_ts}), or max time ({max_ts}) in data. "
            f"Data has shape {df.shape}."
        )

    seconds_in_period = pd.to_timedelta(freq).total_seconds()
    periods_between = (max_ts - creation_time).total_seconds() / seconds_in_period + 1
    new_ts = pd.timedelta_range(
        0,
        periods=int(periods_between),
        freq=freq,
    )
    new_ts = creation_time + new_ts

    # Get empty df at the frequency we want
    interpolated_df = pd.DataFrame({time_col: new_ts, "NEW": True})

    # Filter to only when we have data available
    interpolated_df = interpolated_df[interpolated_df[time_col] >= min_ts]

    # Get stable cols that should be the same for all entries
    for col in stable_cols:
        if df[col].nunique() > 1:
            raise ValueError(f"Multiple values for {col}")
        interpolated_df[col] = df[col].unique()[0]

    df["NEW"] = False

    # Merge to actual df
    df_int = pd.concat([df, interpolated_df], axis=0)

    # Interpolate
    for interpolation_method, cols_to_interpolate in interpolation.items():
        for col in cols_to_interpolate:
            if interpolation_method == "ffill":
                df_int = df_int.sort_values([time_col, "NEW"]).reset_index(drop=True)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    df_int[col] = df_int.set_index(time_col)[col].ffill().values
            elif interpolation_method == "bfill":
                df_int = df_int.sort_values(
                    [time_col, "NEW"], ascending=[True, False]
                ).reset_index(drop=True)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    df_int[col] = df_int.set_index(time_col)[col].bfill().values
            elif interpolation_method == "max_ffill":
                df_int = df_int.sort_values([time_col, "NEW"]).reset_index(drop=True)
                df_int[col] = max_ffill(df_int[col])
            else:
                df_int = df_int.sort_values([time_col, "NEW"]).reset_index(drop=True)
                df_int[col] = (
                    df_int.set_index(time_col)[col]
                    .interpolate(method=interpolation_method)
                    .values
                )

    # Filter out original entries
    df_int = df_int[df_int["NEW"] == True].drop(columns=["NEW"]).reset_index(drop=True)

    return df_int


########################################################################################################################

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
        "time_freq", "dev", "use_backup_tweets", "use_bookmark_tweets", "volatile_tweet_filtering", "max_date"
    ]
    for c in necessary_config:
        if c not in config.keys():
            raise ConfigError(
                f"Please specify config value '{c}' in config file {config_path}."
            )

    # Drop unneeded config values
    config = {c: config[c] for c in necessary_config}

    # Fill in the dev value based on whether this is a local run
    if config["dev"] == "DEVICE_DEFAULT":
        config["dev"] = socket.gethostname() == "is-is28m16x"

    # Convert max date to right type
    config["max_date"] = pd.to_datetime(config["max_date"], utc=True)

    tweet_ids = pd.read_csv(
        f"{input_data_dir}/{'dev' if config['dev'] else 'prod'}_tweet_ids.csv",
        header=None,
    )[0].astype(str)

    intermediate_dir = "cn_effect_intermediate" + ("_dev" if config["dev"] else "_prod")

    # Get a logger
    logger, logger_name = get_logger(
        local_data_root / intermediate_dir, return_logger_name=True
    )

    # Run artifact dir
    artifact_dir = Path(logger_name.rstrip(".log"))
    (local_data_root / intermediate_dir / artifact_dir).mkdir(
        exist_ok=True, parents=True
    )

    # Give a warning if we are running a dev run on the cluster, or a prod run locally
    check_run_type(config["dev"], logger)

    # Log config path and config
    logger.info(f"Config path: {config_path}")
    logger.info(f"Config: {config}")

    # Save environment
    save_environment("src/pipeline/a_preprocess.yml", logger)

    # Load note history
    history_processor = NoteHistoryPreprocessor(
        config["time_freq"],
        config["dev"],
        config["use_backup_tweets"],
        config["use_bookmark_tweets"],
        config["volatile_tweet_filtering"],
        config["max_date"],
    )
    history_processor.load()
    history_processor.log_slap_times()

    # Load metrics and save
    metrics_processor = TweetSnapshotProcessor(
        config["time_freq"],
        config["dev"],
        config["use_backup_tweets"],
        config["use_bookmark_tweets"],
        config["volatile_tweet_filtering"],
        config["max_date"],
    )
    metrics_processor.load()
    metrics_processor.filter_and_log_stats(history_processor.slap_times)

    # Load and write structural metrics
    structural_metrics_processor = StructuralMetricsProcessor(
        config["dev"],
        config["time_freq"],
        config["use_backup_tweets"],
        config["use_bookmark_tweets"],
        config["volatile_tweet_filtering"],
        config["max_date"]
    )
    structural_metrics_processor.load(metrics_processor.creation_times)

    # Instantiate calculated metrics
    calculated_metrics_processor = CalculatedMetricsProcessor(
        config["time_freq"],
        config["dev"],
        config["use_backup_tweets"],
        config["use_bookmark_tweets"],
        config["volatile_tweet_filtering"],
        config["max_date"],
    )

    # Find the final data collection times for the calculated metrics
    calculated_metrics_processor.load_data_collection_times(
        metrics_processor.creation_times,
        history_processor.slap_times,
    )

    # Load the calculated metrics values
    calculated_metrics_processor.load_metrics()

    # Save the calculated metrics to disk
    calculated_metrics_processor.save()
    logger.info("Saved calculated metrics dataset")

    # Now return to metrics: interpolate and save
    metrics_processor.interpolate()
    metrics_processor.save()
    logger.info("Saved metrics dataset")

    # Now that we have tweet creation times from metrics_processor, use this to filter, interpolate,
    # then save dataset
    history_processor.filter(metrics_processor.creation_times)
    history_processor.interpolate()
    history_processor.save()
    logger.info("Saved note history dataset")
