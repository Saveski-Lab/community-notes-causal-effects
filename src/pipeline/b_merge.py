from pathlib import Path
import sys
import json
import argparse
import socket
from copy import deepcopy

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add current directory to path for imports
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.utils import (
    get_logger,
    save_environment,
    check_run_type,
    clear_and_write,
    read_from_table,
    ConfigError, local_data_root,
)
from src.pipeline.a_preprocess import (
    structural_metrics_keys,
)

########################################################################################################################
class NoteTweetMerger:
    def __init__(
        self,
        metrics,
        history,
        calculated_retweets,
        calculated_replies,
        dev,
        time_freq,
        use_backup_tweets,
        use_bookmark_tweets,
        volatile_tweet_filtering,
        max_date,
    ):
        self.metrics = metrics
        logger.info(
            f"After loading, there are {len(self.metrics):,} observations in the metrics dataset,"
            f" with {len(self.metrics['tweet_id'].unique()):,} unique tweets."
            f" The dataset spans from {self.metrics['pulled_at'].min()} to {self.metrics['pulled_at'].max()}."
        )

        self.history = history
        logger.info(
            f"After loading, there are {len(self.history):,} observations in the history dataset"
            f" with {len(self.history['tweet_id'].unique()):,} unique tweets, and "
            f"{len(self.history['note_id'].unique()):,} unique notes. The dataset "
            f"spans from {self.history['timestamp'].min()} to {self.history['timestamp'].max()}."
        )

        self.calculated_retweets = calculated_retweets
        logger.info(
            f"After loading, there are {len(self.calculated_retweets):,} observations in the calculated retweets"
            f" dataset, with {len(self.calculated_retweets['tweet_id'].unique()):,} unique tweets."
            f" The dataset spans from {self.calculated_retweets['timestamp'].min()} to "
            f"{self.calculated_retweets['timestamp'].max()}."
        )

        self.calculated_replies = calculated_replies
        logger.info(
            f"After loading, there are {len(self.calculated_replies):,} observations in the calculated replies"
            f" dataset, with {len(self.calculated_replies['tweet_id'].unique()):,} unique tweets."
            f" The dataset spans from {self.calculated_replies['timestamp'].min()} to "
            f"{self.calculated_replies['timestamp'].max()}."
        )

        self.note_info = None
        self.dev = dev
        self.time_freq = time_freq
        self.use_backup_tweets = use_backup_tweets
        self.use_bookmark_tweets = use_bookmark_tweets
        self.volatile_tweet_filtering = volatile_tweet_filtering
        self.max_date = max_date

    def _calculate_creation_times(self):
        self.history = self.history.rename(
            columns={
                "status": "twitter_status",
                "status_change": "twitter_status_change",
            }
        )

        creation_times = {
            "metrics": self.metrics[["tweet_id", "created_at"]].drop_duplicates(),
            "history": self.history[["tweet_id", "created_at"]].drop_duplicates(),
            "calculated_retweets": self.calculated_retweets[
                ["tweet_id", "created_at"]
            ].drop_duplicates(),
            "calculated_replies": self.calculated_replies[
                ["tweet_id", "created_at"]
            ].drop_duplicates(),
        }

        # Check that all creation times are the same
        creation_times_df = pd.concat(creation_times.values(), axis=0).drop_duplicates()
        creation_times_df = creation_times_df.groupby("tweet_id").agg(
            {"created_at": "nunique"}
        )
        creation_times_df = creation_times_df[creation_times_df["created_at"] > 1]
        if len(creation_times_df) > 0:
            logger.error(
                f"Found {len(creation_times_df)} tweets with multiple creation times: "
                f"{creation_times_df.index.to_list()}"
            )

        # Drop creation times, as we can merge them in later
        self.metrics = self.metrics.drop(columns="created_at")
        self.history = self.history.drop(columns="created_at")
        self.calculated_retweets = self.calculated_retweets.drop(columns="created_at")
        self.calculated_replies = self.calculated_replies.drop(columns="created_at")

        self.creation_times = pd.concat(
            creation_times.values(), axis=0
        ).drop_duplicates()
        self.creation_times = (
            self.creation_times.groupby("tweet_id")["created_at"].min().reset_index()
        )

    def _merge_metrics_and_retweets(self):

        # Merge metrics and calculated retweets
        self.merged_metrics = self.metrics.merge(
            self.calculated_retweets,
            left_on=["tweet_id", "pulled_at"],
            right_on=["tweet_id", "timestamp"],
            how="outer",
            indicator=True,
        )

        # Condense two time columns into one
        self.merged_metrics["timestamp"] = self.merged_metrics["timestamp"].fillna(
            self.merged_metrics["pulled_at"]
        )

        # Drop the extra time column
        self.merged_metrics = self.merged_metrics.drop(columns="pulled_at")

        # Count how many tweets make the merge
        merge_counts_by_tweet = (
            (
                self.merged_metrics[["tweet_id", "_merge"]]
                .value_counts()
                .to_frame(name="observation_count")
                .reset_index()
                .pivot(index="tweet_id", columns="_merge", values="observation_count")
            )
            .fillna(0)
            .astype(int)
            .reset_index()
        )

        logger.warning(
            f"Of the {len(self.merged_metrics):,} tweet snapshots in the merged metrics "
            f"and calculated retweets data, "
            f"{merge_counts_by_tweet['left_only'].sum():,} were only in the metrics data, "
            f"{merge_counts_by_tweet['right_only'].sum():,} were only in the calculated retweets data, "
            f"and {merge_counts_by_tweet['both'].sum():,} were in both datasets."
        )

        # Add indicators for which data is missing
        self.merged_metrics["present_in_metrics"] = (
            self.merged_metrics["_merge"] == "left_only"
        ) | (self.merged_metrics["_merge"] == "both")
        self.merged_metrics["present_in_calculated_retweets"] = (
            self.merged_metrics["_merge"] == "right_only"
        ) | (self.merged_metrics["_merge"] == "both")

        # Drop the merge indicator
        self.merged_metrics = self.merged_metrics.drop(columns="_merge")

    def _merge_replies(self):

        # Merge metrics/calc rt and calculated replies
        self.merged_metrics = self.merged_metrics.merge(
            self.calculated_replies,
            left_on=["tweet_id", "timestamp"],
            right_on=["tweet_id", "timestamp"],
            how="outer",
            indicator=True,
        )

        # Count how many tweets make the merge
        merge_counts_by_tweet = (
            (
                self.merged_metrics[["tweet_id", "_merge"]]
                .value_counts()
                .to_frame(name="observation_count")
                .reset_index()
                .pivot(index="tweet_id", columns="_merge", values="observation_count")
            )
            .fillna(0)
            .astype(int)
            .reset_index()
        )

        left_count = (
            merge_counts_by_tweet["left_only"].sum()
            if "left_only" in merge_counts_by_tweet.columns
            else 0
        )
        right_count = (
            merge_counts_by_tweet["right_only"].sum()
            if "right_only" in merge_counts_by_tweet.columns
            else 0
        )
        both_count = (
            merge_counts_by_tweet["both"].sum()
            if "both" in merge_counts_by_tweet.columns
            else 0
        )
        logger.warning(
            f"Of the {len(self.merged_metrics):,} tweet snapshots in the merged metrics "
            f", calculated retweets, and calculated replies data, "
            f"{left_count:,} were only in the merged metrics/calculated retweets data, "
            f"{right_count:,} were only in the calculated replies data, "
            f"and {both_count:,} were in both datasets."
        )

        # Add indicators for which data is missing
        self.merged_metrics["present_in_calculated_replies"] = (
            self.merged_metrics["_merge"] == "right_only"
        ) | (self.merged_metrics["_merge"] == "both")

        # Drop the merge indicator
        self.merged_metrics = self.merged_metrics.drop(columns="_merge")

        # Fill in NAs for the missing data indicators
        self.merged_metrics = self._fill_missing_data_indicators(self.merged_metrics)

    @staticmethod
    def _fill_missing_data_indicators(dataset):
        present_cols = [c for c in dataset.columns if c.startswith("present_in_")]
        # Add indicators for which data is missing
        for c in present_cols:
            dataset[c] = np.where(
                dataset[c].isna(),
                False,
                dataset[c],
            )
        return dataset

    @staticmethod
    def _load_structural_metrics(tid, metric_type, config):
        input_config = deepcopy(config)
        structural_metrics = read_from_table(
            local_data_root / intermediate_dir / f"a_{metric_type}" / f"{tid}.parquet",
            config=input_config,
        )
        return structural_metrics

    def _merge_structural_metrics(self, tid, merged_tweet, config):
        for key in structural_metrics_keys.values():
            fp = local_data_root / intermediate_dir / f"a_{key}" / f"{tid}.parquet"
            if not fp.exists():
                merged_tweet[f"present_in_{key}"] = False
                continue
            merged_tweet = merged_tweet.merge(
                self._load_structural_metrics(tid, key, config).drop(
                    columns=["created_at"]
                ),
                left_on=["tweet_id", "timestamp"],
                right_on=["tweet_id", "time"],
                how="outer",
                indicator=True,
            ).drop(columns="time")
            merged_tweet[f"present_in_{key}"] = (merged_tweet["_merge"] == "both") | (
                merged_tweet["_merge"] == "right_only"
            )
            merged_tweet = merged_tweet.drop(columns="_merge")

            merged_tweet = self._fill_missing_data_indicators(merged_tweet)

        post_num_rows = len(merged_tweet)
        return merged_tweet

    def merge(self):
        if self.note_info is None:
            self._calculate_creation_times()
            self._merge_metrics_and_retweets()
            self._merge_replies()

            # Get tweet IDs that still need to be merged
            tids = [
                tid
                for tid in self.history["tweet_id"].unique()
                if (not (
                        local_data_root / intermediate_dir / f"b_merged" / f"{tid}.parquet"
                ).exists())
                   or (len(
                    read_from_table(
                        local_data_root / intermediate_dir / f"b_merged" / f"{tid}.parquet",
                        config=deepcopy(
                            {
                                "time_freq": self.time_freq,
                                "dev": self.dev,
                                "use_backup_tweets": self.use_backup_tweets,
                                "use_bookmark_tweets": self.use_bookmark_tweets,
                                "volatile_tweet_filtering": self.volatile_tweet_filtering,
                                "max_date": self.max_date,
                            }
                        ))) == 0)
            ]

            logger.info(
                f"Found {len(tids):,} tweets that still need to be merged, out of "
                f"{len(self.history['tweet_id'].unique()):,} total."
            )

            for tid in tqdm(tids, desc="Merging tweets", smoothing=0):
                tweet_info = self.history[self.history["tweet_id"] == tid]

                merged_tweet = self._pivot_and_merge_note_history(
                    tid, tweet_info.copy(), self.time_freq
                )

                merged_tweet = self._merge_structural_metrics(
                    tid, merged_tweet.copy(), config
                )

                # Merge back to creation times
                merged_tweet = merged_tweet.merge(
                    self.creation_times,
                    on="tweet_id",
                    how="left",
                )

                # Calculate like_through_rate
                merged_tweet["like_through_rate"] = (
                    merged_tweet["likes"] / merged_tweet["impressions"]
                )

                # Calculate time_since_publication
                if merged_tweet["created_at"].isna().any():
                    logger.error(
                        f"Tweet {tid} has NA in created_at field. Removing tweet"
                    )
                    continue
                else:
                    # Add how long it has been since the tweet was published
                    merged_tweet["time_since_publication"] = (
                        merged_tweet["timestamp"] - merged_tweet["created_at"]
                    )

                # Make sure note IDs and note history times are always present

                # Save
                merged_tweet["dev"] = self.dev
                merged_tweet["time_freq"] = self.time_freq
                merged_tweet["use_backup_tweets"] = self.use_backup_tweets
                merged_tweet["use_bookmark_tweets"] = self.use_bookmark_tweets
                merged_tweet["volatile_tweet_filtering"] = self.volatile_tweet_filtering
                merged_tweet["max_date"] = self.max_date

                # Define output path and make sure parent exists
                output_path = (
                    local_data_root / intermediate_dir / f"b_merged" / f"{tid}.parquet"
                )
                output_path.parent.mkdir(parents=True, exist_ok=True)
                clear_and_write(
                    merged_tweet,
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

        return self.note_info

    def _pivot_and_merge_note_history(self, tid, note_history, time_freq):
        # We're now going to pivot the note info so that the unit of observation is a tweet at a
        # given time, and each note is a column.

        # To do so, find out when each note was first rated helpful (as this will be something
        # we want to sort on)
        crhs = note_history[note_history["twitter_status"] == "CURRENTLY_RATED_HELPFUL"]
        first_crh_time = (
            crhs.groupby(["tweet_id", "note_id"])
            .agg({"timestamp": "min"})
            .reset_index()
        )
        first_crh_time = first_crh_time.rename(columns={"timestamp": "first_crh_time"})

        # Next, get the list of all notes, and label each with the first crh time, if there was one
        all_notes = note_history[["tweet_id", "note_id"]].drop_duplicates()
        all_notes = all_notes.merge(
            first_crh_time, on=["tweet_id", "note_id"], how="left"
        )

        # Now sort based on first CRH time, and create an index so that for each tweet,
        # the note with the earliest CRH time is first, and so on
        all_notes = all_notes.sort_values(["tweet_id", "first_crh_time", "note_id"])
        all_notes["note_index"] = all_notes.groupby("tweet_id", dropna=False).cumcount()
        total_num_notes = all_notes["note_index"].max() + 1

        # Now, join back to main data
        note_history = note_history.merge(
            all_notes, on=["tweet_id", "note_id"], how="left"
        )

        note_history["time_since_first_crh"] = (
            note_history["timestamp"] - note_history["first_crh_time"]
        )

        # Use the newly found index to pivot
        note_history = note_history.pivot(
            index=["tweet_id", "timestamp"],
            columns="note_index",
            values=[
                c
                for c in note_history.columns
                if c not in ["tweet_id", "timestamp", "note_index"]
            ],
        )

        # Rename columns
        note_history.columns = [
            f"note_{c[1]}_{c[0]}" for c in note_history.columns.to_flat_index()
        ]

        # Reset tweet id to column
        note_history = note_history.reset_index()
        # Note history and metrics
        note_history = note_history.merge(
            self.merged_metrics[self.merged_metrics["tweet_id"] == tid],
            left_on=["tweet_id", "timestamp"],
            right_on=["tweet_id", "timestamp"],
            indicator=True,
            how="outer",
        )

        # Make sure we have timestamps for all relevant times
        note_history = note_history.merge(
            pd.date_range(
                note_history["timestamp"].min(),
                note_history["timestamp"].max(),
                freq=pd.to_timedelta(time_freq),
            ).to_series(name="timestamp"),
            how="outer",
        )

        # Create indicators for what data is missing
        note_history["present_in_metrics"] = np.where(
            note_history["present_in_metrics"].isna(),
            False,
            note_history["present_in_metrics"],
        )
        note_history["present_in_calculated_retweets"] = np.where(
            note_history["present_in_calculated_retweets"].isna(),
            False,
            note_history["present_in_calculated_retweets"],
        )
        note_history["present_in_calculated_replies"] = np.where(
            note_history["present_in_calculated_replies"].isna(),
            False,
            note_history["present_in_calculated_replies"],
        )
        note_history["present_in_note_history"] = (
            (note_history["_merge"] == "left_only") | (note_history["_merge"] == "both")
        ).fillna(False)

        # Drop the merge indicator
        note_history = note_history.drop(columns=["_merge"])

        # Fill in tweet_id column
        note_history["tweet_id"] = tid

        # Fill in columns that should always be present
        to_fill = [
            "note_{note_idx}_note_created_at",
            "note_{note_idx}_note_id",
            "note_{note_idx}_first_crh_time",
        ]

        for note in range(total_num_notes):
            for col in to_fill:
                col_value = note_history[col.format(note_idx=note)].unique()
                # Filter out NAs
                col_value = col_value[~pd.isna(col_value)]
                if len(col_value) == 0:
                    continue
                note_history[col.format(note_idx=note)] = col_value[0]

            # Now calculate time since CRH
            if note_history[f"note_{note}_first_crh_time"].notna().all():
                note_history[f"note_{note}_time_since_first_crh"] = (
                    note_history["timestamp"]
                    - note_history[f"note_{note}_first_crh_time"]
                )

        return note_history


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
    ]
    for c in necessary_config:
        if c not in config.keys():
            raise ConfigError(
                f"Please specify config value '{c}' in config file '{config_path}.'"
            )

    # Drop unneeded config values
    config = {c: config[c] for c in necessary_config}

    if config["dev"] == "DEVICE_DEFAULT":
        config["dev"] = socket.gethostname() == "is-is28m16x"

    config["max_date"] = pd.to_datetime(config["max_date"], utc=True)

    intermediate_dir = "cn_effect_intermediate" + ("_dev" if config["dev"] else "_prod")

    # Get a logger
    logger = get_logger(local_data_root / intermediate_dir)

    # Check that we aren't running a dev run on the cluster, or vice versa
    check_run_type(config["dev"], logger)

    # Save environment
    save_environment("src/pipeline/b_merge.yml", logger)

    # Load datasets
    metrics = read_from_table(
        local_data_root / intermediate_dir / "a_metrics.parquet", config=config
    )
    note_history = read_from_table(
        local_data_root / intermediate_dir / "a_note_history.parquet",
        config=config,
    )
    calculated_retweets = read_from_table(
        local_data_root / intermediate_dir / "a_calculated_retweets.parquet",
        config=config,
    )
    calculated_replies = read_from_table(
        local_data_root / intermediate_dir / "a_calculated_replies.parquet",
        config=config,
    )

    merger = NoteTweetMerger(
        metrics,
        note_history,
        calculated_retweets,
        calculated_replies,
        config["dev"],
        config["time_freq"],
        config["use_backup_tweets"],
        config["use_bookmark_tweets"],
        config["volatile_tweet_filtering"],
        config["max_date"],
    )
    merger.merge()
