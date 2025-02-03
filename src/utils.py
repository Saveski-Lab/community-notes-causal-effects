import json
import pathlib
import gzip
import re
import os
import sys
import subprocess
from copy import deepcopy

import yaml
import socket
from json.decoder import JSONDecodeError

import pandas as pd

from datetime import datetime
from loguru import logger
from tqdm import tqdm

from src.pipeline.a_preprocess import local_data_root


def anti_join(x, y, on):
    """Return rows in x which are not present in y"""
    ans = pd.merge(left=x, right=y, how="left", indicator=True, on=on)
    ans = ans.loc[ans._merge == "left_only", :].drop(columns="_merge")
    return ans


def get_logger(dir, return_logger_name=False, log_name=None, enqueue=False):
    """Return a logger with a file and stdout handler."""

    # Remove the stderr logger, if it is present
    for hid, handler in logger._core.handlers.items():
        if handler._name == "<stderr>":
            logger.remove(hid)

    # Create location to save log
    save_dir = os.path.join(dir, "logs")
    os.makedirs(save_dir, exist_ok=True)

    # Get save path and format for logger (save name is based on script name)
    if log_name is None:
        log_name = (
            os.path.basename(sys.argv[0]).replace(".py", "")
            + f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')}.log"
        )

    save_path = os.path.join(save_dir, log_name)

    # Set log format, with color for metadata
    format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<green>{level: <8}</green> | "
        "<green>{name}:{function}:{line}</green> | "
        "{message}"
    )

    # Add the log file handler, if it is not already present
    log_file_present = False
    for hid, handler in logger._core.handlers.items():
        log_file_present = (
            log_file_present or os.path.basename(handler._name.strip("'")) == log_name
        )
    if not log_file_present:
        logger.add(
            save_path, level="INFO", serialize=False, format=format, enqueue=enqueue
        )

    # Add the stdout handler, if it is not already present
    stdout_present = False
    for hid, handler in logger._core.handlers.items():
        stdout_present = stdout_present or handler._name == "<stdout>"
    if not stdout_present:
        logger.add(
            sys.stdout, level="INFO", serialize=False, format=format, enqueue=enqueue
        )

    return (logger, log_name) if return_logger_name else logger


def save_environment(fp, logger=None):
    """Save environment to a file by combining outputs from conda env export."""

    if not fp.endswith(".yml"):
        raise ValueError("Filepath must end with .yml")

    # Export the full environment without builds but with pip packages
    full_env = subprocess.run(
        ["conda", "env", "export", "--no-builds"], stdout=subprocess.PIPE, text=True
    ).stdout

    # Export just the explicit installations from history
    explicit_env = subprocess.run(
        ["conda", "env", "export", "--from-history"], stdout=subprocess.PIPE, text=True
    ).stdout

    # Convert exports to dictionaries
    full_env_dict = yaml.safe_load(full_env)
    explicit_env_dict = yaml.safe_load(explicit_env)

    # Get list of explicitly installed packages (names only)
    explicit_names = set(pkg.split("=")[0] for pkg in explicit_env_dict["dependencies"])
    explicit_names.add("pip")

    # Filter full dependencies by those names
    filtered_dependencies = [
        pkg
        for pkg in full_env_dict["dependencies"]
        if isinstance(pkg, str)
        and pkg.split("=")[0] in explicit_names
        or isinstance(pkg, dict)
        and "pip" in pkg
    ]

    # Replace dependencies in full env with filtered list
    full_env_dict["dependencies"] = filtered_dependencies

    # Remove prefix line if present
    if "prefix" in full_env_dict:
        del full_env_dict["prefix"]

    # Save the filtered environment to a file
    with open(fp, "w") as outfile:
        yaml.safe_dump(full_env_dict, outfile)

    if logger:
        logger.info(f"Environment saved to {fp}")


def check_run_type(dev_run, logger):
    """Check if the run type is what we would expect on the current machine"""
    device_name = socket.gethostname()
    LAPTOP_NAME = "is-is28m16x"
    if dev_run and device_name != LAPTOP_NAME:
        logger.warning(
            f"Performing a dev run on device {device_name}. Are you sure this is what you want to do?"
        )
    elif not dev_run and device_name == LAPTOP_NAME:
        logger.warning(
            f"Performing a prod run on device {device_name}. Are you sure this is what you want to do?"
        )
    else:
        logger.info(
            f'Performing a {"dev" if dev_run else "prod"} run on device {device_name}'
        )


def json_gzip_reader(path):
    path = str(path)
    try:
        with gzip.open(
            path,
            "r",
        ) as fin:
            loaded = json.loads(fin.read())

    # Files can be corrupted, so we need to handle this
    except JSONDecodeError:
        loaded = []
    return loaded


def json_gzip_writer(data, path):
    path = str(path)
    with gzip.open(
        path,
        "wt",
    ) as fout:
        fout.write(json.dumps(data))


def clear_and_write(
    df: pd.DataFrame | list, fp: pathlib.Path, config: dict, logger=None
) -> None:
    if fp.suffix == ".csv":
        reader = pd.read_csv
    elif fp.suffix == ".parquet":
        reader = pd.read_parquet
    elif str(fp).endswith(".json.gz"):
        reader = json_gzip_reader
    else:
        raise ValueError("Unsupported file type")

    if fp.exists():
        prev_entries = reader(fp)
        if logger:
            logger.info(
                f"When saving dataset {fp.name}, found {len(prev_entries):,} previous observations."
            )
        # Filter out any entries with the same configuration as the current run
        if fp.suffix in [".csv", ".parquet"]:
            same_config_indexer: None | pd.Series = None
            for config_key, config_value in config.items():
                indexer_for_this_key = (
                    prev_entries[config_key.isna()]
                    if config_value is None
                    else (prev_entries[config_key] == config_value)
                )
                same_config_indexer = (
                    indexer_for_this_key
                    if same_config_indexer is None
                    else (same_config_indexer & indexer_for_this_key)
                )

            prev_entries = prev_entries[~(same_config_indexer)]
            all_entries = pd.concat([prev_entries, df], axis=0)
        elif str(fp).endswith(".json.gz"):
            other_entries = []
            # Loop through previous entries in json
            for e in prev_entries:
                # Assume they have the same config
                same_config = True
                for k in config.keys():
                    # Convert max_date to datetime
                    if k == "max_date":
                        e_value = pd.to_datetime(e[k], utc=True)
                        config_value = pd.to_datetime(config[k], utc=True)
                    else:
                        e_value = e[k]
                        config_value = config[k]
                    # If there are any config values that differ then the config is different
                    if e_value != config_value:
                        if logger:
                            logger.info(
                                "Not filtering out entry when saving because of non-matching ",
                                k,
                            )
                        same_config = False
                        break
                if not same_config:
                    other_entries.append(e)
            prev_entries = other_entries
            all_entries = df + prev_entries

        if logger:
            logger.info(
                f"Filtered out observations with the same configuration as the current run. There are now "
                f"{len(prev_entries):,} observations remaining from prior runs in {fp.name}."
            )

    else:
        if logger:
            logger.info(
                f"When saving dataset {fp.name}, found 0 previous observations."
            )
        all_entries = df

    if fp.suffix == ".csv":
        all_entries.to_csv(
            fp,
            index=False,
        )
    elif fp.suffix == ".parquet":
        all_entries.to_parquet(
            fp,
            index=False,
        )
    elif str(fp).endswith(".json.gz"):
        with gzip.open(fp, "wt") as fout:
            fout.write(json.dumps(all_entries))
    else:
        raise ValueError("Unsupported file type")


def read_from_table(fp, config, keep_metadata=False) -> pd.DataFrame | list:
    if fp.suffix == ".parquet":
        df = pd.read_parquet(
            fp,
            filters=[(k, "==", v) for k, v in config.items()],
        )
    elif fp.suffix == ".csv":
        df = pd.read_csv(fp)
        # Filter out any entries with the same configuration as the current run
        same_config_indexer: None | pd.Series = None
        for config_key, config_value in config.items():
            indexer_for_this_key = df[config_key] == config_value
            same_config_indexer = (
                indexer_for_this_key
                if same_config_indexer is None
                else (same_config_indexer & indexer_for_this_key)
            )
        df = df[same_config_indexer]
    elif str(fp).endswith(".json.gz"):
        prev_entries = json_gzip_reader(fp)
        matching_entries = []
        # Loop through previous entries in json
        for e in prev_entries:
            # Assume they have the same config
            same_config = True
            for k in config.keys():
                if k == "max_date":
                    e_value = pd.to_datetime(e[k], utc=True)
                    config_value = pd.to_datetime(config[k], utc=True)
                else:
                    e_value = e[k]
                    config_value = config[k]
                # If there are any config values that differ, then the config is different
                if e_value != config_value:
                    same_config = False
                    break
            if same_config:
                matching_entries.append(e)
        df = matching_entries
    else:
        raise ValueError("Unsupported file type")

    if not keep_metadata:
        if fp.suffix in [".parquet", ".csv"]:
            df = df.drop(columns=config.keys())
        elif str(fp).endswith(".json.gz"):
            for k in config.keys():
                for i in range(len(df)):
                    if k in df[i].keys():
                        del df[i][k]

    return df


def informative_merge(
    ds1, ds2, ds1_name, ds2_name, on, logger=None, *merge_args, **merge_kwargs
):
    if not isinstance(on, str):
        raise ValueError("'on' must be a string in current implementation")

    merged = ds1.merge(ds2, on=on, *merge_args, **merge_kwargs, indicator=True)

    made_it_from_left = ds1[on].isin(merged[on].values).sum()
    didnt_make_it_from_left = (~ds1[on].isin(merged[on].values)).sum()

    made_it_from_right = ds2[on].isin(merged[on].values).sum()
    didnt_make_it_from_right = (~ds2[on].isin(merged[on].values)).sum()

    left_only = (merged["_merge"] == "left_only").sum()
    right_only = (merged["_merge"] == "right_only").sum()
    both = (merged["_merge"] == "both").sum()

    merged = merged.drop(columns="_merge")

    if logger:
        messenger = logger.info
    else:
        messenger = print

    messenger(
        f"{ds1_name} started with {len(ds1):,} records. "
        f"After merging by {on}, {made_it_from_left:,} records "
        f"made it to the merged data, while {didnt_make_it_from_left:,} did not.\n"
        f"{ds2_name} started with {len(ds2):,} records. "
        f"After merging by {on}, {made_it_from_right:,} records "
        f"made it to the merged data, while {didnt_make_it_from_right:,} did not.\n"
        f"The merged dataset contains {len(merged):,} records. "
        f"Of these records, {left_only:,} were only present in {ds1_name}, "
        f"{right_only:,} were only present in {ds2_name}, while {both:,} were in both.\n\n"
    )
    return merged


class ConfigError(Exception):
    pass


def camel_to_snake(s):
    return re.sub(r"(?<!^)(?=[A-Z])", "_", s).lower()


from urlextract import URLExtract
from typing import List


def remove_urls(text: str) -> str:
    """
    Remove all URLs from a string using urlextract library.

    Args:
        text (str): Input text containing URLs

    Returns:
        str: Text with all URLs removed and cleaned up
    """
    # Initialize the extractor
    extractor = URLExtract()

    # Get all URLs from the text
    urls = extractor.find_urls(text)

    # Replace each URL with empty string
    for url in urls:
        text = text.replace(url, '')

    # Clean up extra whitespace
    return ' '.join(text.split())


def read_weights(
    intermediate_dir: str,
    config: dict,
    subdir: None | str = None,
    treatment_tweet_ids: None | list = None,
):
    """
    Read the various synthetic control weights that have been calculated for each control sequence

    subdir can be provided for permutations

    """
    weights_dir = local_data_root / intermediate_dir / "f_calc_weights"

    if subdir:
        weights_dir = weights_dir / subdir

    weights_fnames = [fn for fn in os.listdir(weights_dir) if fn.endswith(".parquet")]

    if treatment_tweet_ids is not None:
        weights_fnames = [
            fn
            for fn in weights_fnames
            if fn.split(".parquet")[0] in treatment_tweet_ids
        ]

    weights_fpaths = [weights_dir / fn for fn in weights_fnames]

    config = deepcopy(config)
    config["train_backdate"] = -(
        pd.to_timedelta(config["train_backdate"]).total_seconds() / 3600
    )

    config["matching_metrics"] = ",".join(config["matching_metrics"])

    # Create the config file to use for filtering to the right weights
    unneeded_config = [
        "target_metrics",
        "backup_tweet_metrics",
        "tweet_metrics",
        "include_bias_correction",
        "pre_break_bias_correction_time",
        "bias_correction_missing_actions",
        "bias_correction_model",
        "author_metrics",
        "restrict_donor_pool",
        "restricted_pool_size",
        "sample_bias_correction_controls",
        "intermediate_dir",
        "artifact_dir",
    ]

    for k in unneeded_config:
        if k in config.keys():
            del config[k]

    # Read the weights into a dictionary, with the tweet ID as the key
    weights = {
        fn.replace(".parquet", ""): read_from_table(fp, config)
        for fn, fp in tqdm(
            zip(weights_fnames, weights_fpaths),
            total=len(weights_fnames),
            desc="Reading weights",
        )
    }

    # Filter to only solved weights
    for tid, w in weights.items():
        weights[tid] = w[w["solver_status"] == "Solved"]

    # Delete any treatment tweets that don't have weights
    keys_to_check = list(weights.keys())
    for k in keys_to_check:
        if weights[k].shape[0] == 0:
            del weights[k]

    # Make sure there's a max of one set of weights for each tweet
    for tid, w in weights.items():
        if len(w) > 1:
            raise ValueError(f"More than one set of weights found for tweet {tid}")

    return weights


def read_control_configs(
    intermediate_dir: str,
    config: dict,
):
    metadata_for_step_c = {}
    necessary_config = [
        "time_freq",
        "dev",
        "use_bookmark_tweets",
        "max_date",
        "train_backdate",
        "pre_break_min_time",
        "pre_break_max_time",
        "post_break_min_time",
        "matching_metrics",
        "replace_calculated_when_missing",
        "missing_metric_action",
    ]
    for k in necessary_config:
        metadata_for_step_c[k] = deepcopy(config[k])

    return {
        tweet_fname.replace(".json.gz", ""): read_from_table(
            local_data_root / intermediate_dir / "c_find_controls" / tweet_fname,
            metadata_for_step_c,
        )
        for tweet_fname in tqdm(
            os.listdir(local_data_root / intermediate_dir / "c_find_controls"),
            desc="Reading control configurations",
            smoothing=0,
        )
        if tweet_fname.endswith(".json.gz")
    }


def combine_sequence(
    seq, first_weights, second_weights=None, treatment_column_name="treatment"
):
    """Calculate the synthetic controls based on specified weights."""

    cids = [
        col
        for col in first_weights.columns[~first_weights.isna().any()]
        if col in seq.columns
    ]

    # Get the treatment sequence
    treatment = seq[treatment_column_name].copy()

    # Multiply the control sequences by the weights
    first_sums = seq[cids].values * first_weights[cids].values

    # Filter out columns that have NAs
    first_sums = first_sums[:, ~pd.isna(first_sums).any(axis=0)]

    # Sum along the weighted control sequences
    first_sums = pd.DataFrame(
        first_sums.sum(axis=1),
        columns=["first_weighted_control_sequence"],
        index=seq.index,
    )

    if second_weights is None:
        combined = first_sums
    else:
        # Multiply the control sequences by the weights
        second_sums = seq[cids].values * second_weights[cids].values
        second_sums = second_sums[:, ~pd.isna(second_sums).any(axis=0)]
        second_sums = pd.DataFrame(
            second_sums.sum(axis=1),
            columns=["second_weighted_control_sequence"],
            index=seq.index,
        )

        # Merge the treatment and control sequences
        combined = pd.concat([first_sums, second_sums], axis=1)

    # Set treatment
    assert (combined.index == treatment.index).all()
    combined["treatment"] = treatment.values

    return combined
