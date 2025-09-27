import argparse
import json
import sys
import socket

from itertools import chain
from pathlib import Path
from copy import deepcopy

# Add current directory to path for imports
sys.path.append(str(Path(__file__).resolve().parents[2]))

import pandas as pd
from src.utils import (
    get_logger,
    ConfigError,
    check_run_type,
    informative_merge, read_weights, read_control_configs,
    data_root,
)
from src.pipeline.c_find_controls import (
    metric_parents,
    read_trt_and_ctrl,
    tweet_level_datasets,
)
from src.analysis.plotting_utils import load_tes_for_metrics, get_trt_and_control_ids, get_metadata


def main(config_path: str) -> None:
    """Main function to gather and process treatment effect data.

    Args:
        config_path (str): Path to the configuration file.
    """
    with open(config_path, "r") as fin:
        config = json.load(fin)

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
        "include_bias_correction",
        "pre_break_bias_correction_time",
        "bias_correction_model",
        "bias_correction_missing_actions",
        "target_metrics",
        "restrict_donor_pool",
        "restricted_pool_size",
        "sample_bias_correction_controls",
        "lambda",
        "shuffle",
    ]
    optional_config = [
        "control_list",
    ]
    for c in necessary_config:
        if c not in config.keys():
            raise ConfigError(
                f"Please specify config value '{c}' in config file '{config_path}.'"
            )
    for c in optional_config:
        if c not in config.keys():
            config[c] = None

    config = {c: config[c] for c in chain(necessary_config, optional_config) if c in config.keys()}

    if len(config["lambda"]) == 1:
        config["lambda"] = config["lambda"][0]
    else:
        raise ConfigError("Multiple lambda values in config.")

    config["dev"] = socket.gethostname() == "is-is28m16x" if config["dev"] == "DEVICE_DEFAULT" else config["dev"]

    config["tweet_metrics"] = [m for m in config["matching_metrics"] if metric_parents[m] in tweet_level_datasets]
    config["backup_tweet_metrics"] = [m.replace("calculated_", "") for m in config["matching_metrics"] if "calculated_" in m] if config["replace_calculated_when_missing"] else []
    config["author_metrics"] = [m for m in config["matching_metrics"] if metric_parents[m] == "author"]

    if len(config["tweet_metrics"]) + len(config["author_metrics"]) != len(config["matching_metrics"]):
        unknown_metrics = [m for m in config["metrics"] if m not in metric_parents]
        raise ConfigError(f"Unknown metrics: {unknown_metrics}.")

    intermediate_dir = f"cn_effect_intermediate_{'dev' if config['dev'] else 'prod'}"
    output_dir = data_root / "cn_effect_output" / "treatment_effects" / Path(config_path).stem
    output_dir.mkdir(parents=True, exist_ok=True)

    logger, log_name = get_logger(data_root / intermediate_dir, enqueue=True, return_logger_name=True)
    check_run_type(config["dev"], logger)
    logger.info(f"Config path: {config_path}")
    logger.info(f"Config: {config}")

    logger.info("Loading treatment effects")
    metrics = sorted(config["target_metrics"])
    te_config = deepcopy(config)
    te = load_tes_for_metrics(metrics, te_config, intermediate_dir)
    num_tes = {metric: df["tweet_id"].nunique() for metric, df in te.items()}
    logger.info(f"Treatment effects loaded. Number of TEs per metric: {num_tes}")

    final_te = pd.concat(
        [
            te_df[["tweet_id", "note_0_hours_since_first_crh", "treatment", "control", "bias_adjusted_control", "bias_adjusted_treatment_effect", "unadjusted_treatment_effect"]]
            .rename(columns={
                "bias_adjusted_control": f"bcc_{metric}",
                "treatment": f"t_{metric}",
                "control": f"c_{metric}",
                "bias_adjusted_treatment_effect": f"te_{metric}",
                "unadjusted_treatment_effect": f"te_unadjusted_{metric}",
            })
            .set_index(["tweet_id", "note_0_hours_since_first_crh"])
            for metric, te_df in te.items()
        ],
        axis=1,
    ).reset_index()

    logger.info(f"Final treatment effects merged. Number of tweets with a TE: {final_te['tweet_id'].nunique():,}")

    weights = read_weights(intermediate_dir, config)
    logger.info(f"Weights loaded for {len(weights):,} metrics")

    treatment_tids, control_tids = get_trt_and_control_ids(metrics, te, weights)
    logger.info(f"Treatment tweet IDs identified for {len(control_tids):,} control tweets and {len(treatment_tids):,} treatment tweets.")

    # Get trt and control tweets
    trt_and_ctrl_config = deepcopy(config)
    trt_and_ctr_config_keys = [
        "time_freq",
        "dev",
        "volatile_tweet_filtering",
        "train_backdate",
        "pre_break_min_time",
        "backup_tweet_metrics",
        "tweet_metrics",
        "author_metrics",
        "pre_break_max_time",
        "post_break_min_time",
        "matching_metrics",
        "replace_calculated_when_missing",
        "missing_metric_action",
        "shuffle"
    ]
    for c in list(trt_and_ctrl_config.keys()):
        if c not in trt_and_ctr_config_keys:
            del trt_and_ctrl_config[c]

    control_tweets, treatment_tweets = read_trt_and_ctrl(intermediate_dir, trt_and_ctrl_config)
    logger.info(f"Metrics read for {len(control_tweets):,} control tweets and {len(treatment_tweets):,} treatment tweets.")

    control_configs = read_control_configs(intermediate_dir, config)
    logger.info(f"Control configs read for {len(control_configs):,} tweets, {len([v for v in control_configs.values() if v[0]['use_tweet']]):,} used for matching.")

    tweet_metadata = get_metadata()
    final_te_with_metadata = informative_merge(final_te, tweet_metadata, "final_treatment_effects", "tweet_metadata", on="tweet_id", how="left", logger=logger)

    try:
        for c in final_te_with_metadata.columns:
            if final_te_with_metadata[c].dtype.name in ["Interval", "category"] or 'timedelta' in str(final_te_with_metadata[c].dtype):
                try:
                    final_te_with_metadata[c] = final_te_with_metadata[c].astype(str)
                    print(f"Converted column {c} to string")
                except Exception as e:
                    print(f"Error converting column {c}: {e}")

        final_te_with_metadata.to_parquet(output_dir / "final_treatment_effects.parquet")
    except Exception as e:
        print(f"Error saving to parquet: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to config file")
    args = parser.parse_args()
    main(args.config)