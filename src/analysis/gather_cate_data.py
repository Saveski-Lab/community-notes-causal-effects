import argparse
import json
import socket
import sys
from pathlib import Path
import pandas as pd
from src.utils import (
    save_environment,
    get_logger,
    ConfigError,
    check_run_type,
    informative_merge, read_weights, read_control_configs, local_data_root,
)
from src.pipeline.c_find_controls import (
    metric_parents,
    read_trt_and_ctrl,
    tweet_level_datasets,
)
from src.analysis.plot_treatment_effects import (
    load_tes_for_metrics,
    get_trt_and_control_ids,
    get_metadata,
    get_artifact_dir,
)

def main(config_path: str) -> None:
    """Main function to gather and process treatment effect data.

    Args:
        config_path (str): Path to the configuration file.
    """
    with open(config_path, "r") as fin:
        config = json.load(fin)

    necessary_config = [
        "time_freq", "dev", "use_backup_tweets", "use_bookmark_tweets",
        "volatile_tweet_filtering", "max_date", "train_backdate",
        "pre_break_min_time", "pre_break_max_time", "post_break_min_time",
        "matching_metrics", "replace_calculated_when_missing",
        "missing_metric_action", "include_bias_correction",
        "pre_break_bias_correction_time", "bias_correction_model",
        "bias_correction_missing_actions", "target_metrics",
        "restrict_donor_pool", "restricted_pool_size",
        "sample_bias_correction_controls", "lambda",
    ]
    for c in necessary_config:
        if c not in config:
            raise ConfigError(f"Missing config value '{c}' in '{config_path}'.")

    config = {c: config[c] for c in necessary_config}

    if len(config["lambda"]) == 1:
        config["lambda"] = config["lambda"][0]
    else:
        raise ConfigError("Multiple lambda values in config.")

    config["dev"] = socket.gethostname() == "is-is28m16x" if config["dev"] == "DEVICE_DEFAULT" else config["dev"]
    config["max_date"] = pd.to_datetime(config["max_date"], utc=True)

    config["tweet_metrics"] = [m for m in config["matching_metrics"] if metric_parents[m] in tweet_level_datasets]
    config["backup_tweet_metrics"] = [m.replace("calculated_", "") for m in config["matching_metrics"] if "calculated_" in m] if config["replace_calculated_when_missing"] else []
    config["author_metrics"] = [m for m in config["matching_metrics"] if metric_parents[m] == "author"]

    if len(config["tweet_metrics"]) + len(config["author_metrics"]) != len(config["matching_metrics"]):
        unknown_metrics = [m for m in config["metrics"] if m not in metric_parents]
        raise ConfigError(f"Unknown metrics: {unknown_metrics}.")

    intermediate_dir = f"cn_effect_intermediate_{'dev' if config['dev'] else 'prod'}"
    output_dir = local_data_root / "cn_effect_output" / "treatment_effects" / Path(config_path).stem
    output_dir.mkdir(parents=True, exist_ok=True)

    logger, log_name = get_logger(local_data_root / intermediate_dir, enqueue=True, return_logger_name=True)
    check_run_type(config["dev"], logger)
    logger.info(f"Config path: {config_path}")
    logger.info(f"Config: {config}")

    logger.info("Loading treatment effects")
    metrics = sorted(config["target_metrics"] + [m + "_pct_change" for m in config["target_metrics"]])
    te = load_tes_for_metrics(metrics, config, intermediate_dir)
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

    treatment_tids, control_tids = get_trt_and_control_ids(config["target_metrics"], te, weights)
    logger.info(f"Treatment tweet IDs identified for {len(control_tids):,} control tweets and {len(treatment_tids):,} treatment tweets.")

    control_tweets, treatment_tweets = read_trt_and_ctrl(intermediate_dir, config)
    logger.info(f"Metrics read for {len(control_tweets):,} control tweets and {len(treatment_tweets):,} treatment tweets.")

    control_configs = read_control_configs(intermediate_dir, config)
    logger.info(f"Control configs read for {len(control_configs):,} tweets, {len([v for v in control_configs.values() if v[0]['use_tweet']]):,} used for matching.")

    tweet_metadata = get_metadata(treatment_tids, treatment_tweets, control_configs, intermediate_dir, get_artifact_dir(config), config, config["target_metrics"], logger=logger)
    final_te_with_metadata = informative_merge(final_te, tweet_metadata, "final_treatment_effects", "tweet_metadata", on="tweet_id", how="left", logger=logger)

    try:
        final_te_with_metadata.to_csv(output_dir / "final_treatment_effects.csv", index=False)
    except Exception as e:
        print(f"Error saving to csv: {e}")

    try:
        final_te_with_metadata.to_pickle(output_dir / "final_treatment_effects.pkl.gz")
    except Exception as e:
        print(f"Error saving to pickle: {e}")

    try:
        for c in final_te_with_metadata.columns:
            if final_te_with_metadata[c].dtype.name == "Interval" or 'timedelta' in str(final_te_with_metadata[c].dtype):
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