import os
import sys
import socket
import json
import argparse
import traceback
from pathlib import Path
from copy import deepcopy
from joblib import Parallel, delayed
from collections import defaultdict
from typing import Optional, Any
from itertools import product, chain

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from tqdm import tqdm
import sklearn.linear_model as lm
from sklearn.metrics import mean_absolute_error, r2_score
from loguru._logger import Logger

# Add current directory to path for imports
sys.path.append(str(Path(__file__).resolve().parents[2]))

NUM_PROCESSES = 10

from src.utils import (
    get_logger,
    ConfigError,
    check_run_type,
    clear_and_write, read_weights, read_control_configs, combine_sequence, data_root,
)
from src.pipeline.c_find_controls import (
    metric_parents,
    read_trt_and_ctrl,
    author_path,
    tweet_level_datasets,
)

# Load tweet seeds
tweet_seeds_path = data_root / "__tweet_seeds.json"
TWEET_SEEDS = json.load(open(tweet_seeds_path))


########################################################################################################################


def impute_nas(df: pd.DataFrame, to_impute: None | pd.DataFrame = None) -> tuple[DataFrame, DataFrame | Any]:
    if to_impute is None:
        to_impute = df.mean(axis=1)

    # Make sure all necessary indexes are contained in the imputation
    df = df.reindex(to_impute.index, fill_value=np.NaN, axis=0)

    # Get indicator for missing values
    all_nas = df.isna().astype(int)

    # Fill in missing values
    df = df.T.fillna(to_impute).T

    # Get interactions between all variables and missing indicators
    interactions = all_nas * df

    # Add new columns to each of the dataframes to identify them
    all_nas["predictor_type"] = "na_indicator"
    df["predictor_type"] = "filled_value"
    interactions["predictor_type"] = "interaction"

    # Join the dataframes together
    df = pd.concat([df, all_nas, interactions], axis=0)

    # Add the new predictor_type column to the multi-index
    df = df.set_index("predictor_type", append=True)

    return df, to_impute


def fit_bias_correction_models(
    control_predictors: pd.DataFrame, control_outcomes: pd.DataFrame, target_metric: str
) -> list:
    """
    Fit bias correction models for each control tweet
    :param control_predictors: Dataframe of metrics for control tweets prior to treatment time of treatment tweet
    :param control_outcomes: Dataframe of metrics for control tweets after treatment time of treatment tweet
    :return:
    """
    models = list()
    r2s = list()
    maes = list()

    predictors_filtered = control_predictors.transpose().copy()

    for ts, y0 in tqdm(
        control_outcomes.iterrows(),
        total=len(control_outcomes),
        position=0,
        leave=True,
    ):
        # Fit model
        model = lm.LinearRegression().fit(y=y0, X=predictors_filtered.to_numpy())

        # Save model
        models.append(model)

        # Measure test performance
        r2s.append(
            (
                ts,
                r2_score(
                    y_true=y0,
                    y_pred=model.predict(predictors_filtered.to_numpy()),
                ),
            )
        )
        maes.append(
            (
                ts,
                mean_absolute_error(
                    y_true=y0,
                    y_pred=model.predict(predictors_filtered.to_numpy()),
                ),
            )
        )

    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        logger.info(f"{target_metric} R2s: {r2s}")
        logger.info(f"{target_metric} MAEs: {maes}")
        logger.info(f"{target_metric} Columns: {control_predictors.index}")

    return models


def apply_bias_correction(
    control_predictors: pd.DataFrame,
    control_outcomes: pd.DataFrame,
    treatment_predictors: pd.DataFrame,
    treatment_outcomes: pd.DataFrame,
    seq_weights: pd.DataFrame,
    models: list,
    var_means: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply bias correction to estimate a treatment effect for an individual treatment tweet

    :param control_predictors: Dataframe of metrics for control tweets prior to treatment time of treatment tweet
    :param control_outcomes: Dataframe of metrics for control tweets after treatment time of treatment tweet
    :param treatment_predictors: Dataframe of metrics for treatment tweet prior to treatment time
    :param treatment_outcomes: Dataframe of metrics for treatment tweet after treatment time
    :param seq_weights: Dataframe of weight assigned to each control tweet
    :param models: List of bias correction models, one for timestamp in control_outcomes
    :param var_means: Imputed values for each variable
    :return: Dataframe of bias-corrected treatment effects
    """
    # Initialize lists for storing results
    bias_corrected = []

    control_preds_imputed, _ = impute_nas(control_predictors, var_means)
    trt_preds_imputed, _ = impute_nas(treatment_predictors, var_means)

    # Iterate through points in time
    for model, (_, y0), (_, y1) in zip(
        models,
        control_outcomes.iterrows(),
        treatment_outcomes.iterrows(),
    ):
        if "impute_na_predictors" in config["bias_correction_missing_actions"]:
            # Get predictions for control and treatment
            mu_hat_0_control = model.predict(
                control_preds_imputed.transpose().to_numpy()
            )
        else:
            mu_hat_0_control = model.predict(control_predictors.transpose().to_numpy())

        if "impute_na_predictors" in config["bias_correction_missing_actions"]:
            mu_hat_0_treatment = model.predict(trt_preds_imputed.transpose().to_numpy())
        else:
            mu_hat_0_treatment = model.predict(
                treatment_predictors.transpose().to_numpy()
            )

        # Apply bias correction
        bias_corrected.append(
            (y1 - mu_hat_0_treatment).values[0]
            - (seq_weights * (y0 - mu_hat_0_control)).sum(axis=1).values[0]
        )

    return pd.DataFrame(
        {"bias_adjusted_treatment_effect": bias_corrected},
        index=control_outcomes.index,
    )


def filter_to_correct_predictors(predictor_matrix, target_metric, config):
    """
    Based on a predictor matrix for bias correction, filter down to only the predictor variables
    that will be used for this target metric.
    """

    valid_metrics = config["matching_metrics"] + [target_metric]
    valid_metrics += [f"{m}_filled" for m in valid_metrics]

    sliced_predictors = predictor_matrix[
        predictor_matrix.index.get_level_values(1).isin(valid_metrics)
    ]

    return sliced_predictors


def get_save_config(config):
    config_keys_to_save = [
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
        "restricted_pool_size",
        "lambda",
        "shuffle",
        "control_list",
    ]
    config_to_save = deepcopy({k: config[k] for k in config_keys_to_save if k in config})
    for k in config_to_save.keys():
        if type(config_to_save[k]) == list:
            config_to_save[k] = ",".join(config_to_save[k])

    return config_to_save


def build_predictor_matrix(
    tid,
    treatment_tweets,
    control_tweets,
    weights_to_use,
    config,
    control_config,
    author_df,
    log_name,
    sample_bias_correction_controls,
    run_name,
):
    matrix_building_logger = get_logger(
        dir=data_root / intermediate_dir,
        log_name=log_name,
        enqueue=True,
    )
    try:
        if matrix_building_logger:
            matrix_building_logger.info(f"Building predictor matrices for tweet {tid}")

        # In the metric sequence files, the treatment tweet is stored in a column
        # named "treatment." If we're calculating using the observed data, we need
        # to use this column name. Otherwise, (if we're calculating used the
        # permuted data), we need to swap in the tweet that is being used as
        # treatment
        treatment_column_name = (
            "treatment"
            if weights_to_use["tweet_id"].iloc[0] == tid
            else weights_to_use["tweet_id"].iloc[0]
        )

        break_time = (
            treatment_tweets[
                treatment_tweets["note_0_time_since_first_crh"]
                == -pd.to_timedelta(config["train_backdate"])
                ]
            .loc[tid]
            .index[0]
        )

        predict_until = break_time + pd.to_timedelta(config["post_break_min_time"])

        # Find out which covariates were used to match on
        predictor_variables = deepcopy(control_config["metrics_present_for_tweet"])
        outcome_variables = [
            m
            for m in control_config["metrics_present_for_tweet"]
            if m in config["target_metrics"]
        ]

        # If a new metric is being used as a target, even though it hasn't been matched on,
        # add it to the predictors and matching time ranges here
        for metric in config["target_metrics"]:

            # Check if the metric is already being used
            if metric in config["matching_metrics"]:
                continue

            # Get observations of this metric prior to the break
            pre_break_observations = treatment_tweets.loc[tid][
                (treatment_tweets.loc[tid]["note_0_time_since_first_crh"]
                 < -pd.Timedelta(config["train_backdate"]))
                & (
                        treatment_tweets.loc[tid]["note_0_time_since_first_crh"]
                        >= -(
                        pd.Timedelta(config["pre_break_min_time"])
                        + pd.Timedelta(config["train_backdate"])
                )
                )
                ][metric]

            # Skip if no observations
            if pre_break_observations.isna().any():
                continue

            # Get timestamps where metric is present
            pre_break_observations = (
                pre_break_observations[
                    treatment_tweets.loc[tid][f"present_in_{metric_parents[metric]}"]
                    & pre_break_observations.notna()
                    ]
                .index.total_seconds()
                .astype(int)
                .to_list()
            )

            # Get the post-break observations
            post_break_observations = treatment_tweets.loc[tid][
                (
                        treatment_tweets.loc[tid]["note_0_time_since_first_crh"]
                        >= -pd.Timedelta(config["train_backdate"])
                )
                & (
                        treatment_tweets.loc[tid]["note_0_time_since_first_crh"]
                        <= pd.Timedelta(config["post_break_min_time"])
                )
                ][metric]

            # Make sure we have observations for this metric post-break
            if post_break_observations.isna().any():
                continue
            else:
                # Add to the list of predictors in the regression
                predictor_variables.append(metric)

                # Add to the list of valid outcomes
                outcome_variables.append(metric)

                # Add to the list of matching timestamps
                control_config["matching_timestamps"].append(
                    [metric, pre_break_observations]
                )

        if config["include_bias_correction"]:

            # Convert to a list of dataframes, one for each predictor metric
            # (Slice to only the amount of time before the break that is required)
            treatment_predictors = {
                metric: treatment_tweets.loc[
                    [
                        (tid, td)
                        for td in pd.to_timedelta(matching_ts, unit="seconds")[
                            (
                                pd.to_timedelta(matching_ts, unit="seconds")
                                >= break_time
                                - pd.to_timedelta(
                                    config["pre_break_bias_correction_time"]
                                )
                            )
                            & (
                                pd.to_timedelta(matching_ts, unit="seconds")
                                < break_time
                            )
                        ]
                    ]
                ][[metric, "note_0_time_since_first_crh"]]
                .rename(columns={metric: tid})
                .reset_index()
                for metric, matching_ts in control_config["matching_timestamps"]
            }
            for metric in predictor_variables:
                if metric in config["author_metrics"]:
                    pass
                else:
                    treatment_predictors[metric]["metric"] = metric
                    treatment_predictors[metric] = (
                        treatment_predictors[metric]
                        .set_index(["note_0_time_since_first_crh", "metric"])
                        .drop(columns=["tweet_id"])
                    ).copy()

            treatment_predictors = pd.concat(treatment_predictors.values(), axis=0)

            # Add in author metrics:
            for metric in predictor_variables:
                if metric in config["author_metrics"]:
                    index_names = treatment_predictors.index.names
                    treatment_predictors = pd.concat(
                        [
                            treatment_predictors,
                            pd.DataFrame(
                                [author_df.loc[[tid]][[metric]].values[0][0]],
                                columns=[tid],
                                index=[(np.nan, metric)],
                            ),
                        ]
                    )
                    treatment_predictors.index.names = index_names

            # Get control tweets that were used
            ctids_to_use = sorted(
                set(control_config["control_tweet_ids"])
                .intersection(set(control_tweets.index.get_level_values(0)))
                .intersection(
                    set(weights_to_use.columns[weights_to_use.notna().values.flatten()])
                )
            )

            # Make sure we aren't somehow using more control tweets than we should be,
            # based on the donor pool size limitation
            if config["restrict_donor_pool"]:
                if len(ctids_to_use) > config["restricted_pool_size"]:
                    raise ConfigError(
                        f"Number of control tweets ({len(ctids_to_use):,}) exceeds "
                        f"restricted pool size ({config['restricted_pool_size']}:,) "
                        f"for tweet {tid}."
                    )

            control_predictors = {
                metric: control_tweets.loc[
                    [
                        i
                        for i in product(
                            ctids_to_use,
                            # Filter to 1 hr before treatment
                            pd.to_timedelta(matching_ts, unit="seconds")[
                                (
                                    pd.to_timedelta(matching_ts, unit="seconds")
                                    >= break_time
                                    - pd.to_timedelta(
                                        config["pre_break_bias_correction_time"]
                                    )
                                )
                                # Don't include any time after treatment
                                & (
                                    pd.to_timedelta(matching_ts, unit="seconds")
                                    < break_time
                                )
                            ],
                        )
                    ]
                ][[metric]]
                .pivot_table(index="time_since_publication", columns="tweet_id")
                .droplevel(0, axis=1)
                for metric, matching_ts in control_config["matching_timestamps"]
            }

            # Rename columns to just be TID
            for metric, _ in control_config["matching_timestamps"]:
                control_predictors[metric]["metric"] = metric
                control_predictors[metric] = (
                    control_predictors[metric]
                    .reset_index()
                    .set_index(["time_since_publication", "metric"])
                )

            # Concatenate into a single sequence
            control_predictors = pd.concat(control_predictors.values())

            # Add in author metrics:
            for metric in predictor_variables:
                if metric in config["author_metrics"]:
                    # Find which control_predictors columns are missing from author_df
                    missing_ids = [tid for tid in control_predictors.columns if tid not in author_df.index]
                    present_ids = [tid for tid in control_predictors.columns if tid in author_df.index]

                    # Get author values for present IDs
                    author_slice = author_df.loc[present_ids][[metric]].transpose()

                    # For missing IDs, fill with mean of present tweets, as we do in step f
                    if missing_ids:
                        mean_val = author_slice.loc[metric].mean()
                        for mid in missing_ids:
                            author_slice[mid] = mean_val

                    author_slice = author_slice[control_predictors.columns]

                    author_slice["metric"] = metric
                    author_slice["time_since_publication"] = np.nan
                    author_slice = author_slice.reset_index(drop=True).set_index(
                        ["time_since_publication", "metric"]
                    )
                    control_predictors = pd.concat([control_predictors, author_slice])

            # Currently control predictors are indexed on time_since_publication, while
            # treatment predictors are indexed on time since slap. We need to port the
            # treatment index over to the controls, now that they have the same time range
            # (First, double check that the time ranges are the same)
            assert (
                    (
                            treatment_predictors.index.get_level_values(1)
                            == control_predictors.index.get_level_values(1)
                    )
                    & (
                            (
                                    treatment_predictors["time_since_publication"]
                                    == control_predictors.index.get_level_values(0)
                            )
                            | (
                                    pd.isna(treatment_predictors["time_since_publication"])
                                    & pd.isna(control_predictors.index.get_level_values(0))
                            )
                    )
            ).all(), (
                "Time ranges for control and treatment predictors are not the same."
            )
            control_predictors.index = treatment_predictors.index
            treatment_predictors = treatment_predictors.drop(
                columns="time_since_publication"
            )

            if "replace_calculated" in config["bias_correction_missing_actions"]:
                for metric in config["backup_tweet_metrics"]:
                    # Create an indicator for whether non-calculated metrics are replacing
                    # calculated metrics for this tweet
                    calculated_metric_was_filled = (
                        (
                                (metric in control_config["metrics_present_for_tweet"])
                                and (f"calculated_{metric}" not in control_config["metrics_present_for_tweet"]))
                    )
                    index_names = control_predictors.index.names
                    control_predictors = pd.concat(
                        [
                            control_predictors,
                            pd.DataFrame(
                                calculated_metric_was_filled,
                                index=control_predictors.columns,
                                columns=[(np.nan, f"calculated_{metric}_filled")],
                            ).transpose(),
                        ]
                    )
                    control_predictors.index.names = index_names
                    treatment_predictors = pd.concat(
                        [
                            treatment_predictors,
                            pd.DataFrame(
                                calculated_metric_was_filled,
                                index=treatment_predictors.columns,
                                columns=[(np.nan, f"calculated_{metric}_filled")],
                            ).transpose(),
                        ]
                    )
                    treatment_predictors.index.names = index_names
                    # Actually fill in the metric
                    if calculated_metric_was_filled:

                        # Make sure we fill in the outcomes too
                        outcome_variables += [f"calculated_{metric}"]

                        timestamps_for_metric = control_predictors.index[
                            control_predictors.index.get_level_values(1) == metric
                        ].get_level_values(0)

                        for ts in timestamps_for_metric:
                            if pd.isna(ts):
                                continue

                            index_names = control_predictors.index.names
                            treatment_predictors = pd.concat(
                                [
                                    treatment_predictors,
                                    pd.DataFrame(
                                        treatment_predictors.loc[(ts, metric)].values,
                                        index=treatment_predictors.columns,
                                        columns=[(ts, f"calculated_{metric}")],
                                    ).transpose(),
                                ]
                            )
                            treatment_predictors.index.names = index_names
                            control_predictors = pd.concat(
                                [
                                    control_predictors,
                                    pd.DataFrame(
                                        control_predictors.loc[(ts, metric)].values,
                                        index=control_predictors.columns,
                                        columns=[(ts, f"calculated_{metric}")],
                                    ).transpose(),
                                ]
                            )
                            control_predictors.index.names = index_names

            else:
                if matrix_building_logger:
                    matrix_building_logger.error(
                        f"Do not know how to handle value other than 'replace_calculated' "
                        f"for 'bias_correction_missing_actions': "
                        f"{config['bias_correction_missing_actions']}"
                    )
                return None

            # Merge the predictor matrices for different sets of predictors together
            control_predictors = control_predictors.sort_index()
            treatment_predictors = treatment_predictors.sort_index()

            # Now, create the outcome matrices for control tweets,
            # which consist of all observations for the target variable post
            # treatment

            # Get the outcome matrices for the treatment tweet
            treatment_outcomes = {
                target_metric: (
                    treatment_tweets.loc[
                        [
                            (tid, td)
                            for td in pd.timedelta_range(
                                break_time, predict_until, freq=config["time_freq"]
                            )
                        ]
                    ][[target_metric, "note_0_time_since_first_crh"]]
                    .rename(columns={target_metric: tid})
                    .set_index("note_0_time_since_first_crh")
                )
                for target_metric in outcome_variables
            }

            control_outcomes = {
                target_metric: (
                    control_tweets.loc[
                        product(
                            ctids_to_use,
                            pd.timedelta_range(
                                break_time, predict_until, freq=config["time_freq"]
                            ),
                        )
                    ][[target_metric]]
                    .pivot_table(index="time_since_publication", columns="tweet_id")
                    .droplevel(0, axis=1)
                )
                for target_metric in outcome_variables
            }

            # If calculated_replies or calculated_retweets were filled in for replies or retweets,
            # we want to fill in the calculated outcome values for these metrics as well
            if "replace_calculated" in config["bias_correction_missing_actions"]:
                for metric in config["backup_tweet_metrics"]:
                    # Create an indicator for whether non-calculated metrics are replacing
                    # calculated metrics for this tweet
                    calculated_metric_was_filled = (
                        (
                                (metric in control_config["metrics_present_for_tweet"])
                                and (f"calculated_{metric}" not in control_config["metrics_present_for_tweet"]))
                    )
                    if calculated_metric_was_filled:
                        # Get the outcome matrices for the treatment tweet
                        treatment_outcomes[f"calculated_{metric}"] = (
                            (
                                treatment_tweets.loc[
                                    [
                                        (tid, td)
                                        for td in pd.timedelta_range(
                                        break_time, predict_until, freq=config["time_freq"]
                                    )
                                    ]
                                ][[metric, "note_0_time_since_first_crh"]]
                                .rename(columns={metric: tid})
                                .set_index("note_0_time_since_first_crh")
                            )
                        )
                        control_outcomes[f"calculated_{metric}"] = (
                            control_tweets.loc[
                                product(
                                    ctids_to_use,
                                    pd.timedelta_range(
                                        break_time, predict_until, freq=config["time_freq"]
                                    ),
                                )
                            ][[metric]]
                            .pivot_table(index="time_since_publication", columns="tweet_id")
                            .droplevel(0, axis=1)
                        )

            for target_metric in outcome_variables:
                control_outcomes[target_metric].index = treatment_outcomes[
                    target_metric
                ].index

            # Now, filter out this treatment tweet if it doesn't actually have enough data to be used
            # First, find if this treatment tweet does not have any data post-treatment
            zero_outcome_len = any(
                [len(treatment_outcomes[tm]) == 0 for tm in outcome_variables]
            )

            # Next, find if the treatment tweet does not have any data pre-treatment
            zero_trt_len = len(treatment_predictors) == 0

            if zero_outcome_len or zero_trt_len:
                if matrix_building_logger:
                    matrix_building_logger.info(
                        f"Removing {tid} from estimates for directory, due to a lack "
                        f"of data either pre or post treatment."
                    )
                return None

        else:
            control_predictors = None
            treatment_predictors = None
            control_outcomes = None
            treatment_outcomes = None

    except Exception as e:
        if matrix_building_logger:
            matrix_building_logger.error(f"Error with tweet id {tid}")
            matrix_building_logger.error(e)
            matrix_building_logger.error(traceback.format_exc())
        return None

    # Go through, and find which target metrics this tweet has enough data to
    # be used for estimating
    valid_target_metrics = []
    for target_metric in outcome_variables:
        necessary_predictors = treatment_predictors.index.get_level_values(1).isin(
            control_config["metrics_present_for_tweet"] + [target_metric]
        )

        if (
            treatment_predictors[necessary_predictors].isna().any().any()
            or np.isinf(treatment_predictors[necessary_predictors]).any().any()
        ):
            if matrix_building_logger:
                matrix_building_logger.info(
                    f"Found NAs or Infinities in the treatment predictor matrix for tweet {tid} for "
                    f"metric {target_metric}. "
                    f"These will later be imputed, assuming 'impute_na_predictors' is a "
                    f"bias_correction_missing_action in this config."
                )

        if (
            control_predictors[necessary_predictors].isna().any().any()
            or np.isinf(control_predictors[necessary_predictors]).any().any()
        ):
            if matrix_building_logger:
                matrix_building_logger.info(
                    f"Found NAs or Infinities in the control predictor matrix for tweet {tid} for "
                    f"metric {target_metric}. "
                    f"These will later be imputed, assuming 'impute_na_predictors' is a "
                    f"bias_correction_missing_action in this config."
                )

        if (
            treatment_outcomes[target_metric].isna().any().any()
            or np.isinf(treatment_outcomes[target_metric]).any().any()
        ):
            if matrix_building_logger:
                matrix_building_logger.info(
                    f"Removing {tid} due to NAs or Infinities in the "
                    f"treatment outcome matrix for metric {target_metric}. "
                    f"(The treatment effect cannot be calculated if there are NAs in "
                    f"the treatment metric of interest after the slap occurs)."
                )
            continue

        if (
            control_outcomes[target_metric].isna().any().any()
            or np.isinf(control_outcomes[target_metric]).any().any()
        ):
            if matrix_building_logger:
                matrix_building_logger.info(
                    f"Removing {tid} due to NAs or Infinities in the "
                    f"control outcome matrix for metric {target_metric}. "
                    f"(The synthetic control cannot be calculated if there are NAs in "
                    f"the control outcome matrix, as some weight may be assigned to NA control tweets, "
                    f"which would result in the control overall being NA)."
                )
            continue

        valid_target_metrics.append(target_metric)

    if sample_bias_correction_controls > 0:
        # Save the predictor matrices
        cache_dir = (
            data_root
            / intermediate_dir
            / f".controls_{run_name.replace('.log', '')}"
        )
        cache_dir.mkdir(exist_ok=True, parents=True)

        cache_path = cache_dir / f"{tid}_control_predictors.parquet"
        control_predictors.to_parquet(cache_path)

        for metric in valid_target_metrics:
            cache_path = cache_dir / f"{tid}_control_outcomes_{metric}.parquet"
            control_outcomes[metric].to_parquet(cache_path)

        # set seed & sort by tweet seeds
        rng = np.random.default_rng(TWEET_SEEDS[tid])

        control_predictors_columns = control_predictors.columns
        control_predictors_columns = control_predictors_columns.sort_values(
            key=lambda x: [TWEET_SEEDS[c.replace(".parquet", "")] for c in x]
        )

        sample_control_ids = rng.choice(
            control_predictors_columns,
            size=min(sample_bias_correction_controls, len(control_predictors_columns)),
            replace=False,
        )

        control_predictors = control_predictors[sample_control_ids]
        control_outcomes = {
            k: v[[c for c in sample_control_ids if c in v.columns]]
            for k, v in control_outcomes.items()
        }

    matrix_building_logger.info(f"Predictor matrix build for tid {tid}")

    return (
        tid,
        treatment_column_name,
        control_predictors,
        treatment_predictors,
        control_outcomes,
        treatment_outcomes,
        valid_target_metrics,
    )


def calculate_tweet_level_effects(
    target_metric: str,
    tid: str,
    log_name: str,
    control_config: dict,
    control_outcome_matrix: pd.DataFrame,
    treatment_outcome_matrix: pd.DataFrame,
    control_predictor_matrix: pd.DataFrame,
    treatment_predictor_matrix: pd.DataFrame,
    treatment_column_name: str,
    config: dict,
    weights_to_use: pd.DataFrame,
    save_dir: Path,
    var_means: pd.DataFrame,
    bias_correction_models: list,
    sample_bias_correction_controls: int = 0,
):

    te_estimation_logger = get_logger(
        dir=data_root / intermediate_dir,
        log_name=log_name,
        enqueue=True,
    )

    te_estimation_logger.info(
        f"Calculating treatment effects for metric {target_metric} for tweet {tid}"
    )

    if sample_bias_correction_controls > 0:
        te_estimation_logger.info(
            f"Reading in control predictors and outcomes for bias correction from disk "
            f"for metric {target_metric} for tweet {tid}."
        )
        control_predictor_matrix_path = (
            data_root
            / intermediate_dir
            / f".controls_{log_name.replace('.log','')}"
            / f"{tid}_control_predictors.parquet"
        )
        control_outcome_matrix_path = (
            data_root
            / intermediate_dir
            / f".controls_{log_name.replace('.log','')}"
            / f"{tid}_control_outcomes_{target_metric}.parquet"
        )

        control_predictor_matrix = filter_to_correct_predictors(
            pd.read_parquet(control_predictor_matrix_path),
            target_metric,
            config,
        )
        control_outcome_matrix = pd.read_parquet(control_outcome_matrix_path)
        te_estimation_logger.info(
            f"Control predictors and outcomes for bias correction read from disk "
            f"for metric {target_metric} for tweet {tid}."
        )

    ctids_to_use = sorted(
        set(control_config["control_tweet_ids"])
        .intersection(control_predictor_matrix.columns)
        .intersection(
            set(weights_to_use.columns[weights_to_use.notna().values.flatten()])
        )
    )

    te_estimation_logger.info(
        f"Using {len(ctids_to_use):,} control tweets for metric {target_metric} for tweet {tid}."
    )

    if len(ctids_to_use) < len(control_config["control_tweet_ids"]):
        if config["restrict_donor_pool"]:
            if len(ctids_to_use) < config["restricted_pool_size"]:
                te_estimation_logger.warning(
                    f"Using {len(ctids_to_use):,} Control IDs for tweet {tid}, which is less than the "
                    f"{len(control_config['control_tweet_ids']):,} specified in the control configuration file. "
                    f"This should only occur if the post had duplicates dropped during weight calculation."
                )
        else:
            te_estimation_logger.error(
                f"Using {len(ctids_to_use):,} Control IDs for tweet {tid}, which is less than the "
                f"{len(control_config['control_tweet_ids']):,} specified in the control configuration file. "
                f"This should only occur if the post had duplicates dropped during weight calculation."
            )

    te_estimation_logger.info(
        f"Starting calculation of unadjusted TEs "
        f"for metric {target_metric} for tweet {tid}."
    )

    seq_df = control_outcome_matrix.copy()
    seq_df[treatment_column_name] = treatment_outcome_matrix[tid]

    # Get unadjusted control sequence
    treatment_and_control = combine_sequence(
        seq_df,
        weights_to_use[ctids_to_use],
        treatment_column_name=treatment_column_name,
    )

    # Calculate unadjusted treatment effect
    treatment_and_control["unadjusted_treatment_effect"] = (
        treatment_and_control["treatment"] - treatment_and_control["control"]
    )

    te_estimation_logger.info(
        f"Finished calculation of unadjusted TEs "
        f"for metric {target_metric} for tweet {tid}."
    )

    if config["include_bias_correction"]:
        try:
            te_estimation_logger.info(
                f"Starting calculation of bias-corrected TEs "
                f"for metric {target_metric} for tweet {tid}."
            )

            # Get adjusted treatment effect
            bias_adjusted_seq = apply_bias_correction(
                control_predictors=control_predictor_matrix[ctids_to_use],
                treatment_predictors=treatment_predictor_matrix,
                control_outcomes=control_outcome_matrix[ctids_to_use],
                treatment_outcomes=treatment_outcome_matrix,
                seq_weights=weights_to_use[list(ctids_to_use)],
                models=bias_correction_models,
                var_means=var_means,
            )

            te_estimation_logger.info(
                f"Finished calculation of bias-corrected TEs "
                f"for metric {target_metric} for tweet {tid} ."
            )

            # Merge the unadjusted and adjusted
            treatment_and_control = treatment_and_control.merge(
                bias_adjusted_seq,
                how="left",
                left_on="note_0_time_since_first_crh",
                right_index=True,
            )

            te_estimation_logger.info(
                f"Merged raw and bias-corrected TEs "
                f"for metric {target_metric} for tweet {tid}."
            )

        except Exception as e:
            if te_estimation_logger:
                te_estimation_logger.error(
                    f"Skipping tweet {tid} for metric {target_metric}: {e}"
                )
            return "Error"

    # Find the save path
    effects_dir = save_dir / target_metric
    effects_dir.mkdir(exist_ok=True, parents=True)
    effects_path = effects_dir / f"{tid}.parquet"

    # Include weight ID here
    treatment_and_control["weight_id"] = weights_to_use["weight_id"].iloc[0]
    treatment_and_control["tweet_id"] = tid

    # Revert time-since-CRH to being a columns
    treatment_and_control = treatment_and_control.reset_index()

    # Save!
    save_config = get_save_config(config)
    for k in save_config:
        treatment_and_control[k] = save_config[k]
    clear_and_write(treatment_and_control, effects_path, config=save_config)

    te_estimation_logger.info(f"TE complete for metric {target_metric} for tweet {tid}")

    return "Complete"


def calc_treatment_effects(
    treatment_tweets: dict[str, pd.DataFrame],
    control_tweets: dict[str, pd.DataFrame],
    control_configs: dict[str],
    weights: dict[str, list],
    author_df: pd.DataFrame,
    save_dir: Path,
    logger: Optional[Logger] = None,
    log_name: Optional[str] = None,
    sample_bias_correction_controls: int = 0,
) -> None:
    """
    Calculate average treatment effect across all treatment tweets

    :param tids: List of tweet IDs
    :param seqs: Dictionary of sequences of metrics for each tweet
    :param weights: Amount of weight assigned to each control tweet for each treatment tweet
    :param save_dir: Directory to save treatment effects
    :param logger: Logger object (optional)
    :return: None
    """

    tid_results = Parallel(
        n_jobs=NUM_PROCESSES,
        backend="loky",
        prefer="processes",
        temp_folder=data_root / intermediate_dir / "tmp",
        max_nbytes=10,
    )(
        delayed(build_predictor_matrix)(
            tid=tid,
            treatment_tweets=treatment_tweets,
            control_tweets=control_tweets,
            weights_to_use=weights[tid].iloc[[0]],
            config=deepcopy(config),
            control_config=control_configs[tid][0],
            author_df=author_df,
            log_name=log_name,
            sample_bias_correction_controls=sample_bias_correction_controls,
            run_name=log_name,
        )
        for tid in tqdm(
            weights.keys(),
            desc="Building predictor matrices",
            position=0,
            leave=True,
        )
    )

    logger.info("Predictor matrices built")

    # Initialize objects for storing tweet-level info that will need to be reused
    treatment_column_names = {}
    control_predictor_matrices = {}
    treatment_predictor_matrices = {}
    control_outcome_matrices = {}
    treatment_outcome_matrices = {}
    valid_target_metrics = {}

    # Some TIDs don't have data directly before/after slap, and we filter those out here
    tids_with_adequate_data = []

    # Iterate through treatment tweets
    for tid_metadata in tid_results:
        if tid_metadata is not None:
            tid = tid_metadata[0]
            if tid not in tids_with_adequate_data:
                tids_with_adequate_data.append(tid)
                (
                    _,
                    treatment_column_names[tid],
                    control_predictor_matrices[tid],
                    treatment_predictor_matrices[tid],
                    control_outcome_matrices[tid],
                    treatment_outcome_matrices[tid],
                    valid_target_metrics[tid],
                ) = tid_metadata

    # Check which tweets can get TE for which variables
    tids_for_each_target_metric = defaultdict(list)
    not_used_tids = defaultdict(list)
    for target_metric in config["target_metrics"]:
        for tid in tids_with_adequate_data:
            if target_metric in valid_target_metrics[tid]:
                tids_for_each_target_metric[target_metric].append(tid)
            else:
                not_used_tids[target_metric].append(tid)

    # Log which tweets are used for which target variables
    for target_metric in set(tids_for_each_target_metric.keys()).union(
        not_used_tids.keys()
    ):
        if logger:
            messenger = (
                logger.warning if len(not_used_tids[target_metric]) else logger.info
            )
            messenger(
                f"{len(tids_for_each_target_metric[target_metric]):,} Tweet IDs used for metric {target_metric}"
                + (
                    f": {tids_for_each_target_metric[target_metric]}"
                    if len(tids_for_each_target_metric[target_metric])
                    else "."
                )
            )
            messenger(
                f"{len(not_used_tids[target_metric]):,} Tweet IDs NOT USED for metric {target_metric}"
                + (
                    f": {not_used_tids[target_metric]}"
                    if len(not_used_tids[target_metric])
                    else "."
                )
            )

    #############################################################################################
    if config["include_bias_correction"]:
        # Now merge the data for all tweets together

        variables_used = {}
        bias_correction_models = {}
        predictor_means = defaultdict(list)
        for metric in tids_for_each_target_metric.keys():
            logger.info("Constructing control predictor bias-correction matrices")

            # Create predictor matrix
            all_control_predictors = pd.concat(
                {
                    tid: filter_to_correct_predictors(
                        control_predictor_matrices[tid], metric, config
                    )
                    for tid in tids_for_each_target_metric[metric]
                },
                axis=1,
            )

            logger.info(
                f"Bias-correction predictors constructed for target metric {metric}. "
                f"Shape: {all_control_predictors.shape}.\n"
                f"Axis: {all_control_predictors.index}"
            )

            # Create outcome matrix
            logger.info("Constructing control outcome bias-correction matrices")
            all_control_outcomes = pd.concat(
                {
                    tid: control_outcome_matrices[tid][metric]
                    for tid in tids_for_each_target_metric[metric]
                },
                axis=1,
            )

            logger.info(
                f"Bias-correction outputs constructed for target metric {metric}. "
                f"Shape: {all_control_outcomes.shape}.\n"
                f"Axis: {all_control_outcomes.index}"
            )

            # Get the subset of columns in both multi-indexes
            columns_in_both = all_control_predictors.columns.intersection(
                all_control_outcomes.columns
            )

            if len(all_control_predictors.columns.difference(columns_in_both)) > 0:
                logger.error(
                    f"Columns in predictors but not outcomes for metric {metric}: "
                    f"{all_control_predictors.columns.difference(columns_in_both)}"
                )
                all_control_predictors = all_control_predictors[columns_in_both]
                all_control_outcomes = all_control_outcomes[columns_in_both]
            if len(all_control_outcomes.columns.difference(columns_in_both)) > 0:
                logger.error(
                    f"Columns in outcomes but not predictors for metric {metric}: "
                    f"{all_control_outcomes.columns.difference(columns_in_both)}"
                )
                all_control_predictors = all_control_predictors[columns_in_both]
                all_control_outcomes = all_control_outcomes[columns_in_both]

            if all_control_outcomes.isna().any().any():
                logger.error(
                    f"There are NAs in the control outcomes for metric {metric}. This should "
                    "not be possible!"
                )

            # Handle NAs
            if "impute_na_predictors" in config["bias_correction_missing_actions"]:
                all_control_predictors, predictor_means[metric] = impute_nas(
                    all_control_predictors
                )
            else:
                raise ConfigError(
                    f"Unsure how to handle NAs in controls for bias correction: "
                    f"{config['bias_correction_missing_actions']}"
                )

            variables_used[metric] = all_control_predictors.index.copy()

            #############################################################################################
            # Fit the models adjustment/bias-correction regressions
            try:
                bias_correction_models[metric] = fit_bias_correction_models(
                    control_predictors=all_control_predictors,
                    control_outcomes=all_control_outcomes,
                    target_metric=metric,
                )
                logger.info("Outcome models fit")

            except Exception as e:
                if logger:
                    logger.error(f"Error fitting bias correction models for {metric}")
                    logger.error(e)

            del all_control_predictors, all_control_outcomes

    #############################################################################################

    for target_metric in tids_for_each_target_metric.keys():
        if target_metric in bias_correction_models.keys():
            # Calculate treatment effects
            Parallel(
                n_jobs=NUM_PROCESSES,
                backend="threading",
                temp_folder=data_root / intermediate_dir / "tmp",
                max_nbytes=10,
            )(
                delayed(
                    calculate_tweet_level_effects,
                )(
                    target_metric=target_metric,
                    tid=tid,
                    log_name=log_name,
                    control_config=control_configs[tid][0],
                    control_outcome_matrix=control_outcome_matrices[tid][target_metric],
                    treatment_outcome_matrix=treatment_outcome_matrices[tid][
                        target_metric
                    ],
                    control_predictor_matrix=filter_to_correct_predictors(
                        control_predictor_matrices[tid],
                        target_metric,
                        config,
                    ),
                    treatment_predictor_matrix=filter_to_correct_predictors(
                        treatment_predictor_matrices[tid],
                        target_metric,
                        config,
                    ),
                    treatment_column_name=treatment_column_names[tid],
                    weights_to_use=weights[tid].iloc[[0]],
                    config=config,
                    save_dir=save_dir,
                    var_means=predictor_means[target_metric],
                    bias_correction_models=bias_correction_models[target_metric],
                    sample_bias_correction_controls=sample_bias_correction_controls,
                )
                for tid in tqdm(
                    tids_for_each_target_metric[target_metric],
                    desc="Calculating tweet-level effects",
                    position=0,
                    leave=True,
                )
            )


def condense_columns(tweet_df):

    cols_to_use = (
        [
            "tweet_id",
            "created_at",
            "timestamp",
            "time_since_publication",
            "note_0_time_since_first_crh",
            "note_0_first_crh_time",
        ]
        + config["tweet_metrics"]
        + config["backup_tweet_metrics"]
        + [
            m
            for m in config["target_metrics"]
            if m not in config["tweet_metrics"]
            and m not in config["backup_tweet_metrics"]
        ]
        + np.unique(
            [
                f"present_in_{metric_parents[m]}"
                for m in config["tweet_metrics"]
                + config["backup_tweet_metrics"]
                + config["target_metrics"]
            ]
        ).tolist()
    )

    tweet_df = tweet_df[[c for c in cols_to_use if c in tweet_df.columns]].copy()
    return tweet_df


if __name__ == "__main__":
    # Read config file
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to config file")
    args = parser.parse_args()
    config_path = args.config
    with open(config_path, "r") as fin:
        config = json.loads(fin.read())

    # Make sure config has everything we need at this point in pipeline
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

    # Drop unneeded config values
    config = {c: config[c] for c in chain(necessary_config, optional_config) if c in config.keys()}

    # Convert lambda from list
    if len(config["lambda"]) == 1:
        config["lambda"] = config["lambda"][0]
    else:
        raise ConfigError("Unsure how to handle multiple lambda values in same config.")

    # Fill in the dev value based on whether this is a local run
    if config["dev"] == "DEVICE_DEFAULT":
        config["dev"] = socket.gethostname() == "is-is28m16x"

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
    output_dir = data_root / intermediate_dir / "h_treatment_effects"
    output_dir.mkdir(parents=True, exist_ok=True)

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

    # Read sequences of metrics (likes, retweets, etc.)
    control_tweets, treatment_tweets = read_trt_and_ctrl(
        intermediate_dir, trt_and_ctrl_config, log_name=log_name, sample_size=None
    )

    # Discard unused columns
    for tid in control_tweets.keys():
        control_tweets[tid] = condense_columns(control_tweets[tid])
        control_tweets[tid] = control_tweets[tid].drop(
            columns=[
                "note_0_time_since_first_crh",
                "note_0_first_crh_time",
            ],
            errors="ignore",
        )
    for tid in treatment_tweets.keys():
        treatment_tweets[tid] = condense_columns(treatment_tweets[tid])

    control_configs = read_control_configs(intermediate_dir, config)

    # Make sure there's only one configuration per tweet
    for tid in control_configs.keys():
        if len(control_configs[tid]) != 1:
            logger.error(
                f"Control config for {tid} has {len(control_configs[tid])} entries"
            )

    # Read weights
    logger.info("Loading calculated_weights")
    weights_config = deepcopy(config)
    weights = read_weights(
        intermediate_dir,
        weights_config,
        treatment_tweet_ids=list(treatment_tweets.keys()),
    )
    logger.info(
        f"Weights loaded for {len(weights)} tweets. "
        f"Treatment tweets whose weights could not be loaded: "
        f"{[tid for tid in treatment_tweets.keys() if tid not in weights.keys()]}"
    )

    # Preprocess treatment tweets
    for tid in tqdm(
        sorted(treatment_tweets.keys()),
        desc="Preprocessing treatment tweets",
        smoothing=0,
    ):
        # Set index to time since first CRH
        treatment_tweets[tid] = (
            treatment_tweets[tid]
            .set_index(["tweet_id", "time_since_publication"])
            .sort_index()
        )

    treatment_tweets = pd.concat(treatment_tweets.values(), axis=0)

    logger.info("Built treatment df")

    for cid in tqdm(
        sorted(control_tweets.keys()), desc="Preprocessing control tweets", smoothing=0
    ):
        # Set the index to time since tweet was created
        control_tweets[cid] = (
            control_tweets[cid]
            .set_index(["tweet_id", "time_since_publication"])
            .sort_index()
        )

    control_tweets = pd.concat(control_tweets.values(), axis=0)

    logger.info("Built control df")

    calc_treatment_effects(
        treatment_tweets,
        control_tweets,
        control_configs,
        weights,
        author_df,
        save_dir=output_dir,
        logger=logger,
        log_name=log_name,
        sample_bias_correction_controls=config["sample_bias_correction_controls"],
    )
