import glob
import sys
import os
import re
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd
import numpy as np
from tqdm import tqdm

# Add current directory to path for imports
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.utils import camel_to_snake, informative_merge, remove_urls
from src.pipeline.a_preprocess import local_data_root, shared_data_root
from src.analysis.plot_treatment_effects import get_artifact_dir
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

dev = False
SAMPLE = False

intermediate_dir = local_data_root / Path(
    "cn_effect_intermediate" + ("_dev" if dev else "_prod")
)
artifact_dir = local_data_root / intermediate_dir / get_artifact_dir({"dev": dev})
prod_dir = local_data_root / "cn_effect_intermediate_prod"
merged_dir = prod_dir / "b_merged"

def find_note_tweet_pairs(config: Dict[str, Any], force: bool = False) -> None:
    """
    Finds note-tweet pairs and calculates the amount of time a note spent CRH (Currently Rated Helpful)
    in the 48 hours after the first CRH of the tweet.

    Args:
        config (dict): Configuration dictionary containing 'post_break_min_time' and 'train_backdate'.
        force (bool): If True, forces the function to run even if the output file already exists.

    Returns:
        None
    """
    if not force and (artifact_dir / "analyzed_notes_and_tweets.csv").exists():
        return

    # Get paths for metrics for all tweets
    merged_files = [Path(f) for f in glob.glob(str(merged_dir / "*.parquet"))]

    # Get the tweet IDs of these tweets
    tweet_ids = [f.name.replace(".parquet", "") for f in merged_files]

    # Create a list to store the note-tweet pairs in
    note_tweet_pairs = []

    # Create a list to store the total CRH times post slap
    total_crh_times_post_slap = []

    # Iterate through all tweets
    for tweet_id in tqdm(tweet_ids, "Creating Note Tweet Pairs DF"):

        # Read the tweet's metrics data
        tweet_df = pd.read_parquet(
            merged_dir / f"{tweet_id}.parquet",
        )

        # If there was a slap for this tweet, filter to only the 48h after the slap
        if tweet_df["note_0_time_since_first_crh"].notna().any():
            tweet_df = tweet_df[
                (
                    (
                        tweet_df["note_0_time_since_first_crh"]
                        <= pd.Timedelta(config["post_break_min_time"])
                        - pd.Timedelta(config["train_backdate"])
                    )
                    & (
                        tweet_df["note_0_time_since_first_crh"]
                        >= -pd.Timedelta(config["train_backdate"])
                    )
                )
            ]

        # Get the total amount of time that the tweet spent with at
        # least one CRH note in the 48h post slap
        total_crh_times_post_slap.append(
            {
                "tweet_id": tweet_id,
                "total_crh_times_post_slap": (
                    tweet_df[
                        [
                            col
                            for col in tweet_df.columns
                            if re.match(r"note_.*_twitter_status", col)
                        ]
                    ]
                    == "CURRENTLY_RATED_HELPFUL"
                )
                .any(axis=1)
                .sum()
                / 4,
            }
        )

        # Count number of notes by getting columns that are of form "note_*_note_id"
        num_notes = len(
            [col for col in tweet_df.columns if re.match(r"note_.*_note_id", col)]
        )

        # Iterate through all notes for this tweet
        for note in range(num_notes):

            # Get the note ID
            note_id = [
                nid
                for nid in tweet_df[f"note_{note}_note_id"].unique()
                if nid is not None
            ][0]

            # Find out whether this note ever had a CRH rating in the 48h post slap
            ever_crh = (
                tweet_df[f"note_{note}_twitter_status"] == "CURRENTLY_RATED_HELPFUL"
            ).any()

            # Get the amount of time that this note spent CRH in the 48h post slap
            total_crh_time_in_48h_post_slap = (
                tweet_df[f"note_{note}_twitter_status"]
                .value_counts()
                .get("CURRENTLY_RATED_HELPFUL", 0)
                / 4
            )

            # Append to the list of note-tweet pairs
            note_tweet_pairs.append(
                {
                    "tweet_id": tweet_id,
                    "note_id": note_id,
                    "ever_crh_in_48h_post_slap": ever_crh,
                    "approx_num_hours_crh_in_48h_post_slap": total_crh_time_in_48h_post_slap,
                }
            )

    # Save the note-tweet pairs to a CSV
    note_tweet_pairs_df = pd.DataFrame(note_tweet_pairs)
    note_tweet_pairs_df.to_csv(
        artifact_dir / "analyzed_notes_and_tweets.csv", index=False
    )

    # Save the total CRH times post slap to a CSV
    total_crh_times_post_slap = pd.DataFrame(total_crh_times_post_slap)
    total_crh_times_post_slap.to_csv(
        artifact_dir / "total_crh_times_post_slap.csv", index=False
    )

if __name__ == "__main__":

    ################################
    # Create note-tweet pairs dataframe
    find_note_tweet_pairs(
        config={"post_break_min_time": "48h", "train_backdate": "0h"}, force=not SAMPLE
    )

    ################################
    # read note-tweet pairs
    df_tweets_notes_all = pd.read_csv(
        artifact_dir / "analyzed_notes_and_tweets.csv",
        dtype={"tweet_id": str, "note_id": str},
    )

    # Filter to only notes that were CRH in the 48h post slap
    df_tweets_notes = df_tweets_notes_all[
        df_tweets_notes_all["ever_crh_in_48h_post_slap"]
    ]

    ################################
    # read note ratings
    cn_csvs_dir = Path(shared_data_root / "public-releases-csvs" / "downloads/")

    # List the dates where we downloaded data
    date_dirs = os.listdir(cn_csvs_dir)

    # Sample the dates to read
    if SAMPLE:
        date_dirs = np.random.choice(date_dirs, 20, replace=False)

    # Create an empty dataframe that we can later append ratings to
    notes = pd.DataFrame({})

    # Iterate through
    for date_dir in tqdm(
        date_dirs,
        smoothing=0,
        desc="Reading note ratings of tweets (e.g. notes-00000.tsv)",
    ):
        try:
            data_for_this_date = pd.read_csv(
                cn_csvs_dir / date_dir / "notes-00000.tsv",
                sep="\t",
                dtype={"tweetId": str, "noteId": str},
                low_memory=False,
            )
            data_for_this_date["date_dir"] = date_dir
            notes = pd.concat(
                [
                    notes,
                    data_for_this_date,
                ]
            ).drop_duplicates()
        except Exception:
            print(f"Error reading from directory {date_dir}")

    # Drop duplicates, only keeping the last observation of the rating
    notes = notes.sort_values("date_dir")
    notes = notes.drop_duplicates(subset=["noteId", "tweetId"], keep="last")

    # Select right columns
    note_columns = [
        "noteId",
        "tweetId",
        "misleadingOther",
        "misleadingFactualError",
        "misleadingManipulatedMedia",
        "misleadingOutdatedInformation",
        "misleadingMissingImportantContext",
        "misleadingUnverifiedClaimAsFact",
        "misleadingSatire",
        "notMisleadingOther",
        "notMisleadingFactuallyCorrect",
        "notMisleadingOutdatedButNotWhenWritten",
        "notMisleadingClearlySatire",
        "notMisleadingPersonalOpinion",
        "trustworthySources",
        "summary",
    ]
    notes = notes[note_columns]

    # Rename columns
    notes = notes.rename(columns=camel_to_snake)

    # Rename columns to clarify source
    notes = notes.rename(columns=lambda x: f"tweet_rated_{x}").rename(
        columns={
            "tweet_rated_note_id": "note_id",
            "tweet_rated_tweet_id": "tweet_id",
            "tweet_rated_summary": "note_full_text",
        }
    )

    if SAMPLE:
        notes = notes.sample(frac=1).reset_index(drop=True)

    ################################
    # Label using valence, subjectivity etc. lexica
    #

    # Make sure index is range:
    notes = notes.reset_index(drop=True)

    # Fill NAs with empty strings
    notes["note_full_text"] = notes["note_full_text"].fillna("")

    # NB: removing some punctuation at the beginning of the note, and remove_urls
    stripped_notes = [t.lstrip(".,:& ") for t in notes["note_full_text"]]
    stripped_notes = [remove_urls(t) for t in stripped_notes]

    # Measure readability
    note_readability = pd.DataFrame(
        [
            measure_readability(stripped_notes[i])
            for i in tqdm(
                range(len(notes)), desc="Measure note readability", smoothing=0
            )
        ]
    )

    # Tokenize
    tokenized_notes = [
        parse_headline(remove_urls(notes["note_full_text"].iloc[i]))._.clf_token_texts
        for i in tqdm(range(len(notes)), desc="Tokenizing Notes", smoothing=0)
    ]

    # Measure valence, subjectivity, etc. with lexicons
    lexicons, valued_lexicons, fields = load_all_lexicons()
    note_lexical_analysis = pd.DataFrame(
        [
            process_tokens(tokenized_notes[i], lexicons, valued_lexicons, fields)
            for i in tqdm(
                range(len(notes)),
                desc="Calculating text stats with lexicons for notes",
                smoothing=0,
            )
        ]
    )

    # Clarify that these columns are for the tweet text
    note_readability = note_readability.add_prefix("note_text_")
    note_lexical_analysis = note_lexical_analysis.add_prefix("note_text_")

    # Merge to other covars
    notes = pd.concat([notes, note_readability, note_lexical_analysis], axis=1)

    ################################
    # Read slap times
    #
    slap_times = pd.read_csv(
        artifact_dir / "a_slap_times.csv",
        dtype={"tweet_id": str, "note_id": str},
    )
    slap_times["first_crh"] = pd.to_datetime(
        slap_times["first_crh"], format="mixed", utc=True
    )

    # Merge ratings with slap times
    slap_times = informative_merge(
        df_tweets_notes,
        slap_times,
        "Note-tweet pairs for slapped tweets",
        "Slap times",
        on="tweet_id",
    )

    ################################
    # Read ratings data

    # List the dates where we downloaded data
    date_dirs = os.listdir(cn_csvs_dir)

    # Sample the dates to read
    if SAMPLE:
        date_dirs = np.random.choice(date_dirs, 8, replace=False)

    # Create an empty dataframe that we can later append ratings to
    ratings = pd.DataFrame({})

    # Iterate through days we have downloaded data for
    for date_dir in tqdm(
        date_dirs, smoothing=0, desc="Reading ratings data (e.g. ratings-00000.tsv)"
    ):

        # Find the ratings files (there can be more than 1 on a given day)
        ratings_files = [
            f for f in os.listdir(cn_csvs_dir / date_dir) if f.startswith("ratings-")
        ]

        # Iterate through the ratings files
        for ratings_file in ratings_files:
            try:
                # Read new file
                new_ratings = pd.read_csv(
                    cn_csvs_dir / date_dir / ratings_file,
                    sep="\t",
                    dtype={"noteId": str},
                    low_memory=False,
                )

                # Merge to slap times
                new_ratings = new_ratings.merge(
                    slap_times.rename(columns={"note_id": "noteId"})
                )

                # Join the previous ratings to the new ones
                ratings = pd.concat([ratings, new_ratings])

                # Convert to datetime
                ratings["rating_time"] = pd.to_datetime(
                    ratings["createdAtMillis"], unit="ms", utc=True
                )

                # Filter to pre-slap ratings
                ratings = ratings[ratings["rating_time"] <= ratings["first_crh"]]

                # Drop duplicated ratings
                ratings = ratings.drop_duplicates(
                    subset=["noteId", "raterParticipantId"]
                )

            except Exception:
                print(f"Error reading from directory {date_dir}")

    # Select right columns
    ratings_columns_to_sum = [
        "agree",
        "disagree",
        "helpfulOther",
        "helpfulInformative",
        "helpfulClear",
        "helpfulEmpathetic",
        "helpfulGoodSources",
        "helpfulUniqueContext",
        "helpfulAddressesClaim",
        "helpfulImportantContext",
        "helpfulUnbiasedLanguage",
        "notHelpfulOther",
        "notHelpfulIncorrect",
        "notHelpfulSourcesMissingOrUnreliable",
        "notHelpfulOpinionSpeculationOrBias",
        "notHelpfulMissingKeyPoints",
        "notHelpfulOutdated",
        "notHelpfulHardToUnderstand",
        "notHelpfulArgumentativeOrBiased",
        "notHelpfulOffTopic",
        "notHelpfulSpamHarassmentOrAbuse",
        "notHelpfulIrrelevantSources",
        "notHelpfulOpinionSpeculation",
        "notHelpfulNoteNotNeeded",
    ]

    # Aggregate ratings
    total_ratings = ratings.groupby(["noteId"])[ratings_columns_to_sum].sum()

    # Aggregate helpfulness, which is stored as a string
    helpfulness_counts = (
        ratings.groupby("noteId")["helpfulnessLevel"]
        .value_counts()
        .reset_index()
        .pivot(index="noteId", columns="helpfulnessLevel", values="count")
        .fillna(0)
        .astype(int)
    )
    helpfulness_counts.columns = [
        helpfulness_counts.lower() for helpfulness_counts in helpfulness_counts.columns
    ]

    # Merge ratings/helpfulness
    total_ratings = informative_merge(
        total_ratings.reset_index(),
        helpfulness_counts.reset_index(),
        "Sums of ratings for each note",
        "Counts of helpfulness levels for each note",
        on="noteId",
        how="outer",
    ).set_index("noteId")

    # Count number of ratings done of each note
    ratings_counts = ratings["noteId"].value_counts().to_frame(name="ratings_count")

    # Merge ratings counts
    total_ratings = informative_merge(
        total_ratings.reset_index(),
        ratings_counts.reset_index(),
        "Aggregated ratings (both rating sums and helpfulness counts)",
        "Count of the total number of ratings",
        on="noteId",
    ).set_index("noteId")

    # Switch to snake case
    total_ratings = total_ratings.rename(columns=camel_to_snake)

    # Get the percent versions
    pct_ratings = (
        total_ratings.div(total_ratings["ratings_count"], axis=0) * 100
    ).drop(columns="ratings_count")

    # Rename columns to clarify source
    total_ratings = total_ratings.rename(columns=lambda x: f"total_{x}_ratings")
    total_ratings = total_ratings.rename(
        columns={"total_ratings_count_ratings": "total_number_of_ratings"}
    )
    pct_ratings = pct_ratings.rename(columns=lambda x: f"percent_rated_{x}")

    # Merge percent and non-percent versions
    total_ratings = total_ratings.merge(pct_ratings, left_index=True, right_index=True)

    # Merge with note-level data
    all_note_data = informative_merge(
        total_ratings.reset_index().rename(columns={"noteId": "note_id"}),
        notes,
        "Aggregated ratings of note",
        "Note ratings of tweet",
        on="note_id",
        how="outer",
    )

    # Merge back to amount of time spent CRH
    all_note_data = informative_merge(
        all_note_data,
        df_tweets_notes.drop(columns="tweet_id"),
        "Aggregated ratings of note and note ratings of tweet",
        "Amount of time that a note spent CRH in the 48h post slap",
        on="note_id",
    )

    # Define a lambda function to compute the weighted mean:
    weighted_mean = lambda x: np.average(
        x, weights=all_note_data.loc[x.index, "approx_num_hours_crh_in_48h_post_slap"]
    )

    # Average across all CRH notes for a tweet
    ratings_data = (
        all_note_data.drop(
            columns=[
                "note_id",
                "ever_crh_in_48h_post_slap",
                "approx_num_hours_crh_in_48h_post_slap",
                "note_full_text",
            ]
        )
        .groupby("tweet_id")
        .aggregate(weighted_mean)
        .reset_index()
    )

    note_full_text = all_note_data[
        ["tweet_id", "note_full_text", "approx_num_hours_crh_in_48h_post_slap"]
    ].copy()
    note_full_text = note_full_text[
        note_full_text["approx_num_hours_crh_in_48h_post_slap"] > 0
    ].reset_index(drop=True)
    note_full_text["note_full_text"] = (
        note_full_text["approx_num_hours_crh_in_48h_post_slap"].astype(str)
        + " hours: "
        + note_full_text["note_full_text"].fillna("")
    )
    note_full_text = note_full_text.sort_values(
        "approx_num_hours_crh_in_48h_post_slap", ascending=False
    )
    note_full_text = (
        note_full_text.groupby("tweet_id")[["note_full_text"]]
        .agg(lambda x: "\n\n-------------------\n ".join(x))
        .reset_index()
    )

    # Merge back to ratings data
    ratings_data = ratings_data.merge(note_full_text, on="tweet_id", how="left")

    # Save to csv
    ratings_data.to_csv(intermediate_dir / "pre_slap_ratings.csv", index=False)