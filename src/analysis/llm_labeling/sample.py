import pandas as pd
import numpy as np

from pathlib import Path

_TE_PATH = Path(
    "cn_effect_output/treatment_effects/with_root_and_non_root_rts_prod/final_treatment_effects.csv"
)

_TWEET_COLUMNS = [
    "tweet_id",
    "tweet_full_text",
    "tweet_language",
    "note_full_text"
]

_ANALYSIS_DIR = Path("src/analysis/llm_labeling")

if __name__ == "__main__":
    # Read in tweet text for treated tweets
    te = pd.read_csv(
        _TE_PATH,
        dtype=str,
        usecols=_TWEET_COLUMNS,
    )

    # Only sample English tweets
    te = te[te["tweet_language"] == "English"].drop(columns=["tweet_language"])

    te.to_csv(_ANALYSIS_DIR / "all_english_tweets.csv", index=False)

    # Set seed
    np.random.seed(7936)

    # Sample 30 tweets
    r1_sampled_tweets = te.sample(30, replace=False)

    # Write the sampled tweets to csv
    r1_sampled_tweets.drop(columns="note_full_text").to_csv(_ANALYSIS_DIR / "sampled_tweets_round_1.csv", index=False)

    # Split notes into separate rows:
    te["note_full_text"] = te["note_full_text"].str.split("\n\n-------------------\n ")
    te = te.explode(column="note_full_text")
    r1_sampled_tweets["note_full_text"] = r1_sampled_tweets["note_full_text"].str.split("\n\n-------------------\n ")
    r1_sampled_tweets = r1_sampled_tweets.explode(column="note_full_text")

    # Split of the number of hours tag that the notes start with
    te["note_hours_crh"] = te["note_full_text"].str.extract(r"([\d\.]+) hours:").astype(float)
    te["note_full_text"] = te["note_full_text"].str.replace(r"[\d\.]+ hours:", "",regex=True).str.strip()
    r1_sampled_tweets["note_hours_crh"] = r1_sampled_tweets["note_full_text"].str.extract(r"([\d\.]+) hours:").astype(float)
    r1_sampled_tweets["note_full_text"] = r1_sampled_tweets["note_full_text"].str.replace(r"[\d\.]+ hours:", "",regex=True).str.strip()

    # Write the sampled notes to csv
    te = te.sort_values("tweet_id")
    te.to_csv(_ANALYSIS_DIR / "all_english_notes.csv", index=False)
    r1_sampled_tweets = r1_sampled_tweets.sort_values("tweet_id")
    r1_sampled_tweets.to_csv(_ANALYSIS_DIR / "sampled_notes_round_1.csv", index=False)



