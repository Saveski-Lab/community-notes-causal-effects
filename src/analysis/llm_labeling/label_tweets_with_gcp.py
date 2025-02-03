import os
import time
from typing import Dict, List
from tqdm import tqdm
import pandas as pd
from model_interface import create_model_interface

# Constants
_ACTIVE_MODEL = "claude-3-5-sonnet-v2@20241022"
_SCHEMA_PATH = "src/analysis/llm_labeling/tweet_annotation_schema.json"
_TWEET_SET = "all_english_tweets"
_TWEET_PATH = f"src/analysis/llm_labeling/{_TWEET_SET}.csv"
_INSTRUCTIONS_PATH = "src/analysis/llm_labeling/tweet_annotation_instructions.txt"

# Configuration
CONFIG = {
    "gemini-flash-001": {"model_name": "gemini-1.5-flash-001"},
    "gemini-flash-002": {"model_name": "gemini-1.5-flash-002"},
    "gemini-pro-001": {"model_name": "gemini-1.5-pro-001"},
    "gemini-pro-002": {"model_name": "gemini-1.5-pro-002"},
    "claude-3-5-sonnet-v2@20241022": {
        "model_name": "claude-3-5-sonnet-v2@20241022",
        "region": "us-east5",
        "project_id": os.getenv("ANTHROPIC_VERTEX_PROJECT_ID"),
    },
}

def get_output_path(model_name: str) -> str:
    """Generate the output path for the responses CSV file.

    Args:
        model_name (str): The name of the model.

    Returns:
        str: The output path for the responses CSV file.
    """
    return f"src/analysis/llm_labeling/{_TWEET_SET}_{model_name}_responses.csv"

def load_existing_responses(output_path: str) -> Dict[str, int]:
    """Load existing responses from the CSV file.

    Args:
        output_path (str): The path to the output CSV file.

    Returns:
        Dict[str, int]: A dictionary mapping tweet IDs to the number of responses.
    """
    try:
        existing_df = pd.read_csv(output_path, dtype=str)
        return existing_df.groupby("tweet_id").size().to_dict()
    except FileNotFoundError:
        return {}

def save_response(response_data: Dict, output_path: str):
    """Save a response to the CSV file.

    Args:
        response_data (Dict): The response data to save.
        output_path (str): The path to the output CSV file.
    """
    response_df = pd.DataFrame([response_data])
    mode, header = ("a", False) if os.path.exists(output_path) else ("w", True)
    response_df.to_csv(output_path, mode=mode, header=header, index=False)

def process_tweets(tweets_df: pd.DataFrame, model_interface: "AIModelInterface", schema_path: str) -> List[Dict]:
    """Process tweets using the specified model interface and schema.

    Args:
        tweets_df (pd.DataFrame): DataFrame containing tweets to process.
        model_interface (AIModelInterface): The model interface to use for processing.
        schema_path (str): The path to the schema file.

    Returns:
        List[Dict]: A list of processed responses.
    """
    with open(_INSTRUCTIONS_PATH, "r") as file:
        instructions = file.read()

    output_path = get_output_path(_ACTIVE_MODEL)
    existing_responses = load_existing_responses(output_path)

    tweets_to_process = [
        row for _, row in tweets_df.iterrows() if existing_responses.get(row["tweet_id"], 0) < 3
    ]
    num_responses_needed = [
        3 - existing_responses.get(row["tweet_id"], 0) for row in tweets_to_process
    ]

    if not tweets_to_process:
        print("All tweets have been processed 3 times. Nothing to do.")
        return []

    for (_, row), needed_responses in tqdm(
        zip(tweets_to_process, num_responses_needed),
        desc=f"Processing tweets with {model_interface.__class__.__name__}",
        total=len(tweets_to_process),
    ):
        tweet, tweet_id = row["tweet_full_text"], row["tweet_id"]
        completed, failed = 0, 0

        while completed < needed_responses and failed < needed_responses * 2:
            prompt = f"{instructions}\n\nTweet:{tweet}"
            try:
                response = model_interface.generate_response(
                    prompt=prompt, schema_path=schema_path, max_tokens=2048, temperature=1.0
                )
                response_data = response["content"]
                if isinstance(response_data, dict):
                    response_data["tweet_id"] = tweet_id
                    save_response(response_data, output_path)
                    completed += 1
                else:
                    failed += 1
            except Exception as e:
                failed += 1
                print(f"Error processing tweet {tweet_id}: {str(e)}")
                time.sleep(1)

def main():
    """Main function to initialize the model and process tweets."""
    model_config = CONFIG[_ACTIVE_MODEL]
    model = create_model_interface(_ACTIVE_MODEL, **model_config)
    model.initialize()

    tweets = pd.read_csv(_TWEET_PATH, dtype=str)
    process_tweets(tweets, model, _SCHEMA_PATH)

    output_path = get_output_path(_ACTIVE_MODEL)
    responses_df = pd.read_csv(output_path, dtype=str)
    responses_df.sort_values("tweet_id").reset_index(drop=True)

if __name__ == "__main__":
    main()