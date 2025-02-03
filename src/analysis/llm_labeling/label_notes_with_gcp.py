import time
import os
from typing import List, Dict, Tuple
from tqdm import tqdm
import pandas as pd
from src.analysis.llm_labeling.model_interface import create_model_interface

_ACTIVE_MODEL = "claude-3-5-sonnet-v2@20241022"
_SCHEMA_PATH = "src/analysis/llm_labeling/note_annotation_schema.json"
_NOTE_SET = "all_english_notes"
_NOTE_PATH = f"src/analysis/llm_labeling/{_NOTE_SET}.csv"
_INSTRUCTIONS_PATH = "src/analysis/llm_labeling/note_annotation_instructions.txt"

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
    """Generate the output path for the CSV file.

    Args:
        model_name (str): The name of the model.

    Returns:
        str: The output path for the CSV file.
    """
    return f"src/analysis/llm_labeling/{_NOTE_SET}_{model_name}_responses.csv"

def load_existing_responses(output_path: str) -> Dict[Tuple[str, str], int]:
    """Load existing responses and count completions per (tweet_id, note) pair.

    Args:
        output_path (str): The path to the output CSV file.

    Returns:
        Dict[Tuple[str, str], int]: A dictionary mapping (tweet_id, note_text) to the number of existing responses.
    """
    try:
        existing_df = pd.read_csv(output_path, dtype=str)
        return existing_df.groupby(["tweet_id", "note_full_text"]).size().to_dict()
    except FileNotFoundError:
        return {}

def save_response(response_data: Dict[str, str], output_path: str) -> None:
    """Save a single response to the CSV file.

    Args:
        response_data (Dict[str, str]): The response data to save.
        output_path (str): The path to the output CSV file.
    """
    response_df = pd.DataFrame([response_data])
    mode = "a" if os.path.exists(output_path) else "w"
    response_df.to_csv(output_path, mode=mode, header=not os.path.exists(output_path), index=False)

def process_notes(notes_df: pd.DataFrame, model_interface: "AIModelInterface", schema_path: str) -> List[Dict[str, str]]:
    """Process notes through the selected model interface.

    Args:
        notes_df (pd.DataFrame): DataFrame containing the notes to process.
        model_interface (AIModelInterface): Initialized model interface.
        schema_path (str): Path to the JSON schema file.

    Returns:
        List[Dict[str, str]]: List of processed responses.
    """
    with open(_INSTRUCTIONS_PATH, "r") as file:
        instructions = file.read()

    output_path = get_output_path(_ACTIVE_MODEL)
    existing_responses = load_existing_responses(output_path)

    notes_df["tweet_and_note"] = "Tweet:\n" + notes_df["tweet_full_text"] + "\n\nNote:\n" + notes_df["note_full_text"]

    notes_to_process = []
    num_responses_needed = []

    for _, row in notes_df.iterrows():
        key = (row["tweet_id"], row["note_full_text"])
        existing_count = existing_responses.get(key, 0)
        if existing_count < 3:
            notes_to_process.append(row)
            num_responses_needed.append(3 - existing_count)

    notes_to_process = pd.DataFrame(notes_to_process)

    if notes_to_process.empty:
        print("All notes have been processed 3 times. Nothing to do.")
        return []

    for (_, row), needed_responses in tqdm(zip(notes_to_process.iterrows(), num_responses_needed), total=len(notes_to_process)):
        tweet_note_text = row["tweet_and_note"]
        tweet_id = row["tweet_id"]
        tweet = row["tweet_full_text"]
        note = row["note_full_text"]

        completed = 0
        failed = 0
        while completed < needed_responses and failed < needed_responses * 2:
            prompt = f"{instructions}\n\n{tweet_note_text}"
            try:
                response = model_interface.generate_response(prompt=prompt, schema_path=schema_path, max_tokens=2048, temperature=1.0)
                response_data = response["content"]
                if isinstance(response_data, dict):
                    response_data.update({"tweet_id": tweet_id, "note_full_text": note, "tweet_full_text": tweet})
                    save_response(response_data, output_path)
                    completed += 1
                else:
                    failed += 1
            except Exception as e:
                failed += 1
                print(f"Error processing tweet {tweet_id}: {str(e)}")
                time.sleep(1)

def main() -> None:
    """Main function to initialize the model and process notes."""
    model_config = CONFIG[_ACTIVE_MODEL]
    model = create_model_interface(_ACTIVE_MODEL, **model_config)
    model.initialize()

    notes = pd.read_csv(_NOTE_PATH, dtype=str)
    process_notes(notes, model, _SCHEMA_PATH)

    output_path = get_output_path(_ACTIVE_MODEL)
    responses_df = pd.read_csv(output_path, dtype=str)
    responses_df = responses_df.sort_values(["tweet_id", "note_full_text"]).reset_index(drop=True)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()