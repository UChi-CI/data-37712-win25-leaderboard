import pandas as pd
from evaluate import load

wer = load("wer")


def load_test_data(assignment_test_data_dir):
    """Loads test data for the given assignment.

    Inputs:
        assignment_test_data_dir (Path): Path to the assignment's test data directory.

    Returns:
        dict: A dictionary containing test data for each dataset.

    Raises:
        FileNotFoundError: If a test data file is missing.
        Exception: For other unexpected errors during loading.
    """
    try:
        test_data = pd.read_csv(
            assignment_test_data_dir / "test_ground_truths.csv", index_col="id"
        )
        return test_data
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Test data file not found: {e.filename}")
    except Exception as e:
        print(f"An error occurred while loading test data: {e}")
        raise


def compute_scores(file_name, pred, repo, test_data):
    model_name = file_name.split("_test_wer_predictions.csv")[0]

    if model_name not in ["character_n_gram", "subword_n_gram", "transformer"]:
        print(f"Model name {model_name} not recognized")
        return

    comment = ""

    if pred is None:
        wer_score = None
        comment = "Error reading CSV!"

    else:
        try:
            pred = pred.set_index("id")
            pred.columns = ["sentences"]

            wer_score = wer.compute(
                predictions=pred["sentences"].tolist(),
                references=test_data["sentences"].tolist(),
            )
            wer_score = round(wer_score, 5)

        except:
            wer_score = None
            comment = "Error computing correlation!"

    return [
        {
            # Required: name of leaderboard file.
            "leaderboard": "leaderboard_hub",
            "Score": wer_score,
            "Method": model_name,
            "Member": member,
            "Comment": comment,
        }
        for member in repo["member"]
    ]


def sort_scores(leaderboards):
    """Sort the leaderboard by Score, Member, and Method."""
    return leaderboards.sort_values(["Score", "Member", "Method"], ascending=False)
