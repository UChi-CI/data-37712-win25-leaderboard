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
    # model_name = file_name.split("_test_wer_predictions.csv")[0]

    # if model_name not in ["character_n_gram", "subword_n_gram", "transformer"]:
    #     print(f"Model name {model_name} not recognized")
    #     return

    # comment = ""

    # if pred is None:
    #     wer_score = None
    #     comment = "Error reading CSV!"

    # else:
    #     try:
    #         pred = pred.set_index("id")
    #         pred.columns = ["sentences"]

    #         wer_score = wer.compute(
    #             predictions=pred["sentences"].tolist(),
    #             references=test_data["sentences"].tolist(),
    #         )
    #         wer_score = round(wer_score, 5)

    #     except:
    #         wer_score = None
    #         comment = "Error computing correlation!"

    # return [
    #     {
    #         # Required: name of leaderboard file.
    #         "leaderboard": "leaderboard_hub",
    #         "Score": wer_score,
    #         "Method": model_name,
    #         "Member": member,
    #         "Comment": comment,
    #     }
    #     for member in repo["member"]
    # ]

    # Extract model name from file name
    valid_models = {"character_n_gram", "subword_n_gram", "transformer"}
    model_name = next((m for m in valid_models if file_name.startswith(m)), None)

    # Default values
    wer_score = None
    comment = ""

    if model_name is None:
        comment = f"Model name in file {file_name} not recognized"
    elif pred is None:
        comment = "Error reading CSV!"
    else:
        try:
            # Ensure "id" column exists and identify the second column dynamically
            if "id" not in pred.columns:
                comment = "Error: Missing required 'id' column in predictions DataFrame"
            else:
                content_columns = [col for col in pred.columns if col != "id"]

                if len(content_columns) != 1:
                    comment = "Error: Predictions DataFrame should have exactly one non-'id' column"
                else:
                    content_column = content_columns[0]
                    wer_score = wer.compute(
                        predictions=pred.set_index("id")[content_column].tolist(),
                        references=test_data["sentences"].tolist(),
                    )
                    wer_score = round(wer_score, 5)
        except Exception as e:
            comment = f"Error computing WER score: {e}"

    # Return structured leaderboard results
    return [
        {
            "leaderboard": "leaderboard_hub",
            "Score": wer_score,
            "Method": model_name,
            "Member": member,
            "Comment": comment,
        }
        for member in repo.get("member", [])
    ]


def sort_scores(leaderboards):
    """Sort the leaderboard by Score, Member, and Method."""
    return leaderboards.sort_values(["Score", "Member", "Method"], ascending=False)
