import pandas as pd
from sklearn.metrics import accuracy_score


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
        test_data = {
            "newsgroups": pd.read_csv(
                assignment_test_data_dir / "newsgroups_test_labels.csv"
            ),
            "sst2": pd.read_csv(assignment_test_data_dir / "sst2_test_labels.csv"),
        }
        return test_data
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Test data file not found: {e.filename}")
    except Exception as e:
        print(f"An error occurred while loading test data: {e}")
        raise


def compute_scores(file_name, pred, repo, test_data):
    """Compute scores for a given file and repository."""
    try:
        method, dataset, *_ = file_name.split("_")
        true = test_data[dataset]
        comment = ""
    except:
        return

    try:
        if dataset == "newsgroups":
            accuracy = accuracy_score(true["newsgroup"], pred["newsgroup"]).round(5)
        elif dataset == "sst2":
            accuracy = accuracy_score(true["label"], pred["label"]).round(5)
    except:
        accuracy = None
        comment = "Error computing accuracy!"

    return {
        "leaderboard": "leaderboard_" + dataset,
        "Score": accuracy,
        "Method": method,
        "Member": " ".join(repo["member"]),
        "Comment": comment,
    }


def sort_scores(leaderboards):
    """Sort the leaderboard by Score, Member, and Method."""
    return leaderboards.sort_values(["Score", "Member", "Method"], ascending=False)
