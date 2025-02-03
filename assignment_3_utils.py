import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def get_similarity_scores(E1, E2):
    """
    From the assignment repo (to use the same computation)
    Function to compute the similarity scores between two sets of embeddings.
    """
    similarity_scores = []
    for idx in range(len(E1)):
        sim_score = round(np.dot(E1[idx][1], E2[idx][1]), 6)
        similarity_scores.append(sim_score)
    return similarity_scores


def compute_spearman_correlation(similarity_scores, human_scores):
    """
    From the assignment repo (to use the same computation)
    Function to compute the Spearman correlation between the similarity scores and human scores (labels).
    """
    return round(spearmanr(similarity_scores, human_scores).correlation, 6)


def read_embedding(rows):
    embeddings = []
    for i, row in enumerate(rows):
        word, *vector = row.split()
        embeddings.append((word, [float(x) for x in vector]))
        dim = len(vector)
    return embeddings, dim


def enforce_embedding_size(embedding_list, max_allowed_embed_size=1024):
    """Check if all the embeddings have at most max_allowed_embed_size"""
    for embed in embedding_list:
        if len(embed[1]) > max_allowed_embed_size:
            return False
    return True


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
            "cont": pd.read_csv(assignment_test_data_dir / "contextual_test_y.csv"),
            "isol": pd.read_csv(assignment_test_data_dir / "isolated_test_y.csv"),
        }
        return test_data
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Test data file not found: {e.filename}")
    except Exception as e:
        print(f"An error occurred while loading test data: {e}")
        raise


# TODO: Sort out the args...
def compute_scores(results_file_name, pred, repo, test_data):
    # def compute_scores(result_files, repo):
    """

    Inputs:
        results_file_name: The name of the file in the repo
        pred: The predictions in the file
        repo: The repo object
        test_data: The test data for the assignment
    """
    # pred_embeds = {"cont": {}, "isol": {}}
    # comments = {"cont": "", "isol": ""}
    # scores = {"cont": None, "isol": None}
    # dataset = ""

    # for f, v in result_files.items():
    #     try:
    #         breakpoint()
    #         method, dataset, _, word_order, *_ = f.split("_")
    #         if dataset not in ["cont", "isol"]:
    #             return
    #         pred_embeds[dataset][word_order] = v
    #     except:
    #         print(f)
    #         if dataset != "":
    #             comments[dataset] = "Error reading result embeddings!"
    #         else:
    #             comments["cont"] = "Error reading result embeddings!"
    #             comments["isol"] = "Error reading result embeddings!"
    # for _task in ["cont", "isol"]:
    #     if pred_embeds[_task] == {}:
    #         scores[_task] = None
    #         comments[_task] = "Error reading result embeddings!"
    # if not pred_embeds["cont"] == {} and not pred_embeds["isol"] == {}:
    #     for task in ["cont", "isol"]:
    #         # Check: embedding size
    #         try:
    #             if not enforce_embedding_size(
    #                 pred_embeds[task]["words1"]
    #             ) or not enforce_embedding_size(pred_embeds[task]["words2"]):
    #                 comments[task] = "Embedding size exceeds 1024!"
    #                 continue
    #             else:
    #                 try:
    #                     similarity = get_similarity_scores(
    #                         pred_embeds[task]["words1"], pred_embeds[task]["words2"]
    #                     )
    #                     data = test_data[task].copy()
    #                     data.columns = ["actual"]
    #                     data["predicted"] = similarity
    #                     score = round(spearmanr(data).correlation, 6)
    #                     if pd.isnull(score):
    #                         score = None
    #                         comments[task] = (
    #                             "Error computing correlation: the score is nan"
    #                         )
    #                     else:
    #                         score = score.round(5)
    #                     scores[task] = score
    #                 except:
    #                     comments[task] = "Error computing correlation!"
    #         except:
    #             continue
    # # This is a general error catch, to avoid edge cases (e.g. where people used lfs => will get a None score)
    # for _task in ["cont", "isol"]:
    #     if scores[_task] == None:
    #         comments[_task] = "Error computing correlation!"
    # print("scores and comments", scores, comments)
    # return {
    #     # Required: name of leaderboard file.
    #     "leaderboard": "leaderboard_isol",
    #     "Score": scores["isol"],
    #     "Method": method,
    #     "Member": " ".join(repo["member"]),
    #     "Comment": comments["isol"],
    # }, {
    #     "leaderboard": "leaderboard_cont",
    #     "Score": scores["cont"],
    #     "Method": method,
    #     "Member": " ".join(repo["member"]),
    #     "Comment": comments["cont"],
    # }
    pred_embeds = {"cont": {}, "isol": {}}
    comments = {"cont": "", "isol": ""}
    scores = {"cont": None, "isol": None}
    dataset = ""

    try:
        method, dataset, _, word_order, *_ = results_file_name.split("_")
        if dataset not in ["cont", "isol"]:
            return
        pred_embeds[dataset][word_order] = pred
    except:
        print(results_file_name)
        if dataset != "":
            comments[dataset] = "Error reading result embeddings!"
        else:
            comments["cont"] = "Error reading result embeddings!"
            comments["isol"] = "Error reading result embeddings!"
    # Check if the embeddings are read correctly
    for _task in ["cont", "isol"]:
        if pred_embeds[_task] == {}:
            scores[_task] = None
            comments[_task] = "Error reading result embeddings!"
    if not pred_embeds["cont"] == {} and not pred_embeds["isol"] == {}:
        for task in ["cont", "isol"]:
            # Check: embedding size
            try:
                if not enforce_embedding_size(
                    pred_embeds[task]["words1"]
                ) or not enforce_embedding_size(pred_embeds[task]["words2"]):
                    comments[task] = "Embedding size exceeds 1024!"
                    continue
                else:
                    try:
                        similarity = get_similarity_scores(
                            pred_embeds[task]["words1"], pred_embeds[task]["words2"]
                        )
                        breakpoint()
                        data = test_data[task].copy()
                        data.columns = ["actual"]
                        data["predicted"] = similarity
                        score = round(spearmanr(data).correlation, 6)
                        if pd.isnull(score):
                            score = None
                            comments[task] = (
                                "Error computing correlation: the score is nan"
                            )
                        else:
                            score = score.round(5)
                        scores[task] = score
                    except:
                        comments[task] = "Error computing correlation!"
            except:
                continue
    # This is a general error catch, to avoid edge cases (e.g. where people used lfs => will get a None score)
    for _task in ["cont", "isol"]:
        if scores[_task] == None:
            comments[_task] = "Error computing correlation!"
    print("scores and comments", scores, comments)
    return {
        # Required: name of leaderboard file.
        "leaderboard": "leaderboard_isol",
        "Score": scores["isol"],
        "Method": method,
        "Member": " ".join(repo["member"]),
        "Comment": comments["isol"],
    }, {
        "leaderboard": "leaderboard_cont",
        "Score": scores["cont"],
        "Method": method,
        "Member": " ".join(repo["member"]),
        "Comment": comments["cont"],
    }


def sort_scores(leaderboards):
    """Sort the leaderboard by Score, Member, and Method."""
    return leaderboards.sort_values(["Score", "Member", "Method"], ascending=False)
