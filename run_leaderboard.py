import argparse
import base64
import importlib
import os
from io import StringIO
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv
from github import Github, GithubException
from tqdm import tqdm


def get_assignment_utils(utils_module):
    """Factory function to import scoring and data utilities based on the config file.

    Inputs:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: A dictionary containing `compute_scores`, `sort_scores`, and `load_test_data`.
    """
    module = importlib.import_module(utils_module)

    compute_scores = getattr(module, "compute_scores")
    sort_scores = getattr(module, "sort_scores")
    load_test_data = getattr(module, "load_test_data")

    return {
        "compute_scores": compute_scores,
        "sort_scores": sort_scores,
        "load_test_data": load_test_data,
    }


def main(config_path="config.yaml"):
    # Load environment variables
    load_dotenv()
    GITHUB_USERNAME = os.getenv("GITHUB_USERNAME")
    GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

    # Load YAML configuration
    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)

    # Extract configuration values
    DRY_RUN = config["dry_run"]
    CLASS = config["github"]["organization"]
    LEADERBOARD_REPO_NAME = config["github"]["leaderboard_repo"]
    REPO_ASSIGNMENT_PREFIX = config["github"]["assignment_prefix"]
    LEADERBOARD_ASSIGNMENT_NAME = config["github"]["assignment_name"]
    STAFF = set(config["staff"])
    TEST_DATA_DIR = Path(config["test_data"]["directory"])
    ASSIGNMENT_TEST_DATA_DIR = (
        TEST_DATA_DIR / config["test_data"]["assignment_test_data"]
    )
    RESULTS_FILES = config["results_files"]
    UTILS_MODULE = config["utils_module"]

    # Load utilities based on config
    utils = get_assignment_utils(UTILS_MODULE)

    compute_scores = utils["compute_scores"]
    sort_scores = utils["sort_scores"]
    load_test_data = utils["load_test_data"]

    # Validate GitHub credentials
    if not GITHUB_USERNAME or not GITHUB_TOKEN:
        raise ValueError(
            "GitHub username and token must be provided via .env or YAML config."
        )

    print(f"Using GitHub username: {GITHUB_USERNAME}")
    print(f"DRY_RUN mode is {'enabled' if DRY_RUN else 'disabled'}")

    git = Github(GITHUB_USERNAME, GITHUB_TOKEN)
    org = git.get_organization(CLASS)
    leaderboard_repo = org.get_repo(LEADERBOARD_REPO_NAME)

    print("Loading test data...")
    test_data = load_test_data(ASSIGNMENT_TEST_DATA_DIR)

    print("Loading Repos...")
    repos = [
        {
            "git": repo,
            "name": repo.name,
            "member": sorted(
                [c.login for c in repo.get_collaborators() if c.login not in STAFF]
            ),
        }
        for repo in org.get_repos()
        if repo.name.startswith(REPO_ASSIGNMENT_PREFIX)
    ]

    repos = [
        repo for repo in repos if not any(staff in repo["name"] for staff in STAFF)
    ]

    # Extract results
    for repo in tqdm(repos, desc="Finding files"):
        try:
            res_files = repo["git"].get_contents("results")
        except Exception:
            print(f"Issue: results folder not found for {repo['name']}")
            continue

        repo["files"] = {
            result_file.name: result_file
            for result_file in res_files
            if result_file.name in RESULTS_FILES
        }

    # Download and compute scores
    leaderboards = []
    for repo in tqdm(repos, desc="Downloading files"):
        repo["results"] = {}
        if "files" not in repo:
            continue

        for file_name, path in repo["files"].items():
            content_encoded = repo["git"].get_git_blob(path.sha).content
            content = base64.b64decode(content_encoded).decode("utf-8")
            data = pd.read_csv(StringIO(content))
            repo["results"][file_name] = data

            score = compute_scores(file_name, data, repo, test_data)
            if score:
                leaderboards.append(score)

    leaderboards = sort_scores(pd.DataFrame(leaderboards))

    print("Updating leaderboards...")
    for name, board in leaderboards.groupby("leaderboard"):
        del board["leaderboard"]
        csv_content = board.to_csv(index=False)
        csv_name = name + ".csv"

        if DRY_RUN:
            Path("public").mkdir(exist_ok=True)
            with open(f"public/{csv_name}", "w") as f:
                f.write(csv_content)
        else:
            try:
                leaderboard_file = leaderboard_repo.get_contents(
                    LEADERBOARD_ASSIGNMENT_NAME + "/" + csv_name
                )
                leaderboard_repo.update_file(
                    leaderboard_file.path,
                    "Leaderboard Update",
                    csv_content,
                    leaderboard_file.sha,
                )
                print(f"Updated existing file: {csv_name}")

            except GithubException as e:
                # Check if the exception is a 404 (file not found)
                if e.status == 404:
                    leaderboard_repo.create_file(
                        LEADERBOARD_ASSIGNMENT_NAME + "/" + csv_name,
                        "Create leaderboard",
                        csv_content,
                    )
                    print(f"Created new file: {csv_name}")
                else:
                    raise

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run leaderboard updater.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args()
    main(args.config)
