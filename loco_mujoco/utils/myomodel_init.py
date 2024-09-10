import os
import shutil
from os.path import expanduser
import git
from pathlib import Path

import loco_mujoco


data_dir = Path(loco_mujoco.__file__).resolve().parent / "environments" / "data"


def fetch_git(repo_url, commit_hash, clone_directory, clone_path):
    """
    fetch git repo using provided details
    """

    clone_directory = os.path.join(clone_path, clone_directory)

    try:
        # Create the clone directory if it doesn't exist
        os.makedirs(clone_directory, exist_ok=True)

        # Clone the repository to the specified path
        if not os.path.exists(os.path.join(clone_directory, '.git')):
            repo = git.Repo.clone_from(repo_url, clone_directory)
            print(f"{repo_url} cloned at {clone_directory}")
        else:
            repo = git.Repo(clone_directory)
            origin = repo.remote('origin')
            origin.fetch()

        # Check out the specific commit if not already
        current_commit_hash = repo.head.commit.hexsha
        if current_commit_hash != commit_hash:
            repo.git.checkout(commit_hash)
            print(f"{repo_url}@{commit_hash} fetched at {clone_directory}")
    except git.GitCommandError as e:
        print(f"Error: {e}")

    return clone_directory


def clear_myoskeleton():
    """
    Remove cached MyoSkeleton if it exists.
    """
    print("LocoMujoco:> Clearing MyoSkeleton ...")
    api_path = os.path.join(data_dir, 'myo_model')
    if os.path.exists(api_path):
        shutil.rmtree(api_path)
    else:
        print("LocoMujoco:> MyoSkeleton directory does not exist.")
    print("LocoMujoco:> MyoSkeleton cleared")


def accept_license():
    prompt = """
    A permissive license for non-commercial scientific research of the MyoSkeleton by the MyoLab Inc. is available.
    You can review the license at: https://github.com/myolab/myo_model/blob/main/LICENSE
    Do you accept the terms of the license? (yes/no):
    """
    response = input(prompt).strip().lower()

    if response == 'yes':
        print("Thank you for accepting the license. You may proceed.")
        return True
    elif response == 'no':
        print("You have rejected the license terms. Exiting...")
        return False
    else:
        print("Invalid input. Please enter 'yes' or 'no'.")
        return accept_license()  # Recursively prompt again for valid input


def fetch_myoskeleton():
    """
    Fetch a copy of MyoSkeleton.
    """
    print("\n\nLocoMuJoCo:> Initializing MyoSkeleton...")

    # Inform user about API
    if accept_license():
        # Proceed with the rest of the code
        print("LocoMuJoCo:> MyoSkeleton License accepted. Proceeding initialization ...")
    else:
        # Exit or handle the rejection case
        print("LocoMuJoCo:> MyoSkeleton License rejected. Exiting")
        return

    # Fetch
    print("LocoMuJoCo:> Downloading simulation assets (upto ~100MBs)")
    fetch_git(repo_url="https://github.com/myolab/myo_model.git",
              commit_hash="619b1a876113e91a302b9baeaad6c2341e12ac81",
              clone_directory="myo_model",
              clone_path=data_dir)

    print("LocoMuJoCo:> Successfully initialized MyoSkeleton .")

