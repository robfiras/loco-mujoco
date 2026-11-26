"""
Download and removal of the MyoSkeleton model.

The MyoSkeleton model of MyoLab Inc. is distributed under its own license and is therefore not shipped with
LocoMuJoCo. The functions in this module back the command-line tools ``loco-mujoco-myomodel-init`` and
``loco-mujoco-myomodel-clear``.
"""

import os
import shutil
import git

from loco_mujoco import PATH_TO_CUSTOM_MODELS


def fetch_git(repo_url, commit_hash, clone_directory, clone_path):
    """
    Clone a git repository and check out a specific commit.

    If the repository has already been cloned to the target directory, it is fetched instead of cloned again. Git
    errors are printed and not raised.

    Args:
        repo_url (str): URL of the repository.
        commit_hash (str): Hash of the commit to check out.
        clone_directory (str): Name of the directory the repository is cloned into.
        clone_path (str): Path in which ``clone_directory`` is created.

    Returns:
        str: Path to the cloned repository, ``clone_path/clone_directory``.

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
    Remove the downloaded MyoSkeleton model, if it exists.

    The model is deleted from the directory ``myo_model`` inside ``loco_mujoco.PATH_TO_CUSTOM_MODELS``. This is the
    entry point of the command-line tool ``loco-mujoco-myomodel-clear``.

    """
    print("LocoMujoco:> Clearing MyoSkeleton ...")
    api_path = os.path.join(PATH_TO_CUSTOM_MODELS, 'myo_model')
    if os.path.exists(api_path):
        shutil.rmtree(api_path)
    else:
        print("LocoMujoco:> MyoSkeleton directory does not exist.")
    print("LocoMujoco:> MyoSkeleton cleared")


def accept_license():
    """
    Ask the user on the command line to accept the license of the MyoSkeleton model.

    The question is repeated until the answer is ``yes`` or ``no``.

    Returns:
        bool: True if the user accepted the license, False if the user rejected it.

    """
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
    Download the MyoSkeleton model after the user accepted its license.

    The repository https://github.com/myolab/myo_model is cloned at a fixed commit into the directory ``myo_model``
    inside ``loco_mujoco.PATH_TO_CUSTOM_MODELS``. Nothing is downloaded if the license is rejected. This is the entry
    point of the command-line tool ``loco-mujoco-myomodel-init``.

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
              clone_path=PATH_TO_CUSTOM_MODELS)

    print("LocoMuJoCo:> Successfully initialized MyoSkeleton .")
