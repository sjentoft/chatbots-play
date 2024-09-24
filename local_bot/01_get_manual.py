# Download dapla manual and save to S3
import os
from git import Repo #GitPython
import shutil
import time
import stat
from datetime import date

from config import dato, path_to_manual

def download_html_files(repo_url, branch, dato, folder_path):
    """
    Function to download files from a github repo on statisticsnorway to a bucket.
    """

    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        remove_temp(folder_path)
    os.makedirs(folder_path)
    time.sleep(2) # Wait to ensure folder is made before cloning into it
    print(f"Folder created at: {folder_path}")

    # Clone the repository into a temporary directory
    try:
        Repo.clone_from(repo_url, folder_path, branch=branch, single_branch=True)
        print("Repo cloned")
    except GitCommandError as e:
        print(f"Error: {e}")

    print("All done!")

def remove_temp(folder_path):
    """
    Remove a temporary folder
    """
    try:
        shutil.rmtree(folder_path, onerror=handle_remove_readonly)
    except Exception as e:
        print(f"Error removing directory: {e}")

def handle_remove_readonly(func, path, exc):
    """
    Change the file to be writeable
    """
    os.chmod(path, stat.S_IWRITE)
    func(path)


if __name__ == "__main__":
    # Source repo
    repo_url = 'https://github.com/statisticsnorway/dapla-manual.git'
    branch = 'gh-pages'

    # Destination 
    dato = date.today()
    target_folder = f"{path_to_manual}/{dato}"

    # Run
    download_html_files(repo_url, branch, dato, target_folder)
