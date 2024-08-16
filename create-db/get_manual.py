# Download dapla manual and save to S3

import os
from git import Repo
import shutil
import time
import stat
from datetime import date
import s3fs


def download_html_files(repo_url, branch, target_folder, 
                        bucket ="sjentoft",
                        file_type = ".html"):
    """
    Function to download files from a github repo on statisticsnorway to a bucket.
    """
    # Make a temporary folder
    temp_path = f'{os.getcwd()}/temp'
    if os.path.exists(temp_path) and os.path.isdir(temp_path):
        remove_temp(temp_path)
    os.makedirs(temp_path)
    time.sleep(2) # Wait to ensure folder is made before cloning into it

    # Clone the repository into a temporary directory
    Repo.clone_from(repo_url, temp_path, branch=branch, single_branch=True)
    print("Repo cloned")

    # Set up fs connection
    S3_ENDPOINT_URL = "https://" + os.environ["AWS_S3_ENDPOINT"]
    fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': S3_ENDPOINT_URL})

    # Create folder on gs as target
    dato = date.today()
    dest_directory = f"{bucket}/{target_folder}"
    fs.touch(dest_directory)
    print(f'Folder created for files: {dest_directory}')

    # Copy files to destination
    time.sleep(5)  # Wait for 10 seconds to ensure downloaded
    for filename in os.listdir(f'{temp_path}/statistikkere'):
        if filename == ".git":
            continue
        if filename.endswith(file_type):
            file_path = f'{temp_path}/statistikkere/{filename}'
            file_dest = f'{dest_directory}/{filename}'
            fs.put(file_path, file_dest)          
    print("Html files copied")

    # Delay before cleanup to allow for any file locks to be released
    print("Cleaning up...")
    time.sleep(5)  # Wait for 5 seconds

    # Clean up the temporary clone
    remove_temp(temp_path)
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
    target_folder = f"db-files/dapla-manual/{dato}"

    # Run
    download_html_files(repo_url, branch, target_folder)
