'''
Scheduler requirements:
1. Can call multiple instances at the same time and won't cause overlaps
2. Can stop calling instances, call a new instance, and start working at the correct place
3. Covers all instance
'''

'''
Proposed Solution:
Make each instance to be run a separate file, and have its naming reflect its status: (raw, started, finished).
So for example:
  - raw_"instance_naming".csv
  - started_"instance_naming".csv
  - finished_"instance_naming".csv
The scheduler will then:
1. Check for all raw_*.csv files and put them in a list
2. Before processing begins, the scheduler checks if the raw_*.csv file still exists. If it does, it will rename it to started_*.csv
3. If a thread fails for any reason, started_*.csv file will be renamed back to raw_*.csv
3. If a thread is finished, it will rename the started_*.csv file to finished_*.csv
4. The scheduler will keep running until all raw_*.csv files are finished
'''

'''
Potential Problems:
1. Currently this represents 7 million files, we could put 100 rows into one file instead and cut down dramatically on the number of files:
    1.a) 7 million goes to 70,000 with 100 rows
2. We can also cut the number of files by eliminating scenarios we don't find interesting, let's cut it down by 2 orders of magnitude (100x)
'''

import os
import shutil
import pandas as pd
from git import Repo
from multiprocessing import Pool, cpu_count
from main import process_row
import datetime

# Define folders
BASE_DIR = os.getcwd()
NOT_STARTED_FOLDER = os.path.join(BASE_DIR, "not_started")
STARTED_FOLDER = os.path.join(BASE_DIR, "started")
FINISHED_FOLDER = os.path.join(BASE_DIR, "finished")
REPO_DIR = BASE_DIR  # Assumes this script is run from the root of the GitHub repo

# Ensure required folders exist
for folder in [NOT_STARTED_FOLDER, STARTED_FOLDER, FINISHED_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Initialize the Git repo
repo = Repo(REPO_DIR)
if repo.bare:
    raise Exception("Not a valid Git repository")

def update_git(files):
    """Commit and push changes to Git"""
    for src, dst in files:
        repo.index.add([dst])
        repo.index.remove([src])
    repo.index.commit(f"Moved {len(files)} files in batch.")
    origin = repo.remote(name="origin")
    origin.pull()
    origin.push()

def move_files_in_batch(files):
    """Move a batch of files from their source to their destination."""
    for src, dst in files:
        shutil.move(src, dst)

    update_git(files)



def process_file(file_name):
    """Process a single file: move, process, and finalize."""
    file_path = os.path.join(STARTED_FOLDER, file_name)

    if not file_name.endswith(".csv"):
        print(f"Skipped {file_name}: Not a CSV.")
        return

    # Read and process the file
    df = pd.read_csv(file_path)

    # Processing function where we pass a row from the df and update our axis
    # so in this case we will need to return the relative result
    # we're going to pull this in from main
    print("Now working on processing a file!")
    df[["underdog_mean", "underdog_variance"]] = df.apply(process_row, axis=1, result_type='expand')

    # Save the processed file
    finished_path = os.path.join(FINISHED_FOLDER, file_name)
    df.to_csv(finished_path, index=False)
    print(f"Processed {file_name} and saved to 'finished'.")
    os.remove(file_path)
    return file_path, finished_path


if __name__ == "__main__":
    while True:
        # Pull the latest changes from the remote repository
        origin = repo.remote(name="origin")
        origin.pull()

        # Get the list of files in the 'not_started' folder (up to 1000 files)
        files = os.listdir(NOT_STARTED_FOLDER)[:min(100, len(os.listdir(NOT_STARTED_FOLDER)))]

        if not files:
            print("No more files to process in 'not_started'. Exiting.")
            break

        # Move all selected files to 'started' first in a batch
        batch_moves = []
        for file_name in files:
            src = os.path.join(NOT_STARTED_FOLDER, file_name)
            dst = os.path.join(STARTED_FOLDER, file_name)
            batch_moves.append((src, dst))

        move_files_in_batch(batch_moves)

        # Use multiprocessing Pool to process files concurrently
        num_cores = cpu_count()-2
        now = datetime.datetime.now()
        print(now.time())
        with Pool(num_cores) as pool:
            files = pool.map(process_file, files)

        print("Trying to update git after getting through batch of 100 files.")
        update_git(files)