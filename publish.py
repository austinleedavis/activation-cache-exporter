"""
This script pushes parquet files to the Hugging Face Hub. 
The script loads the dataset from the specified directory and pushes it to the Hugging Face Hub using the provided configuration.
"""
import argparse
import os

import huggingface_hub

DEBUG = False
DEBUG_ARGS = 'data/activations/unsorted test_activations_push'.split()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""Activation Cache Publisher""")
    parser.add_argument("output_dir", type=str, help="Directory where processed files are saved.")
    parser.add_argument("repo_id", type=str, help="""The ID of the 🤗 Hf repository to push to in the following format: `<user>/<dataset_name>` or `<org>/<dataset_name>`. Also accepts `<dataset_name>`, which will default to the namespace of the logged-in user.""")
    
    args = parser.parse_args(DEBUG_ARGS if DEBUG else None)

    ds_args_file = os.path.join(args.output_dir, "args.txt")
    
    report_url = huggingface_hub.create_repo(repo_id=args.repo_id, 
                                repo_type="dataset",
                                exist_ok=True)
    print(f"{report_url=}")

    commit_info = huggingface_hub.upload_file(path_or_fileobj=ds_args_file, 
                                path_in_repo="args.txt",
                                repo_id=args.repo_id,)
    print(f"{commit_info=}")

    data_commit_info =huggingface_hub.upload_folder(repo_id=args.repo_id, 
                                  repo_type='dataset', 
                                  folder_path=args.output_dir,
                                  path_in_repo="data",
                                  allow_patterns="*.parquet.gz")
    
    print(f"{data_commit_info=}")