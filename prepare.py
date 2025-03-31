"""
This script writes activation cache for a given dataset to disk. 

WARNING: An activation caches can be HUGE on disk. So, tune `batches_per_shard`, `batch_size`, and `max_shards_created` to limit disk usage.
A hidden state with dimension 768 and 12 layers averages 34.375kB per token. So, you can get ~1.5k tokens in a 50MB partition

Sorting by length with `--sort_ds_by_len` makes forward passes more efficient by packing same-sized instances into a single mini-batch. But
when verifying a minibatch fits in memory, use `--sort_ds_reversed` to ensure the longest inputs are processed first.


Example Usage:
```python
    python prepare.py --output_dir data/activations/sorted \
        --batches_per_shard 10  \
        --batch_size 2  \
        --max_shards_created 1 \
        --model_checkpoint austindavis/chessGPT2  \
        --ds_config 202302-00000-00009 \
        --ds_repo austindavis/lichess-uci-scored \
        --ds_split train  \
        --ds_input_column Transcript \
        --ds_label_columns Site WhiteElo BlackElo Transcript Scores \
        --n_pos 1024  \
        --log_file log.txt
```

The resulting dataset can be loaded using:
```python
    datasets.load_dataset("parquet", data_files="data/activations/sorted/*.parquet")
```
"""
import argparse
import logging
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor

import datasets
import numpy as np
import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

DEBUG = False
DEBUG_ARGS = '--output_dir data/activations/unsorted --sort_ds_by_len --auto_find_batch_size --batches_per_shard 5 --batch_size 2  --max_shards_created 1 --model_checkpoint austindavis/chessGPT2 --ds_config 202302-00000-00009 --ds_repo austindavis/lichess-uci-scored --ds_split train --ds_input_column Transcript --ds_label_columns Site WhiteElo BlackElo Transcript Scores --n_pos 1024 --log_file log.txt'.split()

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", required=True, type=str, help="Directory where processed files are saved.")
    parser.add_argument("--ds_repo", required=True, type=str, help="Hf 🤗 dataset repository name (e.g., 'user/repo')") 
    parser.add_argument("--ds_config", required=True, type=str, help="Hf 🤗 dataset config name (e.g., '202301')") 
    parser.add_argument("--ds_split", required=True, type=str, help="Hf 🤗 dataset split name (e.g. 'train')") 
    parser.add_argument("--model_checkpoint", required=True, type=str, help="local or Hf 🤗 model used to generate hidden state vectors")

    parser.add_argument("--ds_input_column", type=str, help="Dataset column name that will be tokenized. (Default='Transcript')", default="Transcript")
    parser.add_argument("--ds_label_columns", nargs="+", type=str, help="Dataset column name(s) that will label each output record. (Default=['Site', 'Transcript'])", default=["Site", "Transcript"])

    parser.add_argument("--max_shards_created", type=int, help="Activation caches are massive (~6 GB/shard). This value limits the total number of activation shards created to prevent out of disk errors.", default=2)
    parser.add_argument("--batches_per_shard", type=int, help="Number of minibatches saved to each shard", default=500)
    parser.add_argument("--batch_size", type=int, help="Batch size for each forward pass through llm. (Default: 8)", default=8)
    parser.add_argument("--auto_find_batch_size", action='store_true', help="Automatically deduce the max memory-safe batch size.")
    
    parser.add_argument("--sort_ds_by_len", action='store_true', help="Sort the dataset input column by number of characters. Enabling increases efficiency of the forward pass")
    parser.add_argument("--sort_ds_reversed", action='store_true', help="Sort the dataset input column in reverse (longest first). Especially helpful when checking batch size.")
    parser.add_argument("--ds_shuffle_seed", type=int, help="Seed to shuffle the dataset. If none provided, the original dataset order is used.")
    parser.add_argument("--n_pos", type=int, help="Number of positions to sample. Sampling is done from the final n_pos positions.", default=-1)
    parser.add_argument("--device", type=str, help="Device for LLM.", default='cuda')
    
    parser.add_argument("--log_level", type=str, help="Log level. (Default: INFO)", default="INFO")
    parser.add_argument("--log_file", type=str, help="Specify a log filename to send logging to disk. Otherwise, prints log info to stdout.")

    args = parser.parse_args(DEBUG_ARGS if DEBUG else None)    
    return args

def map_to_length(row, input_column_name, len_col_name):
    return {len_col_name: len(row[input_column_name])}

def init_dataset_shard(ds_label_columns: list[str]):
    main_dataset = Dataset.from_dict({"TokenPosition": [], "HiddenStates": []})
    for label in ds_label_columns:
        assert label in ds.column_names, f"'{label}' not is a column in the source dataset. Check the `ds_label_columns` CLI argument."
        main_dataset = main_dataset.add_column(label, [])
    return main_dataset

def get_output_file(output_dir: str, file_counter: int, total_shards: int):
    format_string = f"0{len(str(total_shards))}" # leading zeros based on max number of shards
    return os.path.join(output_dir, f"activations_shard_{file_counter:{format_string}}.parquet")

def process_single_game(game_id, hidden_states, token_count, n_pos_limit, batch, label_columns):
    """Helper function to process a single game's hidden states and create a record."""
    expanded_records = []

    # Slice out hidden states for a single transcript
    transcript_hidden_states = hidden_states[:, game_id]  # Final shape: (n_layers, n_positions, d_hidden)
    
    min_index = max(0, token_count[game_id] - n_pos_limit)
    max_index = min(min_index + n_pos_limit, token_count[game_id])

    # Shape: (n_layers, n_positions, d_hidden)
    record = {
        "HiddenStates": transcript_hidden_states[:, min_index:max_index].numpy()  
    }

    # Add additional labels from dataset columns
    for label in label_columns:
        record[label] = batch[label][game_id]

    expanded_records.append(record)

    return expanded_records

def find_largest_batch_size_binary_search_synthetic(llm, device, max_length, max_batch_size=1024):
    low, high = 1, max_batch_size
    best = 1
    pbar = tqdm(total=0, bar_format="{desc}", leave=True)
    while low <= high:
        mid = (low + high) // 2
        pbar.set_description_str(f"Trying batch_size={mid} (range: {low}-{high})")
        try:
            input_ids = torch.ones((mid, max_length), dtype=torch.long, device=device)
            attention_mask = torch.ones((mid, max_length), dtype=torch.long, device=device)
            with torch.no_grad():
                _ = llm(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            best = mid
            low = mid + 1
        except RuntimeError as e:
            if 'CUDA out of memory' in str(e):
                torch.cuda.empty_cache()
                high = mid - 1
            else:
                pbar.close()
                raise e
    pbar.set_description_str(f"Final batch_size={best}")
    pbar.close()
    return best

def remove_prefixes(strings: list[str], indices: list[int]) -> list[int]:
    result = []

    for i, s in enumerate(strings[:-1]):
        # if any(other.startswith(s) for other in strings[i+1:]):
        if str(strings[i+1]).startswith(str(s)):
            continue
        result.append(indices[i])

    return result

if __name__ == "__main__":

    args = parse_args()

    # exist is NOT ok because it may cause BIG mistakes
    os.makedirs(args.output_dir, exist_ok=False)

    # Write arguments to the output directory
    with open(os.path.join(args.output_dir,"args.txt"), "x") as arg_log:
        arg_log.writelines([f"{k}: {v}\n" for k,v in args.__dict__.items()])

    ##############
    # Setup Logging
    ##############
    if args.log_file:
        log_file_path = os.path.join(args.output_dir, args.log_file)

    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {args.log_level}')
    logging.basicConfig(level=numeric_level, filename=log_file_path)

    ##############
    # Load Dataset
    ##############
    ds = load_dataset(path=args.ds_repo, name=args.ds_config, split=args.ds_split)

    # deduplicate and remove repeated prefixes
    values, unique_indices = np.unique(ds[args.ds_input_column], return_index=True, axis=0)
    deprefixed_indices = remove_prefixes(values, unique_indices.tolist())
    ds = ds.select(deprefixed_indices)

    # dataset metrics
    num_batches = torch.math.ceil((len(ds)/args.batch_size))
    total_shards = min(args.max_shards_created, torch.math.ceil(num_batches / args.batches_per_shard))
    total_batches = total_shards * args.batches_per_shard
    total_records = total_batches * args.batch_size

    if args.ds_shuffle_seed:
        ds = ds.shuffle(args.ds_shuffle_seed).select(range(total_records))
    elif args.sort_ds_by_len:
        # Hint: This sorts by number of characters, not num tokens!
        len_col_name = f"{args.ds_input_column}_Length"
        num_procs = multiprocessing.cpu_count()
        half_procs = max(1,num_procs // 2)
        ds = ds.map(map_to_length, num_proc=half_procs, fn_kwargs=dict(input_column_name=args.ds_input_column, len_col_name=len_col_name), desc="Compute Input Column Len")
        ds = ds.sort(column_names=len_col_name, reverse=args.sort_ds_reversed)

    ##############
    # Load Model and Tokenizer
    ##############
    DEVICE = torch.device(args.device)

    llm = (
        AutoModelForCausalLM.from_pretrained(args.model_checkpoint)
            .requires_grad_(False) # huge memory savings here
            .to(device=DEVICE) # big speed improvement
        )

    # Setup tokenizer here because dataset mapping ruins forces parallelism=False
    os.environ['TOKENIZERS_PARALLELISM'] = "true"
    tok = AutoTokenizer.from_pretrained(args.model_checkpoint)
    logging.info(f"Loaded tokenizer:\n{tok}")

    n_layers = llm.config.n_layer
    n_embd = llm.config.n_embd
    max_length = llm.config.n_positions

    assert args.n_pos <= max_length and args.n_pos > 0, f"n_pos (={args.n_pos}) must be in: 0 < n_pos <= {max_length}"

    logging.info(f"Loaded LLM to Device: '{DEVICE}'\n{llm}\n{llm.config}")

    ##############
    # Find safe batch size
    ##############
    if args.auto_find_batch_size:
        logging.info(f"Finding max safe batch size: {args.batch_size}")
        args.batch_size = find_largest_batch_size_binary_search_synthetic(llm, DEVICE, max_length)
        logging.info(f"Auto-selected batch size: {args.batch_size}")

    ##############
    # Main loop
    ##############
    datasets.disable_progress_bar()

    # Initialize an empty Huggingface dataset
    main_dataset = init_dataset_shard(args.ds_label_columns)

    logging.info(f"{total_batches=} {total_shards=}")
    shard_progress_bar = tqdm(range(total_shards), total=total_shards, desc="Exporting Shards")
    batches_progress_bar = tqdm(range(total_batches), total=total_batches, desc="Processing Batches")

    file_counter = 0
    with ProcessPoolExecutor(max_workers=args.batch_size) as executor:    

        for batch_idx, record_idx in enumerate(range(0, len(ds), args.batch_size)):
            
            batches_progress_bar.update()

            if (batch_idx + 1) % args.batches_per_shard == 0:
                # Save the current main_dataset to disk
                file_path = get_output_file(args.output_dir, file_counter, total_shards)
                main_dataset.to_parquet(file_path)
                # print(f"Saved dataset chunk to {file_path}")

                # Reset main_dataset and increment file counter
                main_dataset = init_dataset_shard(args.ds_label_columns)
                file_counter += 1
                shard_progress_bar.update()
                if file_counter >= args.max_shards_created:
                    logging.info(f"Processing finished due to max shards (={args.max_shards_created}) created. Exiting.")
                    break

            batch = ds[record_idx: record_idx + args.batch_size]
            text = batch[args.ds_input_column]

            tokenized_games = tok(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
                return_token_type_ids=False,
            )

            # Compute number of positions in each game transcript
            token_count = tokenized_games['attention_mask'].sum(dim=-1).tolist()

            # Batch forward pass for computationally efficiency
            with torch.no_grad():
                outputs = llm(**tokenized_games.to(DEVICE), output_hidden_states=True)
                hidden_states = outputs.hidden_states  # List of hidden states from each layer

                # Stack hidden states into a single tensor of shape (n_layers, batch_size, n_positions, d_hidden)
                # Skip the first hidden element which is actually just the embeddings
                # hidden_states.shape = (n_layers, batch_size, n_positions, d_hidden)
                hidden_states = torch.stack(hidden_states[1:]).to(torch.device('cpu'))  

            ####
            # Create a record for each game (can be parallelized)
            ####
            futures = [
                executor.submit(process_single_game, index, hidden_states, token_count, args.n_pos, batch, args.ds_label_columns)
                for index in range(args.batch_size)
            ]

            # Collect results for the current batch and create a Huggingface Dataset
            batch_records = []
            for future in futures:
                batch_records.extend(future.result())

            batch_dataset = Dataset.from_dict({
                "HiddenStates": [record["HiddenStates"] for record in batch_records]
            })

            # Add other columns for the labels
            for label in args.ds_label_columns:
                batch_dataset = batch_dataset.add_column(label, [record[label] for record in batch_records])

            # Concatenate the batch dataset to the main dataset
            main_dataset = concatenate_datasets([main_dataset, batch_dataset])
        
        # Save remaining entries (if any) to disk
        if len(main_dataset) > 0:
            # This code is only reachable if the max shard count exceeds len(dataset) / batches_per_shard
            file_path = get_output_file(args.output_dir, file_counter, total_shards)
            main_dataset.to_parquet(file_path)