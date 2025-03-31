# Activation Cache Exporter

This repository provides a CLI tool for exporting transformer hidden states (activations) from HuggingFace language models for a specified HuggingFace 🤗 dataset. It is primarily intended for mechanistic interpretability research and enables efficient serialization of hidden states to disk in a sharded Parquet format.

## Features

- Stream hidden states to disk in memory-efficient shards
- Configure batch size, number of batches per shard, and maximum number of shards
- Sort input data by length to improve forward pass throughput
- Filter, sample, or shuffle dataset inputs before activation extraction
- Compatible with any causal language model checkpoint on HuggingFace

## Requirements

Install the required packages with:

```bash
pip install -r requirements.txt
```

Note: The tokenizer is loaded from a custom fork:
```text
git+https://github.com/austinleedavis/tokenizers.git#subdirectory=bindings/python
```

## Usage

```bash
python prepare.py \
    --output_dir data/activations/sorted \
    --batches_per_shard 50 \
    --auto_find_batch_size \
    --max_shards_created 900 \
    --model_checkpoint austindavis/chessGPT2 \
    --ds_config 202302-00000-00009 \
    --ds_repo austindavis/lichess-uci-scored \
    --ds_split train \
    --ds_input_column Transcript \
    --ds_label_columns Site WhiteElo BlackElo Scores \
    --n_pos 1024 \
    --log_file log.txt
```

## Output

The script produces a set of `.parquet` files under the specified `--output_dir`. Each file contains a fixed number of activation batches, along with optional labels from the dataset. The output can be loaded using:

```python
from datasets import load_dataset
ds = load_dataset("parquet", data_files="data/activations/sorted/*.parquet")
```

## Notes

- Activation caches can be very large. Tune `--batches_per_shard`, `--batch_size`, and `--max_shards_created` accordingly.
- Sorting by input length (`--sort_ds_by_len`) improves performance by reducing the number of PAD tokens added to a batch, but may affect sampling distributions. If used, the resulting dataset should be shuffled before being used for (say) training.
- Reversing sort order (`--sort_ds_reversed`) ensures memory safety by placing longest samples first, allowing the user to check for out of memory errors.

## License

MIT
