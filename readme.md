<p align="center">

  <h3 align="center">Dense Enzyme Retrieval (DEER)</h3>

  <p align="center">
    Supporting code for the paper
  </p>
</p>

## Table of Contents

* [Installation](#installation)
* [Getting Started](#getting-started)
  * [Download Resources](#download-resources)
  * [Running the Example](#running-the-retrieval-example)
* [Citation](#citation)
* [References](#references)

## About this repository

DEER (Dense Enzyme Retrieval) provides a method for finding functionally related human-bacteria isozymes using learned dense vector representations (embeddings). This repository contains the code, pre-trained models, and example data necessary to reproduce the results and apply DEER to new enzyme sequences, as presented in our paper.

## Installation

We recommend using Conda for managing dependencies.

First, create the Conda environment with one of the following two installation options.

**Option 1. standard installation**
```bash
conda env create -f env.yml
```
**Option 2. with flash attention**

Alternatively, if the machine supports flash attention, run the following to install the environment with flash attention. This is recommended for compatible GPUs for a significant speed-up.
```bash
conda env create -f env_fa.yml
```

Then activate the environment.
```bash
conda activate deer
```

## Getting started

Follow these steps to download the necessary resources and run the example enzyme retrieval task.

### Download resources

The pre-trained model checkpoints are available on servers. To download them to perform inference, run the following.
```bash
https://proj.cse.cuhk.edu.hk/aihlab/gmps/api/download?filename=ckpt.pt -O ckpt.pt
```
Alternatively, in case the above link is unavailable, the checkpoint can also be downloaded manually using <a href="https://drive.google.com/file/d/1C8drHpS4-9ONblpR_lUi5iijcJeL0irZ/view?usp=drive_link">this link</a>.


Then decompress the file.
```bash
unzip -d ckpt.zip
```
The `./ckpt/` directory should now contain:
*   `saprot_35m/`: Files required for the underlying SaProt protein language model [1].
*   `deer_checkpoint.ckpt`: The pre-trained DEER model checkpoint.

### Running the retrieval example

We provide an example dataset in the `data/` directory to demonstrate the retrieval process. This dataset contains 5,849 enzyme sequences (1,636 Eukaryota templates and 4,213 Bacteria enzymes) and was used for benchmarking in our paper. The tokenized dataset for the model are also present in `data/`.

There are two steps to perform retrieval on the working example dataset.

**Step 1. generate embeddings**

Use the `test.py` script to compute embeddings for all sequences in the example dataset.
```bash
python test.py --config ./config/config_test.yaml
```

This script will process the sequences defined in the configuration file.
Embeddings will be saved to a `./results/reprs.pkl` file (or other places specified within `config_test.yaml`).

**Note**: You may need to adjust settings in `config_test.yaml`, such as GPU allocation (`devices`) or batch size (`batch_size`), based on your hardware.

**Step 2. perform dense retrieval**

Once embeddings are generated, use `do_retrieval.py` to calculate pairwise similarities.
```bash
python do_retrieval.py
```
This script loads the generated embeddings and calculates the Euclidean distance between template (Eukaryota) and query (Bacteria) embeddings.

Results are saved to `/results/retrieval_results.pkl` by default, containing a Pandas DataFrame with the columns:
* `eukaryota_id`: Identifier for the template enzyme.
* `bacteria_id`: Identifier for the bacteria enzyme.
* `distance`: Euclidean distance between embeddings. Lower distance indicates higher similarity.

This will output a `.pkl` file in the `./reuslt/` folder storing a dataframe for the pairwise similarities between each template enzyme and each bacteria enzyme. There are three columns in this dataframe: `eukaryota_id`, `bacteria_id`, and `distance`, which is the Euclidean distance between embeddings (hence smaller distance indicates higher degree of similarity).


## Citation
If you use DEER or this codebase in your research, please cite our paper:
```
@misc{liu2025Exploring,
  author={Liu, H. and Shen, J. and others},
  title={Exploring Functional Insights into the Human Gut Microbiome via the Structural Proteome},
  year={2025},
  note={Manuscript under revision}
}
```

## References

[1] Su, Jin, et al. "Saprot: Protein language modeling with structure-aware vocabulary." bioRxiv (2023): 2023-10.
