<!-- <p align="center">

  <h3 align="center">Dense Enzyme Retrieval (DEER)</h3>

  <p align="center">
    Supporting code for the paper
  </p>
</p> -->
<p align="center">
  <h2 align="center">Dense Enzyme Retrieval (DEER)</h2>
  <p align="center">
    <!-- Official PyTorch implementation for finding human-bacteria isozymes using learned dense vector representations.
    <br /> -->
    Supporting code for the paper: "Exploring Functional Insights into the Human Gut Microbiome via the Structural Proteome" (Liu et al., 2025, Manuscript under revision)
    <br />
    <!-- <a href="#about-this-repository"><strong>Explore the docs »</strong></a>
    <br /> -->
    <br />
    <a href="https://github.com/WangJiuming/deer/issues">Report Bug</a>
    ·
    <a href="https://github.com/WangJiuming/deer/issues">Request Feature</a>
  </p>
</p>

## Table of Contents

* [About this repository](#about-this-repository)
* [Installation](#installation)
* [Getting started](#getting-started)
  * [Download resources](#download-resources)
  * [Running the example](#running-the-retrieval-example)
* [Citation](#citation)
* [References](#references)

## About this repository

DEER (Dense Enzyme Retrieval) provides a method for finding functionally related human-bacteria isozymes using learned dense vector representations (embeddings). This repository contains the code, pre-trained models, and example data necessary to reproduce the results and apply DEER to new enzyme sequences, as presented in our paper.

## Installation

First, clone the repository:
```bash
git clone https://github.com/WangJiuming/deer.git
cd deer
```

We recommend using Conda for managing dependencies. Choose one of the following options based on your hardware:

**Option 1. GPU with Flash Attention (Recommended)**

If the hardware supports Flash Attention (see the <a href="https://github.com/Dao-AILab/flash-attention">official Flash Attention repository</a> [1] for compatibility), this option offers significant speed-ups.

```bash
conda env create --name deer --file env/env_gpu_fa.yml
```

**Option 2. Standard GPU**

If your GPU is not compatible with Flash Attention, use this standard GPU installation.

```bash
conda env create --name deer --file env/env_gpu.yml
```

**Option 3. CPU Only**

If no GPU is available, you can install the CPU-only version. Note that this will be significantly slower than GPU versions.
```bash
conda env create --name deer --file env/env_cpu.yml
```

After installation using any of the above options, activate the Conda environment:
```bash
conda activate deer
```

## Getting started

Follow these steps to download the necessary resources and run the example enzyme retrieval task.

### Download resources

The pre-trained model checkpoints are available on servers. To download them to perform inference, run the following.
```bash
wget https://huggingface.co/jmwang9/deer/resolve/main/ckpt.zip
```
Alternatively, in case the above link is unavailable, the checkpoint can also be downloaded manually using <a href="https://drive.google.com/file/d/1C8drHpS4-9ONblpR_lUi5iijcJeL0irZ/view?usp=drive_link">this link</a>.


Then decompress the file.
```bash
unzip -d ckpt.zip
```
The `./ckpt/` directory should now contain:
*   `saprot_35m/`: Files required for the underlying SaProt protein language model [2].
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

**Note**: Based on the hardware, you may need to adjust settings in `config_test.yaml`, such as GPU allocation (`devices`), batch size (`batch_size`), or worker number for the data loader (`num_workers`). 

**Important**: If you installed the CPU-only version (Option 3), ensure the `devices` parameter in `config_test.yaml` is set to `cpu`.

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

[1] Dao, Tri, et al. "Flashattention: Fast and memory-efficient exact attention with io-awareness." Advances in neural information processing systems 35 (2022): 16344-16359.

[2] Su, Jin, et al. "Saprot: Protein language modeling with structure-aware vocabulary." bioRxiv (2023): 2023-10.


