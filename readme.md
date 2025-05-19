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

### 1. Download resources

#### 1.1 Model checkpoints
The pre-trained model checkpoints are available on servers. To download them to perform inference, run the following.
```bash
wget https://huggingface.co/cuhkaih/deer/resolve/main/ckpt.zip
```
Alternatively, in case the above link is unavailable, the checkpoint can also be downloaded manually using <a href="https://drive.google.com/file/d/1C8drHpS4-9ONblpR_lUi5iijcJeL0irZ/view?usp=drive_link">this link</a>.


Then decompress the file.
```bash
unzip ckpt.zip
```
The `./ckpt/` directory should now contain:
*   `saprot_35m/`: Files required for the underlying SaProt protein language model [2].
*   `esm2_t12_35M_UR50D/`: Files required for the underlying ESM2 language model [3].
*   `deer_checkpoint.ckpt`: The pre-trained DEER model checkpoint.

#### 1.2 Dataset

Additionally, we provide an working example dataset to demonstrate the retrieval process. This dataset contains 5,849 enzyme structures and was used for benchmarking in our paper. To download the dataset, run the following.
```bash
wget https://huggingface.co/datasets/cuhkaih/deer/resolve/main/data.zip
```

Then decompress the file.
```bash
unzip data.zip
```
The `data/` directory should now contain:
*   `example/template_pdb/`: 1,636 eukaryota templates' PDB files.
*   `example/database_pdb/`: 4,213 bacteria enzymes' PDB files.

### 2. Running the retrieval example

To perform retrieval using a group of template structures against a database using the default options:
```bash
python do_retrieval.py --template_pdb_dir ./data/template_pdb/ \
                       --database_pdb_dir ./data/database_pdb/
```

More options can be set according to the `--help` argument.
```bash
python do_retrieval.py --help
```

Note that if Flash Attention is installed, then the `--use_fa` flag argument can be set to accelerate the process. By default, the model will use all available GPUs whenever GPU is detected in the system, to overide this behavior and use a specific device or opt for CPU, users may set the environment variable `CUDA_VISIBLE_DEVICES` when running the script.

- For using two specific GPU devices:
```bash
CUDA_VISIBLE_DEVICES="0,1" python do_retrieval.py ...
```

- For doing CPU-only inference:
```bash
CUDA_VISIBLE_DEVICES="" python do_retrieval.py ...
```

Results are saved to `./results/similarity.csv` by default, containing a Pandas DataFrame with the columns:
* `eukaryota_id`: Identifier for the template enzyme.
* `bacteria_id`: Identifier for the bacteria enzyme.
* `distance`: Euclidean distance between embeddings. Lower distance indicates higher similarity.

Note that if multiple templates are used, the retrieval results for all templates will be saved and sorted together in one file. Users may separate them during further processing.

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

[3] Lin, Zeming, et al. "Language models of protein sequences at the scale of evolution enable accurate structure prediction." BioRxiv 2022 (2022): 500902.

