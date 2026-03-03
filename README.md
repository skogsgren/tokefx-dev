# Tokenization Effects in Multilingual LLMs

## Quick Start

First download `ud-data`, assuming `$DATA_DIR` stands for where you want to store the data:

```{bash}
curl -L -o ud-treebanks.zip \
  "https://lindat.mff.cuni.cz/repository/server/api/core/items/b4fcb1e0-f4b2-4939-80f5-baeafda9e5c0/allzip?handleId=11234/1-6036"
mkdir -p "$DATA_DIR"
unzip ud-treebanks.zip -d "$DATA_DIR"
rm ud-treebanks.zip
```

Then you can edit the `/configs/debug_config.toml` and edit the directories to what you want. The `ud_base` directory would be `$DATA_DIR` in the stated commands, i.e.\ the folder with folders like:

```
data/
└── UD_PUD
    ├── UD_Arabic-PUD
    │   ├── ar_pud-ud-test.conllu
    │   ├── ar_pud-ud-test.txt
    │   ├── LICENSE.txt
    │   ├── README.md
    │   └── stats.xml
    ├── UD_Chinese-PUD
    │   ├── LICENSE.txt
    │   ├── README.md
    │   ├── stats.xml
    │   ├── zh_pud-ud-test.conllu
    │   └── zh_pud-ud-test.txt
    ...
```

Clone the repo:

```{bash}
git clone https://github.com/skogsgren/tokefx-dev
cd tokefx-dev
```

Then set up the Python environment, e.g.\

```{bash}
python3 -m venv /path/to/venv
source /path/to/venv/bin/activate
pip3 install -e .
```

To run experiments see the [`/scripts`](/scripts) folder. For example, to run attention analyses using the debug configuration (which runs fine on low-powered non-CUDA machines, using Qwen 0.6B):

```{bash}
python3 scripts/attention.py configs/debug_config.toml
```

Which would create plots and data files under `./out/debug`.

### ALVIS/SLURM Notes

In the [`/slurm`](/slurm) folder are SLURM scripts to run on your favorite SLURM cluster. They were made with [ALVIS](https://www.c3se.chalmers.se/about/Alvis/) in mind, so details for your local SLURM may vary. For ALVIS the setup is as you suspect, e.g.:

```{bash}
module purge
module load "PyTorch/2.7.1-foss-2024a-CUDA-12.6.0"
python3 -m venv /path/to/venv
source /path/to/venv/bin/activate
pip3 install -e .
sbatch slurm/attention_toksuite.sh
```

## Compiling Documents

There's a [`Makefile`](docs/Makefile) included to compile $\LaTeX{}$ files
automatically using `tectonic`. For example, to compile the project proposal
with slides:

```
make -C docs proposal
```

Or the entire thesis:

```
make -C docs thesis
```
