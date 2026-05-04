# Detokenization Effects in Multilingual LLMs

## Quick Start
Clone the repo:

```
git clone https://github.com/skogsgren/tokefx
cd tokefx
```

Then set up the Python environment, e.g.\

```
python3 -m venv venv
source venv/bin/activate
pip3 install -e .
```

Then download `ud-data` (exact commands might change slightly when versions change):

```
curl -L -o ud-treebanks.zip \
  "https://lindat.mff.cuni.cz/repository/server/api/core/items/b4fcb1e0-f4b2-4939-80f5-baeafda9e5c0/allzip?handleId=11234/1-6036"
unzip ud-treebanks.zip
tar xvfz  ud-treebanks-v2.17.tgz
mkdir -p ./data
mv ud-treebanks-v2.17 data/UD_PUD
```

(and then cleanup of the archives if desired):

```
rm ud-*.tgz
rm ud-treebanks.zip
```

Then you either directly edit the config files directly to point `ud_base` to the correct path, or set the environment variable `$MIMER_DIR` to the `realpath` of the parent of the cloned `tokefx-dev`, and then making sure there's a subfolder called `data/UD_PUD` containing all the language specific folders. In other words `./tokefx-dev/data` should contain:

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

There's a runscript at [`/scripts/run`](/scripts/run) which acts as a pipeline wrapper. To run the `toksuite` experiments you would run it like so:

```
./scripts/run configs/toksuite_config.toml
```

**NOTE:** the `toksuite` experiment run expects access to the [Aya model](https://huggingface.co/CohereLabs/aya-expanse-8b), which requires terms and conditions to be accepted and `HFTOKEN` to be set.

## Individual Experiments

To run individual experiments see the [`/scripts`](/scripts) folder. For example, to run attention analyses using the debug configuration:

```{bash}
python3 scripts/attention.py configs/debug_config.toml
```

Which would create plots and data files under `./out/debug`.

You can inspect the raw `parquet` file which contains every individual instance using the included `./debug_parquet` script. For example, if using `configs/debug_config.toml`:

```{bash}
./debug_parquet out/debug/full_attn.parquet
```

It contains some nifty optional flags like `--no_layers` if you want to use it to debug token boundaries. See `./debug_parquet -h` for all the options.

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
