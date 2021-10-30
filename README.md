# freqgym
Combining freqtrade with OpenAI Reinforcement Learning environments

## Installation with Conda

freqgym can be installed with Miniconda or Anaconda.

### What is Conda?

Conda is a package, dependency and environment manager for multiple programming languages: [conda docs](https://docs.conda.io/projects/conda/en/latest/index.html)

### Installation with conda

Prepare freqgym environment, using file `environment_cpuonly.yml` or `environment_cuda.yml`, which exist in main freqgym directory

```bash
conda env create -n freqgym -f environment_cpuonly.yml
```

#### Enter/exit freqtrade-conda environment

To check available environments, type

```bash
conda env list
```

Enter installed environment

```bash
# enter conda environment
conda activate freqgym

# exit conda environment
conda deactivate
```

Download data with:
`
freqtrade download-data --timerange=20201201-20211015 --timeframe 5m --exchange=binance --erase --pairs BTC/USDT
`

TODO

- Multiple pair training
- Vectorized envs
- Normalization
