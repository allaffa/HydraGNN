# HPC installation

Use exactly one installation entry point for the target machine:

```bash
# Frontier (default ROCm 7.1; also supports 6.4, 7.2, and 7.13)
bash scripts/hpc/olcf/frontier/installation/install.sh

# Aurora
bash scripts/hpc/alcf/aurora/installation/install.sh

# Perlmutter
bash scripts/hpc/nersc/perlmutter/installation/install.sh
```

Each entry point creates one environment under `INSTALL_ROOT`, installs the
machine accelerator and compiled dependencies into that environment, installs
HydraGNN, and only then installs model-specific dependencies. `fairchem-core`
is included so FAIR-Chem UMA classes can be imported. Downloading gated UMA
checkpoints also requires approval for `facebook/UMA` on Hugging Face and an
authenticated `huggingface-cli` session or `HF_TOKEN`.

The environments remain dependent on facility-provided drivers, compilers, and
MPI libraries. All installable Python packages are collected behind the single
environment activation command printed by each script. Aurora additionally
inherits Intel's XPU-enabled PyTorch from the facility `frameworks` module.

Useful overrides:

```bash
INSTALL_ROOT=/path/to/install VENV_PATH=/path/to/venv bash <installer>
RECREATE_ENV=1 bash <installer>
FRONTIER_ROCM_VERSION=7.13 bash scripts/hpc/olcf/frontier/installation/install.sh
FAIRCHEM_CORE_VERSION=2.22.0 bash <installer>
```

The version-named Frontier scripts are internal profiles selected by
`frontier/installation/install.sh`; they are not separate user workflows.

## Other facility assets

Facility-specific environment, monitoring, and launch assets are organized as:

```text
scripts/hpc/<facility>/<system>/
```

For example, `olcf/frontier/environments` contains interactive Frontier setup
scripts, while `olcf/frontier/omnistat` contains the corresponding Omnistat
collector configurations and `olcf/frontier/installation` contains installation
scripts. Facility-wide helpers, such as `olcf/proxy-env.sh`, live at the facility
level. Keep portable HydraGNN utilities outside this hierarchy.
