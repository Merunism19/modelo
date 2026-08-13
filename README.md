# modelo — single-file edition

A one-file CLI that manages and serves local Hugging Face models. Everything
lives in this folder; there is no package to install.

## Install globally (one command)

**macOS / Linux** (installs `modelo` into `~/.local/bin` and its Python deps):

```bash
curl -fsSL https://raw.githubusercontent.com/Merunism19/modelo/main/install.sh | sh
```

**Windows** (PowerShell — installs `modelo.cmd` into `%USERPROFILE%\.local\bin`):

```powershell
irm https://raw.githubusercontent.com/Merunism19/modelo/main/install.ps1 | iex
```

The installer downloads `modelo.py`, creates a global `modelo` command, and
installs `typer`, `huggingface_hub` and `requests`. Add the install folder to
your `PATH` (the installer prints the exact line), then run `modelo --help`.

> The installer is hosted at `https://github.com/Merunism19/modelo`. The
> `BASE_URL` used to fetch `modelo.py` lives in the installer scripts themselves,
> so you can fork it, set a different `MODELO_URL` env var, or point it at a
> Gist by changing that one line.

## Files & folders

```
modelo.py        the entire program (run this, or the installed modelo command)
install.sh       macOS / Linux one-line installer
install.ps1      Windows one-line installer
.modelo/         auto-created on first run; holds all data
  ├── config.json        settings (models_dir, host, port, default_model, ...)
  ├── models/            the model store (one subfolder per repo)
  └── harness-config/    reference copies of generated harness configs
```

Set the **MODELO_HOME** environment variable to keep that data folder somewhere
else (e.g. `set MODELO_HOME=D:\my-models-home` before running).

## Requirements

Python 3.10+. Core commands need only a few pip packages:

```bash
pip install typer huggingface_hub requests
```

Serving additionally needs llama.cpp + uvicorn:

```bash
pip install llama-cpp-python uvicorn
```

## Run it

```bash
python modelo.py --help
python modelo.py store --help
python modelo.py version
```

Typical flow:

```bash
python modelo.py store download Qwen/Qwen2.5-1.5B-Instruct-GGUF --gguf --quant Q4_K_M
python modelo.py store list
python modelo.py store scan Qwen2.5-1.5B-Instruct-GGUF
python modelo.py store serve --ctx 8192
python modelo.py store setup --all     # wire up Kilo Code / pi / Reasonix
python modelo.py store status
```

Use `--dir <folder>` on any store command to use a different model store
(overrides the config default for that run).