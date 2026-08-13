#!/usr/bin/env python3
"""modelo — a single-file CLI to manage and serve local Hugging Face models.

Run with:  python modelo.py --help

Everything the tool needs lives inside this folder's ``.modelo/`` directory:

    config.json        settings (models_dir, host, port, default_model, ...)
    models/            the model store (one subfolder per downloaded repo)
    harness-config/    reference copies of generated agent-harness configs

Set the MODELO_HOME environment variable to store that data somewhere else
(e.g. ``set MODELO_HOME=C:\\somewhere\\else`` before running).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import typer

from huggingface_hub import HfApi, snapshot_download
import requests

__version__ = "0.1.0"

APP_DIR_NAME = ".modelo"
DEFAULT_MODELS_DIR_NAME = "models"
OUT_DIR_NAME = "harness-config"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def app_dir() -> Path:
    """Root folder for modelo's data (override with MODELO_HOME)."""
    override = os.environ.get("MODELO_HOME")
    if override:
        return Path(override)
    return Path(__file__).resolve().parent / APP_DIR_NAME


def default_models_dir() -> Path:
    return app_dir() / DEFAULT_MODELS_DIR_NAME


def config_path() -> Path:
    return app_dir() / "config.json"


@dataclass
class Config:
    models_dir: str = str(default_models_dir())
    default_model: str | None = None
    host: str = "127.0.0.1"
    port: int = 8080
    gpu_layers: int = 0
    n_ctx: int = 2048

    @property
    def models_path(self) -> Path:
        return Path(self.models_dir).expanduser()

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"


def load_config(path: Path | None = None) -> Config:
    p = Path(path) if path else config_path()
    cfg = Config()
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                known = {
                    k: v
                    for k, v in data.items()
                    if k in Config.__dataclass_fields__ and v is not None
                }
                cfg = Config(**known)
        except (json.JSONDecodeError, TypeError, ValueError, UnicodeDecodeError):
            # Corrupt config: fall back to defaults, keep the file for inspection.
            pass
    # Normalize and coerce values so a hand-edited file cannot crash commands.
    try:
        cfg.models_dir = str(Path(str(cfg.models_dir)).expanduser())
    except (TypeError, ValueError):
        cfg.models_dir = str(default_models_dir())
    for name in ("port", "gpu_layers", "n_ctx"):
        try:
            setattr(cfg, name, int(getattr(cfg, name)))
        except (TypeError, ValueError):
            setattr(cfg, name, getattr(Config(), name))
    if not isinstance(cfg.default_model, str) or not cfg.default_model:
        cfg.default_model = None
    return cfg


def save_config(cfg: Config, path: Path | None = None) -> None:
    p = Path(path) if path else config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(asdict(cfg), indent=2) + "\n", encoding="utf-8")


def ensure_models_dir(cfg: Config) -> Path:
    """Create the model store if needed and return its path."""
    models_dir = cfg.models_path
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


# ---------------------------------------------------------------------------
# The model store: scanning and validating downloaded models
# ---------------------------------------------------------------------------

WEIGHT_KINDS = ("gguf", "safetensors", "bin")
SKIP_DIR_NAMES = {".cache", ".git", "cache", "__pycache__"}

# Matches common GGUF quantization tags: Q4_K_M, Q4_0, Q6_K, IQ3_XS, F16, FP16, BF16, FP8...
QUANT_RE = re.compile(r"(?i)(iq\d+(?:_[a-z0-9]+)*|q\d+(?:_[a-z0-9]+)*|bf16|fp16|fp8|f16|f32)")


class ModelError(Exception):
    """User-facing error about the model store."""


@dataclass
class ModelFile:
    name: str
    size: int
    kind: str  # "gguf" | "safetensors" | "bin" | "other"

    @property
    def is_weight(self) -> bool:
        return self.kind in WEIGHT_KINDS


def classify_file(name: str) -> str:
    ext = Path(name).suffix.lower()
    if ext == ".gguf":
        return "gguf"
    if ext == ".safetensors":
        return "safetensors"
    if ext in (".bin", ".pth", ".pt", ".onnx", ".ggml"):
        return "bin"
    return "other"


def infer_quant(filename: str) -> str | None:
    m = QUANT_RE.search(filename)
    return m.group(1).upper() if m else None


@dataclass
class ModelEntry:
    name: str
    path: Path
    files: list[ModelFile] = field(default_factory=list)
    incomplete: bool = False

    @property
    def total_size(self) -> int:
        return sum(f.size for f in self.files)

    @property
    def weight_files(self) -> list[ModelFile]:
        return [f for f in self.files if f.is_weight]

    @property
    def gguf_files(self) -> list[ModelFile]:
        return [f for f in self.files if f.kind == "gguf"]

    @property
    def has_weights(self) -> bool:
        return bool(self.weight_files)

    @property
    def healthy(self) -> bool:
        return self.has_weights and not self.incomplete

    @property
    def kind(self) -> str:
        if self.gguf_files:
            return "gguf"
        if any(f.kind == "safetensors" for f in self.files):
            return "safetensors"
        if self.has_weights:
            return "bin"
        return "other"

    @property
    def quants(self) -> list[str]:
        found = {infer_quant(f.name) for f in self.gguf_files}
        return sorted(q for q in found if q)

    @property
    def has_config(self) -> bool:
        return (self.path / "config.json").is_file()


def human_size(num: int) -> str:
    value = float(num)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} TB"


def _iter_files(model_dir: Path) -> list[ModelFile]:
    """Collect files under ``model_dir`` without following directory symlinks.

    Hidden files/dirs and known cache dirs (``.cache``, ``.git``...) are skipped.
    ``*.incomplete`` markers surface partial downloads as unhealthy.
    """
    files: list[ModelFile] = []
    for root, dirs, names in os.walk(model_dir):
        dirs[:] = [
            d
            for d in dirs
            if d not in SKIP_DIR_NAMES and not d.startswith(".")
            and not (Path(root) / d).is_symlink()
        ]
        for name in sorted(names):
            if name.startswith("."):
                continue
            p = Path(root) / name
            if p.is_symlink():
                if not p.is_file():
                    continue
            try:
                size = p.stat().st_size
            except OSError:
                size = 0
            rel = p.relative_to(model_dir).as_posix()
            if name.endswith(".incomplete"):
                files.append(ModelFile(name=rel, size=size, kind="other"))
            else:
                files.append(ModelFile(name=rel, size=size, kind=classify_file(name)))
    return files


def scan_model(name: str, model_dir: Path) -> ModelEntry:
    files = _iter_files(model_dir)
    incomplete = any(f.name.endswith(".incomplete") for f in files) or any(
        f.size == 0 and f.kind == "other" for f in files
    )
    return ModelEntry(name=name, path=model_dir, files=files, incomplete=incomplete)


def list_models(models_dir: Path) -> list[ModelEntry]:
    """Scan the store. Supports flat (``Model``) and namespaced (``org/Model``) layouts."""
    if not models_dir.is_dir():
        return []
    entries: list[ModelEntry] = []
    for p1 in sorted(models_dir.iterdir()):
        if not p1.is_dir() or p1.name in SKIP_DIR_NAMES or p1.name.startswith("."):
            continue
        subdirs = [d for d in p1.iterdir() if d.is_dir()]
        if subdirs and not any(f.is_file() for f in p1.iterdir()):
            for p2 in sorted(subdirs):
                if p2.name in SKIP_DIR_NAMES or p2.name.startswith("."):
                    continue
                entry = scan_model(f"{p1.name}/{p2.name}", p2)
                if entry.files:
                    entries.append(entry)
        else:
            entry = scan_model(p1.name, p1)
            if entry.files:
                entries.append(entry)
    return entries


def find_model(name: str, models_dir: Path) -> ModelEntry:
    """Resolve a user-supplied model name to a scanned entry.

    Accepts the exact name (``org/model`` or ``model``) or a unique
    unambiguous suffix (``model`` matching ``org/model``).
    """
    entries = list_models(models_dir)
    for entry in entries:
        if entry.name == name:
            return entry
    matches = [e for e in entries if e.name == name or e.name.rsplit("/", 1)[-1] == name]
    if not matches:
        raise ModelError(f"no model named {name!r} in {models_dir}")
    if len(matches) > 1:
        raise ModelError(
            f"model name {name!r} is ambiguous; use one of: {', '.join(m.name for m in matches)}"
        )
    return matches[0]


# ---------------------------------------------------------------------------
# Downloading models from the Hugging Face Hub
# ---------------------------------------------------------------------------

# Files that accompany a GGUF weights file (tokenizer, config, license...).
_GGUF_SUPPORT = ("*.json", "*.txt", "*.md", "*.model", "*.py")


def _validate_repo_id(repo_id: str) -> None:
    """Reject repo ids that could escape the store (traversal / absolute paths)."""
    repo = Path(repo_id)
    if not repo_id.strip() or repo == Path(".") or repo.is_absolute() or any(
        part in ("", ".", "..") for part in repo.parts
    ):
        raise ModelError(f"invalid repo id {repo_id!r}")


def download_model(
    repo_id: str,
    models_dir: Path,
    *,
    revision: str | None = None,
    allow: list[str] | None = None,
    exclude: list[str] | None = None,
    gguf: bool = False,
    quant: str | None = None,
) -> Path:
    """Download ``repo_id`` into ``models_dir`` and return the model directory.

    With ``gguf=True`` the download is restricted to ``.gguf`` weight files plus
    their support files (tokenizer/config/license). If the repo contains several
    GGUFs, ``quant`` picks one by case-insensitive substring match.
    """
    _validate_repo_id(repo_id)
    target = models_dir / repo_id
    target.mkdir(parents=True, exist_ok=True)

    patterns = list(allow or [])
    api = HfApi()

    if gguf:
        gguf_files = [f for f in api.list_repo_files(repo_id, revision=revision) if f.lower().endswith(".gguf")]
        if not gguf_files:
            raise ModelError(f"repo {repo_id!r} has no .gguf files to download")
        if quant:
            quant_lower = quant.lower()
            picked = [f for f in gguf_files if quant_lower in f.lower()]
            if not picked:
                raise ModelError(
                    f"no GGUF file matches quant {quant!r} in {repo_id!r}\n"
                    f"  available: {', '.join(gguf_files)}"
                )
        elif len(gguf_files) == 1:
            picked = gguf_files
        else:
            raise ModelError(
                f"repo {repo_id!r} has multiple GGUF files; pick one with --quant\n"
                f"  available: {', '.join(gguf_files)}"
            )
        patterns += picked + list(_GGUF_SUPPORT)

    snapshot_download(
        repo_id,
        revision=revision,
        local_dir=str(target),
        allow_patterns=patterns or None,
        ignore_patterns=exclude or None,
        max_workers=4,
    )
    return target


# ---------------------------------------------------------------------------
# Agent harnesses (Kilo Code, pi, Reasonix): generate config to use the model
# ---------------------------------------------------------------------------

# llama.cpp ignores authentication by default; a dummy key satisfies harnesses
# that require a non-empty apiKey.
DUMMY_KEY = "modelo"


def harness_output_dir() -> Path:
    return app_dir() / OUT_DIR_NAME


def _save_reference(filename: str, content: str) -> Path:
    out = harness_output_dir() / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(content, encoding="utf-8")
    return out


def _read_or_error(path: Path, what: str) -> str:
    """Read a user config file; missing files yield '', anything unreadable raises."""
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""
    except (OSError, UnicodeDecodeError) as exc:
        raise ModelError(f"cannot read {what} ({path}): {exc}") from None


def _pi_models_path() -> Path:
    base = Path(os.environ.get("PI_CODING_AGENT_DIR", Path.home() / ".pi"))
    return base / "agent" / "models.json"


def setup_pi(cfg: Config, model_id: str) -> str:
    """Write a 'modelo' provider into pi's ~/.pi/agent/models.json."""
    path = _pi_models_path()
    data: dict = {}
    if path.exists():
        try:
            data = json.loads(_read_or_error(path, "pi models.json"))
        except json.JSONDecodeError:
            raise ModelError(f"{path} exists but is not valid JSON; fix or remove it, then retry") from None
    if not isinstance(data, dict):
        raise ModelError(f"{path} is not a JSON object; fix or remove it, then retry")
    providers = data.setdefault("providers", {})
    providers["modelo"] = {
        "baseUrl": cfg.base_url,
        "api": "openai-completions",
        "apiKey": DUMMY_KEY,
        # llama.cpp does not understand the 'developer' role or reasoning_effort.
        "compat": {"supportsDeveloperRole": False, "supportsReasoningEffort": False},
        "models": [
            {
                "id": model_id,
                "name": f"modelo local ({model_id})",
                "contextWindow": cfg.n_ctx,
                "maxTokens": 4096,
            }
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    _save_reference("pi-models.json", json.dumps(data, indent=2) + "\n")
    return (
        f"wrote provider 'modelo' to {path}\n"
        "  next: restart pi, open /model, pick 'modelo' -> the served model, and use it."
    )


def _reasonix_home() -> Path:
    return Path(os.environ.get("REASONIX_HOME", Path.home() / ".reasonix"))


def _upsert_toml_provider(text: str, block: str, name: str) -> str:
    """Replace the [[providers]] block containing ``name = "<name>"``, or append ``block``."""
    lines = text.splitlines(keepends=True)
    blocks: list[tuple[int, int, str]] = []  # (start_line, end_line, raw)
    start = None
    first_start: int | None = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("["):  # table header: [x] or [[x]]
            if start is not None:
                blocks.append((start, i, "".join(lines[start:i])))
            if first_start is None:
                first_start = i
            start = i
    preamble_end = first_start if first_start is not None else len(lines)
    preamble = "".join(lines[:preamble_end])
    if start is not None:
        blocks.append((start, len(lines), "".join(lines[start:])))
    new_lines = []
    replaced = False
    for s, e, raw in blocks:
        if raw.strip().startswith("[[providers]]") and re.search(rf'name\s*=\s*"{name}"', raw):
            new_lines.append(block)
            replaced = True
        else:
            new_lines.extend(lines[s:e])
    body = "".join(new_lines)
    result = preamble + body
    if not replaced:
        # Never glue the new block onto a comment/key line lacking a newline.
        if result and not result.endswith("\n"):
            result += "\n"
        result += block
    return result


def _upsert_env(text: str, key: str, value: str) -> str:
    prefix = f"{key}="
    export_prefix = f"export {key}="
    out: list[str] = []
    replaced = False
    for line in text.splitlines():
        if line.strip().startswith(prefix) or line.strip().startswith(export_prefix):
            out.append(f"{key}={value}")
            replaced = True
        else:
            out.append(line)
    if not replaced:
        out.append(f"{key}={value}")
    return "\n".join(out) + "\n"


def setup_reasonix(cfg: Config, model_id: str) -> str:
    """Write a 'modelo' provider into ~/.reasonix/config.toml + secrets into .env."""
    home = _reasonix_home()
    config_path = home / "config.toml"
    env_path = home / ".env"
    key = "MODELO_API_KEY"
    block = (
        "[[providers]]\n"
        f'name        = "modelo"\n'
        f'kind        = "openai"\n'
        f'base_url    = "{cfg.base_url}"\n'
        f'models      = ["{model_id}"]\n'
        f'default     = "{model_id}"\n'
        f'api_key_env = "{key}"\n'
    )
    existing_config = _read_or_error(config_path, "reasonix config.toml")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(_upsert_toml_provider(existing_config, block, "modelo"), encoding="utf-8")

    existing_env = _read_or_error(env_path, "reasonix .env")
    env_path.write_text(_upsert_env(existing_env, key, DUMMY_KEY), encoding="utf-8")
    _save_reference("reasonix-provider.toml", block)
    return (
        f"wrote provider 'modelo' to {config_path} and {key}={DUMMY_KEY} to {env_path}\n"
        "  next: run 'reasonix' and select modelo/<model> — or set it as default_model."
    )


def setup_kilocode(cfg: Config, model_id: str) -> str:
    """Write a reference kilo.json (openai-compatible provider) to the harness-config dir."""
    data = {
        "$schema": "https://app.kilo.ai/config.json",
        "model": f"openai-compatible/{model_id}",
        "provider": {
            "openai-compatible": {
                "options": {
                    "apiKey": DUMMY_KEY,
                    "baseURL": cfg.base_url,
                },
                "models": {
                    model_id: {
                        "name": f"modelo local ({model_id})",
                        "tool_call": True,
                        "limit": {"context": cfg.n_ctx, "output": 4096},
                    }
                },
            }
        },
    }
    content = json.dumps(data, indent=2) + "\n"
    out = _save_reference("kilo.json", content)
    return (
        f"wrote {out}\n"
        "  next: copy it into the Kilo Code project as kilo.json (or kilo.jsonc), "
        "or merge its 'provider' section into your global ~/.config/kilo config; "
        "then select openai-compatible/<model> in the model picker."
    )


_SETUPTERS = {
    "kilocode": setup_kilocode,
    "pi": setup_pi,
    "reasonix": setup_reasonix,
}

# Order controls the "--all" run and error messages, so keep it explicit.
HARNESSES = tuple(_SETUPTERS)


def _setup_one(name: str, cfg: Config, model_id: str) -> None:
    fn = _SETUPTERS.get(name)
    if fn is None:
        raise ModelError(f"unknown harness {name!r}; supported: {', '.join(HARNESSES)}")
    message = fn(cfg, model_id)
    typer.secho(f"[{name}] {message}", fg=typer.colors.GREEN)


def run_setup(*, harness: str | None, all: bool, cfg: Config, model: str | None) -> None:
    model_id = model or cfg.default_model
    if not model_id:
        raise ModelError(
            "no model to advertise — pass --model, or download one first (see 'modelo store download --help')"
        )
    if all:
        for name in HARNESSES:
            _setup_one(name, cfg, model_id)
    else:
        _setup_one(harness, cfg, model_id)  # raises for unknown/missing
    typer.secho(
        f"endpoint: {cfg.base_url}  model: {model_id}\n"
        "start the server with 'modelo store serve' before using the harness.",
        fg=typer.colors.CYAN,
    )


def harness_state(cfg: Config) -> dict[str, bool]:
    """Which harnesses are already configured to point at the local endpoint."""
    state: dict[str, bool] = {}
    pi_path = _pi_models_path()
    state["pi"] = pi_path.exists() and _json_has_provider(pi_path, "modelo")
    config_path = _reasonix_home() / "config.toml"
    try:
        config_text = _read_or_error(config_path, "reasonix config.toml")
    except ModelError:
        config_text = ""
    state["reasonix"] = config_path.exists() and bool(
        re.search(r'name\s*=\s*"modelo"', config_text)
    )
    state["kilocode"] = (harness_output_dir() / "kilo.json").exists()
    return state


def _json_has_provider(path: Path, name: str) -> bool:
    try:
        data = json.loads(_read_or_error(path, "pi models.json"))
    except (json.JSONDecodeError, ModelError):
        return False
    return isinstance(data, dict) and name in data.get("providers", {})


# ---------------------------------------------------------------------------
# Serving: OpenAI-compatible API for a GGUF model via llama-cpp-python
# ---------------------------------------------------------------------------

# Preferred quantization when a model directory holds several GGUFs.
_QUANT_PREFERENCE = ("q4_k_m", "q4_0", "q8_0", "q5_k_m", "q5_0", "q6_k", "f16", "f32")


def _pick_models_dir(cfg: Config, dir_override: Path | None) -> Path:
    return Path(dir_override) if dir_override else cfg.models_path


def _pick_model(
    cfg: Config,
    dir_override: Path | None,
    name: str | None,
) -> ModelEntry:
    models_dir = _pick_models_dir(cfg, dir_override)
    if name:
        return find_model(name, models_dir)
    if cfg.default_model:
        try:
            return find_model(cfg.default_model, models_dir)
        except ModelError:
            pass
    gguf_entries = [e for e in list_models(models_dir) if e.gguf_files]
    if not gguf_entries:
        raise ModelError(
            "no GGUF model in the store to serve — run 'modelo store download <repo> --gguf'"
        )
    if len(gguf_entries) > 1:
        names = ", ".join(e.name for e in gguf_entries)
        raise ModelError(f"multiple GGUF models available; pick one with 'modelo store serve <model>':\n  {names}")
    return gguf_entries[0]


def pick_gguf(entry: ModelEntry) -> ModelFile:
    """Choose one GGUF file from a model directory (preferred quant, else smallest)."""
    ggu = sorted(entry.gguf_files, key=lambda f: f.size)
    for pref in _QUANT_PREFERENCE:
        for f in ggu:
            if pref in f.name.lower():
                return f
    return ggu[0]


def _build_app(gguf_path: Path, alias: str, host: str, port: int, gpu_layers: int, n_ctx: int):
    """Build the llama-cpp-python FastAPI app, tolerating old/new settings APIs."""
    from llama_cpp.server.app import create_app
    from llama_cpp.server import settings as s

    try:
        server_settings = s.ServerSettings(host=host, port=port)
        model_settings = [
            s.ModelSettings(
                model=str(gguf_path), model_alias=alias, n_gpu_layers=gpu_layers, n_ctx=n_ctx
            )
        ]
        return create_app(server_settings=server_settings, model_settings=model_settings)
    except AttributeError:
        # Older llama-cpp-python: a single combined Settings object.
        settings = s.Settings(
            model=str(gguf_path),
            model_alias=alias,
            host=host,
            port=port,
            n_gpu_layers=gpu_layers,
            n_ctx=n_ctx,
        )
        return create_app(settings=settings)


def run_server(
    *,
    model: str | None,
    cfg: Config,
    dir_override: Path | None = None,
    host: str | None = None,
    port: int | None = None,
    gpu_layers: int | None = None,
    n_ctx: int | None = None,
) -> None:
    """Serve one GGUF model on an OpenAI-compatible endpoint (foreground)."""
    entry = _pick_model(cfg, dir_override, model)
    gguf = pick_gguf(entry)
    host = host or cfg.host
    port = port or cfg.port
    gpu_layers = cfg.gpu_layers if gpu_layers is None else gpu_layers
    n_ctx = cfg.n_ctx if n_ctx is None else n_ctx

    try:
        import uvicorn  # noqa: F401
        from llama_cpp.server.app import create_app  # noqa: F401
    except ImportError:
        raise ModelError(
            "serving needs llama-cpp-python and uvicorn — install them, e.g. "
            "pip install llama-cpp-python uvicorn"
        ) from None

    app = _build_app(entry.path / gguf.name, alias=entry.name, host=host, port=port, gpu_layers=gpu_layers, n_ctx=n_ctx)
    typer.secho(
        f"serving {entry.name} [{gguf.name}] on http://{host}:{port}/v1  "
        f"(ctx={n_ctx}, gpu_layers={gpu_layers})  (Ctrl+C to stop)",
        fg=typer.colors.GREEN,
        bold=True,
    )
    uvicorn.run(app, host=host, port=port, log_level="info")


def probe_server(host: str, port: int, timeout: float = 2.0) -> tuple[bool, list[str]]:
    """Return (server_up, model_ids) by hitting the OpenAI /v1/models endpoint."""
    try:
        resp = requests.get(f"http://{host}:{port}/v1/models", timeout=timeout)
    except requests.RequestException:
        return False, []
    if resp.status_code != 200:
        return False, []
    try:
        return True, [m.get("id", "?") for m in resp.json().get("data", [])]
    except ValueError:
        return False, []


def server_status(cfg: Config) -> None:
    """Print server state and store summary."""
    up, models = probe_server(cfg.host, cfg.port)
    if up:
        typer.secho(f"server running at {cfg.base_url}", fg=typer.colors.GREEN, bold=True)
        for mid in models:
            typer.echo(f"  - {mid}")
    else:
        typer.echo(f"no server responding at {cfg.base_url}")

    default = cfg.default_model or "(none — set by first download)"
    typer.echo(f"default model: {default}")

    state = harness_state(cfg)
    typer.echo("harness configs: " + ", ".join(
        f"{name}=configured" if on else f"{name}=missing" for name, on in sorted(state.items())
    ))

    entries = list_models(cfg.models_path)
    typer.echo(f"store: {len(entries)} model(s) in {cfg.models_path}")
    for e in entries:
        typer.echo(f"  - {e.name} [{e.kind}, {e.total_size} bytes]" + ("" if e.healthy else " [INCOMPLETE]"))


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

app = typer.Typer(
    name="modelo",
    help="Download Hugging Face models into a local store, scan them, serve them "
    "via an OpenAI-compatible API, and wire them into agent harnesses.",
    no_args_is_help=True,
)
store_app = typer.Typer(help="Download, scan and manage the local model store.")
app.add_typer(store_app, name="store", help="Manage the local model store.")

_DEFAULT_MODELS_DIR_OPTION = typer.Option(
    None, "--dir", "-d", help="Model store directory (default: from config)."
)


def _cfg() -> Config:
    return load_config()


def _resolve_models_dir(cfg: Config, override: Path | None) -> Path:
    return Path(override) if override else ensure_models_dir(cfg)


def _print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    typer.echo(fmt.format(*headers))
    typer.echo("  ".join("-" * w for w in widths))
    for row in rows:
        typer.echo(fmt.format(*row))


def _health_label(entry: ModelEntry) -> str:
    if not entry.has_weights:
        return "no weights"
    if entry.incomplete:
        return "INCOMPLETE"
    return "ok"


def _print_model_summary(entry: ModelEntry) -> None:
    typer.secho(f"{entry.name}  ({entry.path})", fg=typer.colors.CYAN, bold=True)
    flags = []
    if entry.has_config:
        flags.append("config.json")
    if entry.gguf_files:
        flags.append(f"GGUF quants: {', '.join(entry.quants) or 'unknown'}")
    if entry.incomplete:
        flags.append("INCOMPLETE download")
    if entry.kind == "other":
        flags.append("no weight files")
    typer.echo(f"  type: {entry.kind} | size: {human_size(entry.total_size)} | " + " | ".join(flags or ["empty"]))
    for f in entry.files:
        mark = "" if f.kind == "other" else f" [{f.kind}]"
        typer.echo(f"  {human_size(f.size):>10}  {f.name}{mark}")


@store_app.command("download")
def download(
    repo: str = typer.Argument(..., help="Hugging Face repo id, e.g. org/model"),
    models_dir: Path | None = _DEFAULT_MODELS_DIR_OPTION,
    revision: str | None = typer.Option(None, "--revision", help="Repo revision (branch, tag or commit)."),
    allow: list[str] = typer.Option(None, "--allow", help="File patterns to download (repeatable)."),
    exclude: list[str] = typer.Option(None, "--exclude", help="File patterns to skip (repeatable)."),
    gguf: bool = typer.Option(False, "--gguf", help="Download only GGUF weights + tokenizer/config files."),
    quant: str | None = typer.Option(None, "--quant", help="With --gguf: pick this quantization (case-insensitive)."),
):
    """Download a model from the Hugging Face Hub into the store."""
    cfg = _cfg()
    models_dir = _resolve_models_dir(cfg, models_dir)
    try:
        target = download_model(
            repo, models_dir, revision=revision, allow=allow, exclude=exclude, gguf=gguf, quant=quant
        )
    except ModelError as exc:
        typer.secho(f"error: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    entry = scan_model(repo, target)
    if cfg.default_model is None:
        cfg.default_model = repo
        save_config(cfg)
    typer.secho(f"downloaded {repo!r} -> {target}", fg=typer.colors.GREEN)
    _print_model_summary(entry)


@store_app.command("list")
def list_models_cmd(
    models_dir: Path | None = _DEFAULT_MODELS_DIR_OPTION,
):
    """List models in the store."""
    cfg = _cfg()
    models_dir = _resolve_models_dir(cfg, models_dir)
    entries = list_models(models_dir)
    if not entries:
        typer.echo(f"no models in {models_dir} — run 'modelo store download <repo>'")
        return
    rows = [
        [
            e.name,
            e.kind,
            human_size(e.total_size),
            ",".join(e.quants) or "-",
            f"{len(e.weight_files)}",
            _health_label(e),
        ]
        for e in entries
    ]
    _print_table(["model", "type", "size", "quant", "weights", "health"], rows)


@store_app.command("scan")
def scan_cmd(
    model: str | None = typer.Argument(None, help="Model name to inspect (default: all models)."),
    models_dir: Path | None = _DEFAULT_MODELS_DIR_OPTION,
):
    """Inspect model directories: files, sizes, health."""
    cfg = _cfg()
    models_dir = _resolve_models_dir(cfg, models_dir)
    entries = [find_model(model, models_dir)] if model else list_models(models_dir)
    if not entries:
        typer.echo(f"no models in {models_dir} — run 'modelo store download <repo>'")
        return
    for entry in entries:
        _print_model_summary(entry)
        typer.echo("")


@store_app.command("info")
def info_cmd(
    model: str = typer.Argument(...),
    models_dir: Path | None = _DEFAULT_MODELS_DIR_OPTION,
):
    """Show details for one model, including config.json when present."""
    cfg = _cfg()
    models_dir = _resolve_models_dir(cfg, models_dir)
    entry = find_model(model, models_dir)
    _print_model_summary(entry)
    cfg_file = entry.path / "config.json"
    if cfg_file.is_file():
        typer.secho("\nconfig.json:", fg=typer.colors.CYAN, bold=True)
        typer.echo(cfg_file.read_text(encoding="utf-8"))


@store_app.command("remove")
def remove_cmd(
    model: str = typer.Argument(...),
    models_dir: Path | None = _DEFAULT_MODELS_DIR_OPTION,
    yes: bool = typer.Option(False, "--yes", "-y", help="Do not ask for confirmation."),
):
    """Delete a model from the store."""
    cfg = _cfg()
    models_dir = _resolve_models_dir(cfg, models_dir)
    entry = find_model(model, models_dir)
    if not yes:
        typer.confirm(f"delete {entry.name!r} ({human_size(entry.total_size)}) from {models_dir}?", abort=True)
    shutil.rmtree(entry.path)
    # Prune now-empty namespace parents (e.g. org/ left over from org/model).
    parent = entry.path.parent
    while parent != models_dir and parent.is_dir() and not any(parent.iterdir()):
        parent.rmdir()
        parent = parent.parent
    if cfg.default_model == entry.name:
        cfg.default_model = None
        save_config(cfg)
    typer.secho(f"removed {entry.name}", fg=typer.colors.GREEN)


@store_app.command("serve")
def serve_cmd(
    model: str | None = typer.Argument(None, help="Model to serve (default: config default_model)."),
    models_dir: Path | None = _DEFAULT_MODELS_DIR_OPTION,
    host: str | None = typer.Option(None, "--host", help="Bind address (default: from config)."),
    port: int | None = typer.Option(None, "--port", "-p", help="Port (default: from config)."),
    gpu_layers: int | None = typer.Option(None, "--gpu-layers", "-ngl", help="Layers offloaded to GPU (0 = CPU)."),
    ctx: int | None = typer.Option(None, "--ctx", help="Context window in tokens (default: from config, 2048)."),
):
    """Serve a GGUF model behind an OpenAI-compatible API."""
    run_server(
        model=model,
        cfg=_cfg(),
        dir_override=models_dir,
        host=host,
        port=port,
        gpu_layers=gpu_layers,
        n_ctx=ctx,
    )


@store_app.command("status")
def status_cmd(
    models_dir: Path | None = _DEFAULT_MODELS_DIR_OPTION,
):
    """Show whether a server is running and which harnesses are configured."""
    cfg = _cfg()
    server_status(cfg)


@store_app.command("setup")
def setup_cmd(
    harness: str | None = typer.Argument(None, help="harness name, or --all"),
    all: bool = typer.Option(False, "--all", help="Generate config for every supported harness."),
    model: str | None = typer.Option(None, "--model", help="Model id to advertise (default: config default_model)."),
):
    """Generate config so an agent harness (kilocode, pi, reasonix) can use the served model."""
    run_setup(harness=harness, all=all, cfg=_cfg(), model=model)


@app.command("version")
def version_cmd():
    """Print the version."""
    typer.echo(f"modelo {__version__}")


def main() -> None:
    try:
        app()
    except ModelError as exc:
        typer.secho(f"error: {exc}", fg=typer.colors.RED)
        sys.exit(1)


if __name__ == "__main__":
    main()
