"""Configuration system for the Doom SNN pipeline using typed dataclasses."""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, field, fields, is_dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Type, TypeVar, Union

import yaml

T = TypeVar("T")


@dataclass
class ObsConfig:
    """Observation and image preprocessing configuration."""

    grayscale: bool = True
    height: int = 24
    width: int = 32
    normalize_min: float = 0.0
    normalize_max: float = 1.0
    flatten: bool = True
    input_channels: int = 768


@dataclass
class ActionConfig:
    """Action space configuration."""

    num_actions: int = 6
    names: List[str] = field(
        default_factory=lambda: [
            "noop",
            "left",
            "right",
            "shoot",
            "left_shoot",
            "right_shoot",
        ]
    )
    action_repeat: int = 4


@dataclass
class NeuronConfig:
    """Spiking neuron dynamics configuration."""

    type: str = "lif"
    beta: float = 0.75
    threshold: float = 0.4
    reset: str = "soft"
    surrogate: str = "fast_sigmoid"
    T: int = 8


@dataclass
class ModelConfig:
    """SNN model architecture configuration."""

    track: str = "c_rsnn_ac"
    core_neurons: int = 96
    recurrent: bool = True
    readout: str = "membrane"
    actor_outputs: int = 6
    critic_outputs: int = 1
    q_outputs: int = 6


@dataclass
class TrainingConfig:
    """Actor-Critic / Policy Gradient training configuration."""

    algorithm: str = "a2c_gae"
    episodes: int = 500
    rollout_length: int = 8
    gamma: float = 0.99
    gae_lambda: float = 0.95
    lr: float = 3e-4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    grad_clip: float = 0.5
    eval_every: int = 50
    save_best: bool = True


@dataclass
class DQNTrainingConfig:
    """Deep Q-Network baseline training configuration."""

    lr: float = 1e-3
    gamma: float = 0.99
    buffer_size: int = 50000
    batch_size: int = 64
    target_update_interval: int = 500
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 10000
    min_history: int = 1000


@dataclass
class EvalConfig:
    """Policy evaluation configuration."""

    episodes_per_seed: int = 100
    seeds: List[int] = field(default_factory=lambda: [6201, 6202, 6203])
    greedy: bool = True
    record_action_distribution: bool = True
    record_spike_stats: bool = True
    record_latency: bool = True


@dataclass
class QuantizationConfig:
    """Hardware quantization configuration."""

    enabled: bool = False
    mode: str = "fake_quant_int16"
    weight_bits: int = 16
    membrane_bits: int = 16
    readout_bits: int = 16
    qat: bool = False
    calibration_episodes: int = 10
    int8_exploratory: bool = False


@dataclass
class LoggingConfig:
    """Telemetry and logging configuration."""

    log_interval: int = 10
    csv: bool = True
    json_summary: bool = True
    tensorboard: bool = False


@dataclass
class EnvConfig:
    """ViZDoom game environment configuration."""

    vizdoom_config: str = ""
    scenario: str = "basic"
    headless: bool = True
    dummy_env: bool = False


@dataclass
class RunConfig:
    """Top-level unified experiment run configuration."""

    name: str = "default"
    seed: int = 6201
    device: str = "auto"
    output_dir: str = "runs"
    obs: ObsConfig = field(default_factory=ObsConfig)
    actions: ActionConfig = field(default_factory=ActionConfig)
    neuron: NeuronConfig = field(default_factory=NeuronConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    dqn: DQNTrainingConfig = field(default_factory=DQNTrainingConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    env: EnvConfig = field(default_factory=EnvConfig)


def _parse_override_value(val_str: str) -> Any:
    """Parse a string representation of an override value into a Python object."""
    stripped = val_str.strip()
    if stripped.lower() == "true":
        return True
    if stripped.lower() == "false":
        return False
    if stripped.lower() in ("none", "null"):
        return None
    try:
        return ast.literal_eval(stripped)
    except (ValueError, SyntaxError):
        return stripped


def _dict_to_dataclass(target_cls: Type[T], data: Dict[str, Any]) -> T:
    """Recursively convert nested dictionary into a dataclass instance."""
    if not is_dataclass(target_cls):
        return data  # type: ignore[return-value]

    dataclass_registry = {
        "ObsConfig": ObsConfig,
        "ActionConfig": ActionConfig,
        "NeuronConfig": NeuronConfig,
        "ModelConfig": ModelConfig,
        "TrainingConfig": TrainingConfig,
        "DQNTrainingConfig": DQNTrainingConfig,
        "EvalConfig": EvalConfig,
        "QuantizationConfig": QuantizationConfig,
        "LoggingConfig": LoggingConfig,
        "EnvConfig": EnvConfig,
        "RunConfig": RunConfig,
    }

    class_fields = {f.name: f for f in fields(target_cls)}
    kwargs: Dict[str, Any] = {}

    for name, value in data.items():
        if name not in class_fields:
            continue
        field_info = class_fields[name]
        field_type = field_info.type

        target_sub_cls = None
        if isinstance(field_type, str) and field_type in dataclass_registry:
            target_sub_cls = dataclass_registry[field_type]
        elif isinstance(field_type, type) and is_dataclass(field_type):
            target_sub_cls = field_type
        elif callable(field_info.default_factory) and is_dataclass(field_info.default_factory):
            target_sub_cls = field_info.default_factory

        if target_sub_cls is not None and isinstance(value, dict):
            kwargs[name] = _dict_to_dataclass(target_sub_cls, value)
        else:
            kwargs[name] = value

    return target_cls(**kwargs)


def _apply_override(config: RunConfig, override: str) -> None:
    """Apply a dot-notation key=value override to a RunConfig instance."""
    if "=" not in override:
        raise ValueError(f"Invalid override format '{override}', expected 'key=value'")

    key_path, val_str = override.split("=", 1)
    keys = [k.strip() for k in key_path.strip().split(".") if k.strip()]
    if not keys:
        raise ValueError(f"Empty key path in override: '{override}'")

    parsed_val = _parse_override_value(val_str)

    target: Any = config
    for key in keys[:-1]:
        if not hasattr(target, key):
            raise AttributeError(f"Config object has no attribute '{key}' in path '{key_path}'")
        target = getattr(target, key)

    last_key = keys[-1]
    if not hasattr(target, last_key):
        raise AttributeError(f"Config object has no attribute '{last_key}' in path '{key_path}'")

    setattr(target, last_key, parsed_val)


def load_config(
    yaml_path: Optional[Union[str, Path]] = None,
    cli_overrides: Optional[Sequence[str]] = None,
) -> RunConfig:
    """Load RunConfig from YAML file and merge with optional dot-notation CLI overrides.

    Args:
        yaml_path: Path to YAML configuration file. If None, default RunConfig is used.
        cli_overrides: Sequence of 'section.field=value' overrides (e.g., ['neuron.beta=0.5']).

    Returns:
        Configured RunConfig instance.
    """
    if yaml_path is not None:
        path = Path(yaml_path)
        if not path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {path.resolve()}")
        with path.open("r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f)
        if not isinstance(raw_data, dict):
            raw_data = {}
        config = _dict_to_dataclass(RunConfig, raw_data)
    else:
        config = RunConfig()

    if cli_overrides:
        for override in cli_overrides:
            _apply_override(config, override)

    return config


def save_config(config: RunConfig, path: Union[str, Path]) -> None:
    """Serialize RunConfig to YAML format and write to destination path.

    Args:
        config: The RunConfig to serialize.
        path: Output file destination path.
    """
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    config_dict = asdict(config)
    with save_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config_dict, f, default_flow_style=False, sort_keys=False)


def config_hash(config: RunConfig) -> str:
    """Compute a deterministic MD5 hash of the serialized configuration.

    Args:
        config: The RunConfig instance to hash.

    Returns:
        32-character hexadecimal MD5 hash string.
    """
    serialized = json.dumps(asdict(config), sort_keys=True)
    return hashlib.md5(serialized.encode("utf-8")).hexdigest()
