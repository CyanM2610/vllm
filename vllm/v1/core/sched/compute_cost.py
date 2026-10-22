"""Static dense-Qwen prefill FLOP proxy, independent of measured latencies.

The proxy counts decoder matrix multiplications (two FLOPs per multiply-add),
including GQA projections and all query heads' causal attention.  It is not a
GPU time model. Output projection/sampling are excluded; the frozen runtime may
also execute these at partial chunks. See prepared/compute-key/COST-MODEL.md.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from numbers import Integral


def _field(config, name, default=None):
    return config.get(name, default) if isinstance(config, dict) else getattr(config, name, default)


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


@dataclass(frozen=True)
class ComputeCostModel:
    """Integer-valued ordering proxy derived only from model structure."""

    num_hidden_layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int

    @classmethod
    def from_config(cls, hf_config):
        if _field(hf_config, "model_type") != "qwen2":
            raise ValueError("compute-key baseline supports the frozen dense qwen2 model only")
        if _field(hf_config, "hidden_act") != "silu":
            raise ValueError("compute-key cost assumes the Qwen SwiGLU/silu MLP")
        if _field(hf_config, "use_sliding_window", False):
            raise ValueError("compute-key cost assumes full causal attention, not sliding windows")
        values = {
            name: _positive_int(_field(hf_config, name), name)
            for name in (
                "num_hidden_layers", "hidden_size", "intermediate_size",
                "num_attention_heads", "num_key_value_heads", "vocab_size",
            )
        }
        h, nq, nkv = values["hidden_size"], values["num_attention_heads"], values["num_key_value_heads"]
        if h % nq or nq % nkv:
            raise ValueError("compute-key cost requires integral dense GQA head geometry")
        values["head_dim"] = _positive_int(_field(hf_config, "head_dim", h // nq), "head_dim")
        if values["head_dim"] * nq != h:
            raise ValueError("compute-key cost requires query width equal to hidden size")
        # The experiment is deliberately not a general model scheduler. Fail
        # closed if the deployed model differs from the frozen Qwen2.5-7B.
        expected = (28, 3584, 18944, 28, 4, 128, 152064)
        actual = tuple(values[name] for name in cls.__dataclass_fields__)
        if actual != expected:
            raise ValueError(f"compute-key cost outside frozen Qwen2.5-7B geometry: {actual!r}")
        return cls(**values)

    @property
    def kv_width(self):
        return self.num_key_value_heads * self.head_dim

    @property
    def a(self):
        h, i, k = self.hidden_size, self.intermediate_size, self.kv_width
        return self.num_hidden_layers * (4 * h * h + 4 * h * k + 6 * h * i)

    @property
    def b(self):
        return 4 * self.num_hidden_layers * self.hidden_size

    def estimate(self, r, p):
        for name, value in (("r", r), ("p", p)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
        r, p = int(r), int(p)
        return self.a * r + self.b * (r * p + r * (r + 1) // 2)

    def to_dict(self):
        return {
            "schema": "qwen-prefill-structural-cost-v1",
            "model_geometry": asdict(self),
            "kv_width": self.kv_width,
            "a": self.a,
            "b": self.b,
            "formula": "a*r + b*(r*p + r*(r+1)/2)",
            "units": "theoretical decoder matrix-multiplication FLOPs (2 per MAC)",
            "output_projection_flops_per_logits_position_excluded": 2 * self.hidden_size * self.vocab_size,
            "omitted": [
                "LM head and sampling, including discarded partial-chunk positions",
                "RMSNorm, SiLU, elementwise products, softmax, RoPE, bias and residual additions",
                "padding, kernel tiling, launch, memory traffic, cache transfer and control waits",
            ],
            "fitted_to_measurements": False,
            "gpu_time_prediction": False,
        }
