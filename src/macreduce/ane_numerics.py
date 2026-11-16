"""Explicit ANEForge 0.4.0 lowering for large Qwen RMSNorm inputs.

Forward rescales the native L2 reduction; backward keeps the original input
and epsilon so small epsilon values are not rounded away by fp16 arithmetic.
This preserves the backend's existing RMSNorm epsilon-floor approximation.
"""

from importlib.metadata import version
from types import SimpleNamespace

_REGISTERED = False


def scaled_rms_norm(value, gamma, eps):
    global _REGISTERED
    if not _REGISTERED:
        if version("aneforge") != "0.4.0":
            raise RuntimeError("scaled RMSNorm requires the validated ANEForge 0.4.0 lowering")
        from aneforge._compile import _e_rms_norm, op
        from aneforge.autograd import _vjp_rms_norm, vjp

        @op("macreduce_scaled_rms_norm")
        def emit(em, tensor, name, sources):
            scale = 128.0
            em.line(f'fp16 {name}_scale = const()[name=string("{name}_scale"), '
                    f'val=fp16({float(1 / scale).hex()})];')
            em.line(f'{em.ty(tensor.shape)} {name}_scaled = mul(x={sources[0]}, '
                    f'y={name}_scale)[name=string("{name}_scaled")];')
            proxy = SimpleNamespace(shape=tensor.shape,
                                    attrs={**tensor.attrs, "eps": tensor.attrs["eps"] / scale**2})
            _e_rms_norm(em, proxy, name, [f"{name}_scaled"])

        vjp("macreduce_scaled_rms_norm")(_vjp_rms_norm)
        _REGISTERED = True
    node = value.rms_norm(gamma, eps)
    if node.op != "rms_norm":
        raise ValueError("scaled RMSNorm requires a frozen array gamma")
    node.op = "macreduce_scaled_rms_norm"
    return node


def scale_lora_factors(values):
    """Keep A @ B unchanged while lifting tiny B factors before fp16 upload."""
    return [value / 16.0 if i % 2 == 0 else value * 16.0 for i, value in enumerate(values)]


def restore_lora_gradients(values):
    """Chain rule back to the original master-factor coordinates."""
    return [value / 16.0 if i % 2 == 0 else value * 16.0 for i, value in enumerate(values)]
