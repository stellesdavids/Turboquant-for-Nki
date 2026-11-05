# TurboQuant for NKI

This project implements the TurboQuant KV-cache compression pipeline using **AWS Neuron Kernel Interface (NKI)**, with standalone interfaces for KV compression and decompression.

The implementation covers L2 normalization, orthogonal rotation, Lloyd-Max scalar quantization, and KV reconstruction. Computation is distributed across Trainium's VectorE, TensorE, GpSIMD, and DMA engines. The caller runs QKV projection and attention, integrating the codec through `compress()` and `decompress()`.

## Implementation

Compression:

```text
K / V -> L2 normalization -> Orthogonal rotation -> Lloyd-Max binary search -> UINT8 codes
                 |
                 +-------------------------------------------------------> FP32 L2 scales
```

Decompression:

```text
Codes -> Centroid lookup -> Inverse rotation -> Apply L2 scales -> Reconstructed K / V
```

Each tile contains 128 tokens for one K or V head. Compression and decompression each use one standalone NKI kernel invocation per tile. Tiles execute sequentially.

| Engine | Operations |
|---|---|
| VectorE | L2 normalization, conversion to ordered integer representations, binary-search midpoint calculation, scale restoration |
| TensorE | Matrix multiplication for orthogonal rotation and inverse rotation |
| GpSIMD | Threshold lookup, integer comparison and index selection, centroid lookup, SWDGE address descriptor generation |
| DMA | Data loading, quantization-code byte transfers, output writeback |

GpSIMD search uses ordered integer representations to implement floating-point threshold comparisons, handling signed zeros and integer subtraction overflow. Rotation operands are rounded to BF16, with results stored in FP32. L2 scales and reconstructed K/V also use FP32. Zero vectors use a scale floor of `sqrt(1e-20)`.

## Interface and Data Format

The entry point is `TurboQuantNKI` in `turboquant.py`.

| Parameter | Shape and requirements |
|---|---|
| `rotation` | `[128, 128]`, a finite orthogonal matrix; decompression uses its transpose |
| `thresholds` | `[2**b - 1]`, strictly increasing finite thresholds |
| `centroids` | `[2**b]`, finite quantization centroids; supports `1 <= b <= 8` |
| `K`, `V` | FP32 arrays with the same shape, `[tokens, heads, 128]`; the token count must be a positive multiple of 128 |

The caller supplies the rotation matrix, Lloyd-Max thresholds, and centroids. Inputs must be finite, and their L2 reductions must remain within the FP32 range accepted by the implementation.

`compress()` returns a `KVCache` containing:

- `codes`: UINT8 array of shape `[2, heads, tokens, 128]`. The first dimension selects K or V, in that order.
- `scales`: FP32 array of shape `[2, heads, tokens, 1]`.
- `bits`: Quantization bit width `b`.
- `codec_id`: An identifier for the rotation matrix, thresholds, and centroids, used to check cache compatibility.

The storage format uses one UINT8 byte per quantization code; `b=3` selects eight quantization levels. Each token in each K or V head requires 128 bytes of codes and a 4-byte scale.

`decompress()` returns two FP32 arrays, each with shape `[tokens, heads, 128]`. The public interfaces use NumPy arrays, so cache data and reconstructed outputs are transferred between host and device.

## Usage

The runtime environment requires a Trainium-compatible AWS Neuron SDK, the NKI compilation interfaces and runtime, and NumPy. The code currently targets **Trn1, LNC1, NeuronCore 0**.

Run your calling script from this directory using a Neuron Python environment. The environment available on this machine is:

```bash
cd /home/ubuntu/basic_kv_compression
/opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin/python -B your_script.py
```

Prepare `R`, `T`, `Gamma`, `K`, and `V` according to the interface requirements above:

```python
from turboquant import TurboQuantNKI

codec = TurboQuantNKI(
    rotation=R,
    thresholds=T,
    centroids=Gamma,
)

cache = codec.compress(K, V)
K_hat, V_hat = codec.decompress(cache)

# Pass Q, K_hat, and V_hat to your attention kernel using its required layout.
```

Creating an instance compiles the required kernels when cached artifacts are unavailable. Compilation artifacts default to `~/.cache/turboquant_nki/`, respect `XDG_CACHE_HOME`, and can be redirected with `artifact_dir`. Set `trace=True` to save execution traces.

## Source Files

| File | Contents |
|---|---|
| `turboquant.py` | Parameter validation, the `KVCache` data structure, compilation and runtime management, compression and decompression interfaces |
| `native_kernels.py` | GpSIMD threshold search and the complete compression kernel |
| `kernels.py` | Normalization, rotation, centroid lookup, scale restoration, and the complete decompression kernel |
