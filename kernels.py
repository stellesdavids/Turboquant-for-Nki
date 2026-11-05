"""TurboQuant for NKI: tensor primitives and KV-cache decompression."""
import nki
import nki.language as nl
import nki.isa as ni

@nki.jit
def normalize(x):
    """VectorE L2 normalization, including a VectorE Newton reciprocal square root.

    Avoid activation(rsqrt), which would silently introduce ScalarE. Five Newton
    steps refine a bit-pattern seed to FP32 precision for normal positive sums.
    The norm floor is sqrt(1e-20), including for zero rows.
    """
    out = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.shared_hbm)
    scales = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    a = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.sbuf)
    square = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.sbuf)
    total = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    half = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    bits = nl.ndarray((128, 1), dtype=nl.uint32, buffer=nl.sbuf)
    seed = nl.ndarray((128, 1), dtype=nl.uint32, buffer=nl.sbuf)
    ni.dma_copy(dst=a, src=x)
    ni.tensor_tensor(dst=square, data1=a, data2=a, op=nl.multiply, engine=ni.engine.vector)
    ni.tensor_reduce(dst=total, data=square, op=nl.add, axis=1)
    ni.tensor_scalar(dst=total, data=total, op0=nl.maximum, operand0=1e-20, engine=ni.engine.vector)
    ni.tensor_scalar(dst=half, data=total, op0=nl.multiply, operand0=0.5, engine=ni.engine.vector)
    ni.tensor_scalar(dst=bits, data=total.ap(pattern=[[1,128],[1,1]], dtype=nl.uint32),
                     op0=nl.right_shift, operand0=1, engine=ni.engine.vector)
    ni.tensor_scalar(dst=seed, data=bits, op0=nl.subtract, operand0=1597463007,
                     reverse0=True, engine=ni.engine.vector)
    inv = seed.ap(pattern=[[1,128],[1,1]], dtype=nl.float32)
    for iteration in nl.static_range(5):
        squared = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
        product = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
        correction = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
        refined = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
        ni.tensor_tensor(dst=squared, data1=inv, data2=inv, op=nl.multiply, engine=ni.engine.vector)
        ni.tensor_tensor(dst=product, data1=half, data2=squared, op=nl.multiply, engine=ni.engine.vector)
        ni.tensor_scalar(dst=correction, data=product, op0=nl.subtract,
                         operand0=1.5, reverse0=True, engine=ni.engine.vector)
        ni.tensor_tensor(dst=refined, data1=inv, data2=correction, op=nl.multiply, engine=ni.engine.vector)
        inv = refined
    norm = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    unit = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.sbuf)
    ni.tensor_tensor(dst=norm, data1=total, data2=inv, op=nl.multiply, engine=ni.engine.vector)
    ni.tensor_scalar(dst=unit, data=a, op0=nl.multiply, operand0=inv, engine=ni.engine.vector)
    ni.dma_copy(dst=out, src=unit)
    ni.dma_copy(dst=scales, src=norm)
    return out, scales



@nki.jit
def rotate(x, rotation):
    """TensorE dense multiplication; operands rounded to BF16, FP32 result."""
    out = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.shared_hbm)
    a = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
    r = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
    trans = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.psum)
    trans_bf = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
    acc = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.psum)
    result = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.sbuf)
    ni.dma_copy(dst=a, src=x)
    ni.dma_copy(dst=r, src=rotation)
    ni.nc_transpose(dst=trans, data=a)
    ni.tensor_copy(dst=trans_bf, src=trans)
    ni.nc_matmul(dst=acc, stationary=trans_bf, moving=r)
    ni.tensor_copy(dst=result, src=acc)
    ni.dma_copy(dst=out, src=result)
    return out



@nki.jit
def unpack_bytes(codes):
    out = nl.ndarray((128, 128), dtype=nl.uint32, buffer=nl.shared_hbm)
    ids = nl.ndarray((128, 128), dtype=nl.uint32, buffer=nl.sbuf)
    ni.dma_copy(dst=ids, src=codes, dge_mode=ni.dge_mode.swdge)
    ni.dma_copy(dst=out, src=ids)
    return out



@nki.jit
def centroid_lookup(indices, table):
    out = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.shared_hbm)
    ids = nl.ndarray((128, 128), dtype=nl.uint32, buffer=nl.sbuf)
    lut = nl.ndarray(table.shape, dtype=nl.float32, buffer=nl.sbuf)
    values = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.sbuf)
    ni.dma_copy(dst=ids, src=indices)
    ni.dma_copy(dst=lut, src=table)
    ni.nc_n_gather(dst=values, data=lut, indices=ids)
    ni.dma_copy(dst=out, src=values)
    return out



@nki.jit
def restore_scale(unit, scales):
    out = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.shared_hbm)
    u = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.sbuf)
    s = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    z = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.sbuf)
    ni.dma_copy(dst=u, src=unit)
    ni.dma_copy(dst=s, src=scales)
    ni.tensor_scalar(dst=z, data=u, op0=nl.multiply, operand0=s, engine=ni.engine.vector)
    ni.dma_copy(dst=out, src=z)
    return out



unpack_fn = unpack_bytes.func
lookup_fn = centroid_lookup.func
rotate_fn = rotate.func
restore_fn = restore_scale.func


@nki.jit
def decode_tile(codes, scales, table, inverse_rotation):
    indices = unpack_fn(codes)
    rotated = lookup_fn(indices, table)
    unit = rotate_fn(rotated, inverse_rotation)
    return restore_fn(unit, scales)
