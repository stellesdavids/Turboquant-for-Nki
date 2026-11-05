"""TurboQuant for NKI: native GpSIMD Lloyd-Max encoding.

Ordered FP32 integer keys preserve comparisons, with signed zeros canonicalized.
Sign-aware selection handles subtraction overflow. VectorE computes ordered-key
representations and binary-search midpoints; GpSIMD looks up thresholds,
compares keys, selects indices, and generates DMA byte-copy descriptors.
"""
import nki
import nki.language as nl
import nki.isa as ni
import kernels as tile_ops

normalize_fn=tile_ops.normalize.func
rotate_fn=tile_ops.rotate.func


def range_flag(data, subtract):
    small=nl.ndarray((128,128),dtype=nl.uint8,buffer=nl.sbuf)
    result=nl.ndarray((128,128),dtype=nl.int32,buffer=nl.sbuf)
    # The NISA scalar field is signed i32; -2147483648 supplies the UINT32
    # bit pattern 0x80000000 required by the unsigned address range check.
    ni.tensor_scalar(dst=small,data=data,op0=nl.subtract,operand0=subtract,
                     op1=nl.less,operand1=-2147483648,engine=ni.engine.gpsimd)
    ni.tensor_copy(dst=result,src=small,engine=ni.engine.gpsimd)
    return result


def ordered_keys(values):
    raw=values.ap(pattern=[[128,128],[1,128]],dtype=nl.int32)
    sign=nl.ndarray((128,128),dtype=nl.int32,buffer=nl.sbuf)
    negative_sign=nl.ndarray((128,128),dtype=nl.int32,buffer=nl.sbuf)
    flip=nl.ndarray((128,128),dtype=nl.int32,buffer=nl.sbuf)
    key=nl.ndarray((128,128),dtype=nl.int32,buffer=nl.sbuf)
    magnitude=nl.ndarray((128,128),dtype=nl.int32,buffer=nl.sbuf)
    result=nl.ndarray((128,128),dtype=nl.int32,buffer=nl.sbuf)
    ni.tensor_scalar(dst=sign,data=raw,op0=nl.right_shift,operand0=31,engine=ni.engine.vector)
    # NKI right_shift is logical even for int32: expand its 0/1 result to 0/-1.
    ni.tensor_scalar(dst=negative_sign,data=sign,op0=nl.multiply,operand0=-1,engine=ni.engine.vector)
    ni.tensor_scalar(dst=flip,data=negative_sign,op0=nl.bitwise_and,operand0=2147483647,engine=ni.engine.vector)
    ni.tensor_tensor(dst=key,data1=raw,data2=flip,op=nl.bitwise_xor,engine=ni.engine.vector)
    ni.tensor_scalar(dst=magnitude,data=raw,op0=nl.bitwise_and,operand0=2147483647,engine=ni.engine.vector)
    nonzero=range_flag(magnitude.ap(pattern=[[128,128],[1,128]],dtype=nl.uint32),1)
    ni.tensor_tensor(dst=result,data1=key,data2=nonzero,op=nl.multiply,engine=ni.engine.gpsimd)
    return result


def signed_ge(a,b,a_positive,one):
    diff=nl.ndarray((128,128),dtype=nl.int32,buffer=nl.sbuf)
    ni.tensor_tensor(dst=diff,data1=a,data2=b,op=nl.subtract,engine=ni.engine.gpsimd)
    nonnegative=range_flag(diff.ap(pattern=[[128,128],[1,128]],dtype=nl.uint32),0)
    b_positive=range_flag(b.ap(pattern=[[128,128],[1,128]],dtype=nl.uint32),0)
    sign_delta=nl.ndarray((128,128),dtype=nl.int32,buffer=nl.sbuf)
    sign_square=nl.ndarray((128,128),dtype=nl.int32,buffer=nl.sbuf)
    same_sign=nl.ndarray((128,128),dtype=nl.int32,buffer=nl.sbuf)
    b_negative=nl.ndarray((128,128),dtype=nl.int32,buffer=nl.sbuf)
    cross_positive=nl.ndarray((128,128),dtype=nl.int32,buffer=nl.sbuf)
    within_sign=nl.ndarray((128,128),dtype=nl.int32,buffer=nl.sbuf)
    result=nl.ndarray((128,128),dtype=nl.int32,buffer=nl.sbuf)
    ni.tensor_tensor(dst=sign_delta,data1=a_positive,data2=b_positive,op=nl.subtract,engine=ni.engine.gpsimd)
    ni.tensor_tensor(dst=sign_square,data1=sign_delta,data2=sign_delta,op=nl.multiply,engine=ni.engine.gpsimd)
    ni.tensor_tensor(dst=same_sign,data1=one,data2=sign_square,op=nl.subtract,engine=ni.engine.gpsimd)
    ni.tensor_tensor(dst=b_negative,data1=one,data2=b_positive,op=nl.subtract,engine=ni.engine.gpsimd)
    ni.tensor_tensor(dst=cross_positive,data1=a_positive,data2=b_negative,op=nl.multiply,engine=ni.engine.gpsimd)
    ni.tensor_tensor(dst=within_sign,data1=same_sign,data2=nonnegative,op=nl.multiply,engine=ni.engine.gpsimd)
    ni.tensor_tensor(dst=result,data1=within_sign,data2=cross_positive,op=nl.add,engine=ni.engine.gpsimd)
    return result


def search(values,threshold_keys,bits):
    keys=ordered_keys(values)
    positive=range_flag(keys.ap(pattern=[[128,128],[1,128]],dtype=nl.uint32),0)
    lut=nl.ndarray(threshold_keys.shape,dtype=nl.int32,buffer=nl.sbuf)
    ni.dma_copy(dst=lut,src=threshold_keys)
    low=nl.ndarray((128,128),dtype=nl.int32,buffer=nl.sbuf)
    high=nl.ndarray((128,128),dtype=nl.int32,buffer=nl.sbuf)
    one=nl.ndarray((128,128),dtype=nl.int32,buffer=nl.sbuf)
    ni.memset(dst=low,value=0,engine=ni.engine.gpsimd)
    ni.memset(dst=high,value=2**bits-1,engine=ni.engine.gpsimd)
    ni.memset(dst=one,value=1,engine=ni.engine.gpsimd)
    for step in nl.static_range(bits):
        total=nl.ndarray((128,128),dtype=nl.int32,buffer=nl.sbuf)
        mid=nl.ndarray((128,128),dtype=nl.int32,buffer=nl.sbuf)
        threshold=nl.ndarray((128,128),dtype=nl.int32,buffer=nl.sbuf)
        ni.tensor_tensor(dst=total,data1=low,data2=high,op=nl.add,engine=ni.engine.vector)
        ni.tensor_scalar(dst=mid,data=total,op0=nl.right_shift,operand0=1,engine=ni.engine.vector)
        ni.nc_n_gather(dst=threshold,data=lut,indices=mid.ap(pattern=[[128,128],[1,128]],dtype=nl.uint32))
        predicate=signed_ge(keys,threshold,positive,one)
        next_mid=nl.ndarray((128,128),dtype=nl.int32,buffer=nl.sbuf)
        low_delta=nl.ndarray((128,128),dtype=nl.int32,buffer=nl.sbuf)
        low_step=nl.ndarray((128,128),dtype=nl.int32,buffer=nl.sbuf)
        high_delta=nl.ndarray((128,128),dtype=nl.int32,buffer=nl.sbuf)
        high_step=nl.ndarray((128,128),dtype=nl.int32,buffer=nl.sbuf)
        new_low=nl.ndarray((128,128),dtype=nl.int32,buffer=nl.sbuf)
        new_high=nl.ndarray((128,128),dtype=nl.int32,buffer=nl.sbuf)
        ni.tensor_tensor(dst=next_mid,data1=mid,data2=one,op=nl.add,engine=ni.engine.gpsimd)
        ni.tensor_tensor(dst=low_delta,data1=next_mid,data2=low,op=nl.subtract,engine=ni.engine.gpsimd)
        ni.tensor_tensor(dst=low_step,data1=low_delta,data2=predicate,op=nl.multiply,engine=ni.engine.gpsimd)
        ni.tensor_tensor(dst=new_low,data1=low,data2=low_step,op=nl.add,engine=ni.engine.gpsimd)
        ni.tensor_tensor(dst=high_delta,data1=high,data2=mid,op=nl.subtract,engine=ni.engine.gpsimd)
        ni.tensor_tensor(dst=high_step,data1=high_delta,data2=predicate,op=nl.multiply,engine=ni.engine.gpsimd)
        ni.tensor_tensor(dst=new_high,data1=mid,data2=high_step,op=nl.add,engine=ni.engine.gpsimd)
        low,high=new_low,new_high
    return low


@nki.jit
def search_only(values,threshold_keys,bits: int):
    a=nl.ndarray((128,128),dtype=nl.float32,buffer=nl.sbuf)
    ni.dma_copy(dst=a,src=values)
    low=search(a,threshold_keys,bits)
    out=nl.ndarray((128,128),dtype=nl.uint8,buffer=nl.shared_hbm)
    ni.dma_copy(dst=out,src=low.ap(pattern=[[512,128],[4,128]],dtype=nl.uint8),dge_mode=ni.dge_mode.swdge)
    return out


@nki.jit
def encode_tile(x,rotation,threshold_keys,bits: int):
    unit,scales=normalize_fn(x)
    rotated=rotate_fn(unit,rotation)
    values=nl.ndarray((128,128),dtype=nl.float32,buffer=nl.sbuf)
    ni.dma_copy(dst=values,src=rotated)
    low=search(values,threshold_keys,bits)
    codes=nl.ndarray((128,128),dtype=nl.uint8,buffer=nl.shared_hbm)
    ni.dma_copy(dst=codes,src=low.ap(pattern=[[512,128],[4,128]],dtype=nl.uint8),
                dge_mode=ni.dge_mode.swdge)
    return codes,scales
