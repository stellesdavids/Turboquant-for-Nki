"""TurboQuant for NKI.

K/V inputs: FP32 [tokens, heads, 128], tokens a positive multiple of 128.
Cache: UINT8 [2, heads, tokens, 128], FP32 scales [2, heads, tokens, 1].
Each quantization code occupies one byte. QKV/attention are external.
Generated compilation artifacts default to ~/.cache/turboquant_nki.
"""
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn1")
os.environ.setdefault("NEURON_LOGICAL_NC_CONFIG", "1")
os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "0")
os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")

import numpy as np
from nki.framework.compiled import CompileKernel
from nki.runtime import SpikeModel, SpikeTensor
import native_kernels
import kernels


@dataclass
class KVCache:
    codes: np.ndarray
    scales: np.ndarray
    bits: int
    codec_id: str


def ordered_thresholds(thresholds):
    raw = np.asarray(thresholds, np.float32).view(np.int32)
    result = raw ^ ((raw >> 31) & 0x7fffffff)
    return np.where(thresholds == 0, 0, result).astype(np.int32)


class TurboQuantNKI:
    def __init__(self, rotation, thresholds, centroids, *, artifact_dir=None, trace=False):
        self.rotation = np.asarray(rotation, dtype=np.float32)
        self.thresholds = np.asarray(thresholds, dtype=np.float32)
        self.centroids = np.asarray(centroids, dtype=np.float32)
        count = self.centroids.size
        if self.centroids.ndim != 1 or count not in (2,4,8,16,32,64,128,256):
            raise ValueError("centroids must contain 2**b entries, 1 <= b <= 8")
        self.bits = count.bit_length()-1
        if self.thresholds.shape != (count-1,) or not np.all(np.diff(self.thresholds)>0):
            raise ValueError("thresholds must be a strictly increasing vector of 2**b-1 entries")
        if self.rotation.shape != (128,128):
            raise ValueError("rotation must have shape [128,128]")
        if not all(np.isfinite(x).all() for x in (self.rotation,self.thresholds,self.centroids)):
            raise ValueError("rotation, thresholds and centroids must be finite")
        if not np.allclose(self.rotation@self.rotation.T, np.eye(128), atol=2e-3, rtol=2e-3):
            raise ValueError("rotation must be orthogonal: decompression uses its transpose")
        self.codec_id = hashlib.sha256(b"".join(x.tobytes() for x in
            (self.rotation,self.thresholds,self.centroids))).hexdigest()
        self.artifact_root = Path(artifact_dir) if artifact_dir is not None else (
            Path(os.environ.get("XDG_CACHE_HOME", str(Path.home()/".cache")))
            / "turboquant_nki"
        )
        self.models = {}
        self.trace = trace
        self.events = []
        self._compile()
        self.r = self.upload(self.rotation, "rotation")
        self.rt = self.upload(self.rotation.T.copy(), "rotation")
        self.gamma = self.upload(np.tile(self.centroids,(128,1)), "table")

        self.threshold_keys = self.upload(self._key_table(), "threshold_keys")


    @staticmethod
    def upload(a, name):
        return SpikeTensor.from_numpy(np.ascontiguousarray(a), name=name, core_id=0)


    @staticmethod
    def download(tensor, dtype):
        return tensor.numpy().view(dtype)


    def _key_table(self):
        # Last entry is padding. A b-round search over 2**b bins only reads the
        # first 2**b-1 thresholds; no artificial +infinity decision is inserted.
        return np.tile(np.r_[ordered_thresholds(self.thresholds),np.int32(0)],(128,1))


    def _compile(self):
        paths=("native_kernels.py","kernels.py","turboquant.py")
        digest=hashlib.sha256(b"".join((ROOT/p).read_bytes() for p in paths)).hexdigest()[:16]
        self.artifacts=self.artifact_root/digest/f"b{self.bits}"
        self.artifacts.mkdir(parents=True,exist_ok=True)
        specifications={
            "encode_tile":(native_kernels.encode_tile.func,
                           dict(x=np.zeros((128,128),np.float32),rotation=self.rotation,
                                threshold_keys=self._key_table()),dict(bits=self.bits)),
            "decode_tile":(kernels.decode_tile.func,
                           dict(codes=np.zeros((128,128),np.uint8),scales=np.ones((128,1),np.float32),
                                table=np.tile(self.centroids,(128,1)),inverse_rotation=self.rotation.T.copy()),{}),
        }
        for name,(function,data,static) in specifications.items():
            directory=self.artifacts/name
            directory.mkdir(exist_ok=True)
            neff=directory/"kernel.neff"
            if not neff.exists():
                print("compile TurboQuant for NKI",name,flush=True)
                CompileKernel(function,target="trn1",lnc=1,artifacts_dir=str(directory))(**data,**static)
            self.models[name]=SpikeModel.load_from_neff(str(neff),core_id=0)


    def call(self, name, **arguments):
        model = self.models[name]
        binding = {
            n:n.removesuffix(".must_alias_input") for n in model.input_tensors_info}
        inputs = {n:arguments[arg] for n,arg in binding.items()}
        index = len(self.events)
        outputs = model(inputs, save_trace=self.trace,
                        ntff_name=str(self.artifacts/f"{index:05d}_{name}.ntff"))
        # NRT execution is synchronous; return is the stage completion boundary.
        self.events.append(name)
        return [outputs[n] for n in sorted(outputs)]


    def encode_tile(self,tile):
        tile=np.asarray(tile,dtype=np.float32)
        if tile.shape!=(128,128) or not np.isfinite(tile).all():
            raise ValueError("tile must be finite FP32 [128,128]")
        if np.any(np.sum(tile.astype(np.float64)**2,axis=1)>np.finfo(np.float32).max/2):
            raise ValueError("tile L2 reduction would overflow FP32")
        output=self.call("encode_tile",x=self.upload(tile,"x"),rotation=self.r,
                         threshold_keys=self.threshold_keys)
        return (next(t for t in output if t.shape==(128,128)),
                next(t for t in output if t.shape==(128,1)))


    def decode_tile(self,codes,scales):
        return self.call("decode_tile",codes=codes,scales=scales,table=self.gamma,
                         inverse_rotation=self.rt)[0]


    def compress(self, k, v):
        k, v = np.asarray(k,dtype=np.float32), np.asarray(v,dtype=np.float32)
        if k.shape != v.shape or k.ndim != 3 or k.shape[2] != 128:
            raise ValueError("K and V must share shape [tokens,heads,128]")
        tokens, heads, _ = k.shape
        if tokens == 0 or tokens % 128 or heads == 0:
            raise ValueError("tokens must be a positive multiple of 128; heads must be positive")
        codes = np.empty((2,heads,tokens,128),np.uint8)
        scales = np.empty((2,heads,tokens,1),np.float32)
        # QKV is external. Its completion and input availability precede this call.
        for start in range(0,tokens,128):
            for head in range(heads):
                for kind, source in enumerate((k,v)):
                    c,s = self.encode_tile(source[start:start+128,head,:])
                    codes[kind,head,start:start+128,:] = self.download(c,np.uint8)
                    scales[kind,head,start:start+128,:] = self.download(s,np.float32)
        return KVCache(codes,scales,self.bits,self.codec_id)


    def decompress(self, cache):
        if cache.bits != self.bits or cache.codec_id != self.codec_id:
            raise ValueError("cache belongs to a different codec")
        c,s = cache.codes,cache.scales
        if c.dtype != np.uint8 or c.ndim != 4 or c.shape[0] != 2 or c.shape[3] != 128:
            raise ValueError("cache codes must be UINT8 [2,heads,tokens,128]")
        _, heads, tokens, _ = c.shape
        if not heads or not tokens or tokens%128 or s.shape != (2,heads,tokens,1):
            raise ValueError("cache shape is invalid")
        if np.any(c >= 2**self.bits) or s.dtype != np.float32 or not np.isfinite(s).all() or np.any(s<=0):
            raise ValueError("invalid cache codes or scales")
        restored = np.empty((2,tokens,heads,128),np.float32)
        for start in range(0,tokens,128):
            for head in range(heads):
                for kind in range(2):
                    z = self.decode_tile(self.upload(c[kind,head,start:start+128],"codes"),
                                         self.upload(s[kind,head,start:start+128],"scales"))
                    restored[kind,start:start+128,head,:] = self.download(z,np.float32)
        # Every tile is complete on return; caller may now invoke attention.
        return restored[0],restored[1]


    def save_events(self, path):
        Path(path).write_text(json.dumps({"synchronous_stage_order":self.events},indent=2)+"\n")

