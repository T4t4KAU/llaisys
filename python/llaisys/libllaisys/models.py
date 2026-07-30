import ctypes

from .llaisys_types import llaisysDataType_t, llaisysDeviceType_t


llaisysQwen2Model_t = ctypes.c_void_p


class LlaisysQwen2Config(ctypes.Structure):
    _fields_ = [
        ("vocab_size", ctypes.c_size_t),
        ("hidden_size", ctypes.c_size_t),
        ("intermediate_size", ctypes.c_size_t),
        ("num_hidden_layers", ctypes.c_size_t),
        ("num_attention_heads", ctypes.c_size_t),
        ("num_key_value_heads", ctypes.c_size_t),
        ("rms_norm_eps", ctypes.c_float),
        ("rope_theta", ctypes.c_float),
        ("eos_token_id", ctypes.c_int64),
        ("device_type", llaisysDeviceType_t),
        ("device_id", ctypes.c_int),
    ]


def load_models(lib):
    lib.llaisysQwen2Create.argtypes = [ctypes.POINTER(LlaisysQwen2Config)]
    lib.llaisysQwen2Create.restype = llaisysQwen2Model_t

    lib.llaisysQwen2Destroy.argtypes = [llaisysQwen2Model_t]
    lib.llaisysQwen2Destroy.restype = None

    lib.llaisysQwen2LoadWeight.argtypes = [
        llaisysQwen2Model_t,
        ctypes.c_char_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_size_t,
        llaisysDataType_t,
    ]
    lib.llaisysQwen2LoadWeight.restype = ctypes.c_int

    lib.llaisysQwen2Finalize.argtypes = [llaisysQwen2Model_t]
    lib.llaisysQwen2Finalize.restype = ctypes.c_int

    lib.llaisysQwen2Generate.argtypes = [
        llaisysQwen2Model_t,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_size_t,
    ]
    lib.llaisysQwen2Generate.restype = ctypes.c_size_t
