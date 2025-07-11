from typing import List, Optional, Tuple, Union
import blosc
import ctypes
import numpy as np
import torch
import umsgpack

EXT_TYPE = 0x69


def torch_dtype_from_str(dtype: Union[bytes, str]) -> Optional[torch.dtype]:
    """
    Checks attribute is a `torch.dtype` for safety.
    """
    if isinstance(dtype, bytes):
        torch_dtype = dtype.decode()
    elif isinstance(dtype, str):
        torch_dtype = dtype
    else:
        raise ValueError(f"Expected `bytes` or `str`, got {type(dtype)}.")
    torch_dtype = getattr(torch, torch_dtype, None)
    if torch_dtype is None:
        return None
    if isinstance(torch_dtype, torch.dtype):
        return torch_dtype
    else:
        return None


def compress(data: bytes, typesize: int) -> bytes:
    """
    Best compression with tensor dtype's itemsize as `typesize`
    `SHUFFLE` is better on average
    `BITSHUFFLE` compresses better in some cases and worse in others
    `zlib` is best trade-off between compression and performance
    `zstd` is best compression but slower
    """
    return blosc.compress(data, typesize, shuffle=blosc.SHUFFLE, cname="zlib")


def decompress(data: bytes) -> bytearray:
    """
    `bytearray` due to "non-writable" warning from `torch.frombuffer`
    """
    return blosc.decompress(data, as_bytearray=True)


def tensor_data(tensor: torch.Tensor) -> Tuple[str, List[int], bytes]:
    total_bytes = tensor.numel() * tensor.dtype.itemsize
    data_ptr = tensor.data_ptr()
    if data_ptr == 0:
        data = b""
    else:
        newptr = ctypes.cast(data_ptr, ctypes.POINTER(ctypes.c_ubyte))
        data = np.ctypeslib.as_array(newptr, (total_bytes,)).tobytes()
        data = compress(data, tensor.dtype.itemsize)
    return str(tensor.dtype).split(".")[-1], list(tensor.shape), data


def from_tensor_data(dtype: bytes, shape: List[int], data: bytes) -> torch.Tensor:
    torch_dtype = torch_dtype_from_str(dtype)
    if torch_dtype is None:
        raise ValueError(f"`torch.dtype` with name `{dtype=}` not found.")
    return torch.frombuffer(decompress(data), dtype=torch_dtype).reshape(shape)


def pack_tensor(tensor: torch.Tensor):
    return umsgpack.Ext(EXT_TYPE, umsgpack.packb(tensor_data(tensor)))


def unpack_tensor(ext: umsgpack.Ext):
    return from_tensor_data(*umsgpack.unpackb(ext.data))


def packb(obj, **options):
    """
    Serialize a Python object into MessagePack bytes.

    Args:
        obj: a Python object

    Keyword Args:
        ext_handlers (dict): dictionary of Ext handlers, mapping a custom type
                             to a callable that packs an instance of the type
                             into an Ext object
        force_float_precision (str): "single" to force packing floats as
                                     IEEE-754 single-precision floats,
                                     "double" to force packing floats as
                                     IEEE-754 double-precision floats

    Returns:
        bytes: Serialized MessagePack bytes

    Raises:
        UnsupportedTypeException(PackException):
            Object type not supported for packing.

    Example:
        >>> umsgpack.packb({u"compact": True, u"schema": 0})
        b'\\x82\\xa7compact\\xc3\\xa6schema\\x00'
    """
    ext_handlers = {torch.Tensor: pack_tensor}
    ext_handlers.update(options.pop("ext_handlers", {}))
    return umsgpack.packb(obj, ext_handlers=ext_handlers, **options)


def unpackb(s, **options):
    """
    Deserialize MessagePack bytes into a Python object.

    Args:
        s (bytes, bytearray): serialized MessagePack bytes

    Keyword Args:
        ext_handlers (dict): dictionary of Ext handlers, mapping integer Ext
                             type to a callable that unpacks an instance of
                             Ext into an object
        use_ordered_dict (bool): unpack maps into OrderedDict, instead of dict
                                 (default False)
        use_tuple (bool): unpacks arrays into tuples, instead of lists (default
                          False)
        allow_invalid_utf8 (bool): unpack invalid strings into instances of
                                   :class:`InvalidString`, for access to the
                                   bytes (default False)

    Returns:
        Python object

    Raises:
        TypeError:
            Packed data type is neither 'bytes' nor 'bytearray'.
        InsufficientDataException(UnpackException):
            Insufficient data to unpack the serialized object.
        InvalidStringException(UnpackException):
            Invalid UTF-8 string encountered during unpacking.
        UnsupportedTimestampException(UnpackException):
            Unsupported timestamp format encountered during unpacking.
        ReservedCodeException(UnpackException):
            Reserved code encountered during unpacking.
        UnhashableKeyException(UnpackException):
            Unhashable key encountered during map unpacking.
            The serialized map cannot be deserialized into a Python dictionary.
        DuplicateKeyException(UnpackException):
            Duplicate key encountered during map unpacking.

    Example:
        >>> umsgpack.unpackb(b'\\x82\\xa7compact\\xc3\\xa6schema\\x00')
        {'compact': True, 'schema': 0}
    """
    ext_handlers = {EXT_TYPE: unpack_tensor}
    ext_handlers.update(options.pop("ext_handlers", {}))
    return umsgpack.unpackb(s, ext_handlers=ext_handlers, **options)


if __name__ == "__main__":
    x = {
        "last_hidden_state": torch.randn(1, 77, 768, dtype=torch.float32),
        "pooler_output": torch.randn(1, 768, dtype=torch.float16),
    }

    packed = packb(x)

    y = unpackb(packed)
