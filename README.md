# msgpack-tensor

## Usage

```python
import umsgpack
import msgpack_tensor
import torch

x = {
    "last_hidden_state": torch.randn(1, 77, 768, dtype=torch.float32),
    "pooler_output": torch.randn(1, 768, dtype=torch.float16),
}

packed = umsgpack.packb(
    x,
    ext_handlers={torch.Tensor: msgpack_tensor.pack_tensor},
)

y = umsgpack.unpackb(
    packed, ext_handlers={msgpack_tensor.EXT_TYPE: msgpack_tensor.unpack_tensor}
)
```

OR

```python
import msgpack_tensor
import torch

x = {
    "last_hidden_state": torch.randn(1, 77, 768, dtype=torch.float32),
    "pooler_output": torch.randn(1, 768, dtype=torch.float16),
}

packed = msgpack_tensor.packb(x)

y = msgpack_tensor.unpackb(packed)
```
