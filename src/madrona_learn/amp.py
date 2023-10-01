import jax
from jax import lax, random, numpy as jnp

from typing import Optional
from contextlib import contextmanager
from dataclasses import dataclass

__all__ = [ "amp" ]

@dataclass(init=False)
class AMPState:
    device_type: str
    enabled: bool
    compute_dtype: jnp.dtype

    def __init__(self):
        pass

    def init(self, dev, enable_mixed_precision):
        self.device_type = dev.type

        if enable_mixed_precision:
            self.enabled = True

            if dev.type == 'cuda':
                self.compute_dtype = torch.float16
            else:
                self.compute_dtype = torch.bfloat16
        else:
            self.enabled = False
            self.compute_dtype = torch.float32

    @contextmanager
    def enable(self):
        if not self.enabled:
            try:
                yield
            finally:
                pass
        else:
            with torch.autocast(self.device_type, dtype=self.compute_dtype):
                try:
                    yield
                finally:
                    pass

    @contextmanager
    def disable(self):
        if not self.enabled:
            try:
                yield
            finally:
                pass
        else:
            with torch.autocast(self.device_type, enabled=False):
                try:
                    yield
                finally:
                    pass

amp = AMPState() 
