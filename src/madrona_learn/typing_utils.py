from dataclasses import dataclass
from typing import Protocol, runtime_checkable

@runtime_checkable
@dataclass
class DataclassProtocol(Protocol):
    pass
