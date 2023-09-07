from dataclasses import dataclass
from typing import Protocol

@runtime_checkable
@dataclasses.dataclass
class DataclassProtocol(Protocol):
    pass
