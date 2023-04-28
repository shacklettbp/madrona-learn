from dataclasses import dataclass

@dataclass(frozen=True, kw_only=True)
class TrainConfig:
    num_epochs: int
    lr: float
    steps_per_update: int
    gamma: float

    def __repr__(self):
        rep = "TrainConfig:"

        for k, v in self.__dict__.items():
            rep += f"\n  {k}: {v}" 

        return rep
