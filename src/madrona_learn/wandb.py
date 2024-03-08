import wandb
from madrona_learn.tensorboard import SummaryWriter
from time import time
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary

# Make a class that extends summarywriter to overwrite _add_event to include wandb.log, and overwrite __init__ to include wandb.init and args
class WandbWriter(SummaryWriter):
    def __init__(self, logdir, queue_size=20, write_interval=10, args=None):
        super().__init__(logdir, queue_size, write_interval)
        wandb.init(
            project="puzzle_bench_jax",
            entity=None,
            sync_tensorboard=True,
            config=vars(args),
            name=args.run_name,
            save_code=True,
        )
    def _add_event(self, step, summary_values):
        super()._add_event(step, summary_values)

        wandb.log({
            summary_values.tag: summary_values.simple_value,
            "update_step": step,
        })