# Copied and modified from objax repo, original copyright below:


# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import enum
import os
from time import time
from typing import Union, Callable, Tuple, ByteString

import numpy as np
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary, SummaryMetadata
from tensorboard.summary.writer.event_file_writer import EventFileWriter
from tensorboard.util.tensor_util import make_tensor_proto

class SummaryWriter:
    """Writes entries to event files in the logdir to be consumed by Tensorboard."""

    def __init__(self, logdir: str, queue_size: int = 20, write_interval: int = 10):
        """Creates SummaryWriter instance.

        Args:
            logdir: directory where event file will be written.
            queue_size: size of the queue for pending events and summaries
                        before one of the 'add' calls forces a flush to disk.
            write_interval: how often, in seconds, to write the pending events and summaries to disk.
        """
        if not os.path.isdir(logdir):
            os.makedirs(logdir, exist_ok=True)

        # I do not care about tensorflow informational warnings, just let me
        # write my tensorboard file!
        old_min_log_level = os.environ.get('TF_CPP_MIN_LOG_LEVEL', None)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        self.writer = EventFileWriter(logdir, queue_size, write_interval)

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = old_min_log_level

    def scalar(self, tag, scalar, step: int):
        self._add_event(step, Summary.Value(
            tag=tag,
            simple_value=scalar,
        ))

    def text(self, tag, textdata, step: int):
        metadata = SummaryMetadata(
            plugin_data=SummaryMetadata.PluginData(plugin_name='text'))

        self._add_event(step, Summary.Value(
            tag=tag,
            metadata=metadata,
            tensor=make_tensor_proto(
                values=textdata.encode('utf-8'), shape=(1,)),
        ))

    def image(self, tag, image, step: int):
        self._add_event(step, Summary.Value(
            tag=tag,
            value=Summary.Image(
                encoded_image_string=image,
                colorspace=image.shape[0],
                height=image.shape[1],
                width=image.shape[2]),
            ),
        )

    def flush(self):
        self.writer.flush()

    def close():
        self.writer.close()

    def _add_event(self, step, summary_values):
        if isinstance(summary_values, Summary.Value):
            summary_values = [summary_values]

        self.writer.add_event(Event(
            step=step, 
            summary=Summary(value=summary_values),
            wall_time=time(),
        ))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.flush()
