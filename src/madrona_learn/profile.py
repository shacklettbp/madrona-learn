from contextlib import contextmanager
from time import time
import torch

__all__ = [ "profile" ]

class DummyGPUEvent:
    def __init__(self, enable_timing):
        pass

    def record():
        pass

if torch.cuda.is_available():
    GPUTimingEvent = torch.cuda.Event
else:
    GPUTimingEvent = DummyGPUEvent


class TimingData:
    def __init__(self):
        self.cpu_mean = 0
        self.N = 0

    def start(self):
        self.cpu_start = time()

    def end(self):
        end = time()

        self.N += 1
        diff = end - self.cpu_start
        self.cpu_mean += (diff - self.cpu_mean) / self.N

    def reset(self):
        self.cpu_mean = 0
        self.N = 0

    def commit(self):
        pass

    def __repr__(self):
        return f"{self.cpu_mean:.3f}"


class GPUTimingData(TimingData):
    def __init__(self):
        super().__init__()

        self.gpu_mean = 0
        self.gpu_N = 0
        self.cur_event_idx = 0
        self.start_events = []
        self.end_events = []

    def start(self):
        super().start()

        if self.cur_event_idx >= len(self.start_events):
            self.start_events.append(GPUTimingEvent(enable_timing=True))
            self.end_events.append(GPUTimingEvent(enable_timing=True))

        self.start_events[self.cur_event_idx].record()

    def end(self):
        super().end()
        self.end_events[self.cur_event_idx].record()
        self.cur_event_idx += 1

    def reset(self):
        super().reset()
        self.gpu_mean = 0
        self.gpu_N = 0
        self.cur_event_idx = 0

    def commit(self):
        super().commit()
        for start, end in zip(self.start_events, self.end_events):
            self.gpu_N += 1
            diff = start.elapsed_time(end) / 1000
            self.gpu_mean += (diff - self.gpu_mean) / self.gpu_N

        cur_event_idx = 0

    def __repr__(self):
        return f"CPU: {super().__repr__()}, GPU: {self.gpu_mean:.3f}"

class Profiler:
    def __init__(self):
        self.timings = {}

    @contextmanager
    def __call__(self, name, gpu=False):
        try:
            timing_data = self.timings[name]
        except KeyError:
            if gpu:
                timing_data = GPUTimingData()
            else:
                timing_data = TimingData()
            self.timings[name] = timing_data

        try:
            timing_data.start()
            yield
        finally:
            timing_data.end()

    def commit(self):
        for timing in self.timings.values():
            timing.commit()

    def reset(self):
        for timing in self.timings.values():
            timing.reset()

    def clear(self):
        self.timings.clear()

    def report(self, indent='    '):
        print(f"{indent}Timings:")
        for name, timing in self.timings.items():
            print(f"{indent * 2}{name}: {timing}")


profile = Profiler()
