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
    def __init__(self, name):
        self.name = name
        self.cpu_mean = 0
        self.N = 0
        self.children = {}

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
        return f"{self.name} => CPU: {self.cpu_mean:.3f}"


class GPUTimingData(TimingData):
    def __init__(self, name):
        super().__init__(name)

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
        return f"{self.name} => CPU: {self.cpu_mean:.3f}, GPU: {self.gpu_mean:.3f}"

class Profiler:
    def __init__(self):
        self.top = {}
        self.parents = []
        self.iter_stack = []

    @contextmanager
    def __call__(self, name, gpu=False):
        if len(self.parents) > 0:
            cur_timings = self.parents[-1].children
        else:
            cur_timings = self.top

        try:
            timing_data = cur_timings[name]
        except KeyError:
            if gpu:
                timing_data = GPUTimingData(name)
            else:
                timing_data = TimingData(name)
            cur_timings[name] = timing_data

        self.parents.append(timing_data)

        try:
            timing_data.start()
            yield
        finally:
            timing_data.end()

        self.parents.pop()

    def _iter_timings(self, fn):
        for timing in self.top.values():
            self.iter_stack.append((timing, 0))

        while len(self.iter_stack) > 0:
            cur, depth = self.iter_stack.pop()
            fn(cur, depth)
            for child in cur.children.values():
                self.iter_stack.append((child, depth + 1))

    def commit(self):
        assert(len(self.parents) == 0)
        self._iter_timings(lambda x, d: x.commit())

    def reset(self):
        self._iter_timings(lambda x, d: x.reset())

    def clear(self):
        assert(len(self.parents) == 0)
        self.top.clear()

    def report(self, base_indent='    ', depth_indent='  '):
        print(f"{base_indent}Timings:")
        def print_timing(timing, depth):
            print(f"{base_indent}{depth_indent * depth}{timing}")

        self._iter_timings(print_timing)


profile = Profiler()
