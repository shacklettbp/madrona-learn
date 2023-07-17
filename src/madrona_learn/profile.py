from contextlib import contextmanager
from time import time
import torch

__all__ = [ "profile" ]

class DummyGPUEvent:
    def __init__(self, enable_timing):
        pass

    def record(self):
        pass

    def elapsed_time(self, e):
        return 0.0

if torch.cuda.is_available():
    GPUTimingEvent = torch.cuda.Event
else:
    GPUTimingEvent = DummyGPUEvent


class TimingData:
    def __init__(self, name):
        self.name = name
        self.cur_sum = 0
        self.cpu_mean = 0
        self.N = 0
        self.children = {}

    def start(self):
        self.cpu_start = time()

    def end(self):
        end = time()

        diff = end - self.cpu_start
        self.cur_sum += diff

    def reset(self):
        self.cur_sum = 0
        self.cpu_mean = 0
        self.N = 0

    def commit(self):
        self.N += 1
        self.cpu_mean += (self.cur_sum - self.cpu_mean) / self.N
        self.cur_sum = 0

    def __repr__(self):
        return f"CPU: {self.cpu_mean:.3f}"


class GPUTimingData(TimingData):
    def __init__(self, name):
        super().__init__(name)

        self.gpu_mean = 0
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
        self.cur_event_idx = 0

    def commit(self):
        super().commit()

        gpu_sum = 0
        for start, end in zip(self.start_events, self.end_events):
            gpu_sum += start.elapsed_time(end) / 1000

        self.gpu_mean += (gpu_sum - self.gpu_mean) / self.N
        self.cur_event_idx = 0

    def __repr__(self):
        return f"CPU: {self.cpu_mean:.3f}, GPU: {self.gpu_mean:.3f}"

class Profiler:
    def __init__(self):
        self.top = {}
        self.parents = []
        self.iter_stack = []
        self.disabled = False

    @contextmanager
    def __call__(self, name, gpu=False):
        if self.disabled:
            try:
                yield
            finally:
                pass
            return

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

    def _iter_timings(self, start, starting_depth, fn):
        for timing in reversed(start.values()):
            self.iter_stack.append((timing, starting_depth))

        while len(self.iter_stack) > 0:
            cur, depth = self.iter_stack.pop()
            fn(cur, depth)
            for child in reversed(cur.children.values()):
                self.iter_stack.append((child, depth + 1))

    def commit(self):
        commit_lambda = lambda x, d: x.commit()

        if len(self.parents) == 0:
            self._iter_timings(self.top, 0, commit_lambda)
        else:
            self._iter_timings(
                self.parents[-1].children, len(self.parents), commit_lambda)

    def reset(self):
        assert(len(self.parents) == 0)
        self._iter_timings(self.top, 0, lambda x, d: x.reset())

    def clear(self):
        assert(len(self.parents) == 0)
        self.top.clear()

    def disable(self):
        self.disabled = True

    def enable(self):
        self.disabled = False

    def report(self, base_indent='    ', depth_indent='  '):
        assert(len(self.parents) == 0)

        def pad(depth):
            return f"{base_indent}{depth_indent * depth}"

        max_len = 0
        def compute_max_len(timing, depth):
            nonlocal max_len

            prefix_len = len(f"{pad(depth)}{timing.name}")
            if prefix_len > max_len:
                max_len = prefix_len

        self._iter_timings(self.top, 0, compute_max_len)

        def print_timing(timing, depth):
            prefix = f"{pad(depth)}{timing.name}"
            right_pad_amount = max_len - len(prefix)

            print(f"{pad(depth)}{timing.name}{' ' * right_pad_amount} => {timing}")

        self._iter_timings(self.top, 0, print_timing)


profile = Profiler()
