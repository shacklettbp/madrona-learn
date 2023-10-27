from contextlib import contextmanager
import jax

__all__ = [ "profile" ]

class Profiler:
    def __init__(self):
        self.disabled = False

    @contextmanager
    def __call__(self, name):
        if self.disabled:
            try:
                yield
            finally:
                pass
            return

        try:
            with jax.named_scope(name), jax.profiler.TraceAnnotation(name):
                yield
        finally:
            pass

    def disable(self):
        self.disabled = True

    def enable(self):
        self.disabled = False


profile = Profiler()
