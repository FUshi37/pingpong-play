"""Keep cyclic GC pauses outside an armed command stream.

Reference counting remains enabled. Only this dedicated control process's
automatic cyclic collector is deferred; idle collection retains normal memory
reclamation, and the previous collector state is restored on shutdown.
"""
import gc
import time


class DeferredControlGC:
    def __init__(self, idle_period_s=5.0):
        self.was_enabled = gc.isenabled()
        self.idle_period_s = idle_period_s
        self.next_collect = 0.0
        self.collections = 0
        self.last_duration_ms = None
        self.started = False

    def start(self):
        if not self.started:
            gc.disable()
            self.started = True

    def collect_if_idle(self, probe):
        now = time.monotonic()
        if not self.started or now < self.next_collect:
            return False
        if (bool(getattr(probe.runtime_options, 'publish_commands', False))
                or probe.command_stream_active):
            return False
        # The main command consumer and the control tick share this lock.
        # A Send/reset request cannot arm publication during the collection.
        with probe.lock:
            options = probe.runtime_options
            if (bool(getattr(options, 'publish_commands', False))
                    or probe.command_stream_active):
                return False
            start = time.perf_counter()
            gc.collect()
            self.last_duration_ms = (time.perf_counter() - start) * 1000
            self.collections += 1
            self.next_collect = time.monotonic() + self.idle_period_s
        return True

    def payload(self):
        return dict(automaticDeferred=self.started, idleCollections=self.collections,
                    lastIdleCollectionMs=self.last_duration_ms)

    def close(self):
        if self.started and self.was_enabled:
            gc.enable()
        self.started = False
