"""
Timer utility for profiling SLAM pipeline stages.

Author: Loping151 (MIT License)
Adapted from: https://github.com/Loping151/pytools151
"""
import time


class Timer:
    """Simple timer with lap tracking and interval measurement."""

    def __init__(self, auto_start=True, log_file="time.log", time_func=time.perf_counter):
        self.log_file = log_file
        self.log_csv = log_file.rsplit(".", 1)[0] + ".csv"
        self.time_func = time_func
        self.start_time = None
        self.last_time = None
        self.interval_time = {}
        self.interval_pool = {}

        if auto_start:
            self.start()

    def start(self):
        """Start the timer."""
        self.start_time = self.time_func()
        self.last_time = self.time_func()
        with open(self.log_file, "a") as f:
            f.write(f"New timer start: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    def refresh(self):
        """Reset lap reference without logging."""
        self.last_time = self.time_func()

    def lap(self, tag="", stage=None):
        """Record a lap time."""
        end_time = self.time_func()
        with open(self.log_file, "a") as f:
            f.write(f"Lap: {tag}: {end_time - self.last_time:.6f}s\n")
        self.last_time = end_time

    def start_interval(self, name):
        """Start a named interval."""
        self.interval_time[name] = self.time_func()

    def stop_interval(self, name):
        """Stop a named interval."""
        if name not in self.interval_time or self.interval_time[name] < 0:
            return
        end_time = self.time_func()
        if name not in self.interval_pool:
            self.interval_pool[name] = []
        self.interval_pool[name].append(end_time - self.interval_time[name])
        self.interval_time[name] = -1

    def end(self):
        """Stop the timer and write summary."""
        if self.start_time is None:
            return
        end_time = self.time_func()
        with open(self.log_file, "a") as f:
            f.write(f"Timer end after: {end_time - self.start_time:.6f}s\n")
        # Write CSV summary
        with open(self.log_csv, "a") as f:
            f.write(f"Timer: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("Name, Count, Total/s, Average/s\n")
            for name, times in self.interval_pool.items():
                f.write(f"{name}, {len(times)}, {sum(times):.4f}, {sum(times)/len(times):.6f}\n")
        self.start_time = None
        self.interval_pool = {}
