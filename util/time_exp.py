from time import time
import datetime

_tic_toc_last_time = 0
_tic_toc_dict = {}

tic_toc_always_enabled = False


class TicToc:
    def __init__(self, update_interval: int = 1):
        self.start_time = self.last_time = time()
        self.index = 0
        self.recent_runtime = self.avg_runtime = 0.
        self.update_interval = update_interval

    def tic(self):
        self.last_time = time()

    def toc(self):
        self.index += 1
        if self.index != 1 and self.index % self.update_interval == 0:
            last_time = self.last_time
            self.last_time = time()
            self.recent_runtime = (self.last_time - last_time) / self.update_interval
            self.avg_runtime = (self.last_time - self.start_time) / self.index
            return True
        else:
            return False


def tic(msg: str = None, key=None):
    if __debug__ or tic_toc_always_enabled:
        if msg:
            print("{} ({}).".format(msg, datetime.datetime.now().strftime("%I:%M %p on %B %d, %Y")))

        if key is None:
            global _last_time
            _last_time = time()
        else:
            _tic_toc_dict[key] = time()


def toc(msg: str = None, key=None):
    if __debug__ or tic_toc_always_enabled:
        curr_time = time()

        global _last_time
        if key is None:
            last_time = _last_time
            _last_time = curr_time
        else:
            last_time = _tic_toc_dict[key]
            del _tic_toc_dict[key]

        if msg:
            print("{} ({:.5f} secs elapsed).".format(msg, curr_time - last_time))
        else:
            print("{:.5f} secs elapsed.".format(curr_time - last_time))
