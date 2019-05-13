from time import time


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
