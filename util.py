class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
