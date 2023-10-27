class RunningAverage():
    def __init__(self):
        self.val = 0.
        self.norm = 0

    def update(self, value):
        self.norm += 1
        self.val += (value - self.val) / self.norm

    def __repr__(self) -> str:
        return self.format(self.val)

    def __format__(self, __format_spec: str) -> str:
        return ("{:"+__format_spec+"}").format(self.val)
