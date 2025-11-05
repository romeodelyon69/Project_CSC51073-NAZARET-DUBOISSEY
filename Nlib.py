class Buffer:
    def __init__(self, size):
        self.size = size
        self.data = []

    def add(self, value):
        if len(self.data) >= self.size:
            self.data.pop(0)
        self.data.append(value)

    def get_median(self):
        if not self.data:
            return None
        return sorted(self.data)[len(self.data) // 2]

    def get_mean(self):
        if not self.data:
            return None
        return sum(self.data) / len(self.data)
