from typing import List


class Buffer:
    def __init__(self, size: int) -> None:
        self.size: int = size
        self.data: List[int] = []

    def add(self, value: int) -> None:
        if len(self.data) >= self.size:
            self.data.pop(0)
        self.data.append(value)

    def get_median(self) -> int:
        if not self.data:
            raise ValueError("Buffer is empty")
        return sorted(self.data)[len(self.data) // 2]

    def get_mean(self) -> float:
        if not self.data:
            raise ValueError("Buffer is empty")
        return sum(self.data) / len(self.data)
