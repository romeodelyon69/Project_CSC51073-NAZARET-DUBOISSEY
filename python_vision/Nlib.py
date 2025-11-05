from typing import List, Optional


class Buffer:
    def __init__(self, size: int) -> None:
        self.size: int = size
        self.data: List[float] = []

    def add(self, value: float) -> None:
        if len(self.data) >= self.size:
            self.data.pop(0)
        self.data.append(value)

    def get_median(self) -> Optional[float]:
        if not self.data:
            return None
        return sorted(self.data)[len(self.data) // 2]

    def get_mean(self) -> Optional[float]:
        if not self.data:
            return None
        return sum(self.data) / len(self.data)
