from __future__ import annotations
from collections import deque

def create_circular_queue(n: int) -> deque[int]:
    """1부터 n까지의 숫자로 deque를 생성함"""
    return deque(range(1, n + 1))

def rotate_and_remove(queue: deque[int], k: int) -> int:
    """
    큐에서 k번째 원소를 제거하고 반환함
    """
    queue.rotate(-(k - 1))  # k-1번 왼쪽으로 회전
    return queue.popleft()  # k번째 요소 제거 후 반환
