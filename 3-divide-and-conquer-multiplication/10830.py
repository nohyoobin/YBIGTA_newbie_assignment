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

# ----- main.py -----



"""
TODO:
- simulate_card_game 구현하기
    # 카드 게임 시뮬레이션 구현
        # 1. 큐 생성
        # 2. 카드가 1장 남을 때까지 반복
        # 3. 마지막 남은 카드 반환
"""

def simulate_card_game(n: int, k: int) -> list[int]:
    q = create_circular_queue(n)
    result = []
    while q:
        removed = rotate_and_remove(q, k)
        result.append(removed)
    return result

def solve_yosephus() -> None:
    n, k = map(int, input().split())
    result = simulate_card_game(n, k)
    print("<" + ", ".join(map(str, result)) + ">")

if __name__ == "__main__":
    solve_yosephus()