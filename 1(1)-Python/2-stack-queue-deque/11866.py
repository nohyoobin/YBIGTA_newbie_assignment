from lib import create_circular_queue, rotate_and_remove

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
