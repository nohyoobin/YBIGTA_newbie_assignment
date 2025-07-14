from lib import create_circular_queue

"""
TODO:
- simulate_card_game 구현하기
    # 카드 게임 시뮬레이션 구현
        # 1. 큐 생성
        # 2. 카드가 1장 남을 때까지 반복
        # 3. 마지막 남은 카드 반환
"""

def simulate_card_game(n: int) -> int:
    # 1부터 n까지 숫자가 들어간 큐 생성함
    q = create_circular_queue(n)
    while (len(q) > 1):
        q.popleft()  # 맨 위 카드 버림
        q.append(q.popleft())  # 다음 카드를 맨 아래로 옮김
    return q[0]

def solve_card2() -> None:
    n: int = int(input())
    result: int = simulate_card_game(n)
    print(result)

if __name__ == "__main__":
    solve_card2()
