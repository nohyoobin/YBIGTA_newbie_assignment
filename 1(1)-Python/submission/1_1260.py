
from __future__ import annotations
import copy
from collections import deque
from collections import defaultdict
from typing import DefaultDict, List

class Graph:
    def __init__(self, n: int) -> None:
        # 정점 개수 저장함
        self.n = n
        self.adj: DefaultDict[int, List[int]] = defaultdict(list)  # 인접 리스트 초기화함

    def add_edge(self, u: int, v: int) -> None:
        # 양방향 간선 추가함
        self.adj[u].append(v)
        self.adj[v].append(u)

    def dfs(self, start: int) -> list[int]:
        # DFS 탐색 결과 저장할 리스트
        result: List[int] = []
        visited: List[bool] = [False] * (self.n + 1)

        def recursive(node: int) -> None:
            visited[node] = True
            result.append(node)
            for neighbor in sorted(self.adj[node]):
                if (not visited[neighbor]):
                    recursive(neighbor)

        recursive(start)
        return result

    def bfs(self, start: int) -> list[int]:
        result: List[int] = []
        visited: List[bool] = [False] * (self.n + 1)
        q: deque[int] = deque()
        q.append(start)
        visited[start] = True

        while q:
            node = q.popleft()
            result.append(node)
            for neighbor in sorted(self.adj[node]):
                if (not visited[neighbor]):
                    visited[neighbor] = True
                    q.append(neighbor)

        return result

    def search_and_print(self, start: int) -> None:
        dfs_result = self.dfs(start)
        bfs_result = self.bfs(start)
        print(' '.join(map(str, dfs_result)))
        print(' '.join(map(str, bfs_result)))


# ----- main.py -----


from typing import Callable
import sys

"""
-아무것도 수정하지 마세요!
"""

def main() -> None:
    intify: Callable[[str], list[int]] = lambda l: [*map(int, l.split())]
    lines: list[str] = sys.stdin.readlines()
    N, M, V = intify(lines[0])
    graph = Graph(N)  # 그래프 생성
    for i in range(1, M + 1):  # 간선 정보 입력
        u, v = intify(lines[i])
        graph.add_edge(u, v)
    graph.search_and_print(V)  # DFS와 BFS 수행 및 출력

if __name__ == "__main__":
    main()
