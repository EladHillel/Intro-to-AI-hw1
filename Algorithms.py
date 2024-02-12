import collections

import numpy as np

from DragonBallEnv import DragonBallEnv
from typing import List, Tuple
from collections import deque as deque
import heapdict

class Node():
    def __init__(self, state, papa, papaToSonAction):
        self.state = state
        self.papa = papa
        self.papaToSonAction = papaToSonAction

class BFSAgent():
    def __init__(self) -> None:
        self.open = deque([])
        self.closed = {}
        self.nodes = {}
        self.expanded = 0

    def search(self, env: DragonBallEnv) -> Tuple[List[int], float, int]:
        startState = env.get_initial_state()
        self.open.append(startState)
        while len(self.open):
            currentState = self.open.popleft()
            self.closed.add(currentState)

            for action, successor in env.succ():
                env.set_state(currentState)
                childState = env.step(action)
                if env.is_final_state(childState):
                    # return something
                    pass
                if (not successor[2]) and (childState not in self.closed) and (childState not in self.open):
                    self.open.append(childState)

        # return failure





class WeightedAStarAgent():
    def __init__(self) -> None:
        raise NotImplementedError

    def search(self, env: DragonBallEnv, h_weight) -> Tuple[List[int], float, int]:
        raise NotImplementedError



class AStarEpsilonAgent():
    def __init__(self) -> None:
        raise NotImplementedError
        
    def ssearch(self, env: DragonBallEnv, epsilon: int) -> Tuple[List[int], float, int]:
        raise NotImplementedError