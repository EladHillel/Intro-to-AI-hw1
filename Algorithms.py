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
        self.openNodes = deque([])
        self.closed = {}
        self.nodes = dict() # state: Node(state, prevState, action)
        self.expanded = 0

    def search(self, env: DragonBallEnv) -> Tuple[List[int], float, int]:
        startState = env.get_initial_state()
        self.open.append(startState)
        startNode = Node((startState), None, None)
        self.nodes[currentState] = startNode
        while len(self.open):
            currentState = self.open.popleft()
            self.expanded += 1
            self.closed.add(currentState)
            for action, successor in env.succ():
                env.set_state(currentState)
                childState = env.step(action)
                if env.is_final_state(childState):
                    steps = retrace_steps(currentState)
                    steps.append(action)
                    cost = calculate_cost(steps)
                    return (steps, cost, self.expanded)
                if (not successor[2]) and (childState not in self.closed) and (childState not in self.open):
                    self.open.append(childState)
                    self.nodes[childState] = Node(childState, currentState, action)

        # return failure
        def retrace_steps(finalState: Tuple) -> list[int]:
            actions = []
            while self.nodes[finalState][2] != None:
                actions.append(self.nodes[finalState][2])
                finalState = self.nodes[finalState][1]
            actions.reverse()
            return actions
        def calculate_cost(actions: list[int]) -> int:
            # set initial and than walk by the actions
            state = env.get_initial_state()
            env.set_state(state)
            cost = 0
            index = 0
            terminated = False
            while terminated != True:
                state, tempCost, terminated = env.step(actions[index])
                index += 1
                cost += tempCost
            return cost




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