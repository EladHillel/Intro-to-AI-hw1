import collections

import numpy as np

from DragonBallEnv import DragonBallEnv
from typing import List, Tuple
from collections import deque as deque
from heapdict import heapdict

class Node():
    def __init__(self, state, papa, papaToSonAction, totalCost):
        self.state = state
        self.papa = papa
        self.papaToSonAction = papaToSonAction
        self.totalCost = totalCost


class BFSAgent():
    def __init__(self) -> None:
        self.open = deque([])
        self.closed = set()
        self.nodes = dict() # state: Node(state, prevState, action)
        self.expanded = 0

    def search(self, env: DragonBallEnv) -> Tuple[List[int], float, int]:
        env.reset()
        startState = env.get_state()
        self.open.append(startState)
        self.nodes[startState] = Node(startState, None, None, 0)
        while len(self.open):
            currentState = self.open.popleft()
            self.expanded += 1
            self.closed.add(currentState)
            for action, successor in env.succ(currentState).items():
                env.reset()
                env.set_state(currentState)
                childState, cost, _ = env.step(action)
                totalCost = cost + self.nodes[currentState].totalCost
                if env.is_final_state(childState):
                    steps = self.retrace_steps(currentState)
                    steps.append(action)
                    return (steps, totalCost, self.expanded)
                if (not successor[2]) and (childState not in self.closed) and (childState not in self.open):
                    self.open.append(childState)
                    self.nodes[childState] = Node(childState, currentState, action, totalCost)
        return Node
    def retrace_steps(self, finalState: Tuple) -> list[int]:
        actions = []
        while self.nodes[finalState].papaToSonAction != None:
            actions.append(self.nodes[finalState].papaToSonAction)
            finalState = self.nodes[finalState].papa
        actions.reverse()
        return actions
    def calculate_cost(self, actions: list[int]) -> int:
        # set initial and than walk by the actions
        state = env.get_initial_state()
        env.reset()
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
        self.open = heapdict() # key: state, value: (fValue, node)
        self.closed = set()
        self.nodes = dict() # state: Node(state, prevState, action)
        self.expanded = 0

    def retrace_steps(self, finalState: Tuple) -> list[int]:
        actions = []
        while self.nodes[finalState].papaToSonAction != None:
            actions.append(self.nodes[finalState].papaToSonAction)
            finalState = self.nodes[finalState].papa
        actions.reverse()
        return actions
    def calc_heuristic(self, state):
        pass
    def calc_fval(self, hVal, gVal):
        return self.weight * hVal + (1-self.weight) * gVal


    def search(self, env: DragonBallEnv, h_weight) -> Tuple[List[int], float, int]:
        self.weight = h_weight
        env.reset()
        startState = env.get_state()
        startHVal = self.calc_heuristic(startState)
        startNode = Node(startState, None, None, 0);
        self.open[startState] = (self.calc_fval(startHVal, 0), startState,  startNode) #secondary comparision by state as required
        self.nodes[startState] = startNode

        while len(self.open):
            (currentFval, currentState , currentNode) = self.open.popitem()[1]
            self.closed.add(currentState)
            if env.is_final_state(currentState):
                steps = self.retrace_steps(currentState)
                return (steps, currentNode.totalCost, self.expanded)

            self.expanded += 1
            for action, successor in env.succ(currentState).items():
                env.reset()
                env.set_state(currentState)
                childState, cost, terminated = env.step(action)

                if (terminated and not env.is_final_state(childState)):
                    continue

                newGVal = currentNode.totalCost + cost
                childHVal = self.calc_heuristic(childState)
                newFVal = self.calc_fval(childHVal, newGVal)
                if (childState not in self.open) and (childState not in self.closed):
                    self.open[childState] = (newFVal, childState, Node(childState, currentState, action, newGVal))
                elif childState in self.open :
                    childExistingFVal, childState, childExistingNode = self.open[childState]
                    if newFVal < childExistingFVal:
                        newChildNode = Node(childState, currentState, action, newGVal)
                        self.open[childState] = (newFVal,childState, newChildNode)
                else:
                    childExistingNode = self.nodes[childState]
                    childExistingFVal = self.calc_fval(childHVal, childExistingNode.totalCost)
                    if newFVal < childExistingFVal:
                        newChildNode = Node(childState, currentState, action, newGVal)
                        self.open[childState] = (newFVal,childState, newChildNode)
                        self.closed.remove(childState)
        return None









class AStarEpsilonAgent():
    def __init__(self) -> None:
        self.open = heapdict() # key: state, value: (fValue, node)
        self.closed = set()
        self.nodes = dict() # state: Node(state, prevState, action)
        self.expanded = 0

    def retrace_steps(self, finalState: Tuple) -> list[int]:
        actions = []
        while self.nodes[finalState].papaToSonAction != None:
            actions.append(self.nodes[finalState].papaToSonAction)
            finalState = self.nodes[finalState].papa
        actions.reverse()
        return actions
    def calc_heuristic(self, state):
        pass
    def calc_fval(self, hVal, gVal):
        return hVal + gVal

    def getNextToExpand(self, epsilon):
        minFval = self.open.peekitem[1][0]
        focal = heapdict((key, self.calc_heuristic(key)) for
                         key, (fval, state, node) in self.open.items() if fval < minFval * (1 + epsilon))
        return focal.peekitem()[0];
def search(self, env: DragonBallEnv, epsilon: int) -> Tuple[List[int], float, int]:
        env.reset()
        startState = env.get_state()
        startHVal = self.calc_heuristic(startState)
        startNode = Node(startState, None, None, 0);
        self.open[startState] = (
        self.calc_fval(startHVal, 0), startState, startNode)  # secondary comparision by state as required
        self.nodes[startState] = startNode

        while len(self.open):
            currentState = self.getNextToExpand(epsilon)
            (currentFval, currentState, currentNode) = self.open[currentState]
            self.open.pop(currentState)
            self.closed.add(currentState)
            if env.is_final_state(currentState):
                steps = self.retrace_steps(currentState)
                return (steps, currentNode.totalCost, self.expanded)

            self.expanded += 1
            for action, successor in env.succ(currentState).items():
                env.reset()
                env.set_state(currentState)
                childState, cost, terminated = env.step(action)

                if (terminated and not env.is_final_state(childState)):
                    continue

                newGVal = currentNode.totalCost + cost
                childHVal = self.calc_heuristic(childState)
                newFVal = self.calc_fval(childHVal, newGVal)
                if (childState not in self.open) and (childState not in self.closed):
                    self.open[childState] = (newFVal, childState, Node(childState, currentState, action, newGVal))
                elif childState in self.open:
                    childExistingFVal, childState, childExistingNode = self.open[childState]
                    if newFVal < childExistingFVal:
                        newChildNode = Node(childState, currentState, action, newGVal)
                        self.open[childState] = (newFVal, childState, newChildNode)
                else:
                    childExistingNode = self.nodes[childState]
                    childExistingFVal = self.calc_fval(childHVal, childExistingNode.totalCost)
                    if newFVal < childExistingFVal:
                        newChildNode = Node(childState, currentState, action, newGVal)
                        self.open[childState] = (newFVal, childState, newChildNode)
                        self.closed.remove(childState)
        return None
    
    
    
    
##### Testing area:
import time
from IPython.display import clear_output



from DragonBallEnv import DragonBallEnv
from typing import List, Tuple
# from Algorithms import *
DOWN = 0
RIGHT = 1
UP = 2
LEFT = 3
MAPS = {
    "4x4": ["SFFF",
            "FDFF",
            "FFFD",
            "FFFG"],
    "8x8": [
        "SFFFFFFF",
        "FFFFFTAL",
        "TFFHFFTF",
        "FFFFFHTF",
        "FAFHFFFF",
        "FHHFFFHF",
        "DFTFHDTL",
        "FLFHFFFG",
    ],
}
def print_solution(actions,env: DragonBallEnv) -> None:
    env.reset()
    total_cost = 0
    print(env.render())
    print(f"Timestep: {1}")
    print(f"State: {env.get_state()}")
    print(f"Action: {None}")
    print(f"Cost: {0}")
    time.sleep(1)

    for i, action in enumerate(actions):
      state, cost, terminated = env.step(action)
      total_cost += cost
      clear_output(wait=True)

      print(env.render())
      print(f"Timestep: {i + 2}")
      print(f"State: {state}")
      print(f"Action: {action}")
      print(f"Cost: {cost}")
      print(f"Total cost: {total_cost}")
      
      time.sleep(1)

      if terminated is True:
        break
    
env = DragonBallEnv(MAPS["8x8"])
BFS_agent = BFSAgent()
actions, total_cost, expanded = BFS_agent.search(env)
print(f"Total_cost: {total_cost}")
print(f"Expanded: {expanded}")
print(f"Actions: {actions}")

assert total_cost == 119.0, "Error in total cost returned"

env = DragonBallEnv(MAPS["8x8"])
wAgent = WeightedAStarAgent()
actions, total_cost, expanded = wAgent.search(env, 0.5)
print(f"Total_cost: {total_cost}")
print(f"Expanded: {expanded}")
print(f"Actions: {actions}")