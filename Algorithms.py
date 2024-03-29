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
        ## reset agent:
        self.open = deque([])
        self.closed = set()
        self.nodes = dict() # state: Node(state, prevState, action)
        self.expanded = 0
        ##
        env.reset()
        startState = env.get_state()
        self.open.append(startState)
        self.nodes[startState] = Node(startState, None, None, 0)
        while len(self.open):
            currentState = self.open.popleft()
            self.expanded += 1
            self.closed.add(currentState)
            if currentState[0] in [goal[0] for goal in env.get_goal_states()]:
                continue
            for action, successor in env.succ(currentState).items():
                if successor[0] == None:
                    break
                env.reset()
                env.set_state(currentState)
                childState, cost, _ = env.step(action)
                totalCost = cost + self.nodes[currentState].totalCost
                if env.is_final_state(childState):
                    steps = self.retrace_steps(currentState)
                    steps.append(action)
                    return (steps, totalCost, self.expanded)
                if (childState not in self.closed) and (childState not in self.open):
                    self.open.append(childState)
                    self.nodes[childState] = Node(childState, currentState, action, totalCost)
        return None, None, None
    def retrace_steps(self, finalState: Tuple) -> list[int]:
        actions = []
        while self.nodes[finalState].papaToSonAction != None:
            actions.append(self.nodes[finalState].papaToSonAction)
            finalState = self.nodes[finalState].papa
        actions.reverse()
        return actions


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

    def initHeuristicCalc(self, env, state):
        positionsForHueristic = {goal[0] for goal in env.goals}
        if not state[1]:
            positionsForHueristic.add(env.d1[0])
        if not state[2]:
            positionsForHueristic.add(env.d2[0])
        self.pointsForHeuristic = {(position // env.ncol, position % env.ncol) for position in positionsForHueristic}

    def calc_heuristic(self, env, state):
        self.initHeuristicCalc(env, state)
        point = (state[0] // env.ncol, state[0] % env.ncol)
        distances = {abs(point[0] - otherPoint[0]) + abs(point[1] - otherPoint[1]) for otherPoint in
                     self.pointsForHeuristic}
        return min(distances)

    def calc_fval(self, hVal, gVal):
        return 1.0 * (self.weight * hVal + (1.0 - self.weight) * gVal)


    def search(self, env: DragonBallEnv, h_weight) -> Tuple[List[int], float, int]:
        ## reset agent:
        self.open = heapdict() # key: state, value: (fValue, node)
        self.closed = set()
        self.nodes = dict() # state: Node(state, prevState, action)
        self.expanded = 0
        ##
        self.weight = h_weight
        env.reset()
        startState = env.get_state()
        startHVal = self.calc_heuristic(env, startState)
        startNode = Node(startState, None, None, 0)
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

                if terminated and not env.is_final_state(childState):
                    continue

                newGVal = currentNode.totalCost + cost
                childHVal = self.calc_heuristic(env,childState)
                newFVal = self.calc_fval(childHVal, newGVal)
                newChildNode = Node(childState, currentState, action, newGVal)
                if (childState not in self.open) and (childState not in self.closed):
                    self.open[childState] = (newFVal, childState, newChildNode)
                    self.nodes[childState] = newChildNode
                elif childState in self.open :
                    childExistingFVal, childState, childExistingNode = self.open[childState]
                    if newFVal < childExistingFVal:
                        self.nodes[childState] = newChildNode
                        self.open[childState] = (newFVal,childState, newChildNode)
                else:
                    childExistingNode = self.nodes[childState]
                    childExistingFVal = self.calc_fval(childHVal, childExistingNode.totalCost)
                    if newFVal < childExistingFVal:
                        self.open[childState] = (newFVal,childState, newChildNode)
                        self.nodes[childState] = newChildNode
                        self.closed.remove(childState)
        return None, None, None



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
    def calc_fval(self, hVal, gVal):
        return hVal + gVal

    def initHeuristicCalc(self, env, state):
        positionsForHueristic = {goal[0] for goal in env.goals}
        if not state[1]:
            positionsForHueristic.add(env.d1[0])
        if not state[2]:
            positionsForHueristic.add(env.d2[0])
        self.pointsForHeuristic = {(position // env.ncol, position % env.ncol) for position in positionsForHueristic}

    def calc_heuristic(self, env, state):
        self.initHeuristicCalc(env, state)
        point = (state[0] // env.ncol, state[0] % env.ncol)
        distances = {abs(point[0] - otherPoint[0]) + abs(point[1] - otherPoint[1]) for otherPoint in
                     self.pointsForHeuristic}
        return min(distances)
    def getNextToExpand(self, epsilon):
        minFval = self.open.peekitem()[1][0]
        focal = heapdict((key, self.open[key][2].totalCost) for
                         key, (fval, state, node) in self.open.items() if fval < minFval * (1 + epsilon))
        return focal.peekitem()[0]
    def search(self, env: DragonBallEnv, epsilon: int) -> Tuple[List[int], float, int]:
        ## reset agent:
        self.open = heapdict() # key: state, value: (fValue, node)
        self.closed = set()
        self.nodes = dict() # state: Node(state, prevState, action)
        self.expanded = 0
        ##
        env.reset()
        startState = env.get_state()
        startHVal = self.calc_heuristic(env, startState)
        startNode = Node(startState, None, None, 0)
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

                if terminated and not env.is_final_state(childState):
                    continue

                newGVal = currentNode.totalCost + cost
                childHVal = self.calc_heuristic(env, childState)
                newFVal = self.calc_fval(childHVal, newGVal)
                newChildNode = Node(childState, currentState, action, newGVal)
                if (childState not in self.open) and (childState not in self.closed):
                    self.open[childState] = (newFVal, childState, newChildNode)
                    self.nodes[childState] = newChildNode
                elif childState in self.open:
                    childExistingFVal, childState, childExistingNode = self.open[childState]
                    if newFVal < childExistingFVal:
                        self.nodes[childState] = newChildNode
                        self.open[childState] = (newFVal, childState, newChildNode)
                else:
                    childExistingNode = self.nodes[childState]
                    childExistingFVal = self.calc_fval(childHVal, childExistingNode.totalCost)
                    if newFVal < childExistingFVal:
                        self.open[childState] = (newFVal, childState, newChildNode)
                        self.nodes[childState] = newChildNode
                        self.closed.remove(childState)
        return None, None, None
