from DragonBallEnv import DragonBallEnv
from typing import List, Tuple
import Algorithms
import random
import os

def randomTuple():
    while True:
        p = (random.randint(0, 7), random.randint(0, 7))
        if p != (0, 0):
            return p


def generateMap(maxNumOfHoles, maxNumOfGoals):
    map = [['' for _ in range(8)] for _ in range(8)]
    regularLetters = "FTAL"

    for i in range(8):
        for j in range(8):
                map[i][j] = random.choice(regularLetters)

    for i in range(maxNumOfHoles):
        p = randomTuple()
        map[p[0]][p[1]] = 'H'

    for i in range(maxNumOfGoals):
        p = randomTuple()
        map[p[0]][p[1]] = 'G'

    for i in range(2):
        while(True):
            p = randomTuple();
            if map[p[0]][p[1]] != 'G' and map[p[0]][p[1]] !='D':
                map[p[0]][p[1]] = 'D'
                break
    map[0][0] = 'S'
    return map


file_path = os.path.join(os.getcwd(), "tests")
os.makedirs(file_path, exist_ok=True)
os.chdir(file_path)


def run_print(testNum, subNum, agent, map, agentParam):
    env = DragonBallEnv(map)
    if isinstance(agent, Algorithms.BFSAgent):
        actions, total_cost, expanded = agent.search(env)
    if isinstance(agent, Algorithms.WeightedAStarAgent):
        actions, total_cost, expanded = agent.search(env, 1/agentParam)
    if isinstance(agent, Algorithms.AStarEpsilonAgent):
        actions, total_cost, expanded = agent.search(env, agentParam)

    with open("test_" + str(testNum) + "_" + agentsSTR[subNum] + ".txt", "w") as file:
            file.write(f"PARAM: {agentParam}\n")
            file.write(f"Total_cost: {total_cost}\n")
            file.write(f"Expanded: {expanded}\n")
            file.write(f"Actions: {actions}")

agentsSTR = ["bfs", "weighted", "epsilon"]

def matrix_to_text(matrix):
    text = ""
    for row in matrix:
        text += " ".join(map(str, row)) + "\n"
    return text

def create_test(testNum, maxNumHoles, maxNumGoals, agentParam):
    bfs = Algorithms.BFSAgent()
    wa = Algorithms.WeightedAStarAgent()
    ea = Algorithms.AStarEpsilonAgent()
    agents = [bfs, wa, ea]
    map = generateMap(maxNumHoles, maxNumGoals)
    with open("map_" + str(testNum) + ".txt" , "w") as file:
        file.write(matrix_to_text(map))
    with open("param_" + str(testNum) + ".txt", "w") as file:
        file.write(str(agentParam))
    for index, agent in enumerate(agents):
        run_print(testNum, index ,agent, map, agentParam)



def generate():
    for i in range(50):
        print("gen " + str(i))
        create_test(i, random.randint(0, 20), random.randint(1, 5), random.randint(1, 10))


if __name__ == "__main__":
    generate()
