import os
import Algorithms
from DragonBallEnv import DragonBallEnv

tests_path = os.path.join(os.getcwd(), "tests")
agentsSTR = ["bfs", "weighted", "epsilon"]
runs_path = os.path.join(os.getcwd(), "runs")
os.makedirs(runs_path, exist_ok=True)

def get_map_and_param(num):
    map_path = os.path.join(tests_path, "map_"+str(num) + ".txt")
    param_path = os.path.join(tests_path, "param_" + str(num)+ ".txt")
    with open(map_path, "r") as file:
        matrix = [line.strip().split() for line in file]
    with open(param_path, "r") as file:
        param = int(file.read())
    return matrix, param

# Example usage



def run_test(num):
    bfs = Algorithms.BFSAgent()
    wa = Algorithms.WeightedAStarAgent()
    ea = Algorithms.AStarEpsilonAgent()
    agents = [bfs, wa, ea]
    map, agentParam = get_map_and_param(num)
    for index, agent in enumerate(agents):
        env = DragonBallEnv(map)
        if isinstance(agent, Algorithms.BFSAgent):
            actions, total_cost, expanded = agent.search(env)
        if isinstance(agent, Algorithms.WeightedAStarAgent):
            actions, total_cost, expanded = agent.search(env, 1 / agentParam)
        if isinstance(agent, Algorithms.AStarEpsilonAgent):
            actions, total_cost, expanded = agent.search(env, agentParam)
        agent_run_path = os.path.join(runs_path, "test_" + str(num) + "_" + agentsSTR[index] + ".txt")
        with open(agent_run_path, "w") as file:
            file.write(f"PARAM: {agentParam}\n")
            file.write(f"Total_cost: {total_cost}\n")
            file.write(f"Expanded: {expanded}\n")
            file.write(f"Actions: {actions}")

if __name__ == "__main__":
    for i in range(50):
        print("run " + str(i))
        run_test(i)
