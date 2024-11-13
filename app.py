from utils import init_pygame, quit_pygame
from environment import Environment
from agent import Agent

if __name__ == '__main__':
    size = 10
    zombies = 8
    supplies = 5
    obstacles = 8

    load_grid = False
    load_QTable = False

    env = Environment(size, zombies, supplies, obstacles, load_grid)
    agent = Agent(env, load_QTable)

    screen, cellSize = init_pygame(env)

    agent.train_agent(screen, cellSize)

    status, collected_supplies, steps = agent.test_agent(screen, cellSize)

    quit_pygame()