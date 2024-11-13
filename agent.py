import numpy as np
import random
import pygame
from utils import handle_pygame_events
import pickle

class Agent:
   def __init__(self, env, load=False):
      self.episodes = 10000
      self.max_steps = (env.size * env.size)
      self.learning_rate = 0.1
      self.discount_factor = 0.99
      self.epsilon = 1.0
      self.min_epsilon = 0.01
      self.epsilon_decay = 0.001
      
      self.size = env.size
      self.supply_states = env.supply_states
      self.env = env

      if load and self.load_qTable:
         self.qTable = self.load_qTable()
      else:
         self.qTable = np.zeros((self.size, self.size, 2 ** len(self.supply_states), 4))

   def greedy_policy(self, state, collected_supplies):
      present_index = int(''.join(['1' if (i, j) in collected_supplies else '0' for (i, j) in self.supply_states]), 2)

      if random.uniform(0, 1) < self.epsilon:
         return random.randint(0, 3)
      else:
         return np.argmax(self.qTable[state[0]][state[1]][present_index])

   def train_agent(self, screen, cellSize):
      for episode in range(self.episodes):
         state, collected_supplies = self.env.reset()
         done = False
         t = 0

         while not done and t < self.max_steps:
            handle_pygame_events()
            action = self.greedy_policy(state, collected_supplies)
            
            nextState, nextPresents, reward, done, status = self.env.step(action)
            
            present_index = int(''.join(['1' if (i, j) in collected_supplies else '0' for (i, j) in self.env.supply_states]), 2)
            nextpresent_index = int(''.join(['1' if (i, j) in nextPresents else '0' for (i, j) in self.env.supply_states]), 2)
            
            self.qTable[state[0]][state[1]][present_index][action] += self.learning_rate * \
                  (
                     reward + self.discount_factor * np.max(self.qTable[nextState[0]][nextState[1]][nextpresent_index]) - \
                     self.qTable[state[0]][state[1]][present_index][action]
                  )
            
            state, collected_supplies = nextState, nextPresents
            t += 1

         self.epsilon = max(self.min_epsilon, self.epsilon * (1 - self.epsilon_decay))

         if episode % 1000 == 0:
            self.env.render(screen, cellSize)
            pygame.time.wait(350)
      
      self.save_table()

   def test_agent(self, screen, cellSize):
      state, collected_supplies = self.env.reset()
      done = False
      steps = 0

      while not done:
         handle_pygame_events()

         present_index = int(''.join(['1' if (i, j) in collected_supplies else '0' for (i, j) in self.env.supply_states]), 2)
         action = np.argmax(self.qTable[state[0]][state[1]][present_index])

         nextState, nextPresents, reward, done, status = self.env.step(action)

         self.env.render(screen, cellSize)
         pygame.time.wait(500)
         state, collected_supplies = nextState, nextPresents
         
         steps+=1
      
      return status, collected_supplies, steps
   
   def save_table(self):
        with open('table.pkl', 'wb') as f:
            pickle.dump(self.qTable, f)

   def load_qTable(self):      
      try:
         with open('table.pkl', 'rb') as f:
               return pickle.load(f)
      except FileNotFoundError:
         return None