import pygame
import sys

def handle_pygame_events():
   for event in pygame.event.get():
      if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()


def init_pygame(env):
   pygame.init()
   
   cellSize = 100 - (env.size * 4)

   if cellSize <= 0:
      cellSize = 10

   screen = pygame.display.set_mode((env.size * cellSize, env.size * cellSize))
   pygame.display.set_caption('Reinforcement Learning')

   return screen, cellSize


def quit_pygame():
   pygame.quit()
   sys.exit()