import pygame
import random
import pickle
import numpy as np

class Environment:
    def __init__(self, size=10, number_of_zombies=8, number_of_supplies=5, number_of_rocks=3, load=False):
        self.size = size
        self.grid = np.zeros((size, size), dtype=int) # preenche todo o grid com zeros
        self.initial_state = (0, 0)
        self.goal_state = (size-1, size-1)
        self.number_of_supplies = number_of_supplies

        if load and self.load_grid():
            self.zombie_states, self.supply_states, self.rock_states = self.load_grid()
        else:
            self.zombie_states = self.place_random(number_of_zombies)
            self.supply_states = self.place_random(number_of_supplies, exclude=self.zombie_states)
            self.rock_states = self.place_random(number_of_rocks, exclude=self.zombie_states + self.supply_states)
            self.save_grid()
        
        self.collected_supplies = set()
        
        # seta no grid as respectivas posições. 1 é zombie, 2 é suprimento e 3 é preda        
        for i, j in self.zombie_states:
            self.grid[i][j] = 1
        for i, j in self.supply_states:
            self.grid[i][j] = 2
        for i, j in self.rock_states:
            self.grid[i][j] = 3
    
    def place_random(self, num_items, exclude=[]):
        """
            Randomiza a posição de itens na matriz tratando exceções e posições que são fixas.
            
            exclude [] : posições em que não pode randomizar itens
        """
        items = []
        while len(items) < num_items:
            i, j = random.randint(0, self.size-1), random.randint(0, self.size-1)
            
            if (i, j) not in items and (i, j) not in exclude and (i, j) != self.initial_state and (i, j) != self.goal_state:
                items.append((i, j))
        
        return items
    
    def reset(self):
        self.currentState = self.initial_state
        self.collected_supplies = set()
        return self.currentState, tuple(self.collected_supplies)
    
    def step(self, action):
        status = ""
        i, j = self.currentState
        if action == 0: # CIMA
            i = max(i-1, 0)
        elif action == 1: # BAIXO
            i = min(i+1, self.size-1)
        elif action == 2: # ESQUERDA
            j = max(j-1, 0)
        elif action == 3: # DIREITA
            j = min(j+1, self.size-1)
        
        if (i, j) in self.rock_states:
            i, j = self.currentState
        
        self.currentState = (i, j)
        
        if self.currentState == self.goal_state:
            if len(self.collected_supplies) == len(self.supply_states):
                reward = 10
                done = True
                status = "SAIU"
            else:
                reward = -1
                done = True
                status = "SAIU SEM TODOS PRESENTES"
        elif self.currentState in self.zombie_states:
            reward = -5
            done = True
            status = "MORTO PELO ZUMBI"
        elif self.currentState in self.supply_states and self.currentState not in self.collected_supplies:
            self.collected_supplies.add(self.currentState)
            reward = 2
            done = False
        else:
            reward = -0.1
            done = False
        
        return self.currentState, tuple(self.collected_supplies), reward, done, status

    def render(self, screen, cellSize=60):
        images = {
            "robot": pygame.image.load("./assets/robot.png"),
            "goal": pygame.image.load("./assets/goal.png"),
            "zombie": pygame.image.load("./assets/zombie.png"),
            "supply": pygame.image.load("./assets/supply.png"),
            "wall": pygame.image.load("./assets/rock.png"),
            "empty": pygame.image.load("./assets/empty.png")
        }

        # Redimensionar imagens para o tamanho da célula
        for key in images:
            images[key] = pygame.transform.scale(images[key], (cellSize, cellSize))

        # Limpar a tela com uma imagem de fundo ou cor de fundo
        screen.fill((200, 200, 200))

        for i in range(self.size):
            for j in range(self.size):
                image = images["empty"]
                if (i, j) == self.currentState:
                    image = images["robot"]
                elif (i, j) == self.goal_state:
                    image = images["goal"]
                elif self.grid[i][j] == 1:
                    image = images["zombie"]
                elif self.grid[i][j] == 2:
                    if (i, j) not in self.collected_supplies:
                        image = images["supply"]
                elif self.grid[i][j] == 3:
                    image = images["wall"]

                # Desenhar a imagem na tela
                screen.blit(image, (j * cellSize, i * cellSize))

        pygame.display.flip()

    def save_grid(self):        
        gridData = {
            'zombie_states': self.zombie_states,
            'supply_states': self.supply_states,
            'rock_states': self.rock_states
        }

        with open('grid.pkl', 'wb') as f:
            pickle.dump(gridData, f)

    def load_grid(self):
        try:
            with open('grid.pkl', 'rb') as f:
                gridData = pickle.load(f)
                return gridData['zombie_states'], gridData['supply_states'], gridData['rock_states']
        except FileNotFoundError:
            return None