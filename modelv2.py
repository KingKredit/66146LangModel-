import mesa 
import numpy as np 
import pandas as pd 



import random 
import networkx as nx 
import matplotlib.pyplot as plt
from mesa.discrete_space import CellAgent, OrthogonalMooreGrid



from mesa import Agent
class LanguageAgent(CellAgent):
    def __init__(self, model, cell, language):
        super().__init__(model)
        self.cell = cell
        self.pos = cell.coordinate
        self.language = language
        # self.random = model.random


    @classmethod
    def create_agents(cls, model, num_agents, cells, languages):
        """Mesa-3 style helper, similar to MoneyAgent in the Adding Space tutorial."""
        agents = []
        for i in range(num_agents):
            cell = cells[i]
            language = languages[i]

            agent = cls(model, cell, language)
            # attach to cell
            cell.add_agent(agent)
            # register with the model's AgentSet (see Model.register_agent in API) 
            model.register_agent(agent)
            agents.append(agent)
        return agents

    moves = {
        "U": (0, 1),
        "D": (0, -1),
        "L": (-1, 0),
        "R": (1, 0),
        "S": (0, 0),
    }

    def distance(x0, x1, y0, y1):
        return ((x1-x0)**2 + (y1 -y0)**2)**0.5

    def getValidMoves(self):
        validMoves = []
        for move, (dx, dy) in self.moves.items():
            newX = self.pos[0] + dx
            newY = self.pos[1] + dy
            if not (0 <= newX < self.model.grid.width and 0 <= newY < self.model.grid.height):
                # validMoves.append((newX, newY))
                continue

            targetPos = (newX, newY)
            if targetPos == self.pos:
                validMoves.append(targetPos)
                continue
                    

            targetCell = None
            for cell in self.model.grid.all_cells.cells:
                if cell.coordinate == targetPos:
                    targetCell = cell
                    break

            if targetCell is None:
                continue  # shouldn't happen, but be safe

            # only allow move if the cell is empty (no agents there)
            if len(targetCell.agents) == 0:
                validMoves.append(targetPos)
                
        return validMoves
    

    def getInteractableNeighbours(self):
    # Neighbor cells around our current cell
        neighbor_cells = self.cell.get_neighborhood(
            radius=self.model.interactionRadius,
            include_center=False,
        )

        interactableNeighbours = []
        P = self.model.P

        # Each neighbor cell can contain multiple agents
        for nbr_cell in neighbor_cells:
            for neighbor in nbr_cell.agents:
                if P[self.language, neighbor.language] >= self.model.probThreshold:
                    interactableNeighbours.append(neighbor)
        print(interactableNeighbours)
        return interactableNeighbours

    
    def binaryMoveHelper(self, pos1, pos2):
        x1, y1 = pos1
        x2, y2 = pos2

        dx = int(np.sign(x2 - x1))
        dy = int(np.sign(y2 - y1))

        if dx == 0 and dy == 0:
            return "S"
        elif abs(dx) >= abs(dy):
            return "R" if dx > 0 else "L"
        else:
            return "U" if dy > 0 else "D"
        

    def moveHelper(self):
        interactableNeighbours = self.getInteractableNeighbours()
        eps = 1e-6
        scores = {d: eps for d in self.moves}
        for neighbor in interactableNeighbours:
            direction = self.binaryMoveHelper(self.pos, neighbor.pos)
            weight = self.model.P[self.language, neighbor.language]
            scores[direction] += weight

        directions = list(scores.keys())   
        weights    = list(scores.values()) 
        if not interactableNeighbours:
            return self.pos
        else:
            chosen_dir = self.random.choices(directions, weights=weights, k=1)[0]
            direction = self.moves[chosen_dir]
            dx, dy = direction
            return (self.pos[0] + dx, self.pos[1] + dy)
        

        
    def move(self):
        validMoves = self.getValidMoves()
        newPos = self.moveHelper()

        # stay if invalid or no move
        if newPos not in validMoves or newPos == self.pos:
            return

        # find the target cell with that coordinate
        target_cell = None
        for cell in self.model.grid.all_cells.cells:
            if cell.coordinate == newPos:
                target_cell = cell
                break
        if target_cell is None:
            return  # shouldn't happen, but be safe

        # update occupancy
        self.cell.remove_agent(self)
        target_cell.add_agent(self)

        # update our references
        self.cell = target_cell
        self.pos = newPos
        print(self.pos)




    def mutate(self):
        neighbours = self.getInteractableNeighbours()
        if not neighbours:
            return
        else:
            interactAgent = self.random.choice(neighbours)
            P = self.model.P
            p = P[self.language, interactAgent.language]
            u = self.random.random()
            if p > u:
                if self.random.random() < self.model.mutationProb:
                    newLang = self.model.createLanguageMutuation(self, interactAgent)
                    self.language = newLang
                    interactAgent.language = newLang
                    print(newLang)
                else:
                    self.language = interactAgent.language

    def step(self):
    # Use Mesa's AgentSet API
        self.move()
        self.mutate()



from mesa import Model
from mesa.space import MultiGrid
class LanguageModel(Model):
    def __init__(
        self,
        width=20,
        height=20,
        numAgents=100,
        P=None,
        r=1,
        probThresh=0.1,
        mutationProb=0.1,
        languageNum=2
    ):
        super().__init__()
        self.width = width
        self.height = height 
        self.numAgents = numAgents
        self.interactionRadius = r 
        self.probThreshold = probThresh
        self.mutationProb = mutationProb
        self.languageNum = languageNum
        self.languageNetwork = None


        



        if P is None:
            P = np.zeros((languageNum, languageNum))
            for i in range(languageNum):
                stay_prob = np.random.uniform(0.7, 0.9)  
                mutate_prob = 1 - stay_prob
                if i < languageNum - 1:
                    P[i, i] = stay_prob
                    P[i, i + 1] = mutate_prob
                else:
                    P[i, i] = 1.0 
        P = np.round(P, 3)
        self.P = P

        self.grid = OrthogonalMooreGrid(
                (width, height),
                torus=False,          # or True if you want wrapping
                capacity=None,        # or an int if you want cell capacity limits
                random=self.random
            )

        self._create_agents()
        self.languageNetwork = self.createNetwork()


    def _create_agents(self):
        cells = self.random.choices(self.grid.all_cells.cells, k=self.numAgents)
        languages = self.random.choices(range(self.languageNum), k=self.numAgents)
        LanguageAgent.create_agents(self, self.numAgents, cells, languages)


    def createNetwork(self):
            graph = nx.Graph()
            for i in range(self.languageNum):
                graph.add_node(i)

            for j in range(self.languageNum):
                for k in range(self.languageNum):
                    if self.P[j, k] >= self.probThreshold:
                        graph.add_edge(j, k, weight=self.P[j, k])

            return graph


    
    def updateLanguageNetwork(self, agent1, agent2):
        lang1 = agent1.language
        lang2 = agent2.language
        mutationOrigin = np.random.choice([lang1, lang2])
        network = self.languageNetwork
        network.add_node((lang1, lang2))
        new_node = (lang1, lang2)
        if new_node not in network:
            network.add_node(new_node, origin=mutationOrigin)

            for target_lang, prob in enumerate(self.P[mutationOrigin]):
                if prob > 0:
                    network.add_edge(new_node, target_lang, weight=prob)
        self.languageNetwork = network
        pass

    def createLanguageMutuation(self, agent1, agent2):
        lang1 = agent1.language
        lang2 = agent2.language
        oldP = self.P
        nOld = self.languageNum
        mutation_origin = np.random.choice([lang1, lang2])
        newP = np.zeros((self.languageNum + 1, self.languageNum + 1))
        for i in range(self.languageNum):
            for j in range(self.languageNum):
                newP[i, j] = oldP[i, j]

        newId = nOld
        for j in range(nOld):
            newP[newId, j] = oldP[mutation_origin, j]  
            newP[j, newId] = oldP[j, mutation_origin]

        newP[newId, newId] = oldP[mutation_origin, mutation_origin] * 0.9  
        self.P = newP
        self.languageNum += 1
        self.updateLanguageNetwork(agent1, agent2)
        return newId
    

    
    def step(self):
    # Use Mesa's AgentSet: shuffle and call .step() on each agent
        self.agents.shuffle_do("step")


    def debug_cell_occupancy(self):
        total_agents = len(self.agents)
        multi_cells = 0

        for cell in self.grid.all_cells.cells:
            if len(cell.agents) > 1:
                multi_cells += 1

        print(f"Total agents: {total_agents}, cells with >1 agent: {multi_cells}")

    

    
    def validate_state(self):
        # Check transition matrix and language count
        assert self.P.shape[0] == self.P.shape[1], "P must be square"
        assert self.P.shape[0] == self.languageNum, "P rows must equal languageNum"

        # Probabilities must be between 0 and 1
        assert np.all(self.P >= 0) and np.all(self.P <= 1), "P entries must be in [0,1]"

        # Grid shape consistent
        assert self.grid.width == self.width, "Grid width mismatch"
        assert self.grid.height == self.height, "Grid height mismatch"

        # All agents must be on the grid and have valid languages
        for agent in self.agents:
            x, y = agent.pos
            assert 0 <= x < self.width and 0 <= y < self.height, "Agent out of bounds"
            assert 0 <= agent.language < self.languageNum, "Agent language out of range"

        # Network basic sanity: all languages should be nodes
        if self.languageNetwork is not None:
            for lang in range(self.languageNum):
                assert lang in self.languageNetwork.nodes, "Missing language node in network"
