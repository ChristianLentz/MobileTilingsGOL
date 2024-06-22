# =======================================================

"""
This file includes two classes which work together to 
run our simulation. 

1) BuildGoL: Build a deterministic or stochasitc implementation 
of the GoL. Add seeds to the board and record their data for 
fitness evaluation. 

2) GAManager: Oversee the evolution of seeds/agents and link the 
front end with the genetic algorithm.

Reference the following files in Turney's ModelS code: 

- model_classes
- model_functions 

Authors: Christian Lentz, Ana Espeleta
"""

# =======================================================

import argparse
import numpy as np
import random as rand
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from matplotlib.animation import PillowWriter
from tileAgent import Seed
from GeneticAlg import GA

# =======================================================

class BuildGoL: 

    """
    Build deterministic (classic) or stochastic (probabalistic) version of 
    Conway's GoL.
    """ 

    def __init__(self, withSim, save=None): 

        """
        Constructor for BuildGoL class. Builds a 2D array of cells which 
        are OFF/dead. Size of the array is based on the current largest 
        member of the population. 
        """
        
        """
        An important note on how the simulation proceeds: we start by 
        running the sim and evaluating fitness for each seed in the 
        initial population. Once this has concludued, we start the GA. 
        The GA will proceed, where at each "generation" we replace the 
        least fit member of the current population with a ne child. This 
        proceeds maxGens times. 
        """
        
        # ============================== parameters for the game of life
        
        self.gridSize = 200                                            # size of the grid (initially 100)
        self.gridCenter = int(self.gridSize/2)                         # the center of the current board
        self.isClassic = True                                          # rules for the GoL
        self.withSim = withSim                                         # run with or without visuals
        self.done = False
        self.save = save
        
        # ============================== parameters for the genetic algorithm 
        
        self.popSize = 20                                              # size of the population (remains the same)
        self.isClassic = True                                          # rules for the sim 
        self.ga_manager = GAManager(self.gridSize, self.popSize)       # genetic algorithm manager
        self.population = self.ga_manager.initPop                      # members of the current population 
        self.maxGens = 3000 * self.popSize                             # number of generations to run the GA
        self.currGen = 0                                               # current generation of GA
        self.currSeed = self.population[0]
        
        # ============================== value for on/off cells
        
        self.ON = 255  
        self.OFF = 0                    
        
        # ============================== hold data for evaluating agent fitness                       
       
        self.posArray = []                                            # position data 
        self.sizeArray = []                                           # size data 
        
        # ============================== counters
        
        self.stepCounter = 0                                          # current step for the sim of a single seed
        self.maxSteps = 100                                           # max time steps for sim of a single seed
        self.agentCounter = 0                                         # number of agents we have added to the game board 
                                                                      # (zero indexed to match the population array)                                    
        # ============================== collect GA data 
        
        self.currBestFitScore = 0                                     # current best fitness score in the population
        self.mostFitSeed = None                                       # most fit member of current population
        self.avgPopFitnessArr = []                                    # records average fitness of population over the sim

        # build the game grid
        self.grid = self.clear_grid()
        
        # add first seed to the grid 
        self.add_agent(self.currSeed)
        
        # run the sim
        if self.withSim:
            # build the GUI
            self.fig, self.ax = plt.subplots()
            self.img = self.ax.imshow(self.grid) 
            # animate the GUI
            self.animation = ani.FuncAnimation(fig = self.fig, 
                                        func = self.deterministic_run,
                                        fargs = (self.img, False),  
                                        frames = 100, 
                                        interval = 0.5)
            plt.show()
        if (not self.withSim) or (self.save): 
            # run without visuals
            for i in range(self.maxGens + self.popSize):
                for j in range(self.maxSteps+1):
                    self.deterministic_run(None, None, False)
        
    def add_agent(self, seed):
        
        """
        Add a single seed to the game board.
        """ 
        
        # make a copy of the grid 
        newGrid = self.clear_grid()
        # get x and y offset to center agent on the board
        xOffset = self.gridCenter - int(seed.xSpan/2) 
        yOffset = self.gridCenter - int(seed.ySpan/2)
        # add the agent to the copy 
        for i in range(seed.xSpan):
            for j in range(seed.ySpan):
                if seed.cells[i][j] == 1: 
                    newGrid[i+xOffset][j+yOffset] = self.ON
        # update board with the copy
        self.grid[:] = newGrid[:] 
        # add the current size and location of seed to data arrays 
        self.posArray.append(self.get_agent_position(seed.area))
        self.sizeArray.append(seed.area)

    def deterministic_run(self, frame, img, save_mode):

        """
        Run the simulation using the deterministic (classic) rules. 
        """
        
        # evaluate initial population 
        if self.agentCounter == 0 and self.stepCounter == 0 and self.currGen == 0: 
            print(f"========== Evaluating Initial Seeds ==========")
            print()
        # run the GA steps 
        if self.currGen < self.maxGens and not self.done:
            self.sim_one_seed(img, False)
        # save the most fit seed if in save mode
        if save_mode:
            self.sim_one_seed(img, True)
                
    def sim_one_seed(self, img, save_mode): 
        
        """
        Run the simulation for one single seed. This includes: 
            - Update tiles according to rules at each step of stepCounter
            - Collect data every five steps of stepCounter
        """
        
        newGrid = self.grid.copy()
        totalCells = 0 
        # for each cell grid[i,j]
        for i in range(self.gridSize):
            for j in range(self.gridSize):                
			    # get live neighbor counts for each cell 
                live_neighs = self.get_neighbor_counts(i, j)
			    # apply Conway's rules
                if self.grid[i][j] == self.ON:
                    totalCells += 1
                    if (live_neighs < 2) or (live_neighs > 3):
                        newGrid[i][j] = self.OFF
                        totalCells -= 1
                else:
                    if live_neighs == 3:
                        newGrid[i][j] = self.ON
                        totalCells += 1
        # update grid 
        self.grid[:] = newGrid[:]
        if self.withSim or save_mode:
            img.set_data(self.grid)
        # collect data or remove current seed
        if not save_mode:
            if self.stepCounter < self.maxSteps: 
                if (self.stepCounter % 5 == 0 and self.stepCounter != 0): 
                    self.collect_data(totalCells)
                self.stepCounter+=1
            else: 
                self.update_for_new_seed(img)
    
    def get_neighbor_counts(self, i, j):

        """
        Count the number of neighbors of cell [i,j] which are currently 
        ON/alive. Total should be in {0, ..., 8}. We compute 8-neighbor 
        sum and do not assume toroidal boundary conditions. 
        """
        
        N = self.gridSize
        total = 0
        if (i < N-1) and (j < N-1):
            total = (self.grid[i][(j-1)] + 
                self.grid[i][(j+1)] + 
                self.grid[(i-1)][j] + 
                self.grid[(i+1)][j] + 
                self.grid[(i-1)][(j-1)] + 
                self.grid[(i-1)][(j+1)] + 
                self.grid[(i+1)][(j-1)] + 
                self.grid[(i+1)][(j+1)])

        return int(total/255)
    
    def get_agent_position(self, totalCells):
        
        """ 
        Compute the position of the agent by taking an average position of 
        current living cells
        """
        
        totalSumX = 0
        totalSumY = 0
        for i in range(self.gridSize):
            for j in range(self.gridSize):
                if self.grid[i][j] == self.ON:
                    totalSumX += i
                    totalSumY += j 
        return totalSumX/totalCells, totalSumY/totalCells
            
    def collect_data(self, totalCells): 
        
        """
        Collect data for fitness eval of the current seed after each time 
        step. 
        """
        
        if self.currGen == self.maxGens: 
            pass
        else:
            if (totalCells != 0):
                # make sure position or size has changed 
                newPos = self.get_agent_position(totalCells)
                currPos = self.posArray[len(self.posArray)-1]
                currSize = self.sizeArray[len(self.sizeArray)-1]
                # if newPos != currPos or totalCells != currSize:
                # record new data 
                self.posArray.append(self.get_agent_position(totalCells))
                self.sizeArray.append(totalCells)
            
    def update_for_new_seed(self, img): 
        
        """
        Update the board once a seed hits max steps.
            - If we are in the first generation, get the next seed in the 
            inital popultion. 
            - If we are past the first generation, run the GA, eval 
            fitness of the new child, replace lest fit member of population.
        """
        
        # evaluate fitness for current seed 
        # we only collect data every fifth step, so we pass maxSteps/5 
        self.currSeed.eval_fitness(self.posArray, self.sizeArray, int(self.maxSteps/5))
        self.clear_data()
        if self.currSeed.fitness > self.currBestFitScore: 
            self.mostFitSeed = self.currSeed
            self.currBestFitScore = self.currSeed.fitness
        self.stepCounter = 0
        # if first generation 
        if self.currGen == 0:
            print(f"Fitness for seed {self.currSeed.address} is {self.currSeed.fitness}")
            self.update_for_seed_first_gen()
            # if self.agentCounter == self.popSize-1: 
            #     self.record_avg_pop_fit()
            if img != None:
                img.set_data(self.grid)
        # if generations between first and last 
        elif self.currGen <= self.maxGens - 2: 
            print(f"New child fitness: {self.currSeed.fitness}")
            print()    
            self.record_avg_pop_fit()
            print(f"========== GA Step {self.currGen} ==========")
            print()
            self.update_for_seed_GA() 
            if img != None:
                img.set_data(self.grid)
        # if last generation 
        else: 
            print(f"New child fitness: {self.currSeed.fitness}")
            self.record_avg_pop_fit()
            self.print_summary()
            # if we are running in 'sim' mode
            if img != None:
                plt.close()
            else:
                if self.save: 
                    # if we are running in 'save' mode
                    self.save_best_seed()
                else: 
                    # if we are running in 'data' mode
                    self.plot_fit_data()
            self.done = True
    
    def update_for_seed_first_gen(self):
        
        """
        Update the game board for a new seed when we are currentlty evaluating 
        the population of initial seeds. 
        """  
         
        # continue evaluating the initial population 
        if self.agentCounter < self.popSize-1:  
            self.agentCounter+=1
            self.currSeed = self.population[self.agentCounter]
            self.add_agent(self.currSeed)
        # OR call the GA if we are done with inital population 
        else: 
            self.record_avg_pop_fit()
            self.update_for_seed_GA()
            self.currGen+=1
        
    def update_for_seed_GA(self):
        
        """
        Update the game board for a new seed when we are currently executing 
        the GA. This means that we have already evaluated the initial 
        population, and are now evaluating one seed at a time, rather than 
        evaluating a whole population. 
        """
         # call one GA step
        roul_seed = self.population[self.ga_manager.GA.roulette_select(self.population)]
        if self.currGen % 20 == 0: 
            # adjust mutation rate every fifth generation 
            avg_fit = self.avgPopFitnessArr[len(self.avgPopFitnessArr) - 1]
            self.population, self.currSeed = self.ga_manager.one_step(
                roul_seed, 
                avg_fit=avg_fit)
        else: 
            self.population, self.currSeed = self.ga_manager.one_step(roul_seed)

        # make sure the current seed is the new child so we can evaluate it 
        self.add_agent(self.currSeed)
        self.currGen+=1 
    
    def clear_data(self): 
        
        """
        Clear the data array for the current seed whose fitness we have 
        evaluated. 
        """
        
        self.posArray.clear()
        self.sizeArray.clear()
        
    def record_avg_pop_fit(self): 
        
        """
        Record the current average fitness of the population. 
        """
        
        sum = 0
        for seed in self.population: 
            sum += seed.fitness
        self.avgPopFitnessArr.append(sum / self.popSize)
        
    def print_summary(self):
        
        """
        Print the results of the simulation
        """
        
        print() 
        print(f"========== End of Simulation ==========")
        print()
        print(f"Most fit is seed: {self.mostFitSeed.address}")
        print(f"Best seed score: {self.mostFitSeed.fitness}")
        i = len(self.avgPopFitnessArr)
        print(f"Average fitness of inital population: {self.avgPopFitnessArr[0]}")
        print(f"Average fitness of final population: {self.avgPopFitnessArr[i-1]}")
        print()
    
    def plot_fit_data(self): 
        
        """
        Use matplotlib to plot the fitness data that we collected 
        over the course of our genetic algorithm.  
        """
        
        xvals = list(range(0,len(self.avgPopFitnessArr)))
        yvals = self.avgPopFitnessArr
        fig, ax = plt.subplots()
        ax.plot(xvals, yvals, '-')
        ax.set_title("Evolution Using Sexual Layer Only")
        ax.set_xlabel("GA Steps")
        ax.set_ylabel("Average Fitness")
        plt.show()
    
    def save_best_seed(self): 
        
        """
        When running in save mode, save a gif of the most fit seed at 
        the end of the simulation. 
        """
        
        print("saving most fit seed . . . ")
        print()
        # prepare the visual 
        pillow = PillowWriter(fps=30)
        self.clear_grid()
        seed = self.mostFitSeed
        self.add_agent(seed)
        fig, ax = plt.subplots()
        img = ax.imshow(self.grid)
        # animate and save
        animation = ani.FuncAnimation(fig = fig, 
                                    func = self.deterministic_run,
                                    fargs = (img, True,),  
                                    frames = 100, 
                                    interval = 0.5)
        animation.save('bestSeed.gif', writer=pillow)
        plt.show()
    
    def clear_grid(self):
        
        """
        Reset the game board to initial state. Set all cells to OFF. 
        """
        
        return [[self.OFF for x in range(self.gridSize)] for y in range(self.gridSize)]
    
# ======================================================= GA Manager Class

class GAManager: 

    """
    Oversee the GA, and link each new population of seeds generated by 
    GeneticAlg to the front end. 
    """

    def __init__(self, gridSize, popSize):

        """
        Constructor for GAManager class
        """

        self.gridSize = gridSize
        self.popSize = popSize
        self.minSpan = int(gridSize/5 - 10)
        self.maxSpan = int(gridSize/5 + 10) 
        self.initPop = self.init_population()
        self.GA = GA(self.initPop)
        self.firstGAStep = True

    def init_population(self): 
        
        """
        Randomly initialize the population of seeds. 
        Agents should be small and sparse.
        """
        
        population = []
        for i in range(self.popSize):
            rand_x_span = rand.randrange(self.minSpan, self.maxSpan)
            rand_y_span = rand.randrange(self.minSpan, self.maxSpan)
            seed = Seed(rand_x_span, rand_y_span, self.popSize)
            population.append(seed)
            seed.address = i 
        
        return population
    
    def one_step(self, candidate_seed, avg_fit=None): 
        
        """
        Run a single step of the genetic algorithm. This creates one new child 
        seed from one of the following: 
            - asexual reproduction with a mutation 
            - sexual reproduction with roulette selection
        Note that the GA call will update the GeneticAlg class' reference to 
        the population. 
            
        Once we have completed the GA step, we:
            - return the new child and population to the GOL class
            - update the GOL's reference to the population 
            - alter mutation rate according to similarity of the population
            - evaluate the fitness of the new child seed 
        """
        
        if self.firstGAStep: 
            print()
            print(f"========== Initialize GA ==========")
            print()
            self.firstGAStep = False
        # alter mutation rate ß
        if avg_fit != None: 
            print("running with mutation rate change. . .")
            self.GA.alter_mutation_rate(avg_fit)

        # call the GA step - asexual, sexual, similarity
        # newChild = self.GA.uniform_asexual(candidate_seed)
        # newChild = self.GA.sexual(candidate_seed)
        newChild = self.GA.sexual_similarity(candidate_seed)

        # return pop and child
        return self.GA.pop, newChild
        
# =======================================================

def main(): 
    
    # collect command line input
    parser = argparse.ArgumentParser(description="Runs RunSim.py.")
    parser.add_argument('--data', action='store_true', required=False)
    parser.add_argument('--sim', action='store_true', required=False)
    parser.add_argument('--save', action='store_true', required=False)
    args = parser.parse_args()
    
    # run the sim
    if args.data:
        print()
        print("Running in data mode . . . ")
        print()
        BuildGoL(False)
    elif args.sim:
        print()
        print("Running in simulation mode . . . ")
        print()
        BuildGoL(True)
    elif args.save:
        print()
        print("Running in save mode . . . ")
        print()
        BuildGoL(False, save=True)
    else:
        print()
        print("Please run this file in the terminal!")
        print("Specify which mode you would like to run using one of the following flags:") 
        print()
        print("     python RunSim.py --data")
        print("     python RunSim.py --sim")
        print("     python RunSim.py --save")
        print()

if __name__ == '__main__':
    main()