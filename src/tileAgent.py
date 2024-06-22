# =======================================================

"""
This file contins one class, which defines a seed for a 
tile agent that exists within the GoL. Each seed is an 
initial tiling. It is the seeds that our genetic algorthm 
will evolve. 

This class defines constructors, attributes, functions 
and fitness evaluation of the seed. 

Authors: Christian Lentz, Ana Espeleta
"""

# =======================================================

import numpy as np
import numpy.random as nprand
import random as rand 
import math as math

# =======================================================

class Seed:
    
    """
    Defines a seed for a tile agent within the game of life. 
    """ 

    def __init__(self, xSpan, ySpan, popSize):
        
        """
        Constructor for the seed. 
        """ 

        # instance data   
        self.xSpan = xSpan 
        self.ySpan = ySpan
        self.area = 0
        self.currPos = (0,0)

        # seed density
        self.maxDensity = 0.13
        self.minDensity = 0.18
        self.density = self.gen_random_density()

        # initialize matrix which records the cells for the seed 
        self.cells = np.zeros((xSpan, ySpan))

        # populate the cells matrix 
        self.random_fill(self.density)
        
        # agent fitness 
        self.fitness = 0

        # agent similarity
        self.similarities = np.zeros(popSize) 

        # agent position in population array
        self.address = None 

    def gen_random_density(self): 
        
        """
        Randomly generate the density for the new seed.
        """
        
        densityRange = self.maxDensity - self.minDensity 
        randPercent = rand.random()
        randDensity = self.minDensity + (densityRange*randPercent)
        return randDensity

    def random_fill(self, seedDensity):  
        
        """
        Randomly populate the seed with living cells based on the randomly generated denisty. 
        Each cell will store this information in self.cells defined above. 
        """

        # for each cell 
        for x in range(self.xSpan): 
            for y in range(self.ySpan): 
                if rand.random() <= seedDensity:
                    self.cells[x][y] = 1
                    self.area += 1      
    
    def eval_fitness(self, points, areas, steps): 
        
        """ 
        Evaluate the movement and growth of an agent to determine fitness. 
        """
        
        # get fitness measures 
        move_fitness = self.score_movement(points, steps)
        growth_fitness = self.score_growth(areas, steps)
        
        # combine these, weight the movement score
        overall_fit = (move_fitness + growth_fitness) / 2 
        
        # set fitness for the agent
        self.fitness = overall_fit
                
    def score_movement(self, points, steps): 
        
        """
        Score the movement of the agent.
        We will favor directed, linear movement which persists over time steps.  
        """
        
        # k counts the total number of moves that are "good"
        k = 1 
        move_fitness = 0.0
        numPoints = len(points)
        maxAngles = steps-2
        
        for i in range(numPoints - 1): 
            if i != 0: 
                # get three adjacent points
                prev_pt = points[i-1]
                curr_pt = points[i]
                next_pt = points[i+1]
                same = self.are_same_points(prev_pt, curr_pt, next_pt)
                # evalute step if the triplet of points is unique 
                if not same:
                    # convert to two vectors 
                    v1 = ((curr_pt[0] - prev_pt[0]), (curr_pt[1] - prev_pt[1]))
                    v2 = ((next_pt[0] - curr_pt[0]), (next_pt[1] - curr_pt[1]))
                    # score movement based on angle between vectors
                    # angles closer to 0 are preferred 
                    norm_angle = self.angle(v1, v2) / 180
                    step_score = 1 - norm_angle
                    if step_score >= 0.95:
                        k+=1
                    # increment fitness
                    move_fitness += step_score
                    
        # normalize and return
        k_term = k/(maxAngles + 1)
        return (move_fitness/maxAngles) * k_term
    
    def are_same_points(self, pt1, pt2, pt3):

        """
        Determine if the points used to measure a single movmement step actually
        form an angle. If either pair of points is the same, we skip that step, as
        the seed either didn't move or oscillated. 
        """

        if pt1 == pt2 or pt2 == pt3 or pt1 == pt3: 
            return True 
        else: 
            return False
        
    def dotprod(self, v1, v2): 
        
        """
        Dot two vectors v1 and v2. Used for evaluation of fitness. 
        """   
        
        return sum((a*b) for a, b in zip(v1, v2))

    def length(self, vec):
        
        """
        Get the length of a vector. Used for evaluation of fitness. 
        """
        
        dp = self.dotprod(vec,vec)
        return math.sqrt(dp)

    def angle(self, v1, v2):
        
        """
        Get the angle between two vectors. Used for evaluation of fitness. 
        """
        
        norm = self.dotprod(v1, v2) / (self.length(v1) * self.length(v2))
        if norm >= -1 and norm <= 1: 
            return math.acos(norm) * 180 / math.pi
        else:
            return math.acos(norm%1) * 180 / math.pi
    
    def score_growth(self, areas, steps): 
        
        """
        Score the growth of the agent. We will favor cells which do not grow 
        indefinitley or fizzle-out, and maintain a roughly constant area over 
        time steps. 
        """
        
        # k counts the total number of moves that are "good"
        k = 1
        growth_fitness = 0
        initArea = areas[0]
        numAreas = len(areas)
        
        # eval movement fitness 
        # better steps are scored closer to 1 
        for i in range(numAreas): 
            if i != 0: 
                stepScore = 0
                prop = areas[i]/initArea
                # growth case
                if prop >= 1: 
                    if prop >= 1.05 and prop <= 1.20:
                        k+=1
                    stepScore = 1/prop
                # shrink case: 
                if prop < 1:
                    if prop >= 0.8 and prop <= 0.95:   
                        k+=1
                    stepScore = prop
                growth_fitness += stepScore
                
        # normalize and return
        k_term = k/(steps + 1) 
        return (growth_fitness/steps) * k_term
    
    def mutate(self, mutation_rate):
        
        """
        Mutate a seed by randomly flipping bits. Assumes the seed
        contains 0s and 1s.
        """
        
        num_mutations = 0
        for s_x in range(self.xSpan):
            for s_y in range(self.ySpan):
                if (rand.uniform(0, 1) < mutation_rate):
                    # flip cell value: 0 becomes 1 and 1 becomes 0
                    self.cells[s_x][s_y] = 1 - self.cells[s_x][s_y]
                    # count the number of mutations so far
                    num_mutations = num_mutations + 1
        # force a minimum of one mutation -- there is no value
        # in having duplicates in the population
        if (num_mutations == 0):
            s_x = rand.randrange(self.xSpan)
            s_y = rand.randrange(self.ySpan)
            self.cells[s_x][s_y] = 1 - self.cells[s_x][s_y]