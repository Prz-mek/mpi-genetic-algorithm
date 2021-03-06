# imports
import numpy as np
import random
import matplotlib.pyplot as plt

def readFile(name):
    with open(name, 'r') as file:
        text = file.read()
        values = text.split()
        knapsack_size = int(values[0])
        values = values[1:]
        weights = np.asarray([int(i) for i in values[0::2]], dtype="int32")
        costs = np.asarray([int(i) for i in values[1::2]], dtype="int32")
        return knapsack_size, len(costs), weights, costs

class Individual:

    def get_fitness(self):
        pass

    def mutate(self):
        pass

    def crossover(self, other_individual):
        pass

class KnapsackProblem:
    def __init__(self, knapsack_size, weights, costs):
        self.knapsack_size = knapsack_size
        self.weights = weights
        self.costs = costs

    def get_cost(self, chromosome):
        return chromosome.dot(self.costs)

    def get_weight(self, chromosome):
        return self.knapsack_size - chromosome.dot(self.weights)


class KnapsackProblemIndividual(Individual):
    def __init__(self, chromoseome_length, problem, chromosome=None):
        self.chromosome_length = chromoseome_length
        self.problem = problem
        self.chromosome = chromosome

    def random(self):
        self.chromosome = np.random.choice([0, 1], size=self.chromosome_length)


    # TODO adjust fitness function
    def get_fitness(self):
        cost = self.problem.get_cost(self.chromosome)
        weight = self.problem.get_weight(self.chromosome)
        if weight < 0:
            return cost - 100_000_000
        return cost

    def mutate(self):
        index = random.randint(0,self.chromosome_length-1)
        chromosome = self.chromosome[:]
        if chromosome[index] == 0:
            chromosome[index] == 1
        else:
            chromosome[index] == 0
        return KnapsackProblemIndividual(self.chromosome_length, self.problem, chromosome)

    def crossover(self, other_individual):
        index = random.randint(0,self.chromosome_length-1)
        return KnapsackProblemIndividual(self.chromosome_length, self.problem, np.concatenate((self.chromosome[:index], other_individual.chromosome[index:])))

    def __str__(self):
        return str(self.chromosome)

class Selection:
    def select(population):
        pass


class RouletteSelection(Selection):
    def __init__(self):
        pass

    def select(self, population, num):
        n = len(population)
        wheel = np.zeros(len(population))
        sum = 0.0
        for i in range(n):
            sum += population[i].get_fitness()
            wheel[i] = sum
        
        wheel /= sum
        new_population=[]
        for i in range(num):
            val = random.random()
            j = 0
            while (wheel[j] < val):
                j += 1
            new_population.append(population[j])

        return new_population
                    
class KnapsackProblemPolulation:
    def __init__(self, chromoseome_length, problem, selection, crossover_prob=0.4, mutation_prob=0.03, size=100):
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.selection = selection
        self.population = []
        for _ in range(size):
            individual = KnapsackProblemIndividual(chromoseome_length, problem)
            individual.random()
            self.population.append(individual)

    def crossover(self):
        n = len(self.population)
        for i in range(n):
            for j in range(i+1,n):
                val = random.random()
                if val < self.crossover_prob:
                    self.population.append(self.population[i].crossover(self.population[j]))

    def mutate(self):
        n = len(self.population)
        for i in range(n):
            val = random.random()
            if val < self.crossover_prob:
                self.population.append(self.population[i].mutate())

    def select(self, num):
        self.population = self.selection.select(self.population, num)


    def get_best(self):
        best = self.population[0]
        for ind in self.population:
            if best.get_fitness() < ind.get_fitness():
                best = ind
        return best

    def get_all_results(self):
        return self.population

def genetic_main():
    size = 200
    iter = 100
    s, ch_len, w, c = readFile("data/data_big")
    problem = KnapsackProblem(s, w, c)

    population = KnapsackProblemPolulation(ch_len, problem, RouletteSelection(), size=size)
    best_inds = []
    for i in range(iter):
        population.crossover()
        population.mutate()
        best_inds.append(population.get_best().get_fitness())
        population.select(size)

    plt.plot([i for i in range(iter)], best_inds)
    plt.show()

genetic_main()