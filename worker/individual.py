import numpy as np

class Individual:

    def get_fitness(self):
        pass

    def count_fitness(self):
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
        self.fitness = self.count_fitness()

    def random(self):
        self.chromosome = np.random.choice([0, 1], size=self.chromosome_length)

    def get_fitness(self):
        return self.fitness

    def set_fitness(self, fitness):
        self.fitness = fitness
    
    def count_fitness(self):
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