# imports
import numpy as np
import random
import string

class Individual:

    def get_fitness(self):
        pass

    def mutate(self):
        pass

    def crossover(self, other_individual):
        pass


# some example
class HelloWorldIndividual(Individual):
    def __init__(self, chromosome='aaaaaaaaaaa'):
        self.chromosome = chromosome

    def random():
        return HelloWorldIndividual(''.join(random.choices(string.ascii_letters + ' ', k=11)))

    def get_fitness(self):
        a = 'Hello World'
        b = self.chromosome
        return 11 - sum (a[i] != b[i] for i in range(len(a)))

    def mutate(self):
        index = random.randint(0,10)
        temp = list(self.chromosome)
        temp[index] = random.choice(string.ascii_letters + ' ')
        return HelloWorldIndividual(''.join(temp))

    def crossover(self, other_individual):
        index = random.randint(0,10)
        return HelloWorldIndividual(self.chromosome[:index] + other_individual.chromosome[index:])

    def __str__(self):
        return self.chromosome


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
    s = 15
    ch_len = 7
    w = np.transpose(np.asarray([12, 1, 4, 1, 2, 14, 1]))
    c = np.transpose(np.asarray([4, 2, 10, 1, 2, 16, 5]))
    problem = KnapsackProblem(s, w, c)

    population = KnapsackProblemPolulation(ch_len, problem, RouletteSelection())
    best = population.get_best()
    for i in range(100):
        population.select(70)
        population.crossover()
        population.mutate()
        temp_best = population.get_best()
        if temp_best.get_fitness() > best.get_fitness():
            best = temp_best

    temp_best = population.get_best()
    if temp_best.get_fitness() > best.get_fitness():
        best = temp_best
    print(best)
    print(best.get_fitness())