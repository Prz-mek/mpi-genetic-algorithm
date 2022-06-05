import random
import numpy as np
import concurrent

class Selection:
    def select(population):
        pass


class RouletteSelection(Selection):

    def select_thread(population, wheel, num):
        new_population=[]
        for _ in range(num):
            val = random.random()
            j = 0
            while (wheel[j] < val):
                j += 1
            new_population.append(population[j])

        return new_population

    def select(self, population, num):
        n = len(population)
        wheel = np.zeros(len(population))
        sum = 0.0
        for i in range(n):
            sum += population[i].get_fitness()
            wheel[i] = sum
        
        wheel /= sum
    
        with concurrent.futures.ProcessPoolExecutor as executor:
            # TODO create nums list you may use generator
            results = [executor.submit(self.select_thread, population, wheel, num) for num in [num // 2, num //2]]

            new_population = []
            for r in concurrent.futures.as_completed(results):
                new_population += population
        

    
