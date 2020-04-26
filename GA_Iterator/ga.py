'''
This class is for general genetic algorithms

The lifecycle of a genetic algorithm is:
    - select parents
    - generate children by randomly selecting parents to breed then mutate the offspring
    - Evaluate the fitness of offspring
    - Select best members of population to be parents of the next generation

The values in this class are"
    - ga.population -> dictionary of members
    - ga.population[member]: dictionary with sub dictionaries 
        ["chromosom", "expression", "fitness"] 
    - ga.parents -> list of population members to be parents of the generation

To use this class, you will need to overload the expression function to fit your needs

The lifecycle after initialization looks like looks like
    1. ga.load_population()
        + first checks if there is a population in memory, if so, just uses that
        + then checks if there are any generations on file, and grabs the latest
        + In either of those cases, it will select the parents based off of the 
            elitism ratio 
        + if both those come up short, or the `start_fresh` argument is marked true
            then children will be randomly generated rather then bred 
    2. ga.gen_child():
        + breeds and mutates a child, adds it to the population
        + adds the chromosome to the population
        + returns child_id, child_chromosome
    3. use your custom express_chromosome function to express the provided chromosome
        + feel free to add it to the population dictionary with
            self.population[child_id]["expression"] = express_chromosome(child_chromosome)
    4. run whatver is needed to get the fitness of the child
    5. ga.population[child_name]["fitness"] = fitness
        + add the fitness to the child's information
    6. run ga.save_child_to_file(child_name)
        + this will save the child's info to the output_dir/population/ subdir as a json
    7. run steps 2-6 for however members you want in your generations
    8. run ga.save_generation()
        + this will save a json to the "generations" file, which will be a dictionary with 
            the entries 
                ++ "memebers" (a list of the member names) 
                ++ "parents" (a list of member names which were used as parents)
    9. head back to step 1
'''

import json
import numpy as np
import os
import pandas as pd
import time

class ga():

    def __init__(self, output_dir="ga_output/", 
            gene_map=[("chromosome", 0, 1)],
            elitism_ratio=0.2,
            mutation_ratio=0.2,
            fitness_lower_is_better=True):
        '''
        input output_dir: str, will be created if it does not exist already
        input gene_map: list of tuples, ("gene_name", min_int, max_int)
        input elitism_ratio: float, ratio of population to be used as parents
        input fitness_lower_is_better: bool
        '''
        self.population = {}
        self.output_dir = output_dir 
        self.fitness_lower_is_better = fitness_lower_is_better
        self.elitism_ratio = elitism_ratio
        self.mutation_ratio = mutation_ratio
        os.makedirs(self.output_dir, exist_ok=True)
        self.generation_dir = os.path.join(self.output_dir, "generations")
        os.makedirs(self.generation_dir, exist_ok=True)
        self.population_dir = os.path.join(self.output_dir, "population")
        os.makedirs(self.population_dir, exist_ok=True)
        self.chromosome_len = len(gene_map)
        self.mod_array = np.zeros(self.chromosome_len, dtype="int")
        self.base_array = np.zeros(self.chromosome_len, dtype="int")
        self.gene_map = gene_map
        for i in range(self.chromosome_len):
            self.mod_array[i] = gene_map[i][2] - gene_map[i][1]
            self.base_array[i] = gene_map[i][1]
        self.gene_max = self.mod_array.prod()

    def load_population(self, start_fresh=False):
        '''
        checks if there is a population on file, if there is and start fresh is false,
            then it uses that population for parents
        '''
        if start_fresh:
            self.new_population=True
        elif len(self.population.keys()) > 0:
            self.new_population=False
        elif len(os.listdir(self.generation_dir)) > 0:
            self.new_population=False
            generations = os.listdir(self.generation_dir)
            generations.sort()
            with open(os.path.join(self.generation_dir, generations[-1]), 'r') as f:
                self.population = json.load(f)
            for member_id in self.population.keys():
                with open(os.path.join(self.population_dir, member_id + ".json"), 'r') as f:
                    self.population[member_id] = json.load(f)
        else:
            raise ValueError("For ga.load_population() either specify `start_fesh=True` or ensure there are generations in `output_dir`/generations/")
        if not self.new_population:
            self.choose_parents()
            new_population = {}
            for parent_id in self.parents:
                new_population[parent_id] = self.population[parent_id]
            self.population = new_population

    def choose_parents(self):
        df = pd.DataFrame.from_dict(self.population, orient="index")
        df.index.rename("member_id", inplace=True)
        df.reset_index(inplace=True)
        df = df[["member_id", "fitness"]].copy()
        df.sort_values("fitness", ascending=self.fitness_lower_is_better, inplace=True)
        self.parents = df["member_id"][:max(2,
            int(df.shape[0] * self.elitism_ratio))].tolist()
        self.num_parents = len(self.parents)

    def gen_child(self):
        if self.new_population:
            child = np.random.randint(low=0, high=self.gene_max,  size=(self.chromosome_len,))
        else: 
            parent_ids = np.random.choice(self.num_parents, 2, replace=False)
            child = self.mutate(self.crossover(self.parents[parent_ids[0]], 
                self.parents[parent_ids[1]]))
        child = np.mod(child, self.mod_array) + self.base_array 
        child_id = int(time.time()*10e6) 
        self.population[child_id] = {"chromosome": child}
        return child_id, child

    def save_member(self, member_id):
        '''
        input member_id: int
        '''
        with open(os.path.join(self.population_dir, str(member_id) + ".json") , 'w') as f:
            out_dict = {}
            for k in self.population[member_id]:
                if k == "chromosome":
                    out_dict[k] = self.population[member_id][k].tolist()
                else: 
                    out_dict[k] = self.population[member_id][k]
            json.dump(out_dict, f)

    def save_generation(self):
        out_dict = {}
        for k in self.population:
            out_dict[k] = {"chromosome": self.population[k]["chromosome"].tolist(),
                    "fitness": self.population[k]["fitness"]}
        with open(os.path.join(self.generation_dir, 
            str(int(time.time() * 10e6)) + ".json") , 'w') as f:
            json.dump(out_dict, f)

    def crossover(self, left_chromosome_member_id, right_chromosome_member_id):
        '''
        input left_chromosome_index: int, row index in population of left chromosome
        input right_chromosome_index: int, row index in population of right chromosome
        This is the breeding part of the genetic algorithm
        '''
        crossover_index = np.random.randint(0, self.chromosome_len-1)
        left_mask = np.where(np.arange(self.chromosome_len) <= crossover_index, 1, 0)
        return self.population[left_chromosome_member_id]["chromosome"]*left_mask + \
                self.population[right_chromosome_member_id]["chromosome"]*(1 - left_mask)

    def mutate(self, chromosome):
        '''
        input genes_to_mutate: int, number of genest to mutate per chromosome, in probability
        '''
        mutate_map = np.where(
            np.random.uniform(size=self.chromosome_len) < self.mutation_ratio, 1, 0)
        chromosome +=  mutate_map * np.random.randint(0, self.gene_max,
                size=self.chromosome_len).astype(int)
        return chromosome
