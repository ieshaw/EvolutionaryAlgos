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
import shutil
import time

class ga():

    def __init__(self, output_dir="ga_output/", 
            gene_map=[("chromosome", 0, 1)],
            elitism_ratio=0.2,
            mutation_ratio=0.2,
            fitness_lower_is_better=True,
            verbose=False):
        '''
        input output_dir: str, will be created if it does not exist already
        input gene_map: list of tuples, ("gene_name", min_int, max_int)
        input elitism_ratio: float, ratio of population to be used as parents
        input fitness_lower_is_better: bool
        '''
        self.population = {}
        self.verbose=verbose
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
            self.mod_array[i] = gene_map[i][2] - gene_map[i][1] + 1
            self.base_array[i] = gene_map[i][1]
        self.gene_max = self.mod_array.prod()

    def load_population(self, start_fresh=False, min_member_count=0, generation_file=""):
        '''
        checks if there is a population on file, if there is and start fresh is false,
            then it uses that population for parents
        input min_member_count: int, Will search for most recent generation whose
            len(members) >= min_member_count. Default 0 will just give the most recent 
            generation file.
        input generation_file: str, if specified, will load that specific generation
            file regardless of recency or memmebr count
        '''
        self.previous_generation_file_name = ""
        if (~start_fresh) & (len(self.population.keys()) > 0) & (
                len(generation_file) < 1):
            self.new_population=False
            if self.verbose:
                print("Seeding new generation with population from memory")
        elif (~start_fresh) & (len(os.listdir(self.generation_dir)) > 0):
            self.new_population=False
            self.population = {}
            prev_generation_id, self.prev_generation, \
                    self.population = self.load_generation_from_file(
                            min_member_count=min_member_count, 
                            generation_file=generation_file) 
        else:
            if self.verbose:
                print("Starting from fresh, randomly breeding new generation")
            self.new_population=True
            self.parents = []
            self.population = {}
        if not self.new_population:
            self.choose_parents()
            new_population = {}
            for parent_id in self.parents:
                new_population[parent_id] = self.population[parent_id]
            self.population = new_population
            if self.verbose:
                print("Selected {} parents to breed new generation".format(self.num_parents))

    def load_generation_from_file(self, min_member_count=0, generation_file=""):
        '''
        input generation_file: str
        input min_member_count: int, Will search for most recent generation whose
            len(members) >= min_member_count. Default 0 will just give the most recent 
            generation file.
        input generation_file: str, if specified, will load that specific generation
            file regardless of recency or memmebr count
        output generation_id: int
        output generation: dict, keys ["members", "parents"]
        '''
        generations = os.listdir(self.generation_dir)
        if self.verbose:
            print("Found {} generations on file, seeding new generation".format(
                len(generations)))
        generations.sort()
        num_generations = len(generations)
        population = {}
        num_members = -1
        i = 0
        if len(generation_file) < 1:
            while num_members < min_member_count:
                i+= 1
                if i > num_generations:
                    raise ValueError("No Generations meet min_members >= {} requirement".format(
                        min_members))
                generation_file = generations[-i]
                with open(os.path.join(self.generation_dir, generation_file), 'r') as f:
                    generation = json.load(f)
                num_members = len(generation["members"])
        else:
            with open(os.path.join(self.generation_dir, generation_file), 'r') as f:
                generation = json.load(f)
        generation_id = generation_file.split(".")[0]
        for member_id in generation["members"]:
            with open(os.path.join(self.population_dir, str(member_id) + ".json"), 'r') as f:
                population[member_id] = json.load(f)
        return generation_id, generation, population

    def create_fitness_df(self, population):
        df = pd.DataFrame.from_dict(population, orient="index")
        df.index.rename("member_id", inplace=True)
        df.reset_index(inplace=True)
        df = df[["member_id", "fitness"]].copy()
        df.sort_values("fitness", ascending=self.fitness_lower_is_better, inplace=True)
        if self.verbose:
            print("Fitness DF")
            print("="*30)
            print(df.head())
            print("="*30)
        return df

    def choose_parents(self):
        self.prev_generation_df = self.create_fitness_df(self.population)
        self.parents = self.prev_generation_df["member_id"][:max(2,
            int(self.prev_generation_df.shape[0] * self.elitism_ratio))].tolist()
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

    def save_generation(self, delete_partials=False):
        '''
        input delete_partials: bool, will delete earlier saved files of this same generations.
                                    Useful for 
        '''
        out_dict = {"members": list(self.population.keys()), "parents": self.parents}
        generation_file_name = os.path.join(self.generation_dir, 
                str(int(time.time() * 10e6)) + ".json")
        with open(generation_file_name , 'w') as f:
            json.dump(out_dict, f)
        if delete_partials and (len(self.previous_generation_file_name)>1):
            os.remove(self.previous_generation_file_name)
        self.previous_generation_file_name = generation_file_name

    def last_generation_info_from_file(self, min_member_count=0, 
            save_to_new_dir=False, new_dir=""):
        '''
        This function is meant to look through the file system for the most recent generation.
        Useful if you want to just transfer the latest generation to a new computer and seed 
            the algorithm futher (think saving generation to git to pick up at a later date) 
        input min_member_count: int, Will search for most recent generation whose
            len(members) >= min_member_count. Default 0 will just give the most recent 
            generation file.
        input save_to_new_dir: bool, if True, will create new file directory with just the 
            most recent generatio.
        input new_dir: str, if empty string, will just append "_latest" to ga.output_dir
        output parents: list, parent id's
        output fitness_df: dataframe, columns ["member_id", "fitness"]
        '''
        generation_id, generation, population = self.load_generation_from_file(
                        min_member_count=min_member_count) 
        if self.verbose:
            print("Generation ID: {}".format(generation_id))
            print("Parents: {}".format(generation["parents"]))
        fitness_df = self.create_fitness_df(population)
        if save_to_new_dir:
            if len(new_dir) < 1:
                path_tup = os.path.split(self.output_dir)
                new_dir = os.path.join(path_tup[0],path_tup[1] + "_latest")
            os.makedirs(new_dir, exist_ok=True)
            new_generation_dir = os.path.join(new_dir, "generations")
            os.makedirs(new_generation_dir, exist_ok=True)
            generation_file = str(generation_id) + ".json"
            shutil.copyfile(os.path.join(self.generation_dir, generation_file),
                os.path.join(new_generation_dir, generation_file))
            new_population_dir = os.path.join(new_dir, "population")
            os.makedirs(new_population_dir, exist_ok=True)
            for member_id in fitness_df["member_id"].tolist():
                member_file = str(member_id) + ".json"
                shutil.copyfile(os.path.join(self.population_dir, member_file),
                    os.path.join(new_population_dir, member_file))
        return generation["parents"], fitness_df

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
        mutation_array = np.around(mutate_map 
                * self.mod_array * 0.5 * np.random.randn(self.chromosome_len))
        return chromosome + mutation_array.astype(int)
