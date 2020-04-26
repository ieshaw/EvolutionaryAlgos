import numpy as np
import os
import shutil
import time
import unittest

from ga import ga

np.random.seed(0)

def express_chromosome(chromosome):
    '''
    input chromosome: numpy array
    output expression: however the chromosome is to be expressed, 
            in the this test case, a dictionayr
    '''
    express_dict = {}
    express_dict["int_test"] = chromosome[0]
    some_list = ["a", "b", "c"]
    express_dict["list_test"] = some_list[chromosome[1]]
    express_dict["bool_test"] = chromosome[2]
    return express_dict

class TestGA(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.out_dir = "test_output/"
        cls.gene_map = [("int_test", 1, 4),
                    ("list_test", 0, 2),
                    ("bool_test", 0, 1)]

    def test_init(cls):
        test_ga = ga(
                output_dir=cls.out_dir,
                gene_map= cls.gene_map)
        cls.assertEqual(3, test_ga.chromosome_len)
        np.testing.assert_array_equal(np.array([4, 3, 2]), test_ga.mod_array)
        np.testing.assert_array_equal(np.array([1, 0, 0]), test_ga.base_array)
        cls.assertTrue(os.path.exists(cls.out_dir))

    def test_gen_child_new(cls):
        test_ga = ga(
                output_dir=os.path.join(cls.out_dir, str(int(time.time()*10e6))),
                gene_map= cls.gene_map)
        with cls.assertRaises(ValueError) as e:
            test_ga.load_population()
        test_ga.load_population(start_fresh=True)
        child_id, chromosome = test_ga.gen_child()

    def test_save_member(cls):
        test_ga = ga(
                output_dir=os.path.join(cls.out_dir, str(int(time.time()*10e6))),
                gene_map= cls.gene_map)
        test_ga.load_population(start_fresh=True)
        child_id, chromosome = test_ga.gen_child()
        test_ga.save_member(child_id)
        cls.assertTrue(os.path.exists(os.path.join(test_ga.population_dir, 
                    str(child_id) + ".json")))

    def test_save_generation(cls):
        test_ga = ga(
                output_dir=os.path.join(cls.out_dir, str(int(time.time()*10e6))),
                gene_map= cls.gene_map)
        test_ga.load_population(start_fresh=True)
        for i in range(20):
            child_id, chromosome = test_ga.gen_child()
            test_ga.population[child_id]["fitness"] = i
        cls.assertEqual(len(os.listdir(test_ga.generation_dir)), 0)
        test_ga.save_generation()
        cls.assertTrue(len(os.listdir(test_ga.generation_dir)) > 0)

    def test_load_generation_from_memory(cls):
        test_ga = ga(
                output_dir=os.path.join(cls.out_dir, str(int(time.time()*10e6))),
                gene_map= cls.gene_map,
                elitism_ratio = 0.2,
                fitness_lower_is_better=True)
        test_ga.load_population(start_fresh=True)
        for i in range(20):
            child_id, chromosome = test_ga.gen_child()
            test_ga.population[child_id]["fitness"] = i
        cls.assertEqual(len(list(test_ga.population.keys())), 20)
        test_ga.load_population(start_fresh=False)
        cls.assertEqual(len(list(test_ga.population.keys())), 4)
        cls.assertEqual(test_ga.population[test_ga.parents[0]]["fitness"], 0)

    def test_load_generation_from_memory_descending_fitness(cls):
        test_ga = ga(
                output_dir=os.path.join(cls.out_dir, str(int(time.time()*10e6))),
                gene_map= cls.gene_map,
                elitism_ratio = 0.2,
                fitness_lower_is_better=False)
        test_ga.load_population(start_fresh=True)
        for i in range(20):
            child_id, chromosome = test_ga.gen_child()
            test_ga.population[child_id]["fitness"] = i
        cls.assertEqual(len(list(test_ga.population.keys())), 20)
        test_ga.load_population(start_fresh=False)
        cls.assertEqual(len(list(test_ga.population.keys())), 4)
        cls.assertEqual(test_ga.population[test_ga.parents[0]]["fitness"], 19)

    def test_load_generation_from_file(cls):
        out_dir = os.path.join(cls.out_dir, str(int(time.time()*10e6)))
        test_ga = ga(
                output_dir=out_dir,
                gene_map= cls.gene_map,
                elitism_ratio = 0.2,
                fitness_lower_is_better=True)
        test_ga.load_population(start_fresh=True)
        for i in range(20):
            child_id, chromosome = test_ga.gen_child()
            test_ga.population[child_id]["fitness"] = i
            test_ga.save_member(child_id)
        test_ga.save_generation()
        #Generate some extras not in the generation
        for i in range(20):
            child_id, chromosome = test_ga.gen_child()
            test_ga.population[child_id]["fitness"] = -1*i
            test_ga.save_member(child_id)
        del test_ga
        new_ga = ga(
                output_dir=out_dir,
                gene_map= cls.gene_map,
                elitism_ratio = 0.2,
                fitness_lower_is_better=True)
        new_ga.load_population(start_fresh=False)
        cls.assertEqual(len(list(new_ga.population.keys())), 4)
        cls.assertEqual(new_ga.population[new_ga.parents[0]]["fitness"], 0)

    def test_crossover(cls):
        test_ga = ga(
                output_dir=os.path.join(cls.out_dir, str(int(time.time()*10e6))),
                gene_map= cls.gene_map,
                elitism_ratio = 0.2,
                mutation_ratio = 1,
                fitness_lower_is_better=True)
        test_ga.population[0] = {"chromosome":np.zeros(len(cls.gene_map))}
        test_ga.population[1] = {"chromosome":np.ones(len(cls.gene_map))}
        chromosome = test_ga.crossover(0,1)
        cls.assertEqual(chromosome.max(), 1)
        cls.assertEqual(chromosome.min(), 0)

    def test_mutate(cls):
        test_ga = ga(
                output_dir=os.path.join(cls.out_dir, str(int(time.time()*10e6))),
                gene_map= cls.gene_map,
                elitism_ratio = 0.2,
                mutation_ratio = 1,
                fitness_lower_is_better=True)
        chromosome = np.zeros(len(cls.gene_map))
        cls.assertTrue(test_ga.mutate(chromosome).sum() > 0)

    def test_breeding(cls):
        test_ga = ga(
                output_dir=os.path.join(cls.out_dir, str(int(time.time()*10e6))),
                gene_map= cls.gene_map,
                elitism_ratio = 0.2,
                fitness_lower_is_better=True)
        test_ga.load_population(start_fresh=True)
        for i in range(20):
            child_id, chromosome = test_ga.gen_child()
            test_ga.population[child_id]["fitness"] = i
        test_ga.load_population(start_fresh=False)
        _, c1 = test_ga.gen_child() 

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.out_dir)

if __name__ == '__main__':
    unittest.main()
