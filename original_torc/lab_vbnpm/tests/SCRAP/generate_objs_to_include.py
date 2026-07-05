"""
given a set of object names, randomly generate a list of objects to include in the test
"""
import random
def random_gen(n_objs=7):
    obj_list = ["001_chips_can", "003_cracker_box", "004_sugar_box", "005_tomato_soup_can", "006_mustard_bottle",
                "008_pudding_box", "010_potted_meat_can", "011_banana", "013_apple", "017_orange"]
    return random.sample(obj_list, n_objs)

if __name__ == "__main__":
    print(random_gen(7))