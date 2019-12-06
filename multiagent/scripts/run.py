# Manipulate the yaml configuration file for running tests.

import subprocess
import yaml
import numpy as np
from itertools import product


# Take a yaml filename and a dictionary, and use the key-value pairs to
# update the fields and values in the file.
def update(filename, changesdict):
    with open(filename) as f:
        current_dict = yaml.load(f)
    for key in changesdict:
        current_dict[key] = changesdict[key]
    with open(filename, 'w') as f:
        yaml.dump(current_dict, f)
        

# Take a yaml filename and return a tuple of the keys and a tuple of
# the values.
def get_tuples(filename):
    with open(filename) as f:
        d = yaml.load(f)
    keys = tuple(d.keys())
    vals = tuple(d.values())
    return keys, vals
        

# First read in the metaconfig file with all the different hyperparameters
# to be tested. Then change the config file, run train.py, and repeat.
if __name__ == '__main__':
    keys, vals = get_tuples('metaconfig.yml')

    combos = list(product(*vals))
    
    for elem in combos:
        mydict = dict(zip(keys, elem))
        update('maopac_config.yml', mydict)
        subprocess.call(['python', 'maopac_test.py'])


















# end
