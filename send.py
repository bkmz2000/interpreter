from sys import argv
from robo_parser import tokenize, Parser
from robots import RobotManager
from time import time

# TODO runner

if len(argv) != 3:
    raise RuntimeError(f'I expect exactly two arguments, got {argv}')

name = argv[2]

if '.' not in name:
    print('added')
    name += '.txt'

file = open(argv[1], 'r')
new = open(name, 'w')

text = file.read()

new.write(text)

file.close()
new.close()

