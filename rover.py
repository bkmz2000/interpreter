from random import randint
from time import time, sleep
from os.path import exists
from os import remove
from robots import RobotManager
from robo_parser import tokenize, Parser


def main():
    id = randint(1000000, 10000000)
    name = str(id) + '.txt'

    print(f'Rover started, id={id}')
    worldf = open('map.txt', 'r')

    world = [list(line) for line in worldf.read().split('\n')]
    if not world[-1]:
        world = world[:-1]

    start = time()
    rm = RobotManager(world)

    while time() - start < 60 * 3:
        sleep(1)
        if exists(name):
            print('Got a command')

            file = open(name, 'r')

            code = file.read()

            tokens = tokenize(code)
            parser = Parser(tokens)
            ast = parser.getAst()

            ast.check({})

            start = time()
            ast.execute({}, rm)
            stop = time()

            print(f'Done in {stop - start}ms')
            file.close()
            remove(name)

    print(f'Rover {id} is out of power')


if __name__ == '__main__':
    main()
