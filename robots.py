from enum import IntEnum, auto
from random import choice


class Direction(IntEnum):
    Up = auto()
    Down = auto()
    Left = auto()
    Right = auto()


class Robot:
    def __init__(self, x: int, y: int, d: Direction, parent: 'RobotManager' = None):
        self.x = x
        self.y = y
        self.minerals = 0
        self.direction = d
        self.message = ''
        self.parent = parent
        self.got_new_message = False

    def set_parent(self, par: 'RobotManager'):
        self.parent = par

    def next_pos(self):
        dr = {
            Direction.Up: (0, 1),
            Direction.Down: (0, -1),
            Direction.Right: (1, 0),
            Direction.Left: (-1, 0)
        }

        dx, dy = dr[self.direction]

        return self.x + dx, self.y + dy

    def move(self):
        self.x, self.y = self.next_pos()

    def turnRight(self):
        right = {
            Direction.Up: Direction.Right,
            Direction.Down: Direction.Left,
            Direction.Right: Direction.Down,
            Direction.Left: Direction.Up
        }

        self.direction = right[self.direction]

    def turnLeft(self):
        left = {
            Direction.Up: Direction.Left,
            Direction.Down: Direction.Right,
            Direction.Right: Direction.Up,
            Direction.Left: Direction.Down
        }

        self.direction = left[self.direction]

    # Implemented this way to be compatible with a method in the language
    def getDir(self) -> int:
        index = {Direction.Up: 0,
                 Direction.Down: 1,
                 Direction.Left: 2,
                 Direction.Right: 3}

        return index[self.direction]

    def what_is_under(self) -> int:
        return self.parent.at(self.x, self.y)

    def what_is_in_front(self):
        return self.parent.in_front(0)

    def drill(self):
        if self.parent.at(self.x, self.y) == ord('D'):
            self.minerals += 1

    def explode(self):
        x, y = self.next_pos()

        if self.parent.at(x, y) == ord('X'):
            self.parent.explode(x, y)

    def build(self):
        x, y = self.next_pos()

        if self.parent.at(x, y) == ord(' '):
            self.parent.build(x, y)

    def melt(self):
        x, y = self.next_pos()

        if self.parent.at(x, y) == ord('I'):
            self.parent.melt(x, y)

    def freeze(self):
        x, y = self.next_pos()

        if self.parent.at(x, y) == ord('W'):
            self.parent.freeze(x, y)

    def print_info(self):
        d = self.direction._name_
        print(f'Robot(x={self.x},'
              f' y={self.y}, '
              f'dir={d}, '
              f'in_front="{chr(self.what_is_in_front())}", '
              f'under="{chr(self.what_is_under())}")')


class RobotManager:
    def __init__(self, world: list[list[str]], robots=None):
        self.world = world
        self.robots = robots

        # I assume that world is rectangular and each line has the same length
        self.width = len(world)
        self.height = len(world[0])

        empty = []

        for x in range(self.width):
            for y in range(self.height):
                if self.world[x][y] == ' ':
                    empty.append((x, y))

        if not robots:
            x, y = choice(empty)
            d = choice([Direction.Right,
                        Direction.Left,
                        Direction.Down,
                        Direction.Up])

            self.robots = [Robot(x, y, d, self)]

    def drill(self, index=0):
        self.robots[index].drill()

    def print_info(self, index=0):
        self.robots[index].print_info()

    def turn_right(self, index=0):
        self.robots[index].turnRight()

    def turn_left(self, index=0):
        self.robots[index].turnLeft()

    def move(self, index=0):
        x, y = self.robots[index].next_pos()

        if self.at(x, y) not in [ord('X'), ord('W')]:
            self.robots[index].move()

    def in_front(self, index=0) -> int:
        rover = self.robots[index]
        x, y, d = rover.x, rover.y, rover.direction

        dr = {
            Direction.Up: (0, 1),
            Direction.Down: (0, -1),
            Direction.Right: (1, 0),
            Direction.Left: (-1, 0)
        }

        dx, dy = dr[d]

        x += dx
        y += dy

        return self.at(x, y)

    def under(self, index) -> int:
        rover = self.robots[index]
        x, y = rover.x, rover.y

        return self.at(x, y)

    def at(self, x, y) -> int:
        return ord(self.world[x][y])

    def freeze(self, index):
        x, y = self.robots[index].next_pos()

        if self.at(x, y) == ord('W'):
            self.world[x][y] = 'I'

    def melt(self, index):
        x, y = self.robots[index].next_pos()
        if self.at(x, y) == ord('I'):
            self.world[x][y] = 'W'

    def explode(self, index):
        x, y = self.robots[index].next_pos()
        if self.at(x, y) == ord('X'):
            self.world[x][y] = ' '

    def build(self, index):
        x, y = self.robots[index].next_pos()

        if self.at(x, y) == ord(' '):
            self.world[x][y] = 'X'

    def print_map(self):
        arrows = {Direction.Left: '←',
                  Direction.Right: '→',
                  Direction.Up: '↑',
                  Direction.Down: '↓'}

        # weird iteration, but that is intentional
        for y in range(self.height - 1, -1, -1):
            for x in range(self.width):
                symb = self.world[x][y]

                if (x, y) == (self.robots[0].x, self.robots[0].y):
                    symb = arrows[self.robots[0].direction]

                print(symb, end='')
            print()
