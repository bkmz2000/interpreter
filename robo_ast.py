from typing import Union, Any
from re import fullmatch, compile
from node_type import NodeType

# These should be the same as in robo_parser!

Reals = compile(r"-?(0|[1-9][0-9]*)(\.[0-9]+)?")  # noqa
Nums = compile("-?(0|[1-9][0-9]*)")  # noqa
Ids = compile("[a-zA-Z_][a-zA-Z_0-9]*")  # noqa


class AstNode:
    def __init__(self, *chlds: Union['AstNode', str]):
        self._children: list[Union['AstNode', str]] = list(chlds)
        self._type: NodeType = NodeType.Null()
        self.result: Any = None

    def get_type(self):
        return self._type

    def execute(self, ctx, robo_manager):
        pass

    def check(self, ctx: dict[str, NodeType]):
        for ch in self._children:
            if isinstance(ch, AstNode):
                ch.check(ctx)

        self.check_type(ctx)

    def check_type(self, ctx: dict[str, NodeType]):
        pass

    def make_repr(self, level: int = 0):
        # ``level`` for level of nesting

        ret = '  ' * level + f'{type(self).__name__}('

        if not self._children:
            return ret + ')'

        if len(self._children) == 1:
            r = repr(self._children[0])
            if r.count('\n') == 0:
                return ret + r + ')'

        ret += '\n'

        chlds = []

        for chld in self._children:
            if type(chld) == str:
                chlds.append('  ' * (level + 1) + '"' + chld + '"')

            elif isinstance(chld, AstNode):
                chlds.append(chld.make_repr(level + 1))

        chlds = ',\n'.join(chlds)
        ret += chlds + '\n' + '  ' * level + ')'
        return ret

    def __repr__(self):
        # The output of this function is a valid Python code!
        # It's also quite readable

        return self.make_repr()

    def add_child(self, node: Union['AstNode', str]):
        self._children.append(node)

    def error(self, s):
        raise Exception(s)


class Program(AstNode):
    def __init__(self, block):
        super().__init__(block)
        self.block: 'Block' = block

    def execute(self, ctx, robo_manager):
        self.block.execute(ctx, robo_manager)


class Block(AstNode):
    def __init__(self, decls, stmts):
        super().__init__(decls, stmts)
        self.decls: 'Decls' = decls
        self.stmts: 'Stmts' = stmts

    def execute(self, ctx, robo_manager):
        self.decls.execute(ctx, robo_manager)
        self.stmts.execute(ctx, robo_manager)


class Decls(AstNode):
    def execute(self, ctx, robo_manager):
        for ch in self._children:
            ch.execute(ctx, robo_manager)


class Decl(AstNode):
    def __init__(self, t: 'Type', name: str):
        super().__init__(t, name)

        self.name: str = name
        # This is possible because the type of Type node is known in "compile time"
        t.check({})
        self._type: NodeType = t.get_type()

    def execute(self, ctx, defauolt=None):
        if self.name in ctx.keys():
            self.error('Runtime error: {self.name} is already defined')

        ctx[self.name] = self._type.default_value()

    def check_type(self, ctx: dict[str, NodeType]):
        ctx[self.name] = self._type


class Type(AstNode):
    # Here type can be derived in "compile time"
    def __init__(self, basic: str, *dims: int):
        super().__init__(basic, *dims)
        if dims:
            t = NodeType.from_str(basic)

            self._type = NodeType.Array(t, dims)

        else:
            self._type = NodeType.from_str(basic)


class Stmts(AstNode):
    def execute(self, ctx, robo_manager):
        for ch in self._children:
            ch.execute(ctx, robo_manager)


class Stmt(AstNode):
    def __init__(self, name, *args):
        super().__init__(name, *args)
        self.name = name

    def execute_assign(self, ctx, robo_manager):
        lhs = self._children[1]
        rhs = self._children[2]

        rhs.execute(ctx, robo_manager)

        name = lhs._children[0]

        if type(ctx[name]) == list:
            def change(li, idx, v, to_be_reversed=True):
                if len(idx) == 1:
                    li[idx[0]] = v
                else:
                    if to_be_reversed:
                        idx = list(reversed(idx))

                    change(li[idx[-1]], idx[:-1], v, False)

            idx = []
            var = ctx[name]
            for ch in lhs._children[1:]:
                ch.execute(ctx, robo_manager)
                idx.append(ch.result)

            change(var, idx, rhs.result)

        else:
            ctx[name] = rhs.result

    def execute_if(self, ctx, robo_manager):
        cond = self._children[1]
        body = self._children[2]
        cond.execute(ctx, robo_manager)

        if cond.result:
            body.execute(ctx, robo_manager)

        elif len(self._children) > 3:
            else_body = self._children[3]
            else_body.execute(ctx, robo_manager)

    def execute_while(self, ctx, robo_manager):
        cond = self._children[1]
        body = self._children[2]
        cond.execute(ctx, robo_manager)

        while cond.result:
            body.execute(ctx, robo_manager)
            cond.execute(ctx, robo_manager)

    def execute(self, ctx, robo_manager):
        if self.name == 'assign':
            self.execute_assign(ctx, robo_manager)

        if self.name == 'if':
            self.execute_if(ctx, robo_manager)

        if self.name == 'while':
            self.execute_while(ctx, robo_manager)

        if self.name == 'block':
            self._children[1].execute(ctx, robo_manager)


class Method(AstNode):
    def __init__(self, name):
        super().__init__(name)
        self.name = name

    def execute(self, ctx, robo_manager):
        if self.name == 'turn_right':
            robo_manager.turn_right(0)
        if self.name == 'turn_left':
            robo_manager.turn_left(0)
        if self.name == 'move':
            robo_manager.move(0)

        if self.name == 'freeze':
            robo_manager.freeze(0)
        if self.name == 'melt':
            robo_manager.melt(0)
        if self.name == 'build':
            robo_manager.build(0)
        if self.name == 'explode':
            robo_manager.explode(0)

        if self.name == 'print_info':
            robo_manager.print_info(0)
        if self.name == 'drill':
            robo_manager.drill(0)


class Bif(AstNode):
    def __init__(self, name, arg):
        super().__init__(name, arg)
        self.name = name
        self.arg = arg

    def execute(self, ctx, robo_manager):
        if self.name == 'print':
            self.arg.execute(ctx, robo_manager)
            print(self.arg.result)

        if self.name == 'print_map':
            robo_manager.print_map()


class Loc(AstNode):
    def __init__(self, name, *dims: AstNode):
        super().__init__(name, *dims)
        self.name: str = name
        self.dims: list[AstNode] = list(dims)

    def execute(self, ctx, robo_manager):
        res = ctx[self.name]

        if self.dims:
            for d in self.dims:
                d.execute(ctx, robo_manager)
                res = res[d.result]

        self.result = res

    def check_type(self, ctx: dict[str, NodeType]):
        if self.name not in ctx.keys():
            self.error(f'Variable {self.name} is not defined')

        if self.dims:
            if not ctx[self.name].is_array():
                self.error(f'Type {ctx[self.name]} is not subscriptable')

            dim = len(ctx[self.name].get_dims()) - len(self.dims)

            if dim < 0:
                self.error(f'Type {ctx[self.name]} subscripted too many times')

            if dim == 0:
                self._type = NodeType.same_as(ctx[self.name])

            else:
                self._type = NodeType.Array(ctx[self.name].get_type(), ctx[self.name].get_dims()[:-dim])

        else:
            self._type = ctx[self.name]


class Bool(AstNode):
    def execute(self, ctx, robo_manager):
        if len(self._children) == 1:
            self._children[0].execute(ctx, robo_manager)
            self.result = self._children[0].result
            return

        for ch in self._children:
            ch.execute(ctx, robo_manager)
            if ch.result is True:
                self.result = True
                return

        self.result = False

    def check_type(self, ctx: dict[str, NodeType]):
        self._type = self._children[0]._type


class Join(AstNode):
    def execute(self, ctx, robo_manager):
        if len(self._children) == 1:
            self._children[0].execute(ctx, robo_manager)
            self.result = self._children[0].result
            return

        for ch in self._children:
            ch.execute(ctx, robo_manager)
            if ch.result is False:
                self.result = False
                return

        self.result = True

    def check_type(self, ctx: dict[str, NodeType]):
        self._type = self._children[0]._type


class BinOp(AstNode):
    def __init__(self, lhs: AstNode, op: str, rhs: AstNode):
        super().__init__(lhs, op, rhs)
        self.lhs = lhs
        self.op = op
        self.rhs = rhs

    def check_type(self, ctx: dict[str, NodeType]):
        lhst: NodeType = self._children[0]._type
        rhst: NodeType = self._children[2]._type

        if self.op in '+-*/':
            self.check_arithmetic(lhst, rhst)

        elif self.op in ['<', '>', '>=', '<=', '==', '!=']:
            self.check_comp(lhst, rhst)

        else:
            self.error('Unknown binary operator')  # Hopefully is never executed

    def error(self, s=''):
        lhst = self._children[0]._type
        rhst = self._children[2]._type

        super().error(f'Binary operator {self.op} is not defined for type {lhst} and {rhst}.')

    def check_arithmetic(self, lhst: NodeType, rhst: NodeType):
        if rhst != lhst or rhst not in [NodeType.Int(), NodeType.Double()]:
            self.error()

        self._type = rhst

    def check_comp(self, lhst: NodeType, rhst: NodeType):
        if rhst != lhst or rhst not in [NodeType.Int(), NodeType.Double()]:
            self.error()

        self._type = NodeType.Bool()

    def execute(self, ctx, robo_manager):
        actions = {'+': lambda x, y: x + y,
                   '-': lambda x, y: x - y,
                   '/': lambda x, y: x / y,
                   '*': lambda x, y: x * y,
                   '<': lambda x, y: x < y,
                   '>': lambda x, y: x > y,
                   '<=': lambda x, y: x <= y,
                   '>=': lambda x, y: x >= y,
                   '==': lambda x, y: x == y,
                   '!=': lambda x, y: x != y}

        self.lhs.execute(ctx, robo_manager)
        self.rhs.execute(ctx, robo_manager)

        lhs = self.lhs.result
        rhs = self.rhs.result

        t = int

        if self.rhs.get_type() == NodeType.Double():
            t = float

        if self.op in ['>', '<', '>=', '<=', '==', '!=']:
            t = bool

        self.result = t(actions[self.op](lhs, rhs))


class Unary(AstNode):
    def __init__(self, prefix: str, value: AstNode):
        super().__init__(prefix, value)

        self.prefix = prefix
        self.value = value

    def check_type(self, ctx: dict[str, NodeType]):
        if self._children[1].get_type() in [NodeType.Int(), NodeType.Double()]:
            if '!' in self.prefix:
                self.error(f'Unary operator ! is not defined for {self._children[1]._type}.')

        if self._children[1].get_type() == NodeType.Bool():
            if '-' in self.prefix:
                self.error(f'Unary operator - is not defined for boolean values.')

        self._type = self.value._type

    def execute(self, ctx, robo_manager):
        self.value.execute(ctx, robo_manager)
        res = self.value.result

        for c in self.prefix:
            if c == '!':
                res = not res

            if c == '-':
                res = - res

        self.result = res


class Property(AstNode):
    def __init__(self, name):
        super().__init__(name)
        self.name = name

    def execute(self, ctx, robo_manager):
        if self.name == 'x':
            self.result = robo_manager.robots[0].x

        if self.name == 'y':
            self.result = robo_manager.robots[0].y

        if self.name == 'dir':
            self.result = int(robo_manager.robots[0].direction)

        if self.name == 'under':
            self.result = robo_manager.under(0)

        if self.name == 'in_front':
            self.result = robo_manager.in_front(0)

    def check_type(self, ctx: dict[str, NodeType]):
        self._type = NodeType.Int()


class Factor(AstNode):
    def __init__(self, val: Union[AstNode, str]):
        super().__init__(val)
        self.val = val

    def execute(self, ctx, robo_manager):
        if type(self.val) == str:
            if self.val in ['true', 'false']:
                self.result = {'true': True, 'false': False}[self.val]
                return

            elif fullmatch(Ids, self.val):
                if self.val not in ctx.keys():
                    self.error(f'{self.val} is not defined')

                self.result = ctx[self.val]

            elif fullmatch(Nums, self.val):
                self.result = int(self.val)

            elif fullmatch(Reals, self.val):
                self.result = float(self.val)

        else:
            self.val.execute(ctx, robo_manager)
            self.result = self.val.result

    def check_type(self, ctx: dict[str, NodeType]):
        if type(self.val) == Bool:
            self._type = self.val._type

        elif type(self.val) == Loc:
            self._type = self.val._type

        elif self.val in ['true', 'false']:
            self._type = NodeType.Bool()

        elif fullmatch(Ids, self.val):
            if self.val not in ctx.keys():
                self.error(f'{self.val} is not defined')

            self._type = ctx[self.val]

        elif fullmatch(Nums, self.val):
            self._type = NodeType.Int()

        elif fullmatch(Reals, self.val):
            self._type = NodeType.Double()
