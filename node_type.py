from typing import Optional


# Works like np.full
# There should be a more verbose way to implement this
# but this is good for now
def fill_array(dims: list[int], v):
    if len(dims) == 1:
        return [v for _ in range(int(dims[0]))]

    return [fill_array(dims[:len(dims) - 1], v) for _ in range(int(dims[-1]))]


# Type.Null means that the node has no type
class NodeType:
    _Int, _Double, _Bool, _Null, _Char, _Array = range(6)

    def __init__(self, t: int, basic=None):
        if basic is None:
            basic = t

        self._type: int = t
        self._basic: int = basic
        self._dims: Optional[list[int]] = None

    @classmethod
    def Int(cls):
        ret = NodeType(NodeType._Int)
        return ret

    @classmethod
    def Double(cls):
        ret = NodeType(NodeType._Double)
        return ret

    @classmethod
    def Bool(cls):
        ret = NodeType(NodeType._Bool)
        return ret

    @classmethod
    def Null(cls):
        ret = NodeType(NodeType._Null)
        return ret

    @classmethod
    def Char(cls):
        ret = NodeType(NodeType._Char)
        return ret

    @classmethod
    def Array(cls, basic, dims):
        ret = NodeType(basic)
        ret._type = NodeType._Array
        ret._dims = list(dims)
        return ret

    def same_as(self) -> 'NodeType':
        rets: dict[int, 'NodeType'] = {NodeType._Int: NodeType.Int(),
                                       NodeType._Double: NodeType.Double(),
                                       NodeType._Bool: NodeType.Bool(),
                                       NodeType._Null: NodeType.Null(),
                                       NodeType._Char: NodeType.Char()}

        return rets[self._basic]

    def get_type(self):
        return self._type

    def get_dims(self):
        return self._dims

    def get_basic(self):
        return self._basic

    @classmethod
    def from_str(cls, s):
        named = {'int': NodeType.Int(), 'double': NodeType.Double(), 'bool': NodeType.Bool(),
                 'char': NodeType.Char()}

        return named[s]

    def default_value(self):
        defaults = {
            NodeType._Int: 0,
            NodeType._Double: 0,
            NodeType._Bool: False,
            NodeType._Array: None,
            NodeType._Char: ' '
        }

        if self.is_array():
            return fill_array(self._dims, defaults[self._basic])

        return defaults[self._type]

    def __eq__(self, other):
        if type(other) != NodeType:
            return False

        return (self._type, self._basic, self._dims) == (other.get_type(), other.get_basic(), other.get_dims())

    def is_array(self):
        return self._dims is not None

    def __str__(self):
        names = {NodeType._Int: 'Int',
                 NodeType._Double: 'Double',
                 NodeType._Bool: 'Bool',
                 NodeType._Null: 'NullType',
                 NodeType._Char: 'Char'}

        if self._type == NodeType._Array:
            return f'Array({names[self._basic]}, {self._dims})'

        return names[self._basic]

    def __repr__(self):
        return str(self)
