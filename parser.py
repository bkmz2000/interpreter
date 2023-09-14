from typing import Union, Optional
from enum import Enum
from sys import argv

from re import compile, fullmatch

'''
    This program can parse such as the following, which contains every syntactic feature of the language.
    
    {
     int x ;
        bool [ 2 ] [ 4 ] [ 9 ] [ 5 ] _ ;
        char C123_nlksfnsfkdn ;
        double __00___11 ;
    
        x [ 2 + 3 * 5 ] = 3 ;
        x [ 2 ]                                    = w ;
        x [ 2            == 3 ] = 2 ;
        x [ 2 && 3 || false && true ] = 2 ;
        x [    2 || 3 ] = 2 ;
        x [ ( 2 ==             3 ) || p ] = 2 ;
        x [ 2 && 3 || 4 && 5 ] = 2 ;
    
        x [ ( 2 == ! ! ! - - - -8 ) && ( 3 < 1 ) || ( 4 - 4 == 0 ) && ( 5 == 6 ) ] = 2 ;
        x [ ( 2 == 8 + 1 ) && ( 3 - 4 < 1 ) || ( 4 - 4 == 0 ) && ( 5 == 6 ) ] = 2 ;
        x [ ( 2 == 8 + 1 ) && ( 3 - 4 < 1 ) || ( 4 - 4 == 0 ) && ( 5 == 6 ) ] = x [ ( 2 == ! ! ! - - - -8 ) && ( 3 < 1 ) || ( 4 - 4 == 0 ) && ( 5 == 6 ) ] ;
        if ( 2 ) if ( 3 )      { if ( ( 2 == ! ! ! - - - -8 ) && ( 3 < 1 ) || ( 4 - 4 == 0 ) && ( 5 == 6 ) ) while ( q == p ) { x = 3 ; x = -3 ; } }
    }  
    
    It also provides some basic error handling via exceptions. To test that out, try making some syntax errors in your 
    program, you will get some readable output and a (hopefully valid) suggestion on how to fix it.
    
    Using regex could be an overkill, but it would be easy to add new features.
    
    gettoken() is implemented as a method of ``TokenStream`` class and is never used, because functions ``is_exact``, 
    ``is_any``, ``advance_if_is`` and ``advance_if_any`` provide the same functionality as well as easier error handling. 
'''

'''
    Here is the initial grammar:
	<program> ::= <block>
	<block> ::= { <decls> <stmts> }
	<decls> ::= <decls> <decl> | ε
	<decl> ::= <type> ID ;
	<type> ::= <type> [ NUM ] | BASIC
	<stmts> ::= <stmts>hstmt>| ε
	<stmt> ::= <loc>=<bool>;
	| IF (<bool>) <stmt>
	| IF (<bool>) <stmt> ELSE <stmt>
	| WHILE ( <bool> ) <stmt>
	| <block>
	<loc> ::= <loc> [ <bool> ] | ID
	<bool> ::= <bool> || <join> | <join>
	<join> ::= <join> && <equality> | <equality>
	<equality> ::= <equality>==<rel>
	| <equality> != <rel>
	| <rel>
	<rel> ::= <expr><<expr>
	| <expr> <= <expr>
	| <expr> >= <expr>
	| <expr> > <expr>
	| <expr>
	<expr> ::= <expr> + <term>
	| <expr> - <term>
	| <term>
	<term> ::= <term> * <unary>
	| <term> / <unary>
	| <unary>
	<unary> ::= !<unary>
	| -<unary>
	| <factor>
	<factor >::= (<bool>)
	| <loc>
	| NUM
	| REAL
	| TRUE
	| FALSE

It is impossible to immediately write a recursively descending parser because of two reasons:
1) Some of the rules are left recursive ie can be simplified to the form of 
	A ::= A ...
2) The rule on line 48 is a prefix of the rule on line 49.

So here is the grammar that does not have these flaws and is equivalent to the former (I use ebnf for readability):
	program ::= block
	block ::=  "{" decls stmts "}"
	decls ::= (decl)*
	decl ::= type ID ";"
	type ::= BASIC ("[" NUM "]")*
	stmts ::= (stmt)*
	stmt ::= loc "=" bool ";"
	| "if" "(" bool ")" stmt ("else" stmt)?
	| "while" "(" bool ")" stmt
	| block
	loc ::= ID ("[" bool "]")*
	bool ::= join ("||" join)*
	join ::= eq ("&&" eq)*
	eq ::= rel (("==" | "!=") rel)*
	rel ::= expr (("<" | ">" | "<=" | ">=") expr)*
	expr ::= term (("+" | "-") term)*
	term ::= unary (("*" | "/") unary)*
	unary ::=  ("!" | "-")* factor
	factor ::=  "(" bool ")"
	| loc
	| NUM
	| REAL
	| TRUE
	| FALSE

I used several technics:
1) Rules <decls>, <loc>, <bool>, <join>, <equality>, <term>, <expr>, <unary> and <stmts> were rewritten from 
	A ::= A B | C
to 
A ::= C (B)*
2) Rule <stmt> was rewritten from 
	A ::= ... 
	| B
	| B C
	...
to 
	A ::=  ...
	| B C?
	...

'''

Semi = compile(';')
Brackets = compile('[()]')
SqBrackets = compile('[\[\]]')
CBrackets = compile('[\{\}]')
If = compile('if')
Else = compile('else')
While = compile('while')
Ops = compile("(=)|\|\||&&|!=|==|<|>|<=|>=|\+|-|\*|/|\!")

true = compile('true')
false = compile('false')
Basics = compile('bool|int|char|double')

Reals = compile(r"-?(0|[1-9][0-9]*)(\.[0-9]+)?")
Nums = compile("-?(0|[1-9][0-9]*)")
Ids = compile("[a-zA-Z_][a-zA-Z_0-9]*")

# It is important that the patterns are in the same order as TokenTypes
Patterns = [Semi, Brackets, SqBrackets, CBrackets, If, Else, While, Ops, true, false, Basics, Ids, Nums, Reals]


class TokenType(Enum):
    # See the previous comment
    Semi, Bracket, SqBracket, CBracket, If, Else, While, Op, true, false, Basic, Id, Num, Real = range(14)


class Token:
    def __init__(self, content: str, t: TokenType, line: int):
        # At first, I expected to use ``line`` for error handling, but now it seems to be a little bit superfluous.
        # I needed to add some additional info though.

        self.content = content
        self.type = t
        self.line = line

    def match_exact(self, s: str) -> bool:
        return self.content == s

    def match_type(self, t: TokenType) -> bool:
        return self.type == t

    def __repr__(self) -> str:
        return f'Token({self.content}, {self.type}, line={self.line})'


def tokenize(code: str) -> list[Token]:
    lines = code.split('\n')
    raw_tokens = []
    for i, line in enumerate(lines):
        raw_tokens.extend([(s.strip(), i) for s in line.split(' ') if s.strip() != ''])

    ret = []
    for text, line in raw_tokens:
        found = False
        for i, pat in enumerate(Patterns):
            if fullmatch(pat, text):
                ret.append(Token(text, TokenType(i), line))
                found = True
                break

        if found:
            continue

        raise Exception(f'Lexer error: unsupported token "{text}", line {line}')

    return ret


class TokenStream:
    def __init__(self, tokens: list[Token], parent: 'Parser'):
        # Parent is passed for error handling

        self.tokens = tokens
        self.size: int = len(self.tokens)
        self.parent = parent
        self.ind: int = 0

    def error(self, expected):
        ctx = []
        underline = []
        for i in range(max(0, self.ind - 5),
                       min(self.size, self.ind + 5)):

            cnt = self.tokens[i].content
            ctx.append(cnt)
            if i != self.ind:
                underline.append(' ' * len(cnt))

            else:
                underline.append('^' * len(cnt))

        ctx = ' '.join(ctx)
        underline = ' '.join(underline)
        raise Exception(
            f'Parser error: \n\t line={self.tokens[self.ind].line}, code \n \t\t {ctx}\n\t\t {underline}\n\t in rule ' +
            f'{self.parent.rule}: expected {expected}, {self.tokens[self.ind]} found')

    def is_exact(self, s: str) -> bool:
        return self.tokens[self.ind].content == s

    def is_type(self, t: TokenType) -> bool:
        return self.tokens[self.ind].type == t

    def is_any(self, *pats: Union[str, TokenType]) -> bool:
        for pat in pats:
            if type(pat) == str:
                if self.is_exact(pat):
                    return True
            if type(pat) == TokenType:
                if self.is_type(pat):
                    return True

        return False

    def advance_if_is(self, what: Union[str, TokenType]) -> str:
        if type(what) == str:
            if not self.is_exact(what):
                self.error(what)
            self.ind += 1
            return self.tokens[self.ind - 1].content

        if type(what) == TokenType:
            if not self.is_type(what):
                self.error(what)
            self.ind += 1
            return self.tokens[self.ind - 1].content

    def advance_if_any(self, *pats: Union[str, TokenType]):
        if not self.is_any(*pats):
            self.error(pats)

        self.ind += 1
        return self.tokens[self.ind - 1].content

    def gettoken(self) -> Optional[str]:
        self.ind += 1
        if self.ind < self.size:
            return self.tokens[self.ind - 1].content

        return None

    def __repr__(self) -> str:
        return repr(self.tokens[self.ind])


class AstNode:
    def __init__(self, *chlds: Union['AstNode', str]):
        self.children = list(chlds)
        self.type = 'AstNode'

    def make_repr(self, level: int = 0):
        # ``level`` for level of nesting

        ret = '  ' * level + f'{type(self).__name__}('

        if not self.children:
            return ret + ')'

        if len(self.children) == 1:
            r = repr(self.children[0])
            if r.count('\n') == 0:
                return ret + r + ')'

        ret += '\n'

        chlds = []

        for chld in self.children:
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
        self.children.append(node)


# The following bunch of classes are defined only for readability
class Program(AstNode):
    pass


class Block(AstNode):
    pass


class Decls(AstNode):
    pass


class Decl(AstNode):
    pass


class Type(AstNode):
    pass


class Stmts(AstNode):
    pass


class Stmt(AstNode):
    pass


class Loc(AstNode):
    pass


class Bool(AstNode):
    pass


class Join(AstNode):
    pass


class BinOp(AstNode):
    pass


class Unary(AstNode):
    pass


class Factor(AstNode):
    pass


class Parser:
    def __init__(self, tokens: list[Token]):
        self.current_token = TokenStream(tokens, self)
        self.rule = ''  # For error handling
        self.ast = self.parseProgram()

    def getAst(self) -> Program:
        return self.ast

    def parseProgram(self) -> Program:
        self.rule = 'Program'
        return Program(self.parseBlock())

    def parseBlock(self) -> Block:
        self.rule = 'Block'

        self.current_token.advance_if_is('{')

        decls = self.parseDecls()
        stmts = self.parseStmts()

        self.current_token.advance_if_is('}')

        return Block(decls, stmts)

    def parseDecls(self) -> Decls:
        self.rule = 'Decls'

        ret = Decls()
        while self.current_token.is_type(TokenType.Basic):
            ret.add_child(self.parseDecl())

        return ret

    def parseDecl(self) -> Decl:
        self.rule = 'Decl'

        _type = self.parseType()
        _id = self.current_token.advance_if_is(TokenType.Id)

        self.current_token.advance_if_is(';')

        return Decl(_type, _id)

    def parseType(self) -> Type:
        self.rule = 'Type'

        _basic = self.current_token.advance_if_is(TokenType.Basic)

        dims = []
        while self.current_token.is_exact('['):
            self.current_token.advance_if_is('[')

            token = self.current_token.advance_if_is(TokenType.Num)

            dims.append(token)

            self.current_token.advance_if_is(']')

        if dims:
            return Type(_basic, *dims)

        return Type(_basic)

    def parseStmts(self) -> Stmts:
        self.rule = 'Stmts'
        should_continue = self.current_token.is_any(
            TokenType.Id,
            TokenType.If,
            TokenType.While,
            '{'
        )

        stmts = Stmts()
        while should_continue:
            stmts.add_child(self.parseStmt())
            should_continue = self.current_token.is_any(
                TokenType.Id,
                TokenType.If,
                TokenType.While,
                '{'
            )

        return stmts

    def parseStmt(self) -> Stmt:
        self.rule = 'Stmt'

        if self.current_token.is_type(TokenType.Id):
            lhs = self.parseLoc()

            self.current_token.advance_if_is('=')

            rhs = self.parseBool()

            self.current_token.advance_if_is(';')

            return Stmt('assign', lhs, rhs)

        if self.current_token.is_exact('if'):
            self.current_token.advance_if_is('if')
            self.current_token.advance_if_is('(')

            cond = self.parseBool()

            self.current_token.advance_if_is(')')

            then = self.parseStmt()

            if self.current_token.is_exact('else'):
                self.current_token.advance_if_is('else')
                elseb = self.parseStmt()

                return Stmt('if', cond, then, elseb)

            return Stmt('if', cond, then)

        if self.current_token.is_exact('while'):
            self.current_token.advance_if_is('while')
            self.current_token.advance_if_is('(')

            cond = self.parseBool()

            self.current_token.advance_if_is(')')

            body = self.parseStmt()

            return Stmt('while', cond, body)

        if self.current_token.is_exact('{'):
            return Stmt(self.parseBlock())

    def parseLoc(self) -> Loc:
        self.rule = 'Loc'

        name = self.current_token.advance_if_is(TokenType.Id)
        dims = []

        while self.current_token.is_exact('['):
            self.current_token.advance_if_is('[')
            dim = self.parseBool()
            dims.append(dim)
            self.current_token.advance_if_is(']')

        return Loc(name, *dims)

    def parseBool(self) -> Bool:
        self.rule = 'Bool'

        ret = Bool(self.parseJoin())

        while self.current_token.is_exact('||'):
            self.current_token.advance_if_is('||')
            ret.add_child(self.parseJoin())

        return ret

    def parseJoin(self) -> Join:
        self.rule = 'Join'

        ret = Join(self.parseEq())

        while self.current_token.is_exact('&&'):
            self.current_token.advance_if_is('&&')
            ret.add_child(self.parseEq())

        return ret

    def parseEq(self) -> Union[BinOp, Unary]:
        self.rule = 'Eq'

        ret = self.parseRel()

        should_continue = self.current_token.is_any('==', '!=')

        while should_continue:
            op = self.current_token.advance_if_is(TokenType.Op)
            rhs = self.parseRel()

            ret = BinOp(ret, op, rhs)

            should_continue = self.current_token.is_any('==', '!=')

        return ret

    def parseRel(self) -> Union[BinOp, Unary]:
        self.rule = 'Rel'

        ret = self.parseExpr()

        should_continue = self.current_token.is_any('>', '<', '>=', '<=')

        while should_continue:
            op = self.current_token.advance_if_is(TokenType.Op)
            rhs = self.parseExpr()

            ret = BinOp(ret, op, rhs)

            should_continue = self.current_token.is_any('>', '<', '>=', '<=')

        return ret

    def parseExpr(self) -> Union[BinOp, Unary]:
        self.rule = 'Expr'

        ret = self.parseTerm()
        should_continue = self.current_token.is_any('-', '+')

        while should_continue:
            op = self.current_token.advance_if_is(TokenType.Op)
            rhs = self.parseTerm()

            ret = BinOp(ret, op, rhs)
            should_continue = self.current_token.is_any('-', '+')

        return ret

    def parseTerm(self) -> Union[BinOp, Unary]:
        self.rule = 'Term'

        ret = self.parseUnary()
        should_continue = self.current_token.is_any('*', '/')

        while should_continue:
            op = self.current_token.advance_if_is(TokenType.Op)
            rhs = self.parseUnary()

            ret = BinOp(ret, op, rhs)
            should_continue = self.current_token.is_any('*', '/')

        return ret

    def parseUnary(self) -> Unary:
        self.rule = 'Unary'

        prefix = ''
        should_continue = self.current_token.is_any('!', '-')

        while should_continue:
            op = self.current_token.advance_if_is(TokenType.Op)
            prefix = op + prefix

            should_continue = self.current_token.is_any('!', '-')

        factor = self.parseFactor()

        return Unary(prefix, factor)

    def parseFactor(self) -> Factor:
        self.rule = 'Factor'

        if self.current_token.is_exact('('):
            self.current_token.advance_if_is('(')

            ret = Factor(self.parseBool())

            self.current_token.advance_if_is(')')

            return ret

        if self.current_token.is_type(TokenType.Id):
            return Factor(self.parseLoc())

        if self.current_token.is_type(TokenType.Num):
            token = self.current_token.advance_if_is(TokenType.Num)
            return Factor(token)

        if self.current_token.is_type(TokenType.Real):
            token = self.current_token.advance_if_is(TokenType.Real)
            return Factor(token)

        if self.current_token.is_any(TokenType.true, TokenType.false):
            token = self.current_token.advance_if_any(TokenType.true, TokenType.false)
            return Factor(token)

        # I do really hope that this one will never happen!
        # This line may seem strange, but it provides a readable error message.
        # It always reises an Exception, that is intended.

        self.current_token.advance_if_is('something like a factor')


if len(argv) != 2:
    raise RuntimeError(f'I expect exactly one argument, got {argv}')

file = open(argv[1], 'r')
code = file.read()

tokens = tokenize(code)
parser = Parser(tokens)
ast = parser.getAst()

print(ast)

file.close()
