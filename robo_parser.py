from __future__ import annotations
import robo_ast as ast
from re import compile, fullmatch
from enum import Enum
from typing import Union, Optional

'''
program ::= block
block ::=  "{" decls stmts "}"
decls ::= (decl)*
decl ::= type ID ";"
type ::= BASIC ("[" NUM "]")*
stmts ::= (stmt)*
stmt ::= loc "=" bool ";"
| "print" "(" bool ")"
| "rover" "." method "(" ")"
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
| "rover" "." property
method ::= "turn_right" 
| "turn_left"
| "move"
| "broadcast"
| "get_message"
| "get_dir" 
| "print_info"
| "drill"
property ::= "under"
| "in_front"
| "has_new_message"
'''

Semi = compile(';')  # noqa
Brackets = compile('[()]')  # noqa
SqBrackets = compile('[\[\]]')  # noqa
CBrackets = compile('[\{\}]')  # noqa
If = compile('if')  # noqa
Else = compile('else')  # noqa
While = compile('while')  # noqa
Ops = compile("(=)|\|\||&&|!=|==|<|>|<=|>=|\+|-|\*|/|\!|\.")  # noqa

true = compile('true')  # noqa
false = compile('false')  # noqa
Basics = compile('bool|int|char|double')  # noqa

Reals = compile(r"-?(0|[1-9][0-9]*)(\.[0-9]+)?")  # noqa
Nums = compile("-?(0|[1-9][0-9]*)")  # noqa
Ids = compile("[a-zA-Z_][a-zA-Z_0-9]*")  # noqa

Methods = compile("turn_right|turn_left|move|freeze|melt|build|explode|print_info|drill")
Properties = compile("x|y|dir|under|in_front")

Bifs = compile("print|print_map")
# It is important that the patterns are in the same order as TokenTypes
Patterns = [Semi, Brackets, SqBrackets, CBrackets, If, Else, While, Ops, true, false, Basics, Nums, Reals, Methods,
            Properties, Bifs, Ids]


class TokenType(Enum):
    # See the previous comment
    Semi, Bracket, SqBracket, CBracket, If, Else, While, Op, true, \
    false, Basic, Num, Real, Method, Property, Bif, Id = range(17)


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


class Parser:
    def __init__(self, tokens: list[Token]):
        self.current_token = TokenStream(tokens, self)
        self.rule = ''  # For error handling
        self.ast = self.parseProgram()

    def getAst(self) -> ast.Program:
        return self.ast

    def parseProgram(self) -> ast.Program:
        self.rule = 'Program'
        return ast.Program(self.parseBlock())

    def parseBlock(self) -> ast.Block:
        self.rule = 'Block'

        self.current_token.advance_if_is('{')

        decls = self.parseDecls()
        stmts = self.parseStmts()

        self.current_token.advance_if_is('}')

        return ast.Block(decls, stmts)

    def parseDecls(self) -> ast.Decls:
        self.rule = 'Decls'

        ret = ast.Decls()
        while self.current_token.is_type(TokenType.Basic):
            ret.add_child(self.parseDecl())

        return ret

    def parseDecl(self) -> ast.Decl:
        self.rule = 'Decl'

        _type = self.parseType()
        _id = self.current_token.advance_if_is(TokenType.Id)

        self.current_token.advance_if_is(';')

        return ast.Decl(_type, _id)

    def parseType(self) -> ast.Type:
        self.rule = 'Type'

        _basic = self.current_token.advance_if_is(TokenType.Basic)

        dims = []
        while self.current_token.is_exact('['):
            self.current_token.advance_if_is('[')

            token = self.current_token.advance_if_is(TokenType.Num)

            dims.append(token)

            self.current_token.advance_if_is(']')

        if dims:
            return ast.Type(_basic, *dims)

        return ast.Type(_basic)

    def parseStmts(self) -> ast.Stmts:
        self.rule = 'Stmts'
        should_continue = self.current_token.is_any(
            TokenType.Bif,
            TokenType.Id,
            TokenType.If,
            TokenType.While,
            '{'
        )

        stmts = ast.Stmts()
        while should_continue:
            stmts.add_child(self.parseStmt())
            should_continue = self.current_token.is_any(
                TokenType.Bif,
                TokenType.Id,
                TokenType.If,
                TokenType.While,
                '{'
            )

        return stmts

    def parseStmt(self) -> Union[ast.Stmt, ast.Method, ast.Bif]:
        self.rule = 'Stmt'

        if self.current_token.is_type(TokenType.Bif):
            bif = self.current_token.advance_if_is(TokenType.Bif)
            self.current_token.advance_if_is("(")

            argument = None
            if not self.current_token.is_exact(")"):
                argument = self.parseBool()

            self.current_token.advance_if_is(")")
            self.current_token.advance_if_is(";")

            return ast.Bif(bif, argument)

        if self.current_token.is_exact("rover"):
            self.current_token.advance_if_is("rover")
            self.current_token.advance_if_is(".")

            method = self.current_token.advance_if_is(TokenType.Method)

            self.current_token.advance_if_is("(")
            self.current_token.advance_if_is(")")
            self.current_token.advance_if_is(";")

            return ast.Method(method)

        if self.current_token.is_type(TokenType.Id):
            lhs = self.parseLoc()

            self.current_token.advance_if_is('=')

            rhs = self.parseBool()

            self.current_token.advance_if_is(';')

            return ast.Stmt('assign', lhs, rhs)

        if self.current_token.is_exact('if'):
            self.current_token.advance_if_is('if')
            self.current_token.advance_if_is('(')

            cond = self.parseBool()

            self.current_token.advance_if_is(')')

            then = self.parseStmt()

            if self.current_token.is_exact('else'):
                self.current_token.advance_if_is('else')
                elseb = self.parseStmt()

                return ast.Stmt('if', cond, then, elseb)

            return ast.Stmt('if', cond, then)

        if self.current_token.is_exact('while'):
            self.current_token.advance_if_is('while')
            self.current_token.advance_if_is('(')

            cond = self.parseBool()

            self.current_token.advance_if_is(')')

            body = self.parseStmt()

            return ast.Stmt('while', cond, body)

        if self.current_token.is_exact('{'):
            return ast.Stmt('block', self.parseBlock())

    def parseLoc(self) -> ast.Loc:
        self.rule = 'Loc'

        name = self.current_token.advance_if_is(TokenType.Id)
        dims: list[ast.AstNode] = []

        while self.current_token.is_exact('['):
            self.current_token.advance_if_is('[')
            dim = self.parseBool()
            dims.append(dim)
            self.current_token.advance_if_is(']')

        return ast.Loc(name, *dims)

    def parseBool(self) -> ast.Bool:
        self.rule = 'Bool'

        ret = ast.Bool(self.parseJoin())

        while self.current_token.is_exact('||'):
            self.current_token.advance_if_is('||')
            ret.add_child(self.parseJoin())

        return ret

    def parseJoin(self) -> ast.Join:
        self.rule = 'Join'

        ret = ast.Join(self.parseEq())

        while self.current_token.is_exact('&&'):
            self.current_token.advance_if_is('&&')
            ret.add_child(self.parseEq())

        return ret

    def parseEq(self) -> Union[ast.BinOp, ast.Unary]:
        self.rule = 'Eq'

        ret = self.parseRel()

        should_continue = self.current_token.is_any('==', '!=')

        while should_continue:
            op = self.current_token.advance_if_is(TokenType.Op)
            rhs = self.parseRel()

            ret = ast.BinOp(ret, op, rhs)

            should_continue = self.current_token.is_any('==', '!=')

        return ret

    def parseRel(self) -> Union[ast.BinOp, ast.Unary]:
        self.rule = 'Rel'

        ret = self.parseExpr()

        should_continue = self.current_token.is_any('>', '<', '>=', '<=')

        while should_continue:
            op = self.current_token.advance_if_is(TokenType.Op)
            rhs = self.parseExpr()

            ret = ast.BinOp(ret, op, rhs)

            should_continue = self.current_token.is_any('>', '<', '>=', '<=')

        return ret

    def parseExpr(self) -> Union[ast.BinOp, ast.Unary]:
        self.rule = 'Expr'

        ret = self.parseTerm()
        should_continue = self.current_token.is_any('-', '+')

        while should_continue:
            op = self.current_token.advance_if_is(TokenType.Op)
            rhs = self.parseTerm()

            ret = ast.BinOp(ret, op, rhs)
            should_continue = self.current_token.is_any('-', '+')

        return ret

    def parseTerm(self) -> Union[ast.BinOp, ast.Unary]:
        self.rule = 'Term'

        ret = self.parseUnary()
        should_continue = self.current_token.is_any('*', '/')

        while should_continue:
            op = self.current_token.advance_if_is(TokenType.Op)
            rhs = self.parseUnary()

            ret = ast.BinOp(ret, op, rhs)
            should_continue = self.current_token.is_any('*', '/')

        return ret

    def parseUnary(self) -> ast.Unary:
        self.rule = 'Unary'

        prefix = ''
        should_continue = self.current_token.is_any('!', '-')

        while should_continue:
            op = self.current_token.advance_if_is(TokenType.Op)
            prefix = op + prefix

            should_continue = self.current_token.is_any('!', '-')

        factor = self.parseFactor()

        return ast.Unary(prefix, factor)

    def parseFactor(self) -> ast.Factor:
        self.rule = 'Factor'

        if self.current_token.is_exact("rover"):
            self.current_token.advance_if_is("rover")
            self.current_token.advance_if_is(".")

            prop = self.current_token.advance_if_is(TokenType.Property)

            return ast.Property(prop)

        if self.current_token.is_exact('('):
            self.current_token.advance_if_is('(')

            ret = ast.Factor(self.parseBool())

            self.current_token.advance_if_is(')')

            return ret

        if self.current_token.is_type(TokenType.Id):
            return ast.Factor(self.parseLoc())

        if self.current_token.is_type(TokenType.Num):
            token = self.current_token.advance_if_is(TokenType.Num)
            return ast.Factor(token)

        if self.current_token.is_type(TokenType.Real):
            token = self.current_token.advance_if_is(TokenType.Real)
            return ast.Factor(token)

        if self.current_token.is_any(TokenType.true, TokenType.false):
            token = self.current_token.advance_if_any(TokenType.true, TokenType.false)
            return ast.Factor(token)

        # I do really hope that this one will never happen!
        # This line may seem strange, but it provides a readable error message.
        # It always reises an Exception, that is intended.

        self.current_token.advance_if_is('something like a factor')
