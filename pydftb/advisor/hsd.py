"""A read-only parser for DFTB+'s HSD input format, keeping line numbers.

Covers what DFTB+ inputs, ASE-written inputs and DFTB+'s ``dftb_pin.hsd`` contain::

    Tag = value                      # a value: the rest of the line (several tokens)
    Tag [unit] = value               # with a modifier
    Tag = Method { ... }             # a method block: Tag has one child, Method
    Tag = { ... }  /  Tag { ... }    # a plain block
    1 0 0                            # data lines inside a block (no '=' / '{')
    <<< "file"                       # data lines from a file (e.g. a .gen geometry)
    <<+ "file.hsd"                   # HSD read from a file, as if written here

Comments start with ``#``; ``;`` ends a statement like a newline.  Names are matched
case-insensitively, as DFTB+ does.  Unknown tags are fine: the parser builds the tree
the text describes and leaves the meaning to the rules.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


class HSDError(ValueError):
    """The text is not HSD (unbalanced braces, a value followed by '{', ...)."""


@dataclass(eq=False)
class Node:
    """One HSD node.

    ``value``: the text of ``Tag = value`` (quotes removed); ``children``: the nodes of a
    block (the method of ``Tag = Method {...}`` is the single child, ``is_method``);
    ``data``: data lines as (line, tokens); ``line``/``end_line``: where it is written.
    """
    name: str
    modifier: str | None = None
    value: str | None = None
    children: list = field(default_factory=list)
    data: list = field(default_factory=list)
    file: str | None = None
    line: int = 0
    end_line: int = 0
    assigned: bool = False            # written as Tag = ...
    is_method: bool = False
    raw_file: str | None = None       # the file of a <<< include
    missing_includes: list = field(default_factory=list)
    parent: "Node | None" = field(default=None, repr=False)

    # -- navigation ------------------------------------------------------------------
    def child(self, name: str) -> "Node | None":
        low = name.lower()
        return next((c for c in self.children if c.name.lower() == low), None)

    def find(self, path: str) -> "Node | None":
        """Node at ``path`` ("Hamiltonian/DFTB/Filling"), case-insensitive."""
        node = self
        for part in path.strip("/").split("/"):
            if part:
                node = node.child(part)
                if node is None:
                    return None
        return node

    @property
    def method(self) -> "Node | None":
        """The method of ``Tag = Method { ... }``; None for values and plain blocks."""
        if len(self.children) == 1 and self.children[0].is_method:
            return self.children[0]
        return None

    @property
    def path(self) -> str:
        parts, node = [], self
        while node is not None and node.parent is not None:
            parts.append(node.name)
            node = node.parent
        return "/".join(reversed(parts))

    @property
    def where(self) -> str:
        """``file:line`` or ``file:first-last`` for messages."""
        name = Path(self.file).name if self.file else "?"
        if self.end_line and self.end_line != self.line:
            return f"{name}:{self.line}-{self.end_line}"
        return f"{name}:{self.line}"

    def walk(self):
        """This node and every node below it (depth first)."""
        yield self
        for c in self.children:
            yield from c.walk()

    def text(self, path: str = "", default: str | None = None) -> str | None:
        """The value at ``path``; a block's data (``SetForAll { Yes }``) counts as text."""
        node = self.find(path) if path else self
        if node is None:
            return default
        if node.value is not None:
            return node.value
        if node.method is not None:
            return node.method.text(default=default)
        if node.data:
            return " ".join(" ".join(tok) for _, tok in node.data)
        return default

    def flag(self, path: str = "", default: bool | None = None) -> bool | None:
        """``Yes``/``No`` at ``path`` as a bool."""
        txt = self.text(path)
        if txt is None:
            return default
        t = txt.strip().lower()
        if t in ("yes", "true", ".true."):
            return True
        if t in ("no", "false", ".false."):
            return False
        return default

    def number(self, path: str = "", default: float | None = None) -> float | None:
        txt = self.text(path)
        try:
            return float(txt.split()[0].lower().replace("d", "e"))
        except (AttributeError, ValueError, IndexError):
            return default


# -- tokenizer ----------------------------------------------------------------------------
_TOKEN = re.compile(r'''
    (?P<ws>[ \t\r\f\v]+)
  | (?P<nl>\n)
  | (?P<comment>\#[^\n]*)
  | (?P<inc><<[<+])
  | (?P<str>"[^"\n]*"|'[^'\n]*')
  | (?P<mod>\[[^\]\n]*\])
  | (?P<punct>[={};])
  | (?P<word>[^\s={}\[\];#"']+)
''', re.VERBOSE)


def _tokens(text: str) -> list:
    """(kind, text, line) with kind one of nl, inc, str, mod, punct, word."""
    out, line, pos = [], 1, 0
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if m is None:
            raise HSDError(f"line {line}: cannot read {text[pos:pos + 20]!r}")
        kind, tok = m.lastgroup, m.group()
        if kind == "nl":
            out.append(("nl", "\n", line))
            line += 1
        elif kind not in ("ws", "comment"):
            out.append((kind, tok, line))
        pos = m.end()
    out.append(("nl", "\n", line))
    return out


def _unquote(tok: str) -> str:
    if len(tok) >= 2 and tok[0] == tok[-1] and tok[0] in "\"'":
        return tok[1:-1]
    return tok


class _Parser:
    def __init__(self, text: str, path: str | None, depth: int = 0):
        self.toks = _tokens(text)
        self.i = 0
        self.path = path
        self.depth = depth

    def peek(self, k: int = 0):
        j = self.i + k
        return self.toks[j] if j < len(self.toks) else ("eof", "", self.toks[-1][2])

    def take(self):
        tok = self.peek()
        self.i += 1
        return tok

    def _end_of_statement(self, kind: str, tok: str) -> bool:
        return kind in ("nl", "eof") or (kind == "punct" and tok in (";", "}"))

    def block(self, parent: Node, closing: bool) -> None:
        """Statements and data lines until '}' (``closing``) or the end of the text."""
        while True:
            while self.peek()[0] == "nl" or self.peek()[1] == ";" and self.peek()[0] == "punct":
                self.i += 1
            kind, tok, line = self.peek()
            if kind == "eof":
                if closing:
                    raise HSDError(f"{self.path}: '{{' of {parent.name or 'the input'} "
                                   f"(line {parent.line}) is never closed")
                return
            if kind == "punct" and tok == "}":
                if not closing:
                    raise HSDError(f"{self.path}:{line}: '}}' without a matching '{{'")
                self.i += 1
                parent.end_line = line
                return
            if kind == "inc":
                self.include(parent)
            elif self.is_statement():
                self.statement(parent)
            else:
                self.data_line(parent)

    def is_statement(self) -> bool:
        """``word [mod] =`` or ``word [mod] {`` starts a tag; anything else is data."""
        if self.peek()[0] not in ("word", "str"):
            return False
        k = 2 if self.peek(1)[0] == "mod" else 1
        nxt = self.peek(k)
        return nxt[0] == "punct" and nxt[1] in ("=", "{")

    def statement(self, parent: Node) -> None:
        _, name, line = self.take()
        node = Node(_unquote(name), file=self.path, line=line, end_line=line, parent=parent)
        parent.children.append(node)
        if self.peek()[0] == "mod":
            node.modifier = self.take()[1][1:-1].strip()
        _, op, _ = self.take()
        if op == "{":                                             # Tag { ... }
            self.block(node, closing=True)
            return
        node.assigned = True
        kind, tok, _ = self.peek()
        if kind == "punct" and tok == "{":                       # Tag = { ... }
            self.i += 1
            self.block(node, closing=True)
            return
        if kind in ("word", "str") and self.peek(1)[:2] == ("punct", "{"):
            _, mname, mline = self.take()                         # Tag = Method { ... }
            self.i += 1
            method = Node(_unquote(mname), file=self.path, line=mline, end_line=mline,
                          is_method=True, parent=node)
            node.children.append(method)
            self.block(method, closing=True)
            node.end_line = method.end_line
            return
        words = []                                                # Tag = value
        while True:
            kind, tok, ln = self.peek()
            if self._end_of_statement(kind, tok):
                break
            if kind == "punct" and tok == "{":
                raise HSDError(f"{self.path}:{ln}: unexpected '{{' after the value of {node.name}")
            # 'A = 1  B = 2' on one line: DFTB+ takes the rest of the line as the value of A
            # (and stops with "Superfluous data found"); so does this parser
            words.append(_unquote(tok) if kind == "str" else tok)
            node.end_line = ln
            self.i += 1
        node.value = " ".join(words)

    def data_line(self, parent: Node) -> None:
        tokens, line = [], self.peek()[2]
        while True:
            kind, tok, ln = self.peek()
            if self._end_of_statement(kind, tok):
                break
            if kind == "punct":
                raise HSDError(f"{self.path}:{ln}: unexpected {tok!r} in a data line")
            tokens.append(_unquote(tok) if kind == "str" else tok)
            self.i += 1
        if tokens:
            parent.data.append((line, tokens))

    def include(self, parent: Node) -> None:
        _, kind, line = self.take()
        k2, name, _ = self.take()
        if k2 not in ("str", "word"):
            raise HSDError(f"{self.path}:{line}: {kind} needs a file name")
        fname = _unquote(name)
        target = (Path(self.path).parent if self.path else Path(".")) / fname
        try:
            text = target.read_text(errors="replace")
        except OSError:
            parent.missing_includes.append(fname)
            return
        if kind == "<<<":                                         # raw data lines
            parent.raw_file = str(target)
            for n, raw in enumerate(text.splitlines(), 1):
                toks = raw.split("#", 1)[0].split()
                if toks:
                    parent.data.append((n, toks))
            return
        if self.depth > 20:
            raise HSDError(f"{self.path}:{line}: <<+ nested too deeply")
        _Parser(text, str(target), self.depth + 1).block(parent, closing=False)


def parse(text: str, path: str | None = None) -> Node:
    """The tree of HSD ``text`` (``path``: for messages and relative includes)."""
    root = Node("", file=path, line=1)
    _Parser(text, path).block(root, closing=False)
    root.end_line = text.count("\n") + 1
    return root


def read(path) -> Node:
    path = Path(path)
    return parse(path.read_text(errors="replace"), str(path))
