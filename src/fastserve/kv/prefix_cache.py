"""A reference prefix cache: a radix tree over token sequences with LRU eviction (SGLang's RadixAttention).

Requests that start with the same tokens (a system prompt, earlier turns of a conversation) produce the same
keys and values for those tokens, because attention is causal: a token's K and V depend only on the tokens
before it. A prefix cache keeps them, so a new request only prefills the part nobody has seen.

The tree:
- Each edge holds a run of tokens and a *payload* for exactly those tokens (in nanoserve: their K and V).
  A payload must support slicing along tokens (`payload[a:b]`), so an edge can be split where two sequences
  diverge. Plain lists work, which keeps the tree testable without a GPU.
- `match(tokens)` walks down as far as the tokens agree and returns the matched length and the payloads along
  the way. Every node it touches becomes most recently used.
- `insert(tokens, payload)` adds whatever part of the sequence isn't cached yet.
- With a token budget, inserting evicts least-recently-used *leaves* first. An inner node is shared by its
  children, so it only becomes evictable once they are gone.

vLLM's automatic prefix caching gets the same effect differently: it hashes each full 16-token block together
with the hash of the block before it, so equal prefixes produce equal block hashes. A radix tree matches at
token granularity; block hashing only at block boundaries.

Stdlib only.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any


@dataclass(eq=False)
class Node:
    tokens: tuple[int, ...] = ()
    payload: Any = None
    parent: Node | None = None
    children: dict[int, Node] = field(default_factory=dict)  # keyed by the child edge's first token
    last_used: int = 0
    hits: int = 0  # how many matches passed through (or ended inside) this edge
    id: int = 0


class RadixCache:
    def __init__(self, capacity_tokens: int | None = None):
        self._ids = itertools.count()
        self.root = Node(id=next(self._ids))
        self.capacity = capacity_tokens
        self.size = 0  # tokens held
        self._clock = itertools.count(1)
        self.queried_tokens = 0
        self.hit_tokens = 0
        self.evicted_tokens = 0

    # -- lookup

    def match(self, tokens: list[int] | tuple[int, ...]) -> tuple[int, list[Any]]:
        """Length of the longest cached prefix of `tokens`, and the payload pieces covering it, in order."""
        now, node, matched, pieces = next(self._clock), self.root, 0, []
        while matched < len(tokens):
            child = node.children.get(tokens[matched])
            if child is None:
                break
            common = _common_length(child.tokens, tokens[matched:])
            child.last_used, child.hits = now, child.hits + 1
            pieces.append(child.payload[:common] if common < len(child.tokens) else child.payload)
            matched += common
            if common < len(child.tokens):
                break
            node = child
        self.queried_tokens += len(tokens)
        self.hit_tokens += matched
        return matched, pieces

    # -- insertion

    def insert(self, tokens: list[int] | tuple[int, ...], payload: Any) -> int:
        """Cache `tokens` (payload covers all of them); returns how many tokens were new."""
        tokens, now, node, i = tuple(tokens), next(self._clock), self.root, 0
        while i < len(tokens):
            child = node.children.get(tokens[i])
            if child is None:
                break
            common = _common_length(child.tokens, tokens[i:])
            if common < len(child.tokens):
                child = self._split(child, common)
            child.last_used = now
            node, i = child, i + common
        if i == len(tokens):
            return 0
        new = Node(tokens=tokens[i:], payload=payload[i:], parent=node, last_used=now, id=next(self._ids))
        node.children[tokens[i]] = new
        self.size += len(new.tokens)
        if self.capacity is not None and self.size > self.capacity:
            self.evict(self.size - self.capacity, protect=new)
        return len(new.tokens)

    def _split(self, child: Node, at: int) -> Node:
        """Cut child's edge after `at` tokens: a new inner node takes the first part."""
        parent = child.parent
        head = Node(
            tokens=child.tokens[:at],
            payload=child.payload[:at],
            parent=parent,
            last_used=child.last_used,
            hits=child.hits,
            id=next(self._ids),
        )
        child.tokens, child.payload, child.parent = child.tokens[at:], child.payload[at:], head
        head.children[child.tokens[0]] = child
        parent.children[head.tokens[0]] = head
        return head

    # -- eviction

    def evict(self, n_tokens: int, protect: Node | None = None) -> int:
        """Drop least-recently-used leaves until `n_tokens` are freed (or nothing evictable is left)."""
        freed = 0
        while freed < n_tokens:
            leaves = [n for n in self.nodes() if not n.children and n is not protect]
            if not leaves:
                break
            leaf = min(leaves, key=lambda n: n.last_used)
            del leaf.parent.children[leaf.tokens[0]]
            freed += len(leaf.tokens)
        self.size -= freed
        self.evicted_tokens += freed
        return freed

    # -- inspection

    def nodes(self) -> list[Node]:
        """Every node except the root, parents before children."""
        out, stack = [], list(self.root.children.values())
        while stack:
            node = stack.pop()
            out.append(node)
            stack.extend(node.children.values())
        return out

    def hit_rate(self) -> float:
        return self.hit_tokens / self.queried_tokens if self.queried_tokens else 0.0


def _common_length(a: tuple[int, ...], b: list[int] | tuple[int, ...]) -> int:
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n


def replay(prompts: list[list[int]], capacity_tokens: int | None = None) -> tuple[RadixCache, list[int]]:
    """Serve prompts in order through a radix cache (payload = the tokens themselves).

    Returns the cache and, per prompt, how many of its tokens were already cached: the prefill a perfect
    prefix cache skips. Generated tokens aren't inserted; this models what the *prompts* share.
    """
    cache, cached = RadixCache(capacity_tokens), []
    for prompt in prompts:
        matched, _ = cache.match(prompt)
        cached.append(matched)
        cache.insert(prompt, list(prompt))
    return cache, cached
