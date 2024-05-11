"""Reentrancy container classes.

"""
from __future__ import annotations
__author__ = 'Paul Landes'
from typing import Tuple, Set, Dict, Any, Iterable, Type, ClassVar
from dataclasses import dataclass, field
import sys
from io import TextIOBase
from itertools import chain
from frozendict import frozendict
from zensols.config import Dictable
from zensols.persist import PersistableContainer, persisted
from . import ConceptGraphNode, GraphEdge


@dataclass
class EdgeFlow(PersistableContainer, Dictable):
    """The flow over a graph edge.  This keeps the flow of the edge as a
    "snapshot" of the value at a particular point in the algorithm, before it is
    modified to fix the issue.

    """
    edge: GraphEdge = field()
    """The outgoing (in the reverse graph) edge of the reentrancy."""

    flow: float = field(default=None)
    """The flow value at the time of the algorithm."""

    def __post_init__(self):
        super().__init__()
        self.flow = self.edge.flow

    def __str__(self) -> str:
        return f'{self.edge}: {self.flow}'


@dataclass
class Reentrancy(PersistableContainer, Dictable):
    """Reentrancies are concept nodes with multiple parents (in the forward
    graph) and have side effects when running the algorithm.

    Note: an AMR (always acyclic) graph with no reentrancies are trees.

    """
    _DICTABLE_ATTRIBUTES: ClassVar[Set[str]] = frozenset({'has_zero_flow'})

    concept_node: ConceptGraphNode = field()
    """The concept node of the reentrancy"""

    concept_node_vertex: int = field()
    """The :obj:`igraph.Vertex.index` associated with the node."""

    edge_flows: Tuple[EdgeFlow] = field()
    """The outgoing edges connected to the reentrant :obj:`concept_node`."""

    def __post_init__(self):
        super().__init__()

    @property
    @persisted('_zero_flows', transient=True)
    def zero_flows(self) -> Tuple[GraphEdge]:
        """The edges that have no flow."""
        return tuple(filter(lambda ef: ef.flow == 0, self.edge_flows))

    @property
    def has_zero_flow(self) -> bool:
        """Whether the reentracy has any edges with no flow."""
        return len(self.zero_flows) > 0

    @property
    def total_flow(self) -> float:
        """The total flow of all outgoing (in the reverse graph) edges."""
        return sum(map(lambda ef: ef.flow, self.edge_flows))

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        self._write_line(f'node: {self.concept_node}', depth, writer)
        ef: EdgeFlow
        for ef in self.edge_flows:
            self._write_line(str(ef), depth + 1, writer)


@dataclass
class ReentrancySet(PersistableContainer, Dictable):
    """A set of reentrancies, one for each iteration of the algorithm.

    """
    _DICTABLE_WRITABLE_DESCENDANTS: ClassVar[bool] = True

    reentrancies: Tuple[Reentrancy] = field(default=())
    """Concept nodes with multiple parents."""

    @property
    @persisted('_by_vertex')
    def by_vertex(self) -> Dict[int, Reentrancy]:
        return frozendict({r.concept_node_vertex: r for r in self.reentrancies})

    @property
    def stats(self) -> Dict[str, Any]:
        """Get the stats for this set of reentrancies."""
        zflows = filter(lambda r: r.has_zero_flow, self.reentrancies)
        return {'total': len(self.reentrancies),
                'zero_flow': sum(1 for _ in zflows)}

    @classmethod
    def combine(cls: Type[ReentrancySet], sets: Iterable[ReentrancySet]) -> \
            ReentrancySet:
        """Combine by adding reentrancy links for all in ``sets``."""
        rs: Tuple[Reentrancy] = tuple(chain.from_iterable(
            map(lambda rs: rs.reentrancies, sets)))
        return cls(rs)
