"""Defines a class that aligns components of a (bipartite) graph.

"""
__author__ = 'Paul Landes'

from typing import Tuple, Iterable, Dict, List, Type
from dataclasses import dataclass, field
import logging
from itertools import chain
from igraph import Graph, Vertex, Edge
from zensols.persist import persisted
from . import (
    GraphEdge, ComponentAlignmentGraphEdge, GraphComponent, DocumentGraph,
    TerminalGraphEdge, TerminalGraphNode,
)

logger = logging.getLogger(__name__)


@dataclass
class GraphAlignmentConstructor(object):
    """Adds additional nodes and edges that enable the maxflow algorithm to be
    used on the graph.  This include component alignment edges, source node and
    sink node.  Capacities for the component alignment edges are also set.

    """
    doc_graph: DocumentGraph = field(default=None)
    """A document graph that contains the graph to be aligned.

    """
    source_flow: float = field(default=GraphEdge.MAX_CAPACITY)
    """The capacity to use for the source node of the transporation graph."""

    def __post_init__(self):
        self._persisted_works: List[str] = ['_source_sink_flow_nodes']

    @property
    def _doc_graph(self) -> DocumentGraph:
        return self.__doc_graph

    @_doc_graph.setter
    def _doc_graph(self, doc_graph: DocumentGraph):
        self.__doc_graph = doc_graph
        if doc_graph is not None:
            for attr in self._persisted_works:
                if hasattr(self, attr):
                    getattr(self, attr).clear()

    def requires_reversed_edges(self) -> bool:
        return False

    def add_edges(self, capacities: Iterable[Tuple[int, int, float]],
                  cls: Type[GraphEdge] = ComponentAlignmentGraphEdge) -> \
            List[GraphEdge]:
        """Add ``capacities`` as graph capacities to the graph.

        :param capacities: the vertexes and capacities in the form: ``(<source
                           vertex index>, <summary index>, <capacity>)``

        :param cls: the type of object to instantiate for the
                    :class:`.GraphEdge` alignment

        """
        edges: List[GraphEdge] = []
        g: Graph = self.doc_graph.graph
        for source, target, capacity in capacities:
            e: Edge = g.add_edge(source, target)
            ge: GraphEdge = GraphComponent.set_edge(
                e, cls(
                    context=self.doc_graph.graph_attrib_context,
                    capacity=capacity))
            edges.append(ge)
        self.doc_graph.invalidate()
        return edges

    def set_capacities(self, edges: Iterable[GraphEdge],
                       capacity: float = GraphEdge.MAX_CAPACITY):
        """Set ``capacity`` on all ``edges``."""
        e: GraphEdge
        for e in edges:
            e.capacity = capacity

    def update_capacities(self, caps: Dict[int, int]) -> Dict[int, int]:
        """Update the capacities of the graph component.

        :param caps: the capacities with key/value pairs as ``<edge
                     ID>/<capacity>``

        :return: the ``caps`` parameter

        """
        container: DocumentGraph = self.doc_graph
        for vix, cap in caps.items():
            ge: GraphEdge = container.edge_by_id(vix)
            ge.capacity = cap
        return caps

    def _get_sinks(self, s: Vertex, t: Vertex) -> \
            Iterable[Tuple[int, int, float]]:
        """Get terminal node(s), which are those used to connect to the sink.

        :return: tuples of ``(<source vertex>, <target vertex>, <capacity>)``

        """
        ti: int = t.index
        cap: int = self.doc_graph.graph_attrib_context.sink_capacity
        v: Vertex
        for v in self.doc_graph.select_vertices(_outdegree=0):
            yield (v.index, ti, cap)

    def _get_sources(self, s: Vertex, t: Vertex) -> \
            Iterable[Tuple[int, int, float]]:
        """Like :meth:`_get_sinks` but creates the source(s)."""
        root: Vertex = self.doc_graph.root
        return ((s.index, root.index, self.source_flow),)

    def _connect_terminals(self, s: Vertex, t: Vertex):
        capacities: Iterable[Tuple[int, int, float]] = \
            chain.from_iterable(
                (self._get_sinks(s, t), self._get_sources(s, t)))
        self.add_edges(capacities, cls=TerminalGraphEdge)

    @persisted('_source_sink_flow_nodes')
    def _get_source_sink_flow_nodes(self) -> Tuple[Vertex, Vertex]:
        """The source and sink terminal nodes in that order.  This connects them
        as a side effect.

        """
        g: Graph = self.doc_graph.graph
        s: Vertex = GraphComponent.set_node(
            g.add_vertex(), TerminalGraphNode(
                context=self.doc_graph.graph_attrib_context,
                is_source=True))[0]
        t: Vertex = GraphComponent.set_node(
            g.add_vertex(), TerminalGraphNode(
                context=self.doc_graph.graph_attrib_context,
                is_source=False))[0]
        self._connect_terminals(s, t)
        nodes: Tuple[Vertex, Vertex] = (s, t)
        v: Vertex
        for v in nodes:
            v[GraphComponent.ROOT_ATTRIB_NAME] = False
        return nodes

    @property
    def source_flow_node(self) -> Vertex:
        """The source flow node."""
        return self._get_source_sink_flow_nodes()[0]

    @property
    def sink_flow_node(self) -> Vertex:
        """The sink flow node."""
        return self._get_source_sink_flow_nodes()[1]

    def build(self):
        """Build the graph by adding component alignment capacities."""
        pass


GraphAlignmentConstructor.doc_graph = GraphAlignmentConstructor._doc_graph
