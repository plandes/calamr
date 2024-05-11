"""Graph alignment constructors for summary graphs.

"""
__author__ = 'Paul Landes'

from typing import Tuple, Iterable, Callable
from dataclasses import dataclass, field
import logging
from itertools import chain
import pandas as pd
from igraph import Vertex, Edge
from zensols.util import time
from zensols.persist import persisted
from .. import (
    ComponentAlignmentError,
    GraphAttributeContext, GraphEdge, GraphNode, GraphComponent,
    DocumentGraphComponent, DocumentGraph, GraphAlignmentConstructor,
    ComponentAlignmentGraphEdge,
)
from .factory import SummaryConstants
from .capacity import CapacityCalculator

logger = logging.getLogger(__name__)


@dataclass
class SummaryGraphAlignmentConstructor(GraphAlignmentConstructor):
    """A graph alignment constructor for summary graphs.

    """
    capacity_calculator: CapacityCalculator = field(default=None)
    """Calculates and the component alignment capacities."""

    def __post_init__(self):
        super().__post_init__()
        self._persisted_works.append('_connections')
        self._persisted_works.append('_component_alignment_capacities')

    def _get_component(self, name: str) -> DocumentGraphComponent:
        """The document and AMR nodes that make up the source content of the
        document.

        """
        dc: DocumentGraphComponent = self.doc_graph.components_by_name.get(name)
        if dc is None:
            names: str = ', '.join(self.doc_graph.components_by_name.keys())
            raise ComponentAlignmentError(
                f"Missing component: '{name}', keys: {names}")
        return dc

    @property
    def source(self) -> DocumentGraphComponent:
        """The document and AMR nodes that make up the source content of the
        document.

        """
        return self._get_component(SummaryConstants.SOURCE_COMP)

    @property
    def summary(self) -> DocumentGraphComponent:
        """The document and AMR nodes that make up the summary content of the
        document.

        """
        return self._get_component(SummaryConstants.SUMMARY_COMP)

    def _set_doc_graph(self, doc_graph: DocumentGraph):
        super()._set_doc_graph(doc_graph)
        self._connections.clear()
        self._component_alignment_capacities.clear()

    @property
    @persisted('_connections')
    def connections(self) -> pd.DataFrame:
        """The connections, capacities and similaries of the graph."""
        ctx: GraphAttributeContext = self.doc_graph.graph_attrib_context
        with time('calculated capacities'):
            cons: pd.DataFrame = self.capacity_calculator(self.doc_graph)
        cons = cons.copy()
        cons = cons[cons['sim'] > ctx.similarity_threshold]
        cons['capacity'] = cons['sim'] * ctx.component_alignment_capacity
        return cons

    def _get_doc_node_capacities(self, capacity: float) -> \
            Iterable[Tuple[int, int, float]]:
        srcs: Iterable[Vertex] = self.doc_graph.source.doc_vertices
        smys: Iterable[Vertex] = self.doc_graph.summary.doc_vertices
        for srcv, smyv in zip(srcs, smys):
            yield (srcv.index, smyv.index, capacity)

    @property
    @persisted('_component_alignment_capacities')
    def component_alignment_capacities(self) -> Tuple[Tuple[int, int, float]]:
        """The component alignment edges (alignment) and capacities edges as
        tuples of::

            ``(<source vertex index>, <summary index>, <capacity>)``

        """
        return tuple(self._create_component_alignment_capacities())

    def _create_component_alignment_capacities(self) -> \
            Tuple[Tuple[int, int, float]]:
        ctx: GraphAttributeContext = self.doc_graph.graph_attrib_context
        amr_caps = self.connections['src_ix smy_ix capacity'.split()].\
            itertuples(index=False, name=None)
        doc_node_caps = self._get_doc_node_capacities(ctx.doc_capacity)
        return chain.from_iterable((amr_caps, doc_node_caps))

    def _get_sinks(self, s: Vertex, t: Vertex) -> \
            Iterable[Tuple[int, int, float]]:
        """This implementation returns all the summary verticies that have no
        out degere, which are the leaf/terminal nodes of the summary graph.

        :return: tuples of ``(<source vertex>, <target vertex>, <capacity>)``

        """
        ti: int = t.index
        cap: int = self.doc_graph.graph_attrib_context.sink_capacity
        v: Vertex
        for v in self.doc_graph.summary.select_vertices(_outdegree=0):
            yield (v.index, ti, cap)

    def _get_sources(self, s: Vertex, t: Vertex) -> \
            Iterable[Tuple[int, int, float]]:
        root: Vertex = self.doc_graph.source.root
        return ((s.index, root.index, self.source_flow),)

    def find_flow_diffs(self, diff_value: float = 0,
                        compare: Callable = lambda x, y: x != y) -> \
            Tuple[Edge, GraphNode]:
        """Find edges having a flow to capacity delta.

        :param diff_value: the value matching ``capacity - flow``

        :param compare: the function to compare, which defaults to an inequality

        :return: tuples of edges to graph edges

        """
        def filter_edge(e: Edge, ge: GraphEdge):
            return compare(float(ge.capacity) - float(ge.flow), diff_value)

        return filter(lambda x: filter_edge(*x), self.doc_graph.es.items())

    def _add_alignments(self):
        self.add_edges(self.component_alignment_capacities)

    def build(self):
        ctx: GraphAttributeContext = self.doc_graph.graph_attrib_context
        edges: Iterable[GraphEdge] = self.doc_graph.es.values()
        self._add_alignments()
        self.set_capacities(edges, ctx.default_capacity)
        # force create source and sink nodes so they can be referenced in
        # capacity setting algos rather than lazy create
        self._get_source_sink_flow_nodes()


@dataclass
class ReverseFlowGraphAlignmentConstructor(SummaryGraphAlignmentConstructor):
    """A constructor that::

      * reverses the edges of the graph

      * connects the source flow node to the leaf AMR nodes of the source and
        summary components

      * connects the sink flow node to the sentence nodes of the source
        component

    """
    reverse_alignments: bool = field(default=False)
    """If ``True``, the bipartite alignments flow from the summary to the
    source.  Otherwise, they flow from the source to the summary.

    """
    def requires_reversed_edges(self) -> bool:
        return True

    def _create_component_alignment_capacities(self) -> \
            Tuple[Tuple[int, int, float]]:
        caps = self.connections['src_ix smy_ix capacity'.split()].\
            itertuples(index=False, name=None)
        if self.reverse_alignments:
            caps = map(lambda x: (x[1], x[0], x[2]), caps)
        return caps

    def _get_sinks(self, s: Vertex, t: Vertex) -> \
            Iterable[Tuple[int, int, float]]:
        ti: int = t.index
        cap: int = self.doc_graph.graph_attrib_context.sink_capacity
        sink_ix: Vertex = self._sink_vertex
        yield (sink_ix.index, ti, cap)

    def _get_sources(self, s: Vertex, t: Vertex) -> \
            Iterable[Tuple[int, int, float]]:
        si: int = s.index
        v: Vertex
        for v in self._summary_terminals:
            yield (si, v.index, self.source_flow)

    def build(self):
        flow_to_comp: DocumentGraphComponent
        flow_from_comp: DocumentGraphComponent

        if self.reverse_alignments:
            flow_to_comp = self.source
            flow_from_comp = self.summary
            self._sink_vertex: Vertex = self.source.root
        else:
            flow_to_comp = self.summary
            flow_from_comp = self.source
            self._sink_vertex: Vertex = self.summary.root

        # grab the terminal nodes of the summary graph before they're connected,
        # after which, they'll be connected and harder to find
        self._summary_terminals = tuple(
            flow_from_comp.select_vertices(_indegree=0))

        super().build()
        dg: DocumentGraph = self.doc_graph
        # set source AMR edges capacities
        self.set_capacities(map(lambda e: dg.es[e], flow_to_comp.es.keys()),
                            GraphEdge.MAX_CAPACITY)
        # set source AMR edges capacities
        self.set_capacities(map(lambda e: dg.es[e], flow_from_comp.es.keys()),
                            GraphEdge.MAX_CAPACITY)


@dataclass
class SharedGraphAlignmentConstructor(ReverseFlowGraphAlignmentConstructor):
    """Creates graph alignment :class:`..ComponentAlignmentGraphEdge` instances
    from another graph so they can share capacity updates.

    """
    org_constructor: ReverseFlowGraphAlignmentConstructor = field(default=None)
    """The original constructor that has the graph to query and copy alignments
    (see class docs).

    """
    def _add_alignments(self):
        oc: ReverseFlowGraphAlignmentConstructor = self.org_constructor
        og: DocumentGraph = oc.doc_graph
        cg: DocumentGraph = self.doc_graph
        e: Edge
        ge: GraphEdge
        for e, ge in og.es.items():
            if isinstance(ge, ComponentAlignmentGraphEdge):
                os: GraphNode = og.node_by_id(e.source)
                ot: GraphNode = og.node_by_id(e.target)
                cs: Vertex = cg.node_to_vertex[os]
                ct: Vertex = cg.node_to_vertex[ot]
                if oc.reverse_alignments:
                    tmp: Vertex = cs
                    cs = ct
                    ct = tmp
                ce: Edge = cg.graph.add_edge(cs, ct)
                GraphComponent.set_edge(ce, ge)
        cg.invalidate()
