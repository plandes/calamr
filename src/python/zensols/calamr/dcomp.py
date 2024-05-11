"""A document centric graph component.

"""
__author__ = 'Paul Landes'

from typing import (
    Tuple, List, Dict, Set, Iterable, Callable, ClassVar, Optional
)
from dataclasses import dataclass, field
import logging
import sys
from itertools import chain
from io import TextIOBase
from frozendict import frozendict
from igraph import Graph, Vertex, Edge
from zensols.config import Dictable
from zensols.persist import persisted
from zensols.amr import RelationSet, AmrFeatureSentence, AmrFeatureDocument
from . import (
    GraphAttribute, GraphNode, AmrDocumentNode, DocumentGraphNode,
    SentenceGraphNode, ConceptGraphNode, AttributeGraphNode, GraphComponent,
)
from .comp import _RestoreContext

logger = logging.getLogger(__name__)


@dataclass
class SentenceEntry(Dictable):
    """Contains the sentence node of a sentence, and the respective concept and
    attribute nodes.

    """
    node: SentenceGraphNode = field(default=None)
    """The sentence node, which is the root of the sentence subgraph."""

    concepts: Tuple[ConceptGraphNode] = field(default=None)
    """The AMR concept nodes of the sentence."""

    attributes: Tuple[AttributeGraphNode] = field(default=None)
    """The AMR attribute nodes of the sentence."""

    @property
    @persisted('_by_variable', transient=True)
    def concept_by_variable(self) -> Dict[str, ConceptGraphNode]:
        return frozendict({n.variable: n for n in self.concepts})

    def _get_state(self):
        return (self.node.id,
                tuple(map(lambda n: n.id, self.concepts)),
                tuple(map(lambda n: n.id, self.attributes)))

    def _set_state(self, state, ctx: _RestoreContext):
        node, concepts, attributes = state
        map_id: Callable = ctx.graph_node_by_id
        self.node = map_id(node)
        self.concepts = tuple(map(map_id, concepts))
        self.attributes = tuple(map(map_id, attributes))


@dataclass
class SentenceIndex(Dictable):
    """An index of the sentences of a :class:`.DocumentGraphComponent`.

    """
    entries: Tuple[SentenceEntry] = field(default=None)
    """Then entries of the index, each of which is a sentence."""

    @property
    @persisted('_by_sentence', transient=True)
    def by_sentence(self) -> Dict[AmrFeatureSentence, SentenceEntry]:
        return frozendict({n.node.sent: n for n in self.entries})

    def _persist_state(self, ctx: _RestoreContext):
        ctx.state['sent_index'] = tuple(map(
            lambda e: e._get_state(), self.entries))

    def _restore_state(self, ctx: _RestoreContext):
        def map_state(entry_state):
            e = SentenceEntry()
            e._set_state(entry_state, ctx)
            return e
        self.entries = tuple(map(map_state, ctx.state['sent_index']))


@dataclass
class DocumentGraphComponent(GraphComponent):
    """A class containing the root information of the document tree and the
    :class:`igraph.Graph` vertex.  When the :class:`igraph.Graph` is set with
    the :obj:`graph` property, a strongly connected subgraph component is
    induced.  It does this by traversing all reachable verticies and edges from
    the :obj:`root`.  Examples of these induced components include *source* and
    *summary* components of a document AMR graph.

    Instances are created by :class:`.DocumentGraphFactory`.

    """
    _PERSITABLE_TRANSIENT_ATTRIBUTES: ClassVar[Set[str]] = \
        GraphComponent._PERSITABLE_TRANSIENT_ATTRIBUTES | \
        {'_root_vertex', 'sent_index'}

    root_node: AmrDocumentNode = field()
    """The root of the document tree."""

    sent_index: SentenceIndex = field(default_factory=SentenceIndex)
    """An index of the sentences of a :class:`.DocumentGraphComponent`."""

    description: str = field(default=None)
    """A description of the component used for debugging."""

    def __post_init__(self):
        super().__post_init__()
        self._root_vertex = None
        assert isinstance(self.root_node, AmrDocumentNode)

    @property
    def name(self) -> str:
        """Return the name of the AMR document node."""
        return self.root_node.name

    @property
    def root(self) -> Optional[Vertex]:
        """The roots of the graph, which are usually top level
        :class:`.DocumentNode` instances.

        """
        return self._root_vertex

    @root.setter
    def root(self, vertex: Vertex):
        """The roots of the graph, which are usually top level
        :class:`.DocumentNode` instances.

        """
        self._root_vertex = vertex

    def _set_graph(self, graph: Graph):
        # check for attribute since this is called before __post_init__ when
        # this instance created by the super class dataclass setter
        if hasattr(self, '_root_vertex') and self._root_vertex is not None:
            self._induce_component(self._root_vertex, self.graph, graph)
            self._igraph = graph
        else:
            super()._set_graph(graph)

    def _induce_component(self, root: Vertex, org_graph: Graph, graph: Graph):
        """Reset data structures so that this graph becomes a strongly connected
        partition of the larger graph.

        """
        # new and old graphs
        ng: Graph = graph
        # find the vertex in the new graph using the old graph and root
        old_root_doc: GraphNode
        new_root: Vertex
        old_root_doc, new_root = self._resolve_root(org_graph, root, ng)
        # finds all vertices reachable in the new graph from the root vertex in
        # the same new graph, which gives us the component, which is a subset of
        # the graph nodes if disconnected and yields the new compoennt
        subcomp_verticies: List[int] = ng.subcomponent(new_root)
        # get all edges connected to the new (sub)component we have identified
        # from the vertexes
        subcomp_edges: Iterable[Edge] = ng.es.select(_within=subcomp_verticies)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'{old_root_doc} -> {new_root}')
            logger.debug(f'subgraph: {sorted(subcomp_verticies)}')

        # udpate data structures to use the subcomponent
        self._invalidate(reset_subsets=False)
        self._v_sub = set(subcomp_verticies)
        self._e_sub = set(map(lambda e: e.index, subcomp_edges))
        self._root_vertex = new_root

    @property
    def doc_vertices(self) -> Iterable[Vertex]:
        """Get the vertices of :class:`.DocuemntGraphNode`.  This only fetches
        those document nodes that do not branch.

        """
        v: Vertex = self.root
        while True:
            gn: GraphNode = self.to_node(v)
            if not isinstance(gn, DocumentGraphNode):
                break
            yield v
            ns: List[Vertex] = v.neighbors()
            if len(ns) >= 2:
                break
            v = ns[0]

    @property
    def relation_set(self) -> RelationSet:
        """The relations in the contained root node document."""
        doc: AmrFeatureDocument = self.root_node.doc
        return doc.relation_set

    def get_attributes(self) -> Iterable[GraphAttribute]:
        return chain.from_iterable((
            super().get_attributes(), (self.root_node,)))

    def clone(self, reverse_edges: bool = False, deep: bool = True,
              **kwargs) -> GraphComponent:
        params = dict(root_node=self.root_node, sent_index=self.sent_index)
        params.update(kwargs)
        if 'description' not in params:
            params['description'] = self.description
        inst = super().clone(reverse_edges, deep, **params)
        inst._induce_component(self.root, self.graph, inst.graph)
        return inst

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        super().write(depth, writer)
        self._write_object(self.root_node, depth, writer)

    def deallocate(self):
        super().deallocate()
        del self.root_node
        del self.sent_index

    def _get_restore_context(self, skip_nodes: Set[Vertex] = None,
                             skip_edges: Set[Edge] = None) -> _RestoreContext:
        ctx: _RestoreContext = super()._get_restore_context(
            skip_nodes, skip_edges)
        gn: GraphNode = self.to_node(self.root)
        ctx.state['root_node'] = gn.id
        self.sent_index._persist_state(ctx)
        return ctx

    def _set_restore_context(self, ctx: _RestoreContext):
        super()._set_restore_context(ctx)
        gn_id: int = ctx.state['root_node']
        v_id: int = ctx.gn2v[gn_id]
        self.root: Vertex = self.graph.vs[v_id]
        self.sent_index = SentenceIndex()
        self.sent_index._restore_state(ctx)

    def __str__(self) -> str:
        return f'{self.name} ({self.description})'
