"""Base graph component class.

"""
from __future__ import annotations
__author__ = 'Paul Landes'
from typing import (
    ClassVar, List, Iterable, Dict, Tuple, Set, Optional, Union, Any, Type
)
from dataclasses import dataclass, field
import logging
import sys
import itertools as it
from itertools import chain
from io import TextIOBase
from frozendict import frozendict
import igraph as ig
from igraph import Graph, Vertex, Edge
from zensols.util import APIError
from zensols.config import Writable
from zensols.persist import persisted, PersistedWork, PersistableContainer
from . import ComponentAlignmentError, GraphAttribute, GraphNode, GraphEdge

logger = logging.getLogger(__name__)


@dataclass
class _GlobalRestoreContext(object):
    """A global context that is passed through the entire graph tree structure.
    This has all the :class:`.GraphAttribute` instances that are shared between
    all graphs (root nascent factory built, it's components, then its flow
    children).

    """
    nodes: Dict[int, GraphNode] = field(default_factory=dict)
    edges: Dict[int, GraphEdge] = field(default_factory=dict)


@dataclass
class _RestoreContext(Writable):
    """Used to persist the state of a :class:`.GraphComponent`.

    """
    name: str = field()
    roots: Set[int] = field()
    nodes: List[int] = field()
    edges: List[Tuple[int, int, int]] = field()
    state: Dict[str, Any] = field(default_factory=dict)
    gn2v: Dict[int, int] = field(default_factory=dict)
    comp: GraphComponent = field(default=None)

    def graph_node_by_id(self, gn_id: int) -> GraphNode:
        v: int = self.gn2v[gn_id]
        return self.comp.node_by_id(v)

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        self._write_line(str(self), depth, writer)
        self._write_line('nodes:', depth, writer)
        id2n = {n.id: n for n in self.nodes}
        gn: GraphNode
        for gn in self.nodes:
            self._write_line(f'{gn} ({gn.id})', depth + 1, writer)
        self._write_line('edges:', depth, writer)
        ge: GraphEdge
        for src, targ, ge in self.edges:
            sg: GraphNode = id2n.get(src, f'<{src}>')
            tg: GraphNode = id2n.get(targ, f'<{targ}>')
            self._write_line(f'{sg} <- {ge} -> {tg}', depth + 1, writer)

    def __str__(self) -> str:
        skeys = ', '.join(self.state.keys())
        return (f'name: {self.name}: roots: {self.roots}, ' +
                f'n_nodes: {len(self.nodes)} ' +
                f'n_edges: {len(self.edges)}, state: {skeys}, ' +
                f'gn2v size: {len(self.gn2v)}')


@dataclass
class GraphComponent(PersistableContainer, Writable):
    """A container class for an :class:`igraph.Graph`, which also has caching
    data structures for fast access to graph attributes.

    """
    _PERSITABLE_TRANSIENT_ATTRIBUTES: ClassVar[Set[str]] = \
        set('_igraph _v_sub _e_sub'.split())

    ROOT_ATTRIB_NAME: ClassVar[str] = 'root'
    """The attribute that identifies the root vertex."""

    GRAPH_ATTRIB_NAME: ClassVar[str] = 'ga'
    """The name of the graph attributes on igraph nodes and edges."""

    graph: Graph = field()
    """The graph used for computational manipulation of the synthesized AMR
    sentences.

    """
    def __post_init__(self):
        super().__init__()
        # sets of Vertex and Edge instances
        self._vs = PersistedWork('_vs', self, transient=True)
        self._es = PersistedWork('_es', self, transient=True)
        # graph to edge
        self._gete = PersistedWork('_gete', self, transient=True)
        # dict of GraphNode to iGraph Vertex instances
        self._ntov = PersistedWork('_ntov', self, transient=True)
        # graph node index to vertex index
        self._gn2v = PersistedWork('_gn2v', self, transient=True)
        # graph edge index to vertex index
        self._ge2e = PersistedWork('_ge2e', self, transient=True)
        # adjacency list of vectors as list of list of vertex ints
        self._adjacency_list = PersistedWork(
            '_adjacency_list', self, transient=True)
        self._edges_reversed = False

    @staticmethod
    def graph_instance() -> Graph:
        """Create a new directory nascent graph."""
        return Graph(directed=True)

    @property
    def _graph(self) -> Graph:
        """The :mod:`igraph` graph."""
        return self._igraph

    @_graph.setter
    def _graph(self, graph: Graph):
        """The :mod:`igraph` graph."""
        self._set_graph(graph)

    def _set_graph(self, graph: Graph):
        self._igraph = graph
        self._invalidate(True)

    def _invalidate(self, reset_subsets: bool, vs: Set[GraphNode] = None,
                    es: Set[GraphEdge] = None):
        """Clear cached data structures to force them to be recreated after
        igraph level data has changed.  Graph and edges indexes are reset and
        taken from the current graph.

        :param reset_subsets: whether to recreate vertex and edge indexes sets

        """
        attr: str = self.GRAPH_ATTRIB_NAME
        if hasattr(self, '_vs'):
            pws = (self._vs, self._es, self._gete, self._ntov,
                   self._gn2v, self._ge2e, self._adjacency_list)
            for p in pws:
                p.clear()
        if reset_subsets:
            self._v_sub: Set[int] = set(self._igraph.vs.indices)
            self._e_sub: Set[int] = set(self._igraph.es.indices)
        if vs is not None:
            vsp: Dict[Vertex, GraphNode] = {}
            v_sub: Set[int] = set()
            v: Vertex
            for v in self.graph.vs:
                gn: GraphNode = v[attr]
                if gn in vs:
                    vsp[v] = gn
                    v_sub.add(v.index)
            self._v_sub: Set[int] = v_sub
            self._vs.set(frozendict(vsp))
        if es is not None:
            esp: Dict[Edge, GraphNode] = {}
            e_sub: Set[int] = set()
            e: Edge
            for e in self.graph.es:
                ge: GraphEdge = e[attr]
                if ge in es:
                    esp[e] = ge
                    e_sub.add(e.index)
            self._e_sub: Set[int] = e_sub
            self._es.set(frozendict(esp))

    def invalidate(self):
        """Clear cached data structures to force them to be recreated after
        igraph level data has changed.

        """
        self._invalidate(True)

    @staticmethod
    def _resolve_root(org_graph: Graph, org_root: Vertex,
                      new_graph: Graph) -> Tuple[GraphNode, Vertex]:
        ga: str = GraphComponent.GRAPH_ATTRIB_NAME
        # the old root graph node is given by the currently set root vertex
        old_root_doc: GraphNode = org_graph.vs[org_root.index][ga]
        # find the root vertex in the new graph by finding the root graph node
        # since all graph nodes/edges are shared across all graphs
        new_root: Vertex = next(iter(filter(
            lambda v: id(v[ga]) == id(old_root_doc), new_graph.vs)))
        return old_root_doc, new_root

    @staticmethod
    def _reverse_edges(g: Graph, root: Vertex = None) -> Graph:
        """Reverse the direction on all edges in the graph."""
        ga: str = GraphComponent.GRAPH_ATTRIB_NAME
        ra: str = GraphComponent.ROOT_ATTRIB_NAME
        # vertices that make up a component having the root as a vertex
        subgraph_vs: List[int]
        if root is None:
            subgraph_vs = g.vs.indices
        else:
            subgraph_vs = sorted(g.subcomponent(root))
        # edges in the component aka subgraph
        es: Tuple[Edge] = tuple(g.es.select(_incident_in=subgraph_vs))
        # graph edge instances to be set on the new graph
        edge_ctx: List[GraphEdge] = g.es.select(_incident_in=subgraph_vs)[ga]
        # reversed edges
        el: Iterable[Tuple[int, int]] = map(lambda e: (e.target, e.source), es)
        # create the new graph and set the collected data on it
        ng = ig.Graph(el, directed=True)
        ng.es[ga] = edge_ctx
        ng.vs[ga] = g.vs[subgraph_vs][ga]
        ng.vs[ra] = g.vs[subgraph_vs][ra]
        return ng

    def copy_graph(self, reverse_edges: bool = False,
                   subgraph_type: str = None) -> Graph:
        """Return a copy of the class:`igraph.Graph`.

        :param reverse_edges: whether to reverse the direction on all edges in
                              the graph

        :param subgraph_type: the method of creating a subgraph, which is either
                              ``induced`` to create it from the nodes of the
                              graph; or ``sub`` (the default) to create it from
                              the subcomponent from the root

        """
        g: Graph = self.graph
        subgraph_type = 'sub' if subgraph_type is None else subgraph_type
        # vertices that make up a component having the root as a vertex
        copy: Graph
        if subgraph_type == 'induced':
            copy = self.graph.induced_subgraph(
                self.vs.keys(), 'create_from_scratch')
        elif subgraph_type == 'sub':
            copy = g.subgraph(g.subcomponent(self.root))
        else:
            raise APIError(f'Unknown subgraph type: {subgraph_type}')
        if reverse_edges:
            # find the vertex in the new graph using the old graph and root
            old_root_doc: GraphNode
            new_root: Vertex
            old_root_doc, new_root = self._resolve_root(g, self.root, copy)
            copy = self._reverse_edges(copy, new_root)
        return copy

    def clone(self, reverse_edges: bool = False, deep: bool = True,
              **kwargs) -> GraphComponent:
        """Clone an instance and return it.  The :obj:`graph` is deep copied,
        but all :obj:`.GraphAttribute` instances are not.

        :param reverse_edges: whether to reverse the direction of edges in the
                              directed graph, which is used to create the
                              reverse flow graphs used in the maxflow algorithm

        :param deep: whether to create a deep clone, which detatches a component
                     from any bipartite graph by keeping only the graph composed
                     of :obj:`vs`; otherwise the graph is copied as a
                     subcomponent starting from :obj:`root`

        :param kwargs: arguments to add to as attributes to the clone; include
                       ``cls`` is the type of the new instance

        :return: the cloned instance of this instance

        """
        params: Dict[str, Any] = kwargs
        cls: Type[GraphComponent] = kwargs.pop('cls', self.__class__)
        if 'graph' not in params:
            params = dict(kwargs)
            params['graph'] = self.copy_graph(
                reverse_edges=reverse_edges,
                subgraph_type='induced' if deep else 'sub')
        inst: GraphComponent = cls(**params)
        if reverse_edges:
            inst._edges_reversed = not self._edges_reversed
        else:
            inst._edges_reversed = self._edges_reversed
        return inst

    @property
    def edges_reversed(self) -> bool:
        """Whether the edge direction in the graph is reversed.  This is
        ``True`` for reverse flow graphs.

        :see: :class:`.summary.ReverseFlowGraphAlignmentConstructor`

        """
        return self._edges_reversed

    @property
    def roots(self) -> Iterable[Vertex]:
        """The roots of the graph, which are usually top level
        :class:`.DocumentNode` instances.

        """
        vs: Dict[Vertex, GraphNode] = self.vs
        sel_params: Dict = {f'{self.ROOT_ATTRIB_NAME}_eq': True}
        try:
            verts: Iterable[Vertex] = self.graph.vs.select(**sel_params)
            return filter(lambda v: v in vs, verts)
        except KeyError:
            pass

    @property
    def root(self) -> Optional[Vertex]:
        """The singular (first found) root of the graph, which is usually the
        top level :class:`.DocumentNode` instance.

        """
        try:
            return next(iter(self.roots))
        except StopIteration:
            pass

    @property
    @persisted('_ntov')
    def node_to_vertex(self) -> Dict[GraphNode, Vertex]:
        """A mapping from graph nodes to vertexes."""
        return frozendict({x[1]: x[0] for x in self.vs.items()})

    @property
    @persisted('_gete')
    def graph_edge_to_edge(self) -> Dict[GraphEdge, Edge]:
        """A mapping from graph nodes to vertexes."""
        return frozendict({x[1]: x[0] for x in self.es.items()})

    @property
    @persisted('_vs')
    def vs(self) -> Dict[Vertex, GraphNode]:
        """The igraph to domain object vertex mapping."""
        g: Graph = self._igraph
        ga: str = GraphComponent.GRAPH_ATTRIB_NAME
        vs: Dict[Vertex, GraphNode] = {}
        i: int
        for i in self._v_sub:
            v: Vertex = g.vs[i]
            vs[v] = v[ga]
        return frozendict(vs)

    @property
    @persisted('_es')
    def es(self) -> Dict[Edge, GraphEdge]:
        """The igraph to domain object edge mapping."""
        g: Graph = self._igraph
        ga: str = GraphComponent.GRAPH_ATTRIB_NAME
        es: Dict[Edge, GraphEdge] = {}
        i: int
        for i in self._e_sub:
            e: Edge = g.es[i]
            es[e]: GraphEdge = e[ga]
        return frozendict(es)

    def get_attributes(self) -> Iterable[GraphAttribute]:
        """Return all graph attributes of the component, which include instances
        of both :class:`.GraphNode` and :class:`.GraphEdge`.

        """
        return chain.from_iterable((self.vs.values(), self.es.values()))

    @property
    @persisted('_gn2v')
    def graph_node_id_to_vertex_ref(self) -> Dict[int, int]:
        """Graph node index to vertex index."""
        return frozendict(map(lambda t: (t[1].id, t[0].index), self.vs.items()))

    @property
    @persisted('_ge2e')
    def graph_edge_id_to_edge_ref(self) -> Dict[int, int]:
        """Graph node index to vertex index."""
        return frozendict(map(lambda t: (t[1].id, t[0].index), self.es.items()))

    @classmethod
    def to_node(cls, v: Vertex) -> GraphNode:
        """Narrow a vertex to a node."""
        return v[cls.GRAPH_ATTRIB_NAME]

    @classmethod
    def set_node(cls, v: Vertex, n: GraphNode):
        """Set the graph node data in the igraph vertex."""
        v[cls.GRAPH_ATTRIB_NAME] = n
        return v, n

    @classmethod
    def to_edge(cls, e: Edge) -> GraphEdge:
        """Narrow a vertex to a edge."""
        return e[cls.GRAPH_ATTRIB_NAME]

    @classmethod
    def set_edge(cls, e: Edge, ge: GraphEdge):
        """Set the graph edge data in the igraph edge."""
        e[cls.GRAPH_ATTRIB_NAME] = ge
        return e, ge

    def node_by_id(self, ix: int) -> GraphNode:
        """Return the graph node for the vertex ID."""
        v: Vertex = self.graph.vs[ix]
        return self.to_node(v)

    def edge_by_id(self, ix: int) -> GraphEdge:
        """Return the edge for the vertex ID."""
        v: Vertex = self.graph.es[ix]
        return self.to_edge(v)

    def node_by_graph_node_id(self, gn_id: int) -> GraphNode:
        """Return a node based on the graph (attribute) node id."""
        v_id: int = self.graph_node_id_to_vertex_ref.get(gn_id)
        return self.node_by_id(v_id)

    def edge_by_graph_edge_id(self, ge_id: int) -> GraphEdge:
        """Return a edge based on the graph (attribute) edge id."""
        e_id: int = self.graph_edge_id_to_edge_ref.get(ge_id)
        return self.edge_by_id(e_id)

    def vertex_ref_by_id(self, ix: int) -> Vertex:
        """Get the :class:`igraph.Vertex` instance by its index."""
        return self.graph.vs[ix]

    def edge_ref_by_id(self, ix: int) -> Edge:
        """Get the :class:`igraph.Edge` instance by its index."""
        return self.graph.es[ix]

    def select_vertices(self, **kwargs) -> Iterable[Vertex]:
        """Return matched graph nodes from an :meth:`igraph.Graph.vs.select`."""
        vs: Dict[Vertex, GraphNode] = self.vs
        v: Vertex
        for v in self.graph.vs.select(**kwargs):
            if v in vs:
                yield v

    def select_edges(self, **kwargs) -> Iterable[Edge]:
        """Return matched graph edges from an :meth:`igraph.Graph.vs.select`."""
        es: Dict[Edge, GraphEdge] = self.es
        e: Edge
        for e in self.graph.es.select(**kwargs):
            if e in es:
                yield e

    @property
    @persisted('_adjacency_list')
    def adjacency_list(self) -> List[List[int]]:
        """"An adjacency list of vertexes based on their relation to each other
        in the graph.  The outer list's index is the source vertex and the inner
        list is that vertex's neighbors.

        **Implementation note**: the list is sub-setted at both the inner and
        outer level for those vertexes in this component.

        """
        def map_verts(x: Tuple[int, List[int]]) -> Tuple[int]:
            if x[0] in verts:
                return tuple(filter(lambda v: v in verts, x[1]))
            else:
                return ()

        mode: str = 'in' if self._edges_reversed else 'out'
        verts = set(map(lambda v: v.index, self.vs.keys()))
        al: List[List[int]] = self.graph.get_adjlist(mode=mode)
        return tuple(map(map_verts, zip(it.count(), al)))

    def delete_edges(self, edges: Iterable[GraphEdge],
                     include_nodes: bool = False) -> \
            Tuple[Set[GraphEdge], Set[GraphNode]]:
        """Remove edges from the graph.

        :param edges: the edges to remove form the graph

        :param include_nodes: whether to remove nodes that become orphans after
                              deleting ``edges``

        :return: a tuple the edges and nodes removed

        """
        gattr: str = self.GRAPH_ATTRIB_NAME
        ge2e: Dict[GraphEdge, Edge] = self.graph_edge_to_edge
        e_removed: Set[GraphEdge] = set()
        n_removed: Set[GraphNode]
        eid_removed: Set[int] = set()
        vid_removed: Set[int] = set()
        e: GraphEdge
        for ge in edges:
            e: Edge = ge2e[ge]
            eid_removed.add(e.index)
            e_removed.add(ge)
            if include_nodes:
                vid_removed.add(e.source)
                vid_removed.add(e.target)
        if len(eid_removed) > 0 or len(vid_removed) > 0:
            self.graph.delete_edges(eid_removed)
            v_removed: Tuple[Vertex] = tuple(filter(
                lambda v: v.index in vid_removed,
                self.graph.vs.select(_degree=0)))
            vid_removed = tuple(map(lambda v: v.index, v_removed))
            n_removed = set(map(lambda v: v[gattr], v_removed))
            self.graph.delete_vertices(vid_removed)
            self._invalidate(True)
        else:
            n_removed = set()
        return (e_removed, n_removed)

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        self._write_line('vertexes:', depth, writer)
        v: Vertex
        n: GraphNode
        for v, n in self.vs.items():
            self._write_line(f'v: {v.index}: {n}', depth + 1, writer)
        self._write_line('edges:', depth, writer)
        v: Edge
        for e, n in self.es.items():
            self._write_line(f'v: {e.index}: {n}', depth + 1, writer)

    def deallocate(self):
        if self._vs.is_set() and self.vs is not None:
            for o in self.vs.values():
                self._try_deallocate(o)
        if self._es.is_set() and self.es is not None:
            for o in self.es.values():
                self._try_deallocate(o)
        super().deallocate()
        self._invalidate(True)

    @staticmethod
    def create_vertexes(g: Graph, n: int) -> Tuple[Vertex]:
        """Create ``n`` vertexes and return them after being added."""
        start: int = len(g.vs)
        g.add_vertices(n)
        return tuple(map(lambda i: g.vs[start + i], range(n)))

    @staticmethod
    def create_edges(g: Graph, es: Tuple[Tuple[int, int]]) -> Tuple[Edge]:
        """Add the edges of list ``es`` and return them after being added."""
        start: int = len(g.es)
        g.add_edges(es)
        return tuple(map(lambda i: g.es[start + i], range(len(es))))

    def _get_restore_context(self, skip_nodes: Set[Vertex] = None,
                             skip_edges: Set[Edge] = None) -> _RestoreContext:
        glob: _GlobalRestoreContext = self._glob
        edges: List[Tuple[int, int, GraphEdge]] = []
        # root GraphAttribute IDs are either doc root (source) or terminals
        roots: Set[int] = set(map(lambda v: self.vs[v].id, self.roots))
        # addd and potentially clobber dups from previous shared attributes
        glob.nodes.update(dict(map(lambda gn: (gn.id, gn), self.vs.values())))
        glob.edges.update(dict(map(lambda ge: (ge.id, ge), self.es.values())))
        # add and make vertexes have the same order in the unpersisted instance
        # for uniformity debugging
        nodes: List[GraphNode]
        if skip_nodes is None:
            nodes = list(map(lambda gn: gn.id, self.vs.values()))
        else:
            nodes = list(map(lambda t: t[1].id,
                             filter(lambda t: t[0] not in skip_nodes,
                                    self.vs.items())))
        nodes.sort()
        # record GraphAttribute edges as (source vert, target vert, edge)
        skip_edges = () if skip_edges is None else skip_edges
        e: Edge
        ge: GraphEdge
        for e, ge in self.es.items():
            if e not in skip_edges:
                src: GraphNode = self.node_by_id(e.source)
                targ: GraphNode = self.node_by_id(e.target)
                edges.append((src.id, targ.id, ge.id))
        return _RestoreContext(str(self), roots, nodes, edges)

    def _set_restore_context(self, ctx: _RestoreContext):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'{type(self)}: set restore context: {ctx}')
        gattrib: str = self.GRAPH_ATTRIB_NAME
        glob: _GlobalRestoreContext = self._glob
        nodes: Dict[int, GraphNode] = glob.nodes
        edges: Dict[int, GraphEdge] = glob.edges
        # graph node ID to vertex ID
        gn2v: Dict[int, int] = ctx.gn2v
        g: Graph
        # either create the graph instance or build on what's passed in
        if ctx.comp.graph is None:
            # DocumentGraphComponent won't have this set
            g = self.graph_instance()
        else:
            # DocumentGraph has this set because it's built from the comps
            g = self.graph
        # create and add vertexes
        vs: Tuple[Vertex] = self.create_vertexes(g, len(ctx.nodes))
        g.vs[self.ROOT_ATTRIB_NAME] = False
        v: Vertex
        gn_id: int
        for v, gn_id in zip(vs, ctx.nodes):
            gn: GraphNode = nodes[gn_id]
            v[gattrib] = gn
            dup_vert: Vertex = gn2v.get(gn.id)
            if dup_vert is not None:
                raise ComponentAlignmentError(
                    f'Duplicate vertex found while restoring {ctx}: ' +
                    f'{dup_vert}, gn ID: {gn.id}')
            gn2v[gn.id] = v.index
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'added vertex: {gn.id} -> {v}')
        # set the root nodes
        gn_id: int
        for gn_id in ctx.roots:
            v_id: int = gn2v[gn_id]
            v: Vertex = g.vs[v_id]
            v[self.ROOT_ATTRIB_NAME] = True
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'root: {gn_id}[{v_id}] -> {v}')
        # add edges
        es: Tuple[Edge] = self.create_edges(g, tuple(map(
            lambda t: (gn2v[t[0]], gn2v[t[1]]), ctx.edges)))
        e: Edge
        ge_id: int
        for e, ge_id in zip(es, map(lambda t: t[2], ctx.edges)):
            e[gattrib] = edges[ge_id]
        # this must go at end since it invalidates this component
        self.graph = g
        # save flow module from having to recalculate this
        self._gn2v.set(frozendict(gn2v))

    def __getstate__(self) -> Dict[str, Any]:
        glob_existed: bool = hasattr(self, '_glob')
        state: Dict[str, Any] = super().__getstate__()
        if not glob_existed:
            self._glob = _GlobalRestoreContext()
        try:
            state['restore_context'] = self._get_restore_context()
            if not glob_existed:
                state['glob'] = self._glob
            if logger.isEnabledFor(logging.DEBUG):
                keys = tuple(map(lambda t: t[0],
                                 filter(lambda t: t[1] is not None,
                                        state.items())))
                logger.debug(f'{type(self)}: getstate: {keys}')
            return state
        finally:
            if not glob_existed:
                del self._glob

    def __setstate__(self, state: Dict[str, Any]):
        if logger.isEnabledFor(logging.DEBUG):
            keys = tuple(map(lambda t: t[0],
                             filter(lambda t: t[1] is not None, state.items())))
            logger.debug(f'{type(self)}: restore state: {keys}')
        glob_existed: bool = hasattr(self, '_glob')
        if not glob_existed:
            self._glob = state.pop('glob')
        try:
            restore_context: _RestoreContext = state.pop('restore_context')
            super().__setstate__(state)
            if restore_context is not None:
                restore_context.comp = self
                self._set_restore_context(restore_context)
        finally:
            if not glob_existed:
                del self._glob

    def __getitem__(self, key: Union[Vertex, Edge]) -> GraphAttribute:
        data: Dict = self.vs if isinstance(key, Vertex) else self.es
        return data[key]

    def __len__(self) -> int:
        """A graph is a set of vertexes and edges."""
        return len(self.vs) + len(self.es)


GraphComponent.graph = GraphComponent._graph
