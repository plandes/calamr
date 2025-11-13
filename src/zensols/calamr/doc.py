"""Document based graph container, factory and strategy classes.

"""
from __future__ import annotations
__author__ = 'Paul Landes'
from typing import (
    Dict, Tuple, Any, List, Set, Iterable, ClassVar, Type, Optional
)
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
import sys
from itertools import chain
from functools import reduce
from io import TextIOBase
from frozendict import frozendict
import igraph as ig
from igraph import Graph, Vertex, Edge
from zensols.util import time
from zensols.config import ConfigFactory
from zensols.persist import persisted, PersistedWork
from zensols.amr import Relation, RelationSet, AmrFeatureDocument
from . import (
    ComponentAlignmentError, GraphAttributeContext, GraphComponent,
    GraphNode, GraphEdge, DocumentGraphComponent,
)
from .comp import _RestoreContext, _GlobalRestoreContext

logger = logging.getLogger(__name__)


@dataclass
class DocumentGraph(GraphComponent):
    """A graph containing the text, text features, AMR Penman graph and igraph.

    This class roughly follows a GoF composite pattern with :obj:`children` a
    collection of instance of this class, which are the reversed source and
    summary graphs created for the max flow algorithm.  The root is constructed
    from the :class:`.DocumentGraphFactory` class and the children are built by
    the :class:`.DocumentGraphController` instances.

    The children of this composite are not to be confused with
    :obj:`components`, which are the disconnected source and summary graph
    components in the root graph instance.  Each child also has the reversed
    flow graphs, but are connected as a bipartite flow graph for use by the max
    flow algorithm.

    """
    _PERSITABLE_TRANSIENT_ATTRIBUTES: ClassVar[Set[str]] = \
        GraphComponent._PERSITABLE_TRANSIENT_ATTRIBUTES | \
        {'_graph_attrib_context_val', 'components', 'children'}

    name: str = field()
    """The name of the graph used to identify it.  For now, this is only
    ``reversed_source`` for the graph that flows from the summary to the source,
    and ``reversed_summary for the graph that flows from the source to the
    summary.  These are "reversed" because the flow is reversed from the leaf
    nodes to the root.

    """
    graph_attrib_context: GraphAttributeContext = field()
    """The context given to all nodees and edges of the graph."""

    doc: AmrFeatureDocument = field()
    """The document that represents the graph."""

    components: Tuple[DocumentGraphComponent, ...] = field()
    """The roots of the trees created by the :class:`.DocumentGraphFactory`.

    """
    children: Dict[str, DocumentGraph] = field(default_factory=dict)
    """The children of this instance, which for now, are only instances of
    :class:`.FlowDocumentGraph`.

    """
    def __post_init__(self):
        super().__post_init__()
        self._components_by_name = PersistedWork(
            '_components_by_name', self, transient=True)
        # see clone method
        if self.components is not None:
            self._build()

    @property
    def _graph_attrib_context(self) -> GraphAttributeContext:
        return self._graph_attrib_context_val

    @_graph_attrib_context.setter
    def _graph_attrib_context(self, context: GraphAttributeContext):
        self._graph_attrib_context_val = context
        if hasattr(self, 'components'):
            comp: DocumentGraphComponent
            for comp in self.components:
                gn: GraphNode
                for gn in comp.vs.values():
                    gn.context = context
                ge: GraphEdge
                for ge in comp.es.values():
                    ge.context = context
            child: DocumentGraph
            for child in self.children.values():
                child.graph_attrib_context = context

    @property
    @persisted('_components_by_name')
    def components_by_name(self) -> Dict[str, DocumentGraphComponent]:
        """Get document graph components by name."""
        return frozendict({c.name: c for c in self.components})

    @property
    def components_by_name_sorted(self) -> \
            Tuple[Tuple[str, DocumentGraphComponent], ...]:
        """Get document graph components sorted name."""
        return tuple(sorted(self.components, key=lambda c: c.name))

    def add_child(self, child: DocumentGraph):
        """Add a child graph.

        :see: :obj:`children`

        """
        self.children[child.name] = child

    def component_iter(self) -> Iterable[GraphComponent]:
        """Return an iterable of the components of this graph and recursively
        over the children.

        """
        return chain.from_iterable(
            (self.components,
             self.children.values(),
             chain.from_iterable(
                 map(lambda c: c.component_iter(), self.children.values()))))

    def get_containing_component(self, n: GraphNode) -> DocumentGraphComponent:
        """Return the component that contains graph node ``n``."""
        comp: DocumentGraphComponent
        for comp in self.components:
            v: Vertex = comp.node_to_vertex.get(n)
            if v is not None and v in comp.vs:
                return comp

    def _build(self):
        # get each components induced graph
        graphs: Tuple[Graph, ...] = tuple(map(
            lambda c: c.graph, self.components))
        # union the document graphs in to one igraph instance as disconnected
        # components
        graph: Graph = ig.disjoint_union(graphs)
        # set the component graphs to the union graph
        comp: DocumentGraphComponent
        for comp in self.components:
            comp.graph = graph
        # set our own container graph to the union graph
        self.graph = graph

    def _clone_components(self, reverse_edges: bool, deep: bool,
                          components: Tuple[DocumentGraphComponent, ...]) -> \
            Tuple[DocumentGraphComponent]:
        return tuple(map(lambda c: c.clone(reverse_edges, deep), components))

    def clone(self, reverse_edges: bool = False, deep: bool = True,
              **kwargs) -> GraphComponent:
        if deep:
            return self._clone_deep(reverse_edges, **kwargs)
        else:
            return self._clone_shallow(reverse_edges, **kwargs)

    def _clone_shallow(self, reverse_edges: bool = False, **kwargs) -> \
            GraphComponent:
        params = dict(map(lambda a: (a, getattr(self, a)),
                          'graph_attrib_context doc'.split()))
        params.update(kwargs)
        if 'components' not in params:
            params['components'] = None
        if 'children' not in params:
            params['children'] = {}
        if 'graph' not in params:
            params['graph'] = self.graph_instance()
        if 'name' not in params:
            params['name'] = self.name
        clone = super().clone(**params)
        if clone.components is None:
            # see __post_init__
            clone.components = clone._clone_components(
                reverse_edges, False, self.components)
            clone._build()
        return clone

    def _clone_deep(self, reverse_edges: bool = False, **kwargs) -> \
            GraphComponent:
        def map_edge(e: Edge) -> Optional[Tuple[Tuple[int, int]], GraphEdge]:
            src: int = vmap[self.node_by_id(e.source)]
            targ: int = vmap[self.node_by_id(e.target)]
            st: Tuple[int, int] = (targ, src) if reverse_edges else (src, targ)
            if st not in inst_edges:
                return (st, e[gattr])

        # build the graph based on the components only
        gattr: str = self.GRAPH_ATTRIB_NAME
        params: Dict[str, Any] = {}
        for k, v in self.__dict__.items():
            if not k.startswith('_') and k not in kwargs:
                params[k] = v
        params.update(kwargs)
        params['components'] = None
        params['graph'] = self.graph_instance()
        params['children'] = {}
        if 'doc' not in params:
            params['doc'] = self.doc
        if 'graph_attrib_context' not in params:
            params['graph_attrib_context'] = self._graph_attrib_context_val
        cls: Type[GraphComponent] = params.pop('cls', self.__class__)
        inst = cls(**params)
        inst.components = inst._clone_components(
            reverse_edges, True, self.components)
        # see __post_init__
        inst._build()
        inst_graph: Graph = inst.graph

        # add nodes not in the components (i.e. terminal nodes)
        inst_n2v = dict(map(lambda v: (v[gattr], v.index), inst_graph.vs))
        gn_adds: Tuple[GraphNode] = tuple(filter(
            lambda gn: gn not in inst_n2v, self.vs.values()))
        v_adds: Tuple[Vertex] = inst.create_vertexes(inst_graph, len(gn_adds))
        for v, gn in zip(v_adds, gn_adds):
            v[gattr] = gn

        # add edges not in the components (i.e. component alignment edges)
        vmap: Dict[GraphNode, int] = dict(map(
            lambda v: (v[gattr], v.index), inst.graph.vs))
        inst_edges: Set[Tuple[int, int]] = set(map(
            lambda e: (e.source, e.target), inst_graph.es))
        ge_adds: Tuple[Tuple[Tuple[int, int], GraphEdge], ...] = \
            tuple(filter(lambda e: e is not None,
                         map(map_edge, self.es.keys())))
        etups: Tuple[Tuple[int, int], ...] = tuple(map(lambda t: t[0], ge_adds))
        edges: Tuple[GraphEdge] = tuple(map(lambda t: t[1], ge_adds))
        org_edge: GraphEdge
        en: Edge
        for org_edge, en in zip(edges, inst.create_edges(inst_graph, etups)):
            en[gattr] = org_edge

        # clear cached data structures since we added nodes and edges
        inst.invalidate()

        # clone children
        inst.children = frozendict({
            t[0]: t[1].clone(reverse_edges)
            for t in self.children.items()})

        return inst

    def _sanity_check(self, do_print: bool = False, do_assert: bool = True):
        stats: Dict[str, Dict[str, Any]] = self.stats
        diffs = stats['diffs']
        if do_print:
            from pprint import pprint
            pprint(stats)
        if do_assert:
            if not all(map(lambda x: x == 0, diffs.values())):
                raise ComponentAlignmentError(
                    f'Found extraineous vertexes and/or edges: {diffs}')
        return stats

    def delete_edges(self, edges: Iterable[GraphEdge],
                     include_nodes: bool = False):
        # first save off each component's graph nodes and edges
        comp_vs_es: Tuple[Set[GraphNode], Set[GraphEdge]] = {}
        comp: GraphComponent
        for comp in self.components:
            comp_vs_es[comp.name] = \
                (set(comp.vs.values()), set(comp.es.values()))
        # remove the client call's requested edges and attached nodes
        e_removed: Set[GraphEdge]
        v_removed: Set[GraphNode]
        e_removed, v_removed = super().delete_edges(edges, include_nodes)
        # when edges is non-empty, we'll have deleted something, so now add back
        # the components' graph nodes and edges and invalidate
        if len(e_removed) > 0 or len(v_removed) > 0:
            comp: GraphComponent
            for comp in self.components:
                vs, es = comp_vs_es[comp.name]
                comp._invalidate(False, vs, es)

    @property
    def bipartite_relation_set(self) -> RelationSet:
        """The bipartite relations that span components.  This set includes all
        top level relations that are not self contained in any components.

        """
        comp_rel_set: Set[Relation] = set()
        global_rel_set: Set[Relation] = self.doc.relation_set.as_set()
        for comp in self.components:
            comp_rel_set.update(comp.relation_set.as_set())
        return global_rel_set - comp_rel_set

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        self._write_line('doc:', depth, writer)
        super().write(depth + 1, writer)
        self._write_line('components:', depth, writer)
        comp: DocumentGraphComponent
        for comp in self.components:
            g: Graph = comp.graph
            self._write_line(f'component: (v={len(g.vs)}, e={len(g.es)})',
                             depth + 1, writer)
            self._write_object(comp, depth + 2, writer)

    def deallocate(self):
        for c in self.children.values():
            self._try_deallocate(c)
        for c in self.components:
            self._try_deallocate(c)
        self._try_deallocate(self.doc)
        super().deallocate()
        self.doc = None
        self.components = None
        self.children = None

    def _get_restore_context(self, skip_nodes: Set[Vertex] = None,
                             skip_edges: Set[Edge] = None) -> _RestoreContext:
        # skip (don't in this GraphComponent) verts and edges from components as
        # that is added in in the context state below
        if len(self.components) > 0:
            skip_nodes = reduce(
                lambda x, y: x | y,
                map(lambda c: set(c.vs.keys()), self.components))
            skip_edges = reduce(
                lambda x, y: x | y,
                map(lambda c: set(c.es.keys()), self.components))
        ctx: _RestoreContext = super()._get_restore_context(
            skip_nodes, skip_edges)
        # the context now has only the terminal nodes and edges, so add the
        # components that will have the rest of the graph and used in _build()
        # to reconnect everything on unpersist
        comp_states: Tuple[Tuple[Type, Dict[str, Any]], ...] = []
        child_states: Tuple[Tuple[str, Type, Dict[str, Any]], ...] = []
        # this graph's components and children are stored as restore context
        # state and will be shared in the global context
        ctx.state['components'] = comp_states
        ctx.state['children'] = child_states
        comp: DocumentGraphComponent
        for comp in self.components:
            # add the global context so GraphAttributes (nodes/edges) are shared
            comp._glob: _GlobalRestoreContext = self._glob
            try:
                comp_states.append((type(comp), comp.__getstate__()))
            finally:
                del comp._glob
        child: DocumentGraph
        for name, child in self.children.items():
            # add the global context so GraphAttributes (nodes/edges) are shared
            child._glob: _GlobalRestoreContext = self._glob
            try:
                child_states.append((name, type(child), child.__getstate__()))
            finally:
                del child._glob
        return ctx

    def _set_restore_context(self, ctx: _RestoreContext):
        # first, unpersist components and children graphs so we have that data
        # to build the graph on
        components: List[DocumentGraphComponent] = []
        children: Dict[str, DocumentGraph] = {}
        for cls, state in ctx.state['components']:
            inst = cls.__new__(cls)
            state['glob'] = self._glob
            inst.__setstate__(state)
            assert not hasattr(inst, '_glob')
            components.append(inst)
        for name, cls, state in ctx.state['children']:
            inst = cls.__new__(cls)
            state['glob'] = self._glob
            inst.__setstate__(state)
            children[name] = inst
            assert not hasattr(inst, '_glob')
        self.components = tuple(components)
        self.children = children

        # set this graph's data based on the components unpersisted above
        self._build()

        # add this graph's indexes created in _build() so we can create and
        # connect the terminal nodes (only data not "skipped") in
        # _get_restore_context
        gattr: str = self.GRAPH_ATTRIB_NAME
        ctx.graph = self.graph
        gn2v: Dict[int, int] = ctx.gn2v
        v: Vertex
        for v in self.graph.vs:
            ge: GraphNode = v[gattr]
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'add doc graph: {ge.id} -> {ge}')
            gn2v[ge.id] = v.index
        super()._set_restore_context(ctx)

    def __getitem__(self, name: str) -> DocumentGraphComponent:
        return self.components_by_name[name]

    def __str__(self) -> str:
        return self.name


# keep the dataclass semantics, but allow for a setter
DocumentGraph.graph_attrib_context = DocumentGraph._graph_attrib_context


@dataclass
class DocumentGraphDecorator(ABC):
    """A strategy to create a graph from a document structure.

    """
    @abstractmethod
    def decorate(self, component: DocumentGraphComponent):
        """Creates the graph from a :class:`.DocumentNode` root node.

        :param component: the graph to populate from the decorateing process

        """
        pass


@dataclass
class DocumentGraphFactory(ABC):
    """Creates a document graph.  After the document portion of the graph is
    created, the igraph is built and merged using a
    :class:`.DocumentGraphDecorator`.  This igraph has the corresponding
    vertexes and edges associated with the document graph, which includes AMR
    Penman graph and feature document artifacts.

    """
    config_factory: ConfigFactory = field()
    """Used to create a :class:`.DocumentGraphDecorator`."""

    graph_decorators: Tuple[DocumentGraphDecorator, ...] = field()
    """The name of the section that defines a :class:`.DocumentGraphDecorator`
    instance.

    """
    doc_graph_section_name: str = field()
    """The name of a section in the configuration that defines new instances of
    :class:`.DocumentGraph`.

    """
    graph_attrib_context: GraphAttributeContext = field()
    """The context given to all nodees and edges of the graph."""

    @abstractmethod
    def _create(self, root: AmrFeatureDocument) -> \
            Tuple[DocumentGraphComponent, ...]:
        """Create a document node graph from an AMR feature document and return
        the root.

        """
        pass

    def create(self, root: AmrFeatureDocument) -> DocumentGraph:
        """Create a document graph and return it starting from the root note.
        See class docs.

        :param root: the feature document from which to create the graph

        """
        assert isinstance(root, AmrFeatureDocument)
        # reset the graph attribute counter to make duplicate graph's consistent
        # and debugging easier, otherwise IDs will increement for each new graph
        self.graph_attrib_context.reset_attrib_id()
        components: Tuple[DocumentGraphComponent, ...] = self._create(root)
        graph: Graph = GraphComponent.graph_instance()
        decorator: DocumentGraphDecorator
        for decorator in self.graph_decorators:
            component: DocumentGraphComponent
            for component in components:
                with time(f'decorated {component.name}'):
                    decorator.decorate(component)
        doc_graph: DocumentGraph = self.config_factory.new_instance(
            self.doc_graph_section_name,
            graph=graph,
            doc=root,
            components=components)
        component: DocumentGraphComponent
        for component in components:
            component.description = doc_graph.name
        # evict to force recreate in the application context on next call
        self.config_factory.clear_instance(self.doc_graph_section_name)
        return doc_graph

    def __call__(self, root: AmrFeatureDocument) -> DocumentGraph:
        """See :meth:`create`."""
        return self.create(root)
