"""Document graph controller implementations.

"""
from __future__ import annotations
__author__ = 'Paul Landes'
from typing import (
    Tuple, Dict, Any, Sequence, List, Set, Type, Optional, Iterable
)
from dataclasses import dataclass, field
import logging
from igraph import Graph, Vertex, Edge
from igraph.cut import Flow as Maxflow
from zensols.util import time
from zensols.amr import AmrFeatureDocument
from .render.base import GraphRenderer
from . import (
    ComponentAlignmentError, GraphNode, ConceptGraphNode, GraphEdge,
    ComponentAlignmentGraphEdge, RoleGraphEdge, SentenceGraphEdge,
    DocumentGraph, GraphComponent, DocumentGraphComponent,
    GraphAlignmentConstructor, DocumentGraphController,
    FlowDocumentGraph, EdgeFlow, Reentrancy, ReentrancySet,
)

logger = logging.getLogger(__name__)


@dataclass
class ConstructDocumentGraphController(DocumentGraphController):
    """Constructs the graph that will later be used for the min cut/max flow
    algorithm (see :class:`.MaxflowDocumentGraphController`).  After its
    :meth:`invoke` method is called, :obj:`build_graph` is available, which is
    the constructed graph provided by :obj:`constructor`.

    """
    build_graph_name: str = field()
    """The name given to newly instances of :class:`.DocumentGraph`."""

    constructor: GraphAlignmentConstructor = field()
    """The constructor used to get the source and sink nodes."""

    renderer: GraphRenderer = field()
    """Visually render the graph in to a human understandable presentation."""

    def _set_flow_color_style(self, doc_graph: DocumentGraph,
                              weighted_edge: bool = True,
                              partition_nodes: bool = True):
        """Change the graph style so that edge flow gradients are used and
        udpate the concept color per style

        """
        vst: Dict[str, Any] = self.renderer.visual_style.asdict()
        if weighted_edge:
            vst['weighted_edge_flow'] = True
        if partition_nodes and 'concept_color' in vst:
            cc: Dict[str, str] = vst['concept_color']
            comp: DocumentGraphComponent
            for comp in doc_graph.components:
                cc[comp.name] = cc['component'][comp.name]

    def _build_graph(self, nascent_graph: DocumentGraph) -> DocumentGraph:
        """A (cached) graph that is built from the constructor.  This is
        constructed based on the ``doc_graph`` passed to :meth:`invoke`, which
        is then set :obj:`.DocumentGraph.original_graph`.

        """
        build_graph: DocumentGraph
        rev_edges: bool = self.constructor.requires_reversed_edges()
        with time(self._fmt('clone build graph')):
            build_graph = nascent_graph.clone(
                cls=FlowDocumentGraph,
                reverse_edges=rev_edges,
                # copy as shallow to create a connected bipartite graph with the
                # components sharing the same igraph instance
                deep=False)
        build_graph.name = self.build_graph_name
        self.constructor.doc_graph = build_graph
        with time(self._fmt('constructed build graph')):
            self.constructor.build()
        if logger.isEnabledFor(logging.INFO):
            logger.info(self._fmt(
                f'build graph: {build_graph}, ' +
                f'reverse edges: {rev_edges}'))
        nascent_graph.add_child(build_graph)
        return build_graph

    def _invoke(self, doc_graph: DocumentGraph) -> int:
        def map_cap(e: Edge) -> float:
            ge: GraphEdge = GraphComponent.to_edge(e)
            return ge.capacity

        with time(self._fmt('constructed flow network graph')):
            build_graph: DocumentGraph = self._build_graph(doc_graph)
            self._set_flow_color_style(build_graph)
        return len(doc_graph)


@dataclass
class MaxflowDocumentGraphController(DocumentGraphController):
    """Executes the maxflow/min cut algorithm on a document graph.

    """
    constructor: GraphAlignmentConstructor = field()
    """The constructor used to get the source and sink nodes."""

    def __post_init__(self):
        super().__post_init__()
        self.flows: Dict[int, float] = {}

    def _set_flows(self, maxflow: Maxflow, build_graph: DocumentGraph) -> int:
        g: Graph = build_graph.graph
        es: Dict[Edge, GraphEdge] = build_graph.es
        updates: int = 0
        eix: int
        flow_val: float
        for eix, flow_val in enumerate(maxflow.flow):
            e: Edge = g.es[eix]
            eg: GraphEdge = es[e]
            if isinstance(eg, RoleGraphEdge):
                prev_flow: Optional[float] = self.flows.get(eg.id)
                if prev_flow is not None:
                    diff: float = abs(prev_flow - flow_val)
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f'flow diff ({eg.id}):{prev_flow:.3f}->' +
                                     f'{flow_val:.3f} {diff:.3f}')
                    if diff > 0:
                        updates += 1
                else:
                    updates += 1
                # keep previous flows for diffing (update counts)
                self.flows[eg.id] = flow_val
            eg.flow = flow_val
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'add flow {maxflow} to {eg.id}, ' +
                             f'cap={eg.capacity}, flow={flow_val}')
        return updates

    def _invoke(self, doc_graph: DocumentGraph) -> int:
        def map_cap(e: Edge) -> float:
            ge: GraphEdge = GraphComponent.to_edge(e)
            return ge.capacity

        s: Vertex = self.constructor.source_flow_node
        t: Vertex = self.constructor.sink_flow_node
        g: Graph = doc_graph.graph
        caps = list(map(map_cap, g.es))
        logger.debug(f'running max flow on {s} -> {t}')
        maxflow: Maxflow
        with time(self._fmt(f'maxflow algorithm on {id(doc_graph)}')):
            maxflow = g.maxflow(s.index, t.index, caps)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('maxflow:')
            logger.debug(f'value: {maxflow.value}')
            logger.debug(f'flow: {maxflow.flow}')
            logger.debug(f'cut: {maxflow.cut}')
            logger.debug(f'partition: {maxflow.partition}')
        with time(self._fmt('set max flows')):
            return self._set_flows(maxflow, doc_graph)

    def reset(self):
        """Clears the cached :obj:`build_graph` instance."""
        super().reset()
        self.flows.clear()


@dataclass
class NormFlowDocumentGraphController(DocumentGraphController):
    """Normalizes flow on edges as the flow going through the edge and the total
    number of descendants.  Descendants are counted as the edge's source node
    and all children/descendants of that node.

    This is done recursively to calculate flow per node.  For each call
    recursive iteration, it computes the flow per node of the parent edge(s)
    from the perspective of the nascent graph, (root at top with arrows pointed
    to children underneath).  However, the graph this operates on are the
    reverese flow max flow graphs (flow diretion is taken care of adjacency list
    computed in :class:`.GraphComponent`.

    Since an AMR node can have multiple parents, we keep track of descendants as
    a set rather than a count to avoid duplicate counts when nodes have more
    than one parent.  Otherwise, in multiple parent case, duplicates would be
    counted when the path later converges closer to the root.

    """
    component_names: Set[str] = field()
    """The name of the components to minimize."""

    constructor: GraphAlignmentConstructor = field()
    """The instance used to construct the graph passed in the :meth:`invoke`
    method.

    """
    normalize_mode: str = field(default='fpn')
    """How to normalize nodes (if at all), which is one of:

      * ``fpn``: leaves flow values as they were after the initial flow per node
                 calculation

      * ``norm``: normalize so all values add to one

      * ``vis``: same as ``norm`` but add a ``vis_flow`` attribute to the edges
                 so the original flow is displayed and visualized as the flow
                 color

    """
    def _calc_neigh_flow(self, par: int, neigh: int, neighs: List[int],
                         neigh_desc: Set[int]):
        """Compute the neighbor ``neigh`` flow of parent ``par`` that has (of
        ``par``) neighbors ``neighs`` with descendants ``neigh_desc``.  This is
        then set on the neighbor's edge.

        The neighbors (``neigh`` and ``neighs``) are the children in the nascent
        graph or the parents in the reverse flow graph.

        """
        comp: DocumentGraphComponent = self._comp
        n_descendants: int = len(neigh_desc)
        eid: int = comp.graph.get_eid(neigh, par)
        ge: GraphEdge = comp.edge_by_id(eid)
        flow: float = ge.flow
        fpn: float = flow / n_descendants
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'set flow {par} -> {neigh}: desc={n_descendants}, ' +
                         f'n=, e={ge}, f={ge.flow:.4f}->{fpn:.4f}')
        ge.flow = fpn

    def _calc_flow_per_node(self, par: int, visited: Set[int]) -> \
            Tuple[Set[int], float]:
        """See class docs.

        :param par: the parent node from the perspective of the maxflow graph

        :param visited: the set of nodes already visited in this

        :return: all descendants of ``par`` and flow per node as a tuple

        """
        neighs: List[int] = self._alist[par]
        desc: Set[int] = {par}
        tot_fpn: float = 0
        par_visited: bool = par in visited
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'{par} -> {neighs}, v={par_visited}, ' +
                         f'graph reversed: {self._comp._edges_reversed}')
        # protect against cycles (i.e. proxy report id=20080125_0121); this
        # condition also satisfies for nodes with more than one parent in the
        # nascent graph (benign)
        if par_visited:
            doc: AmrFeatureDocument = self._comp.root_node.root
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('cycle or node with multiple parent ' +
                             f'detected in {par} in {doc}')
        else:
            visited.add(par)
            neigh: int
            # descendants at this level are the immediate children of this node
            for neigh in neighs:
                neigh_desc, tfpn = self._calc_flow_per_node(neigh, visited)
                self._calc_neigh_flow(par, neigh, neighs, neigh_desc)
                desc.update(neigh_desc)
                tot_fpn += tfpn
        return desc, tot_fpn

    def _norm_flow_node(self, tot_fpn: float) -> float:
        comp: DocumentGraphComponent = self._comp
        norm_mode: str = self.normalize_mode
        tot_flow: float = 0
        ge: GraphEdge
        for ge in comp.es.values():
            norm_flow: float = ge.flow / tot_fpn
            if norm_mode == 'norm' or norm_mode == 'vis':
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f'norm set flow {ge.flow} / {tot_fpn} = {norm_flow}')
                ge.flow = norm_flow
                if norm_mode == 'vis':
                    ge.vis_flow = norm_flow * tot_fpn
            else:
                raise ComponentAlignmentError(
                    f'Unknown normalization mode: {norm_mode}')
            tot_flow += norm_flow
        return tot_flow

    def _normalize_flow(self, comp: DocumentGraphComponent) -> int:
        """Normalize the flow so that it sums to one."""
        # add sink edge from the component root to the adjacency list to its
        # edge is also computed
        sink: Vertex = self.constructor.sink_flow_node
        self._alist: List[List[int]] = list(comp.adjacency_list)
        self._alist[sink.index] = (comp.root.index,)
        self._comp = comp
        try:
            desc, tfpn = self._calc_flow_per_node(sink.index, set())
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'flow/node {tfpn:.3f} on {len(desc)} descendants')
            if self.normalize_mode == 'fpn':
                pass
            elif self.normalize_mode in {'portion', 'norm', 'vis'}:
                one = self._norm_flow_node(tfpn)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f'total norm flow: {one}')
            else:
                raise ComponentAlignmentError(
                    f'No such normalization mode: {self.normalize_mode}')
            return len(desc)
        finally:
            del self._alist
            del self._comp

    def _invoke(self, doc_graph: DocumentGraph) -> int:
        updates: int = 0
        comp: DocumentGraphComponent
        for comp in doc_graph.components:
            if comp.name in self.component_names:
                updates += self._normalize_flow(comp)
        return updates


@dataclass
class FlowSetDocumentGraphController(DocumentGraphController):
    """Set a static flow on components based on name and edges based on class.

    """
    component_names: Set[str] = field(default_factory=set)
    """The components on which to set the flow."""

    match_edge_classes: Set[Type[GraphEdge]] = field(default_factory=set)
    """The edge classes (i.e. :class:`.TerminalGraphEdge`) to set the flow.

    """
    flow: float = field(default=0)
    """The flow to set."""

    def _set_component_flow(self, comp: DocumentGraphComponent) -> int:
        ge: GraphEdge
        for ge in comp.es.values():
            ge.flow = self.flow
        return len(comp.es)

    def _is_match_edge(self, ge: GraphEdge) -> bool:
        match_class: Type
        for match_class in self.match_edge_classes:
            if issubclass(type(ge), match_class):
                return True
        return False

    def _invoke(self, doc_graph: DocumentGraph) -> int:
        updates: int = 0
        # set any component flow
        comp: DocumentGraphComponent
        for comp in doc_graph.components:
            if comp.name in self.component_names:
                updates += self._set_component_flow(comp)
        ge: GraphEdge
        for e in doc_graph.graph.es:
            ge: GraphEdge = GraphComponent.to_edge(e)
            if self._is_match_edge(ge):
                ge.flow = self.flow
                updates += 1
        return updates


@dataclass
class FlowDiscountDocumentGraphController(DocumentGraphController):
    """Decrease/constrict the capacities by making the sum of the incoming flows
    from the bipartitie edges the value of :obj:`discount_sum`.  The capacities
    are only updated if the sum of the incoming bipartitie edges have a flow
    greater than :obj:`discount_sum`.

    """
    discount_sum: float = field()
    """The capacity sum will be this value (see class docs)."""

    component_names: Set[str] = field(default_factory=set)
    """The name of the components to discount."""

    def _constrict(self, gn: GraphNode, v: Vertex) -> int:
        comp: DocumentGraphComponent = self._comp
        neighs: Sequence[int] = comp.graph.incident(v, mode='in')
        neighs: GraphEdge = map(comp.edge_by_id, neighs)
        neighs: GraphEdge = tuple(filter(
            lambda ge: isinstance(ge, ComponentAlignmentGraphEdge), neighs))
        flow_sum: float = sum(map(lambda ge: ge.flow, neighs))
        squeeze: float = self.discount_sum / flow_sum
        ge: GraphEdge
        for ge in neighs:
            ge.capacity = ge.flow * squeeze
        return len(neighs)

    def _invoke_component(self, par: int) -> int:
        updates: int = 0
        comp: DocumentGraphComponent = self._comp
        neighs: List[int] = self._alist[par]
        for neigh in neighs:
            updates += self._invoke_component(neigh)
        edge_child: int
        for edge_child in comp.graph.incident(par, mode='in'):
            ge: GraphEdge = comp.edge_by_id(edge_child)
            if isinstance(ge, RoleGraphEdge):
                overflow: float = max(0, ge.flow - self.discount_sum)
                if overflow > 0:
                    e: Edge = comp.edge_ref_by_id(edge_child)
                    vid: int = e.source if e.target == par else e.target
                    gn: GraphNode = comp.node_by_id(vid)
                    updates += self._constrict(gn, vid)
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f'{par} e={ge}, n={gn}: {ge.flow}')
        return updates

    def _invoke(self, doc_graph: DocumentGraph) -> int:
        updates: int = 0
        comp: DocumentGraphComponent
        for comp in doc_graph.components:
            if comp.name in self.component_names:
                self._comp = comp
                self._alist: List[List[int]] = comp.adjacency_list
                try:
                    updates += self._invoke_component(comp.root.index)
                finally:
                    del self._comp
                    del self._alist
        return updates


@dataclass
class FixReentrancyDocumentGraphController(DocumentGraphController):
    """Fix reentrancies by splitting the flow of the last calculated maxflow as
    the capacity of the outgoing edges in the reversed graph.  This fixes the
    issue edges getting flow starved, then later eliminated in the graph
    reduction steps.

    Subsequently, the maxflow algorithm is rerun if we have at least one
    reentrancy after reallocating the capacit(ies).

    """
    component_name: str = field()
    """The name of the components to restore."""

    maxflow_controller: MaxflowDocumentGraphController = field()
    """The maxflow component used to recalculate the maxflow ."""

    only_report: bool = field()
    """Whether to only report reentrancies rather than fix them."""

    def _iterate(self, comp: DocumentGraphComponent, v: Vertex, gn: GraphNode,
                 reentrancies: List[Reentrancy]) -> int:
        updates: int = 0
        neighs: Tuple[GraphEdge, ...] = tuple(
            filter(lambda ge: isinstance(ge, RoleGraphEdge), map(
                comp.edge_by_id, comp.graph.incident(v, mode='out'))))
        # reentrancies have multiple outgoing edges in the reversed graph
        if len(neighs) > 1:
            # rentrancies are always concept nodes with role edges that have
            # multiple in edges in the reversed graph
            assert isinstance(gn, ConceptGraphNode)
            reentrancy = Reentrancy(
                concept_node=gn,
                concept_node_vertex=v.index,
                edge_flows=tuple(map(EdgeFlow, neighs)))
            # at least report it
            reentrancies.append(reentrancy)
            # we care only about at least one that has no flow
            if not self.only_report and reentrancy.has_zero_flow:
                total_flow: float = reentrancy.total_flow
                if total_flow == 0:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug('skipping 0 cap on {len(neighs)} edges')
                else:
                    new_capacity: float = total_flow / len(neighs)
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f'setting capacity {new_capacity} ' +
                                     f'on {len(neighs)} edges')
                    neigh: GraphEdge
                    for neigh in neighs:
                        neigh.capacity = new_capacity
                    updates += len(neighs)
        return updates

    def _iterate_comp(self, comp: DocumentGraphComponent) -> \
            Tuple[int, Tuple[Reentrancy, ...]]:
        reentrancies: List[Reentrancy] = []
        updates: int = 0
        v: Vertex
        gn: GraphNode
        for v, gn in comp.vs.items():
            updates += self._iterate(comp, v, gn, reentrancies)
        return updates, tuple(reentrancies)

    def _invoke(self, doc_graph: DocumentGraph) -> int:
        comp: DocumentGraphComponent
        updates: int = 0
        for comp in doc_graph.components:
            comp_updates: int
            reentrancies: Tuple[Reentrancy, ...]
            comp_updates, reentrancies = self._iterate_comp(comp)
            updates += comp_updates
            comp.reentrancy_set = ReentrancySet(reentrancies)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'setting reentrancy sets on {doc_graph}.{comp}')
        if updates > 0:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('rerunning maxflow after finding ' +
                             f'{len(reentrancies)} reentrancies')
            return self.maxflow_controller.invoke(doc_graph)
        else:
            return 0


@dataclass
class AlignmentCapacitySetDocumentGraphController(DocumentGraphController):
    """Set the capacity on edges if the criteria matches :obj:`min_flow`,
    :obj:`component_names` and :obj:`match_edge_classes`.

    """
    min_capacity: float = field()
    """The minimum capacity to clamping the capacity of a target
    :class:`.GraphEdge` to :obj:`capacity`.

    """
    capacity: float = field()
    """The capacity to set."""

    def _invoke(self, doc_graph: DocumentGraph) -> int:
        updates: int = 0
        ge: GraphEdge
        for ge in doc_graph.es.values():
            if isinstance(ge, ComponentAlignmentGraphEdge):
                if ge.capacity <= self.min_capacity:
                    ge.capacity = self.capacity
                    ge.flow = self.capacity
                    updates += 1
        return updates


@dataclass
class RoleCapacitySetDocumentGraphController(DocumentGraphController):
    """This finds low flow role edges and sets (zeros out) all the capacities of
    all the connected edge alignments recursively for all descendants.  We
    "slough off" entire subtrees (sometimes entire sentences or document nodes)
    for low flow ancestors.

    """
    min_flow: float = field()
    """The minimum amount of flow to trigger setting the capacity of a target
    :class:`.GraphEdge` capacity to :obj:`capacity`.

    """
    capacity: float = field()
    """The capacity (and flow) to set."""

    component_names: Set[str] = field()
    """The name of the components to minimize."""

    def _child_flow(self, comp: DocumentGraphComponent,
                    e: Edge, ge: GraphEdge) -> float:
        """Return the sum of the flow of all children edges of the source
        (parent) node.  If the passed edge's source (parent) node has multiple
        outgoing edges, this might have a different flow than the single passed
        edge.

        """
        child_flow: float = ge.flow
        # get all children edges of the parent of the passed edge; assume
        # reverse flow
        eids: Sequence[int] = comp.graph.incident(e.source, mode='out')
        # this will include the passed edge, but might have others
        if len(eids) > 1:
            child_flow = sum(map(lambda i: comp.edge_by_id(i).flow, eids))
        return child_flow

    def _set_flow_comp(self, comp: DocumentGraphComponent) -> int:
        """Normalize the flow so that it sums to one."""
        def filter_edge(eid: int, ge: GraphEdge):
            return eid not in visited and \
                isinstance(ge, ComponentAlignmentGraphEdge)

        updates: int = 0
        min_flow: float = self.min_flow
        cap: float = self.capacity
        visited: Set[int] = set()
        e: Edge
        ge: GraphEdge
        for e, ge in comp.es.items():
            if isinstance(ge, (RoleGraphEdge, SentenceGraphEdge)) and \
               ge.flow <= min_flow and \
               self._child_flow(comp, e, ge) < min_flow:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f'{ge}({e.index}): flow={ge.flow}: {e.source}')
                # assume reverse flow
                for v in comp.graph.dfs(e.source, mode='in')[0]:
                    n: GraphNode = comp.node_by_id(v)
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f' {n}({v})')
                    eids: Sequence[int] = comp.graph.incident(v, mode='in')
                    es: Iterable[GraphEdge] = map(
                        lambda eid: (eid, comp.edge_by_id(eid)), eids)
                    es = filter(lambda x: filter_edge(*x), es)
                    for eid, sge in es:
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(f'  set cap {n}({v}): {eid}({sge})')
                        sge.capacity = cap
                        updates += 1
                    visited.update(eids)
        return updates

    def _invoke(self, doc_graph: DocumentGraph) -> int:
        updates: int = 0
        comp: DocumentGraphComponent
        for comp in doc_graph.components:
            if comp.name in self.component_names:
                updates += self._set_flow_comp(comp)
        return updates


@dataclass
class SnapshotDocumentGraphController(DocumentGraphController):
    """Record flows, then later restore.  If :obj:`snapshot_source` is not
    ``None``, then this instance restores from it.  Otherwise it records.

    """
    component_names: Set[str] = field()
    """The name of the components on which to record or restore flows."""

    snapshot_source: SnapshotDocumentGraphController = field()
    """The source instance that contains the data from which to restore."""

    def __post_init__(self):
        super().__post_init__()
        self._flows: Dict[int, Tuple[float, float]] = {}

    def _capture(self, comp: DocumentGraphComponent):
        if logger.isEnabledFor(logging.INFO):
            logger.info(self._fmt(f'saving: {comp}'))
        ge: GraphEdge
        for ge in comp.es.values():
            if isinstance(ge, (RoleGraphEdge, SentenceGraphEdge)):
                self._flows[ge.id] = (ge.capacity, ge.flow)

    def _restore(self, comp: DocumentGraphComponent):
        flows: Dict[int, float]
        if self.snapshot_source is None:
            flows = self._flows
        else:
            flows = self.snapshot_source._flows
        if logger.isEnabledFor(logging.INFO):
            logger.info(self._fmt(f'restoring: {comp}'))
        ge: GraphEdge
        for ge in comp.es.values():
            if isinstance(ge, (RoleGraphEdge, SentenceGraphEdge)):
                ge.capacity, ge.flow = flows[ge.id]

    def _invoke(self, doc_graph: DocumentGraph) -> int:
        comp: DocumentGraphComponent
        for comp in doc_graph.components:
            if comp.name in self.component_names:
                if self.snapshot_source is None:
                    self._capture(comp)
                else:
                    self._restore(comp)
        return 0

    def reset(self):
        """Clears the cached :obj:`build_graph` instance."""
        super().reset()
        self._flows.clear()


@dataclass
class RemoveAlignsDocumentGraphController(DocumentGraphController):
    """Removes graph component alignment for low capacity links.

    """
    min_capacity: float = field()
    """The graph component alignment edges are removed if their capacities are
    at or below this value.

    """
    def _invoke(self, doc_graph: DocumentGraph) -> int:
        to_remove: List[GraphEdge] = []
        ge: GraphEdge
        for ge in doc_graph.es.values():
            if isinstance(ge, ComponentAlignmentGraphEdge):
                if ge.capacity <= self.min_capacity:
                    to_remove.append(ge)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'deleting: {to_remove}')
        doc_graph.delete_edges(to_remove)
        return len(to_remove)
