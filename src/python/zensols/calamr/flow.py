"""Provides container classes and computes statistics for graph alignments.

"""
from __future__ import annotations
__author__ = 'Paul Landes'
from typing import (
    List, Tuple, Sequence, Set, Dict, Iterable,
    Union, Optional, Type, Any, ClassVar
)
from dataclasses import dataclass, field
import sys
import logging
from frozendict import frozendict
import collections
import textwrap as tw
from io import TextIOBase
from pathlib import Path
import numpy as np
import pandas as pd
from igraph import Vertex, Edge
from zensols.persist import PersistableContainer, persisted
from zensols.config import Dictable
from zensols.amr import AmrFeatureDocument
from zensols.datdesc import DataFrameDescriber, DataDescriber
from . import (
    ComponentAlignmentError, ComponentAlignmentFailure, GraphAttributeContext,
    RoleGraphEdge, ReentrancySet, GraphNode, GraphEdge, DocumentGraphEdge,
    SentenceGraphNode, ConceptGraphNode, AttributeGraphNode, DocumentGraphNode,
    TerminalGraphNode, DocumentGraphComponent, DocumentGraph, SentenceGraphEdge,
    SentenceGraphAttribute,
    ComponentAlignmentGraphEdge, ComponentCorefAlignmentGraphEdge,
)
from .flowmeta import _DATA_DESC_META
from .render.base import RenderContext, GraphRenderer, rendergroup

logger = logging.getLogger(__name__)


@dataclass
class Flow(Dictable):
    """A triple of a source node, target node and connecting edge from the
    graph.  The connecting edge has a flow value associated with it.

    """
    source: GraphNode = field()
    """The starting node in the DAG."""

    target: GraphNode = field()
    """The ending node (arrow head) in the DAG."""

    edge: GraphEdge = field()
    """The edge that connects :obj:`source` and :obj:`target`."""

    @property
    def edge_type(self) -> str:
        """Whether the edge is an AMR ``role`` or ``align``ment."""
        if isinstance(self.edge, ComponentAlignmentGraphEdge):
            return 'align'
        else:
            return 'role'

    @property
    def to_row(self) -> List[Any]:
        """Create a row from the data in this flow used in
        :meth:`.FlowDocumentGraphComponent.create_df`.

        """
        def tok_str(node: GraphNode):
            if isinstance(node, SentenceGraphAttribute):
                tstr: str = '|'.join(map(lambda t: t.norm, node.tokens))
                return tstr if len(tstr) > 0 else None

        src_toks: str = tok_str(self.source)
        targ_toks: str = tok_str(self.target)
        row: List[Any] = [self.source.description, self.target.description,
                          src_toks, targ_toks]
        rel_id: int = None
        is_bipartite: bool = None
        if isinstance(self.edge, ComponentCorefAlignmentGraphEdge):
            rel_id = self.edge.relation.seq_id
            is_bipartite = self.edge.is_bipartite
        row.extend((self.source.id, self.target.id,
                    self.source.attrib_type, self.target.attrib_type,
                    self.edge_type, rel_id, is_bipartite, self.edge.flow))
        return row

    def __str__(self) -> str:
        return (f'{self.source}[{self.source.id}] -> ' +
                f'{self.target}[{self.target.id}]: ' +
                f'{self.edge.description}')


class FlowGraphComponentResult(Dictable):
    """A container class for the flow data from a :class:`.DocumentComponent`
    flow instance (aka reverse flow graph).  This includes the data as
    dictionaries of statistics, :class:`pandas.DataFrame` and
    :class:`~zensols.datdesc.desc.DataDescriber` instances.

    """
    _NODE_TYPES: Tuple[Type[GraphNode], ...] = (
        ConceptGraphNode, AttributeGraphNode,
        SentenceGraphNode, DocumentGraphNode)
    """Base node types used for statistics."""

    def __init__(self, component: FlowDocumentGraphComponent):
        self._component = component

    @property
    @persisted('_node_counts', transient=True)
    def node_counts(self) -> Dict[Type[GraphNode], int]:
        """The number of nodes by type."""
        cnts: Dict[Type[GraphNode], int] = {}
        node_type: Type[GraphNode]
        for node_type in self._NODE_TYPES:
            cnts[node_type] = 0
        node: GraphNode
        for node in self._component.vs.values():
            node_type: Type[GraphNode] = type(node)
            cnt: int = cnts.get(node_type)
            if cnt is None:
                node_type = next(iter(filter(
                    lambda t: isinstance(node, t), self._NODE_TYPES)))
                cnt = cnts[node_type]
            cnts[node_type] = cnt + 1
        return frozendict(cnts)

    @property
    @persisted('_edge_counts', transient=True)
    def edge_counts(self) -> Dict[Type[GraphEdge], int]:
        """The number of edges by type."""
        cnts: Dict[Type[GraphEdge], int] = {}
        # there are no terminal or component alignment edges since the component
        # was taken from the initial graph instance outside the scope of the
        # bipartite graph
        edge_type: Type[GraphEdge]
        for edge_type in (RoleGraphEdge, SentenceGraphEdge, DocumentGraphEdge):
            cnts[edge_type] = 0
        edge: GraphEdge
        for edge in self._component.es.values():
            edge_type = type(edge)
            cnt: int = cnts[edge_type]
            cnts[edge_type] = cnt + 1
        return frozendict(cnts)

    @property
    def n_alignable_nodes(self) -> int:
        """The number of nodes in the component that can take alignment edges.
        Whether those nodes in the count have edges does not effect the result.

        """
        node_counts: Dict[Type[GraphNode], int] = self.node_counts
        return node_counts[ConceptGraphNode] + \
            node_counts[AttributeGraphNode] + \
            node_counts[SentenceGraphNode]

    @property
    @persisted('_df', transient=True)
    def df(self) -> pd.DataFrame:
        """The data in :obj:`flows` and :obj:`root` as a dataframe.  Note the
        terms *source* and *target* refer to the nodes at the ends of the
        directed edge in a **reversed graph**.

          * ``s_descr``: source node descriptions such as concept names,
                         attribute constants and sentence text

          * ``t_descr``: target node of ``s_descr``

          * ``s_toks``: any source node aligned tokens

          * ``t_toks``: any target node aligned tokens

          * ``s_attr``: source node attribute name give by
                        :obj:`.GraphAttribute.attrib_type`, such as ``doc``,
                        ``sentence``, ``concept``, ``attribute``

          * ``t_attr: target node of ``s_attr``

          * ``s_id``: source node :mod:`igraph` ID

          * ``t_id``: target node :mod:`igraph` ID

          * ``edge_type``: whether the edge is an AMR ``role`` or ``alignment``

          * ``rel_id``: the coreference relation ID or ``null`` if the edge is
                        not a corefernce

          * ``is_bipartite``: whether relation ``rel_id`` spans components or
                             ``null`` if the edge is not a coreference

          * ``flow``: the (normalized/flow per node) flow of the edge

          * ``reentrancy``: whether the edge participates an AMR reentrancy

          * ``align_flow``: the flow sum of the alignment edges for the
                            respective edge

          * ``align_count``: the count of incoming alignment edges to the target
                             node in the :class:`.FlowDocumentGraphComponent`
                             this instance holds

        """
        cols: List[str] = 's_descr t_descr s_toks t_toks'.split()
        cols.extend(('s_id t_id s_attr t_attr edge_type rel_id ' +
                     'is_bipartite flow').split())
        rows: List[Any] = [self._component.root_flow.to_row]
        flow: Flow
        for flow in self._component.flows:
            rows.append(flow.to_row)
        df: pd.DataFrame = pd.DataFrame(rows, columns=cols)
        verts: Set[int] = set(self._component.reentrancy_set.by_vertex.keys())
        df['reentrancy'] = df['s_id'].apply(lambda i: i in verts)
        # create alignment edge only dataframe
        adf = df[df['edge_type'] == 'align']
        # create a role only dataframe to zero flows so AMR role edge flow isn't
        # aggregated (summed) with alignment edge flow
        rdf = df[df['edge_type'] == 'role'].copy()
        rdf['flow'] = 0
        # create a dataframe with the counts of incoming alignment on component
        # edges
        join_col: str = 't_id'
        dfc = adf.groupby(join_col)[join_col].agg('count').\
            to_frame().rename(columns={join_col: 'align_count'})
        # aggregate across incoming alignment flow (excludes role edge flow)
        agg_flow = pd.concat((rdf, adf)).groupby(join_col)['flow'].agg('sum').\
            to_frame().rename(columns={'flow': 'align_flow'})
        # create dataframe with aggregate alignment flow and incoming alignment
        # edge (to node) counts
        inflows = agg_flow.merge(dfc, how='left', on=join_col).fillna(0)
        # mixed NAs create floats, so convert back to int
        inflows['align_count'] = inflows['align_count'].astype(int)
        # merge it with the orignal dataframe
        return df.merge(inflows, on=join_col)

    def _get_data_desc_meta_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            data=tuple(map(lambda t: t[1], _DATA_DESC_META)),
            index=tuple(map(lambda t: t[0], _DATA_DESC_META)),
            columns=['description'])

    def create_data_frame_describer(self) -> DataFrameDescriber:
        """Like :meth:`create_align_df` but includes a human readable
        description of the data.

        """
        doc: AmrFeatureDocument = self._component.root_node.doc
        sent_norm: str = tw.shorten(doc.norm, width=60)
        return DataFrameDescriber(
            name='alignment flow edges',
            desc=f'the alignment flow and graph data for "{sent_norm}"',
            df=self.df,
            meta=self._get_data_desc_meta_df())

    @property
    @persisted('_connected_stats', transient=True)
    def connected_stats(self) -> Dict[str, Union[int, float]]:
        """The statistics on how well the two graphs are aligned by counting as:

          * ``alignable``: the number of nodes that are eligible for having an
                           alignment (i.e. sentence, concept, and attribute
                           notes)

          * ``aligned``: the number aligned nodes in the
                         :class:`.FlowDocumentGraphComponent` this instance
                         holds

          * ``aligned_portion``: the quotient of $aligned / alignable$, which is
                                 a number between $[0, 1]$ representing a score
                                 of how well the two graphs match

        """
        df: pd.DataFrame = self.df
        # no-sink dataframe will scew the score
        nsdf = df[df['t_attr'] != TerminalGraphNode.ATTRIB_TYPE]
        # get the edges that are aligned
        adf = nsdf[(nsdf['edge_type'] == 'align') & (nsdf['align_count'] > 0)]
        # get count of target/component nodes that have at least one alignment
        n_aligned: int = len(adf['t_id'].drop_duplicates())
        # get the count of nodes in this component
        n_alignable: int = self.n_alignable_nodes
        # create the portion of those nodes in the graph connected
        aligned_portion: float = n_aligned / n_alignable
        return frozendict({'aligned': n_aligned,
                           'alignable': n_alignable,
                           'aligned_portion': aligned_portion})

    @property
    @persisted('_stats', transient=True)
    def stats(self) -> Dict[str, Any]:
        """All statistics/scores available for this instances, which include:

          * ``root_flow``: the flow from the root node to the sink

          * ``connected``: :obj:`connected_stats`

        """
        return frozendict({
            'root_flow': self._component.root_flow_value,
            'connected': self.connected_stats,
            'counts': frozendict({
                'node': frozendict(dict(
                    map(lambda c: (c[0].ATTRIB_TYPE, c[1]),
                        self.node_counts.items()))),
                'edge': frozendict(dict(
                    map(lambda c: (c[0].ATTRIB_TYPE, c[1]),
                        self.edge_counts.items())))}),
            'reentrancies': self._component.reentrancy_set.stats['total'],
            'relations': len(self._component.relation_set)})

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        self._write_dict(self.stats, depth, writer)


@dataclass
class FlowDocumentGraphComponent(DocumentGraphComponent):
    """Contains all the flows of a :class:`.DocumentComponent`.

    """
    _PERSITABLE_TRANSIENT_ATTRIBUTES: ClassVar[Set[str]] = \
        DocumentGraphComponent._PERSITABLE_TRANSIENT_ATTRIBUTES | \
        {'_parent'}

    reentrancy_set: ReentrancySet = field(default=None)
    """Concept nodes with multiple parents."""

    def _create_flow(self, e: Edge, ge: GraphEdge) -> Flow:
        parent: DocumentGraph = self._parent
        s: GraphNode = parent.node_by_id(e.source)
        t: GraphNode = parent.node_by_id(e.target)
        return Flow(s, t, ge)

    def _create_flows(self) -> Iterable[Flow]:
        parent: DocumentGraph = self._parent
        parent_comp: DocumentGraphComponent = \
            parent.components_by_name[self.name]
        e: Edge
        ge: GraphEdge
        for e, ge in parent.es.items():
            vt: Vertex = parent.vertex_ref_by_id(e.target)
            if vt not in parent_comp.vs:
                continue
            if isinstance(ge, ComponentAlignmentGraphEdge) or \
               isinstance(ge, (SentenceGraphEdge, RoleGraphEdge)):
                yield self._create_flow(e, ge)

    @property
    @persisted('_flows', transient=True)
    def flows(self) -> Tuple[Flow, ...]:
        """The flows aggregated from the document components."""
        return tuple(self._create_flows())

    def _get_terminal_vertex(self, is_source: bool) -> Vertex:
        def filter_ga(v: Vertex) -> bool:
            gn: GraphNode = v[ga_attr]
            return isinstance(gn, TerminalGraphNode) and \
                gn.is_source == is_source

        ga_attr: str = self.GRAPH_ATTRIB_NAME
        verts: Tuple[Vertex, ...] = tuple(filter(
            filter_ga, self._parent.vs.keys()))
        if len(verts) != 1:
            raise ComponentAlignmentError(f'Not singleton: {verts}')
        return verts[0]

    @property
    @persisted('_root_flow', transient=True)
    def root_flow(self) -> Flow:
        """The root flow of the document component, which has the component's
        :class:`.DocumentGraphNode` as the source node and the sink as the
        target node.

        """
        parent: DocumentGraph = self._parent
        sink: Vertex = self._get_terminal_vertex(False)
        sink_neighs: Sequence[int] = self._parent.graph.incident(
            sink, mode='in')
        assert len(sink_neighs) == 1
        e: Edge = parent.edge_ref_by_id(sink_neighs[0])
        ge: GraphEdge = parent.edge_by_id(sink_neighs[0])
        root: Flow = self._create_flow(e, ge)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'sink ({sink.index}): {root}')
        return root

    @property
    def root_flow_value(self) -> float:
        """The flow from the root node to the sink in the reversed graph."""
        return self.root_flow.edge.flow

    @property
    @persisted('_result', transient=True)
    def result(self) -> FlowGraphComponentResult:
        """The flow results for this component."""
        return FlowGraphComponentResult(self)


@dataclass
class FlowDocumentGraph(DocumentGraph):
    """Contains all the flows of a :class:`.DocumentGraph` and has
    :class:`.FlowDocumentGraphComponent` as components.  Instances of this
    document graph have no children.

    """
    def _clone_components(self, reverse_edges: bool, deep: bool,
                          components: Tuple[DocumentGraphComponent, ...]) -> \
            Tuple[DocumentGraphComponent]:
        def clone_comp(comp: DocumentGraphComponent) -> DocumentGraphComponent:
            c = comp.clone(reverse_edges, deep, cls=FlowDocumentGraphComponent)
            c._parent = self
            return c

        return tuple(map(clone_comp, components))

    def __setstate__(self, state: Dict[str, Any]):
        super().__setstate__(state)
        c: FlowDocumentGraphComponent
        for c in self.components:
            c._parent = self


@dataclass
class _FlowGraphResultContext(object):
    """Contains in memory/interperter session data needed by
    :class:`.FlowGraphResult` when it is created or unpickled.

    """
    renderer: GraphRenderer = field()
    """Visually render the graph in to a human understandable presentation."""

    graph_attrib_context: GraphAttributeContext = field()
    """The context given to all nodees and edges of the graph."""


class FlowGraphResult(PersistableContainer, Dictable):
    """A container class for flow document results, which include the detailed
    data as dictionaries of statistics, :class:`pandas.DataFrame` and
    :class:`~zensols.datdesc.desc.DataDescriber` instances.  This is aggregated
    from :obj:`doc_graph` and the flow children's flow graph components.

    All graphs (from nascent to the reversed flow children graphs) have the
    final state of the actions of the :class:`.DocumentGraphController` as
    coordinated by the :class:`.GraphSequencer`.  Since the flows are copied
    from the reversed source graph to the root level (:obj:`doc_graph`) factory
    built nascent graph, all flows are the same.  However, the nascent graph
    will still be the disconnect source and summary graphs.

    """
    _DICTABLE_ATTRIBUTES: ClassVar[Set[str]] = {'stats'}
    _PERSITABLE_TRANSIENT_ATTRIBUTES: ClassVar[Set[str]] = {'_context'}

    def __init__(self, component_paths: Tuple[Tuple[str, str], ...],
                 context: _FlowGraphResultContext,
                 data: Union[DocumentGraph, ComponentAlignmentFailure]):
        """Initialize the flow results.

        :param data: the root nascent :class:`.DocumentGraphFactory` build graph
                     or an instance of :class:`.ComponentAlignmentFailure` if
                     the alignment failed

        :param component_paths: a set of paths that indicate which flow
                                components to use for the results in the form
                                ``(<child name>, <component name>)``

        """
        super().__init__()
        self._component_paths = component_paths
        self._data = data
        self._set_context(context)

    def _set_context(self, context: _FlowGraphResultContext):
        self._context = context
        if context is not None and not self.is_error:
            doc_graph: DocumentGraph = self._data
            doc_graph.graph_attrib_context = context.graph_attrib_context

    @property
    def failure(self) -> Optional[ComponentAlignmentFailure]:
        """What caused the alignment to fail, or ``None`` if it was a success.

        """
        if isinstance(self._data, ComponentAlignmentFailure):
            return self._data

    @property
    def is_error(self) -> bool:
        """Whether the graph resulted in an error."""
        return self.failure is not None

    @property
    def doc_graph(self) -> DocumentGraph:
        """The root nascent :class:`.DocumentGraphFactory` build graph.

        :throws ComponentAlignmentError: if this instanced resulted in an error

        :see: :obj:`is_error`

        """
        if self.is_error:
            self._data.rethrow()
        return self._data

    @property
    @persisted('__components', transient=True)
    def _components(self) -> FlowDocumentGraphComponent:
        """The flow components from the flow children document graphs.

        :throws ComponentAlignmentError: if this instanced resulted in an error

        :see: :obj:`is_error`

        """
        def map_comp_name(path: Tuple[str, str]) -> FlowDocumentGraphComponent:
            fdg: FlowDocumentGraph = self.doc_graph.children[path[0]]
            return fdg.components_by_name[path[1]]

        return tuple(map(map_comp_name, self._component_paths))

    @property
    @persisted('__components_by_name', transient=True)
    def _components_by_name(self) -> Dict[str, FlowDocumentGraphComponent]:
        """Get document graph components by name.

        :throws ComponentAlignmentError: if this instanced resulted in an error

        :see: :obj:`is_error`

        """
        return frozendict({c.name: c for c in self._components})

    @property
    @persisted('_df')
    def df(self) -> pd.DataFrame:
        """A concatenation of frames created with
        :meth:`.FlowDocumentGraphComponent.create_align_df` with the name of
        each component.

        :throws ComponentAlignmentError: if this instanced resulted in an error

        :see: :obj:`is_error`

        """
        dfs: List[pd.DataFrame] = []
        comp: FlowDocumentGraphComponent
        for comp in self._components:
            df: pd.DataFrame = comp.result.df
            df.insert(0, 'name', comp.name)
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True)

    def create_data_describer(self) -> DataDescriber:
        """Like :meth:`create_align_df` but includes a human readable
        description of the data.

        :throws ComponentAlignmentError: if this instanced resulted in an error

        :see: :obj:`is_error`

        """
        dfds: List[DataFrameDescriber] = []
        comp: FlowDocumentGraphComponent
        for comp in self._components:
            dfd: DataFrameDescriber = comp.result.create_data_frame_describer()
            dfd.name = comp.name
            dfds.append(dfd)
        return DataDescriber(
            describers=tuple(dfds),
            name='alignment flow edges')

    @property
    @persisted('_stats', transient=True)
    def stats(self) -> Dict[str, Any]:
        """The statistics with keys as component names and values taken from
        :obj:`.FlowDocumentGraphComponent.stats`.

        :throws ComponentAlignmentError: if this instanced resulted in an error

        :see: :obj:`is_error`

        """
        cstats: Dict[str, Dict[str, Any]] = collections.OrderedDict()
        astats: Dict[str, Dict[str, Any]] = collections.OrderedDict()
        comp: FlowDocumentGraphComponent
        alignable: List[int] = []
        aligned: List[int] = []
        aps: List[float] = []
        flows: List[float] = []
        for comp in self._components:
            stats: Dict[str, Any] = comp.result.stats
            con: Dict[str, Any] = stats['connected']
            alignable.append(con['alignable'])
            aligned.append(con['aligned'])
            aps.append(con['aligned_portion'])
            flows.append(stats['root_flow'])
            cstats[comp.name] = stats
        alignable: int = sum(alignable)
        aligned: int = sum(aligned)
        aps_sum: float = sum(map(lambda x: 0 if x == 0 else 1 / x, aps))
        fl: int = len(flows)
        nf: bool = (fl == 0)
        aph: float
        if nf:
            aph = np.nan
        elif aps_sum == 0:
            aph = 0
        else:
            aph = len(aps) / aps_sum
        astats['aligned_portion_hmean'] = aph
        astats['mean_flow'] = np.nan if nf else sum(flows) / fl
        astats['tot_alignable'] = np.nan if nf else alignable
        astats['tot_aligned'] = np.nan if nf else aligned
        astats['aligned_portion'] = np.nan if nf else aligned / alignable
        if nf:
            rs = ReentrancySet()
        else:
            sets: Tuple[ReentrancySet] = tuple(map(
                lambda c: c.reentrancy_set, self._components))
            rs = ReentrancySet.combine(sets)
        astats['reentrancies'] = rs.stats['total']
        if len(cstats) == 0 and nf:
            c = {k: np.nan for k in 'alignable aligned aligned_portion'.split()}
            comp = frozendict({
                'connected': frozendict(c),
                'counts': frozendict(
                    {'node':
                     frozendict(
                         dict(map(lambda n: (n.ATTRIB_TYPE, 0),
                                  (ConceptGraphNode, AttributeGraphNode,
                                   SentenceGraphNode, DocumentGraphNode)))),
                     'edge': frozendict(
                         dict(map(lambda n: (n.ATTRIB_TYPE, 0),
                                  (RoleGraphEdge, SentenceGraphEdge))))}),
                'root_flow': np.nan,
                'reentrancies': {}})
            cstats = {
                'source': comp,
                'summary': comp}
        bp_rels: int = -1
        bp_rels = len(self.doc_graph.bipartite_relation_set)
        return frozendict({
            'components': cstats,
            'agg': astats,
            'bipartite_relations': bp_rels})

    @property
    @persisted('_stats_df', transient=True)
    def stats_df(self) -> pd.DataFrame:
        """A Pandas dataframe version of :obj:`stats`."""
        stats: Dict[str, Any] = self.stats
        agg: Dict[str, Any] = stats['agg']
        aks: List[str] = ('aligned_portion_hmean mean_flow tot_alignable ' +
                          'tot_aligned aligned_portion reentrancies').split()
        cols: List[str] = 'component text'.split() + \
            list(map(lambda c: f'agg_{c}', aks)) + \
            'root_flow alignable aligned aligned_portion reentrancies'.split()
        agg_row = tuple(map(lambda a: agg[a], aks))
        rows: List[Tuple] = []
        name: str
        cstats: Dict[str, Any]
        for name, cstats in stats['components'].items():
            comp: FlowDocumentGraphComponent = self._components_by_name[name]
            doc: AmrFeatureDocument = comp.root_node.doc
            con: Dict[str, Any] = cstats['connected']
            rows.append(
                (name, doc.norm, *agg_row, cstats['root_flow'],
                 con['alignable'], con['aligned'], con['aligned_portion'],
                 cstats['reentrancies']))
        return pd.DataFrame(rows, columns=cols)

    def get_render_contexts(self, child_names: Iterable[str] = None,
                            include_nascent: bool = False) -> \
            List[RenderContext]:
        """Get contexts used to render the graphs with
        :class:`.render.base.rendergroup`.

        :param child_names: the name of the :obj:`.DocumentGraph.children` to
                            render, which defaults the the nascent grah and the
                            final bipartite graph rendered ("restore previous
                            flow on source")

        :param include_nascent: whether to include the nascent graphs

        """
        initial_child_name: str = self._component_paths[0][0]
        renderer_class: str = self._context.renderer.__class__.__name__
        ctxs: List[RenderContext] = []
        if include_nascent:
            ctxs.append(RenderContext(
                doc_graph=self.doc_graph,
                heading=f"Nascent {initial_child_name.replace('_', ' ')}"))
        if child_names is None:
            child_names = (initial_child_name,)
        child_name: str
        for child_name in child_names:
            fdg: FlowDocumentGraph = self.doc_graph.children[child_name]
            ctxs.append(RenderContext(
                doc_graph=fdg,
                heading=child_name.replace('_', ' ').capitalize()))
        # reverse the directory of the rendered graph to be consistent with the
        # graph building rendering; only applies to the graphvis rendering
        if renderer_class.startswith('GraphVis'):
            for ctx in ctxs:
                if ctx.doc_graph.name.startswith('reversed'):
                    ctx.visual_style = {'attributes': {'rankdir': 'BT'}}
        return ctxs

    def render(self, contexts: Tuple[RenderContext] = None,
               graph_id: str = 'graph', display: bool = True,
               directory: Path = None):
        """Render several graphs at a time, then optionally display them.

        :param contexts: the data to render, which defaults to the output of
                         :meth:`get_render_contexts`

        :param graph_id: a unique identifier prefixed to files generated if none
                         provided in the call method

        :param display: whether to display the files after generated

        :param directory: the directory to create the files in place of the
                          temporary directory; if provided the directory is not
                          removed after the graphs are rendered

        """
        renderer: GraphRenderer = self._context.renderer
        contexts = self.get_render_contexts() if contexts is None else contexts
        with rendergroup(renderer, graph_id=graph_id, display=display,
                         directory=directory) as rg:
            ctx: RenderContext
            for ctx in contexts:
                rg(ctx)

    def _from_dictable(self, *args, **kwargs) -> Dict[str, Any]:
        if self.is_error:
            return {'error': self.failure.asdict()}
        else:
            return super()._from_dictable(*args, **kwargs)

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout,
              include_doc: bool = True):
        if self.is_error:
            super().write(depth, writer)
        else:
            if include_doc:
                self.doc_graph.doc.amr.write(depth, writer, limit_sent=0)
            self._write_line('statistics:', depth, writer)
            self._write_object(self.stats, depth + 1, writer)

    def deallocate(self):
        self._try_deallocate(self._data)
        del self._data
        del self._component_paths
        del self._context
