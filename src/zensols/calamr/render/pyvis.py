"""Renderer AMR graphs using :mod:`pyvis`.

"""
__author__ = 'Paul Landes'

from typing import Dict, Any, Tuple, List, Set, Iterable
from dataclasses import dataclass, field
import logging
import textwrap
import json
from pathlib import Path
from igraph import Graph, Vertex, Edge
from pyvis.network import Network
from zensols.util import APIError
from zensols.config import Settings
from .base import GraphRenderer, RenderContext
from .. import (
    GraphNode, GraphEdge, RoleGraphEdge, DocumentGraph,
    SentenceGraphAttribute, DocumentGraphNode, SentenceGraphNode,
    ConceptGraphNode, AttributeGraphNode, TerminalGraphNode,
    ComponentAlignmentGraphEdge, DocumentGraphComponent,
)
from .util import ColorUtil, Formatter

logger = logging.getLogger(__name__)


@dataclass
class PyvisGraphRenderer(GraphRenderer):
    """A graph renderer using the :class:`pyvis.network.Network` API.

    """
    extension: str = field(default='html')
    """The output file's extension."""

    def __post_init__(self):
        self._formatter: Formatter = None

    def display(self, out_file: Path):
        super().display(out_file)

    @staticmethod
    def _update_style(name: str, vst: Dict[str, Any], params: Dict[str, Any]):
        key = f'{name}_add'
        if key in vst:
            params.update(vst[key])

    def _populate_nodes(self, nodes: Iterable[Tuple[Vertex, GraphNode]],
                        net: Network, g: Graph = None, root: Vertex = None,
                        vcolors: Tuple[str] = None, root_off: int = None,
                        name: str = None, longest_path: int = None):
        def update_style(name: str):
            self._update_style(name, vst, params)

        vst: Dict[str, Any] = self.visual_style.asdict()
        node_fmt: str = vst['label_format']['node']
        is_comp: bool = g is not None
        cparams: Dict[str, Dict[str, Any]] = self.visual_style.component
        cparams: Dict[str, Any] = cparams.get(name, cparams['default'])
        v: Vertex
        gn: GraphNode
        for v, gn in nodes:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('adding node:', gn)
            nid: int = v.index
            title: str = None
            level: int = None
            params: Dict[str, Any] = dict(cparams)
            if is_comp:
                level = len(g.get_shortest_paths(root, to=v)[0])
                color: str = vst['concept_color'][name]
                if color == 'rainbow':
                    ix: int = max(0, level - root_off)
                    if ix < len(vcolors):
                        color = vcolors[ix]
                    else:
                        color = 'AntiqueWhite'
                params['color'] = color
            if gn.partition is not None:
                style: Dict[str, Any] = \
                    vst['concept_color']['partition'][gn.partition]
                params.update(style)
            params['label'] = node_fmt.format(v=v, gn=gn)
            if self.rooted:
                params['level'] = level
            if isinstance(gn, ConceptGraphNode):
                update_style('concept')
            elif isinstance(gn, AttributeGraphNode):
                update_style('attribute')
            elif isinstance(gn, DocumentGraphNode):
                update_style('doc')
            elif isinstance(gn, SentenceGraphNode):
                update_style('sent')
            elif isinstance(gn, TerminalGraphNode):
                update_style('terminal')
                if not gn.is_source and longest_path is not None:
                    params['level'] = longest_path
                if not gn.is_source:
                    update_style('sink')
            else:
                raise APIError(f'Unknown doc node type: {type(gn)}')
            params['title'] = self._formatter.node(v.index, gn, '\n')
            if is_comp and v.index == root.index:
                update_style('root')
            net.add_node(nid, **params)

    def _populate_edges(self, edges: Iterable[Tuple[Edge, GraphEdge]],
                        net: Network):
        def update_style(name: str):
            self._update_style(name, vst, params)

        vst: Dict[str, Any] = self.visual_style.asdict()
        role_colors: Dict[str, str] = self.visual_style.role_colors
        label_fmt: str = vst['label_format']['edge']
        weights: Dict[str, Any] = self.visual_style.weights
        weighted_scale: float = weights['scale']
        weighted_edge_flow: bool = weights['flow']
        flow_color_ends: int = weights['color_ends']
        flow_color_buckets: int = weights['color_buckets']
        max_capacity: int = weights['max_capacity']
        max_edge_width: int = weights['max_edge_width']
        # create flow gradients of red (rgb=0.01) for component alignment edges
        flow_colors_component_alignment: Tuple[str] = ColorUtil.gradient_colors(
            n=flow_color_buckets, rgb=0.01)
        # create flow edge gradients of blue (rgb=0.01) for all others
        flow_colors_non_component_alignment: Tuple[str] = \
            ColorUtil.gradient_colors(n=flow_color_buckets, rgb=0.7)
        params = {}
        update_style('edge')
        n_params = dict(params)
        e: Edge
        ge: GraphEdge
        for e, ge in edges:
            params = dict(n_params)
            params['label'] = label_fmt.format(ge=ge, e=e)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('adding edge:', ge)
            title = f'edge: {e.index}\ndesc: {ge.description}'
            if isinstance(ge, SentenceGraphAttribute):
                fmt: str = self._formatter.edge(e.index, ge, '\n')
                if fmt is not None:
                    title = title + '\n' + fmt
            # color AMR role edges
            if isinstance(ge, RoleGraphEdge):
                if ge.role.is_inverted:
                    params['color'] = role_colors.get('inverse')
                else:
                    color: str = role_colors.get(ge.role.prefix)
                    if color is not None:
                        params['color'] = color
            # color cross source/summary edges with width
            if 'weighted_edge_add' in vst:
                update_style('weighted_edge')
                width = ge.capacity * weighted_scale
                if width < GraphEdge.MAX_CAPACITY:
                    width = round(width)
                params['width'] = min(max_edge_width, (width + params['width']))
            if weighted_edge_flow:
                if isinstance(ge, ComponentAlignmentGraphEdge):
                    flow_colors = flow_colors_component_alignment
                else:
                    flow_colors = flow_colors_non_component_alignment
                # compute edge color gradient based on flow
                capacity: float = min(ge.capacity, max_capacity) + 1
                flow: float = min(ge.flow, capacity - 1)
                flow_bucket: int = flow_color_buckets - \
                    int(flow / capacity *
                        (flow_color_buckets - (flow_color_ends * 2))) - \
                    flow_color_ends - 1
                params['color'] = flow_colors[flow_bucket]
            # add flow and capacity to pop up window
            capacity_str = ge.capacity_str(vst['edge_capacity_strlen'])
            flow_str = ge.flow_str(vst['edge_flow_strlen'])
            title = f'{title}\ncapacity: {capacity_str}\nflow: {flow_str}'
            if len(title) > 0:
                params['title'] = title
            net.add_edge(e.source, e.target, **params)

    def _populate_graph_comp(self, comp: DocumentGraphComponent, net: Network):
        vst: Dict[str, Any] = self.visual_style.asdict()
        root_off: int = vst['root_offset']
        g: Graph = comp.graph
        root: Vertex = comp.root
        shortest_paths: List[List[int]] = g.get_shortest_paths(root)
        longest_path: int = max(root_off + 1, max(map(len, shortest_paths)))
        lights: Dict[str, float] = vst['color_lightness']
        light: float = lights.get(comp.name, lights['default'])
        vcolors: Tuple[str] = ColorUtil.rainbow_colors(
            n=abs(longest_path - root_off) + 1, light=light)
        # add vertexes
        self._populate_nodes(comp.vs.items(), net, g, root, vcolors,
                             root_off, comp.name)
        # add edges
        self._populate_edges(comp.es.items(), net)
        return longest_path

    def _populate_graph(self, dg: DocumentGraph, net: Network):
        comp: DocumentGraphComponent
        vertexes: Set[Vertex] = set()
        edges: Set[Edge] = set()
        longest_path: int = 0
        for comp in dg.components:
            longest_path = max(self._populate_graph_comp(comp, net),
                               longest_path)
            vertexes.update(comp.vs.keys())
            edges.update(comp.es.keys())
        left_vertexs: Set[Vertex] = set(dg.vs.keys()) - vertexes
        left_edges: Set[Edge] = set(dg.es.keys()) - edges
        self._populate_nodes(map(lambda e: (e, dg[e]), left_vertexs),
                             net, longest_path=longest_path - 1)
        self._populate_edges(map(lambda e: (e, dg[e]), left_edges), net)

    def _set_options(self, net: Network):
        """Set pyvis Network options.

        :see: `Options <https://visjs.github.io/vis-network/docs/network/layout.html>`

        """
        both_opts: Dict[str, Dict[str, Any]] = self.visual_style.options
        rooted_key: str = 'rooted' if self.rooted else 'non_rooted'
        opts: Dict[str, Any] = both_opts[rooted_key]
        opt_str: str = 'var options = ' + json.dumps(opts)
        net.set_options(opt_str)

    def _render_to_file(self, context: RenderContext, out_file: Path):
        def map_comp(c: DocumentGraphComponent):
            text = textwrap.shorten(c.node.doc.text, 50)
            return f'{{{c.node.name}}}{text}'

        vst: Settings = self.visual_style
        self._formatter = Formatter(width=vst.label_format['wrap_width'])
        heading: str = context.heading
        net = Network(
            directed=True,
            width='100%',
            height='750px' if vst.show_layout_buttons else '100%')
        net.set_edge_smooth('dynamic')
        self._populate_graph(context.doc_graph, net)
        if heading is None and False:
            heading = ' '.join(map(map_comp, context.doc_graph.components))
        if heading is not None:
            net.heading = heading
        self._set_options(net)
        if 0:
            net.toggle_stabilization(False)
        net.show(str(out_file))
