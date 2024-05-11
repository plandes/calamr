"""Renderer AMR graphs using :mod:`graphviz`.

"""
__author__ = 'Paul Landes'

from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import logging
from pathlib import Path
from igraph import Graph, Vertex, Edge
from graphviz import Digraph
from .. import (
    ComponentAlignmentError, GraphEdge, GraphNode,
    ComponentAlignmentGraphEdge, ComponentCorefAlignmentGraphEdge,
    DocumentGraph, DocumentGraphComponent,
)
from .base import GraphRenderer, RenderContext
from .util import ColorUtil, Formatter

logger = logging.getLogger(__name__)


@dataclass
class GraphVisGraphRenderer(GraphRenderer):
    """A graph renderer using :class:`graphviz.Digraph`.

    """
    extension: str = field(default='pdf')
    """The output file's extension."""

    def __post_init__(self):
        self._formatter: Formatter = None

    def _get_style(self, g: Graph, root: int) -> Dict[str, Any]:
        visual_style = dict(self.visual_style)
        return visual_style

    def _get_comp(self, doc_graph: DocumentGraph, v: Vertex) -> Optional[str]:
        comp: DocumentGraphComponent
        for comp in doc_graph.components:
            if v in comp.vs:
                return comp.name

    def _add_nodes(self, doc_graph: DocumentGraph, dg: Digraph):
        style: Dict[str, Any] = self.visual_style.node
        attr_style: Dict[str, str] = style['attr']
        comp_style: Dict[str, str] = style['comp']
        df_style: Dict[str, str] = {}
        v: Vertex
        gn: GraphNode
        for v, gn in doc_graph.vs.items():
            title: str = None
            at: str = gn.attrib_type
            comp_name: str = self._get_comp(doc_graph, v)
            cparams: Dict[str, str] = dict(comp_style.get(comp_name, df_style))
            aparams: Dict[str, str] = attr_style.get(at, attr_style['default'])
            if 'style' in aparams and 'style' in cparams:
                aparams['style'] = ','.join(
                    [aparams['style'], cparams['style']])
            cparams.update(aparams)
            title: str = self._formatter.node(v.index, gn, '\n')
            cparams['tooltip'] = title
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'creating vertex: {gn.label}, type: {at}, ' +
                             f'attr: ({v.index})')
            label: str
            if style['label']['add_id']:
                label = f'{gn.label}[{gn.id}]'
            else:
                label = gn.label
            dg.node(str(v.index), label=label, **cparams)

    def _add_edges(self, doc_graph: DocumentGraph, dg: Digraph):
        style: Dict[str, Any] = self.visual_style.edge
        attr_style: Dict[str, str] = style['attr']
        weights: Dict[str, Any] = style['weights']
        weighted_scale: float = weights['weighted_scale']
        weighted_edge_flow: bool = weights['flow']
        max_edge_width: int = weights['max_edge_width']
        flow_scale: float = weights['flow_scale']
        flow_color_ends: int = weights['color_ends']
        flow_color_buckets: int = weights['color_buckets']
        max_capacity: int = weights['max_capacity']
        max_capacity: int = weights['max_capacity']
        # create flow gradients of red (rgb=0.01) for component alignment edges
        flow_colors_component_alignment: Tuple[str] = ColorUtil.gradient_colors(
            n=flow_color_buckets, rgb=0.01)
        # create flow edge gradients of blue (rgb=0.01) for all others
        flow_colors_non_component_alignment: Tuple[str] = \
            ColorUtil.gradient_colors(n=flow_color_buckets, rgb=0.7)
        title: str
        e: Edge
        ge: GraphEdge
        for e, ge in doc_graph.es.items():
            at: str = ge.attrib_type
            params: Dict[str, str] = attr_style.get(at, attr_style['default'])
            params = dict(params)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'edge {ge.label}: {e.source} -> {e.target}')
            if 'weighted_edge_add' in style:
                params.update(style['weighted_edge_add'])
                width = ge.capacity * weighted_scale
                if width < GraphEdge.MAX_CAPACITY:
                    width = round(width)
                params['penwidth'] = int(min(
                    max_edge_width, (width + params['penwidth'])))
            if weighted_edge_flow:
                if isinstance(ge, ComponentCorefAlignmentGraphEdge):
                    params['style'] = 'dashed'
                if isinstance(ge, ComponentAlignmentGraphEdge):
                    flow_colors = flow_colors_component_alignment
                else:
                    flow_colors = flow_colors_non_component_alignment
                # compute edge color gradient based on flow
                capacity: float = min(ge.capacity, max_capacity) + 1
                flow_val: float = ge.vis_flow if hasattr(ge, 'vis_flow') \
                    else ge.flow
                flow: float = min(flow_val * flow_scale, capacity - 1)
                flow_bucket: int = flow_color_buckets - \
                    int(flow /
                        capacity *
                        (flow_color_buckets - (flow_color_ends * 2))) - \
                    flow_color_ends - 1
                params['color'] = flow_colors[flow_bucket]
            title = self._formatter.edge(e.index, ge, '\n')
            params['tooltip'] = title
            for k, v in params.items():
                params[k] = str(v)
            label: str
            if style['label']['add_id']:
                label = f'{ge.label}[{ge.id}]'
            else:
                label = ge.label
            dg.edge(str(e.source), str(e.target), label=label, **params)

    def _construct_graph(self, doc_graph: DocumentGraph,
                         out_file: Path) -> Digraph:
        style: Dict[str, Any] = self.visual_style.graph
        style: Dict[str, str] = {
            k: str(style[k]) for k in self.visual_style.graph}
        dg = Digraph('amr_graph', filename=out_file, format=self.extension)
        dg.attr(**style)
        self._add_nodes(doc_graph, dg)
        self._add_edges(doc_graph, dg)
        return dg

    def _render_to_file(self, context: RenderContext, out_file: Path):
        style: Dict[str, Any] = self.visual_style
        vstyle: Dict[str, Any] = style.attributes
        ext: str = self.extension
        self._formatter = Formatter(width=self.visual_style.label['wrap_width'])
        if out_file.suffix != f'.{ext}':
            raise ComponentAlignmentError(
                f'Expecting {ext} but got: {out_file.suffix}')
        out_file = out_file.parent / out_file.stem
        if context.visual_style is not None:
            attribs: Dict[str, Any] = context.visual_style.get('attributes')
            if attribs is not None:
                vstyle = dict(vstyle)
                vstyle.update(attribs)
        try:
            dg: Digraph = self._construct_graph(context.doc_graph, out_file)
            dg.attr(**vstyle)
            if style['add_heading'] and context.heading is not None:
                dg.attr(label=context.heading)
            dg.render(quiet=True)
        finally:
            if out_file.is_file():
                out_file.unlink()
