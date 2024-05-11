"""Renderer AMR graphs using :mod:`plotly`.

"""
__author__ = 'Paul Landes'

from typing import Dict, Any, Tuple, List
from dataclasses import dataclass, field
import logging
import itertools as it
from itertools import chain
from pathlib import Path
import math
from igraph import Graph, Vertex, Edge, Layout
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly
from zensols.persist import persisted
from .. import (
    GraphEdge, GraphNode, DocumentGraph, DocumentGraphComponent, GraphComponent,
    ComponentAlignmentGraphEdge, ComponentCorefAlignmentGraphEdge,
)
from .base import GraphRenderer, RenderContext
from .util import ColorUtil, Formatter

logger = logging.getLogger(__name__)


@dataclass
class _CompPlot(object):
    comp: DocumentGraphComponent
    node_trace: Tuple[go.Scatter3d, ...]
    edge_traces: Tuple[Tuple[go.Scatter3d, ...], ...]

    @property
    def traces(self) -> Tuple[go.Scatter3d]:
        return [self.node_trace] + list(self.edge_traces)


class _CompPlotSet(object):
    def __init__(self):
        self._plots: Dict[str, _CompPlot] = {}
        self._coords: Dict[GraphNode, Tuple[float, float, float]] = {}

    def add(self, plot: _CompPlot):
        self._plots[plot.comp.name] = plot

    def find_node(self, gn: GraphNode):
        coord: Tuple[float, float, float] = self._coords.get(gn)
        if coord is None:
            for name, plot in self._plots.items():
                v: Vertex = plot.comp.node_to_vertex.get(gn)
                if v is not None:
                    trace: go.Scatter3d = plot.node_trace
                    coord = trace.x[v.index], trace.y[v.index], trace.z[v.index]
                    self._coords[gn] = coord
        return coord

    @property
    def traces(self) -> Tuple[go.Scatter3d]:
        return tuple(chain.from_iterable(
            map(lambda p: p.traces,
                sorted(self._plots.values(),
                       key=lambda c: c.comp.name,
                       reverse=True))))


@dataclass(unsafe_hash=True)
class _LineStyle(object):
    color: str = field(default=None)
    width: float = field(default=None)
    dash: str = field(default='solid')


@dataclass
class _EdgeSet(object):
    name: str = field()
    style: Dict[str, Any] = field()
    line_style: _LineStyle = field()
    formatter: Formatter = field()
    edges: List[str] = field(default_factory=list)
    coords: List[Tuple] = field(default_factory=list)

    @property
    @persisted('_numpy_coords')
    def numpy_coords(self) -> np.ndarray:
        return np.array(self.coords)

    def _create_line_trace(self) -> go.Scatter3d:
        c: np.ndarray = self.numpy_coords
        x = tuple(chain.from_iterable(
            zip(c[:, 0, 0], c[:, 1, 0], it.repeat(None))))
        y = tuple(chain.from_iterable(
            zip(c[:, 0, 1], c[:, 1, 1], it.repeat(None))))
        z = tuple(chain.from_iterable(
            zip(c[:, 0, 2], c[:, 1, 2], it.repeat(None))))
        return go.Scatter3d(
            x=x,
            y=y,
            z=z,
            name=self.name,
            mode='lines',
            line=self.line_style.__dict__,
            hoverinfo='none')

    def _create_label_trace(self) -> go.Scatter3d:
        c: np.ndarray = self.numpy_coords
        mids: np.ndarray = (c[:, 0, :] + c[:, 1, :]) / 2
        rows: List[Tuple] = []
        for (e, ge), (x, y, z) in zip(self.edges, mids):
            desc = self.formatter.edge(e.index, ge, '<br>')
            rows.append((e.index, str(ge), desc, x, y, z))
        df = pd.DataFrame(rows, columns='id label desc x y z'.split())
        return go.Scatter3d(
            x=df['x'],
            y=df['y'],
            z=df['z'],
            name=self.name,
            mode='markers',
            opacity=0.25,
            marker=dict(
                symbol='circle',
                size=((self.line_style.width * 0.6) + 1)),
            text=df['label'],
            customdata=np.stack((df['desc'],), axis=-1),
            textposition='top center',
            line=self.line_style.__dict__,
            hovertemplate='%{customdata[0]}')

    def _create_arrow_tip_traces(self) -> go.Scatter3d:
        """Render arrows by forcing the vector field rendering API to create 3d
        cones each arrow.

        """
        ca: np.ndarray = self.numpy_coords
        start_ix: int = 0
        end_ix: int = 1
        # despite this thread, absolute scaling does not work and still based on
        # what seems to be the cross product from eye-balling it
        #
        # https://github.com/plotly/plotly.js/issues/3613
        sr: float = .2 + (self.line_style.width * 1e-3)
        if self.name == 'alignment':
            z_sep: float = self.style['node']['z-sep']
            grow = 1 / (1 + math.exp(-z_sep))
            sr = sr - (grow * 0.15)
        else:
            start_ix, end_ix = 1, 0
        color: str = self.line_style.color
        # distance between points, shape = (<N points>, 3)
        cv: np.ndarray = (ca[:, end_ix, :] - ca[:, start_ix, :])
        # magintiude of all points, shape = (<N points>,)
        mag: np.ndarray = np.sqrt((cv ** 2).sum(axis=1))
        mag = np.expand_dims(mag, 1)
        # divide by error are probably from underflows
        if len(mag[mag == 0]) > 0:
            logger.warning('found 0 magnitude')
            mag[mag == 0] = 1e-10
        # create unit vectors to attempt uniform size for u, v, w vector field,
        # shape = (<N of points>, 3)
        uv: np.ndarray = cv / mag
        # all vector data renders different cone sizes, probably because it
        # interpretes as a vector field--create a separate trace for each arrow
        for i in range(ca.shape[0]):
            yield go.Cone(
                x=(ca[i, end_ix, 0],),
                y=(ca[i, end_ix, 1],),
                z=(ca[i, end_ix, 2],),
                u=(uv[i, 0],),
                v=(uv[i, 1],),
                w=(uv[i, 2],),
                sizemode='absolute',
                sizeref=sr,
                showlegend=False,
                showscale=False,
                anchor='tip',
                hoverinfo='skip',
                # force color to that of the line rather than default vector
                # field gradient
                colorscale=[[0, color], [1, color]])

    def create_traces(self) -> go.Scatter3d:
        return (self._create_line_trace(),
                self._create_label_trace(),
                *tuple(self._create_arrow_tip_traces()))


@dataclass
class PlotlyGraphRenderer(GraphRenderer):
    """Render teh graph in plotly using a tree layout with each separated on the
    Z-axis.  This makes it easier to see each component separately and easier to
    follow where the component alignment lie.

    To do this, the components have to be detached from the connected bipartite
    graph to be rendered separately.  The alignment edges are then rendered
    afterwward.

    """
    extension: str = field(default='html')
    """The output file's extension."""

    def __post_init__(self):
        self._formatter: Formatter = None

    def _create_edges(self, name: str, graph: Graph,
                      layt: Dict[Edge, Tuple[GraphEdge,
                                             Tuple[float, float, int],
                                             Tuple[float, float, int]]]) -> \
            Tuple[go.Scatter3d]:
        style: Dict[str, Any] = self.visual_style
        estyle: Dict[str, Any] = style['edge']
        weights: Dict[str, Any] = estyle['weights']
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
        edge_sets: List[_EdgeSet] = {}
        e: Edge
        for e in graph.es:
            ge: GraphEdge = GraphComponent.to_edge(e)
            line_style = _LineStyle()
            coord: Tuple[Tuple[float, float, None]] = layt.get(e)
            if coord is None:
                continue
            flow_val: float = ge.vis_flow if hasattr(ge, 'vis_flow') \
                else ge.flow
            capacity: float = min(ge.capacity, max_capacity) + 1
            flow: int = min(flow_val * flow_scale, capacity - 1)
            flow_bucket: int = flow_color_buckets - \
                int(flow /
                    capacity *
                    (flow_color_buckets - (flow_color_ends * 2))) - \
                flow_color_ends - 1
            if weighted_edge_flow:
                if isinstance(ge, ComponentCorefAlignmentGraphEdge):
                    line_style.dash = 'dot'
                if isinstance(ge, ComponentAlignmentGraphEdge):
                    flow_colors = flow_colors_component_alignment
                else:
                    flow_colors = flow_colors_non_component_alignment
            line_style.color = flow_colors[flow_bucket]
            width = ge.capacity * weighted_scale
            if width < GraphEdge.MAX_CAPACITY:
                width = round(width)
            line_style.width = int(min(
                max_edge_width,
                ((width + estyle['line_width']) *
                 (estyle['capacity_line_scale']
                  if name == 'alignment' else 1.))))
            eset = edge_sets.get(line_style)
            if eset is None:
                eset = _EdgeSet(name, style, line_style, self._formatter)
                edge_sets[line_style] = eset
            eset.coords.append(coord)
            eset.edges.append((e, ge))
        return tuple(chain.from_iterable(map(
            lambda es: es.create_traces(), edge_sets.values())))

    def _add_comp(self, comp: DocumentGraphComponent, z: int) -> _CompPlot:
        """Render component ``comp`` in the (separate) Z-axis space ``z``.  See
        class docs.

        """
        def map_coord(e: Edge) -> Tuple[GraphEdge,
                                        Tuple[float, float, int],
                                        Tuple[float, float, int]]:
            src: int
            targ: int
            if render_reversed_edges:
                src, targ = e.target, e.source
            else:
                src, targ = e.source, e.target
            return (e, ((*layt[src], z), (*layt[targ], z)))

        edges_reversed: bool = comp.edges_reversed
        comp: DocumentGraphComponent = comp.clone(
            # we have to reverse the edges on the reversed flow graphs since
            # igraph.Layout ('rt') a tree, otherwise the tree is inverted
            reverse_edges=edges_reversed,
            # must be a deep clone to detach the bipartite graph into a separate
            # component, which is then later connected with alignments
            deep=True)
        render_reversed_edges: bool = not edges_reversed
        style: Dict[str, Any] = self.visual_style
        nstyle: Dict[str, Any] = style['node']
        graph: Graph = comp.graph
        layt: Layout = graph.layout('rt')
        # flip along x-axis since the tree is drawn upside down; if not, the
        # sentences are rendered in the reverse order
        layt.mirror(dim=1 if edges_reversed else 0)
        n_nodes: int = graph.vcount()
        add_id: bool = nstyle['label']['add_id']
        rows: List[Tuple[Any, ...]] = []
        i: int
        for i in range(n_nodes):
            gn: GraphNode = comp.node_by_id(i)
            lb: str = f'{gn.label}[{i}]' if add_id else gn.label
            title: str = self._formatter.node(i, gn, '<br>')
            rows.append((*layt[i], z, lb, title))
        df = pd.DataFrame(data=rows, columns='x y z label title'.split())
        node_trace = go.Scatter3d(
            x=df['x'],
            y=df['y'],
            z=df['z'],
            name=comp.name,
            mode='markers+text',
            marker=nstyle['comp'][comp.name],
            text=df['label'],
            textposition='top center',
            customdata=np.stack((df['label'], df['title']), axis=-1),
            hovertemplate='%{customdata[1]}')
        coords = dict(map(map_coord, graph.es))
        return _CompPlot(
            comp,
            node_trace,
            self._create_edges(comp.name, graph, coords))

    def _add_capacities(self, doc_graph: DocumentGraph,
                        comp_plot_set: _CompPlotSet) -> Tuple[go.Scatter3d]:
        def filter_edge(t: Tuple[Edge, GraphEdge]) -> bool:
            return isinstance(t[1], ComponentAlignmentGraphEdge)

        layt: Dict[Edge, Tuple] = {}
        for t in filter(filter_edge, doc_graph.es.items()):
            e: Edge = t[0]
            sgn: GraphNode = doc_graph.node_by_id(e.source)
            tgn: GraphNode = doc_graph.node_by_id(e.target)
            sc: Tuple[float, float, float] = comp_plot_set.find_node(sgn)
            tc: Tuple[float, float, float] = comp_plot_set.find_node(tgn)
            if sc is None:
                logger.error(f'missing capacity source node : {sgn}')
                continue
            if tc is None:
                logger.error(f'missing capacity target node: {tgn}')
                continue
            layt[e] = (sc, tc)
        return self._create_edges('alignment', doc_graph.graph, layt)

    def _render_to_file(self, context: RenderContext, out_file: Path):
        source_on_top: bool = True
        style: Dict[str, Any] = self.visual_style
        nstyle: Dict[str, Any] = style['node']
        z_sep: float = nstyle['z-sep']
        comp_plot_set = _CompPlotSet()
        doc_graph: DocumentGraph = context.doc_graph
        self._formatter = Formatter(width=style['label']['wrap_width'])
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'rendering: {doc_graph}, {context}')
        cix: int
        comps: List[DocumentGraphComponent] = sorted(
            doc_graph.components,
            key=lambda c: c.name,
            reverse=source_on_top)
        first_comp_rev: bool = comps[0].edges_reversed
        comp: DocumentGraphComponent
        for cix, comp in enumerate(comps):
            comp_plot_set.add(self._add_comp(comp, cix))
        axis = dict(showbackground=False,
                    showline=False,
                    zeroline=False,
                    showgrid=False,
                    showticklabels=False,
                    title='')
        cam_x_up: int = 1 if first_comp_rev else -1
        layout = go.Layout(
            title=dict(text=context.heading, x=0, xref='paper'),
            margin=dict(l=40, r=40, t=40, b=40),
            showlegend=False,
            scene=dict(
                xaxis=dict(axis, showspikes=False),
                yaxis=dict(axis, showspikes=False),
                zaxis=dict(axis, showspikes=False),
                camera=dict(
                    up=dict(x=cam_x_up, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=0, y=0, z=2.5)),
                aspectratio=dict(x=1, y=1, z=z_sep)),
            hovermode='closest',
            shapes=(go.layout.Shape(
                type='rect',
                xref='paper',
                yref='paper',
                x0=0,
                y0=0,
                x1=1,
                y1=1,
                line={'width': 1, 'color': 'black'}),))
        caps: Tuple[go.Scatter3d] = self._add_capacities(
            doc_graph, comp_plot_set)
        fig = go.Figure(
            data=list(comp_plot_set.traces) + list(caps),
            layout=layout)
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'wrote: {out_file}')
        plotly.offline.plot(
            fig,
            filename=str(out_file),
            auto_open=False)
