"""Utilities such as gradient color generators.

"""
__author__ = 'Paul Landes'

from typing import Tuple
from dataclasses import dataclass, field
from io import StringIO
from colorsys import hls_to_rgb
import textwrap as tw
from zensols.nlp import FeatureToken
from zensols.amr import Relation, AmrFeatureSentence
from .. import (
    GraphNode, GraphEdge, DocumentGraphNode, SentenceGraphNode,
    SentenceGraphAttribute, ComponentCorefAlignmentGraphEdge,
)


class ColorUtil(object):
    @staticmethod
    def rainbow_colors(n: int = 10, light: float = 0.5, end: float = 1.):
        def int_to_rgb(i: int):
            ctup: Tuple[float] = hls_to_rgb(end * i / (n - 1), light, 1)
            r, g, b = tuple(map(lambda x: int(255 * x), ctup))
            return "#{0:02x}{1:02x}{2:02x}".format(r, g, b)

        return tuple(map(int_to_rgb, range(n)))

    @staticmethod
    def gradient_colors(n: int = 10, rgb: float = 0.5, end: float = 1.):
        def int_to_rgb(i: int):
            ctup: Tuple[float] = hls_to_rgb(rgb, (end * i / (n - 1)), 1)
            r, g, b = tuple(map(lambda x: int(255 * x), ctup))
            return "#{0:02x}{1:02x}{2:02x}".format(r, g, b)

        return tuple(map(int_to_rgb, range(n)))


@dataclass
class Formatter(object):
    width: int = field(default=60)
    desc_width: int = field(default=500)

    def _wrap(self, s: str, delim: str) -> str:
        s = s.replace('\n', ' ')
        s = tw.shorten(s, width=self.desc_width)
        s = delim.join(tw.wrap(s, width=self.width))
        return s

    def node(self, vertex: int, gn: GraphNode, delim: str) -> str:
        title: str = ''
        if isinstance(gn, DocumentGraphNode):
            title = self._wrap(gn.doc_node.description, delim)
        elif isinstance(gn, SentenceGraphAttribute):
            title = self._sent_graph_node(gn)
            title = '' if title is None else title
            desc: str = self._wrap(gn.description, delim)
            if len(title) > 0:
                title += delim
            title += f'description: {desc}'
            if title is not None:
                if delim != '\n':
                    title = title.replace('\n', delim)
        elif isinstance(gn, SentenceGraphNode):
            sent: AmrFeatureSentence = gn.sent
            title = self._wrap(sent.norm, delim)
        vstr: str = f"""\
vertex: {gn.id}{delim}\
ig vertex: {vertex}{delim}\
type: {gn.attrib_type}"""
        if len(title) == 0:
            desc: str = self._wrap(gn.description, delim)
            title = f'{vstr}{delim}description: {desc}'
        elif len(title) > 0:
            title = f'{vstr}{delim}{title}'
        else:
            title = vstr
        return title

    def edge(self, edge: int, ge: GraphEdge, delim: str) -> str:
        title: str = f"""\
edge: {ge.id}{delim}\
ig edge: {edge}{delim}\
desc: {ge.description}{delim}\
capacity: {ge.capacity_str()}{delim}\
flow: {ge.flow_str()}"""
        if isinstance(ge, ComponentCorefAlignmentGraphEdge):
            title += (delim + delim.join(self.coref_edge(ge)))
        return title

    def _sent_graph_node(self, dn: SentenceGraphNode) -> str:
        tokens: Tuple[FeatureToken] = dn.tokens
        if len(tokens) > 0:
            sio = StringIO()
            for i, tok in enumerate(tokens):
                if i > 0:
                    sio.write(('_' * 30) + '\n')
                tok.write_attributes(
                    writer=sio,
                    include_type=False,
                    include_none=False,
                    feature_ids='norm ent_ tag_ pos_'.split())
            return sio.getvalue().strip().replace('=', ': ')

    def coref_edge(self, edge: ComponentCorefAlignmentGraphEdge) -> Tuple[str]:
        rel: Relation = edge.relation
        return (f'relation: {repr(rel)}',
                f'bipartite: {edge.is_bipartite}')
