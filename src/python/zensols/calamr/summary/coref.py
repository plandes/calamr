"""Adds coreferences to the graph.

"""
__author__ = 'Paul Landes'


from typing import Iterable, Tuple, List, Dict
from dataclasses import dataclass, field
import logging
from igraph import Graph, Vertex, Edge, EdgeSeq
from zensols.config import Dictable
from zensols.amr import RelationSet, Relation, Reference
from .. import (
    SentenceIndex, SentenceEntry, GraphNode, GraphEdge, ConceptGraphNode,
    ComponentAlignmentGraphEdge, ComponentCorefAlignmentGraphEdge,
    GraphComponent, DocumentGraphComponent, GraphAttributeContext,
    DocumentGraphController, DocumentGraph, GraphAlignmentConstructor,
)
from .factory import SummaryConstants

logger = logging.getLogger(__name__)


@dataclass
class _GraphReference(Dictable):
    reference: Reference = field()
    concept: Vertex = field()
    alignments: Tuple[Edge] = field()

    def __str__(self) -> str:
        concept: ConceptGraphNode = GraphComponent.to_node(self.concept)
        return f'{concept} ({len(self.alignments)})'


@dataclass
class _GraphRelation(Dictable):
    relation: Relation = field()
    references: Tuple[_GraphReference] = field()

    @property
    def non_zero_references(self) -> Iterable[_GraphReference]:
        return filter(lambda r: len(r.alignments) > 0, self.references)


@dataclass
class CorefDocumentGraphController(DocumentGraphController):
    """Creates graph alignment :class:`..ComponentAlignmentGraphEdge` instances
    based on coreferent concept nodes.

    """
    constructor: GraphAlignmentConstructor = field()
    """The constructor used to get the source and sink nodes."""

    # source_component_name: str = field()
    # """The source component for bipartitie coreference directed edges."""

    def _find_concept(self, comp: DocumentGraphComponent, ref: Reference) -> \
            ConceptGraphNode:
        sent_index: SentenceIndex = comp.sent_index
        entry: SentenceEntry = sent_index.by_sentence.get(ref.sent)
        if entry is not None:
            node: ConceptGraphNode = entry.concept_by_variable[ref.variable]
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'found concept: {node}')
            return node

    def _find_concepts(self, comp: DocumentGraphComponent, rel: Relation) -> \
            Iterable[_GraphReference]:
        def filter_edge(e: Edge) -> bool:
            ge: GraphEdge = GraphComponent.to_edge(e)
            return isinstance(ge, ComponentAlignmentGraphEdge)

        n2v: Dict[GraphNode, Vertex] = comp.node_to_vertex
        ref: Reference
        for ref in rel:
            gn: ConceptGraphNode = self._find_concept(comp, ref)
            if gn is not None:
                v: Vertex = n2v[gn]
                edges: Tuple[Edge] = filter(filter_edge, v.incident('in'))
                yield _GraphReference(ref, n2v[gn], tuple(edges))

    def _get_relations(self, comp: DocumentGraphComponent) -> \
            List[_GraphRelation]:
        relset: RelationSet = comp.relation_set
        graph_rels: List[_GraphRelation] = []
        if logger.isEnabledFor(logging.DEBUG):
            relset.write_to_log(logger, logging.DEBUG)
            logger.debug(f"component '{comp.name}': {len(relset)} relations")
        rel: Relation
        for rel in relset:
            graph_rels.append(_GraphRelation(
                relation=rel,
                references=tuple(self._find_concepts(comp, rel))))
        return graph_rels

    def _connect_ref(self, src: _GraphReference, dst: _GraphReference,
                     g: Graph, rel: _GraphRelation) -> int:
        se: Edge
        for se in src.alignments:
            src_align: int = se.source
            ge: GraphEdge = GraphComponent.to_edge(se)
            same_sent: bool = dst.reference.sent == src.reference.sent
            capacity: float = ge.capacity
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f'{src}[{src_align}] -> ' +
                    f'{dst}[{dst.concept.index}]: align={src_align}, ' +
                    f'edge={ge}, ss={same_sent}, cap={capacity}')
            e: Edge = g.add_edge(src_align, dst.concept)
            e[GraphComponent.GRAPH_ATTRIB_NAME] = \
                ComponentCorefAlignmentGraphEdge(
                    context=ge.context,
                    capacity=capacity,
                    relation=rel.relation)
        return len(src.alignments)

    def _add_comp_corefs(self, comp: DocumentGraphComponent) -> int:
        n_edge_adds: int = 0
        g: Graph = comp.graph
        rel: _GraphRelation
        for rel in self._get_relations(comp):
            align_refs: Tuple[_GraphReference] = tuple(rel.non_zero_references)
            # we can only add coreference alignments if there's at least one of
            # that has an alignemnt
            if len(align_refs) > 0:
                dst: _GraphReference
                for dst in rel.references:
                    src: _GraphReference
                    # avoid self referential links
                    for src in filter(lambda r: id(r) != id(dst), align_refs):
                        n_edge_adds += self._connect_ref(src, dst, g, rel)
        return n_edge_adds

    def _connect_bipartite_refs(self, doc_graph: DocumentGraph, rel: Relation,
                                src: _GraphReference, smy: _GraphReference):
        # assume initial reverse (summary flow to source) graph
        sv: int = smy.concept.index
        tv: int = src.concept.index
        eseq: EdgeSeq = doc_graph.graph.es.select(_source=sv, _target=tv)
        edge: Edge = eseq[0] if len(eseq) > 0 else None
        ctx: GraphAttributeContext
        if edge is not None:
            ge: GraphEdge = GraphComponent.to_edge(edge)
            ctx = ge.context
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'existing bipartite edge: {edge}')
        else:
            gn: GraphNode = GraphComponent.to_node(src.concept)
            ctx = gn.context
            edge = doc_graph.graph.add_edge(sv, tv)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('no previous edge--created one')
        edge[GraphComponent.GRAPH_ATTRIB_NAME] = \
            ComponentCorefAlignmentGraphEdge(
                context=ctx,
                capacity=1.,
                relation=rel,
                is_bipartite=True)

    def _add_bipartite_corefs(self, doc_graph: DocumentGraph):
        updates: int = 0
        relset: RelationSet = doc_graph.bipartite_relation_set
        rel: Relation
        for rel in relset:
            src = doc_graph.components_by_name[SummaryConstants.SOURCE_COMP]
            smy = doc_graph.components_by_name[SummaryConstants.SUMMARY_COMP]
            smy_refs: Tuple[_GraphReference] = \
                tuple(self._find_concepts(smy, rel))
            src_ref: _GraphReference
            for src_ref in self._find_concepts(src, rel):
                smy_ref: _GraphReference
                for smy_ref in smy_refs:
                    self._connect_bipartite_refs(
                        doc_graph, rel, src_ref, smy_ref)
                    updates += 1
        return updates

    def _invoke(self, doc_graph: DocumentGraph) -> int:
        n_edge_adds: int = 0
        comp: DocumentGraphComponent
        for comp in doc_graph.components:
            n_edge_adds += self._add_comp_corefs(comp)
        n_edge_adds += self._add_bipartite_corefs(doc_graph)
        if n_edge_adds > 0:
            doc_graph.invalidate()
        return n_edge_adds
