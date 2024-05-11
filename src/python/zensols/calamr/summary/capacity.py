"""Connection similarity computation.

"""
__author__ = 'Paul Landes'

from typing import (
    Iterable, Dict, List, Tuple, Set, Sequence, Callable,
    Type, Optional, ClassVar
)
from dataclasses import dataclass, field
import logging
import itertools as it
from functools import cmp_to_key
import math
import pandas as pd
from igraph import Graph, Vertex, Edge
import torch
from torch import Tensor
from torch.nn.functional import cosine_similarity
import numpy as np
from zensols.util import time
from zensols.config import Dictable
from zensols.propbankdb import Roleset, Role as FramesetRole
from zensols.propbankdb.embedpop import EmbeddingPopulator
from zensols.datdesc.hyperparam import HyperparamModel
from .. import (
    ComponentAlignmentError, Role, GraphComponent,
    DocumentGraphComponent, DocumentGraph,
    GraphNode, GraphEdge, TripleGraphNode, SentenceGraphNode,
    DocumentGraphNode, ConceptGraphNode, AttributeGraphNode,
)
from .factory import SummaryConstants

logger = logging.getLogger(__name__)


@dataclass
class _Context(object):
    """Context that lasts for the duration of a single call to
    :class:`.CapacityCalculator`.

    """
    dg: DocumentGraph = field()
    """The document graph (having all graph components) used to create alignment
    edges.

    """
    n2v: Dict[GraphNode, Vertex] = field()
    """A mapping of graph node to :class:`igraph.Graph` vertex."""

    sent_emb: Dict[Tuple[int, int], Tuple[Tensor, float]] = \
        field(default_factory=dict)
    """The sentence embeddings with ``(<source>, <summary>)`` tuples as keys."""

    sent_subgraphs: Dict[int, int] = field(default_factory=dict)
    """Concept and attribute vertex IDs to sentence IDs."""

    concept_emb: Dict[int, Tensor] = field(default_factory=dict)
    """Cached concept embeddings: k/vs are graph node ID to embedding in
    :meth:`_concept_embedding`.

    """
    neigh_emb: Dict[int, Tensor] = field(default_factory=dict)
    """Cached neighborhood embeddings: k/vs are graph node ID to embedding."""

    neigh_sim: Dict[Tuple[int, int, str], float] = field(default_factory=dict)
    """Cached neighborhood similiarty keyed as source/summary/type in
    :meth:`_neighboorhood_sim`.

    """
    def clear(self):
        # only concept_emp has a bearing on results
        for o in 'concept_emb neigh_emb neigh_sim'.split():
            getattr(self, o).clear()


@dataclass
class _BipartiteNodeFactory(object):
    """Return the cartesian product of nodes specified in :obj:`connectables`
    across the components of the graph.

    """
    connectables: Tuple[Type[GraphNode]] = field(
        default=(TripleGraphNode, SentenceGraphNode, DocumentGraphNode))
    """Classes of nodes allowed to be connected with alignment edges."""

    def _filter_connectable_node(self, n: GraphNode) -> bool:
        """Keep only nodes that have vectors used for comparison."""
        allowed = self.connectables
        return isinstance(n, allowed)

    def _connectable_nodes(self, comp: DocumentGraphComponent) -> \
            Iterable[GraphNode]:
        """Keep only component nodes that have vectors used for comparison.

        """
        return filter(self._filter_connectable_node, comp.vs.values())

    def _filter_connectable_nodes(self, s: GraphNode, m: GraphNode) -> bool:
        """Keep only summary/source node pairs that have identical attribute
        type (i.e. sentence with sentence, concept with concept, attribute
        with attribute etc).

        """
        return s.attrib_type == m.attrib_type

    def __call__(self, doc_graph: DocumentGraph) -> \
            Iterable[Tuple[GraphNode, GraphNode]]:
        """Return the cartesian product of nodes.

        :param doc_graph: the graph used to compute capacities

        :return: a Pandas dataframe of the cartesian product of graph alignment
                 capacities

        """
        # get summary and source nodes across both respective components of the
        # graph
        cbn: Dict[str, DocumentGraphComponent] = doc_graph.components_by_name
        src_comp: DocumentGraphComponent = cbn[SummaryConstants.SOURCE_COMP]
        smy_comp: DocumentGraphComponent = cbn[SummaryConstants.SUMMARY_COMP]
        src: Iterable[TripleGraphNode] = self._connectable_nodes(src_comp)
        smy: Iterable[TripleGraphNode] = self._connectable_nodes(smy_comp)
        return filter(lambda ns: self._filter_connectable_nodes(*ns),
                      it.product(src, smy))


@dataclass
class CapacityCalculator(Dictable):
    """Calculates and the component alignment capacities.

    """
    _DICTABLE_WRITABLE_DESCENDANTS: ClassVar[bool] = True

    # components
    embedding_populator: EmbeddingPopulator = field(repr=False)
    """Adds embeddings to frameset instances and used for debugging in this
    class.

    """
    # config
    include_tokens: bool = field()
    """Whether to include tokens in the dataframe for debugging."""

    clear_align_node_iteration: bool = field()
    """Whether to clear (:meth:`_Context.clear) the intermediate data structures
    after each iteration of the node alignment algorithm.  Cursory results on a
    small sample show flow and alignments only slightly change when
    :obj:`_Context.neigh_sim` is cleared.

    This is because the concept embeddings are used in network neighborhood
    calculations and a convergance algorithm (like CRFs) are implemented at this
    time.

    Setting to ``False`` shows a speeds it up by ~75% on a small random sample.

    """
    hyp: HyperparamModel = field()
    """The calculator's hyperparameters.

    Hyperparameters::

        :param similarity_dampen_exp: the exponent that dampens the cosine
                                      similarity for all nodes
        :type similarity_dampen_exp: dict

        :param concept_embedding_role_weights: the weights for concept node's
                                               roles (i.e. arg0 with neighbor
                                               node)
        :type concept_embedding_role_weights: dict

        :param concept_embedding_weights: the weights for concept node's
                                          weighted average across tokens and
                                          role sets
        :type concept_embedding_weights: dict

        :param neighbor_direction: indicate how to find neighbors (from the
                                   point of the reverse/maxflow graph with the
                                   root on the bottom), which is one of
                                   descendant (only children and descendents),
                                   ancestor (only parents and ancestors), or all
                                   (all neighbors)
        :type neighbor_direction: str; one of: descendant, ancestor, all

        :param neighbor_embedding_weights: weights used to scale each neighbor
                                           from the the current node and
                                           immediate neighbor to the furthest
                                           neighbor; if there is only one entry,
                                           the singleton value is multiplied by
                                           the respective nodes embeddings
        :type neighbor_embedding_weights: list

        :param neighbor_skew: neighborhood sidmoid skew settings with y/x
                              translation, and compression (how much to compress
                              or "squeeze" the function to provide a faster
                              transition) with cosine similarity as the input
        :type neighbor_skew: dict

        :param sentence_dampen: the slope for the linear dampening of nodes
                                under a sentence by sentence cosine similarity;
                                the higher the value the lower the sentence
                                similarity, which leads to lower concept and
                                attribute node similarities, must be in the
                                interval [0, 1]
        :type sentence_dampen: float

    """
    @staticmethod
    def _sigmoid_trans(x: float, y_trans: float = 0,
                       x_trans: float = 0, compress: float = 1) -> float:
        """Return the output of a translated and compressed sigmoid function.

        :param x: the function input independent variable

        :param y_trans: the Y-axis translation scalar

        :param x_trans: the X-axis translation scalar

        :param compress: how much to compress or "squeeze" the function to
                         provide a faster transition

        """
        return (1 / (1 + math.exp((-x + x_trans) * compress))) + y_trans

    @staticmethod
    def _exp(x: np.ndarray, s: float):
        """Compute and return a vector component wise power ``X^s`.`"""
        return x ** s

    def _cosine_sim(self, srcv: Tensor, smyv: Tensor,
                    scale_key: str = None) -> float:
        """Compute the cosine similarity between two embedding vectors."""
        # OOV words have 0 vectors
        x: float = cosine_similarity(srcv, smyv, dim=0).item()
        x = max(0, x)
        if scale_key is not None:
            scales: Dict[str, float] = self.hyp.similarity_dampen_exp
            scale: float = scales[scale_key]
            if scale != 1:
                x = self._exp(x, scale)
        return x

    def _log_embedding(self, obj_name: str, *objs):
        """Convenience method to log the sentence that was used to generate an
        embedding.

        :see: :meth:`~zensols.propbankdb.embpop.EmbeddingPopulator.get_sentence`

        """
        if logger.isEnabledFor(logging.DEBUG):
            esent = self.embedding_populator.get_sentence(*objs)
            objs = objs[0] if len(objs) == 1 else objs
            logger.debug(f'embedding from {obj_name}: {objs} -> "{esent}"')

    def _get_kth_order_neighs(self, v: Vertex) -> Iterable[Sequence[GraphNode]]:
        """Get the the set of nodes that are at exactly $k$ hops from ``v``.
        Nodes at $k$ hops distance are returned for the $k$th iteration.

        :param v: the vertex to start the traversal

        :return: an iterable, each having the "ring" of nodes at distance $N$
                 from ``v``

        """
        # translate the human readable config to igraph parlance
        neigh_mode: str = {
            'descendant': 'in',
            'ancestor': 'out',
            'all': 'all'
        }[self.hyp.neighbor_direction]
        g: Graph = self._ctx.dg.graph
        prev = set()
        for desc_level in it.count():
            desc_verts: Set[Vertex] = set(
                g.neighborhood(v.index, order=desc_level, mode=neigh_mode))
            desc_ring: Set[Vertex] = desc_verts - prev
            if len(desc_ring) == 0:
                break
            # skip the node we're on and proceed to neighbors, then
            # neighbors of those neighbors
            else:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f'level: {desc_level}, desc ring: {desc_ring}')
                yield g.vs[desc_ring][GraphComponent.GRAPH_ATTRIB_NAME]
            prev.update(desc_verts)

    def _roles_embedding(self, v: Vertex, n: ConceptGraphNode) -> \
            Optional[Tensor]:
        """Create an embedding from the roles associated with a concept node.
        It does this by combining the frame set role (think role metadata) and
        the function embedding.

        :param v: the vertex that points to ``n``

        :param n: the concept node that have the role set and roles used to
                  compute the embedding

        :return: the embedding if the concept node has any roles

        """
        g: Graph = self._ctx.dg.graph
        roleset: Roleset = n.roleset
        # hyperparams
        weights: Dict[str, float] = self.hyp.concept_embedding_role_weights
        # role embeddings to collect
        e_embs: Tensor = []
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(n)
        e: Edge
        # iterate over the role edges, which in the reverse graph, the edge
        # direction go into the parent node
        for e in v.incident('in'):
            # robustly get the other end of the edge, which is the child node
            # that is the argument of the role of the concept
            neigh_eix: int = e.source if e.source != e.index else e.target
            neigh: GraphNode = GraphComponent.to_node(g.vs[neigh_eix])
            ge: GraphEdge = GraphComponent.to_edge(e)
            role: Role = ge.role
            # the role's index is the N in `:ARGN`
            role_ix: int = role.index
            role_emb: Tensor = None
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'neighbor: {neigh}')
            # add role edge embedding and role function embedding
            if role_ix is None or role_ix > len(roleset.roles) - 1:
                # if this santity check on propbank metadata and annotated role
                # number fails, use the role's edge embedding, which is computed
                # from role relation with EmbeddingResource.get_role_embedding
                role_emb = ge.embedding
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f'no role for {role} in {n} (idx={role_ix})')
            else:
                # if the sanity check passes and we found a propbank role
                # metadata, then add the role edge and function embeddings
                frole: FramesetRole = roleset.roles[role_ix]
                frole_emb: Tensor = frole.embedding
                func_emb: Tensor = frole.function.embedding
                self._log_embedding('frame role', roleset, frole)
                self._log_embedding('funcion', frole.function)
                if func_emb is None:
                    logger.warning(f'no embedding for function: {frole}')
                else:
                    role_emb = frole_emb + func_emb
            # add the role child node's weighted embedding
            neigh_emb: Tensor = weights['neighbor'] * neigh.embedding
            if role_emb is None:
                # of no propbank role was found, just add the role child node's
                # embedding
                e_embs.append(neigh_emb)
            else:
                # otherwise, add the role child node's weighted embedding
                e_embs.append(neigh_emb + (weights['role'] * role_emb))
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('_' * 40)
        # return the sum of the embeddings of the roles if any exist
        if len(e_embs) > 0:
            return torch.sum(torch.stack(e_embs), dim=0)

    def _concept_embedding(self, v: Vertex, n: ConceptGraphNode) -> Tensor:
        """Return the embedding of a concept as a weighted average of the
        concept's aligned tokens, the role set (if it has/belongs to one), and
        the roles (if it has any).  Each of these have their own respective
        embeddings and a weighted mean in
        :obj:`hyp`.``concept_embedding_weights``.

        """
        embs: Dict[int, Tensor] = self._ctx.concept_emb
        emb: Tensor = embs.get(n.id)
        if emb is None:
            weights: Dict[str, float] = self.hyp.concept_embedding_weights
            # embeddings from the aligned sentence tokens (avoid using role set
            # (id) twice when token alignments are missing)
            tok_emb: Tensor = n.embedding if n.has_token_embedding else None
            # computed from role set with
            # GraphAttributeFeatureVectorizerManager.get_roleset_embedding
            roleset_emb: Tensor = n.roleset_embedding
            weight_embs: List[Tuple[float, Tensor]] = []
            if tok_emb is not None:
                weight_embs.append((weights['token'], tok_emb))
            if roleset_emb is not None:
                weight_embs.append((weights['roleset'], roleset_emb))
            # only get role embeddings for verb nodes as some concept nodes
            # represent nounds or abstract meaning
            if n.has_roleset:
                assert roleset_emb is not None
                self._log_embedding('role set', n.roleset)
                roles_emb: Tensor = self._roles_embedding(v, n)
                if roles_emb is not None:
                    weight_embs.append((weights['roles'], roles_emb))
            emb_ts: Tuple[Tensor] = map(lambda we: we[0] * we[1], weight_embs)
            emb: Tensor = torch.sum(torch.stack(tuple(emb_ts)), dim=0)
            embs[n.id] = emb
        return emb

    def _get_neighbor_embedding(self, v: Vertex) -> Tensor:
        """Return the embedding around a neighboorhood.  This computes a
        weighted mean using the weights defined in
        :obj:`hyp`.``neighbor_embedding_weights``.  This currently works by
        setting the first weight component to 0 to not consider the node at
        ``v`` to later in :meth:`._neighboorhood_sim` to scale the concept
        embedding based on the surrounding neighboord embedding.

        """
        def map_emb(n: GraphNode) -> Tensor:
            if isinstance(n, ConceptGraphNode):
                return self._concept_embedding(v, n)
            else:
                return n.embedding

        neigh_emb: Dict[int, Tensor] = self._ctx.neigh_emb
        emb: Tensor = neigh_emb.get(v)
        if emb is None:
            weights: Tuple[float] = self.hyp.neighbor_embedding_weights
            neigh_embs: List[Tensor] = []
            weight: float
            gns: Sequence[GraphNode]
            for weight, gns in zip(weights, self._get_kth_order_neighs(v)):
                # gns are all the descendents from ``v`` at a given level
                embs: Tuple[Tensor] = tuple(map(map_emb, gns))
                # weighted average across neighbor level
                emb: Tensor = torch.stack(embs, dim=0).mean(dim=0) * weight
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f'weight: {weight}, nodes: {gns}, ' +
                                 f'shape: {emb.shape}')
                neigh_embs.append(emb)
            emb = torch.sum(torch.stack(tuple(neigh_embs), dim=0), dim=0)
            neigh_emb[v] = emb
        return emb

    def _neighboorhood_sim(self, s: GraphNode, m: GraphNode,
                           sv: Vertex, mv: Vertex, embedding_fn: Callable,
                           scale_key: str = None) -> float:
        """Compute the similarity between two nodes in their neighboorhoods.

        :param s: the graph node identified by ``sv``

        :param m: the graph node identified by ``mv``

        :param sv: the first (source but doesn't matter) vertex of the parent's
                   edge embedding

        :param mv: like ``sv`` but the node from the summary's component

        :param embedding_fn: a function that returns the embedding by providing
                             the vertex and graph node (in that order)

        :param scale_key: the key having a exponent scale value used in
                          :meth:`._cosine_sim`

        :return: the cosine similarity of the nodes and neighboorhood

        """
        neigh_sim: Dict[Tuple[int, int, str], float] = self._ctx.neigh_sim
        sim: float = neigh_sim.get((sv, mv, scale_key))
        if sim is None:
            s_neigh_emb: Tensor = self._get_neighbor_embedding(sv)
            m_neigh_emb: Tensor = self._get_neighbor_embedding(mv)
            s_node_emb: Tensor = embedding_fn(sv, s)
            m_node_emb: Tensor = embedding_fn(mv, m)
            neigh_sim_val: float = self._cosine_sim(
                s_neigh_emb, m_neigh_emb, 'neighborhood')
            node_sim: float = self._cosine_sim(
                s_node_emb, m_node_emb, scale_key)
            neigh_skew: float = self._sigmoid_trans(
                neigh_sim_val, **self.hyp.neighbor_skew)
            node_skew: float = node_sim + neigh_skew
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f's={s}, m={m}, node_sim={node_sim:.4f}, ' +
                             f'neigh_sim_val={neigh_sim_val:.4f}, ' +
                             f'neigh_skew={neigh_skew:.4f}, ' +
                             f'node_skew={node_skew:.4f}')
            sim = min(max(node_skew, 0), 1)
            neigh_sim[(sv, mv, scale_key)] = sim
        return sim

    def _concept_sim(self, s: ConceptGraphNode, m: ConceptGraphNode,
                     sv: Vertex, mv: Vertex) -> float:
        """Compute the similarity between two concept nodes."""
        return self._neighboorhood_sim(
            s, m, sv, mv,
            embedding_fn=self._concept_embedding,
            scale_key='concept')

    def _sent_sim(self):
        """Create the sentence embeddings."""
        n2v: Dict[GraphNode, Vertex] = self._ctx.n2v
        embs: Dict[Tuple[int, int], Tuple[Tensor, float]] = self._ctx.sent_emb
        subgraphs: Dict[int, int] = self._ctx.sent_subgraphs
        dg: DocumentGraph = self._ctx.dg
        g: Graph = dg.graph
        node_fac = _BipartiteNodeFactory(connectables=(SentenceGraphNode,))
        pairs: Iterable[Tuple[GraphNode, GraphNode]] = node_fac(dg)
        s: SentenceGraphNode
        m: SentenceGraphNode
        for s, m in pairs:
            sv: Vertex = n2v[s]
            mv: Vertex = n2v[m]
            assert isinstance(s, SentenceGraphNode)
            assert isinstance(m, SentenceGraphNode)
            sim: float = self._cosine_sim(s.embedding, m.embedding, 'sentence')
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'sent sim={sim}: <{s.sent.text}> ' +
                             f'<{m.sent.text}>')
            slope: float = self.hyp.sentence_dampen
            sent_skew: float = (sim * slope) + (1 - slope)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'{s.sent}<->{m.sent}: s={sim:.3f}, ' +
                             f'sk={sent_skew:.3f}, ({sv.index}, {mv.index})')
            for sent_root in (sv, mv):
                if sent_root not in subgraphs:
                    # assume reverse flow
                    mems: List[int] = g.bfs(sent_root.index, 'in')[0]
                    subgraphs.update({v: sent_root.index for v in mems})
            embs[(sv.index, mv.index)] = (sim, sent_skew)

    def _dampen_sent(self, s: GraphNode, m: GraphNode,
                     sv: Vertex, mv: Vertex, sim: float) -> float:
        """Apply the sentence dampening measure to a concept or attribute node
        pair.

        """
        s_embs: Dict[Tuple[int, int], Tuple[Tensor, float]] = self._ctx.sent_emb
        s_subgraphs: Dict[int, int] = self._ctx.sent_subgraphs
        # get the sentence ID for the concept/attribute by ID
        sent_sv: int = s_subgraphs.get(sv.index)
        if sent_sv is not None:
            # if the source sent entry exists, get the summary sent index
            sent_mv: int = s_subgraphs[mv.index]
            # get the sentence skew scalar, and scale the similarity
            skew: float = s_embs[(sent_sv, sent_mv)][1]
            sim *= skew
        return sim

    def _sim(self, tups: Iterable[Tuple[GraphNode, GraphNode]],
             rows: List[Tuple]):
        """Compute the similarity between two graph nodes.

        :param tups: a tuple of ``(<source>, <summary>)``

        :param rows: a row that will become a Pandas dataframe of the cartesian
                     product of graph alignment capacities

        """
        def tup_cmp(an: Tuple[GraphNode, GraphNode],
                    bn: Tuple[GraphNode, GraphNode]) -> int:
            an: GraphNode = an[0]
            bn: GraphNode = bn[0]
            a: Type[GraphNode] = type(an)
            b: Type[GraphNode] = type(bn)
            if a != b:
                for c in node_order:
                    if a == c:
                        return -1
                    if b == c:
                        return 1
            return an.id < bn.id

        # node to vertex map and sentence embeddings created in :meth:`sent_sim`
        n2v: Dict[GraphNode, Vertex] = self._ctx.n2v
        s_embs: Dict[Tuple[int, int], Tuple[Tensor, float]] = self._ctx.sent_emb

        # order nodes by class type
        node_order: Tuple[Type[GraphNode]] = (
            DocumentGraphNode, SentenceGraphNode,
            AttributeGraphNode, ConceptGraphNode)
        tups = sorted(tups, key=cmp_to_key(tup_cmp))

        # iterate over each aligned node pair
        i: int
        s: GraphNode
        m: GraphNode
        for i, (s, m) in enumerate(tups):
            sv: Vertex = n2v[s]
            mv: Vertex = n2v[m]
            sim: float
            if isinstance(s, ConceptGraphNode):
                # concept similarity
                sim = self._concept_sim(s, m, sv, mv)
            elif isinstance(s, AttributeGraphNode):
                # attrib similarity
                sim = self._neighboorhood_sim(
                    s, m, sv, mv,
                    embedding_fn=lambda v, n: n.embedding,
                    scale_key='attribute')
            elif isinstance(s, DocumentGraphNode):
                # doc similarity
                sim = self._cosine_sim(s.embedding, m.embedding)
            elif isinstance(s, SentenceGraphNode):
                # sentence similarity
                sim = s_embs[(sv.index, mv.index)][0]
            else:
                raise ComponentAlignmentError(f'Unknown graph node: {type(s)}')
            if isinstance(s, (ConceptGraphNode, AttributeGraphNode)):
                # dampen descendants of sentence nodes by their sentence skew
                sim = self._dampen_sent(s, m, sv, mv, sim)
            row = [sv.index, mv.index, sim, s.attrib_type, m.attrib_type]
            if self.include_tokens:
                row.extend((str(s), str(m)))
            rows.append(row)
            if self.clear_align_node_iteration:
                self._ctx.clear()

    def __call__(self, doc_graph: DocumentGraph) -> pd.DataFrame:
        """Create the component alignment capacities as connection similarity
        distances as a Pandas dataframe between the source and summary AMR
        nodes.

        :param doc_graph: the graph used to compute capacities

        :return: a Pandas dataframe of the cartesian product of graph alignment
                 capacities

        """
        # get summary and source nodes across both respective components of the
        # graph
        node_fac = _BipartiteNodeFactory()
        pairs: Iterable[Tuple[GraphNode, GraphNode]] = node_fac(doc_graph)
        columns: List[str] = 'src_ix smy_ix sim src_type smy_type'.split()
        n2v: Dict[GraphNode, Vertex] = doc_graph.node_to_vertex
        rows: List[Tuple] = []
        self._ctx = _Context(doc_graph, n2v)
        try:
            # preemptively create the sentence embeddings
            with time('computed sentence capacities'):
                self._sent_sim()
            with time('computed pair capacities'):
                self._sim(pairs, rows)
        finally:
            del self._ctx
        if self.include_tokens:
            columns.extend('src_tok smy_tok'.split())
        return pd.DataFrame(rows, columns=columns)
