"""Graph node and edge domain classes.

**Terminology**: *token alignments* refer to the sentence index based token
alignments to AMR nodes.  This is not to be confused with alignment edges (aka
graph component alignment edges).

"""
from __future__ import annotations
__author__ = 'Paul Landes'
from typing import Tuple, Sequence, ClassVar, Union, Dict, Optional, Set, Any
from dataclasses import dataclass, field
from abc import ABCMeta
import logging
import sys
from itertools import chain
import textwrap
from frozendict import frozendict
from io import TextIOBase
import penman
from penman.graph import Instance, Attribute
from penman.surface import Alignment, RoleAlignment
from torch import Tensor
from zensols.persist import persisted, PersistedWork
from zensols.nlp import FeatureToken
from zensols.amr import Relation, AmrFeatureDocument, AmrFeatureSentence
from zensols.util.sci import ScientificUtil
from zensols.propbankdb import RolesetId, Roleset
from . import Role, GraphAttribute

logger = logging.getLogger(__name__)


@dataclass(eq=False, repr=False)
class GraphNode(GraphAttribute):
    """Graph attribute data added to the :class:`.igraph.Graph` vertexes.

    """
    def __post_init__(self):
        super().__post_init__()
        self._partition = None

    def _get_embedding(self) -> Tensor:
        return self.embedding_resource.unknown_node_embedding

    @property
    def partition(self) -> int:
        return self._partition

    @partition.setter
    def partition(self, partition: int):
        self._partition = partition


@dataclass(eq=False, repr=False)
class DocumentNode(GraphNode):
    """A composite of a node in the document tree that are associated with the
    :class:`~zensols.amr.container.FeatureDocument` as root node.  This class
    represents nodes in a graph that either:

      * make up the part of the graph that's disjoint from the AMR sentinel
        subgraphs (i.e. a root ``doc`` node), or

      * the root to an AMR sentence (see :class:`.AmrDocumentNode`)

    The in-memory object graph of these instances are dependent on the type of
    data it represents.  For example, the *Proxy Report* corpus has a top level
    a summary and body nodes with AMR sentences below (root on top).

    """
    name: str = field()
    """The descriptive name of the node such as ``doc`` or ``section``."""

    root: AmrFeatureDocument = field()
    """The owning feature document containing all sentences/tokens of the graph.

    """
    children: Tuple[DocumentNode, ...] = field()
    """The children of this node with respect to the composite pattern."""

    @property
    def sents(self) -> Tuple[AmrFeatureSentence]:
        """The sentences of the this document level."""
        return self._get_sents()

    @property
    @persisted('_children_by_name', transient=True)
    def children_by_name(self) -> DocumentNode:
        """The children's names as keys and respective document nodes as
        capacitys.

        """
        return frozendict({c.name: c for c in self.children})

    def _get_sents(self) -> Tuple[AmrFeatureSentence, ...]:
        """Implementation dependent method to get the sentences."""
        # this returns no sentences for intermediate nodes (i.e. section)
        return ()

    def _get_by_path(self, path: Sequence[str]) -> DocumentNode:
        """Get a document node in the non-AMR in the tree portion of the graph.

        :param path: a dot (``.``) separated list of :obj:`name`

        :return: the node found at the end of the path if found

        """
        c = self.children_by_name.get(path[0])
        if c is not None and len(path) > 1:
            c = c._get_by_path(path[1:])
        return c

    def _get_description(self) -> str:
        return '\n'.join(map(lambda s: s.norm, self.sents))

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        sents = self.sents
        children = self.children
        self._write_line(self.name, depth, writer)
        if len(sents) > 0:
            self._write_line('sentences:', depth + 1, writer)
            sent: AmrFeatureSentence
            for sent in sents:
                self._write_line(sent.text, depth + 2, writer, max_len=True)
        if len(children) > 0:
            self._write_iterable(children, depth + 1, writer)

    def __getitem__(self, key) -> DocumentNode:
        """Index by path keys (see :meth:`_get_by_path`)."""
        if isinstance(key, int):
            return self.children[key]
        else:
            return self._get_by_path(key.split('.'))

    def __len__(self) -> int:
        return sum(map(len, self.children)) + len(self.sents)


@dataclass(eq=False, repr=False)
class AmrDocumentNode(DocumentNode):
    """A composite note containing a subset of the :obj:`.DocumentNode.root`
    sentences.  This includes the text, text features, and AMR Penman graph
    data.

    """
    doc: AmrFeatureDocument = field()
    """A document containing a subset of sentences that fall under this portion
    of the graph.

    """
    def _get_sents(self) -> Tuple[AmrFeatureSentence, ...]:
        return self.doc.sents


@dataclass(eq=False, repr=False)
class GraphEdge(GraphAttribute, metaclass=ABCMeta):
    """Graph attriubte data added to the :class:`.igraph.Graph` edges.

    """
    MAX_CAPACITY: ClassVar[float] = {
        0: float('inf'),
        1: sys.float_info.max,
        2: float(10 ** 10)
    }[2]
    """Maximum value a capacity.

    *Implementation note*: It seems :mod:`igraph` can only handle large values
    to represent infinity, and not float ``inf`` or the system defined largest
    float value.

    """
    capacity: float = field(default=0)
    """The capacity of the edge."""

    flow: float = field(default=0)
    """The flow calculated by the maxflow algorithm."""

    def __post_init__(self):
        super().__post_init__()
        self._value_str = PersistedWork('_value_str', self, transient=True)
        self._label = PersistedWork('_label', self, transient=True)

    @property
    def _flow(self) -> float:
        return self._flow_var

    @_flow.setter
    def _flow(self, flow: float):
        self._flow_var = flow
        self._reset_label()

    @property
    def _capacity(self) -> float:
        return self._capacity_var

    @_capacity.setter
    def _capacity(self, capacity: float):
        self._capacity_var = capacity
        self._reset_label()

    def _reset_label(self):
        if hasattr(self, '_value_str'):
            self._description.clear()
            self._value_str.clear()
            self._label.clear()

    def _get_embedding(self) -> Tensor:
        return self.embedding_resource.unknown_edge_embedding

    def _format_val(self, v: float, precision: int = None) -> str:
        if precision is None:
            if self.context is None:
                precision = 4
            else:
                precision = self.context.default_format_strlen
        if self.MAX_CAPACITY == v:
            # unicode infinity symbol
            s = '\u221e'
        else:
            s = ScientificUtil.fixed_format(v, length=precision)
        return s

    def capacity_str(self, precision: int = None) -> str:
        return self._format_val(self.capacity, precision)

    def flow_str(self, precision: int = None) -> str:
        return self._format_val(self.flow, precision)

    @property
    @persisted('_value_str')
    def value_str(self) -> str:
        if self.capacity == 0 and self.flow == 0:
            return ''
        s: str = f'{self.capacity_str()}/{self.flow_str()}'
        if hasattr(self, 'vis_flow'):
            s += f'/{self.vis_flow:.2f}'
        return s

    def _get_label(self) -> str:
        desc: str = self.description
        vs: str = self.value_str
        lab: str
        if len(vs) == 0:
            lab = desc
        elif desc != vs:
            lab = f'{desc}\n({vs})'
        else:
            lab = f'({vs})'
        return lab

    def _get_description(self) -> str:
        return self.value_str

    def __str__(self) -> str:
        return f'{self.capacity_str()}/{self.flow_str()}'


GraphEdge.flow = GraphEdge._flow
GraphEdge.capacity = GraphEdge._capacity


@dataclass(eq=False, repr=False)
class SentenceGraphAttribute(GraphAttribute):
    """A node containing zero or more tokens with its parent sentence.  Usually
    the AMR node represents a single token, but can have more than one
    token alignment.

    """
    _DICTABLE_WRITE_EXCLUDES: ClassVar[str] = \
        GraphAttribute._DICTABLE_WRITE_EXCLUDES | {'token_aligns'}
    _DICTABLE_ATTRIBUTES: ClassVar[str] = GraphAttribute._DICTABLE_ATTRIBUTES

    sent: AmrFeatureSentence = field(repr=False)
    """The sentence from which this node was created."""

    token_aligns: Tuple[Union[Alignment, RoleAlignment], ...] = field()
    """The node to sentinel token index."""

    def _get_embedding(self) -> Tensor:
        if self.has_token_embedding:
            return self.embedding_resource.get_tokens_embedding(self.tokens)
        else:
            return super()._get_embedding()

    @property
    def has_token_embedding(self) -> bool:
        """Whether this attriubte node has token embeddings."""
        return len(self.tokens) > 0

    @property
    @persisted('_tokens', transient=True)
    def tokens(self) -> Tuple[FeatureToken, ...]:
        """The tokens """
        aligns: Dict[Tuple[str, str, str], Tuple[FeatureToken, ...]] = \
            self.sent.alignments
        toks: Optional[Tuple[FeatureToken]] = aligns.get(tuple(self.triple))
        toks = () if toks is None else toks
        return toks

    @property
    def indices(self) -> Tuple[int, ...]:
        """Return the concatenated list of indices of the alginments."""
        return tuple(chain.from_iterable(
            map(lambda a: a.indices, self.token_aligns)))

    @property
    def token_align_str(self) -> str:
        """A string representation of the AMR Penman representation of the
        token alignment.

        """
        return ','.join(map(str, self.token_aligns))

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        astr = self.token_align_str
        super().write(depth, writer)
        if len(astr) > 0:
            self._write_line(f'token aligns: {astr}', depth + 1, writer)


@dataclass(eq=False, repr=False)
class DocumentGraphNode(GraphNode):
    """A node that has data about the non-AMR parts of the graph, such as the
    unifying top level node that ties the sentences together.  However, it can
    contain the root to an AMR sentence (see :class:`.AmrDocumentNode`).

    """
    ATTRIB_TYPE: ClassVar[str] = 'doc'
    """The attribute type this class represents."""

    level_name: str = field()
    """The descriptive name of the node such as ``doc`` or ``section``."""

    doc_node: DocumentNode = field(repr=False)
    """The document node associated with the attached :mod:`igraph` node."""

    def _get_description(self) -> str:
        return f'{self.doc_node.name} ({self.level_name})'

    def _get_label(self) -> str:
        return self.doc_node.name


@dataclass(eq=False, repr=False)
class DocumentGraphEdge(GraphEdge):
    """An edge that has data about the non-AMR parts of the graph, such as
    *sentence*.

    """
    ATTRIB_TYPE: ClassVar[str] = 'doc'
    """The attribute type this class represents."""

    relation: str = field(default='')
    """The edge relation between two document nodes or document to igraph node.

    """
    def _get_description(self) -> str:
        return self.relation

    def __str__(self) -> str:
        vstr: str = self.value_str
        if len(vstr) > 0:
            vstr = f': {vstr}'
        return super().__str__() + vstr


@dataclass(eq=False, repr=False)
class SentenceGraphNode(GraphNode):
    """A graph node containing the root of a sentence.

    """
    SENT_TEXT_LEN: ClassVar[int] = 20
    """The truncated sentence length"""

    ATTRIB_TYPE: ClassVar[str] = 'sentence'
    """The attribute type this class represents."""

    sent: AmrFeatureSentence = field(repr=False)
    """The sentence from which this node was created."""

    sent_ix: int = field()
    """The sentence index."""

    def _get_embedding(self) -> Tensor:
        return self.embedding_resource.get_sentence_tokens_embedding(self.sent)

    def _get_description(self) -> str:
        return textwrap.shorten(self.sent.norm, self.SENT_TEXT_LEN)


@dataclass(eq=False, repr=False)
class TripleGraphNode(SentenceGraphAttribute, GraphNode):
    """Contains a Penman triple with token alignments used for concepts and AMR
    attributes.  Instances of this class get their embedding via
    :meth:`.SentenceGraphAttribute._get_embedding`.

    """
    _DICTABLE_ATTRIBUTES: ClassVar[str] = {'variable'} | \
        GraphNode._DICTABLE_ATTRIBUTES | \
        SentenceGraphAttribute._DICTABLE_ATTRIBUTES

    triple: Union[Instance, Attribute] = field(repr=False)
    """The AMR Penman graph triple."""

    def __post_init__(self):
        super().__post_init__()
        GraphNode.__post_init__(self)

    @property
    def variable(self) -> str:
        """The variable, which comes from the source of the triple, such as
        ``s0``.

        """
        return self.triple.source


@dataclass(eq=False, repr=False)
class ConceptGraphNode(TripleGraphNode):
    """Attribute data from AMR concept nodes grafted on to the
    :class:`igraph.Graph`.

    """
    _DICTABLE_ATTRIBUTES: ClassVar[str] = {'instance'} | \
        TripleGraphNode._DICTABLE_ATTRIBUTES

    ATTRIB_TYPE: ClassVar[str] = 'concept'
    """The attribute type this class represents."""

    def _get_embedding(self) -> Tensor:
        if self.has_token_embedding:
            # get the SentenceGraphAttribute's impl
            return super()._get_embedding()
        else:
            lemma: str = self.roleset_id.lemma
            if lemma is None:
                lemma = self.roleset_id.label
            return self.embedding_resource.get_token_embedding(lemma)

    @property
    @persisted('_roleset_id', transient=True)
    def roleset_id(self) -> RolesetId:
        return RolesetId(self.instance)

    @property
    @persisted('_roleset', transient=True)
    def roleset(self) -> Roleset:
        if self.roleset_id.is_valid:
            return self.embedding_resource.roleset_stash.get(
                str(self.roleset_id))

    @property
    def token_embedding(self) -> Optional[Tensor]:
        if len(self.tokens) > 0:
            return self.embedding

    @property
    def has_roleset(self) -> bool:
        return self.roleset is not None

    @property
    @persisted('_roleset_embedding', transient=True)
    def roleset_embedding(self) -> Tensor:
        return self.embedding_resource.get_roleset_embedding(
            self.roleset_id, self.roleset)

    @property
    def instance(self) -> str:
        """The concept instance, such as the propbank entry (i.e. *see-01*).
        Other examples include nouns.

        """
        return self.triple.target

    def _get_description(self) -> str:
        return f'{self.variable}/{self.instance}'


@dataclass(eq=False, repr=False)
class AttributeGraphNode(TripleGraphNode):
    """Attribute data from AMR attribute nodes grafted on to the
    :class:`igraph.Graph`.

    """
    _PERSITABLE_PROPERTIES: ClassVar[Set[str]] = {'role'}

    _DICTABLE_ATTRIBUTES: ClassVar[Set[str]] = \
        TripleGraphNode._DICTABLE_ATTRIBUTES | {'constant', 'role'}

    ATTRIB_TYPE: ClassVar[str] = 'attribute'
    """The attribute type this class represents."""

    def _get_embedding(self) -> Tensor:
        if self.has_token_embedding:
            # get the SentenceGraphAttribute's impl
            return super()._get_embedding()
        else:
            const: str = str(self.constant)
            return self.embedding_resource.get_token_embedding(const)

    @property
    @persisted('_constant', transient=True)
    def constant(self) -> Any:
        """The constant defined by the attribute from the Penman graph."""
        return penman.constant.evaluate(self.triple.target)

    @property
    @persisted('_role', transient=True)
    def role(self) -> str:
        """The AMR role taken from Penman graph node."""
        return self.context.to_role(self.triple.role)

    def _get_description(self) -> str:
        return str(self.constant)


@dataclass(eq=False, repr=False)
class SentenceGraphEdge(DocumentGraphEdge):
    """An edge from a document node to a :class:`.SentenceGraphNode`.

    """
    ATTRIB_TYPE: ClassVar[str] = 'sentence'
    """The attribute type this class represents."""

    sent: AmrFeatureSentence = field(default=None, repr=False)
    """The sentence from which this node was created."""

    sent_ix: int = field(default=None)
    """The sentence index."""

    def _get_description(self) -> str:
        return f'{self.relation}[{self.sent_ix}]'


@dataclass(eq=False, repr=False)
class RoleGraphEdge(GraphEdge, SentenceGraphAttribute):
    """Attribute data from the AMR role edges grafted on to the
    :class:`igraph.Graph`.

    """
    _DICTABLE_WRITE_EXCLUDES: ClassVar[Set[str]] = {'role'} | \
        GraphNode._DICTABLE_WRITE_EXCLUDES | \
        SentenceGraphAttribute._DICTABLE_WRITE_EXCLUDES

    _DICTABLE_ATTRIBUTES: ClassVar[str] = \
        GraphNode._DICTABLE_ATTRIBUTES | \
        SentenceGraphAttribute._DICTABLE_ATTRIBUTES

    ATTRIB_TYPE: ClassVar[str] = 'role'
    """The attribute type this class represents."""

    triple: Union[Instance, Attribute] = field(default=None, repr=False)
    """The AMR Penman graph triple."""

    role: Union[str, Role] = field(default=None)
    """The role name of the edge such as ``:ARG0``."""

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.role, str):
            self.role = self.context.to_role(self.role)

    def _get_description(self) -> str:
        return self.role.label

    def _get_embedding(self) -> Tensor:
        return self.embedding_resource.get_role_embedding(self.role)


@dataclass(eq=False, repr=False)
class TerminalGraphNode(GraphNode):
    """A flow control: source or sink.

    """
    ATTRIB_TYPE: ClassVar[str] = 'control'
    """The attribute type this class represents."""

    is_source: bool = field()
    """Whether or not this source (``s``) or sink (``t``)."""

    def _get_description(self) -> str:
        return 'S' if self.is_source else 'T'


@dataclass(eq=False, repr=False)
class ComponentAlignmentGraphEdge(GraphEdge):
    """An edge that spans graph components.

    """
    ATTRIB_TYPE: ClassVar[str] = 'component alignment'
    """The attribute type this class represents."""


@dataclass(eq=False, repr=False)
class ComponentCorefAlignmentGraphEdge(ComponentAlignmentGraphEdge):
    """An edge that spans graph components.

    """
    ATTRIB_TYPE: ClassVar[str] = 'component coref alignment'
    """The attribute type this class represents."""

    relation: Relation = field(default=None)
    """The AMR coreference relation between this node and all other refs."""

    is_bipartite: bool = field(default=False)
    """Whether the coreference spans components."""


@dataclass(eq=False, repr=False)
class TerminalGraphEdge(GraphEdge):
    """An edge that connects to terminal a :class:`.TerminalGraphNode`.

    """
    ATTRIB_TYPE: ClassVar[str] = 'terminal'
    """The attribute type this class represents."""
