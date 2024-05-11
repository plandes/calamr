"""Contain a class to add embeddings to AMR feature documents.

"""
__author__ = 'Paul Landes'

from typing import Dict, Tuple, Any
from dataclasses import dataclass, field
import logging
from zensols.util import time
from zensols.persist import DelegateStash, PrimeableStash
from zensols.amr import AmrSentence, AmrFeatureDocument
from zensols.amr.annotate import (
    AnnotatedAmrFeatureDocumentFactory,
    AnnotatedAmrDocument, AnnotatedAmrSectionDocument
)
from zensols.deepnlp.transformer import (
    WordPieceFeatureDocumentFactory, WordPieceFeatureDocument
)

logger = logging.getLogger(__name__)


@dataclass
class AddEmbeddingsFeatureDocumentStash(DelegateStash, PrimeableStash):
    """Add embeddings to AMR feature documents.  Embedding population is
    disabled by configuring :obj:`word_piece_doc_factory` as ``None``.

    """
    word_piece_doc_factory: WordPieceFeatureDocumentFactory = field(
        default=None)
    """The feature document factory that populates embeddings."""

    def load(self, name: str) -> AmrFeatureDocument:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"loading '{name}'")
        doc: AmrFeatureDocument = self.delegate.load(name)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"loaded '{name}' -> {doc}")
        if self.word_piece_doc_factory is not None:
            with time(f'populated embedding of document {name}'):
                wpdoc: WordPieceFeatureDocument = \
                    self.word_piece_doc_factory(doc)
            wpdoc.copy_embedding(doc)
        return doc

    def get(self, name: str, default: Any = None) -> Any:
        item = self.load(name)
        item = default if item is None else item
        return item


@dataclass
class CalamrAnnotatedAmrFeatureDocumentFactory(
        AnnotatedAmrFeatureDocumentFactory):
    """Adds wordpiece embeddings to
    :class:`~zensols.amr.container.AmrFeatureDocument` instances.

    """
    word_piece_doc_factory: WordPieceFeatureDocumentFactory = field(
        default=None)
    """The feature document factory that populates embeddings."""

    def _populate_embeddings(self, doc: AmrFeatureDocument):
        """Adds the transformer sentinel embeddings to the document."""
        if self.word_piece_doc_factory is not None:
            wpdoc: WordPieceFeatureDocument = self.word_piece_doc_factory(doc)
            wpdoc.copy_embedding(doc)

    def from_dict(self, data: Dict[str, str]) -> AmrFeatureDocument:
        fdoc = super().from_dict(data)
        if self.word_piece_doc_factory is not None:
            self._populate_embeddings(fdoc)
        return fdoc


@dataclass
class ProxyReportAnnotatedAmrDocument(AnnotatedAmrDocument):
    """Overrides the sections property to skip duplicate summary sentences also
    found in the body.

    """
    @property
    def sections(self) -> Tuple[AnnotatedAmrSectionDocument]:
        """The sentences that make up the body of the document."""
        def filter_sents(s: AmrSentence) -> bool:
            return s.text not in sum_sents

        sum_sents = set(map(lambda s: s.text, self.summary))
        secs = super().sections
        for sec in secs:
            sec.sents = tuple(filter(filter_sents, sec.sents))
        return secs
