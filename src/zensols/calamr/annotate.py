"""Contain a class to add embeddings to AMR feature documents.

"""
__author__ = 'Paul Landes'
from typing import Dict, Tuple, Any
from dataclasses import dataclass, field
import logging
from zensols.util import time
from zensols.persist import DelegateStash, PrimeableStash
from zensols.amr import AmrFailure, AmrSentence, AmrFeatureDocument
from zensols.amr.annotate import (
    AnnotatedAmrFeatureDocumentFactory,
    AnnotatedAmrDocument, AnnotatedAmrSectionDocument,
)
from zensols.deepnlp.transformer import (
    WordPieceFeatureDocumentFactory, WordPieceFeatureDocument
)

logger = logging.getLogger(__name__)


@dataclass
class _EmbeddingPopulator(object):
    """Base class for adding word piece doc embeddings.

    """
    word_piece_doc_factory: WordPieceFeatureDocumentFactory = field()
    """The feature document factory that populates embeddings."""

    def _populate_embeddings(self, doc: AmrFeatureDocument, doc_id: str = None):
        """Adds the transformer sentinel embeddings to the document."""
        doc_id = doc.amr.get_doc_id() if doc_id is None else doc_id
        if self.word_piece_doc_factory is not None:
            try:
                with time(f'populated embedding of document {doc_id}'):
                    wpdoc: WordPieceFeatureDocument = \
                        self.word_piece_doc_factory(doc)
                wpdoc.copy_embedding(doc)
            except Exception as e:
                msg: str = f"Could not process adhoc document: '{doc_id}': {e}"
                sent: AmrSentence
                for sent in doc.amr.sents:
                    fail = AmrFailure(exception=e, thrower=self, message=msg)
                    sent.failure = fail


@dataclass
class AddEmbeddingsFeatureDocumentStash(DelegateStash, PrimeableStash,
                                        _EmbeddingPopulator):
    """Add embeddings to AMR feature documents.  Embedding population is
    disabled by configuring :obj:`word_piece_doc_factory` as ``None``.

    """
    def load(self, name: str) -> AmrFeatureDocument:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"loading '{name}'")
        doc: AmrFeatureDocument = self.delegate.load(name)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"loaded '{name}' -> {doc}")
        self._populate_embeddings(doc, name)
        return doc

    def get(self, name: str, default: Any = None) -> Any:
        item = self.load(name)
        item = default if item is None else item
        return item


@dataclass
class CalamrAnnotatedAmrFeatureDocumentFactory(
        AnnotatedAmrFeatureDocumentFactory, _EmbeddingPopulator):
    """Adds wordpiece embeddings to
    :class:`~zensols.amr.container.AmrFeatureDocument` instances.

    """
    def to_annotated_doc(self, doc: AmrFeatureDocument) -> AmrFeatureDocument:
        fdoc = super().to_annotated_doc(doc)
        self._populate_embeddings(fdoc)
        return fdoc

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
