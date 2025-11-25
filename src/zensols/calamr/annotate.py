"""Contain a class to add embeddings to AMR feature documents.

"""
__author__ = 'Paul Landes'

from typing import Dict, Tuple, List, Iterable, Sequence, Any, Union
from dataclasses import dataclass, field
import logging
from itertools import chain
from pathlib import Path
from zensols.util import time
from zensols.persist import (
    persisted, DelegateStash, PrimeableStash,
    PersistedWork, DictionaryStash, ReadOnlyStash
)
from zensols.config import ConfigFactory
from zensols.install import Installer
from zensols.amr import (
    AmrFailure, AmrSentence, AmrDocument, AmrFeatureDocument
)
from zensols.amr.annotate import (
    AnnotatedAmrFeatureDocumentFactory,
    AnnotatedAmrDocument, AnnotatedAmrSectionDocument,
    AnnotatedAmrDocumentStash
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


@dataclass
class CalamrAnnotatedAmrDocumentStash(ReadOnlyStash):
    config_factory: ConfigFactory = field(default=None)
    anon_doc_stash: AnnotatedAmrDocumentStash = field(default=None)
    anon_doc_factory: AnnotatedAmrFeatureDocumentFactory = field(default=None)
    stash_writers: Sequence[Tuple[str, str]] = field(default=None)

    def _replace_persists(self, doc):
        self._corpus_doc = PersistedWork('__corpus_doc', self, initial_value=doc)
        self._corpus_df = PersistedWork('__corpus_df', self)
        self.anon_doc_stash._corpus_doc = self._corpus_doc
        self.anon_doc_stash._corpus_df = self._corpus_df

    def _replace_caching_stashes(self):
        replace_stash = DictionaryStash()
        sec: str
        attr: str
        for sec, attr in self.stash_writers:
            obj = self.config_factory(sec)
            print(f'setting {type(obj)}: {attr} -> {type(replace_stash)}')
            setattr(obj, attr, replace_stash)

    def set_corpus(self, data: Union[Path, Dict, Sequence]):
        self._replace_caching_stashes()
        docs: Tuple[AmrFeatureDocument, ...] = \
            tuple(self.anon_doc_factory(data))
        sents: Iterable[AmrSentence] = map(
            lambda s: s.amr,
            chain.from_iterable(map(lambda d: d.sents, docs)))
        doc: AmrDocument = AmrDocument.to_document(sents)
        #self._set_doc(doc)
        self._replace_persists(doc)

    def load(self, doc_id: str) -> AnnotatedAmrDocument:
        return self.anon_doc_stash.load(doc_id)

    def keys(self) -> Iterable[str]:
        return self.anon_doc_stash.keys()

    def exists(self, doc_id: str) -> bool:
        return self.anon_doc_stash.exists(doc_id)
