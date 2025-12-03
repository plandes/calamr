"""Classes that aid in creating and aligning documents without a corpus.

"""
from __future__ import annotations
from typing import Dict, Tuple, List, Iterable, Sequence, Any, ClassVar
from dataclasses import dataclass, field
import logging
import os
from pathlib import Path
import shutil
import pickle
from zensols.util import APIError, Hasher
from zensols.config import Dictable, Serializer, ConfigFactory, Configurable
from zensols.persist import (
    PersistedWork, PrimeableStash, DelegateStash, ReadOnlyStash, Stash
)
from zensols.amr import AmrError, AmrDocument, AmrFeatureDocument
from zensols.amr.annotate import AnnotatedAmrFeatureDocumentFactory
from zensols.amr import AmrSentence

logger = logging.getLogger(__name__)


@dataclass
class ConfigSwapper(Dictable):
    """Swap file system paths and
    :class:`~zensols.persist.annotation.PersistedWork` instances.  This used to
    temporarily cache files for :class:`.AdhocAnnotatedAmrDocumentStash`.  All
    specified paths in :obj:`swap` are "redirected" to directories (with the
    same names as swapped path) stemming from parent/ancestory directory
    :obj:`root_dir`.

    First :meth:`swap` is called to replace data.  Then :meth:`restore` is
    called to restore the values of all data before :meth:`swap` was called.

    """
    config_factory: ConfigFactory = field()
    """Used to retrieve instances that will have data swapped."""

    root_dir: Path = field()
    """The base directory to create cached files (see class docs)."""

    swaps: Sequence[Tuple[str, str, str]] = field()
    """Target factory objects to swap data (see class docs).  Each is a tuple
    of: ``(config section, attribute, <path|persist>).`` The third string tells
    whether to treat the attribute as a path or
    :class:`~zensols.persist.annotation.PersistedWork`.  If the former, replace
    data will have a new path that starts at :obj:`root_dir`.  If the latter, a
    new uninitialized persisted work is swapped in.

    """
    def __post_init__(self):
        self._prev: Dict[Tuple[str, str], Tuple[Any, Any]] = {}

    def swap(self, replacements: Dict[str, Any] = None):
        """Swap in new temporary data specified by :obj:`swaps`.

        :param replacements: keys are ``(config section, attribute)`` and values
                             are what to swap in as the data

        """
        replacements = {} if replacements is None else replacements
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'swap in: {self.root_dir}')
        config: Configurable = self.config_factory.config
        ser: Serializer = self.config_factory.config.serializer
        sec: str
        attr: str
        swap_type: str
        replace: Any
        to_set: str
        for sec, attr, swap_type, replace, to_set in self.swaps:
            inst: Any = self.config_factory(sec)
            prev: Any = getattr(inst, attr)
            key: Tuple[str, str] = (sec, attr)
            repl: Any = replacements.get(key, replace)
            new_conf: str = None
            prev_conf: str = None
            new: Any
            if swap_type == 'identity':
                new = prev
                new_conf = ser.format_option(new)
            elif swap_type == 'value':
                new = repl
                new_conf = ser.format_option(new)
            elif swap_type == 'path':
                new = self.root_dir / prev.name if repl is None else repl
                new_conf = ser.format_option(new)
            elif swap_type == 'persist':
                if repl is None:
                    new = PersistedWork(attr, prev.owner)
                else:
                    new = PersistedWork(attr, prev.owner, initial_value=repl)
            elif swap_type == 'instance':
                new = self.config_factory(repl)
                new_conf = f'instance: {repl}'
            else:
                raise APIError(f'Unknown swap type: {swap_type}')
            if new_conf is not None and \
               to_set in {'both', 'config'} and\
               config.has_option(attr, sec):
                # some values are set on object instances that have no
                # configuration, such as those with default values
                prev_conf = config.get_option(attr, sec)
            self._prev[key] = (prev, new, prev_conf)
            if to_set in {'both', 'attr'}:
                setattr(inst, attr, new)
            if new_conf is not None:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f'set config: {sec}:{attr} -> {new_conf}')
                config.set_option(attr, new_conf, sec)

    def restore(self):
        """Restore all data replaced by :meth:`swap`."""
        config: Configurable = self.config_factory.config
        #ser: Serializer = self.config_factory.config.serializer
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'swap out: {self.root_dir}')
        sec: str
        attr: str
        prev: Any
        new: Any
        prev_conf: Any
        for (sec, attr), (prev, new, prev_conf) in self._prev.items():
            inst: Any = self.config_factory(sec)
            setattr(inst, attr, prev)
            if prev_conf is not None:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f'restore config: {sec}:{attr} -> {prev_conf}')
                config.set_option(attr, prev_conf, sec)
        self._prev.clear()

    def clear(self):
        """Recursively remove all data in :obj:`root_dir`."""
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'swap clear: {self.root_dir}')
        if self.root_dir.is_dir():
            shutil.rmtree(self.root_dir)

    def __getitem__(self, key: Tuple[str, str]) -> Tuple[Any, Any]:
        return self._prev[key]


@dataclass
class AdhocAmrDocumentPartStash(ReadOnlyStash):
    """A factory stash that creates :class:`~zensols.amr.doc.AmrDocument`
    instances for each entry in a :class:`typing.Dict` in :obj:`corpus`.  This
    is used by :class:`.AdhocAnnotatedAmrDocumentStash` to roll all individual
    documents in a single document.

    """
    anon_doc_factory: AnnotatedAmrFeatureDocumentFactory = field()
    """Parses text data into AMR source and summary documents."""

    corpus: Dict[str, Dict[str, str]] = field(default=None)
    """Documents as sentences given with document ID mapped to
    :class:`typing.Dict` as described in the ``corpus`` parameter in
    :meth:`.AdhocAnnotatedAmrDocumentStash.set_corpus`.

    """
    def load(self, did: str) -> AmrDocument:
        id_field: str = AdhocAnnotatedAmrDocumentStash.ID_FIELD
        doc: Dict[str, str] = self.corpus.get(did)
        if doc is not None:
            try:
                fdoc: AmrFeatureDocument = self.anon_doc_factory(doc)
                return fdoc.amr
            except AmrError as e:
                did: int = doc.get(id_field, did)
                sent = AmrSentence(e.to_failure())
                sent.set_metadata(id_field, f"{did}.0")
                return AmrDocument((sent,))

    def keys(self) -> Iterable[str]:
        return self.corpus.keys()

    def exists(self, did: str) -> bool:
        return did in self.corpus


@dataclass
class AdhocAnnotatedAmrDocumentStash(PrimeableStash, DelegateStash):
    """A stash that generates and cachees instances of
    :class:`~zensols.amr.doc.AmrDocument`.  This rolls sentences from all
    documents given to :meth:`set_corpus` into a single document, which is then
    used to create annotated AMR feature sentences.

    """
    ID_FIELD: ClassVar[str] = 'id'

    hasher: Hasher = field()
    """Used to create unique file cache root directories."""

    doc_part_stash: Stash = field()
    """Leverages :class:`.AdhocAmrDocumentPartStash` (in some object graph) go
    create :class:`~zensols.amr.doc.AmrDocument` instances.  These, in turn, are
    used to create one document from all the sentences.

    """
    root_dir: Path = field()
    """The directory base where cached files are stored."""

    swap_keys: Dict[str, List[str]] = field()
    """Indicates the ``(config section, attribute)`` to locate the persisted
    work to set with the cached document so the factory stash doesn't.

    """
    swapper: ConfigSwapper = field()
    """Used to swap in the adhoc paths and document and then back out."""

    _cache_dir: Path = field(default=None)
    """Set by :meth:`.get_cache_dir` and left as an initializer so it can be
    set by the config factory in child process workers (see ``adhoc.yml``).

    """
    def __post_init__(self):
        self.swap_keys = dict(map(
            lambda k: (k[0], (k[1][0], k[1][1])),
            self.swap_keys.items()))
        self._reset()

    def _reset(self):
        #self._cache_dir: Path = None
        self._corpus: Sequence[Dict[str, str]] = None
        self._created_docs: bool = False

    def _get_doc(self, corpus: Dict[str, Dict[str, str]]) -> AmrDocument:
        """A document containing all the sentences from the corpus.

        :see: :obj:`~zensols.amr.annotate.AnnotatedAmrDocumentStash.corpus_doc`

        """
        sents: List[AmrSentence] = []
        id_field: str = self.ID_FIELD
        did: str
        for did in map(lambda d: d[id_field], self._corpus):
            doc: AmrDocument = self.doc_part_stash[did]
            sents.extend(doc.sents)
        return AmrDocument.to_document(sents)

    def _get_cache_dir(self, corpus_id: str) -> Path:
        if corpus_id is None:
            self.hasher.reset()
            self.hasher.update(self._corpus)
            corpus_id = self.hasher()
        return self.root_dir / corpus_id

    def set_corpus(self, corpus: Sequence[Dict[str, str]],
                   corpus_id: str = None):
        """Set the corpus documents that will be used for parsing and
        annotating.  The data (``corpus``) will immediately be parsed into AMRs
        in this call and the data that writes to the file system will be updated
        to point to a new ``.../adhoc`` directory to not interfere with any
        corpus documents (see :class:`.ConfigSwapper`).

        To restore the configuration after adhoc document processing is
        finished, call :meth:`restore`.

        :param corpus: the AMR summary documents, which is usually a sequence of
                       :class:`~typing.Dict` instances (see
                       :class:`~zensols.arm.annotate.AnnotatedAmrFeatureDocumentFactory`
                       for data structure details)

        :param corpus_id: a unique identifier for ``data``, or ``None`` to use a
                          hashed string, which in turn, is used as the directory
                          name for the cached data

        """
        self._corpus = corpus
        self._cache_dir: Path = self._get_cache_dir(corpus_id)
        self.swapper.root_dir = self._cache_dir
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'set corpus: {self._cache_dir}')

    def _create_docs(self):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'creating docs: cache_dir={self._cache_dir}, ' +
                         f'swap dir: {self.swapper.root_dir}')
        if self._cache_dir is None:
            pid: int = os.getpid()
            raise APIError(f'Must first call set_corpus before access in {pid}')
        amr_doc_file: Path = self._cache_dir / 'amr-docs.dat'
        need_create: bool = not amr_doc_file.exists()
        id_field: str = self.ID_FIELD
        corpus: Dict[str, Dict[str, str]] = None
        doc: AmrDocument
        if self._corpus is not None:
            corpus = dict(map(lambda d: (d[id_field], d), self._corpus))
            self.swapper.swap({self.swap_keys['corpus']: corpus})
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'need create: {amr_doc_file} -> {need_create}')
        if need_create:
            doc = self._get_doc(corpus)
            amr_doc_file.parent.mkdir(parents=True, exist_ok=True)
            with open(amr_doc_file, 'wb') as f:
                pickle.dump(doc, f)
            if logger.isEnabledFor(logging.INFO):
                logger.info(f'wrote adhoc parsed amrs: {amr_doc_file}')
        else:
            with open(amr_doc_file, 'rb') as f:
                doc = pickle.load(f)
            if logger.isEnabledFor(logging.INFO):
                logger.info(f'restored adhoc parsed amrs: {amr_doc_file}')
        doc_key: Tuple[str, str] = self.swap_keys['doc']
        pw_inst: Any = self.swapper.config_factory(doc_key[0])
        #pw_current_corp_doc: PersistedWork = self.swapper[doc_key][1]
        pw_current_corp_doc: PersistedWork = getattr(pw_inst, doc_key[1])
        pw_current_corp_doc.set(doc)
        self._created_docs = True

    def prime(self):
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'priming {type(self)}...')
        if not self._created_docs:
            self._create_docs()
        super().prime()

    def restore(self):
        """Restores the configuration that writes to the file system after the
        call to :meth:`set_corpus`.

        """
        logger.debug('restore')
        self.swapper.restore()
        self._reset()

    def clear(self):
        """Recursively remove all data in :obj:`root_dir`."""
        logger.debug('clear')
        if self._cache_dir is not None:
            self.swapper.clear()
