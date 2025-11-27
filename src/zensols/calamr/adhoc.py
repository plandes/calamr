"""Classes that aid in creating and aligning documents without a corpus.

"""
from __future__ import annotations
from typing import Dict, Tuple, Iterable, Sequence, Union, Any
from dataclasses import dataclass, field
import logging
from itertools import chain
from pathlib import Path
import shutil
import pickle
from zensols.util import APIError, Hasher
from zensols.config import Dictable
from zensols.amr import AmrDocument, AmrFeatureDocument
from zensols.amr.annotate import AnnotatedAmrFeatureDocumentFactory
from zensols.config import ConfigFactory
from zensols.persist import PersistedWork, PrimeableStash, DelegateStash
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
        sec: str
        attr: str
        swap_type: str
        for sec, attr, swap_type in self.swaps:
            inst: Any = self.config_factory(sec)
            prev: Any = getattr(inst, attr)
            key: Tuple[str, str] = (sec, attr)
            repl: Any = replacements.get(key)
            new: Any
            if swap_type == 'path':
                new = self.root_dir / prev.name if repl is None else repl
            elif swap_type == 'persist':
                if repl is None:
                    new = PersistedWork(attr, prev.owner)
                else:
                    new = PersistedWork(attr, prev.owner, initial_value=repl)
            else:
                raise APIError(f'Unknown swap type: {swap_type}')
            self._prev[key] = (prev, new)
            setattr(inst, attr, new)

    def restore(self):
        """Restore all data replaced by :meth:`swap`."""
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'swap out: {self.root_dir}')
        sec: str
        attr: str
        prev: Any
        new: Any
        for (sec, attr), (prev, new) in self._prev.items():
            inst: Any = self.config_factory(sec)
            setattr(inst, attr, prev)
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
class AdhocAnnotatedAmrDocumentStash(PrimeableStash, DelegateStash):
    """A stash that generates and cachees instances of
    :class:`~zensols.amr.doc.AmrDocument`.

    """
    anon_doc_factory: AnnotatedAmrFeatureDocumentFactory = field()
    """Parses text data into AMR source and summary documents."""

    hasher: Hasher = field()
    """Used to create unique file cache root directories."""

    root_dir: Path = field()
    """The directory base where cached files are stored."""

    doc_key: Union[Sequence[str], Tuple[str, str]] = field()
    """Indicates the ``(config section, attribute)`` to locate the persisted
    work to set with the cached document so the factory stash doesn't.

    """
    swapper: ConfigSwapper = field()
    """Used to swap in the adhoc paths and document and then back out."""

    def __post_init__(self):
        if not isinstance(self.doc_key, Tuple):
            self.doc_key = tuple(self.doc_key)
        self._reset()

    def _reset(self):
        self._cache_dir: Path = None
        self._data: Union[Path, Dict, Sequence] = None
        self._created_docs: bool = False

    def _get_doc(self, data: Union[Path, Dict, Sequence]) -> AmrDocument:
        docs: Tuple[AmrFeatureDocument, ...] = \
            tuple(self.anon_doc_factory(data))
        sents: Iterable[AmrSentence] = map(
            lambda s: s.amr,
            chain.from_iterable(map(lambda d: d.sents, docs)))
        return AmrDocument.to_document(sents)

    def _get_cache_dir(self, corpus_id: str) -> Path:
        if corpus_id is None:
            self.hasher.reset()
            self.hasher.update(self._data)
            corpus_id = self.hasher()
        return self.root_dir / corpus_id

    def set_corpus(self, data: Union[Path, Dict, Sequence],
                   corpus_id: str = None):
        """Set the corpus documents that will be used for parsing and
        annotating.  The data will immediately be parsed into AMRs in this call
        and the data that writes to the file system will be updated to point to
        a new ``.../adhoc`` directory to not interfere with any corpus documents
        (see :class:`.ConfigSwapper`).

        To restore the configuration after adhoc document processing is
        finished, call :meth:`restore`.

        :param data: the AMR summary documents, which is usually a sequence of
                     :class:`~typing.Dict` instances (see
                     :class:`~zensols.arm.annotate.AnnotatedAmrFeatureDocumentFactory`
                     for data structure details)

        :param corpus_id: a unique identifier for ``data``, or ``None`` to use a
                          hashed string, which in turn, is used as the directory
                          name for the cached data

        """
        self._data = data
        self._cache_dir: Path = self._get_cache_dir(corpus_id)
        self.swapper.root_dir = self._cache_dir
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'set corpus: {self._cache_dir}')

    def _assert_init(self):
        if self._cache_dir is None:
            raise APIError('Must first call set_corpus before clearing')

    def _create_docs(self):
        self._assert_init()
        amr_doc_file: Path = self._cache_dir / 'amr-docs.dat'
        need_create: bool = not amr_doc_file.exists()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'need create: {need_create}')
        self.swapper.swap()
        if need_create:
            doc = self._get_doc(self._data)
            with open(amr_doc_file, 'wb') as f:
                pickle.dump(doc, f)
            if logger.isEnabledFor(logging.INFO):
                logger.info(f'restored adhoc parsed amrs: {amr_doc_file}')
        else:
            with open(amr_doc_file, 'rb') as f:
                doc = pickle.load(f)
            if logger.isEnabledFor(logging.INFO):
                logger.info(f'wrote adhoc parsed amrs: {amr_doc_file}')
        pw_current_corp_doc: PersistedWork = self.swapper[self.doc_key][1]
        pw_current_corp_doc.set(doc)

    def prime(self):
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'priming {type(self)}...')
        if not self._created_docs:
            self._create_docs()
            self._created_docs = True
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
        self._assert_init()
        self.swapper.clear()
