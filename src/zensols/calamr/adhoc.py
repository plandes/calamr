"""Classes that aid in creating and aligning documents without a corpus.

"""
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
from zensols.persist import PersistedWork, DelegateStash
from zensols.amr import AmrSentence

logger = logging.getLogger(__name__)


@dataclass
class ConfigSwapper(Dictable):
    config_factory: ConfigFactory = field()
    root_dir: Path = field()
    swaps: Tuple[Tuple[str, str, str], ...] = field()

    def __post_init__(self):
        self._prev: Dict[Tuple[str, str], Tuple[Any, Any]] = {}

    def swap(self, replacements: Dict[str, Any] = None):
        replacements = {} if replacements is None else replacements
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
        sec: str
        attr: str
        prev: Any
        new: Any
        for (sec, attr), (prev, new) in self._prev.items():
            inst: Any = self.config_factory(sec)
            setattr(inst, attr, prev)
        self._prev.clear()

    def __getitem__(self, key: Tuple[str, str]) -> Tuple[Any, Any]:
        return self._prev[key]

    def clear(self):
        if self.root_dir.is_dir():
            shutil.rmtree(self.root_dir)


@dataclass
class AdhocAnnotatedAmrDocumentStash(DelegateStash):
    config_factory: ConfigFactory = field()
    anon_doc_factory: AnnotatedAmrFeatureDocumentFactory = field()
    hasher: Hasher = field()
    temporary_dir: Path = field()
    doc_key: Union[Sequence[str], Tuple[str, str]] = field()
    swapper: ConfigSwapper = field()

    def __post_init__(self):
        if not isinstance(self.doc_key, Tuple):
            self.doc_key = tuple(self.doc_key)

    def _get_doc(self, data: Union[Path, Dict, Sequence]) -> AmrDocument:
        docs: Tuple[AmrFeatureDocument, ...] = \
            tuple(self.anon_doc_factory(data))
        sents: Iterable[AmrSentence] = map(
            lambda s: s.amr,
            chain.from_iterable(map(lambda d: d.sents, docs)))
        return AmrDocument.to_document(sents)

    def _get_temp_dir(self, data: Union[Path, Dict, Sequence],
                      corpus_id: str) -> Path:
        if corpus_id is None:
            self.hasher.reset()
            self.hasher.update(data)
            corpus_id = self.hasher()
        return self.temporary_dir / corpus_id

    def set_corpus(self, data: Union[Path, Dict, Sequence],
                   corpus_id: str = None):
        temp_dir: Path = self._get_temp_dir(data, corpus_id)
        amr_doc_file: Path = temp_dir / 'amr-docs.dat'
        need_create: bool = not amr_doc_file.exists()
        self.swapper.root_dir = temp_dir
        self.swapper.swap()
        if need_create:
            doc = self._get_doc(data)
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

    def restore(self):
        self.swapper.restore()

    def clear(self):
        if self.swapper.root_dir is None:
            raise APIError('Must first call set_corpus before clearing')
        self.swapper.clear()
