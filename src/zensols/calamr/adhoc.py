"""Classes that aid in creating and aligning documents without a corpus.

"""
from typing import Dict, List, Tuple, Iterable, Sequence, Union, Any
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
import shutil
from zensols.util.fail import APIError
from zensols.config import Dictable
from zensols.amr import AmrDocument, AmrFeatureDocument
from zensols.amr.annotate import AnnotatedAmrFeatureDocumentFactory
from zensols.config import ConfigFactory
from zensols.persist import persisted, PersistedWork, DictionaryStash, DelegateStash, ReadOnlyStash
from zensols.amr import AmrSentence
from zensols.amr.annotate import AnnotatedAmrDocument, AnnotatedAmrDocumentStash


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


@dataclass
class ConfigSwapper(Dictable):
    config_factory: ConfigFactory = field()
    root_dir: Path = field()
    swaps: Tuple[Tuple[str, str, str], ...] = field()
    corpus_id: str = field()

    def __post_init__(self):
        self._prev: Dict[Tuple[str, str], Tuple[Any, Any]] = {}

    @property
    @persisted('_corpus_dir')
    def corpus_dir(self) -> Path:
        return self.root_dir / self.corpus_id

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
                new = self.corpus_dir / prev.name if repl is None else repl
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
        if self.corpus_dir.is_dir():
            shutil.rmtree(self.corpus_dir)


@dataclass
class AdhocAnnotatedAmrDocumentStash(DelegateStash):
    config_factory: ConfigFactory = field()
    anon_doc_factory: AnnotatedAmrFeatureDocumentFactory = field()
    doc_key: Union[Sequence[str], Tuple[str, str]] = field()
    swapper_name: str = field()

    def __post_init__(self):
        self._swapper = PersistedWork('_swapper', self)
        if not isinstance(self.doc_key, Tuple):
            self.doc_key = tuple(self.doc_key)

    def _get_doc(self, data: Union[Path, Dict, Sequence]) -> AmrDocument:
        docs: Tuple[AmrFeatureDocument, ...] = \
            tuple(self.anon_doc_factory(data))
        sents: Iterable[AmrSentence] = map(
            lambda s: s.amr,
            chain.from_iterable(map(lambda d: d.sents, docs)))
        return AmrDocument.to_document(sents)

    def set_corpus(self, data: Union[Path, Dict, Sequence],
                   corpus_id: str):
        self._swapper = self.config_factory.new_instance(
            self.swapper_name, corpus_id=corpus_id)
        self._swapper.swap()
        doc = self._get_doc(data)
        pw_current_corp_doc: PersistedWork = self._swapper[self.doc_key][1]
        pw_current_corp_doc.set(doc)

    def restore(self):
        self._swapper.restore()

    def clear(self):
        self._swapper.clear()
