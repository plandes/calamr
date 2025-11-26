"""Classes that aid in creating and aligning documents without a corpus.

"""
from typing import Dict, Tuple, Iterable, Sequence, Union, Any
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
import shutil
from zensols.util.fail import APIError
from zensols.config import Dictable
from zensols.amr import AmrDocument, AmrFeatureDocument
from zensols.amr.annotate import AnnotatedAmrFeatureDocumentFactory
from zensols.config import ConfigFactory
from zensols.persist import persisted, PersistedWork, DelegateStash
from zensols.amr import AmrSentence


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
