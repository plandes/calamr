"""Classes that aid in creating and aligning documents without a corpus.

"""
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple, Union
from zensols.config import Dictable
from zensols.amr import AmrDocument, AmrFeatureDocument
from zensols.amr.annotate import AnnotatedAmrFeatureDocumentFactory
from zensols.config import ConfigFactory
from zensols.persist import PersistedWork, DictionaryStash, ReadOnlyStash
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
    pass


@dataclass
class AdhocAnnotatedAmrDocumentStash(ReadOnlyStash):
    def set_corpus(self, data: Union[Path, Dict, Sequence]):
        print('here')
