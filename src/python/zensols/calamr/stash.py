"""Alignment dataframe stash.

"""
__author__ = 'Paul Landes'

from typing import Iterable
from dataclasses import dataclass, field
import sys
import logging
import itertools as it
from zensols.persist import (
    persisted, Stash, ReadOnlyStash, PrimeableStash, DelegateStash
)
from zensols.amr import AmrFeatureDocument
from . import (
    DocumentGraph, DocumentGraphFactory, DocumentGraphAligner, FlowGraphResult
)
from .flow import _FlowGraphResultContext

logger = logging.getLogger(__name__)


@dataclass
class FlowGraphResultFactoryStash(ReadOnlyStash, PrimeableStash):
    """A factory stash that creates aligned :class:`.FlowGraphResult` instances
    or :class:`.ComponentAlignmentFailure` when the document cannot be aligned.

    """
    anon_doc_stash: Stash = field()
    """Contains human annotated AMRs."""

    doc_graph_aligner: DocumentGraphAligner = field()
    """Create document graphs."""

    doc_graph_factory: DocumentGraphFactory = field()
    """Create document graphs."""

    limit: int = field(default=sys.maxsize)
    """The limit of the number of items to create."""

    def load(self, name: str) -> FlowGraphResult:
        if logger.isEnabledFor(logging.INFO):
            logger.info(f"creating alignment: '{name}'")
        prev_render_level: int = self.doc_graph_aligner.render_level
        self.doc_graph_aligner.render_level = 0
        try:
            doc: AmrFeatureDocument = self.anon_doc_stash[name]
            doc_graph: DocumentGraph = self.doc_graph_factory(doc)
            res: FlowGraphResult = self.doc_graph_aligner.align(doc_graph)
            if logger.isEnabledFor(logging.INFO):
                logger.info(f"alignment result: {res}")
            return res
        except Exception as e:
            return self.doc_graph_aligner.create_error_result(
                e, f"Could not align: '{name}'")
        finally:
            self.doc_graph_aligner.render_level = prev_render_level

    @persisted('_keys')
    def keys(self) -> Iterable[str]:
        return set(it.islice(self.anon_doc_stash.keys(), self.limit))

    def exists(self, name: str) -> bool:
        return name in self.keys()

    def prime(self):
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'priming {type(self)}...')
        self.anon_doc_stash.prime()


@dataclass
class FlowGraphRestoreStash(DelegateStash, PrimeableStash):
    """The a stash that restores transient data on :class:`.FlowGraphResult`
    instances.

    """
    flow_graph_result_context: _FlowGraphResultContext = field()
    """Contains in memory/interperter session data needed by
    :class:`.FlowGraphResult` when it is created or unpickled.

    """
    def load(self, name: str) -> FlowGraphResult:
        res: FlowGraphResult = super().load(name)
        res._set_context(self.flow_graph_result_context)
        return res

    def exists(self, name: str) -> bool:
        # the stash decorator instance graph is too deep and dependent on lazy
        # vs. preemtive; so rely on existence in keys
        self.prime()
        return Stash.exists(self, name)

    def get(self, name: str, default: FlowGraphResult = None) -> \
            FlowGraphResult:
        self.prime()
        res: FlowGraphResult = self.load(name)
        if res is None:
            return default
        else:
            return res
