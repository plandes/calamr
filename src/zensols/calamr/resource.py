"""Client facade access to annotated AMR documents and alignment.

"""
from __future__ import annotations
__author__ = 'Paul Landes'
from typing import Dict, List, Tuple, Sequence, Union, Iterable, Optional, Type
from dataclasses import dataclass, field
import logging
import traceback
from pathlib import Path
from zensols.persist import Stash
from zensols.amr import AmrFailure, AmrDocument, AmrFeatureDocument
from zensols.amr.serial import AmrSerializedFactory
from zensols.amr.docfac import AmrFeatureDocumentFactory
from zensols.amr.annotate import AnnotatedAmrFeatureDocumentFactory
from . import (
    DocumentGraph, DocumentGraphFactory, DocumentGraphAligner, FlowGraphResult
)

logger = logging.getLogger(__name__)


class _corpus_resource(object):
    def __init__(self, resource: Resource):
        self._resource = resource

    def __enter__(self) -> Stash:
        return self._resource

    def __exit__(self, cls: Type[Exception], value: Optional[Exception],
                 trace: traceback):
        if value is not None:
            raise value


class _adhoc_resource(object):
    def __init__(self, resource: Resource,
                 data: Union[Path, Dict, Sequence],
                 corpus_id: str = None, clear: bool = False):
        self._resource = resource
        self._doc_stash: 'AdhocAnnotatedAmrDocumentStash' = resource.documents
        self._clear = clear
        self._data = data
        self._corpus_id = corpus_id
        self._doc_stash.set_corpus(self._data, self._corpus_id)

    def __enter__(self) -> Stash:
        self._doc_stash.prime()
        return self._resource

    def __exit__(self, cls: Type[Exception], value: Optional[Exception],
                 trace: traceback):
        if value is not None:
            raise value
        try:
            if self._clear:
                self._doc_stash.clear()
            self._doc_stash.restore()
        except Exception as e:
            logger.error(f'Could not restore state {self.__class__}: {e}',
                         exc_info=True)


@dataclass
class Resource(object):
    """Contains objects that parse AMR annotated documents and align them.

    """
    documents: Stash = field()
    """An stash (:class:`dict` like) collection with AMR doc IDs keys to
    :class:`~zensols.amr.doc.AmrFeatureDocument` values.

    """
    alignments: Stash = field()
    """An stash (:class:`dict` like) collection with AMR doc IDs keys to
    :class:`~zensols.calamr.flow.FlowGraphResult` values.

    """


@dataclass
class Resources(object):
    """A client facade (GoF) for Calamr annotated AMR corpus access and
    alginment.  This object is used as a context manager.

    :see: :class:`.AdhocAnnotatedAmrDocumentStash`

    """
    serialized_factory: AmrSerializedFactory = field()
    """Creates a :class:`.Serialized` from :class:`.AmrDocument`,
    :class:`.AmrSentence` or :class:`.AnnotatedAmrDocument`.

    """
    doc_graph_factory: DocumentGraphFactory = field()
    """Create document graphs."""

    doc_graph_aligner: DocumentGraphAligner = field()
    """Create document graphs."""

    _anon_doc_stash: Stash = field()
    """Contains human annotated AMRs.  This could be from the adhoc (micro)
    corpus (small toy corpus), AMR 3.0 Proxy Report corpus, Little Prince, or
    the Bio AMR corpus.

    """
    _adhoc_doc_stash: Stash = field()
    """A :class:`~zensols.calamr.adhoc.AdhocAnnotatedAmrDocumentStash` instance.
    It is used generate documents without setting up a corpus.

    """
    _flow_results_stash: Stash = field()
    """Creates cached instances of :class:`.FlowGraphResult`."""

    def corpus(self) -> Resource:
        """Return a context manager for corpus access.  A corpus must be created
        before using this method, which amounts to using an AMR parser to create
        the parenthetical text files.  These files are then made available as
        resource to be downloaded or available on the file system.

        Example:

        .. code-block:: python

           with self.resources.corpus() as r:
               # print the keys of the annotated AMR documents
               r.documents.keys()
               # determine if a document is in the stash
               print('some_key' in r.documents)
               # write an AMR document
               r.documents['some_key'].write()

        """
        return _corpus_resource(
            resource=Resource(
                documents=self._anon_doc_stash,
                alignments=self._flow_results_stash))

    def adhoc(self, data: Union[Path, Dict, Sequence], corpus_id: str = None,
              clear: bool = False) -> Resource:
        """Return a context manager for parsing and aligning adhoc documents.
        This sets the corpus documents that will be used for parsing and
        annotating.  The data will immediately be parsed into AMRs in this call
        and the data that writes to the file system will be updated to point to
        a new ``.../adhoc`` directory to not interfere with any corpus
        documents.

        The ``data`` input can be a file name that contains parsed parenthetical
        AMRs, a single document, or a sequence of documents.  The keys of each
        dictionary are the case-insensitive enumeration values of
        :class:`~zensols.amr.annotate.SentenceType`.  Keys ``id`` and
        ``comment`` are the unique document identifier and a comment that is
        added to the AMR sentence metadata.  Both are optional, and if ``id`` is
        missing, :obj:``doc_id``.

        The following example JSON creates a document with ID ``ex1``, a
        ``comment`` metadata, one
        :obj:`~zensols.amr.annotate.SentenceType.SUMMARY` and two
        :obj:`.SentenceType.BODY` sentences::

            corpus = [{
                "id": "ex1",
                "comment": "very short",
                "body": "The man ran to make the train. He just missed it.",
                "summary": "A man got caught in the door of a train he just missed."
            }]

        This source / summary text can then be AMR parsed, aligned, and rendered
        with:

        .. code-block:: python

           with self.resources.adhoc(corpus, clear=True) as r:
               # render an aligned document
               r.alignments['some_key'].render()

        :param data: the AMR summary documents, which is usually a sequence of
                     :class:`~typing.Dict` instances (see
                     :class:`~zensols.arm.annotate.AnnotatedAmrFeatureDocumentFactory`
                     for data structure details)

        :param corpus_id: a unique identifier for ``data``, or ``None`` to use a
                          hashed string, which in turn, is used as the directory
                          name for the cached data

        :param clear: whether or not to deleted the cached files (parsed
                      documents, aligned graphs etc) after leaving the lexical
                      boundaries of the context manager

        """
        return _adhoc_resource(
            resource=Resource(
                documents=self._adhoc_doc_stash,
                alignments=self._flow_results_stash),
            data=data,
            corpus_id=corpus_id,
            clear=clear)


@dataclass
class Toolbox(object):
    """A client facade (GoF) for Calamr annotated AMR corpus access and
    alginment.

    """
    anon_doc_stash: Stash = field()
    """Contains human annotated AMRs.  This could be from the adhoc (micro)
    corpus (small toy corpus), AMR 3.0 Proxy Report corpus, Little Prince, or
    the Bio AMR corpus.

    """
    doc_factory: AmrFeatureDocumentFactory = field()
    """Creates :class:`.AmrFeatureDocument` from :class:`.AmrDocument`
    instances.

    """
    serialized_factory: AmrSerializedFactory = field()
    """Creates a :class:`.Serialized` from :class:`.AmrDocument`,
    :class:`.AmrSentence` or :class:`.AnnotatedAmrDocument`.

    """
    doc_graph_factory: DocumentGraphFactory = field()
    """Create document graphs."""

    doc_graph_aligner: DocumentGraphAligner = field()
    """Create document graphs."""

    flow_results_stash: Stash = field()
    """Creates cached instances of :class:`.FlowGraphResult`."""

    anon_doc_factory: AnnotatedAmrFeatureDocumentFactory = field()
    """Creates instances of :class:`~zensols.amr.annotate..AmrFeatureDocument`.

    """
    def get_corpus_keys(self) -> Iterable[str]:
        """Get the keys of the application configured AMR corpus."""
        return self.anon_doc_stash.keys()

    def get_corpus_document(self, doc_id: str) -> AmrFeatureDocument:
        """Get an AMR feature document by key from the application configured
        corpus.

        :param doc_id: the corpus document ID (i.e. ``liu-example`` or
                       ``20041010_0024``)

        :return: the AMR feature document

        """
        return self.anon_doc_stash.get(doc_id)

    def parse_documents(self, data: Union[Path, Dict, Sequence]) -> \
            Iterable[AmrFeatureDocument]:
        """Parse documents with keys ``id``, ``comment``, ``body``, and
        ``summary`` from a :class:`dict`, sequence of :class:`dict`
        instanaces. or JSON file in the format::

            [{
                "id": "ex1",
                "comment": "very short",
                "body": "The man ran to make the train. He just missed it.",
                "summary": "A man got caught in the door of a train he missed."
            }]

        :return: the parsed AMR feature document(s)

        :see: :class:`~zensols.amr.annotate.AnnotatedAmrFeatureDocumentFactory`

        """
        return self.anon_doc_factory(data)

    def to_feature_doc(self, amr_doc: AmrDocument, catch: bool = False,
                       add_metadata: Union[str, bool] = False,
                       add_alignment: bool = False) -> \
            Union[AmrFeatureDocument,
                  Tuple[AmrFeatureDocument, List[AmrFailure]]]:
        """Create a :class:`.AmrFeatureDocument` from a class:`.AmrDocument` by
        parsing the ``snt`` metadata with a
        :class:`~zensols.nlp.parser.FeatureDocumentParser`.

        :param add_metadata: add missing annotation metadata to ``amr_doc``
                             parsed from spaCy if missing (see
                             :meth:`.AmrParser.add_metadata`) if ``True`` and
                             replace any previous metadata if this value is the
                             string ``clobber``

        :param catch: if ``True``, return caught exceptions creating a
                      :class:`.AmrFailure` from each and return them

        :return: an AMR feature document if ``catch`` is ``False``; otherwise, a
                 tuple of a document with sentences that were successfully
                 parsed and a list any exceptions raised during the parsing

        """
        return self.doc_factory.to_feature_doc(
            amr_doc, catch, add_metadata, add_alignment)

    def to_annotated_doc(self, doc: Union[AmrDocument, AmrFeatureDocument]) \
            -> AmrFeatureDocument:
        """Return an annotated feature document, creating the feature document
        if necessary.  The ``doc.amr`` attribute is set to annotated AMR
        document.

        :param doc: an AMR document or an AMR feature document

        :return: a new instance of a document if ``doc`` is not a
                 ``AmrFeatureDocument`` or if ``doc.amr`` is not an
                 ``AnnotatedAmrDocument``

        """
        fdoc: AmrFeatureDocument
        if isinstance(doc, AmrDocument):
            fdoc = self.to_feature_doc(doc)
        else:
            fdoc = doc
        fdoc = self.anon_doc_factory.to_annotated_doc(fdoc)
        return fdoc

    def create_graph(self, doc: AmrFeatureDocument) -> DocumentGraph:
        """Return a new document graph based on feature document.

        :param doc: the document on which to base the new graph

        :return: a new AMR document graph

        """
        return self.doc_graph_factory(doc)

    def align_corpus_document(self, doc_id: str, use_cache: bool = True) -> \
            Optional[FlowGraphResult]:
        """Create flow results of a corpus AMR document.

        :param doc_id: the corpus document ID (i.e. ``liu-example`` or
                       ``20041010_0024``)

        :param use_cache: whether to cache (and use cached) results

        :return: the flow results for the corpus document or ``None`` if
                 ``doc_id`` is not a valid key

        """
        res: FlowGraphResult = None
        if use_cache:
            res = self.flow_results_stash.get(doc_id)
        else:
            doc = self.anon_doc_stash.get(doc_id)
            if doc is not None:
                doc_graph: DocumentGraph = self.doc_graph_factory(doc)
                res = self.align(doc_graph)
        return res

    def align(self, doc_graph: DocumentGraph) -> FlowGraphResult:
        """Create flow results from a document graph.

        :param doc_graph: the source/summary components to align

        :return: the aligned bipartite graph and its statistics

        """
        al: DocumentGraphAligner = self.doc_graph_aligner
        levels: Tuple[int, int] = al.render_level, al.init_loops_render_level
        al.render_level, al.init_loops_render_level = 0, 0
        try:
            return al.align(doc_graph)
        finally:
            al.render_level, al.init_loops_render_level = levels

    def restore(self, res: FlowGraphResult):
        """Restore the information on a flow graph result needed to render it.
        Without out it, :meth:`.FlowGraphResult.render` will raise errors.

        This is only needed when pickling a :class:`.FlowGraphResult`, and thus,
        bypassing :obj:`flow_results_stash`.

        :param res: has additional context information set

        """
        from .stash import FlowGraphRestoreStash
        stash: FlowGraphRestoreStash = self.flow_results_stash
        stash.restore(res)
