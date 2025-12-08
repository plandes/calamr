"""Client facade access to annotated AMR documents and alignment.

"""
__author__ = 'Paul Landes'
from typing import Dict, Sequence, Optional, Type, Union
from dataclasses import dataclass, field
import logging
import traceback
from pathlib import Path
from zensols.util import APIError
from zensols.persist import Stash
from zensols.amr import AmrFeatureDocument
from zensols.amr.serial import AmrSerializedFactory
from . import (
    DocumentGraph, FlowGraphResult, DocumentGraphFactory, DocumentGraphAligner
)

logger = logging.getLogger(__name__)


class _corpus_resource(object):
    def __init__(self, resource: 'Resource'):
        self._resource = resource

    def __enter__(self) -> Stash:
        return self._resource

    def __exit__(self, cls: Type[Exception], value: Optional[Exception],
                 trace: traceback):
        if value is not None:
            raise value


class _adhoc_resource(object):
    def __init__(self, resource: 'Resource',
                 corpus: Sequence[Dict[str, str]],
                 corpus_id: str = None, clear: bool = False):
        self._resource = resource
        self._doc_stash: 'AdhocAnnotatedAmrDocumentStash' = resource.documents
        self._clear = clear
        self._corpus = corpus
        self._corpus_id = corpus_id
        logger.info('setting corpus')
        self._doc_stash.set_corpus(self._corpus, self._corpus_id)

    def __enter__(self) -> Stash:
        logger.info('priming corpus')
        self._doc_stash.prime()
        return self._resource

    def __exit__(self, cls: Type[Exception], value: Optional[Exception],
                 trace: traceback):
        try:
            if self._clear:
                logger.info('clearing corpus')
                self._doc_stash.clear()
        except Exception as e:
            logger.error(f'Could not clear stash in {self.__class__}: {e}',
                         exc_info=True)
        try:
            logger.info('restoring corpus')
            self._doc_stash.restore()
        except Exception as e:
            logger.error(f'Could not restore state in {self.__class__}: {e}',
                         exc_info=True)
        if value is not None:
            raise value


@dataclass
class Resource(object):
    """Contains objects that parse AMR annotated documents and align them.
    Instance of this class are created with :class:`.Resources`.

    """
    documents: Stash = field()
    """A stash (:class:`dict` like) collection with AMR doc IDs keys to
    :class:`~zensols.amr.container.AmrFeatureDocument` values.

    """
    alignments: Stash = field()
    """A stash (:class:`dict` like) collection with AMR doc IDs keys to
    :class:`~zensols.calamr.flow.FlowGraphResult` values.

    """
    _resources: 'Resources' = field()
    """The object that created this instance."""

    def align(self, doc: Union[AmrFeatureDocument, str],
              render_level: int = None, directory: Path = None) -> \
            FlowGraphResult:
        """Align a document, which allows for the rendering of a document since
        it does not use the cached results from :obj:`alignments`.

        :param doc: either the unique document ID or a unique document ID that
                    indicates which document to align

        :param render_level: how many graphs to render (0 - 10), higher means
                             more

        :param directory: the output directory

        """
        graph_factory: DocumentGraphFactory = self._resources.doc_graph_factory
        graph_aligner: DocumentGraphAligner = self._resources.doc_graph_aligner
        doc: AmrFeatureDocument = self.documents[doc] \
            if isinstance(doc, str) else doc
        doc_graph: DocumentGraph = graph_factory(doc)
        prev_render_level: int = graph_aligner.render_level
        prev_output_dir: Path = graph_aligner.output_dir
        try:
            if render_level is not None:
                graph_aligner.render_level = render_level
            if directory is not None:
                graph_aligner.output_dir = directory
            return graph_aligner.align(doc_graph)
        finally:
            graph_aligner.render_level = prev_render_level
            graph_aligner.output_dir = prev_output_dir


@dataclass
class Resources(object):
    """A client facade (GoF) for Calamr annotated AMR corpus access and
    alginment.  This object is used as a context manager.

    The :meth:`corpus` and :meth:`adhoc` methods provide access to documents and
    alignments.  Use the stashes provided by those methods to clear respective
    cached data.

    :see: :class:`.AdhocAnnotatedAmrDocumentStash`

    """
    serialized_factory: AmrSerializedFactory = field()
    """Creates a :class:`.Serialized` from :class:`.AmrDocument`,
    :class:`.AmrSentence` or :class:`.AnnotatedAmrDocument`.

    """
    doc_graph_factory: DocumentGraphFactory = field()
    """Create document graphs."""

    doc_graph_aligner: DocumentGraphAligner = field()
    """Align document graphs."""

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

           from zensols.calamr import Resources, ApplicationFactory

           resources: Resources = ApplicationFactory.get_resources()

           with resources.corpus() as r:
               # print the keys of the annotated AMR documents
               print(tuple(r.documents.keys()))
               # determine if a document is in the stash
               print('some_key' in r.documents)
               # write an AMR document
               r.documents['some_key'].write()

        """
        return _corpus_resource(
            resource=Resource(
                documents=self._anon_doc_stash,
                alignments=self._flow_results_stash,
                _resources=self))

    def adhoc(self, corpus: Sequence[Dict[str, str]] = None,
              corpus_id: str = None, clear: bool = False) -> Resource:
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
        added to the AMR sentence metadata.

        The following example JSON creates a document with ID ``ex1``, a
        ``comment`` metadata, one
        :obj:`~zensols.amr.annotate.SentenceType.SUMMARY` and two
        :obj:`~zensols.amr.annotate.SentenceType.BODY` sentences::

            corpus = [{
                "id": "ex1",
                "comment": "very short",
                "body": "The man ran to make the train. He just missed it.",
                "summary": "A man got caught in a train he just missed."
            }]

        This source / summary text can then be AMR parsed, aligned, and rendered
        with:

        .. code-block:: python

           from zensols.calamr import Resources, ApplicationFactory

           resources: Resources = ApplicationFactory.get_resources()

           with resources.adhoc(corpus, clear=True) as r:
               # render an aligned document
               r.alignments['some_key'].render()

        The ``clear=True`` means to delete all cached files generated in the
        block.

        Either ``corpus`` and/or ``corpus_id`` must be given.  If ``corpus`` is
        not given but ``corpus_id`` is, it will assume there is an existing set
        of data files to use for accessing.

        :param corpus: the AMR summary documents, which is usually a sequence of
                       :class:`~typing.Dict` instances (see
                       :class:`~zensols.amr.annotate.AnnotatedAmrFeatureDocumentFactory`
                       for data structure details)

        :param corpus_id: a unique identifier for ``data``, or ``None`` to use a
                          hashed string, which in turn, is used as the directory
                          name for the cached data

        :param clear: whether or not to deleted the cached files (parsed
                      documents, aligned graphs etc) after leaving the lexical
                      boundaries of the context manager

        """
        if corpus is None:
            if corpus_id is None:
                raise APIError('Either a corpus or corpus_id must be given')
            corpus = ()
        return _adhoc_resource(
            resource=Resource(
                documents=self._adhoc_doc_stash,
                alignments=self._flow_results_stash,
                _resources=self),
            corpus=corpus,
            corpus_id=corpus_id,
            clear=clear)

    def restore(self, res: FlowGraphResult):
        """Restore the information on a flow graph result needed to render it.
        Without out it, :meth:`.FlowGraphResult.render` will raise errors.  This
        is only needed when unpickling a :class:`.FlowGraphResult`.

        :param res: to instance to have additional context information set

        """
        from .stash import FlowGraphRestoreStash
        stash: FlowGraphRestoreStash = self._flow_results_stash
        stash.restore(res)
