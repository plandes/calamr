"""Alignment entry point application.

"""
from __future__ import annotations
__author__ = 'Paul Landes'
from typing import (
    Dict, Any, List, Tuple, Sequence, Union, Iterable,
    Optional, Callable, TYPE_CHECKING
)
from dataclasses import dataclass, field
import logging
import sys
import traceback
import re
from pathlib import Path
import itertools as it
from io import TextIOWrapper
import pandas as pd
from zensols.util import stdout
from zensols.config import ConfigFactory, DefaultDictable
from zensols.persist import Stash
from zensols.cli import ApplicationError
if TYPE_CHECKING:
    from zensols.datdesc import DataDescriber
from zensols.amr import AmrFailure, AmrDocument, AmrFeatureDocument, Format
from zensols.amr.serial import AmrSerializedFactory
from zensols.amr.docfac import AmrFeatureDocumentFactory
from zensols.amr.annotate import AnnotatedAmrFeatureDocumentFactory
from . import (
    DocumentGraph, DocumentGraphFactory, DocumentGraphAligner, FlowGraphResult
)

logger = logging.getLogger(__name__)


@dataclass
class Resource(object):
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

        :return: the parsed AMR feature document

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


@dataclass
class _AlignmentBaseApplication(object):
    """Base class for applications defined in this module.

    """
    resource: Resource = field()
    """A client facade (GoF) for Calamr annotated AMR corpus access and
    alginment.

    """
    def _get_output_file(self, output_dir: Path, key: str,
                         output_format: Format) -> Path:
        output_dir = output_dir / key
        self.resource.doc_graph_aligner.output_dir = output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        ext: str = Format.to_ext(output_format)
        return output_dir / f'results.{ext}'

    def _output_align(self, output_format: Format, doc: AmrFeatureDocument,
                      fout: TextIOWrapper, res: FlowGraphResult):
        doc: AmrFeatureDocument
        if doc is None:
            doc = res.doc_graph.doc
        if output_format == Format.txt:
            self.resource.serialized_factory(doc.amr).write(writer=fout)
            res.write(writer=fout, include_doc=False)
        elif output_format == Format.json:
            dct = DefaultDictable(
                {'doc': self.resource.serialized_factory(doc.amr).asdict(),
                 'alignment': res.asdict()})
            fout.write(dct.asjson(indent=4))
        elif output_format == Format.csv:
            res.stats_df.to_csv(fout, index=False)
        else:
            raise ApplicationError(
                f'Format not supported: {output_format.name}')

    def _get_align_result(self, key: str, use_cached: bool) -> \
            Tuple[FlowGraphResult, AmrFeatureDocument]:
        res: FlowGraphResult = None
        doc: AmrFeatureDocument = None
        if use_cached:
            res = self.resource.flow_results_stash.get(key)
        else:
            aligner: DocumentGraphAligner = self.resource.doc_graph_aligner
            doc = self.resource.anon_doc_stash.get(key)
            if doc is None:
                raise ApplicationError(f"No such key: '{key}', use keys action")
            doc_graph: DocumentGraph = self.resource.doc_graph_factory(doc)
            res = aligner.align(doc_graph)
        if res is None:
            raise ApplicationError(f"No such key: '{key}', use keys action")
        if res.is_error:
            res.failure.rethrow()
        return res, doc

    def _align_corpus(self, key: str, output_dir: Path, output_format: Format,
                      use_stdout: bool, use_cached: bool) -> FlowGraphResult:
        logger.info(f'aligning on document {key}')
        fout: TextIOWrapper
        output_file: Path
        if use_stdout:
            fout = sys.stdout
        else:
            output_file = self._get_output_file(output_dir, key, output_format)
            fout = open(output_file, 'w')
        try:
            res: FlowGraphResult
            doc: AmrFeatureDocument
            res, doc = self._get_align_result(key, use_cached)
            self._output_align(output_format, doc, fout, res)
            return res
        finally:
            if id(fout) != id(sys.stdout):
                fout.close()
                logger.info(f'wrote: {output_file}')

    def _prep_align(self, output_dir: Path, render_level: int) -> \
            Tuple[Path, bool]:
        assert output_dir is not None
        use_stdout: bool
        if output_dir.name == stdout.STANDARD_OUT_PATH:
            use_stdout = True
        else:
            output_dir = output_dir.expanduser()
            use_stdout = False
        if not use_stdout:
            logger.setLevel(logging.INFO)
        if render_level is not None:
            self.resource.doc_graph_aligner.render_level = render_level
        return output_dir, use_stdout


@dataclass
class CorpusApplication(_AlignmentBaseApplication):
    """AMR graph aligment.

    """
    config_factory: ConfigFactory = field()
    """For prototyping."""

    results_dir: Path = field()
    """The directory where the output results are written, then read back for
    analysis reporting.

    """
    def write_keys(self):
        """Write the keys of the configured corpus."""
        key: str
        for key in self.resource.anon_doc_stash.keys():
            print(key)

    def get_annotated_summary(self, limit: int = None) -> pd.DataFrame:
        """Return a CSV file with a summary of the annotated AMR dataset.

        :param limit: the max of items to process

        """
        rows: List[Tuple[str, int]] = []
        idx: List[str] = []
        limit = sys.maxsize if limit is None else limit
        k: str
        doc: AmrFeatureDocument
        for k, doc in it.islice(self.resource.anon_doc_stash.items(), limit):
            idx.append(k)
            rows.append((doc.token_len, doc.text))
        df = pd.DataFrame(rows, columns='len text'.split(), index=idx)
        df = df.sort_values('len')
        df.index.name = 'id'
        return df

    def dump_annotated(self, limit: int = None, output_dir: Path = None,
                       output_format: Format = Format.csv):
        """Write annotated documents and their keys.

        :param limit: the max of items to process

        :param output_dir: the output directory

        :param output_format: the output format

        """
        if output_dir is None:
            output_dir = self.results_dir
        output_dir = output_dir.expanduser()
        limit = sys.maxsize if limit is None else limit
        fac: AmrSerializedFactory = self.resource.serialized_factory
        docs: Dict[str, Any] = {}
        k: str
        doc: AmrFeatureDocument
        for k, doc in it.islice(self.resource.anon_doc_stash.items(), limit):
            docs[k] = fac(doc.amr).asdict()
        dct = DefaultDictable(docs)
        fname: str = f'annotated.{Format.to_ext(output_format)}'
        output_file: Path = output_dir / fname
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as fout:
            if output_format == Format.txt:
                dct.write(writer=fout)
            elif output_format == Format.csv:
                df: pd.DataFrame = self.get_annotated_summary(limit)
                df.to_csv(fout)
            elif output_format == Format.json:
                fout.write(dct.asjson(indent=4))
            else:
                raise ApplicationError(
                    f'Format not supported: {output_format.name}')
        logger.info(f'wrote: {output_file}')

    def align_corpus(self, keys: str, output_dir: Path = None,
                     output_format: Format = Format.csv,
                     render_level: int = None,
                     use_cached: bool = False):
        """Align an annotated AMR document from the corpus.

        :param keys: comma-separated list of dataset keys or file name

        :param output_dir: the output directory

        :param output_format: the output format

        :param render_level: how many graphs to render (0 - 10), higher means
                             more

        :param use_cached: whether to use a cached result if available

        """
        success_keys: List[str] = []
        results: List[FlowGraphResult] = []
        use_stdout: bool
        output_dir, use_stdout = self._prep_align(output_dir, render_level)
        keys: Sequence[str]
        if keys == 'ALL':
            keys = tuple(self.resource.anon_doc_stash.keys())
        else:
            keys = re.findall(r'[^,\s]+', keys)
        key: str
        for key in keys:
            try:
                fdg: FlowGraphResult = self._align_corpus(
                    key, output_dir, output_format, use_stdout, use_cached)
                results.append(fdg)
                success_keys.append(key)
                if not use_stdout:
                    dd: DataDescriber = fdg.create_data_describer()
                    meth: Callable = dd.save_csv
                    meth(output_dir / 'alignments')
            except Exception as e:
                msg: str = f'Can not align {key}: {e}'
                logger.error(msg)
                if not use_stdout:
                    with open(output_dir / 'errors.txt', 'a') as f:
                        f.write(f'error: {msg}\n')
                        traceback.print_exception(e, file=f)
                        f.write('_' * 80 + '\n')
        if not use_stdout and len(keys) > 1:
            dfs: List[pd.DataFrame] = []
            res: FlowGraphResult
            for key, res in zip(success_keys, results):
                if res.is_error:
                    res.write()
                    continue
                df: pd.DataFrame = res.stats_df
                cols = df.columns.tolist()
                df['id'] = key
                df = df[['id'] + cols]
                dfs.append(df)
            df = pd.concat(dfs)
            df_path: Path = output_dir / 'results.csv'
            df.to_csv(df_path, index=False)
            logger.info(f'wrote: {df_path}')

    def write_adhoc_corpus(self, corpus_file: Path = None):
        """Write the adhoc corpus from the JSON created file.

        :param corpus_file: the file with the source and summary sentences

        """
        from zensols.amr.annotate import FileCorpusWriter
        writer_sec: str = 'calamr_adhoc_corpus'
        if writer_sec not in self.config_factory.config.sections:
            raise ApplicationError(
                'Looks like wrong config: ' +
                'use -f --override calamr_corpus.name=adhoc')
        writer: FileCorpusWriter = self.config_factory(writer_sec)
        if corpus_file is None:
            corpus_file = writer.input_file
        else:
            writer.input_file = corpus_file
        logger.info(f'reading: {corpus_file}')
        if writer.output_file.is_file():
            logger.warning(f'already exists: {writer.output_file}--skipping')
        else:
            writer()
        # clear the annotated feature document stash
        self.resource.anon_doc_stash.clear()


@dataclass
class AlignmentApplication(_AlignmentBaseApplication):
    """This application aligns data in files.

    """
    config_factory: ConfigFactory = field()
    """Application configuration factory."""

    def align_file(self, input_file: Path, output_dir: Path = None,
                   output_format: Format = Format.csv,
                   render_level: int = None):
        """Align annotated documents from a JSON file.

        :param input_file: the input JSON file.

        :param output_dir: the output directory

        :param output_format: the output format

        :param render_level: how many graphs to render (0 - 10), higher means
                             more

        """
        aligner: DocumentGraphAligner = self.resource.doc_graph_aligner
        output_dir, use_stdout = self._prep_align(output_dir, render_level)
        success_keys: List[str] = []
        results: List[FlowGraphResult] = []
        dix: int
        doc: AmrFeatureDocument
        for dix, doc in enumerate(self.resource.parse_documents(input_file)):
            try:
                doc_graph: DocumentGraph = self.resource.doc_graph_factory(doc)
                output_file: Path
                if use_stdout:
                    output_file = stdout.STANDARD_OUT_PATH
                else:
                    output_file: Path = self._get_output_file(
                        output_dir, str(dix), output_format)
                res: FlowGraphResult = aligner.align(doc_graph)
                if res.is_error:
                    res.failure.rethrow()
                with stdout(output_file, 'w') as fout:
                    self._output_align(output_format, doc, fout, res)
                if not use_stdout:
                    logger.info(f'wrote: {output_file}')
                results.append(res)
                success_keys.append(dix)
            except Exception as e:
                msg: str = f'Can not align {dix}th {doc}: {e}'
                logger.error(msg)
                if not use_stdout:
                    with open(output_dir / 'errors.txt', 'a') as f:
                        f.write(f'error: {msg}\n')
                        traceback.print_exception(e, file=f)
                        f.write('_' * 80 + '\n')
        if not use_stdout and dix > 1:
            dfs: List[pd.DataFrame] = []
            res: FlowGraphResult
            for key, res in zip(success_keys, results):
                if res.is_error:
                    res.write()
                    continue
                df: pd.DataFrame = res.stats_df
                cols = df.columns.tolist()
                df['id'] = key
                df = df[['id'] + cols]
                dfs.append(df)
            df = pd.concat(dfs)
            df_path: Path = output_dir / 'results.csv'
            df.to_csv(df_path, index=False)
            logger.info(f'wrote: {df_path}')
