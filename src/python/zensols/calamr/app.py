"""AMR Summarization.

"""
from __future__ import annotations
__author__ = 'Paul Landes'
from typing import (
    Dict, Any, List, Tuple, ClassVar, Sequence, Union,
    Iterable, Optional, Callable
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
from zensols.amr import AmrFeatureDocument, Format
from zensols.amr.serial import AmrSerializedFactory
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
                    from zensols.datdesc import DataDescriber
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


@dataclass
class _ProtoApplication(_AlignmentBaseApplication):
    """An application entry point for prototyping.

    """
    CLI_META: ClassVar[Dict[str, Any]] = {'is_usage_visible': False}

    config_factory: ConfigFactory = field()
    """For prototyping."""

    def _get_doc(self):
        if 1:
            # proxy corpus
            k: str = '20080705_0216'
            # no summary
            k = '20030126_0212'
            # key error bug:
            #k = '20030814_0653'
            # no such alignment (37)
            #k = '20080107_0053'
            # max recursion
            #k = '20080125_0121'
            # unsupporter operand type
            #k = '20080717_0594'
        if k not in self.resource.anon_doc_stash:
            # adhoc
            k = 'liu-example'  # original 2014 Lui et al example
            #k = 'coref-src'  # example has co-reference in only souce
            #k = 'fixable-reentrancy'
            #k = 'soccer'
            #k = 'earthquake'
            #k = 'coref-bipartite'
        if k not in self.resource.anon_doc_stash:
            # little prince
            k = '1943'
        if k not in self.resource.anon_doc_stash:
            # bio
            k = '1563_0473'
        if k not in self.resource.anon_doc_stash:
            print(f'no key: {k} in {tuple(self.resource.anon_doc_stash.keys())}')
        return self.resource.anon_doc_stash[k]

    def _read_docs(self, input_file: Path) -> Iterable[AmrFeatureDocument]:
        from .annotate import AnnotatedAmrFeatureDocumentFactory
        factory: AnnotatedAmrFeatureDocumentFactory = \
            self.config_factory('anon_doc_factory')
        return factory(input_file)

    def _get_corpus_datadescriber(self, limit: int = 2) -> 'DataDescriber':
        from typing import Union
        from zensols.util import Failure
        from zensols.datdesc import DataFrameDescriber, DataDescriber
        stash = self.config_factory('calamr_flow_graph_result_cache_stash')
        dfs: List[pd.DataFrame] = []
        dd: DataDescriber = None
        did: str
        res: Union[FlowGraphResult, Failure]
        for did, res in it.islice(stash, limit):
            if isinstance(res, FlowGraphResult):
                df: pd.DataFrame = res.df
                df.insert(0, 'id', did)
                dfs.append(df)
                if dd is None:
                    dd = res.create_data_describer()
        dfd: DataFrameDescriber = dd.describers[0].derive(
            name=dd.name,
            desc=dd.name,
            df=pd.concat(dfs))
        dfd.meta = pd.concat((
            pd.Series(data={'doc_id': 'AMR document ID'},
                      index=['doc_id'], name='description').to_frame(),
            dfd.meta))
        if 1:
            from zensols.rend import ApplicationFactory
            br = ApplicationFactory.get_browser_manager()
            br(dfd)

    def _align(self, render_level: int = 5, write: bool = True):
        doc: AmrFeatureDocument = self._get_doc()
        doc_graph: DocumentGraph = self.resource.doc_graph_factory(doc)
        self.resource.doc_graph_aligner.render_level = render_level
        #self.resource.doc_graph_aligner.init_loops_render_level = render_level
        res: FlowGraphResult = self.resource.doc_graph_aligner.align(doc_graph)
        if write:
            res.write()
        return res

    def _align_file(self, input_file: Path):
        aligner: DocumentGraphAligner = self.resource.doc_graph_aligner
        doc: AmrFeatureDocument = next(iter(self._read_docs(input_file)))
        doc_graph: DocumentGraph = self.resource.doc_graph_factory(doc)
        res: FlowGraphResult = aligner.align(doc_graph)
        doc.amr.write(limit_sent=0)
        res.write()

    def _write(self):
        doc = self._get_doc()
        self.resource.serialized_factory(doc.amr).write()

    def _read_adhoc(self):
        input_file = Path('corpus/micro/source.json')
        for doc in it.islice(self._read_docs(input_file), 2):
            print(f'doc_id: {doc.amr.doc_id}')
            doc.amr.write(1, limit_sent=0)

    def _proxy_key_splits(self):
        ks = self.config_factory('calamr_amr_corp_split_keys')
        ks.write()

    def _tmp(self, render: bool = True):
        from zensols.util import time
        stash = self.config_factory('calamr_flow_graph_result_stash')
        #stash.clear()
        k = 'aurora-borealis'
        k = 'liu-example'
        #k = '20030814_0653'
        with time():
            res: FlowGraphResult = stash[k]
            if render:
                from .render.base import RenderContext, rendergroup
                renderer = self.config_factory('calamr_resource').doc_graph_aligner.renderer
                with rendergroup(renderer) as rg:
                    ctx: RenderContext
                    for ctx in res.get_render_contexts():
                        rg(ctx)
        res.write()

    def _tmp(self):
        from zensols.util import time
        key = 'liu-example'
        key = 'no-summary'
        key = 'aurora-borealis'
        key = 'falling-man'
        stash = self.config_factory('calamr_flow_graph_result_stash')
        with time(f'alignment {key}'):
            res = stash[key]
        res.write()
        #res.render()
        res.render(res.get_render_contexts(include_nascent=False))

    def _tmp_(self):
        doc: AmrFeatureDocument = self._get_doc()
        doc.amr.write(limit_sent=0)
        doc_graph: DocumentGraph = self.resource.doc_graph_factory(doc)
        #self.resource.doc_graph_aligner.render_level = render_level
        #self.resource.doc_graph_aligner.init_loops_render_level = render_level
        res: FlowGraphResult = self.resource.doc_graph_aligner.align(doc_graph)
        res.write()
        df = res.df
        #df.to_csv('aft-all.csv')
        df = df[df['s_attr'] == 'sentence']
        if 0:
            from zensols.rend import ApplicationFactory
            br = ApplicationFactory.get_browser_manager()
            df2 = pd.read_csv('/d/bef-all.csv', index_col=0)
            df2 = df2[df2['s_attr'] == 'sentence']
            br([df, df2])

    def _tmp(self):
        from .render.base import rendergroup, RenderContext
        stash = self.config_factory('calamr_flow_graph_result_stash')
        res = stash['earthquake']
        doc_graph = res.doc_graph.children['reversed_source']
        doc_graph = doc_graph.clone(reverse_edges=True, deep=True)
        if 1:
            res.write()
            return
        if 0:
            for c in doc_graph.components:
                print(f'{c.name}:')
                for v in c.vs.values():
                    print(' ', v)
            return
        if 1:
            gs = []
            for child in res.doc_graph.children.values():
                print(type(child))
                gs.append(child)
            renderer = self.config_factory('calamr_resource').doc_graph_aligner.renderer
            with rendergroup(renderer, graph_id='graph') as rg:
                for g in gs:
                    rg(RenderContext(doc_graph=g))
                rg(RenderContext(doc_graph=res.doc_graph))
            return
        renderer = self.config_factory('calamr_resource').doc_graph_aligner.renderer
        with rendergroup(renderer, graph_id='graph') as rg:
            rg(RenderContext(doc_graph=doc_graph))

    def _tmp(self):
        factory: AnnotatedAmrFeatureDocumentFactory = \
            self.config_factory('amr_anon_doc_factory')
        for doc in it.islice(factory(Path('short-story.json')), 1):
            print(type(doc))
            doc.write()
            from pprint import pprint
            for s in doc.sents:
                pprint(s.amr.metadata)

    def _tmp(self):
        print('aligning liu example...')
        self.resource.align_corpus_document('liu-example', True)

    def _tmp(self):
        from .render.base import rendergroup, RenderContext
        doc: AmrFeatureDocument = self._get_doc()
        doc.amr.write(include_amr=False)
        doc_graph: DocumentGraph = self.resource.doc_graph_factory(doc)
        renderer = self.config_factory('calamr_resource').doc_graph_aligner.renderer
        with rendergroup(renderer, graph_id='graph') as rg:
            rg(RenderContext(doc_graph=doc_graph))

    def proto(self, run: int = 1):
        """Prototype test."""
        return {
            0: self._tmp,
            # make earthquake more well connected
            # neighbor_embedding_weights: [0, 1, 0.6, 0.5, 0.3, 0.2, 0.1]
            # neighbor_skew: {y_trans: -0.5, x_trans: 0, compress: 1}
            1: lambda: self._align(render_level=5),
            2: lambda: self.align('20080717_0594', Path('-'), Format.txt),
        }[run]()
