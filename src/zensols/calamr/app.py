"""Alignment entry point application.

"""
from __future__ import annotations
__author__ = 'Paul Landes'
from typing import Dict, Any, List, Tuple, Sequence, Callable, TYPE_CHECKING
from dataclasses import dataclass, field
import logging
import sys
import traceback
import re
from pathlib import Path
import itertools as it
from io import TextIOWrapper
import json
import pandas as pd
from zensols.util import stdout
from zensols.config import ConfigFactory, DefaultDictable
from zensols.cli import ApplicationError
if TYPE_CHECKING:
    from zensols.datdesc import DataDescriber
from zensols.amr import AmrFeatureDocument, Format
from zensols.amr.serial import AmrSerializedFactory
from . import (
    DocumentGraph, DocumentGraphAligner, FlowGraphResult
)
from .resource import Resources

logger = logging.getLogger(__name__)


@dataclass
class _AlignmentBaseApplication(object):
    """Base class for applications defined in this module.

    """
    resources: Resources = field()
    """A client facade (GoF) for Calamr annotated AMR corpus access and
    alginment.

    """
    doc_graph_aligner: DocumentGraphAligner = field()
    """Create document graphs."""

    def _get_output_file(self, output_dir: Path, key: str,
                         output_format: Format) -> Path:
        output_dir = output_dir / key
        self.doc_graph_aligner.output_dir = output_dir
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
            aligner: DocumentGraphAligner = self.doc_graph_aligner
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
        """Write the document keys of the configured corpus."""
        with self.resources.corpus() as r:
            print('\n'.join(r.documents.keys()))

    def get_annotated_summary(self, limit: int = None) -> pd.DataFrame:
        """Return a CSV file with a summary of the annotated AMR dataset.

        :param limit: the max of items to process

        """
        rows: List[Tuple[str, int]] = []
        idx: List[str] = []
        limit = sys.maxsize if limit is None else limit
        with self.resources.corpus() as r:
            k: str
            doc: AmrFeatureDocument
            for k, doc in it.islice(r.documents.items(), limit):
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
        fac: AmrSerializedFactory = self.resources.serialized_factory
        docs: Dict[str, Any] = {}
        k: str
        doc: AmrFeatureDocument
        with self.resources.corpus() as r:
            for k, doc in it.islice(r.documents.items(), limit):
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
        output_dir = Path('results') if output_dir is None else output_dir
        output_dir, use_stdout = self._prep_align(output_dir, render_level)
        with self.resources.corpus() as r:
            keys: Sequence[str]
            if keys == 'ALL':
                keys = tuple(r.keys())
            else:
                keys = re.findall(r'[^,\s]+', keys)
            if not use_stdout:
                output_dir.mkdir(parents=True, exist_ok=True)
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
        with self.resources.corpus() as r:
            # clear the annotated feature document stash
            r.documents.clear()


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
        #aligner: DocumentGraphAligner = self.doc_graph_aligner
        output_dir, use_stdout = self._prep_align(output_dir, render_level)
        success_keys: List[str] = []
        results: List[FlowGraphResult] = []
        with open(input_file) as f:
            docs: List[Dict[str, str]] = json.load(f)
        output_dir.mkdir(parents=True, exist_ok=True)
        with self.resources.adhoc(docs) as r:
            #dix: int
            did: str = None
            res: FlowGraphResult
            for did, res in r.alignments.items():
                print(type(res))
                try:
                    doc: AmrFeatureDocument = r.documents[did]
                    # doc_graph: DocumentGraph = self.resource.doc_graph_factory(doc)
                    #doc_graph: DocumentGraph = res.doc_graph
                    output_file: Path
                    if use_stdout:
                        output_file = stdout.STANDARD_OUT_PATH
                    else:
                        output_file: Path = self._get_output_file(
                            output_dir, did, output_format)
                    #res: FlowGraphResult = aligner.align(doc_graph)
                    #res: FlowGraphResult = aligner.align(doc_graph)
                    if res.is_error:
                        res.failure.rethrow()
                    with stdout(output_file, 'w') as fout:
                        self._output_align(output_format, doc, fout, res)
                    if not use_stdout:
                        logger.info(f'wrote: {output_file}')
                    results.append(res)
                    success_keys.append(did)
                except Exception as e:
                    msg: str = f'Can not align ID {did}: {e}'
                    logger.error(msg)
                    if not use_stdout:
                        with open(output_dir / 'errors.txt', 'a') as f:
                            f.write(f'error: {msg}\n')
                            traceback.print_exception(e, file=f)
                            f.write('_' * 80 + '\n')
            if not use_stdout and did is not None:
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
