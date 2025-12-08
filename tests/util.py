import unittest
from pathlib import Path
import sys
import shutil
from zensols.cli import CliHarness, ApplicationFailure
from zensols.persist import Stash
from zensols.config import ConfigFactory
# initialize the NLP system
from zensols import deepnlp
from zensols.amr import AmrFeatureDocument, suppress_warnings
from zensols.calamr import (
    Resources, ApplicationFactory, CorpusApplication,
    DocumentGraph, DocumentGraphFactory, DocumentGraphAligner,
)


class TestBase(unittest.TestCase):
    WRITE: bool = 0

    def setUp(self):
        suppress_warnings()
        self.maxDiff = sys.maxsize
        self._clean_cache()
        self.config_factory: ConfigFactory = self._get_config_factory()
        self._copy_micro()
        deepnlp.init()

    def _copy_micro(self):
        micro_file = Path('download/micro.txt.bz2')
        targ_micro_file = Path('target/download/micro.txt.bz2')
        if not micro_file.is_file():
            raise ValueError(
                f"Missing micro corpus file '{micro_file}'. Run `make micro`")
        if not targ_micro_file.is_file():
            targ_micro_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(micro_file, targ_micro_file)

    def _clean_cache(self):
        self.targ_dir = Path('target')
        if self.targ_dir.is_dir():
            shutil.rmtree(self.targ_dir)

    def _get_config_factory(self, config: str = None) -> ConfigFactory:
        app = self._get_app(config)
        app.write_adhoc_corpus()
        return app.config_factory

    def _get_app(self, config: str = None) -> CorpusApplication:
        from zensols.calamr import DocumentGraphAligner
        harn: CliHarness = ApplicationFactory.create_harness()
        config = 'no-intermediary' if config is None else config
        app = harn.get_application(
            args=f'--level=err --config=test-resources/{config}.conf',
            app_section='capp')
        if app is None:
            raise ValueError('Could not create application')
        if isinstance(app, ApplicationFailure):
            app.raise_exception()
        # turn off graphical rendering
        doc_graph_aligner: DocumentGraphAligner = \
            app.config_factory('calamr_resources').doc_graph_aligner
        doc_graph_aligner.render_level = 0
        return app

    def _get_doc_graph(self) -> DocumentGraph:
        res: Resources = self.config_factory('calamr_resources')
        with res.corpus() as r:
            stash: Stash = r.documents
            doc_graph_factory: DocumentGraphFactory = self.config_factory(
                'calamr_doc_graph_factory')
            self.assertTrue(len(tuple(stash.keys())) > 0)
            doc: AmrFeatureDocument = stash['liu-example']
            self.assertEqual(AmrFeatureDocument, type(doc))
            return doc_graph_factory(doc)

    def _get_doc_graph_aligner(self) -> DocumentGraphAligner:
        doc_graph_aligner: DocumentGraphAligner = \
            self.config_factory('calamr_resources').doc_graph_aligner
        doc_graph_aligner.render_level = 0
        return doc_graph_aligner
