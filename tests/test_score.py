import sys
import re
import warnings
from io import StringIO
from pathlib import Path
import shutil
import pandas as pd
from zensols.cli import CliHarness
from zensols.amr import AmrFeatureDocument
from zensols.calamr import (
    FlowGraphResult, Resources, ApplicationFactory, DocumentGraph
)
from util import TestBase


class TestScore(TestBase):
    def setUp(self):
        super().setUp()
        warnings.filterwarnings(
            'ignore',
            message=r'Deprecated call to `pkg_resources',
            category=DeprecationWarning)
        warnings.filterwarnings(
            'ignore',
            message=r'^pkg_resources is deprecated as an API',
            category=UserWarning)

    def _test_align(self, key: str, resources: Resources):
        should_file = Path(f'test-resources/should-align-results-{key}.txt')
        with resources.corpus() as r:
            doc: AmrFeatureDocument = r.documents[key]
            doc_graph: DocumentGraph = resources.doc_graph_factory(doc)
            res: FlowGraphResult = resources.doc_graph_aligner.align(doc_graph)
            sio = StringIO()
            res.write(writer=sio)
            actual: str = sio.getvalue()
            # reduce precision to 3 significant digits
            actual = '\n'.join(map(
                lambda s: re.sub(r'^(\s+(mean|root)_flow: [0-9.]{4})[0-9.]+', r'\1', s),
                actual.split('\n')))
            if self.WRITE:
                with open(should_file, 'w') as f:
                    f.write(actual)
            with open(should_file) as f:
                should: str = f.read()
            self.assertEqual(should, actual)

    def test_score_align(self):
        if sys.version_info.minor >= 11:
            # disable tests under Python 3.10 since frozendict formats
            # incorrectly in Writable.write
            resource = self._get_app().resources
            self._test_align('earthquake', resource)
            self._test_align('aurora-borealis', resource)

    def test_score(self):
        out_file = Path('target/score.csv')
        res_dir = Path('test-resources')
        should_file = res_dir / 'score.csv'
        if out_file.parent.is_dir():
            shutil.rmtree(out_file.parent)
        harn: CliHarness = ApplicationFactory.create_harness()
        harn.execute(
            '--level=err --config=test-resources/no-intermediary.conf score ' +
            f'{res_dir}/lp-gold.txt ' +
            f'--parsed {res_dir}/lp-parsed.txt --out {out_file}')
        df = pd.read_csv(out_file)
        if self.WRITE:
            df.to_csv(should_file)
        dfs = pd.read_csv(should_file, index_col=0)
        sigdig = 2
        df = df.round(sigdig)
        dfs = dfs.round(sigdig)
        dfc = dfs.compare(df)
        if 0:
            dfc.to_csv('target/diff.csv')
        self.assertEqual(0, len(dfc), f'self is previous results\n{dfc}')
