import sys
import unittest
import re
from io import StringIO
from pathlib import Path
import shutil
import pandas as pd
from zensols import deepnlp; deepnlp.init()
from zensols.cli import CliHarness
from zensols.calamr import FlowGraphResult, Resource, ApplicationFactory


class TestScore(unittest.TestCase):
    def setUp(self):
        # initialize the NLP system
        self.maxDiff = sys.maxsize

    def _test_align(self, key: str, resource: Resource, write: bool = False):
        should_file = Path(f'test-resources/should-align-results-{key}.txt')
        res: FlowGraphResult = resource.align_corpus_document(key, False)
        sio = StringIO()
        res.write(writer=sio)
        actual: str = sio.getvalue()
        # reduce precision to 3 significant digits
        actual = '\n'.join(map(
            lambda s: re.sub(r'^(\s+(mean|root)_flow: [0-9.]{4})[0-9.]+', r'\1', s),
            actual.split('\n')))
        if write:
            with open(should_file, 'w') as f:
                f.write(actual)
        with open(should_file) as f:
            should: str = f.read()
        self.assertEqual(should, actual)

    def test_align(self):
        resource = ApplicationFactory.get_resource('--level err')
        self._test_align('earthquake', resource)
        self._test_align('aurora-borealis', resource)

    def test_score(self):
        WRITE: bool = False
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
        if WRITE:
            df.to_csv(should_file)
        dfs = pd.read_csv(should_file, index_col=0)
        sigdig = 15
        df = df.round(sigdig)
        dfs = dfs.round(sigdig)
        dfc = dfs.compare(df)
        if 0:
            dfc.to_csv('target/diff.csv')
        self.assertEqual(0, len(dfc), f'self is previous results\n{dfc}')
