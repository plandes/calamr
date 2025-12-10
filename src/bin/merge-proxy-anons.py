#!/usr/bin/env python

from typing import Dict
from dataclasses import dataclass
import logging
from pathlib import Path
from zensols.cli import CliHarness
from zensols.config import ConfigFactory
from zensols.install import Installer
from zensols.amr import AmrSentence, AmrDocument

logger = logging.getLogger(__name__)


@dataclass
class CliProxyMerger(object):
    """Creates a Proxy report corpus file that merges everything from the
    ``data/amrs`` AMRs with the alignments in ``data/alignments``.  This creates
    a corpus with both the alignments and the sentence type (source vs. summary
    ``snt-type``) tags.

    """
    def _splice(self, amrs_file: Path, aligns_file: Path) -> AmrDocument:
        """Add all metadata by key to alignment AMRs from the ``data/amrs`` that
        have the ``snt-type`` tags.

        """
        amr_doc: AmrDocument = AmrDocument.from_source(amrs_file)
        align_doc: AmrDocument = AmrDocument.from_source(aligns_file)
        amrs: Dict[str, AmrSentence] = dict(map(
            lambda s: (s.metadata['id'], s), amr_doc.sents))
        sent: AmrSentence
        for sent in align_doc:
            meta: Dict[str, str] = sent.metadata
            amr_sent: AmrSentence = amrs[meta['id']]
            # the alignments are epigraph data so we should have the same number
            # of triples between the two corpus files
            assert len(sent.graph.triples) == len(amr_sent.graph.triples)
            meta['snt-type'] = amr_sent.metadata['snt-type']
            meta['snt-org'] = amr_sent.metadata['snt']
            # the alignments file has the tokenized sentence as metadata
            meta['snt'] = meta['tok'].replace('@-@', '-')
            sent.metadata = meta
        return align_doc

    def create(self):
        harness = CliHarness(
            src_dir_name='src',
            app_factory_class='zensols.calamr.ApplicationFactory')
        fac: ConfigFactory = harness.get_config_factory(
            ['--override', 'calamr_corpus.name=proxy-report'])
        harness.configure_logging(
            loggers={__name__: 'info',
                     'zensols.amr': 'info',
                     'zensols.amr.score': 'debug'})
        # install the corpus if not already
        installer: Installer = fac('amr_anon_corpus_installer')
        installer()
        merge_path: Path = installer.get_singleton_path()
        # probably resolve to same path, but be robust on configured file
        merge_file = (merge_path / '../../../merge/unsplit/amr-release-3.0-alignments-proxy.txt').resolve()
        # has the `snt-type` tags
        amrs_file = (merge_path / '../../../amrs/unsplit/amr-release-3.0-amrs-proxy.txt').resolve()
        # has the alignments, both as metadata but using epigraph notation (`~`)
        aligns_file = (merge_path / '../../../alignments/unsplit/amr-release-3.0-alignments-proxy.txt').resolve()
        assert amrs_file.is_file()
        assert aligns_file.is_file()
        merged_doc: AmrDocument = self._splice(amrs_file, aligns_file)
        merge_file.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f'merging: {amrs_file}, {aligns_file} -> {merge_file}')
        with open(merge_file, 'w') as f:
            merged_doc.write(writer=f)
        logger.info(f'wrote: {merge_file}')


if (__name__ == '__main__'):
    CliProxyMerger().create()
