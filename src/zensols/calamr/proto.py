"""Prototyping and cookbook.

"""
__author__ = 'PaulLandes'

from typing import Dict, Any, List, Tuple, ClassVar
from dataclasses import dataclass, field
from pathlib import Path
import itertools as it
import pandas as pd
from zensols.config import ConfigFactory
from zensols.datdesc import DataDescriber
from zensols.amr import AmrFeatureDocument, Format
from zensols.amr.annotate import AnnotatedAmrFeatureDocumentFactory
from . import DocumentGraph, DocumentGraphAligner, FlowGraphResult
from .app import _AlignmentBaseApplication


@dataclass
class _ProtoApplication(_AlignmentBaseApplication):
    """An application entry point for prototyping.

    """
    CLI_META: ClassVar[Dict[str, Any]] = {'is_usage_visible': False}

    config_factory: ConfigFactory = field()
    """For prototyping."""

    def _get_doc(self):
        # proxy corpus
        k = '20080705_0216'
        if k not in self.resource.anon_doc_stash:
            # adhoc
            k = 'liu-example'  # original 2014 Lui et al example
        if k not in self.resource.anon_doc_stash:
            # little prince
            k = '1943'
        if k not in self.resource.anon_doc_stash:
            # bio
            k = '1563_0473'
        if k not in self.resource.anon_doc_stash:
            keys = tuple(self.resource.anon_doc_stash.keys())
            raise ValueError(f'no key: {k} in {keys}')
        return self.resource.anon_doc_stash[k]

    def _write_default_doc(self):
        doc = self._get_doc()
        self.resource.serialized_factory(doc.amr).write()

    def _align(self, render_level: int = 5, write: bool = True,
               set_render_loops: bool = False):
        doc: AmrFeatureDocument = self._get_doc()
        doc_graph: DocumentGraph = self.resource.doc_graph_factory(doc)
        self.resource.doc_graph_aligner.render_level = render_level
        if set_render_loops:
            self.resource.doc_graph_aligner.init_loops_render_level = \
                render_level
        res: FlowGraphResult = self.resource.doc_graph_aligner.align(doc_graph)
        if write:
            res.write()
        return res

    def _read_meta_file(self, path: Path = Path('corpus/micro/source.json'),
                        show: str = 'align', limit: int = 1):
        factory: AnnotatedAmrFeatureDocumentFactory = \
            self.config_factory('amr_anon_doc_factory')
        aligner: DocumentGraphAligner = self.resource.doc_graph_aligner
        doc: AmrFeatureDocument
        for doc in it.islice(factory(path), limit):
            doc_graph: DocumentGraph = self.resource.doc_graph_factory(doc)
            if show == 'align':
                res: FlowGraphResult = aligner.align(doc_graph)
                doc.amr.write(limit_sent=0)
                res.write()
            elif show == 'long':
                doc.write()
            else:
                print(doc)

    def _proxy_key_splits(self):
        ks = self.config_factory('calamr_amr_corp_split_keys')
        ks.write()

    def _render_micro_by_size(self, size: int = 0, render: bool = False,
                              clear: bool = False):
        from zensols.util import time
        keys: Tuple[str, ...] = ('liu-example', 'no-summary', 'aurora-borealis',
                                 'falling-man', 'earthquake')
        # make earthquake more well connected
        # neighbor_embedding_weights: [0, 1, 0.6, 0.5, 0.3, 0.2, 0.1]
        # neighbor_skew: {y_trans: -0.5, x_trans: 0, compress: 1}
        stash = self.config_factory('calamr_flow_graph_result_stash')
        if clear:
            stash.clear()
        with time():
            res: FlowGraphResult = stash[keys[size]]
            #res.render(res.get_render_contexts(include_nascent=False))
            if render or 1:
                from .render.base import RenderContext, rendergroup
                renderer = self.resource.doc_graph_aligner.renderer
                with rendergroup(renderer) as rg:
                    ctx: RenderContext
                    for ctx in res.get_render_contexts():
                        rg(ctx)
        res.write()

    def _write_flows(self):
        doc: AmrFeatureDocument = self._get_doc()
        doc.amr.write(limit_sent=0)
        doc_graph: DocumentGraph = self.resource.doc_graph_factory(doc)
        res: FlowGraphResult = self.resource.doc_graph_aligner.align(doc_graph)
        df: pd.DataFrame = res.df
        res.write()
        print(df)

    def _render_corpus_data(self, limit: int = 2) -> DataDescriber:
        from typing import Union
        from zensols.util import Failure
        from zensols.datdesc import DataFrameDescriber
        from zensols.rend import ApplicationFactory

        stash = self.config_factory('calamr_flow_graph_result_stash')
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
        br = ApplicationFactory.get_browser_manager()
        br(dfd)

    def _align_corpus_doc(self):
        self.resource.align_corpus_document('liu-example', False)

    def _render_amr_document(self):
        from .render.base import rendergroup, RenderContext
        doc: AmrFeatureDocument = self._get_doc()
        doc.amr.write(include_amr=False)
        doc_graph: DocumentGraph = self.resource.doc_graph_factory(doc)
        renderer = self.resource.doc_graph_aligner.renderer
        with rendergroup(renderer, graph_id='graph') as rg:
            rg(RenderContext(doc_graph=doc_graph))

    def _iterate_align_tokens(self):
        import pickle
        import json
        from . import GraphNode, SentenceGraphAttribute, Flow, FlowDocumentGraph

        def tok_aligns(node: GraphNode) -> str:
            spans: Tuple[Tuple[str, Tuple[int, int]], ...] = None
            if isinstance(node, SentenceGraphAttribute):
                spans = tuple(map(
                    lambda t: (t.norm, t.lexspan.astuple), node.tokens))
            spans = None if spans is not None and len(spans) == 0 else spans
            return None if spans is None else json.dumps(spans)

        cached_file = Path('tmp.pkl')
        #cached_file.unlink()
        if not cached_file.is_file():
            doc: AmrFeatureDocument = self._get_doc()
            doc.amr.write(limit_sent=0)
            doc_graph: DocumentGraph = self.resource.doc_graph_factory(doc)
            res: FlowGraphResult = self.resource.align(doc_graph)
            with open(cached_file, 'wb') as f:
                pickle.dump(res, f)
        with open(cached_file, 'rb') as f:
            res = pickle.load(f)
        #res.write()
        if 0:
            self.resource.restore(res)
            res.render()
        doc_graph = res.doc_graph
        flow_doc_graph: FlowDocumentGraph = res.doc_graph.children['reversed_source']
        for cname, graph in flow_doc_graph.components_by_name.items():
            flow: Flow
            for flow in graph.flows:
                src: str = tok_aligns(flow.source)
                trg: str = tok_aligns(flow.target)
                print(f'{flow}: {src} -> {trg}')
        print('_' * 80)
        print(res.df['name s_descr s_toks'.split()])

    def proto(self, run: int = 6):
        """Prototype test."""
        return {
            1: self._write_default_doc,
            2: lambda: self._read_meta_file(show='align', limit =3),
            3: self._render_corpus_data,
            4: lambda: self._align(render_level=5),
            5: lambda: self.align('20080717_0594', Path('-'), Format.txt),
            6: self._iterate_align_tokens,
        }[run]()
