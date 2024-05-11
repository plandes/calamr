from typing import Dict, Set, Any
import pickle
from io import BytesIO
import pandas as pd
from zensols import deepnlp
from zensols.calamr import (
    GraphAttribute, GraphComponent, GraphNode, GraphEdge,
    DocumentGraph, FlowGraphResult, DocumentGraphAligner,
)
from util import TestBase


# initialize the NLP system
deepnlp.init()


class TestAnnotatedAmr(TestBase):
    def _copy_by_pickle(self, obj: Any) -> Any:
        bio = BytesIO()
        pickle.dump(obj, bio)
        bio.seek(0)
        return pickle.load(bio)

    def _cmp_doc_graph(self, doc_graph: DocumentGraph, n_vertexes, n_edges):
        self.assertEqual(n_vertexes, len(doc_graph.vs))
        self.assertEqual(n_edges, len(doc_graph.es))

        # node instance matches between graph components
        cnt = 0
        for comp in doc_graph.components:
            for gn in comp.vs.values():
                for gn2 in doc_graph.vs.values():
                    if id(gn) == id(gn2):
                        cnt += 1
        self.assertEqual(n_vertexes, cnt)

        # edge instance matches between graph components
        cnt = 0
        for comp in doc_graph.components:
            for gn in comp.es.values():
                for gn2 in doc_graph.es.values():
                    if id(gn) == id(gn2):
                        cnt += 1
        self.assertEqual(n_edges, cnt)

        # node instance matches between sentence index
        cnt = 0
        for e in doc_graph.components_by_name['source'].sent_index.entries:
            for gn in doc_graph.vs.values():
                if str(gn) == str(e.node):
                    cnt += 1
        self.assertEqual(2, cnt)

    def _fuzzyEqual(self, should: float, x: float, epsilon: float = 1e-2):
        diff: float = abs(should - x)
        self.assertTrue(
            diff < epsilon,
            msg=f'{should} is with a distance of {epsilon} with {x}')

    def _cmp_align_stats(self, stats: Dict[str, Any]):
        comps = stats['components']
        agg = stats['agg']
        self.assertEqual(21, agg['tot_alignable'])
        self.assertEqual(18, agg['tot_aligned'])
        self._fuzzyEqual(0.8695652173913044, agg['aligned_portion_hmean'])
        self._fuzzyEqual(0.7131309357900468, agg['mean_flow'])

        src_comp = comps['source']
        src_comp_on = src_comp['connected']
        src_comp_cnt = src_comp['counts']
        self.assertEqual(13, src_comp_on['alignable'])
        self.assertEqual(10, src_comp_on['aligned'])
        self._fuzzyEqual(0.7692307692307693, src_comp_on['aligned_portion'])
        self.assertEqual({'edge': {'role': 9,
                                   'sentence': 4,
                                   'doc': 0},
                          'node': {'attribute': 1,
                                   'concept': 10,
                                   'doc': 1,
                                   'sentence': 2}},
                         src_comp_cnt)

        smy_comp = comps['summary']
        smy_comp_on = smy_comp['connected']
        smy_comp_cnt = smy_comp['counts']
        self.assertEqual(8, smy_comp_on['alignable'])
        self.assertEqual(8, smy_comp_on['aligned'])
        self._fuzzyEqual(1.0, smy_comp_on['aligned_portion'])
        self.assertEqual({'edge': {'role': 6,
                                   'sentence': 2,
                                   'doc': 0},
                          'node': {'attribute': 1,
                                   'concept': 6,
                                   'doc': 1,
                                   'sentence': 1}},
                         smy_comp_cnt)

    def test_doc_graph_build(self):
        doc_graph: DocumentGraph = self._get_doc_graph()
        self._cmp_doc_graph(doc_graph, 23, 21)
        # save test run time by testing pickle in the same method
        doc_graph: DocumentGraph = self._get_doc_graph()
        doc_graph_copy = self._copy_by_pickle(doc_graph)
        self.assertNotEqual(id(doc_graph), id(doc_graph_copy))
        for comp in doc_graph.components:
            comp_copy = doc_graph_copy.components_by_name[comp.name]
            self.assertNotEqual(id(comp), id(comp_copy))
        self._cmp_doc_graph(doc_graph_copy, 23, 21)

    def _assert_no_missing(self, doc_graph: DocumentGraph):
        gn: GraphNode
        for gn in doc_graph.vs.values():
            self.assertTrue(gn is not None)
        ge: GraphEdge
        for ge in doc_graph.es.values():
            self.assertTrue(ge is not None)

    def _assert_share(self, a: DocumentGraph, b: DocumentGraph):
        def gid_to_node(dg: DocumentGraph, gid: int) -> GraphNode:
            try:
                return dg.node_by_graph_node_id(gid)
            except Exception:
                if 0:
                    import traceback
                    traceback.print_exc()

        def gid_to_edge(dg: DocumentGraph, gid: int) -> GraphEdge:
            try:
                return dg.edge_by_graph_edge_id(gid)
            except Exception:
                if 0:
                    import traceback
                    traceback.print_exc()

        # node instance matches between graph and orignal graph
        gn_ids = set(map(lambda gn: gn.id, a.vs.values())) | \
            set(map(lambda gn: gn.id, a.vs.values()))
        ge_ids = set(map(lambda ge: ge.id, a.es.values())) | \
            set(map(lambda ge: ge.id, a.es.values()))
        self.assertTrue(len(gn_ids) > 0)
        self.assertTrue(len(ge_ids) > 0)
        cmps = 0
        for gid in gn_ids:
            ag: GraphNode = gid_to_node(a, gid)
            bg: GraphNode = gid_to_node(b, gid)
            if ag is None or bg is None:
                continue
            self.assertEqual(id(ag), id(bg))
            cmps += 1
        self.assertTrue(cmps > 0)
        cmps = 0
        for gid in ge_ids:
            ag: GraphEdge = gid_to_edge(a, gid)
            bg: GraphEdge = gid_to_edge(b, gid)
            if ag is None or bg is None:
                continue
            self.assertEqual(id(ag), id(bg))
            cmps += 1
        self.assertTrue(cmps > 0)

    def _assert_comps(self, doc_graph: DocumentGraph):
        self.assertEqual(2, len(doc_graph.components))
        for comp in doc_graph.components:
            self.assertTrue(comp.root_node is not None)
            self.assertTrue(comp.sent_index is not None)
            self.assertTrue(len(comp.sent_index.entries) > 0)
            self.assertTrue(comp.description is not None)
            self._assert_no_missing(comp)
            self._assert_share(doc_graph, comp)

    def _get_ids(self, doc_graph: DocumentGraph) -> Set[int]:
        ids: Set[int] = set()
        comp: GraphComponent
        for comp in doc_graph.component_iter():
            assert all(map(lambda c: isinstance(c, GraphAttribute),
                           comp.get_attributes()))
            ids.update(map(lambda c: c.id, comp.get_attributes()))
        return ids

    def _get_mem_ids(self, doc_graph: DocumentGraph) -> Set[int]:
        ids: Set[int] = set()
        comp: GraphComponent
        for comp in doc_graph.component_iter():
            assert all(map(lambda c: isinstance(c, GraphAttribute),
                           comp.get_attributes()))
            ids.update(map(lambda c: id(c), comp.get_attributes()))
        return ids

    def test_flow(self):
        doc_graph_aligner: DocumentGraphAligner = self._get_doc_graph_aligner()
        doc_graph: DocumentGraph = self._get_doc_graph()
        res: FlowGraphResult = doc_graph_aligner.align(doc_graph)
        self._cmp_align_stats(res.stats)
        for comp in doc_graph.components:
            self._assert_share(doc_graph, comp)

    def test_pickle_graph(self):
        doc_graph: DocumentGraph = self._get_doc_graph()
        doc_graph = self._copy_by_pickle(doc_graph)
        self._assert_no_missing(doc_graph)

    def test_pickle_components(self):
        doc_graph: DocumentGraph = self._get_doc_graph()
        doc_graph = self._copy_by_pickle(doc_graph)
        self._assert_comps(doc_graph)

    def test_pickle_children(self):
        doc_graph_aligner: DocumentGraphAligner = self._get_doc_graph_aligner()
        doc_graph: DocumentGraph = self._get_doc_graph()
        res: FlowGraphResult = doc_graph_aligner.align(doc_graph)
        res = self._copy_by_pickle(res)
        doc_graph = res.doc_graph
        for child_doc_graph in doc_graph.children.values():
            self._assert_comps(child_doc_graph)
            self._assert_share(doc_graph, child_doc_graph)
            for child_comp in child_doc_graph.components:
                self._assert_share(doc_graph, child_comp)

    def test_flow_pickle_stats(self):
        doc_graph_aligner: DocumentGraphAligner = self._get_doc_graph_aligner()
        doc_graph: DocumentGraph = self._get_doc_graph()
        res: FlowGraphResult = doc_graph_aligner.align(doc_graph)
        df: pd.DataFrame = res.df
        should_ids: Set[int] = self._get_ids(res.doc_graph)
        should_mem_ids: Set[int] = self._get_mem_ids(res.doc_graph)
        self.assertTrue(len(should_ids) > 0)
        self.assertTrue(len(should_mem_ids) > 0)
        res = self._copy_by_pickle(res)
        self._cmp_align_stats(res.stats)
        df_copy: pd.DataFrame = res.df
        dfd: pd.DataFrame = df_copy.compare(df)
        self.assertEqual(0, len(dfd))
        self.assertEqual(should_ids, self._get_ids(res.doc_graph))
        self.assertNotEqual(should_mem_ids, self._get_mem_ids(res.doc_graph))
        self.assertEqual(len(should_mem_ids), len(self._get_mem_ids(res.doc_graph)))

    def test_flow_pickle_double(self):
        doc_graph_aligner: DocumentGraphAligner = self._get_doc_graph_aligner()
        prev_doc_graph: DocumentGraph = self._get_doc_graph()
        doc_graph = self._copy_by_pickle(prev_doc_graph)
        doc_graph.graph_attrib_context = prev_doc_graph.graph_attrib_context
        res: FlowGraphResult = doc_graph_aligner.align(doc_graph)
        df: pd.DataFrame = res.df
        res = self._copy_by_pickle(res)
        self._cmp_align_stats(res.stats)
        df_copy: pd.DataFrame = res.df
        dfd: pd.DataFrame = df_copy.compare(df)
        self.assertEqual(0, len(dfd))
        doc_graph = res.doc_graph
        self._assert_comps(doc_graph)
        for child_doc_graph in doc_graph.children.values():
            self._assert_comps(child_doc_graph)
            self._assert_share(doc_graph, child_doc_graph)
            for child_comp in child_doc_graph.components:
                self._assert_share(doc_graph, child_comp)
