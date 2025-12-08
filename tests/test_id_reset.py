from zensols.calamr import DocumentGraph
from util import TestBase


class TestIdReset(TestBase):
    def test_id_reset(self):
        doc_graph: DocumentGraph = self._get_doc_graph()
        doc_graph2: DocumentGraph = self._get_doc_graph()
        self.assertNotEqual(id(doc_graph), id(doc_graph2))

        ids = sorted(map(lambda gn: gn.id, doc_graph.es.values()))
        ids2 = sorted(map(lambda gn: gn.id, doc_graph2.es.values()))
        self.assertEqual(ids, ids2)
