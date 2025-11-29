import pickle
from io import BytesIO
from zensols.persist import Stash
from zensols.config import ConfigFactory
from zensols.amr import AmrFeatureSentence, AmrFeatureDocument
from zensols.amr.annotate import AnnotatedAmrDocument, AnnotatedAmrSentence
from zensols.calamr import (
    DocumentGraph, SentenceGraphNode, DocumentGraphFactory
)
from util import TestBase


class TestAnnotatedAmr(TestBase):
    def test_create_doc(self):
        stash: Stash = self.config_factory('amr_anon_doc_stash')
        stash.clear()
        self.assertTrue(len(tuple(stash.keys())) > 0)
        doc: AnnotatedAmrDocument = next(iter(stash.values()))
        self.assertEqual(AnnotatedAmrDocument, type(doc))

    def test_pickle(self):
        stash: Stash = self.config_factory('amr_anon_doc_stash')
        doc: AnnotatedAmrDocument = next(iter(stash.values()))
        anon = BytesIO()
        pickle.dump(doc, anon)
        anon.seek(0)
        doc2 = pickle.load(anon)
        for sold, snew in zip(doc, doc2):
            self.assertEqual(sold.graph_string, snew.graph_string)
            self.assertEqual(AnnotatedAmrSentence, type(sold))
            self.assertEqual(AnnotatedAmrSentence, type(snew))
            self.assertEqual(sold.doc_sent_idx, snew.doc_sent_idx)
            self.assertEqual(sold.sent_type, snew.sent_type)

    def _test_doc_share(self, dix: int, six: int, gix: int,
                        config: str = None, debug: bool = False):
        """Test sharing of parts of document in DocumentGraph."""
        if debug:
            print(f'conf: {config}')
        fac: ConfigFactory
        if config is None:
            fac = self.config_factory
        else:
            fac = self._get_config_factory(config)
        stash: Stash = fac('amr_anon_feature_doc_stash')
        gfac: DocumentGraphFactory = fac('calamr_doc_graph_factory')

        # get an annotated document
        doc: AmrFeatureDocument = stash['liu-example']
        self.assertEqual(AmrFeatureDocument, type(doc))
        if debug:
            print('doc:')
            for s in doc:
                s.write()
        doc_graph: DocumentGraph = gfac(doc)
        if debug:
            doc.write()
        self.assertEqual(AmrFeatureDocument, type(doc_graph.doc))

        # document in root of tree
        self.assertEqual(id(doc), id(doc_graph.doc))
        # first sentence across document and graph
        src = doc_graph.components_by_name['source'].root_node.doc
        if debug:
            print('_' * 80)
            print('src')
            src.write()
            print('_' * 40)
            print('doc')
            doc.write()
            print('_' * 80)
        self.assertEqual(id(src[six]), id(doc[dix]), f'{src[six]} != {doc[dix]}')
        # sentence contained in the sentence node
        sn = tuple(filter(lambda n: isinstance(n, SentenceGraphNode),
                          doc_graph.components_by_name['source'].vs.values()))
        if debug:
            print('sents from nodes:')
            for s in sn:
                print(s.sent.text)
        self.assertEqual(2, len(sn))
        snode: SentenceGraphNode = sn[gix]
        sent: AmrFeatureSentence = snode.sent
        self.assertEqual(AmrFeatureSentence, type(sent))
        self.assertEqual(id(doc[dix]), id(sent), f'{doc[dix]} != {sent}')

        # copy
        bio = BytesIO()
        pickle.dump(doc_graph, bio)
        bio.seek(0)
        doc_graph2 = pickle.load(bio)

        self.assertNotEqual(id(doc_graph), id(doc_graph2))
        doc2 = doc_graph2.doc
        self.assertNotEqual(id(doc), id(doc2))

        # document in root of tree
        self.assertEqual(id(doc2), id(doc_graph2.doc))
        # first sentence across document and graph
        src2 = doc_graph2.components_by_name['source'].root_node.doc
        self.assertNotEqual(id(src), id(src2))
        self.assertEqual(id(src2[six]), id(doc2[dix]), f'{src2[six]} != {doc2[dix]}')
        # sentence contained in the sentence node
        sn = tuple(filter(lambda n: isinstance(n, SentenceGraphNode),
                          doc_graph2.components_by_name['source'].vs.values()))
        self.assertEqual(2, len(sn))
        snode: SentenceGraphNode = sn[gix]
        sent: AmrFeatureSentence = snode.sent
        self.assertEqual(AmrFeatureSentence, type(sent))
        self.assertEqual(id(doc2[dix]), id(sent), f'{doc2[dix]} != {sent}')

    def test_doc_share_no_intermediary_nodes(self):
        self._test_doc_share(1, 0, 0)

    def test_doc_share_with_intermediary_nodes(self):
        self._clean_cache()
        self._copy_micro()
        self._test_doc_share(1, 1, 0, config='with-intermediary')
