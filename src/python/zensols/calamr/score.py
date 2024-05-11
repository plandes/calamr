"""Produces CALAMR scores.

"""
__author__ = 'Paul Landes'

from typing import Iterable, Tuple, Dict, Any, List, Union
from dataclasses import dataclass, field
from collections import OrderedDict
import logging
from zensols.nlp.score import ErrorScore, ScoreMethod, ScoreContext, Score
from zensols.amr import AmrFeatureSentence, AmrFeatureDocument
from zensols.amr.annotate import (
    SentenceType, AnnotatedAmrSentence, AnnotatedAmrDocument,
)
from zensols.deepnlp.transformer import (
    WordPieceFeatureDocumentFactory, WordPieceFeatureDocument
)
from . import (
    DocumentGraph, DocumentGraphFactory, DocumentGraphAligner, FlowGraphResult
)

logger = logging.getLogger(__name__)


@dataclass
class CalamrScore(Score):
    """Contains all CALAMR scores.

    """
    flow_graph_res: FlowGraphResult = field(repr=False)

    def _from_dictable(self, *args, **kwargs) -> Dict[str, Any]:
        a_cols: List[str] = 'mean_flow aligned_portion'.split()
        stats: Dict[str, Any] = self.flow_graph_res.stats
        agg: Dict[str, Any] = stats['agg']
        ret: Dict[str, Any] = OrderedDict({f'agg_{k}': agg[k] for k in a_cols})
        ret['bipartite_relations'] = stats['bipartite_relations']
        for cname, val in stats['components'].items():
            ret[f'{cname}_aligned_portion'] = \
                val['connected']['aligned_portion']
            vk: str
            for vk in val.keys():
                if vk == 'connected' or vk == 'counts':
                    continue
                ret[f'{cname}_{vk}'] = val[vk]
            ck: str
            dv: Dict[str, int]
            for ck, dv in val['counts'].items():
                dk: str
                cnt: int
                for dk, cnt in dv.items():
                    ret[f'{cname}_{ck}_{dk}_count'] = cnt
        return ret

    def __str__(self) -> str:
        return ', '.join(map(lambda s: f'{s[0]}: {s[1]:.4f}',
                             self.asflatdict().items()))


CalamrScore.NAN_INSTANCE = CalamrScore(FlowGraphResult(None, None, None))


@dataclass
class CalamrScoreMethod(ScoreMethod):
    """Computes the smatch scores of AMR sentences.  Sentence pairs are ordered
    ``(<summary>, <source>)``.

    """
    word_piece_doc_factory: WordPieceFeatureDocumentFactory = field(
        default=None)
    """The feature document factory that populates embeddings."""

    doc_graph_factory: DocumentGraphFactory = field(default=None)
    """Create document graphs."""

    doc_graph_aligner: DocumentGraphAligner = field(default=None)
    """Create document graphs."""

    def _to_annotated_sent(self, sent: AmrFeatureSentence,
                           sent_type: SentenceType) -> \
            Tuple[AmrFeatureSentence, bool]:
        mod: bool = False
        if not isinstance(sent.amr, AnnotatedAmrSentence):
            asent = sent.amr.clone(
                cls=AnnotatedAmrSentence,
                sent_type=sent_type,
                doc_sent_idx=0)
            sent = sent.clone()
            sent.amr = asent
            mod = True
        return sent, mod

    def _populate_embeddings(self, doc: AmrFeatureDocument):
        """Adds the transformer sentinel embeddings to the document."""
        wpdoc: WordPieceFeatureDocument = self.word_piece_doc_factory(doc)
        wpdoc.copy_embedding(doc)

    def score_annotated_doc(self, doc: AmrFeatureDocument) -> CalamrScore:
        """Score a document that has an
        :obj:`~zensols.amr.container.AmrFeatureDocument.amr` of type
        :class:`~zensols.amr.annotate.AnnotatedAmrDocument`.

        :raises: [zensols.amr.domain.AmrError]: if the AMR could not be parsed
                 or aligned

        """
        assert isinstance(doc, AmrFeatureDocument)
        doc_graph: DocumentGraph = self.doc_graph_factory(doc)
        prev_render_level: int = self.doc_graph_aligner.render_level
        self.doc_graph_aligner.render_level = 0
        try:
            fgr: FlowGraphResult = self.doc_graph_aligner.align(doc_graph)
            return CalamrScore(fgr)
        finally:
            self.doc_graph_aligner.render_level = prev_render_level

    def score_pair(self, smy: Union[AmrFeatureSentence, AmrFeatureDocument],
                   src: Union[AmrFeatureSentence, AmrFeatureDocument]) -> \
            CalamrScore:
        if isinstance(src, AmrFeatureDocument) and \
           isinstance(smy, AmrFeatureDocument):
            src_mod, smy_mod = False, False
            fsents = tuple(list(src.sents) + list(smy.sents))
            asents = tuple(map(lambda s: s.amr, fsents))
            fdoc = AmrFeatureDocument(sents=fsents)
            fdoc.amr = AnnotatedAmrDocument(sents=asents)
        else:
            assert isinstance(src, AmrFeatureSentence)
            assert isinstance(smy, AmrFeatureSentence)
            src, src_mod = self._to_annotated_sent(src, SentenceType.BODY)
            smy, smy_mod = self._to_annotated_sent(smy, SentenceType.SUMMARY)
            fdoc = AmrFeatureDocument(sents=(smy, src))
            fdoc.amr = AnnotatedAmrDocument(sents=(smy.amr, src.amr))
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'scoring <{smy}> :: <{src}>')
        if src_mod or smy_mod:
            self._populate_embeddings(fdoc)
        return self.score_annotated_doc(fdoc)

    def _score(self, meth: str, context: ScoreContext) -> Iterable[CalamrScore]:
        smy: AmrFeatureSentence
        src: AmrFeatureSentence
        for smy, src in context.pairs:
            try:
                yield self.score_pair(smy, src)
            except Exception as e:
                logger.error(f'can not score: <{src}>::<{smy}>: {e}',
                             stack_info=True, exc_info=True)
                yield ErrorScore(meth, e, CalamrScore.NAN_INSTANCE)

    def clear(self):
        self.doc_graph_aligner.clear()
