"""Contains classes that run the algorithm to compute graph component
alignments.

"""
from __future__ import annotations
__author__ = 'Paul Landes'
from typing import Dict, Any, Sequence, List, Type, Iterable, Optional, ClassVar
from dataclasses import dataclass, field
from abc import ABC, ABCMeta, abstractmethod
import logging
import collections
from pathlib import Path
import re
import json
from itertools import chain
from zensols.util import time
from zensols.config import ConfigFactory, Dictable
from zensols.datdesc.hyperparam import HyperparamModel
from zensols.amr import AmrFeatureDocument
from .render.base import GraphRenderer, RenderContext, rendergroup
from . import (
    ComponentAlignmentError, ComponentAlignmentFailure,
    DocumentGraph, FlowGraphResult
)

logger = logging.getLogger(__name__)


@dataclass
class DocumentGraphAligner(ABC):
    """Aligns the graph components of ``doc_graph`` and visualizes them with
    :obj:`renderer`.

    """
    MAX_RENDER_LEVEL: ClassVar[int] = 10
    """The maximum value for :obj:`render_level`."""

    config_factory: ConfigFactory = field()
    """Used to create the :class:`.GraphSequencer` instance."""

    flow_graph_result_name: str = field()
    """The app configuration section name of :class:`.FlowGraphResult`."""

    doc_graph_name: str = field()
    """The :obj:`.DocumentGraph.name` document return from :meth:`align`."""

    renderer: GraphRenderer = field()
    """Visually render the graph in to a human understandable presentation."""

    render_level: int = field()
    """How many graphs to render on a scale from 0 - 10.  The higher the number
    the more likely a graph is to be rendered.  A value of 0 prevents rendering
    and a setting of 10 will render all graphs.

    :see: :obj:`MAX_RENDER_LEVEL`

    """
    init_loops_render_level: int = field()
    """The :obj:`render_level` to use for all iteration loops except for the
    last before the algorithm converges.

    """
    output_dir: Path = field()
    """If this is set, the graphs are written to this created directory on the
    file system.  Otherwise, they are displayed and cleaned up afterward.

    """
    @classmethod
    def is_valid_render_level(cls: Type, render_level: int,
                              should_raise: bool = False) -> bool:
        """Return whether ``render_level`` is a valid value for
        :obj:`render_level.

        """
        ml: int = cls.MAX_RENDER_LEVEL
        valid: bool = render_level >= 0 and render_level <= ml
        if should_raise and not valid:
            msg = f'Valid render levels are in [0 - {ml}], got: {render_level}'
            raise ComponentAlignmentError(msg)
        return valid

    @abstractmethod
    def _align(self, doc_graph: DocumentGraph, render: rendergroup) -> \
            FlowGraphResult:
        """See :meth:`.align`."""
        pass

    def create_error_result(self, ex: Exception,
                            msg: str = 'Could not align') -> FlowGraphResult:
        """Create an error graph result (rather than an alignment result).  This
        should be called in a try/catch to obtain the error information.

        :param ex: the exception that caused the issue

        :param: msg: the error message for the failure

        """
        return self.config_factory.new_instance(
            self.flow_graph_result_name,
            data=ComponentAlignmentFailure(exception=ex, message=msg))

    def align(self, doc_graph: DocumentGraph) -> FlowGraphResult:
        """Align the graph components of ``doc_graph`` and optionally visualize
        them with :obj:`renderer`.  To disable rendering, set
        :obj:`render_level` to 0.

        :param doc_graph: the graph created by the
                          :class:`.DocumentGraphFactory`

        :return: the alignments available as the in memory graph and object
                 graph, Pandas dataframes, statistics and scores

        """
        def _align(doc_graph: DocumentGraph, render) -> FlowGraphResult:
            try:
                return self._align(doc_graph, render)
            except Exception as e:
                return self.create_error_result(e)

        DocumentGraphAligner.is_valid_render_level(self.render_level, True)
        msg: str = 'constructed and aligned graph for all iterations'
        if self.render_level > 0:
            with rendergroup(
                    self.renderer,
                    directory=self.output_dir,
                    display=self.output_dir is None,
            ) as render:
                with time(msg):
                    return _align(doc_graph, render)
        else:
            with time(msg):
                return _align(doc_graph, None)


@dataclass
class DocumentGraphController(Dictable, metaclass=ABCMeta):
    """Executes the maxflow/min cut algorithm on a document graph.

    """
    _TRIM_PACKGE_PREFIX: ClassVar[re.Pattern] = re.compile(
        r'^calamr_(.+)_controller$')

    name: str = field()
    """The configuration instance name for debugging"""

    def __post_init__(self):
        name: str = re.sub(self._TRIM_PACKGE_PREFIX, r'\1', self.name)
        self._prefix = f'{name}[{type(self).__name__}]: '

    def _fmt(self, msg: str) -> str:
        """Format a log message with the class and instance name."""
        return self._prefix + msg

    def invoke(self, doc_graph: DocumentGraph) -> int:
        """Perform operations on the graph algorithm.

        :param doc_graph: the graph to edit

        :return: the number of edits made to the graph

        """
        if logger.isEnabledFor(logging.INFO):
            logger.info(self._fmt(f'invoking: {doc_graph}'))
        updates: int = self._invoke(doc_graph)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(self._fmt(f'num updates: {updates}'))
        return updates

    @abstractmethod
    def _invoke(self, doc_graph: DocumentGraph) -> int:
        """See :meth:`invoke`."""
        pass

    def reset(self):
        """Reset all state in application context shared objects so new data is
        forced to be created on the next alignment request.

        """
        pass


@dataclass
class GraphSequence(Dictable):
    """A strategy GoF pattern that models what to do during a sequence of graph
    modifications using :class:`.DocumentGraphController`.  It also contains
    rendering information for visualization.

    """
    name: str = field()
    """The name of the sequence, which is used to key its graphs."""

    process_name: str = field()
    """The name of the graph provided to the graph controller.  See
    :obj:`process_graph`.

    """
    render_name: str = field()
    """The name of the graph to render.  See :obj:`render_graph`."""

    heading: str = field()
    """The text used in the heading of the graph rendering."""

    controller: Optional[DocumentGraphController] = field(repr=False)
    """The controller used in the invocation of this strategy."""

    sequencer: GraphSequencer = field(repr=False)
    """Owns and controls this instance."""

    @property
    def process_graph(self) -> DocumentGraph:
        """The graph provided to the graph controller."""
        return self.sequencer.get_graph(self.process_name)

    @property
    def render_graph(self):
        """The graph to render."""
        return self.sequencer.get_graph(self.render_name)

    def invoke(self) -> int:
        """Invoke the strategy.  This implementation calls the controller with
        the :obj:`process_graph` to be processed and passes back the update
        count.

        """
        if self.controller is None:
            # some sequences don't do anything but act as a place holder for
            # rendering
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'{self} has no controller--skipping')
        else:
            return self.controller.invoke(self.process_graph)
        return 0

    def populate_render_context(self, context: RenderContext):
        """Alows the sequence to override the parameters before being sent to
        the graph rendinger API.

        """
        pass

    def reset(self):
        """Reset all state in application context shared objects so new data is
        forced to be created on the next alignment request.

        """
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'reset: {type(self)}')
        if self.controller is not None:
            self.controller.reset()

    def __str__(self):
        return self.name


@dataclass
class RenderUpSideDownGraphSequence(GraphSequence):
    """A graph sequence that tells :mod:`graphviz` to render the diagram upside
    down, which is useful for reverse flow graphs.

    """
    def populate_render_context(self, context: RenderContext):
        context.visual_style = {'attributes': {'rankdir': 'BT'}}


@dataclass
class GraphIteration(Dictable):
    """An iteration of the alignment algorithm.

    """
    sequence: GraphSequence = field()
    """The sequence to use for this iteration."""

    render_level: int = field()
    """Whether to render graphs on a scale from 0 - 10.  The higher the number
    the more likely it is to be rendered with 0 never rendering the graph, and
    10 always rendering the graph.

    :see: :obj:`.DocumentGraphAligner.MAX_RENDER_LEVEL`

    """
    updates: bool = field()
    """Whether to report updates by the iteration, otherwise the iteration
    updates are counted.

    """
    def reset(self):
        """Reset all state in application context shared objects so new data is
        forced to be created on the next alignment request.

        """
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'reset: {type(self)}')
        self.sequence.reset()

    def __str__(self) -> str:
        return str(self.sequence)


class GraphSequencer(object):
    """This invokes the :class:`.GraphSequence` objects in the provided sequence
    to automate the graph alignment algorithm and used by
    :class:`.MaxflowDocumentGraphAligner`.

    """
    def __init__(self, config_factory: ConfigFactory, sequence_path: Path,
                 nascent_graph: DocumentGraph, render: rendergroup,
                 descriptor: str = None, heading_format: str = None):
        """Initialize this instance.

        :param config_factory: used to create the controller in the sequence
                               instances

        :param sequence_path: the path to the JSON file that has the sequences'
                              configuration

        :param nascent_graph: the initial disconnected graph created by
                              :class:`.DocumentGraphFactory`

        :param render: the render object created by :class:`.base.rendergroup`

        """
        self._render = render
        self._descriptor = descriptor
        self._heading_format = heading_format
        self.nascent_graph = nascent_graph
        self._seqs: Dict[str, GraphSequence] = {}
        self._run_iters: Dict[str, int] = collections.defaultdict(lambda: 1)
        self.render_level: int = 0
        with open(sequence_path) as f:
            config: List[Dict[str, Any]] = json.load(f)
        cnf: Dict[str, Any]
        for cnf in config['sequences']:
            name: str = cnf['name']
            p_name: str = cnf.get('process_name')
            r_name: str = cnf['render_name'] if 'render_name' in cnf else p_name
            clsn: str = cnf['class'] if 'class' in cnf else None
            cls: Type = GraphSequence if clsn is None else globals()[clsn]
            ctrl_name: str = cnf['controller'] if 'controller' in cnf else None
            controller: DocumentGraphController = None
            if ctrl_name is not None:
                controller = config_factory.instance(ctrl_name)
            seq = cls(name=name,
                      heading=cnf['heading'] if 'heading' in cnf else None,
                      process_name=p_name,
                      render_name=r_name,
                      controller=controller,
                      sequencer=self)
            self._seqs[name] = seq
        self._iters: Dict[str, List[GraphIteration]] = {}
        gic_sets: Dict[str, List[GraphIteration]] = config['iterations']
        gic_set_name: str
        gic_set: List[Dict[str, Any]]
        for gic_set_name, gic_set in gic_sets.items():
            gic: List[GraphIteration] = []
            self._iters[gic_set_name] = gic
            for gc in gic_set:
                if 'enabled' in gc and not gc['enabled']:
                    continue
                seq: GraphSequence = self._seqs[gc['name']]
                updates: bool = gc.get('updates', False)
                gic.append(GraphIteration(
                    sequence=seq,
                    render_level=gc['render'],
                    updates=updates))

    @property
    def render_level(self) -> int:
        """Whether to render graphs on a scale from 0 - 10.  See
        :obj:`.DocumentGraphAligner.MAX_RENDER_LEVEL`.

        """
        return self._render_level

    @render_level.setter
    def render_level(self, level: int):
        """Whether to render graphs on a scale from 0 - 10.  See
        :obj:`.DocumentGraphAligner.MAX_RENDER_LEVEL`.

        """
        DocumentGraphAligner.is_valid_render_level(level, should_raise=True)
        self._render_level = level

    def get_graph(self, name: str = None) -> DocumentGraph:
        """Return a graph by name."""
        if name is None:
            return self.nascent_graph
        else:
            return self.nascent_graph.children[name]

    def _iterate(self, name: str) -> Iterable[GraphIteration]:
        return self._iters[name]

    def _invoke(self, graph_iter: GraphIteration, iter_name: str,
                run_iter: int) -> int:
        """Invoke the sequence graph modifications on ``seq``."""
        seq: GraphIteration = graph_iter.sequence
        updates: int = seq.invoke()
        max_render_level: int = DocumentGraphAligner.MAX_RENDER_LEVEL
        render_level: int = self.render_level
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'iter level: {graph_iter.render_level} ' +
                         f'max: {max_render_level}, render: {render_level}')
        if graph_iter.render_level > (max_render_level - render_level):
            heading: str = None
            if self._heading_format is not None:
                heading = self._heading_format.format(**locals())
                if self._descriptor is not None:
                    heading += f' {self._descriptor}'
                heading += f' of graph {seq.render_graph}'
            context = RenderContext(
                seq.render_graph,
                heading=heading)
            seq.populate_render_context(context)
            try:
                if logger.isEnabledFor(logging.INFO):
                    logger.info(f'rendering: {context}')
                self._render(context)
            except Exception as e:
                # don't re-raise since we won't get the statistics for an
                # otherwise successful alignment
                doc: AmrFeatureDocument = self.nascent_graph.doc
                logger.error(f'could not render: <{doc}>: {e}',
                             stack_info=True, exc_info=True)
        return updates

    def run(self, name: str) -> int:
        run_iter: int = self._run_iters[name]
        updates: int = 0
        giter: Sequence
        for giter in self._iterate(name):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'invoking: {giter}')
            with time(f'{name}:{giter} updates: {{iter_updates}}'):
                iter_updates: int = self._invoke(giter, name, run_iter)
            if giter.updates:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f'adding updates: {giter} {iter_updates}')
                updates += iter_updates
        self._run_iters[name] += 1
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'{name} total updates: {updates}')
        return updates

    def reset(self):
        """Reset all state in application context shared objects so new data is
        forced to be created on the next alignment request.

        """
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'reset: {type(self)}')
        giter: Sequence
        for giter in chain.from_iterable(self._iters.values()):
            giter.reset()
        self._run_iters.clear()


@dataclass
class MaxflowDocumentGraphAligner(DocumentGraphAligner):
    """Uses the maxflow/min cut algorithm to compute graph component alignments.

    """
    graph_sequencer_name: str = field()
    """The app configuration section name of :class:`.GraphSequencer`."""

    max_sequencer_iterations: int = field()
    """The max number of iterations of the sequencer loop.  This is the max
    number of times the ``loop`` iteration set runs if the maxflow algorithm
    doesn't converge (0 changes on bipartite capacities) first.

    """
    hyp: HyperparamModel = field()
    """The capacity calculator hyperparameters.

    :see: :obj:`.summary.CapacityCalculator.hyp`

    """
    def _get_descriptor(self) -> str:
        """Get a descriptor unique based on the graph configuration."""
        sdamp: float = self.hyp.sentence_dampen
        if 0:
            eweights: Sequence[int] = self.hyp.neighbor_embedding_weights
            ndir: str = self.hyp.neighbor_direction
            weight_name = '-'.join(map(str, eweights))
            return f'{weight_name} (dir={ndir})'
        if 0:
            return f' M sdamp={sdamp}'

    def _align(self, doc_graph: DocumentGraph, render: rendergroup) -> \
            FlowGraphResult:
        sequencer: GraphSequencer = self.config_factory.new_instance(
            self.graph_sequencer_name,
            nascent_graph=doc_graph,
            render=render,
            descriptor=self._get_descriptor())
        try:
            sequencer.render_level = self.render_level
            sequencer.run('construct')
            sequencer.render_level = self.init_loops_render_level
            converges: int = 0
            iteration: int
            for iteration in range(self.max_sequencer_iterations):
                updates: int = sequencer.run('loop')
                if updates == 0:
                    if converges >= 1:
                        logger.info('max flow convergence after ' +
                                    f'{iteration + 1} iterations')
                        break
                    converges += 1
                else:
                    converges = 0
            sequencer.render_level = self.render_level
            sequencer.run('final')
            return self.config_factory.new_instance(
                self.flow_graph_result_name,
                data=doc_graph)
        finally:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'reset: {type(self)}')
            sequencer.reset()
