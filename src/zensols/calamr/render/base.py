"""Renderer AMR graphs.

"""
from __future__ import annotations
__author__ = 'Paul Landes'
from typing import Union, Type, Optional, Sequence, Dict, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
import time
import traceback
from pathlib import Path
from zensols.util.tempfile import tempfile, TemporaryFileName
from zensols.persist import FileTextUtil
from zensols.config import Settings, Dictable
from zensols.rend import Presentation, Location, BrowserManager
from .. import DocumentGraph

logger = logging.getLogger(__name__)


@dataclass
class RenderContext(Dictable):
    """Contains everything that is needed to render a graph.

    """
    doc_graph: DocumentGraph = field()
    """The graph to render."""

    heading: str = field(default=None)
    """The title to use as the heading."""

    visual_style: Dict[str, Any] = field(default=None)
    """Any overriding context for the renderer."""


@dataclass
class GraphRenderer(ABC):
    """Renders an igraph in to a file and then displays it.  To render the
    graph, use :meth:`render`.  Then you can display it with :meth:`show`.

    The implementation of rendering is done in subclasses.

    """
    browser_manager: BrowserManager = field()
    """Detects and controls the screen."""

    extension: str = field()
    """The output file's extension."""

    rooted: bool = field(default=True)
    """Whether the graph should be drawn as a tree."""

    visual_style: Settings = field(default_factory=Settings)
    """Contains the visualization sytle for the graph."""

    sleep: Union[int, float] = field(default=0)
    """The number of seconds to sleep before deleting the generated file to give
    the browser time to display.

    """
    def _name_to_path(self, name: str = 'graph') -> Path:
        return f'{name}.{self.extension}'

    @abstractmethod
    def _render_to_file(self, context: RenderContext, out_file: Path):
        """See :meth:`render_to_file"""
        pass

    def render_to_file(self, context: RenderContext, out_file: Path):
        """Render graph the graph.

        :param context: contains everything that is needed to render a graph

        """
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'rendering graph to {out_file}')
        self._render_to_file(context, out_file)

    def _sleep(self):
        if self.sleep is not None:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'sleeping for {self.sleep}s')
            time.sleep(self.sleep)

    def display(self, out_file: Path):
        """Display the graph generated in file ``out_file``.

        :param out_file: the graph to display


        """
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'displaying {out_file}')
        self.browser_manager.show(out_file)
        self._sleep()

    def display_all(self, out_files: Sequence[Path]):
        """Like :meth:`display` but display several files at once.  In some
        cases, this means a separate browser window with multiple tabs with just
        the selected graphs will be shown.

        """
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'displaying {", ".join(map(str, out_files))}')
        pres = Presentation(locations=tuple(map(Location, out_files)))
        self.browser_manager.show(pres)
        self._sleep()

    def show(self, context: RenderContext):
        """Render and display the document graph."""
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'showing: {context}')
        file_fmt: str = self._name_to_path()
        with tempfile(file_fmt=file_fmt) as out_file:
            self.render_to_file(context, out_file)
            self.display(out_file)


class _TemporaryFileName(TemporaryFileName):
    """Override to include the graph ID from :class:`rendergroup`.

    """
    def _format_name(self, fname: str) -> str:
        return self._file_fmt.format(
            name=fname, index=len(self), graph_id=self.graph_id)


class rendergroup(object):
    """A context manager to render several graphs at a time, then optionally
    display them.  Rendering graphs is done by calling an instance of this
    context manager.

    Example:

    .. code-block:: python

       with rendergroup(self.renderer) as render:
           render(graph, heading='Source Max flow')

    """
    def __init__(self, renderer: GraphRenderer, graph_id: str = 'graph',
                 display: bool = True, directory: Path = None):
        """Initialize the context manager.

        :param renderer: used to render the graphs and optionally display using
                         :meth:`.Renderer.display_all`

        :param graph_id: a unique identifier prefixed to files generated if none
                         provided in the call method

        :param display: whether to display the files after generated

        :param directory: the directory to create the files in place of the
                          temporary directory; if provided the directory is not
                          removed after the graphs are rendered

        """
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'create render with output to {directory}, ' +
                         f'display={display}')
        self._remove = directory is None
        self._file_iter = _TemporaryFileName(
            file_fmt=f'{{index}}-{{graph_id}}.{renderer.extension}',
            directory=directory,
            remove=self._remove)
        self._file_iter.graph_id = graph_id
        self._renderer = renderer
        self._display = display

    def __call__(self, context: RenderContext, graph_id: str = None) -> Path:
        doc_graph: DocumentGraph = context.doc_graph
        heading: str = context.heading
        if graph_id is None and heading is not None:
            graph_id = FileTextUtil.normalize_text(heading)
        if graph_id is not None:
            self._file_iter.graph_id = graph_id
        out_file: Path = next(self._file_iter)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'write: {type(doc_graph)} to {out_file}')
        self._renderer.render_to_file(context, out_file)
        return out_file

    def __enter__(self) -> rendergroup:
        return self

    def __exit__(self, cls: Type[Exception], value: Optional[Exception],
                 trace: traceback):
        if value is not None:
            raise value
        try:
            if self._display:
                if cls is None:
                    self._renderer.display_all(self._file_iter.files)
        finally:
            try:
                if self._remove:
                    self._file_iter.clean()
            except Exception as e:
                logger.warning(f'Could not remove temporary files: {e}',
                               exc_info=True)
        return True
