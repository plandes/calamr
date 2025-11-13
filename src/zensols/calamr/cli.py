"""Command line entry point to the application.

"""
__author__ = 'Paul Landes'

from typing import List, Any, Dict, Type
import sys
from zensols.config import ConfigFactory
from zensols.cli import ActionResult, CliHarness
from zensols.cli import ApplicationFactory as CliApplicationFactory
from . import Resource


class ApplicationFactory(CliApplicationFactory):
    def __init__(self, *args, **kwargs):
        kwargs['package_resource'] = 'zensols.calamr'
        super().__init__(*args, **kwargs)

    @classmethod
    def get_resource(cls: Type, *args, **kwargs) -> Resource:
        """A client facade (GoF) for Calamr annotated AMR corpus access and
        alginment.

        """
        harness: CliHarness = cls.create_harness()
        fac: ConfigFactory = harness.get_config_factory(*args, **kwargs)
        return fac('aapp').resource


def main(args: List[str] = sys.argv, **kwargs: Dict[str, Any]) -> ActionResult:
    harness: CliHarness = ApplicationFactory.create_harness(relocate=False)
    harness.invoke(args, **kwargs)
