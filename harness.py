#!/usr/bin/env python

from zensols import deepnlp


if (__name__ == '__main__'):
    # initialize the NLP system
    deepnlp.init()
    from zensols.cli import ConfigurationImporterCliHarness
    harness = ConfigurationImporterCliHarness(
        src_dir_name='src',
        app_factory_class='zensols.calamr.ApplicationFactory',
        proto_args='proto',
        proto_factory_kwargs={
            'reload_pattern':
            r'^zensols\.calamr\.(?!flow|doc|domain|annotate)'})
    r = harness.run()
