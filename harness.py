#!/usr/bin/env python

from zensols import deepnlp

# initialize the NLP system
deepnlp.init()

if 0:
    import zensols.deepnlp.transformer as tran
    tran.turn_off_huggingface_downloads()


if (__name__ == '__main__'):
    from zensols.cli import ConfigurationImporterCliHarness
    harness = ConfigurationImporterCliHarness(
        src_dir_name='src',
        app_factory_class='zensols.calamr.ApplicationFactory',
        proto_args='proto',
        proto_factory_kwargs={
            'reload_pattern':
            r'^zensols\.calamr\.(?!flow|doc|domain|annotate)'},
    )
    r = harness.run()
