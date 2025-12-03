#!/usr/bin/env python

"""Similar to the ``adhoc.py`` example but use a corpus.  All the same resources
are available in the ``with resource.corpus`` scope.

"""
from zensols.amr import AmrFeatureDocument
from zensols.calamr import FlowGraphResult, Resources, ApplicationFactory

# get the resource bundle
resources: Resources = ApplicationFactory.get_resources()

# access the corpus in the `./download` directory
with resources.corpus() as r:
    # get a document, which parses the document (if they aren't already); this
    # step isn't necessary if you want to go right to the alignments
    doc: AmrFeatureDocument = r.documents['liu-example']
    doc.write()

    # get an alignment, which parses alignments (if not already)
    flow: FlowGraphResult = r.alignments['liu-example']
    # write the metrics (or flow.stats to get a simle dictionary)
    flow.write()
    # render the graph with flow data visually
    flow.render()
