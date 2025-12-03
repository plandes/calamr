#!/usr/bin/env python

"""Example of how to parse and align an adhoc corpus.  An adhoc corpus is any
data defined as a sequence of dictionaries given to the API.

"""

from typing import List, Dict
from zensols.amr import AmrFeatureDocument
from zensols.calamr import (
    DocumentGraph, FlowGraphResult, Resources, ApplicationFactory
)

corpus: List[Dict[str, str]] = \
    [{
        "id": "first",
        "body": "The rulings bolster criticisms of how hastily the prosecutions were brought. The rulings were rushed.",
        "summary": "The rulings suggest the prosecutions were rushed."
    }, {
        "id": "second",
        "comment": "very short",
        "body": "The man ran to make the train. He just missed it.",
        "summary": "A man got caught in the door of a train he just missed."
    }]

# get the resource bundle
resources: Resources = ApplicationFactory.get_resources()

# access an adhoc corpus as defined above with the list of dictionaries above
with resources.adhoc(corpus) as r:
    # list the keys in the corpus, each of which is available as a document or
    # alignment as flow metrics/data
    keys = tuple(r.documents.keys())
    print(keys)

    # get a document, which parses the document (if they aren't already); this
    # step isn't necessary if you want to go right to the alignments
    doc: AmrFeatureDocument = r.documents['first']
    doc.write()

    # get an alignment, which parses alignments (if not already)
    flow: FlowGraphResult = r.alignments['first']
    # write the metrics (or flow.stats to get a simle dictionary)
    flow.write()
    # render the graph with flow data visually
    flow.render()

    # get the document graph and access to the nodes and edges of the graph
    dg: DocumentGraph = flow.doc_graph.children['reversed_source']
    dg.write()
