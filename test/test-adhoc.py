#!/usr/bin/env python

from zensols.amr import AmrSentence, AmrDocument, AmrFeatureDocument
from zensols.calamr import (
    DocumentGraph, FlowGraphResult, Resource, ApplicationFactory
)


def main():
    test_summary = AmrSentence("""\
# ::snt Joe's dog was chasing a cat in the garden.
# ::snt-type summary
# ::id liu-example.0
(c / chase-01
   :ARG0 (d / dog
            :poss (p / person
                     :name (n / name
                              :op1 "Joe")))
   :ARG1 (c2 / cat)
   :location (g / garden))""")
    test_body = AmrSentence("""\
# ::snt I saw Joe's dog, which was running in the garden.
# ::snt-type body
# ::id liu-example.1
(s / see-01
   :ARG0 (ii / i)
   :ARG1 (d / dog
            :poss (p / person
                     :name (n / name
                              :op1 "Joe"))
            :ARG0-of (r / run-02
                        :location (g / garden))))""")
    res: Resource = ApplicationFactory.get_resource()
    adoc = AmrDocument((test_summary, test_body))
    fdoc: AmrFeatureDocument = res.to_annotated_doc(adoc)
    graph: DocumentGraph = res.create_graph(fdoc)
    flow: FlowGraphResult = res.align(graph)
    flow.write()
    flow.render()


if (__name__ == '__main__'):
    main()
