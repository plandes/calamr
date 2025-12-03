#!/usr/bin/env python

from zensols.amr import AmrFeatureDocument
from zensols.calamr import ApplicationFactory, Resources, FlowGraphResult
import pandas as pd


def main():
    # get the application resource for corpus A
    resources: Resources = ApplicationFactory.\
        get_resources('-c config/corp-a.conf')

    # access the corpus in the `./download` directory
    with resources.corpus() as r:
        # align and compute flow results and cache the data
        flow_result: FlowGraphResult = r.alignments['liu-example']

        # get the results as a dataframe
        df: pd.DataFrame = flow_result.df
        # write results
        flow_result.write()
        print(df.head(10))

        # iterate through all documents of the corpus
        key: str
        doc: AmrFeatureDocument
        for key, doc in r.documents.items():
            print(f'{key} -> {doc}')

    # iterate through all results of corpus B
    resources = ApplicationFactory.get_resources('-c config/corp-b.conf')
    with resources.corpus() as r:
        for key, doc in r.documents.items():
            print(f'{key} -> {doc}')


if (__name__ == '__main__'):
    main()
