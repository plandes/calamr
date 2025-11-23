#!/usr/bin/env python

from zensols.calamr import ApplicationFactory, Resource, FlowGraphResult
import pandas as pd


def main():
    # get the application resource for corpus A
    res: Resource = ApplicationFactory.get_resource(['-c config/corp-a.conf'])
    # align and compute flow results and cache the data
    flow_result: FlowGraphResult = res.align_corpus_document('liu-example')
    # get the results as a dataframe
    df: pd.DataFrame = flow_result.df
    # write results
    flow_result.write()
    print(df.head(10))

    # iterate through all results of the corpus
    key: str
    for key, flow_result in res.anon_doc_stash.items():
        print(f'{key} -> {flow_result}')

    res = ApplicationFactory.get_resource(['-c config/corp-b.conf'])
    # iterate through all results of corpus B
    for key, flow_result in res.anon_doc_stash.items():
        print(f'{key} -> {flow_result}')


if (__name__ == '__main__'):
    main()
