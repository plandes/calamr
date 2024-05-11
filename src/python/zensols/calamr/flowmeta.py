"""Metadata for the :mod:`flow` module.

"""
__author__ = 'Paul Landes'

from typing import Tuple

_DATA_DESC_META: Tuple[Tuple[str, str]] = (
    ('s_descr', 'source node descriptions such as concept names, attribute constants and sentence text'),
    ('t_descr', 'target node descriptions such as concept names, attribute constants and sentence text'),
    ('s_toks', 'any source node aligned tokens'),
    ('t_toks', 'any target node aligned tokens'),
    ('s_attr', 'source node attribute name give by such as `doc`, `sentence`, `concept`, `attribute`'),
    ('t_attr', 'target node attribute name give by such as `doc`, `sentence`, `concept`, `attribute`'),
    ('s_id', 'source node unique identifier'),
    ('t_id', 'target node unique identifier'),
    ('edge_type', 'whether the edge is an AMR `role` or `alignment`'),
    ('rel_id', 'the coreference relation ID or `null` if the edge is not a corefernce'),
    ('is_bipartite', 'whether relation `rel_id` spans components or `null` if the edge is not a coreference'),
    ('flow', 'the (normalized/flow per node) flow of the edge'),
    ('reentrancy', 'whether the edge participates an AMR reentrancy'),
    ('align_flow', 'the flow sum of the alignment edges for the respective edge'),
    ('align_count', 'the count of incoming alignment edges to the target node'))
