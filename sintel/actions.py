# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

def _overlap(expected, observed):
    first = expected[0] - observed[1]
    second = expected[1] - observed[0]
    return first * second < 0


def _any_overlap(part, intervals):
    for interval in intervals:
        if _overlap(part, interval):
            return True

    return False

def sample(items, k):
    k_ = min(k, len(items))
    idx = np.random.choice(list(range(len(items))), size=k_, replace=False)
    return [items[i] for i in idx]

def event_to_add(pred, true, k):
    missed = [('a', x) for x in true if not _any_overlap(x, pred)]
    return sample(missed, k)

def event_to_remove(pred, true, k):
    incorrect = [('r', x) for x in pred if not _any_overlap(x, true)]
    return sample(incorrect, k)

def annotator(pred, true, k):
    if not isinstance(pred, list):
        pred = list(pred[['start', 'end']].itertuples(index=False))
    if not isinstance(true, list):
        true = list(true[['start', 'end']].itertuples(index=False))
    
    to_add = event_to_add(pred, true, k)
    to_remove = event_to_remove(pred, true, k)

    annot = sample(to_add + to_remove, k)
    keep = [x for x in pred if _any_overlap(x, true)]
    
    for (action, event) in annot:
        if action == 'a':
            keep.append(event)
            
    return keep

def add_label(df, events):
    df_ = df.copy()
    
    for (action, event) in events:
        if action == 'r':
            label = 0
        elif action == 'a':
            label = 1

        df_.at[event[0]: event[1], 'label'] = label
        
    return df_
