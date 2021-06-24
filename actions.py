# -*- coding: utf-8 -*-

def _overlap(expected, observed):
    first = expected[0] - observed[1]
    second = expected[1] - observed[0]
    return first * second < 0


def _any_overlap(part, intervals):
    for interval in intervals:
        if _overlap(part, interval):
            return True

    return False

def event_to_add(pred, true):
    missed = [x for x in true if not _any_overlap(x, pred)]
    return random.choice(missed)

def event_to_remove(pred, true):
    incorrect = [x for x in pred if not _any_overlap(x, true)]
    return random.choice(incorrect)
