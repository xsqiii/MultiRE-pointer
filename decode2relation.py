import re
import tensorflow as tf

# strict measure
def calculate_measure(decode_re, target_re):
    true_positive = 0
    original_res = 0
    decode_total = 0

    # todo compute as ACL2018
    for de_re, ta_re in zip(decode_re, target_re):
        decode_total += len(de_re)
        original_res += len(ta_re)

        tf.logging.info('de_re:', de_re)
        tf.logging.info('ta_re:', ta_re)

        result = [1 if de in ta_re else 0 for de in de_re]
        tf.logging.info(result)

        true_positive += sum(result)

    p = true_positive * 1.0 / decode_total if decode_total > 0 else 0.
    r = true_positive * 1.0 / original_res if original_res > 0 else 0.
    f1 = (2 * p * r) / (p + r) if p * r > 0 else 0.

    return p, r, f1


def _triplelist2triples_(triple_list):
    """
     >>> _triplelist2triples_([1,2,3, 2,5,0])
     {(1,2,3),(2,5,0)}
     >>> _triplelist2triples_([1,2,3, 1,2,3, 2,5,0])
     {(1,2,3),(2,5,0)}
     >>> _triplelist2triples_([1,2,3, 2,5,0].extend(config.NA_TRIPLE))
     {(1,2,3),(2,5,0)}
    """
    # triple_list = list(triple_list)
    triples = set([tuple(triple_list[i:i + 3]) for i in range(0, len(triple_list), 3)])
    return triples

def calculate_measure_cmp(decode_re, target_re):
    true_positive = 0
    original_res = 0
    decode_total = 0


    for d, t in zip(decode_re, target_re):
        d = list(d.split())
        t = list(t.split())
        d_triples = _triplelist2triples_(d)
        t_triples = _triplelist2triples_(t)
        original_res += len(t_triples)
        decode_total += len(d_triples)

        result = [1 if d_t in t_triples else 0 for d_t in d_triples]
        tf.logging.info(result)
        tf.logging.info('decode:%s', d_triples)
        tf.logging.info('gold:%s', t_triples)
        true_positive += sum(result)

    precision = true_positive * 1.0 / decode_total if decode_total > 0 else 0.
    recall = true_positive * 1.0 / original_res if original_res > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision * recall > 0 else 0.

    return precision, recall, f1
