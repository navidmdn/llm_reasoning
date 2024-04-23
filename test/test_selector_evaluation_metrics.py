import pytest
from typing import List, Dict

# Assuming the function to be tested is in a file named 'metrics.py'
from select_and_deduct.evaluate_selector import permutation_invariant_metrics, has_correct_format


def _permutation_invariant_match(pred: str, target: str) -> bool:
    """Mocking the _permutation_invariant_match function for testing purposes."""
    return sorted(pred) == sorted(target)

@pytest.mark.parametrize("preds,targets,num_pred_seq,expected", [
    (
            ['int2sent & sent2', 'sent1 & int2 & sent2', 'sent1 & sent2 & int1', 'int4 & sent2 & sent1'],
            ['int2 & sent1 & sent2', 'int1 & sent2 & sent1'],
            2,
            {'top1_acc': 0.5, 'top2_acc': 1.0, 'diversity': 0.75}
    ),
    (
            ['int2sent & sent2', 'sent1 & int2 & sent2 & sent2', 'sent1 & sent2 & int1', 'int4 & sent2 & sent1'],
            ['int2 & sent1 & sent2', 'int1 & sent2 & sent1'],
            2,
            {'top1_acc': 0.5, 'top2_acc': 0.5, 'diversity': 0.75}
    ),
    (
            ['int2sent & sent2', 'sent1 & int2 & sent2 & sent2', 'sent1 & sent2 & int1', 'int4 & sent2 & sent1'],
            ['int4 & sent1 & sent2'],
            4,
            {'top1_acc': 0.0, 'top4_acc': 1.0, 'diversity': 0.75}
    ),
])
def test_permutation_invariant_metrics(preds: List[str], targets: List[str], num_pred_seq: int, expected: Dict[str, float]):
    out = permutation_invariant_metrics(preds, targets, num_pred_seq)
    intersection_keys = set(out.keys()).intersection(expected.keys())
    for key in intersection_keys:
        assert out[key] == expected[key]


def test_valid_format():
    assert has_correct_format("int1 & sent2")
    assert has_correct_format("sent100 & int200")
    assert has_correct_format("sent1 & int2 & sent4")

def test_invalid_format():
    assert not has_correct_format("int1 & word")
    assert not has_correct_format("int & sent")

def test_incorrect_use_of_ampersand():
    assert not has_correct_format("int1 sent2")
    assert not has_correct_format("int1 &sent2")
    assert not has_correct_format("int1& sent2")

def test_single_valid_patterns():
    assert not has_correct_format("int1")
    assert not has_correct_format("sent10")

def test_empty_string():
    assert not has_correct_format("")

def test_no_ampersand():
    assert not has_correct_format("int1sent2")