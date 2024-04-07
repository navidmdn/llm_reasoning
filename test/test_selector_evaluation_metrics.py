import pytest
from typing import List, Dict

# Assuming the function to be tested is in a file named 'metrics.py'
from evaluate_selector import permutation_invariant_metrics


def _permutation_invariant_match(pred: str, target: str) -> bool:
    """Mocking the _permutation_invariant_match function for testing purposes."""
    return sorted(pred) == sorted(target)

@pytest.mark.parametrize("preds,targets,num_pred_seq,expected", [
    (
            ['int2sent & sent2', 'sent1 & int2 & sent2', 'sent1 & sent2 & int1', 'int4 & sent2 & sent1'],
            ['int2 & sent1 & sent2', 'int1 & sent2 & sent1'],
            2,
            {'top1_acc': 0.5, 'top2_acc': 1.0}
    ),
    (
            ['int2sent & sent2', 'sent1 & int2 & sent2 & sent2', 'sent1 & sent2 & int1', 'int4 & sent2 & sent1'],
            ['int2 & sent1 & sent2', 'int1 & sent2 & sent1'],
            2,
            {'top1_acc': 0.5, 'top2_acc': 0.5}
    ),
    (
            ['int2sent & sent2', 'sent1 & int2 & sent2 & sent2', 'sent1 & sent2 & int1', 'int4 & sent2 & sent1'],
            ['int4 & sent1 & sent2'],
            4,
            {'top1_acc': 0.0, 'top4_acc': 1.0}
    ),
])
def test_permutation_invariant_metrics(preds: List[str], targets: List[str], num_pred_seq: int, expected: Dict[str, float]):
    assert permutation_invariant_metrics(preds, targets, num_pred_seq) == expected

