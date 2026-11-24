
import pytest

from ml.policy import should_promote, should_retrain


@pytest.mark.parametrize(
    "candidate_auc, champion_auc, expected",
    [
        (0.90, 0.87, True),
        (0.84, None, False),
        (0.875, 0.87, False),
        (0.88, 0.87, True),
        (0.85, None, True),
        (0.86, 0.90, False),
        (0.83, 0.80, False),
        (0.91, 0.91, False),
    ],
)
def test_should_promote(candidate_auc, champion_auc, expected):
    decision, reason = should_promote(
        candidate_auc, champion_auc
    )

    assert decision == expected
    assert reason


@pytest.mark.parametrize(
    "champion_auc, approved_auc, new_data, expected",
    [
        (0.91, 0.91, True, True),
        (0.84, 0.85, False, True),
        (0.87, 0.90, False, True),
        (0.88, 0.90, False, False),
        (0.91, 0.91, False, False),
        (0.85, 0.85, False, False),
    ],
)
def test_should_retrain(
    champion_auc, approved_auc, new_data, expected
):
    decision, reason = should_retrain(
        champion_auc, approved_auc, new_data
    )

    assert decision == expected
    assert reason
