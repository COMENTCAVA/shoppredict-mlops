
from math import isclose

FLOOR = 0.85
MIN_GAIN = 0.01
MAX_DROP = 0.03


def should_retrain(champion_auc, approved_auc, new_data):
    if new_data:
        return True, "New dataset version"

    if champion_auc < FLOOR:
        return True, "Champion below quality floor"

    drop = approved_auc - champion_auc

    if drop >= MAX_DROP or isclose(
        drop, MAX_DROP, rel_tol=0, abs_tol=1e-12
    ):
        return True, "Champion performance has dropped"

    return False, "Champion performance remains acceptable"


def should_promote(candidate_auc, champion_auc):
    if candidate_auc < FLOOR:
        return False, "Candidate below quality floor"

    if champion_auc is None:
        return True, "Initial model meets quality floor"

    gain = candidate_auc - champion_auc

    if gain >= MIN_GAIN or isclose(
        gain, MIN_GAIN, rel_tol=0, abs_tol=1e-12
    ):
        return True, "Candidate meets minimum gain"

    return False, "Candidate improvement is insufficient"
