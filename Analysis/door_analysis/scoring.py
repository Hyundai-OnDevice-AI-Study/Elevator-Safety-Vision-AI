"""채점 기준 — evaluate.py 와 테스트가 같은 기준을 쓰도록 한 곳에 둔다 (v0.4 run_offline.summarize)."""


def passed(expected: int, got: int) -> bool:
    """최고 수준 got 이 기대 수준 expected 에 맞는가.

    기대 3(위험) → 3 이 나와야 함 (미탐 금지)
    기대 2(경계) → 정확히 2 (위험으로 올라가도, 주의로 내려가도 실패)
    기대 0·1     → 그 이하면 통과 (더 낮은 건 허용, 높으면 오탐)
    """
    if expected == 3:
        return got == 3
    if expected == 2:
        return got == 2
    return got <= expected
