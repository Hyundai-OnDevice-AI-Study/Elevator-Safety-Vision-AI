# 공유 코드(common/door_common)를 찾을 수 있게 저장소 루트의 common/ 을 경로에 추가한다.
# pip install 없이 각 팀 폴더에서 `python -m ...` 으로 바로 실행하기 위함.
import os as _os
import sys as _sys

_COMMON = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))), "common")
if _COMMON not in _sys.path:
    _sys.path.insert(0, _COMMON)
