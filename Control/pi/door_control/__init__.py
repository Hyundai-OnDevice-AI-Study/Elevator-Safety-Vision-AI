# 공유 코드(common/door_common)를 찾을 수 있게 저장소 루트의 common/ 을 경로에 추가한다.
# (Control/pi/door_control/__init__.py → 저장소 루트까지 세 단계 위)
import os as _os
import sys as _sys

_COMMON = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))))), "common")
if _COMMON not in _sys.path:
    _sys.path.insert(0, _COMMON)
