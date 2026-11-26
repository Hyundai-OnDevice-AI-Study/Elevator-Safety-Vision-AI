# Control — 센싱·제어 팀 (준비 중)

압력 센서와 미니어처(헥사보드) 제어를 담당한다. 아직 코드는 없고, 다른 팀과 주고받을 메시지만 정해져 있다.

| 방향 | 메시지 (`common/door_common/messages.py`) | 내용 |
|---|---|---|
| Control → Analysis | `PressureSample` | HX711 원시값(raw), 영점·스케일 보정한 값(grams) |
| Control → Analysis | `DeviceEvent` | 문 `open/closing/closed`, 카 `stopped/moving`, 고장 `fault` |
| Analysis → Control | `ActionCommand` | 대상 `door/car/buzzer/beacon/voice`, 동작 `open/block_start/emergency_stop/on/off …` |

예정 구성

```
firmware/   헥사보드 ESP32 (Arduino IDE): 모터·서보·근접센서·HX711
bridge/     Pi 쪽: 시리얼 ↔ 위 메시지 변환
```

압력 판단 임계값은 해석 팀의 `../Analysis/config/rules.json` 의 `pressure` 항목에 둔다 (미니어처에서 직접 눌러 보며 채울 자리).
