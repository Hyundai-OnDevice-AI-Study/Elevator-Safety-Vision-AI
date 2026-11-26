# Control — 센싱·제어 팀 (준비 중)

압력 센싱과 엘리베이터(미니어처) 제어를 담당한다. 아직 코드는 없고, 역할 분담과 다른 팀과 주고받을 메시지만 정해져 있다.

## 역할 분담

| 장치 | 담당 | 내용 |
|---|---|---|
| **ESP32** (헥사보드, Arduino IDE) | **문 제어** | 문 모터·서보 구동, 열림/닫힘 위치 감지(근접센서·리밋), 문 상태 보고 |
| | **카 제어** | 카 모터 구동(이동·정지·비상정지), 카 상태 보고 |
| **Raspberry Pi 4B** (Python) | 압력 센싱 | HX711 + 로드셀을 Pi GPIO 에 직접 연결, 영점·스케일 보정 |
| | 경보 출력 | 부저, 경광등(노랑/빨강), 음성 안내 |
| | ESP32 연결 | 시리얼(USB) 로 문·카 명령 전달, 상태 수신 |

판단(언제 문을 열고 카를 세울지)은 하지 않는다. 판단은 Analysis 의 상태 머신이 하고, Control 은 받은 명령을 실행하고 상태·센서 값을 보고한다.

```
              Analysis (상태 머신)
           ▲ PressureSample        │ ActionCommand
           │ DeviceEvent           ▼
 ┌───────────────── Raspberry Pi (Control/pi) ─────────────────┐
 │  HX711 압력 읽기      부저·경광등·음성      ESP32 시리얼 브리지  │
 └──────────────────────────────────────────────────┬──────────┘
                                                    │ USB 시리얼
                                  ┌─────────────────▼─────────────────┐
                                  │ ESP32 (Control/firmware)           │
                                  │  문 모터·센서        카 모터         │
                                  └───────────────────────────────────┘
```

## 메시지 (`common/door_common/messages.py`)

| 방향 | 메시지 | 만드는 곳 | 내용 |
|---|---|---|---|
| Control → Analysis | `PressureSample` | Pi (HX711) | 원시값(raw), 영점·스케일 보정한 값(grams) |
| Control → Analysis | `DeviceEvent` | ESP32 → Pi 브리지 | 문 `open/closing/closed`, 카 `stopped/moving` |
| Control → Analysis | `DeviceEvent` (`kind="fault"`) | Pi 또는 ESP32 | 센서·통신 이상 |
| Analysis → Control | `ActionCommand` `target=door/car` | Pi 브리지 → ESP32 | 문 `open`, 카 `block_start`(출발 보류) / `emergency_stop` 등 |
| Analysis → Control | `ActionCommand` `target=buzzer/beacon/voice` | Pi 에서 바로 실행 | `on/off`, `yellow/red`, 안내 문구 |

## 예정 구성

```
firmware/   ESP32 (Arduino IDE): 문·카 모터, 위치 센서, 시리얼 명령 처리
pi/         Raspberry Pi: HX711 읽기, 부저·경광등·음성, ESP32 시리얼 브리지
```

## 정할 것

- ESP32 ↔ Pi 시리얼 프로토콜 (명령·응답·하트비트 형식)
- Pi 와의 통신이 끊겼을 때 ESP32 가 스스로 취할 안전 동작 (예: 출발 금지)
- 압력 판단 임계값: Analysis 의 `../Analysis/config/rules.json` → `pressure` 항목 (미니어처에서 직접 눌러 보며 채울 자리)
