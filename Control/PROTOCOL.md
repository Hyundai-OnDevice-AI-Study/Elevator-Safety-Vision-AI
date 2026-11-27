# ESP32 ↔ Raspberry Pi 시리얼 프로토콜 (초안 v0.1)

> 상태: **제안안**. `[합의 필요]` 표시는 제어 팀과 정해야 하는 값/동작이다.
> 수치는 처음 돌려 보기 위한 제안값이며, 미니어처에서 재 보고 고친다.
> 이 문서를 바꾸면 `PROTO_VERSION` 을 올리고, Pi 쪽 `Control/pi/door_control/esp32_link.py` 도 같이 고친다.

## 0. 역할 경계 (가장 먼저 합의할 것)

- **운행은 ESP32 가 결정한다.** 층 선택·출발·도착·문 자동 개폐는 지금처럼 ESP32 가 버튼으로 한다.
- **Pi 는 막기만 한다.** 출발 보류(`block_start/block_end`), 비상정지(`emergency_stop/emergency_reset`), 문 다시 열기(`door open`).
  Pi 가 카를 특정 층으로 보내거나 문을 닫으라고 명령하지 않는다.
- **ESP32 는 상태를 보고한다.** 문 `open/closing/closed`, 카 `stopped/moving`, 고장.
- 판단(언제 막을지)은 Analysis 상태 머신(`Analysis/door_analysis/state.py`)이 한다. ESP32 는 판단하지 않는다.
  단, **Pi 와 연결이 끊겼을 때의 안전 동작(5장)은 ESP32 가 스스로** 한다.

```
Analysis 상태 머신 ──ActionCommand(target=door/car)──▶ Pi 브리지 ──JSON 한 줄──▶ ESP32
                  ◀──DeviceEvent(kind=door/car/fault)── Pi 브리지 ◀──JSON 한 줄── ESP32
```

압력(HX711)·부저·경광등·음성은 Pi 에 직접 붙어 있으므로 이 프로토콜에 없다.

## 1. 물리 계층 · 프레이밍

| 항목 | 값 |
|---|---|
| 연결 | USB 시리얼 (ESP32 `Serial`) |
| 속도 | 115200 bps, 8N1 |
| 한 메시지 | **JSON 객체 한 줄 + `\n`** (최대 128 바이트) |
| 인코딩 | ASCII (한글 쓰지 않음) |
| 디버그 출력 | 반드시 **`#` 로 시작**하는 줄. Pi 는 `#` 줄과 JSON 이 아닌 줄을 무시(로그만)한다 |

지금 펌웨어의 `Serial.println("Moving Up")` 같은 출력은 `Serial.println("# Moving Up")` 처럼 바꾼다.
필드 이름은 `common/door_common/messages.py` 의 `DeviceEvent`·`ActionCommand` 와 같게 쓴다.
`ts`·`type`·`v` 는 ESP32 가 보내지 않는다. **시각은 Pi 가 받은 순간 `time.monotonic()` 으로 붙인다** (ESP32 `millis()` 는 쓰지 않음).

## 2. ESP32 → Pi

| 줄 | 언제 | Pi 에서 바뀌는 것 |
|---|---|---|
| `{"hello":1,"fw":"car-0.3","proto":1}` | 부팅 직후 1번 | 로그. 이어서 현재 상태를 `door`/`car` 줄로 보낸다 |
| `{"kind":"door","value":"open"}` | 문이 다 열렸을 때 | `DeviceEvent("door","open")` |
| `{"kind":"door","value":"closing"}` | 닫기 시작할 때 | `DeviceEvent("door","closing")` |
| `{"kind":"door","value":"closed"}` | 다 닫혔을 때 | `DeviceEvent("door","closed")` |
| `{"kind":"car","value":"moving"}` | 모터 출발 | `DeviceEvent("car","moving")` |
| `{"kind":"car","value":"stopped"}` | 모터 정지 (도착·비상정지·타임아웃 모두) | `DeviceEvent("car","stopped")` |
| `{"kind":"fault","value":"motor_timeout"}` | 고장 감지 (`stop_with_error()` 자리) | `DeviceEvent("fault","motor_timeout")` |
| `{"ack":12}` | 명령 `seq=12` 를 실행했을 때 | 재전송 멈춤 |
| `{"ack":12,"ok":0,"err":"car_moving"}` | 명령을 거부했을 때 | 로그 (3장 거부 사유) |
| `{"hb":34,"door":"closed","car":"stopped","blk":0,"es":0,"fl":2}` | **200 ms 마다** | 연결 확인 + 상태 맞추기 (아래) |

- 문·카 줄은 **값이 바뀔 때만** 보낸다.
- 하트비트에는 현재 상태를 같이 싣는다 (`blk`: 출발 보류 중, `es`: 비상정지 래치 중, `fl`: 현재 층).
  이벤트 줄 하나가 깨져 사라져도, Pi 는 하트비트의 `door`/`car` 가 마지막으로 아는 값과 다르면 `DeviceEvent` 를 만들어 맞춘다.
- 고장 문자열은 짧은 영문 snake_case: `motor_timeout`, `door_timeout`, `floor_sensor` … `[합의 필요]` 목록

## 3. Pi → ESP32

모든 명령에 `seq` (1부터 증가, 65535 다음 1) 를 붙인다.

| 줄 | ESP32 동작 | 거부(`ok:0`) 하는 경우 |
|---|---|---|
| `{"seq":5,"target":"car","action":"emergency_stop"}` | **즉시** 모터 0, `es=1` 래치, `car stopped` 보고. 래치 동안 모든 이동 거부 | 없음 (항상 실행) |
| `{"seq":6,"target":"car","action":"emergency_reset"}` | 래치 해제 (`es=0`). 저절로 움직이지 않음 | 없음 |
| `{"seq":7,"target":"car","action":"block_start"}` | `blk=1`. 출발 버튼 무시. 이미 움직이는 중이면 이번 이동은 끝내고 다음 출발부터 막음 | 없음 |
| `{"seq":8,"target":"car","action":"block_end"}` | `blk=0` | 없음 |
| `{"seq":9,"target":"door","action":"open"}` | 닫히는 중이어도 **다시 연다**. 다 열리면 `door open` 보고, 문 열림 유지 시간을 처음부터 다시 셈 | `car_moving` (카가 움직이는 중) |
| `{"seq":10,"target":"esp32","action":"report"}` | 지금 상태를 `door`/`car` 줄로 다시 보냄 (Pi 재시작 시 사용) | 없음 |
| `{"hb":77}` | Pi 생존 신호. **200 ms 마다**. ack 하지 않음 | — |

- 모르는 `target`/`action` 은 `{"ack":n,"ok":0,"err":"unknown"}`.
- **같은 `seq` 가 다시 오면** (Pi 재전송) 다시 실행하지 말고 ack 만 다시 보낸다. 최근 seq 1개만 기억하면 된다.
- Pi 는 ack 가 **100 ms** 안에 안 오면 같은 줄을 다시 보낸다 (최대 3번). 그래도 없으면 Pi 쪽에서 `fault esp32_no_ack` 로 처리.

## 4. ESP32 가 지켜야 할 시간

| 항목 | 요구 | 비고 |
|---|---|---|
| 명령 수신 → 실행 | **≤ 50 ms** | 이동 중·문 동작 중에도. 그래서 `delay()` 로 기다리는 루프 금지 (6장) |
| `emergency_stop` → 모터 0 | **≤ 20 ms** `[합의 필요]` | 받은 루프에서 바로 |
| 상태 바뀜 → 보고 | ≤ 50 ms | |
| 하트비트 간격 | 200 ms | |

## 5. 연결이 끊겼을 때 (ESP32 단독 안전 동작)

Pi 하트비트(`{"hb":n}`)가 **600 ms** 동안 안 오면 ESP32 는 "Pi 없음" 상태로 간다.

| 그때 카가 | ESP32 동작 |
|---|---|
| 서 있음 | 출발 금지 (`blk` 와 별개의 내부 플래그). 문 자동 개폐는 그대로 |
| 움직이는 중 | **이번 이동은 목표 층까지 끝내고** 그 뒤 출발 금지 `[합의 필요]` (대안: 즉시 정지) |

- 하트비트가 다시 오면 "Pi 없음" 해제. ESP32 는 `hello` 를 다시 보내지 않아도 된다 (하트비트 상태로 맞춰짐).
- 비상정지 래치(`es`)는 연결이 끊겨도 **유지**한다. 해제는 `emergency_reset` 으로만.
- Pi 쪽도 ESP32 하트비트가 600 ms 동안 없으면 `DeviceEvent("fault","esp32_link")` 를 상태 머신에 넣는다 → `FAULT`.
  다시 들어오면 `DeviceEvent("fault","clear")`. (주의: 지금 상태 머신의 `clear` 는 모든 장치 고장을 지운다.)

## 6. 지금 펌웨어(`Control/Arduino_car`)에서 바꿔야 할 것

1. **막히는 루프 없애기.** `move_up()`·`move_down()` 의 `while` 과 `door_control()` 의 `delay(2000)` 동안에는 시리얼을 읽지 못한다.
   → `loop()` 한 번에 조금씩 진행하는 상태 머신으로 바꾼다 (`millis()` 로 시간 확인).
   ```
   loop():
     read_serial_lines()        // 명령·하트비트 처리 (먼저!)
     update_car()               // CAR_IDLE / CAR_UP / CAR_DOWN : ADC 층 감지, 타임아웃
     update_door()              // DOOR_CLOSED / DOOR_OPENING / DOOR_OPEN / DOOR_CLOSING : 시간으로 진행
     check_pi_watchdog()
     send_heartbeat_if_due()
     read_buttons()             // 에지 감지 (while 로 떼기 기다리지 않음)
   ```
2. **문 상태 정의.** 서보는 위치를 알려주지 않는다. `open`/`closed` 를 무엇으로 볼지 정한다 `[합의 필요]`
   - 안 A: 각도 명령 후 `DOOR_MOVE_MS`(예: 600 ms) 지나면 도착으로 봄 (센서 없음)
   - 안 B: 리밋·근접 센서로 확인 (README 의 "열림/닫힘 위치 감지")
3. **출발 조건.** 출발 버튼 처리에 `!blk && !es && pi_alive && door == CLOSED` 를 넣는다.
4. **비상정지 래치.** 기존 `fault` 플래그와 별도로 `es` 를 둔다. 비상정지 뒤 카는 층 사이일 수 있다.
   리셋 뒤 어떻게 할지 `[합의 필요]` (안: 남은 이동을 이어서 하지 않고, 버튼으로 다시 목표 층을 받음. 층 수는 ADC 로 다시 맞춤)
   층 사이에 선 채로 `door open` 이 오면 열지 말지도 `[합의 필요]` (지금 초안·가짜 ESP32 는 카가 서 있으면 연다.
   실제 엘리베이터는 층 사이에서 문을 열지 않으므로 `{"ack":n,"ok":0,"err":"between_floors"}` 로 거부하는 안도 있음)
5. **디버그 출력 앞에 `#`.**

## 7. 시나리오 (펌웨어 검증용)

| # | 상황 | Pi 가 보냄 | ESP32 에서 나와야 하는 줄 |
|---|---|---|---|
| 1 | 부팅 | — | `hello` → `door closed` → `car stopped` → 200 ms 마다 `hb` |
| 2 | 버튼으로 2층 이동 | — | `door closing`·`closed` (닫혀 있으면 생략) → `car moving` → … → `car stopped` → `door open` → (유지 시간 뒤) `door closing` → `door closed` |
| 3 | 닫히는 중 다시 열기 | `door open` | `ack` → `door open` (유지 시간 다시 셈) |
| 4 | 이동 중 비상정지 | `car emergency_stop` | `ack` → `car stopped` → `hb` 에 `es:1`. 버튼 눌러도 안 움직임 |
| 5 | 리셋 | `car emergency_reset` | `ack` → `hb` 에 `es:0`. 저절로 움직이지 않음 |
| 6 | 출발 보류 | `car block_start` 후 출발 버튼 | `ack`, `hb` 에 `blk:1`, `car moving` 이 **나오지 않음** |
| 7 | 이동 중 문 열기 | `door open` | `{"ack":n,"ok":0,"err":"car_moving"}` |
| 8 | 같은 seq 두 번 | 같은 줄 두 번 | ack 두 번, 동작은 한 번 |
| 9 | Pi 끊김 | 하트비트 멈춤 | 600 ms 뒤 출발 버튼 무시 (서 있을 때) |

## 8. 테스트 도구 (`Control/pi/`)

```bash
cd Control/pi
python -m pytest -q                                          # 프로토콜·가짜 ESP32 테스트
python -m door_control.esp32_bridge --fake                   # 하드웨어 없이: 가짜 ESP32 와 대화
python -m door_control.esp32_bridge --port /dev/ttyUSB0      # 실제 ESP32 와 대화
```

브리지 창에서 명령을 손으로 넣는다:
```
estop | reset | block | unblock | open | report      → ESP32 로 명령
go 3                                                  → (가짜 ESP32 만) 3층 버튼
quiet                                                 → 하트비트 송신 멈춤/재개 (끊김 시험)
```
받은 줄과 만들어진 `DeviceEvent` 가 시각과 함께 찍힌다. 시나리오 1~9 를 이걸로 확인한다.

Pi 메인 프로그램에서는 이렇게 붙인다:
```python
bridge = Esp32Bridge(port="/dev/ttyUSB0")
sm = SafetyStateMachine(rules, door="closed", car="stopped", start_ts=now())
while True:
    for ev in bridge.poll():                 # DeviceEvent (문·카·고장·연결)
        route(sm.handle(ev))
    route(sm.tick(now()))
    ...
def route(outs):
    for m in outs:
        if isinstance(m, ActionCommand) and m.target in ("door", "car"):
            bridge.send(m)                   # seq·재전송·ack 는 브리지가 처리
        ...                                  # buzzer/beacon/voice/notify 는 Pi 에서 바로
```
