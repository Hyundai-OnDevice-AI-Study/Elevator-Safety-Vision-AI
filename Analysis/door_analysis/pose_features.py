"""자세 해석 — 설계 문서 3장 ① (영상) 규칙의 참고 구현.

PoseFrame(키포인트)을 시간 순서로 받아 사람마다 '자세 근거 수준'을 낸다.

  수준 0 NORMAL  : 정상
  수준 1 OBSERVE : 주의 관찰   — 뻗은 팔이 문틈의 한 점을 계속 가리킴 / 팔이 문쪽으로 당겨짐
  수준 2 CAUTION : 경계       — 손은 문틈(또는 팔이 당겨짐), 몸은 문 반대쪽으로 기울여 버팀
  수준 3 DANGER  : 위험       — dragged_toward_door  : 걷지 않는데 문틈 근처 손을 따라 몸이 끌려감
                               pulled_then_dragged  : 몸이 멈춘 채 팔이 먼저 당겨진 뒤 몸이 끌려옴 (손이 멀어도)
                               hand_sliding_along_gap: 문에 닿은 손이 문틈을 따라 빠르게 미끄러짐
                               collapse_after_brace : 버티다가 무게중심이 문쪽으로 갑자기 쏠림
  낮춤          : 손이 버튼 패널에 머물거나 등을 기대는 자세면 약한 단계(1, 2)만 한 단계 낮춤.
                 DANGER 는 낮추지 않는다.

설계 원칙
  - 모든 거리는 '몸통 길이(어깨 중점~엉덩이 중점)'로 나눈 값이다. 몸통 길이는 최근 프레임들의
    중앙값을 써서 숙이거나 무릎을 꿇어도 기준이 흔들리지 않게 한다.
  - 모든 속도는 '몸통 길이/초'. 프레임 수가 아니라 실제 시간(ts)으로 계산하므로
    FPS가 흔들려도 결과가 같다.
  - 임계값은 전부 rules.json 의 "pose" 섹션에서 읽는다. 코드에 숫자를 박지 않는다.
  - 한 프레임의 판단으로 바로 DANGER를 내지 않는다. 최근 n 프레임 중 k 번 이상일 때만(투표).
  - 출력 level 은 안정화(최소 지속·유지 시간)를 거친 값이고, raw_level 은 그 프레임의 순간 값이다.
    '문이 열릴 때까지 유지'는 상태 머신(state)이 맡는다.

v0.4(룰 기반 엔진, 2026-11) 에서 가져온 것
  스무딩 신뢰도 게이트·끊김 후 초기화, 몸통 길이 중앙값, 좌우 짝 조건, 팔 당김 → 끌림 순서 규칙(PULLED),
  COLLAPSE, 미끄러짐 기준(2.0 torso/s, 0.3s 창, 문틈 너머 음수 거리 허용), 버튼 패널은 약한 단계만 낮춤,
  수준 안정화(Stabilizer), 잠깐 안 보이는 사람도 안정 수준 유지.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from door_common.messages import PoseFrame
from door_common import keypoints as K
from door_common.door_geometry import DoorGeometry
from .smoothing import KeypointSmoother
from .stabilizer import Stabilizer

LEVEL_NAMES = {0: "NORMAL", 1: "OBSERVE", 2: "CAUTION", 3: "DANGER"}

DEFAULT_RULES = {
    "kp_conf": 0.35,          # 이보다 신뢰도가 낮은 키포인트는 없는 것으로 본다
    "window_s": 1.5,          # 판단에 쓰는 최근 시간 창
    "vel_window_s": 0.5,      # 속도 계산 구간
    "near_gap": 0.35,         # 손목-문틈 거리 / 몸통  (이 안이면 '문틈 근처')
    "contact": 0.15,          # 손목-문틈 거리 / 몸통  (이 안이면 '문에 닿음')
    "arm_extended_deg": 145,  # 팔꿈치 각도가 이 이상이면 '뻗은 팔'
    "reach_min": 0.25,        # (어깨-문틈 거리) - (손목-문틈 거리) / 몸통. 팔을 문쪽으로 내민 정도
    "dwell_s": 0.8,           # 같은 점을 가리키는 최소 지속 시간
    "dwell_jitter": 0.10,     # 그동안 손목이 움직인 범위 / 몸통
    "lean_away": 0.20,        # 상체 기울기(sin) — 문 반대쪽으로 이만큼 기울면 '버팀'
    "brace_s": 0.4,           # 버팀 자세 최소 지속 시간
    "drag_speed": 0.5,        # 엉덩이가 문쪽으로 움직이는 속도 / (몸통/초)
    "walk_rel_ankle": 0.6,    # 두 발목의 상대 속도 / (몸통/초)  — 이 이상이면 '걷는 중'
    "stumble_overrides_walking": True,  # 손이 문틈에 닿은 채 발을 내딛으면 걷기가 아니라 끌림
    "onset_disp": 0.12,       # '움직이기 시작'으로 보는 누적 이동량 / 몸통
    "slide_speed": 2.0,       # 문틈을 따라 손목이 미끄러지는 속도 / (몸통/초). v0.4 실측: 정상 동작 최대 1.26, 사고 2.1~3.4
    "slide_window_s": 0.3,    # 미끄러짐 속도만 짧은 창으로 계산 (v0.4 hand_slide_s). 0.5초면 8fps 에서 평균에 묻힘
    "slide_vote_k": 2,        # 미끄러짐은 0.3~0.5초에 끝나는 짧은 사건 → n 프레임 중 2번이면 인정 (8fps 에서 3번은 못 채움)
    "slide_contact_min": -1.0,  # 미끄러짐 판정용 '문에 닿음' 범위. 2D 영상에서는 문 앞의 손이 문틈 선을
    "slide_contact_max": 0.30,  # 넘어가 보이므로(음수 거리) 음수 쪽을 넓게 둔다 (v0.4 hand_slide)
    "pull_dir_deg": 40,       # 어깨→손목 방향과 '문틈 쪽' 사이 각도. 이 이하면 팔이 문쪽으로 뻗어 있음
    "pull_hold_s": 0.4,       # 그 자세가 이만큼 유지되면 '팔이 문쪽으로 당겨짐' (목줄·끈 끼임)
    "pull_hold_ratio": 0.8,   # 유지 구간 중 이 비율 이상의 프레임이 조건을 만족해야 함
    "pull_brace_lean": 0.10,  # 팔이 당겨진 상태에서 이만큼만 뒤로 기울어도 '버팀' (v0.4 lean_brace_max)
    "pull_drag_speed": 0.3,   # 팔이 당겨진 뒤 엉덩이가 이 속도 이상으로 문쪽 이동 → 끌림 (v0.4 hip_speed_min)
    "pull_memory_s": 1.0,     # '몸이 멈춘 채 팔이 당겨짐'을 이 시간 동안 기억
    "collapse_recent_s": 1.0, # 최근 이 시간 안에 버팀이 있었고
    "collapse_jump": 0.30,    # 엉덩이가 발보다 문쪽으로 이만큼 갑자기 쏠리면 → 무너짐 (v0.4 collapse)
    "back_lean_dist": 0.25,   # 어깨 중점이 문틈에서 이 안이면 '등을 기댐' 후보
    "vote_k": 3,
    "vote_n": 5,
    "lower_on_button_panel": True,
    "track_timeout_s": 1.0,
    "smooth": True,           # One Euro 필터로 키포인트 떨림 제거
    "smooth_min_cutoff": 1.5,
    "smooth_beta": 5.0,
    "stabilize": True,        # 수준 안정화 (v0.4). level = 안정 수준, raw_level = 순간 수준
    "stab_min_on_s": [0.0, 0.2, 0.2, 0.0],   # 수준 0~3: 이만큼 계속돼야 켜짐. DANGER 는 투표(vote)가 이미 걸러서 0
    "stab_hold_s": [0.0, 1.0, 1.5, 2.0],     # 수준 0~3: 켜진 뒤 raw 가 내려가도 이만큼 유지
    "smooth_max_gap_s": 0.3,
    "torso_median_n": 31,     # 몸통 길이 = 최근 n 프레임(좌우 짝이 다 보인 프레임)의 중앙값 (v0.4)  # 관절이 이보다 오래 안 보였다 나타나면 그 관절 필터를 새로 시작 (v0.4)
}


# --------------------------------------------------------------- 기하 도우미
def _pt(kp: np.ndarray, i: int, conf: float) -> Optional[np.ndarray]:
    return kp[i, :2] if kp[i, 2] >= conf else None


def _mid(a, b):
    if a is None or b is None:
        return a if b is None else b
    return (a + b) / 2


def elbow_angle(kp: np.ndarray, arm: Tuple[int, int, int], conf: float) -> Optional[float]:
    s, e, w = (_pt(kp, i, conf) for i in arm)
    if s is None or e is None or w is None:
        return None
    v1, v2 = s - e, w - e
    c = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    return math.degrees(math.acos(float(np.clip(c, -1, 1))))


def slope(ts: np.ndarray, vs: np.ndarray) -> float:
    """최소제곱 기울기(단위/초). 점이 2개 미만이면 0."""
    if len(ts) < 2 or ts[-1] - ts[0] < 1e-3:
        return 0.0
    t = ts - ts.mean()
    return float(np.dot(t, vs - vs.mean()) / (np.dot(t, t) + 1e-12))


# --------------------------------------------------------------- 프레임별 측정값
@dataclass
class Measure:
    ts: float
    torso: float                         # 몸통 길이 (정규화 좌표)
    d_hip: Optional[float]               # 엉덩이 중점 → 문틈 거리 / 몸통 (+: 카 안쪽)
    d_sh: Optional[float]                # 어깨 중점
    wrists: Dict[str, Tuple[float, float]]   # arm → (문틈 거리/몸통, 문틈 따라 위치/몸통)
    elbows: Dict[str, Optional[float]]       # arm → 팔꿈치 각도
    arm_dir: Dict[str, float]                # arm → 어깨→손목 방향과 문틈 쪽 사이 각도(도)
    lean: Optional[float]                # 상체 기울기 sin. +: 문 반대쪽(카 안쪽)으로 기욺
    ankles: Dict[str, Tuple[float, float]]   # 발목 좌표 / 몸통 (걷기 판정용)
    in_panel: bool
    d_ank: Optional[float] = None        # 두 발목 중점 → 문틈 거리 / 몸통 (두 발목 다 보일 때만)


def measure(kp: np.ndarray, ts: float, door: DoorGeometry, conf: float,
            torso_hist: Optional[Deque[float]] = None) -> Optional[Measure]:
    """한 프레임의 측정값.

    v0.4 에서 가져온 두 가지:
      - 몸통 길이는 최근 값들의 중앙값(torso_hist). 숙이거나 무릎을 꿇어 한 프레임의 몸통이
        짧아 보여도 정규화 기준이 흔들리지 않는다.
      - 엉덩이·어깨 위치와 상체 기울기는 좌우 짝이 모두 보일 때만 계산한다. 한쪽만 보일 때
        그 점을 중점 대신 쓰면 중점이 한쪽으로 쏠려 가짜 '버팀'이나 가짜 '이동'이 생긴다.
    """
    ls, rs = _pt(kp, K.L_SHOULDER, conf), _pt(kp, K.R_SHOULDER, conf)
    lh, rh = _pt(kp, K.L_HIP, conf), _pt(kp, K.R_HIP, conf)
    sh, hip = _mid(ls, rs), _mid(lh, rh)
    if sh is None or hip is None:
        return None                     # 몸통이 안 보이면 이 프레임은 판단하지 않는다
    full = ls is not None and rs is not None and lh is not None and rh is not None
    torso_now = float(np.linalg.norm(sh - hip))
    if torso_hist is not None and full and torso_now >= 1e-3:
        torso_hist.append(torso_now)
    torso = float(np.median(torso_hist)) if torso_hist else torso_now
    if torso < 1e-3:
        return None
    wrists, elbows, arm_dir, in_panel = {}, {}, {}, False
    for name, arm in K.ARMS.items():
        w = _pt(kp, arm[2], conf)
        sp = _pt(kp, arm[0], conf)
        if w is not None:
            wrists[name] = (door.signed_distance(w) / torso, door.along(w) / torso)
            in_panel |= door.in_button_panel(w)
            if sp is not None:
                v = w - sp
                c = float(np.dot(v, -door.normal) / (np.linalg.norm(v) + 1e-9))
                arm_dir[name] = math.degrees(math.acos(max(-1.0, min(1.0, c))))
        elbows[name] = elbow_angle(kp, arm, conf)
    up = sh - hip
    lean = float(np.dot(up, door.normal) / (np.linalg.norm(up) + 1e-9)) if full else None
    ankles = {}
    for name, i in (("left", K.L_ANKLE), ("right", K.R_ANKLE)):
        a = _pt(kp, i, conf)
        if a is not None:
            ankles[name] = (float(a[0]) / torso, float(a[1]) / torso)
    d_hip = door.signed_distance(hip) / torso if (lh is not None and rh is not None) else None
    d_sh = door.signed_distance(sh) / torso if (ls is not None and rs is not None) else None
    la, ra = _pt(kp, K.L_ANKLE, conf), _pt(kp, K.R_ANKLE, conf)
    d_ank = door.signed_distance((la + ra) / 2) / torso if (la is not None and ra is not None) else None
    return Measure(ts, torso, d_hip, d_sh, wrists, elbows, arm_dir, lean, ankles, in_panel, d_ank)


# --------------------------------------------------------------- 사람별 상태
@dataclass
class TrackState:
    hist: Deque[Measure] = field(default_factory=deque)
    votes: Dict[str, Deque[bool]] = field(default_factory=dict)
    last_ts: float = 0.0
    smoother: Optional[KeypointSmoother] = None
    torso_hist: Optional[Deque[float]] = None
    stab: Optional[Stabilizer] = None
    pull_still_ts: Optional[float] = None   # 몸이 가만히 있는 동안 팔이 당겨져 있던 마지막 시각
    tension_ts: Optional[float] = None      # 마지막으로 '버팀'(수준 2)이었던 시각
    stab: Optional[Stabilizer] = None
    pull_still_ts: Optional[float] = None   # 몸이 가만히 있는 동안 팔이 당겨져 있던 마지막 시각
    tension_ts: Optional[float] = None      # 마지막으로 '버팀'(수준 2)이었던 시각


@dataclass
class PersonAssessment:
    track_id: int
    level: int                     # 안정 수준 (stabilize=False 면 raw_level 과 같음)
    reasons: List[str]
    flags: Dict[str, bool]
    debug: Dict[str, float]
    raw_level: int = 0             # 이 프레임 규칙만으로 낸 순간 수준


@dataclass
class PoseAssessment:
    ts: float
    level: int
    level_name: str
    persons: List[PersonAssessment]
    pet_near_door: bool = False


class PoseAnalyzer:
    def __init__(self, door: DoorGeometry, rules: Optional[dict] = None):
        self.door = door
        self.r = {**DEFAULT_RULES, **(rules or {})}
        self.tracks: Dict[int, TrackState] = {}
        self._pet_since: Optional[float] = None

    # ------------------------------------------------------------ 공개 API
    def update(self, pf: PoseFrame, pet_rules: Optional[dict] = None) -> PoseAssessment:
        persons = []
        aspect = pf.img_w / pf.img_h if pf.img_w > 0 and pf.img_h > 0 else 1.0
        door = self._door_for(aspect)
        for p in pf.persons:
            kp = np.asarray(p.keypoints, dtype=np.float32)
            kp[:, 0] *= aspect                    # 정규화 좌표 → 가로·세로 같은 단위
            st = self.tracks.setdefault(p.track_id, TrackState())
            if self.r["smooth"]:
                if st.smoother is None:
                    st.smoother = KeypointSmoother(self.r["smooth_min_cutoff"], self.r["smooth_beta"],
                                                   min_conf=self.r["kp_conf"], max_gap_s=self.r["smooth_max_gap_s"])
                kp = st.smoother(kp, pf.ts)
            if st.torso_hist is None:
                st.torso_hist = deque(maxlen=int(self.r["torso_median_n"]))
            m = measure(kp, pf.ts, door, self.r["kp_conf"], st.torso_hist)
            st.last_ts = pf.ts
            if m is not None:
                st.hist.append(m)
                while st.hist and pf.ts - st.hist[0].ts > self.r["window_s"]:
                    st.hist.popleft()
            pa = self._assess(p.track_id, st)
            pa.raw_level = pa.level
            if self.r["stabilize"]:
                if st.stab is None:
                    st.stab = Stabilizer(self.r["stab_min_on_s"], self.r["stab_hold_s"])
                pa.level = st.stab.update(pf.ts, pa.raw_level)
                if pa.level > pa.raw_level:
                    pa.reasons.append(f"hold:{LEVEL_NAMES[pa.level]}")
            persons.append(pa)
        # 이번 프레임에 안 보인 사람: 추적 타임아웃 전까지는 안정 수준을 계속 낸다 (v0.4 pipeline).
        # 끌려가며 가려지거나 검출이 한두 프레임 빠져도 DANGER 가 바로 사라지지 않게.
        seen = {p.track_id for p in pf.persons}
        for tid, st in list(self.tracks.items()):
            if tid in seen:
                continue
            if pf.ts - st.last_ts > self.r["track_timeout_s"]:
                del self.tracks[tid]                      # 오래 안 보인 사람 정리
            elif st.stab is not None:
                lv = st.stab.update(pf.ts, 0)
                if lv > 0:
                    persons.append(PersonAssessment(tid, lv, ["not_visible", f"hold:{LEVEL_NAMES[lv]}"],
                                                    {}, {}, raw_level=0))

        pet_near = self._pets(pf, pet_rules or {})
        level = max([a.level for a in persons], default=0)
        return PoseAssessment(pf.ts, level, LEVEL_NAMES[level], persons, pet_near)

    def _door_for(self, aspect: float) -> DoorGeometry:
        if getattr(self, "_door_aspect", None) != aspect:
            self._door_aspect, self._door_scaled = aspect, self.door.scaled(aspect)
        return self._door_scaled

    def reset(self) -> None:
        """문이 열리면 판정을 초기화한다 (설계 문서 3장 ③)."""
        self.tracks.clear()
        self._pet_since = None

    # ------------------------------------------------------------ 내부
    def _vote(self, st: TrackState, key: str, value: bool, k: Optional[int] = None) -> bool:
        q = st.votes.setdefault(key, deque(maxlen=self.r["vote_n"]))
        q.append(value)
        return sum(q) >= (self.r["vote_k"] if k is None else k)

    def _recent(self, st: TrackState, seconds: float) -> List[Measure]:
        if not st.hist:
            return []
        t_end = st.hist[-1].ts
        return [m for m in st.hist if t_end - m.ts <= seconds]

    def _assess(self, tid: int, st: TrackState) -> PersonAssessment:
        r = self.r
        dbg: Dict[str, float] = {}
        flags = {k: False for k in ("pointing", "bracing", "dragged", "sliding", "arm_pulled", "pulled_drag",
                                    "collapse", "button_panel", "leaning_on_door", "walking")}
        if not st.hist:
            return PersonAssessment(tid, 0, ["no_valid_pose"], flags, dbg)
        cur = st.hist[-1]
        vw = self._recent(st, r["vel_window_s"])

        # 문틈에 가장 가까운 손 (없으면 None)
        near_arm = None
        if cur.wrists:
            near_arm = min(cur.wrists, key=lambda a: abs(cur.wrists[a][0]))
            dbg["wrist_d"] = round(cur.wrists[near_arm][0], 3)
        hand_near = near_arm is not None and abs(cur.wrists[near_arm][0]) <= r["near_gap"]
        hand_contact = near_arm is not None and abs(cur.wrists[near_arm][0]) <= r["contact"]

        # ---- 걷기 여부: 두 발목이 서로 다르게 움직이면 걷는 중
        rel = [m for m in vw if "left" in m.ankles and "right" in m.ankles]
        walking = False
        if len(rel) >= 3:
            dx = np.array([m.ankles["left"][0] - m.ankles["right"][0] for m in rel])
            dy = np.array([m.ankles["left"][1] - m.ankles["right"][1] for m in rel])
            t_rel = np.array([m.ts for m in rel])
            rel_speed = math.hypot(slope(t_rel, dx), slope(t_rel, dy))
            # 발 교차 왕복은 기울기가 0에 가까울 수 있으므로 변화 범위도 함께 본다
            span = (dx.max() - dx.min()) / max(t_rel[-1] - t_rel[0], 1e-3)
            dbg["ankle_rel"] = round(max(rel_speed, span), 3)
            walking = max(rel_speed, span) >= r["walk_rel_ankle"]
        flags["walking"] = walking

        # ---- 엉덩이 속도 (문쪽 = d 감소 = 음수 기울기)
        hip_pts = [(m.ts, m.d_hip) for m in vw if m.d_hip is not None]
        hip_v = -slope(np.array([a for a, _ in hip_pts]), np.array([b for _, b in hip_pts])) if hip_pts else 0.0
        dbg["hip_v_to_door"] = round(hip_v, 3)

        # ---- 손이 먼저 움직였나 (window 전체)
        hand_first = self._hand_first(st) if near_arm else False
        dbg["hand_first"] = float(hand_first)

        # ---- 규칙 ①-3: 걷지 않는데 몸이 문쪽으로 끌려감
        # 끌려가는 사람은 비틀거리며 발을 내딛는다 → '걷는 중'으로 보이지만,
        # 손이 문틈에 붙은 채로 몸이 문쪽으로 빨려 가면 걷기가 아니라 끌림으로 본다.
        stumble = walking and hand_contact and r["stumble_overrides_walking"]
        dragged_now = (hip_v >= r["drag_speed"] and (not walking or stumble) and hand_near and
                       (hand_first or hand_contact))
        flags["dragged"] = self._vote(st, "dragged", dragged_now)

        # ---- 규칙 ①-4: 문에 닿은 손이 문틈을 따라 빠르게 미끄러짐
        slide_now = False
        on_door = lambda d: r["slide_contact_min"] <= d <= r["slide_contact_max"]
        if near_arm and on_door(cur.wrists[near_arm][0]):
            sw = self._recent(st, r["slide_window_s"])      # 짧은 미끄러짐도 잡도록 짧은 창 (v0.4)
            pts = [(m.ts, m.wrists[near_arm][1]) for m in sw if near_arm in m.wrists]
            # 손을 문틈으로 '가져가는' 동작과 구분: 속도 구간 내내 문에 닿아 있어야 한다
            touching = all(near_arm in m.wrists and on_door(m.wrists[near_arm][0]) for m in sw)
            if len(pts) >= 2 and touching:
                sv = abs(slope(np.array([a for a, _ in pts]), np.array([b for _, b in pts])))
                dbg["slide_v"] = round(sv, 3)
                slide_now = sv >= r["slide_speed"]
        flags["sliding"] = self._vote(st, "sliding", slide_now, r["slide_vote_k"])

        # ---- 규칙 ①-2: 손은 문틈, 몸은 문 반대쪽으로 기울여 버팀
        brace_win = self._recent(st, r["brace_s"])
        brace_now = (hand_near and cur.lean is not None and cur.lean >= r["lean_away"] and
                     len(brace_win) >= 2 and
                     all(m.lean is not None and m.lean >= r["lean_away"] * 0.8 for m in brace_win))
        dbg["lean"] = round(cur.lean or 0.0, 3)
        flags["bracing"] = self._vote(st, "bracing", brace_now)

        # ---- 규칙 ①-1: 뻗은 팔이 문틈의 한 점을 계속 가리킴
        point_now = False
        # 팔을 그냥 내리고 있어도 팔꿈치는 펴져 있으므로, 손이 어깨보다 문쪽으로 나가 있어야 '가리킴'
        reach = (cur.d_sh - cur.wrists[near_arm][0]) if (near_arm and cur.d_sh is not None) else 0.0
        dbg["reach"] = round(reach, 3)
        if (near_arm and hand_near and reach >= r["reach_min"]
                and (cur.elbows.get(near_arm) or 0) >= r["arm_extended_deg"]):
            dw = [m for m in self._recent(st, r["dwell_s"]) if near_arm in m.wrists]
            if dw and dw[-1].ts - dw[0].ts >= r["dwell_s"] * 0.9:
                alongs = np.array([m.wrists[near_arm][1] for m in dw])
                dists = np.array([m.wrists[near_arm][0] for m in dw])
                # 최대-최소(ptp)는 키포인트 떨림 한 번에 크게 튄다 → 10~90 백분위 폭을 쓴다
                rng = lambda v: float(np.percentile(v, 90) - np.percentile(v, 10))
                jitter = max(rng(alongs), rng(dists))
                dbg["point_jitter"] = round(float(jitter), 3)
                point_now = jitter <= r["dwell_jitter"]
        flags["pointing"] = self._vote(st, "pointing", point_now)

        # ---- 팔이 문쪽으로 당겨짐 (목줄·끈이 문에 끼어 팽팽해진 상태)
        # 손이 문틈에서 멀어도(끈 길이만큼) 팔은 문쪽으로 곧게 뻗어 있다. 거리 대신 '방향'을 본다.
        def pulled(m: Measure) -> Optional[str]:
            for arm in m.arm_dir:
                if (m.arm_dir[arm] <= r["pull_dir_deg"] and (m.elbows.get(arm) or 0) >= r["arm_extended_deg"]
                        and m.d_sh is not None and arm in m.wrists and m.wrists[arm][0] < m.d_sh):
                    return arm
            return None
        hold = self._recent(st, r["pull_hold_s"])
        if hold and hold[-1].ts - hold[0].ts >= r["pull_hold_s"] * 0.8:
            ratio = sum(pulled(m) is not None for m in hold) / len(hold)
            flags["arm_pulled"] = ratio >= r["pull_hold_ratio"]
            # 버튼을 누르는 손은 패널에 머문다. 예전에는 이때 arm_pulled 자체를 껐지만,
            # 끌려가는 손이 패널 앞을 지날 때 판정이 통째로 사라졌다 (cam03 5.8s).
            # v0.4 처럼 '약한 단계만 낮추는 조건'으로만 쓴다 → 아래 낮춤 조건 참고.
            flags["button_panel"] = sum(m.in_panel for m in hold) / len(hold) >= 0.5
        if near_arm and cur.arm_dir.get(near_arm) is not None:
            dbg["arm_dir"] = round(cur.arm_dir[near_arm], 1)

        # ---- 팔이 당겨진 채 몸이 끌려옴 (v0.4 PULLED)
        # 줄이 팽팽해지는 동안(팔이 당겨짐) 몸은 아직 멈춰 있다가 → 그다음 몸이 문쪽으로 움직인다.
        # 이 '순서'가 비자발성의 근거다. 팔을 뻗은 채 몸 전체가 같이 움직이면(카메라 흔들림,
        # 그냥 문쪽으로 다가감) 팔이 당겨진 시점에 이미 몸이 움직이고 있으므로 걸러진다.
        if flags["arm_pulled"] and hip_v < r["pull_drag_speed"]:
            st.pull_still_ts = cur.ts
        pull_recent = st.pull_still_ts is not None and cur.ts - st.pull_still_ts <= r["pull_memory_s"]
        # 걷는 중이면 인정하지 않는다: 목줄을 쥔 채 개를 앞세워 걸어 나가는 정상 동작과 구분.
        # 비틀거리며 끌려가는 경우는 COLLAPSE(아래)와 손 접촉 끌림(dragged)이 맡는다.
        pulled_drag_now = pull_recent and hip_v >= r["pull_drag_speed"] and not walking
        flags["pulled_drag"] = self._vote(st, "pulled_drag", pulled_drag_now)

        # ---- 버티다 무너짐 (v0.4 COLLAPSE)
        # 엉덩이가 발보다 문쪽으로 갑자기 쏠림 = 버티던 다리가 풀리며 끌려 들어감
        com = [(m.ts, m.d_ank - m.d_hip) for m in self._recent(st, r["collapse_recent_s"])
               if m.d_ank is not None and m.d_hip is not None]
        if com and cur.d_ank is not None and cur.d_hip is not None:
            jump = (cur.d_ank - cur.d_hip) - min(v for _, v in com)
            dbg["com_jump"] = round(jump, 3)
            recent_tension = st.tension_ts is not None and cur.ts - st.tension_ts <= r["collapse_recent_s"]
            flags["collapse"] = recent_tension and jump >= r["collapse_jump"] and hip_v > 0

        # ---- 낮춤 조건
        # 등을 기댐: 어깨가 문틈에 붙어 있고, 손이 몸보다 문쪽으로 나가 있지 않음
        hands_behind = all(d >= (cur.d_sh or 0) - 0.05 for d, _ in cur.wrists.values())
        flags["leaning_on_door"] = (cur.d_sh is not None and abs(cur.d_sh) <= r["back_lean_dist"]
                                    and hands_behind and not flags["pointing"])

        level, reasons = 0, []
        if flags["dragged"]:
            level, reasons = 3, reasons + ["dragged_toward_door"]
        if flags["pulled_drag"]:
            level, reasons = 3, reasons + ["pulled_then_dragged"]
        if flags["sliding"]:
            level, reasons = 3, reasons + ["hand_sliding_along_gap"]
        if flags["collapse"]:
            level, reasons = 3, reasons + ["collapse_after_brace"]
        if level < 3 and flags["bracing"]:
            level, reasons = 2, reasons + ["bracing_against_door"]
        if level < 2 and flags["arm_pulled"] and cur.lean is not None and cur.lean >= r["pull_brace_lean"]:
            level, reasons = 2, reasons + ["bracing_against_pull"]      # 문쪽으로 당겨진 팔 + 몸은 반대로 버팀
        if level == 2:
            st.tension_ts = cur.ts                                      # COLLAPSE 판정용 (낮춤 전 기준)
        if level < 1 and flags["arm_pulled"]:
            level, reasons = 1, reasons + ["arm_pulled_toward_door"]
        if level < 2 and flags["pointing"]:
            level, reasons = 1, reasons + ["arm_fixed_on_gap"]
        # 낮춤은 약한 단계(1, 2)에만. DANGER 는 버튼 패널 근처라도 낮추지 않는다 (v0.4).
        if (level in (1, 2) and r["lower_on_button_panel"]
                and (flags["button_panel"] or flags["leaning_on_door"])):
            level -= 1
            reasons.append("lowered:button_panel_or_back_lean")
        return PersonAssessment(tid, level, reasons, flags, dbg)

    def _hand_first(self, st: TrackState) -> bool:
        """창 안에서 손목이 엉덩이보다 먼저 문쪽으로 움직이기 시작했는가."""
        r = self.r
        h = list(st.hist)
        arms = [a for a in K.ARMS if all(a in m.wrists for m in h[-3:])]
        if not arms or len(h) < 3:
            return False

        def onset(vals: List[Optional[float]]) -> Optional[int]:
            base = next((v for v in vals if v is not None), None)
            if base is None:
                return None
            for i, v in enumerate(vals):
                if v is not None and base - v >= r["onset_disp"]:
                    return i
            return None

        hip_on = onset([m.d_hip for m in h])
        for a in arms:
            w_on = onset([m.wrists[a][0] if a in m.wrists else None for m in h])
            # 손목이 처음부터 문에 닿아 있었으면(이미 끼인 상태) 손이 먼저인 것으로 본다
            caught = all(a in m.wrists and abs(m.wrists[a][0]) <= r["contact"] for m in h[:3])
            if caught or (w_on is not None and (hip_on is None or w_on <= hip_on)):
                return True
        return False

    def _pets(self, pf: PoseFrame, pr: dict) -> bool:
        """2차: 반려동물이 문 주변 영역에 일정 시간 이상 머무르면 True."""
        near = any(p.in_door_zone and p.score >= pr.get("min_score", 0.4) for p in pf.pets)
        if not near:
            self._pet_since = None
            return False
        if self._pet_since is None:
            self._pet_since = pf.ts
        return pf.ts - self._pet_since >= pr.get("dwell_s", 0.5)
