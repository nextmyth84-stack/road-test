# app.py — 도로주행 자동 배정 (오전/오후 분리 · 1종수동 기본배정 · 혼합 최소화 · '수동 전용' 삭제)
# 실행: streamlit run app.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional
import re
import pandas as pd
import streamlit as st

# -----------------------
# 상수/라벨
# -----------------------
CLS_ORDER      = ("1M","1A","2A","2M")   # 내부 코드 순서
CLS_KO         = {"1M":"1종수동", "1A":"1종자동", "2A":"2종자동", "2M":"2종수동"}
AM_PERIODS     = [1,2]
PM_PERIODS     = [3,4,5]
MIX_DAILY_CAP  = 1      # 혼합(헤더와 다른 종) 일일 상한 1건
MIX_PERIOD_CAP = 0      # 혼합 교시 기본 0건(불가피하면 1까지 완화)
SWITCH_PENALTY = 0.8    # 새 종(혼합) 페널티
COURSE_PENALTY = 0.8    # 코스점검 가중치

def ko(cls: str) -> str:
    return CLS_KO.get(cls, cls)

def period_capacity(p: int) -> int:
    return 2 if p in (1,5) else 3  # 1·5교시=2, 2·3·4교시=3

# -----------------------
# 데이터 모델
# -----------------------
@dataclass
class Staff:
    name: str
    skills: Set[str]                 # 가능한 종 집합
    morning_course_check: bool=False # 코스점검 여부(오전/오후 텍스트에서 자동 인식)

@dataclass
class Options:
    switch_penalty: float = SWITCH_PENALTY
    course_penalty: float = COURSE_PENALTY
    mix_daily_cap: int = MIX_DAILY_CAP
    mix_period_cap: int = MIX_PERIOD_CAP

Assignment = Dict[int, Dict[str, Dict[str,int]]]  # [교시][이름][종] = 인원수

# -----------------------
# 텍스트 파서(오전/오후 붙여넣기)
# -----------------------
NAME = r"[가-힣]{2,4}"
PLATE_NAME = re.compile(r"(?:•\s*)?(?:\d+\s*호\s*)?(?P<name>"+NAME+")")
COURSE_LINE = re.compile(r"(?:•\s*)?.*?(합격|불합격)\s*:\s*(?P<name>"+NAME+")")

def parse_staff_text(src: str):
    """
    붙여넣은 원문에서 섹션별 이름을 추출.
    반환:
      {
        'classes': {'1M':[], '1A':[], '2A':[], '2M':[]},
        'course': set(),                 # 코스점검자
        'key': Optional[str],            # 열쇠 담당
        'period_names': set(),           # 교시 라인에 등장한 이름
        'must_1m': set(),                # ★ 1종수동 블록에 등장한 이름(필수 근무자 · 헤더 1종수동)
      }
    """
    sections = {
        "1M": re.compile(r"^\s*1\s*종\s*수동\s*:", re.I),
        "1A": re.compile(r"^\s*1\s*종\s*자동\s*:", re.I),
        "2A": re.compile(r"^\s*2\s*종\s*자동\s*:", re.I),
        "2M": re.compile(r"^\s*2\s*종\s*수동\s*:", re.I),
        "COURSE": re.compile(r"코스\s*점검|코스점검", re.I),
        "KEY": re.compile(r"^\s*열쇠\s*:", re.I),
        "PERIOD": re.compile(r"^\s*[1-5]\s*교시\s*:", re.I),
    }
    out = {
        "classes": {"1M":[], "1A":[], "2A":[], "2M":[]},
        "course": set(),
        "key": None,
        "period_names": set(),
        "must_1m": set(),
    }
    cur = None

    def extract_names_loose(text: str):
        # '호' 뒤의 한글 이름 보조 추출(포맷 변형 대비)
        return re.findall(r"호[\s·:\-]*([가-힣]{2,4})", text)

    for raw in (src or "").splitlines():
        line = raw.strip()
        if not line:
            continue

        # 섹션 전환
        if sections["1M"].search(line):   cur = "1M";  continue
        if sections["1A"].search(line):   cur = "1A";  continue
        if sections["2A"].search(line):   cur = "2A";  continue
        if sections["2M"].search(line):   cur = "2M";  continue
        if sections["COURSE"].search(line): cur = "COURSE";  continue
        if sections["KEY"].search(line):
            m = re.search(r":\s*(?P<name>"+NAME+")", line)
            if m: out["key"] = m.group("name")
            cur = None; continue
        if sections["PERIOD"].search(line):
            # 교시 라인에서도 이름 뽑아 스태프 후보에 포함
            for m in PLATE_NAME.finditer(line):
                out["period_names"].add(m.group("name"))
            cur = None; continue

        # 내용 파싱
        if cur in ("1M","1A","2A","2M"):
            hits = [m.group("name") for m in PLATE_NAME.finditer(line)]
            if cur=="1M" and not hits:
                hits = extract_names_loose(line)
            for name in hits:
                if name and name not in out["classes"][cur]:
                    out["classes"][cur].append(name)
                    if cur == "1M":
                        out["must_1m"].add(name)  # ★ 1종수동 블록 출신 → 필수 근무자
        elif cur == "COURSE":
            m = COURSE_LINE.search(line)
            if m:
                out["course"].add(m.group("name"))

    return out

# -----------------------
# 스킬 정규화(규칙)
# -----------------------
def normalize_skills_by_rule(staffs: List[Staff], manual_capable_names: Set[str]) -> None:
    """
    수동 가능(명단 포함) => {1M,2M,1A,2A}
    수동 불가(명단 외)   => {1A,2A}
    """
    for s in staffs:
        if s.name in manual_capable_names:
            s.skills = {"1M","2M","1A","2A"}
        else:
            s.skills = {"1A","2A"}

# -----------------------
# 배정기(오전/오후 다른 근무자 지원 + 1종수동 기본배정 + 혼합 최소화)
#   ※ '수동 전용' 입력/로직 삭제. 수동 수요는 반드시 수동으로만 처리(하드 제약).
# -----------------------
def auto_assign(
    staffs_by_period: Dict[int, List[Staff]],          # 교시별 근무자 리스트
    header_cls_by_staff: Dict[str,str],                # 각 사람의 대표 종(헤더)
    must_1m_by_period: Dict[int, Set[str]],            # ★ 교시별 1종수동 필수 근무자
    period_demands: Dict[int, Dict[str,int]],          # 교시별 수요 { "1M":x,"1A":y,"2A":z,"2M":w }
    options: Options = Options(),
) -> Tuple[Assignment, Dict[str,float], Dict[int,str]]:
    """
    반환:
      - assignments[p][name][cls] = count
      - daily_load[name]          = 코스점검/혼합 포함 가중 총량
      - unavoidable[p]            = 불가피 사유 로그 문자열
    """
    assignments: Assignment = {p:{} for p in range(1,6)}
    # 일일 가중치 초기화(코스점검 +0.8)
    all_names: Set[str] = set(n for lst in staffs_by_period.values() for n in [s.name for s in lst])
    course_names = set(n for lst in staffs_by_period.values() for s in lst if s.morning_course_check for n in [s.name])
    daily_load = {name: (options.course_penalty if name in course_names else 0.0) for name in all_names}
    classes_taken: Dict[str, Set[str]] = {name:set() for name in all_names}
    unavoidable: Dict[int,str] = {}

    # 수요↔스킬 사전 검증 + ★ 수동 수요 용량 체크(절대 자동으로 대체 금지)
    for p in range(1,6):
        sp = staffs_by_period.get(p, [])
        avail = {
            "1M": any(("1M" in s.skills) for s in sp),
            "2M": any(("2M" in s.skills) for s in sp),
            "1A": any(("1A" in s.skills) for s in sp),
            "2A": any(("2A" in s.skills) for s in sp),
        }
        dem = period_demands.get(p, {})
        for c in CLS_ORDER:
            if dem.get(c,0)>0 and not avail[c]:
                raise RuntimeError(f"{p}교시 {ko(c)} 수요가 있으나 해당 교시에 가능한 근무자가 없습니다.")
        # ★ 수동 수요 총량 ≤ 수동 가능 총용량(용량=해당 교시 cap)
        manual_cap = sum(period_capacity(p) for s in sp if ("1M" in s.skills or "2M" in s.skills))
        if dem.get("1M",0) + dem.get("2M",0) > manual_cap:
            raise RuntimeError(f"{p}교시 수동 수요({dem.get('1M',0)+dem.get('2M',0)})가 수동 가능 용량({manual_cap})을 초과합니다.")

    # 혼합 카운터
    mix_daily  = {name: 0 for name in all_names}               # 하루 혼합 사용 수
    mix_period = {(p,s.name):0 for p, lst in staffs_by_period.items() for s in lst}  # 교시별 혼합 수

    # 교시별 배정
    for p in range(1,6):
        staffs = staffs_by_period.get(p, [])
        assignments[p] = {s.name:{c:0 for c in CLS_ORDER} for s in staffs}
        cap_remain = {s.name: period_capacity(p) for s in staffs}
        dem = {c:int(period_demands.get(p,{}).get(c,0)) for c in CLS_ORDER}
        must_1m_names = set(n for n in must_1m_by_period.get(p, set()) if n in cap_remain)

        # 교시 목표치(base/extra) → 소프트 상한 u_s
        T = sum(dem.values())
        N = max(1, len(staffs))
        base, extra = divmod(T, N)
        # extra 후보 선정: (1) 일일가중치 낮음, (2) 해당 교시 희소 스킬(1M/2M) 보유, (3) 이름
        need_manual = (dem["1M"]>0) or (dem["2M"]>0)
        def scarcity_bonus(s: Staff) -> int:
            if not need_manual: return 0
            return 0 if (("1M" in s.skills) or ("2M" in s.skills)) else 1
        extra_targets = sorted(staffs, key=lambda s: (daily_load.get(s.name,0.0), scarcity_bonus(s), s.name))[:extra]
        u_s = {s.name: min(cap_remain[s.name], base + (1 if s in extra_targets else 0)) for s in staffs}

        # 0) 1종수동 "우선예약": must_1m에게 1M 수요를 먼저 소화(소프트 상한 우선, 부족 시 초과 허용)
        need_1m_prefill = dem["1M"]
        if need_1m_prefill > 0 and must_1m_names:
            for _ in range(need_1m_prefill):
                cand = [s for s in staffs if s.name in must_1m_names and "1M" in s.skills and cap_remain[s.name]>0 and sum(assignments[p][s.name].values()) < u_s[s.name]]
                if not cand:
                    cand = [s for s in staffs if s.name in must_1m_names and "1M" in s.skills and cap_remain[s.name]>0]
                if not cand:
                    break
                pick = min(cand, key=lambda s: (daily_load[s.name], sum(assignments[p][s.name].values()), s.name))
                assignments[p][pick.name]["1M"] += 1
                cap_remain[pick.name] -= 1
                if "1M" not in classes_taken[pick.name]:
                    if len(classes_taken[pick.name])>=1:
                        daily_load[pick.name] += options.switch_penalty
                    classes_taken[pick.name].add("1M")
                daily_load[pick.name] += 1.0
                if sum(assignments[p][pick.name].values()) > u_s[pick.name]:
                    unavoidable[p] = (unavoidable.get(p,"") + f" {ko('1M')}_OVER({pick.name})").strip()
                dem["1M"] -= 1
                if dem["1M"] <= 0:
                    break

        # 1) 잔여 수요 배정(1M→2M→1A→2A)
        for c in ("1M","2M","1A","2A"):
            need = dem[c]
            if need>0:
                _assign_units(
                    p, c, need, staffs, cap_remain, assignments, daily_load, classes_taken,
                    u_s, options, unavoidable, header_cls_by_staff, mix_daily, mix_period, must_1m_names
                )

        # 2) 로컬 스왑: must_1m가 비-1M을 들고 있고, 다른 사람이 1M을 들고 있으면 교환 시도
        changed = True
        while changed:
            changed = False
            for a in [s for s in staffs if s.name in must_1m_names]:
                for cls_other in ("2A","1A","2M"):
                    if assignments[p][a.name][cls_other] <= 0:
                        continue
                    for b in staffs:
                        if assignments[p][b.name]["1M"] <= 0:
                            continue
                        if cls_other in b.skills:
                            # 스왑(종 총량 보존: 수동 수
