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
                            # 스왑(종 총량 보존: 수동 수요는 계속 수동으로 남음)
                            assignments[p][a.name][cls_other] -= 1
                            assignments[p][a.name]["1M"]     += 1
                            assignments[p][b.name]["1M"]     -= 1
                            assignments[p][b.name][cls_other]+= 1
                            changed = True
                            break
                    if changed: break
                if changed: break

    return assignments, daily_load, unavoidable

def _assign_units(
    p:int, cls:str, units:int,
    staffs: List[Staff],
    cap_remain: Dict[str,int],
    assignments: Assignment,
    daily_load: Dict[str,float],
    classes_taken: Dict[str, Set[str]},
    u_s: Dict[str,int],
    options: Options,
    unavoidable: Dict[int,str],
    header_cls_by_staff: Dict[str,str],
    mix_daily: Dict[str,int],
    mix_period: Dict[Tuple[int,str],int],
    must_1m_names: Set[str],
):
    if units<=0:
        return
    eligible = [s for s in staffs if cls in s.skills]
    if not eligible:
        unavoidable[p] = (unavoidable.get(p,"") + f" {ko(cls)}_NO_SKILL").strip()
        raise RuntimeError(f"{p}교시 {ko(cls)} 가능한 근무자 없음")

    def can_mix(name: str) -> bool:
        return (mix_period[(p,name)] < options.mix_period_cap) and (mix_daily[name] < options.mix_daily_cap)

    for _ in range(units):
        # 후보 수집
        buckets = {  # (prio, soft/over, list)
            (0,"soft"):[], (0,"over"):[],
            (1,"soft"):[], (1,"over"):[],
            (2,"soft"):[], (2,"over"):[],
        }
        mix_candidates: List[Staff] = []  # 혼합 잠재 후보(최후 완화용)

        for s in eligible:
            if cap_remain[s.name] <= 0: 
                continue
            cur_in_period = sum(assignments[p][s.name].values())
            header = header_cls_by_staff.get(s.name, "2A")
            is_header = (header == cls)
            is_must = (s.name in must_1m_names)
            is_mix = not is_header

            # 혼합은 기본 보류(최후 수단)
            if is_mix:
                if can_mix(s.name):
                    mix_candidates.append(s)
                continue

            # 우선순위: 1M일 때 must 우선, 그 외에는 must 후순위
            if cls == "1M":
                prio = 0 if is_must else 1
            else:
                prio = 0 if (not is_must) else 2

            bucket = "soft" if cur_in_period < u_s[s.name] else "over"
            buckets[(prio, bucket)].append(s)

        # 선택 순서: 헤더 일치 내에서 soft→over, 그다음(필요 시) 혼합 완화
        chosen_group = (buckets[(0,"soft")] or buckets[(0,"over")] or
                        buckets[(1,"soft")] or buckets[(1,"over")] or
                        buckets[(2,"soft")] or buckets[(2,"over")])
        relaxed_mix = False
        if not chosen_group:
            # 정말로 헤더 일치로 못 채우면 그때만 혼합 1건 완화(수동 수요는 수동 종만 배정이므로 여기선 1A/2A일 때만 의미)
            relaxed_mix = True
            cand_relaxed = []
            for s in mix_candidates:
                if cap_remain[s.name] <= 0:
                    continue
                if mix_period[(p,s.name)] < 1:
                    cand_relaxed.append(s)
            chosen_group = cand_relaxed

        if not chosen_group:
            unavoidable[p] = (unavoidable.get(p,"") + f" {ko(cls)}_CAP_SHORT").strip()
            raise RuntimeError(f"{p}교시 {ko(cls)} 수요 처리 불가(잔여용량/혼합제약)")

        # 점수: 일일가중치 + (새 종이면 +0.8), 교시 현재배정, 이름
        def score(s: Staff):
            cur_in_period = sum(assignments[p][s.name].values())
            pseudo = daily_load[s.name] + (options.switch_penalty if ((cls not in classes_taken[s.name]) and len(classes_taken[s.name])>=1) else 0.0)
            return (pseudo, cur_in_period, s.name)

        pick = min(chosen_group, key=score)

        # 실제 배정 반영
        assignments[p][pick.name][cls] += 1
        cap_remain[pick.name] -= 1

        # 새 종 최초 배정시 혼합 패널티
        if cls not in classes_taken[pick.name]:
            if len(classes_taken[pick.name])>=1:
                daily_load[pick.name] += options.switch_penalty
            classes_taken[pick.name].add(cls)
        daily_load[pick.name] += 1.0

        # 혼합 카운팅
        header = header_cls_by_staff.get(pick.name, "2A")
        is_mix = (cls != header)
        if is_mix:
            mix_period[(p,pick.name)] += 1
            mix_daily[pick.name] += 1
            if relaxed_mix:
                unavoidable[p] = (unavoidable.get(p,"") + f" MIX_RELAX({pick.name})").strip()

        # 소프트 상한 초과 사용 기록
        cur_in_period = sum(assignments[p][pick.name].values())
        if cur_in_period > u_s[pick.name]:
            unavoidable[p] = (unavoidable.get(p,"") + f" {ko(cls)}_OVER({pick.name})").strip()

# -----------------------
# 표 렌더(가로=근무자, 세로=교시 · 상단 2행=대표종/이름 · 각 교시 밑에 합계)
# 괄호는 헤더와 다르거나 혼합일 때만 표기
# -----------------------
def render_sheet(assignments: Assignment,
                 staff_order: List[str],
                 header_cls_by_staff: Dict[str,str],
                 title: str,
                 periods: List[int]):

    if not staff_order:
        st.warning(f"{title}: 근무자 없음")
        return

    # MultiIndex 컬럼: (대표종 한글, 이름)
    cols = pd.MultiIndex.from_tuples([(ko(header_cls_by_staff.get(s,"")), s) for s in staff_order])
    cum = {s:0 for s in staff_order}
    rows = []
    idx  = []

    def fmt_cell(p:int, s:str)->str:
        cell = assignments[p].get(s, {})
        total = sum(cell.values()) if cell else 0
        if total==0:
            return ""
        used = [(ko(c), cell[c]) for c in CLS_ORDER if cell.get(c,0)]
        hdr = header_cls_by_staff.get(s,"")
        hdr_ko = ko(hdr) if hdr else ""
        # 헤더와 단일종이 동일하면 괄호 생략
        if len(used)==1 and used[0][0]==hdr_ko:
            return f"{total}"
        return f"{total} ({', '.join(f'{c}{n}' for c,n in used)})"

    st.markdown(f"### {title}")
    for p in periods:
        # 교시 행
        row = [fmt_cell(p,s) for s in staff_order]
        rows.append(row); idx.append(f"{p}교시")
        # 합계(누적) 행 — 'p교시 합계'
        for s in staff_order:
            cell = assignments[p].get(s, {})
            cum[s] += sum(cell.values()) if cell else 0
        rows.append([str(cum[s]) if cum[s] else "" for s in staff_order]); idx.append(f"{p}교시 합계")

    df = pd.DataFrame(rows, index=idx, columns=cols)
    st.dataframe(df, use_container_width=True)

# -----------------------
# UI
# -----------------------
st.set_page_config(page_title="도로주행 자동 배정(오전/오후)", layout="wide")
st.title("도로주행 자동 배정 (오전/오후 분리 · 1종수동 기본배정 · 혼합 최소화)")

with st.sidebar:
    st.subheader("입력 · 붙여넣기")
    tab_am, tab_pm = st.tabs(["오전 텍스트", "오후 텍스트"])
    with tab_am:
        am_text = st.text_area("오전 텍스트 그대로 붙여넣기", height=240,
            placeholder="예) 25.10.27(월) 오전 교양순서 및 차량배정\n\n열쇠: 김면정\n\n1교시: 김면정\n2교시: 김성연\n\n1종수동: 8호 이호석\n\n1종자동: 21호\n\n2종자동:\n • 22호 조정래\n • 17호 김성연\n • 12호 안유미\n • 14호 김면정\n • 19호 김주현\n\n 코스점검 :\n • A코스 합격: 이호석\n • B코스 불합격: 조정래")
    with tab_pm:
        pm_text = st.text_area("오후 텍스트 그대로 붙여넣기", height=240, placeholder="오후 텍스트(있으면)")

    st.markdown("---")
    st.markdown("**수요 입력(교시 옆에 종별 숫자)**")
    period_demands: Dict[int, Dict[str,int]] = {}
    for p in [1,2,3,4,5]:
        c0,c1,c2,c3,c4 = st.columns([1,1,1,1,1])
        with c0: st.write(f"**{p}교시**")
        with c1: v1m = st.number_input("1종수동", min_value=0, step=1, key=f"d_{p}_1M")
        with c2: v1a = st.number_input("1종자동", min_value=0, step=1, key=f"d_{p}_1A")
        with c3: v2a = st.number_input("2종자동", min_value=0, step=1, key=f"d_{p}_2A")
        with c4: v2m = st.number_input("2종수동", min_value=0, step=1, key=f"d_{p}_2M")
        period_demands[p] = {"1M":v1m, "1A":v1a, "2A":v2a, "2M":v2m}

# 붙여넣기 파싱 → 오전/오후 근무자·코스점검자·must_1m
am = parse_staff_text(am_text or "")
pm = parse_staff_text(pm_text or "")
am_staff_names = sorted(
    set(sum([am["classes"][c] for c in CLS_ORDER], []))
    | set(am.get("period_names", set()))
    | ({am.get("key")} if am.get("key") else set())
    | set(am.get("must_1m", set()))
)
pm_staff_names = sorted(
    set(sum([pm["classes"][c] for c in CLS_ORDER], []))
    | set(pm.get("period_names", set()))
    | ({pm.get("key")} if pm.get("key") else set())
    | set(pm.get("must_1m", set()))
)

# 수동 가능 명단(기본: 저장해둔 팀 + 텍스트의 수동블록 인원)
manual_seed = {"권한솔","김남균","김성연","김주현","이호석","조정래"}
manual_set = set(manual_seed)
manual_set |= set(am["classes"]["1M"]) | set(am["classes"]["2M"])
manual_set |= set(pm["classes"]["1M"]) | set(pm["classes"]["2M"])

# 헤더(대표 종) 자동: ★ 1종수동 블록 출신은 1M 고정, 그 외는 (수동가능? 1M : 2A)
header_cls_by_staff: Dict[str,str] = {}
for nm in set(am_staff_names) | set(pm_staff_names):
    if (nm in am.get("must_1m", set())) or (nm in pm.get("must_1m", set())):
        header_cls_by_staff[nm] = "1M"
    else:
        header_cls_by_staff[nm] = "1M" if nm in manual_set else "2A"

# Staff 객체 구성(오전/오후 분리, 코스점검 반영)
def build_staffs(names: List[str], course_names: Set[str]) -> List[Staff]:
    out = []
    for nm in names:
        out.append(Staff(nm, skills=set(), morning_course_check=(nm in course_names)))
    normalize_skills_by_rule(out, manual_set)
    return out

staffs_by_period: Dict[int, List[Staff]] = {}
am_staffs = build_staffs(am_staff_names, am["course"])
pm_staffs = build_staffs(pm_staff_names, pm["course"])
for p in AM_PERIODS: staffs_by_period[p] = am_staffs
for p in PM_PERIODS: staffs_by_period[p] = pm_staffs

# 교시별 must_1m 집합
must_1m_by_period = {1:set(am.get("must_1m", set())), 2:set(am.get("must_1m", set())),
                     3:set(pm.get("must_1m", set())), 4:set(pm.get("must_1m", set())), 5:set(pm.get("must_1m", set()))}

# 진단 라벨(인식된 근무자/헤더) — 문제 빠르게 캐치용
if am_staff_names:
    st.caption("오전 근무자: " + ", ".join(am_staff_names))
if pm_staff_names:
    st.caption("오후 근무자: " + ", ".join(pm_staff_names))
if header_cls_by_staff:
    st.caption("헤더 종: " + ", ".join(f"{n}={ko(c)}" for n,c in header_cls_by_staff.items()))

# 실행 버튼
st.markdown("---")
run = st.button("배정 실행", type="primary")

if run:
    try:
        assignments, loads, unavoidable = auto_assign(
            staffs_by_period=staffs_by_period,
            header_cls_by_staff=header_cls_by_staff,
            must_1m_by_period=must_1m_by_period,
            period_demands=period_demands,
            options=Options(
                switch_penalty=SWITCH_PENALTY,
                course_penalty=COURSE_PENALTY,
                mix_daily_cap=MIX_DAILY_CAP,
                mix_period_cap=MIX_PERIOD_CAP
            )
        )

        # 오전/오후 렌더
        if am_staff_names:
            render_sheet(assignments, am_staff_names, header_cls_by_staff, "오전 (1~2교시)", AM_PERIODS)
        else:
            st.info("오전: 근무자 없음(텍스트 입력 확인)")

        if pm_staff_names:
            render_sheet(assignments, pm_staff_names, header_cls_by_staff, "오후 (3~5교시)", PM_PERIODS)
        else:
            st.info("오후: 근무자 없음(텍스트 입력 확인)")

        # 총합(일일) — 근무자 가로행(상세 없음)
        totals = {}
        for nm in sorted(set(am_staff_names) | set(pm_staff_names)):
            totals[nm] = sum(sum(assignments[p].get(nm,{}).values()) for p in range(1,6))
        if totals:
            st.markdown("### 총합(일일 · 근무자 가로행)")
            df_total = pd.DataFrame([totals], index=["총합"])
            ordered_cols = am_staff_names + [n for n in pm_staff_names if n not in am_staff_names]
            st.dataframe(df_total[ordered_cols], use_container_width=True)

        # 불가피 로그
        if unavoidable:
            st.markdown("#### 참고: 불가피 로그")
            for p in range(1,6):
                if p in unavoidable and unavoidable[p]:
                    st.caption(f"{p}교시 → {unavoidable[p]}")

    except Exception as e:
        st.error(f"오류: {e}")
