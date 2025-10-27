# app.py — 도로주행 자동 배정(가로=근무자, 세로=교시, 괄호 표기 규칙)
from __future__ import annotations
import streamlit as st
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional, Union
import pandas as pd

# -----------------------
# 기본 상수/유틸
# -----------------------
CLS_ORDER = ("1M","1A","2A","2M")  # 표기 순서: 1종수동, 1종자동, 2종자동, 2종수동
CLS_KO = {"1M":"1종수동","1A":"1종자동","2A":"2종자동","2M":"2종수동"}

def period_capacity(p:int)->int:  # 1·5교시=2, 2·3·4교시=3
    return 2 if p in (1,5) else 3

def ko(cls:str)->str:
    return CLS_KO.get(cls, cls)

# -----------------------
# 데이터 모델
# -----------------------
@dataclass
class Staff:
    name: str
    skills: Set[str]                 # 가능한 종: {"1M","2M","1A","2A"} 중 일부/전체
    morning_course_check: bool=False # 오전 코스점검 시 시작 가중치 +0.5

@dataclass
class Options:
    switch_penalty: float = 0.5      # 새 종 처음 맡을 때 가중치
    verbose: bool = False

Assignment = Dict[int, Dict[str, Dict[str,int]]]  # [교시][이름][클래스] = 인원수

# -----------------------
# 스킬 정규화(규칙 강제)
# -----------------------
def normalize_skills_by_rule(staffs: List[Staff], manual_capable_names: Set[str]) -> None:
    """
    - 수동 가능(명단 포함)  => {1M,2M,1A,2A}
    - 수동 불가(명단 외)    => {1A,2A}
    """
    for s in staffs:
        if s.name in manual_capable_names:
            s.skills = {"1M","2M","1A","2A"}
        else:
            s.skills = {"1A","2A"}

# -----------------------
# 배정기(교시별 균등 base/extra + 그리디)
# -----------------------
def auto_assign(
    staffs: List[Staff],
    period_demands: Dict[int, Dict[str,int]],
    manual_only_counts: Optional[Dict[int, Dict[str,int]]] = None,  # {p: {"1M":x,"2M":y}}
    options: Options = Options(),
) -> Tuple[Assignment, Dict[str,float], Dict[int,Dict[str,str]]]:
    """
    반환:
      assignments: [p][name][cls] = count
      daily_load:  이름별 일일 가중 처리량(코스점검/혼합 포함)
      unavoidable: [p]["reason"] = 텍스트 (불가피 초과 사유 요약)
    """
    names = [s.name for s in staffs]
    name_to_idx = {s.name:i for i,s in enumerate(staffs)}
    daily_load = {s.name:(0.5 if s.morning_course_check else 0.0) for s in staffs}
    classes_taken: Dict[str, Set[str]] = {s.name:set() for s in staffs}
    assignments: Assignment = {p:{s.name:{c:0 for c in CLS_ORDER} for s in staffs} for p in range(1,6)}
    unavoidable: Dict[int, Dict[str,str]] = {}

    # 수요 존재 ↔ 스킬 보유 체크(사전)
    def require(cls:str):
        return any((cls in s.skills) for s in staffs)
    for p in range(1,6):
        for c in CLS_ORDER:
            if period_demands.get(p,{}).get(c,0)>0 and not require(c):
                raise RuntimeError(f"{p}교시 {ko(c)} 수요가 있으나 가능한 근무자가 없습니다.")

    # 각 교시 처리
    for p in range(1,6):
        dem = {c:int(period_demands.get(p,{}).get(c,0)) for c in CLS_ORDER}
        cap_remain = {s.name: period_capacity(p) for s in staffs}

        # 교시 목표치(base/extra) → 각 근무자 소프트 상한 u_s
        T = sum(dem.values())
        N = len(staffs) if len(staffs)>0 else 1
        base, extra = divmod(T, N)
        # 일일 가중치가 적은 순 + 희소 스킬(1M/2M 가능한 수가 적은 사람) 우선으로 extra 명 선택
        scarcity_score = {}
        need_manual = (dem["1M"]>0) or (dem["2M"]>0)
        manual_counts = sum(1 for s in staffs if (("1M" in s.skills) or ("2M" in s.skills)))
        for s in staffs:
            scarce_bonus = 0 if not need_manual else (0 if (("1M" in s.skills) or ("2M" in s.skills)) else 1)
            scarcity_score[s.name] = (daily_load[s.name], scarce_bonus, s.name)
        extra_targets = [n for n,_ in sorted([(s.name, scarcity_score[s.name]) for s in staffs], key=lambda x:x[1])][:extra]
        u_s = {s.name: min(cap_remain[s.name], base + (1 if s.name in extra_targets else 0)) for s in staffs}

        # 1) 수동전용 선배정
        mo = manual_only_counts.get(p,{}) if manual_only_counts else {}
        for c in ("1M","2M"):
            need = int(mo.get(c,0))
            if need>dem[c]:
                raise RuntimeError(f"{p}교시 수동전용 {ko(c)} {need} > 수요 {dem[c]}")
            _assign_units(p, c, need, staffs, cap_remain, assignments, daily_load, classes_taken, u_s, options, unavoidable)

            dem[c] -= need

        # 2) 잔여 수요 배정(우선순위: 1M→2M→1A→2A)
        for c in ("1M","2M","1A","2A"):
            need = dem[c]
            if need>0:
                _assign_units(p, c, need, staffs, cap_remain, assignments, daily_load, classes_taken, u_s, options, unavoidable)

    return assignments, daily_load, unavoidable

def _assign_units(
    p:int, cls:str, units:int,
    staffs: List[Staff],
    cap_remain: Dict[str,int],
    assignments: Assignment,
    daily_load: Dict[str,float],
    classes_taken: Dict[str, Set[str]],
    u_s: Dict[str,int],
    options: Options,
    unavoidable: Dict[int,Dict[str,str]]
):
    if units<=0: return
    eligible = [s for s in staffs if cls in s.skills]
    if not eligible:
        unavoidable.setdefault(p,{}).setdefault("reason","")
        unavoidable[p]["reason"] += f" {ko(cls)}_NO_SKILL"
        raise RuntimeError(f"{p}교시 {ko(cls)} 가능한 근무자 없음")

    for _ in range(units):
        # 1순위: cap>0 이고 (현재교시 배정 < u_s)
        cand1 = []
        cand2 = []  # 불가피 초과(현재교시 배정 >= u_s) 허용 후보
        for s in eligible:
            if cap_remain[s.name] <= 0: 
                continue
            cur_in_period = sum(assignments[p][s.name].values())
            if cur_in_period < u_s[s.name]:
                cand1.append(s)
            else:
                cand2.append(s)

        cand = cand1 if cand1 else cand2
        if not cand:
            # 완전 불가: 용량 부족
            unavoidable.setdefault(p,{}).setdefault("reason","")
            unavoidable[p]["reason"] += f" {ko(cls)}_CAP_SHORT"
            raise RuntimeError(f"{p}교시 {ko(cls)} 수요 처리 불가(잔여용량 없음)")

        # 스코어: 일일가중치 + (해당 종 처음이면 +0.5) , 그 교시 현재배정(적을수록), 이름
        def score(s:Staff):
            cur_in_period = sum(assignments[p][s.name].values())
            pseudo = daily_load[s.name] + (options.switch_penalty if ((cls not in classes_taken[s.name]) and len(classes_taken[s.name])>=1) else 0.0)
            return (pseudo, cur_in_period, s.name)

        pick = min(cand, key=score)

        # 실제 배정 반영
        assignments[p][pick.name][cls] += 1
        cap_remain[pick.name] -= 1

        # 새 종 최초 배정시 혼합 패널티
        if cls not in classes_taken[pick.name]:
            if len(classes_taken[pick.name])>=1:
                daily_load[pick.name] += options.switch_penalty
            classes_taken[pick.name].add(cls)

        daily_load[pick.name] += 1.0

        if cand is cand2:  # 소프트 상한 초과 허용됨
            unavoidable.setdefault(p,{}).setdefault("reason","")
            tag = f"{ko(cls)}_OVER"
            unavoidable[p]["reason"] = (unavoidable[p]["reason"]+" "+tag).strip()

# -----------------------
# 표 렌더링 (MultiIndex: 상단 헤더=대표종 / 2행=이름)
# -----------------------
def render_sheet(assignments: Assignment,
                 staff_order: List[str],
                 header_cls_by_staff: Dict[str,str],
                 title: str,
                 periods: List[int]):

    # MultiIndex 컬럼 생성
    cols = pd.MultiIndex.from_tuples([(ko(header_cls_by_staff.get(s,"")), s) for s in staff_order])

    # 누적합 관리
    cum = {s:0 for s in staff_order}
    rows = []
    idx = []

    def fmt_cell(p:int, s:str)->str:
        cell = assignments[p][s]
        total = sum(cell.values())
        if total==0:
            return ""
        # 비어있지 않은 종만 정렬
        used = [(ko(c), cell[c]) for c in CLS_ORDER if cell.get(c,0)]
        hdr = header_cls_by_staff.get(s,"")
        hdr_ko = ko(hdr) if hdr else ""
        # 괄호 규칙: 헤더와 완전 일치(단일종)면 숫자만, 그 외(다름/혼합)만 괄호 표기
        if len(used)==1 and used[0][0]==hdr_ko:
            return f"{total}"
        inside = ", ".join(f"{c}{n}" for c,n in used)
        return f"{total} ({inside})"

    for p in periods:
        # 교시 행
        row = [fmt_cell(p,s) for s in staff_order]
        rows.append(row)
        idx.append(f"{p}교시")
        # 누적 합계 업데이트 후 합계 행
        for s in staff_order:
            cum[s] += sum(assignments[p][s].values())
        rows.append([str(cum[s]) if cum[s] else "" for s in staff_order])
        idx.append("합계")

    df = pd.DataFrame(rows, index=idx, columns=cols)

    st.markdown(f"### {title}")
    st.dataframe(df, use_container_width=True)

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="도로주행 자동 배정", layout="wide")
st.title("도로주행 자동 배정")

# --- 입력: 근무자 / 수동가능 명단 / 코스점검 ---
with st.sidebar:
    st.subheader("입력")
    today_staff = st.text_area("오늘 근무자 (쉼표/줄바꿈)", value="김성연, 김병욱, 김지은, 김면정")
    staff_list = [s.strip() for s in today_staff.replace("\n",",").split(",") if s.strip()]
    manual_capable_default = ["권한솔","김남균","김성연","김주현","이호석","조정래"]  # 저장된 기본값
    manual_input = st.text_area("수동 가능(1종수동 명단)", value=", ".join(manual_capable_default))
    manual_set = set([s.strip() for s in manual_input.replace("\n",",").split(",") if s.strip()])

    course_morning = st.multiselect("오전 코스점검자(가중치 +0.5)", staff_list, default=[])

    st.markdown("---")
    st.markdown("**교시별 수요 입력**")
    period_demands: Dict[int,Dict[str,int]] = {}
    for p in [1,2,3,4,5]:
        c0, c1, c2, c3, c4 = st.columns([1,1,1,1,1])
        with c0: st.write(f"**{p}교시**")
        with c1: v1m = st.number_input(f"1종수동", min_value=0, step=1, key=f"d_{p}_1M")
        with c2: v1a = st.number_input(f"1종자동", min_value=0, step=1, key=f"d_{p}_1A")
        with c3: v2a = st.number_input(f"2종자동", min_value=0, step=1, key=f"d_{p}_2A")
        with c4: v2m = st.number_input(f"2종수동", min_value=0, step=1, key=f"d_{p}_2M")
        period_demands[p] = {"1M":v1m,"1A":v1a,"2A":v2a,"2M":v2m}

    st.markdown("**(선택) 수동전용 응시생 수**")
    manual_only: Dict[int,Dict[str,int]] = {}
    for p in [1,2,3,4,5]:
        c0, c1, c2 = st.columns([1,1,1])
        with c0: st.write(f"{p}교시")
        with c1: mo1 = st.number_input("1종수동 전용", min_value=0, step=1, key=f"mo_{p}_1M")
        with c2: mo2 = st.number_input("2종수동 전용", min_value=0, step=1, key=f"mo_{p}_2M")
        manual_only[p] = {"1M":mo1,"2M":mo2}

# --- 오늘 근무자 객체 생성 + 스킬 정규화 ---
staff_objs: List[Staff] = []
for nm in staff_list:
    staff_objs.append(Staff(nm, skills=set(), morning_course_check=(nm in course_morning)))
normalize_skills_by_rule(staff_objs, manual_set)

# --- 상단 헤더용 대표 종 선택(열마다 붙일 라벨) ---
st.subheader("상단 헤더용 대표 종 지정")
header_cls_by_staff: Dict[str,str] = {}
hdr_cols = st.columns(len(staff_list) if staff_list else 1)
for i, nm in enumerate(staff_list):
    with hdr_cols[i]:
        default_cls = "1M" if nm in manual_set else "2A"
        sel = st.selectbox(f"{nm}", [("1M","1종수동"),("1A","1종자동"),("2A","2종자동"),("2M","2종수동")],
                           index=[k for k,_ in [("1M","1종수동"),("1A","1종자동"),("2A","2종자동"),("2M","2종수동")]].index(default_cls),
                           format_func=lambda kv: kv[1], key=f"hdr_{nm}")
        header_cls_by_staff[nm] = sel[0]

st.markdown("---")
run = st.button("배정 실행", type="primary")

# --- 실행/출력 ---
if run:
    try:
        assignments, loads, unavoidable = auto_assign(
            staff_objs, period_demands, manual_only_counts=manual_only, options=Options()
        )

        # 오전/오후 표 출력 (괄호는 헤더와 다르거나 혼합일 때만)
        if staff_list:
            render_sheet(assignments, staff_list, header_cls_by_staff, "오전 (1~2교시)", [1,2])
            render_sheet(assignments, staff_list, header_cls_by_staff, "오후 (3~5교시)", [3,4,5])

            # 전체 총합 행 (하단 요약)
            totals = {s: sum(sum(assignments[p][s].values()) for p in range(1,6)) for s in staff_list}
            totals_detail = {s: {c: sum(assignments[p][s][c] for p in range(1,6)) for c in CLS_ORDER} for s in staff_list}
            st.markdown("### 총합(일일)")
            df_total = pd.DataFrame([{
                "근무자": s,
                "총합": totals[s],
                "상세": ", ".join(f"{ko(c)}{totals_detail[s][c]}" for c in CLS_ORDER if totals_detail[s][c])
            } for s in staff_list])
            st.dataframe(df_total, use_container_width=True)

            # 불가피 사유 로그
            if unavoidable:
                st.markdown("#### 참고: 불가피 초과/제약 로그")
                for p, info in unavoidable.items():
                    st.caption(f"{p}교시 → {info.get('reason','')}")
        else:
            st.warning("근무자를 입력하세요.")

    except Exception as e:
        st.error(f"오류: {e}")
