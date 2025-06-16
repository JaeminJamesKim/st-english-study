import pandas as pd
import numpy as np
import requests
import os
import time
from bs4 import BeautifulSoup as bs
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from dotenv import load_dotenv
import base64, ast, mimetypes
from openai import OpenAI
import json
from collections import OrderedDict
import warnings

warnings.filterwarnings("ignore")
load_dotenv()
client = OpenAI()                     # 환경변수 OPENAI_API_KEY 사용


def get_current_all_members():
    '''
    ## 1. 현재 총 멤버 업데이트 (from Somoim)
    '''
    page_url = 'https://www.somoim.co.kr/95381bac-d344-11ec-ba98-0a4d683471cd1'

    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    driver = webdriver.Chrome(options=options)

    # 2) Open the page
    driver.get(page_url)
    time.sleep(2)

    # ————— click your button —————
    wait = WebDriverWait(driver, 10)
    btn = wait.until(EC.presence_of_element_located((
        By.XPATH, "//button[contains(., '모임 멤버 더보기')]"
    )))

    driver.execute_script("""
        const rect = arguments[0].getBoundingClientRect();
        window.scrollBy(0, rect.top - 200);
    """, btn)

    btn.click()
    # ————— wait for the target elements to appear —————
    # WebDriverWait(driver, 10)

    # ————— grab & filter them —————
    elems = driver.find_elements(
        By.CSS_SELECTOR,
        "[class*='leading-tight'][class*='min-h-[20px]']"
        "[class*='text-sm'][class*='font-semibold'][class*='text-fc_black']"
    )

    member_name_lst = []
    for el in elems:
        txt = el.text.strip()
        member_name_lst.append(txt)

    board_elems = driver.find_elements(
        By.CSS_SELECTOR,
        "[class*='text-[15px]'][class*='font-bold'][class*='text-fc_black']"
        "[class*='text-center'][class*='line-clamp-1']"
    )

    board_member_lst = []
    for el in board_elems:
        txt = el.text.strip()
        board_member_lst.append(txt)

    return board_member_lst, member_name_lst

# 성별을 수동으로 업데이트 해줘야함: Default 'M'으로 설정
def compare_and_update_member_list_excel(curr_member_name_lst):
    member_list = pd.read_excel('member_list.xlsx')
    new_members = [n for n in curr_member_name_lst if n not in set(member_list["names"])]
    if len(new_members) == 0:
        new_members = ['없음','']
    for member_name in curr_member_name_lst:
        if member_name not in member_list['name']:
            member_list.loc[len(member_list)] = [member_name, 'M', 0, '하', False]
    member_list = member_list[member_list["name"].isin(curr_member_name_lst)].reset_index(drop=True)
    member_list.to_excel('member_list.xlsx')
    
    print('>> [UPDATE COMPLETE] member_list.xlsx')
    return new_members

def _as_data_url(path: str) -> str:
    """파일을 data:<mime>;base64,<b64> 형태로 변환"""
    mime, _ = mimetypes.guess_type(path)
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:{mime or 'image/jpeg'};base64,{b64}"

def extract_member_names(*image_paths):
    # 1) system/user 프롬프트 구성
    system = "You are an OCR‐post-processing assistant. Only output a Python list of boldface names found to the right of each avatar, e.g.: [홍길동, Jane, 석종훈, 안디, 테리, A]. Do not add line breaks or quotes."
    content = [
        {
            "type":"text",
            "text":(
                "아래 이미지들에 적힌 유저 이름만 파이썬 리스트 형태로 반환해줘.\n"
                "유저의 이름은 굵은 글씨로 되어있으니까, 모든 굵은 글씨들을 반환해줘.\n"
                "반드시 다음 예시 형식만 출력: [\"석종훈\", \"박제용\", \"A.T\", \"James\"]\n"
                "리스트 외의 설명·줄바꿈·불필요 문자 모두 금지."
            )
        }
    ]
    for p in image_paths:
        content.append({"type":"image_url", "image_url":{"url":_as_data_url(p)}})

    # 2) OpenAI API 호출
    resp = client.chat.completions.create(
        # model="gpt-4o-vision-preview",   # 또는 gpt-4o-mini 등 vision 지원 모델
        # model="gpt-4o-mini",   # 또는 gpt-4o-mini 등 vision 지원 모델
        model="gpt-4o",   # 또는 gpt-4o-mini 등 vision 지원 모델
        messages=[
            {"role":"system", "content":system},
            {"role":"user", "content":content}
            ],
        # functions=[{
        #     "name":"collect_names",
        #     "parameters":{"type":"array","items":{"type":"string"}}
        # }],
        # function_call="auto",
        temperature=0,
        top_p=0.1,
        n=3,
        max_tokens=512,
    )
    
    # best = max(resp.choices, key=lambda c: c.finish_reason=="stop")
    # print(best)
    # names = json.loads(best.message.function_call.arguments)
    # 3) 파싱 & 후처리
    raw = resp.choices[0].message.content.strip()
    # names = ast.literal_eval(best)
    names = ast.literal_eval(raw)        # 문자열 → 실제 list
    names = list(dict.fromkeys(names))   # (선택) 중복 제거 & 순서 유지
    return names




import math
import random
from collections import Counter, defaultdict
import pandas as pd
import pulp


# -------------------- 이번 주 인원 리스트 불러오기 --------------------


# -------------------- 이번 주 인원 출석 데이터에 병합하기 --------------------
def update_attendance(TODAY, curr_participants):
    member_list = pd.read_excel('member_list.xlsx')
    curr_participant_info_list = member_list.loc[(member_list['name']).isin(curr_participants)]
    curr_participant_info_list["attend_date"] = TODAY
    curr_participant_info_list["attend_date"] = pd.to_datetime(curr_participant_info_list["attend_date"]).dt.date
    attendance = pd.read_excel('attendance.xlsx')
    attendance["attend_date"] = pd.to_datetime(attendance["attend_date"]).dt.date

    curr_participant_info_list['is_new'] = np.where(
        curr_participant_info_list['total_attend'] == 0, True, False) 
    
    curr_participant_info_list['prev_group_code1'] = None
    curr_participant_info_list['prev_group_code2'] = None
    curr_participant_info_list['prev_group_code3'] = None
    for i in curr_participant_info_list['name']:
        try:
            curr_participant_info_list['prev_group_code1'].loc[curr_participant_info_list['name']==i] = attendance['group_code'].loc[attendance['name']==i][-3:].values[-1]
            curr_participant_info_list['prev_group_code2'].loc[curr_participant_info_list['name']==i] = attendance['group_code'].loc[attendance['name']==i][-3:].values[-2]
            curr_participant_info_list['prev_group_code3'].loc[curr_participant_info_list['name']==i] = attendance['group_code'].loc[attendance['name']==i][-3:].values[-3]
        except:
            pass

    new_attendance = pd.concat([attendance, curr_participant_info_list])
    return new_attendance


# -------------------- 출석 데이터 불러오기 --------------------
def load_today(df, today: str) -> pd.DataFrame:
    """
    today      : 'YYYY-MM-DD' 형식
    반환값     : 오늘 날짜의 참석자 데이터프레임
    """
    today = pd.to_datetime(today).date()
    return df[df["attend_date"] == today].reset_index(drop=True)


# -------------------- 최적화 모델 생성 --------------------
# def choose_group_count(n):
#     """
#     n명 참석 시 4·3인 그룹 배치를 위한 최소 그룹 수와
#     허용되는 3인 그룹 개수 k (k ≤ 2) 를 반환.
#     """
#     g = math.ceil(n / 4)          # 우선 4명 기준 최소 그룹 수
#     while True:
#         three_groups = 4 * g - n  # 3명 그룹 개수 (4명 그룹 g−three_groups)
#         if 0 <= three_groups <= 2:
#             print(f'{g=}')
#             print(f'{three_groups=}')
#             return g, three_groups
#         g += 1                    # 그룹 수를 늘려서 three_groups 감소

# def build_groups(df):
#     """
#     df columns:
#         name, gender(M/F), is_admin(bool), is_new(bool),
#         english(상/중/하), prev_code1, prev_code2
#     returns OrderedDict group → [names …]  (group_1, group_2, … 순)
#     """
#     # 가중치 설정
#     W_F   = 1_000_000    # f-조건 위반 페널티 (거의 불가)
#     W_SZ  =    10_000     # 그룹 인원 dev (4명 우선)  ← 성비보다 큼
#     W_C   =     1_000     # 성비 dev
#     W_ENG =       500     # 영어 dev
#     W_ADM =       300     # 신규-운영진 dev
#     W_SLV =        50     # 리더 부족 slack

#     n = len(df)

#     # ====== 그룹 수 결정 (4명 그룹 위주, 3명 그룹 ≤2) ======
#     G_total, max_three = choose_group_count(n)

#     # 리더 수가 너무 적으면 그룹 줄이기
#     leaders = (df["is_admin"] | (df["english"] == "상")).sum()
#     if leaders and leaders < G_total:
#         G_total = leaders                # 최소 '리더 수' 만큼만 그룹
#     G, I = range(G_total), range(n)

#     # ====== 특성 ======
#     is_admin = df["is_admin"].astype(int).tolist()
#     is_leader = (df["is_admin"] | (df["english"] == "상")).astype(int).tolist()
#     is_new   = df["is_new"].astype(int).tolist()
#     male     = (df["gender"] == "M").astype(int).tolist()
#     eng_lvl  = df["english"].map({"상":2,"중":1,"하":0}).tolist()
#     prev1 = df.get("prev_group_code1", "")
#     prev2 = df.get("prev_group_code2", "")
#     prev3 = df.get("prev_group_code3", "")

#     # ====== 변수 ======
#     x       = pulp.LpVariable.dicts("x", (I, G), 0, 1, "Binary")
#     dev_sz  = pulp.LpVariable.dicts("dev_sz",  G, 0, 1)   # 0(4명) or 1(3명)
#     dev_gen = pulp.LpVariable.dicts("dev_gen", G, 0)
#     dev_eng = pulp.LpVariable.dicts("dev_eng", G, 0)
#     dev_adm = pulp.LpVariable.dicts("dev_adm", G, 0)
#     no_lead = pulp.LpVariable.dicts("no_lead", G, 0, 1, "Binary")

#     prob = pulp.LpProblem("GroupAssign", pulp.LpMinimize)

#     # ① 각 사람은 정확히 한 그룹
#     for i in I:
#         prob += pulp.lpSum(x[i][g] for g in G) == 1

#     # ② 그룹 인원 3 또는 4  => dev_sz = 4 - size
#     for g in G:
#         size = pulp.lpSum(x[i][g] for i in I)
#         prob += size >= 3
#         prob += size <= 4
#         prob += dev_sz[g] == 4 - size     # size 4→0 , size 3→1

#     # ③ “3인 그룹은 ≤ 2개”  => Σ dev_sz ≤ 2   (dev_sz는 0/1)
#     prob += pulp.lpSum(dev_sz[g] for g in G) <= max_three

#     # ④ 리더 ≥1  (없으면 no_lead slack)
#     for g in G:
#         prob += pulp.lpSum(is_leader[i]*x[i][g] for i in I) + no_lead[g] >= 1

#     # ⑤ 신규 포함 그룹엔 운영진 ≥1  (1/N trick)
#     for g in G:
#         new_cnt = pulp.lpSum(is_new[i]*x[i][g]   for i in I)
#         adm_cnt = pulp.lpSum(is_admin[i]*x[i][g] for i in I)
#         prob += adm_cnt >= new_cnt * (1/n)

#     # ⑥ f-조건 (최근 3회 그룹 동일 → 오늘 함께 금지)
#     f_pairs = []
#     def same_recent(i, j):
#         return (
#             (bool(prev1.iloc[i]) and prev1.iloc[i] == prev1.iloc[j]) or
#             (bool(prev2.iloc[i]) and prev2.iloc[i] == prev2.iloc[j]) or
#             (bool(prev3.iloc[i]) and prev3.iloc[i] == prev3.iloc[j])
#         )
#     for i in I:
#         for j in range(i+1, n):
#             if same_recent(i,j):
#                 for g in G:
#                     p = pulp.LpVariable(f"pair_{i}_{j}_{g}", 0, 1, "Binary")
#                     prob += p <= x[i][g]; prob += p <= x[j][g]
#                     prob += p >= x[i][g] + x[j][g] - 1
#                     f_pairs.append(p)

#     # ⑦ dev(성비·영어·운영진)
#     avg_male = sum(male)/G_total
#     avg_eng  = sum(eng_lvl)/G_total
#     avg_adm  = sum(is_admin)/G_total
#     for g in G:
#         m_cnt = pulp.lpSum(male[i]*x[i][g]  for i in I)
#         e_sum = pulp.lpSum(eng_lvl[i]*x[i][g]  for i in I)
#         a_cnt = pulp.lpSum(is_admin[i]*x[i][g] for i in I)
#         prob += dev_gen[g] >=  m_cnt - avg_male
#         prob += dev_gen[g] >= -m_cnt + avg_male
#         prob += dev_eng[g] >=  e_sum - avg_eng
#         prob += dev_eng[g] >= -e_sum + avg_eng
#         prob += dev_adm[g] >=  a_cnt - avg_adm
#         prob += dev_adm[g] >= -a_cnt + avg_adm

#     # ====== 목적함수 ======
#     prob += (
#           W_F   * pulp.lpSum(f_pairs)
#         + W_SZ  * pulp.lpSum(dev_sz[g]    for g in G)
#         + W_C   * pulp.lpSum(dev_gen[g]  for g in G)
#         + W_ENG * pulp.lpSum(dev_eng[g]  for g in G)
#         + W_ADM * pulp.lpSum(dev_adm[g]  for g in G)
#         + W_SLV * pulp.lpSum(no_lead[g]  for g in G)
#     )

#     prob.solve(pulp.PULP_CBC_CMD(msg=False))

#     # ====== 결과 ======
#     raw = {g: [] for g in G}
#     for i in I:
#         for g in G:
#             if pulp.value(x[i][g]) > 0.5:
#                 raw[g].append(i)

#     # 그룹 내부 정렬: 운영진 → 영어 상 → 나머지
#     def sort_key(idx):
#         return (-is_admin[idx], -(df.iloc[idx]['english']=='상'), idx)

#     ordered = OrderedDict()
#     for g in sorted(raw):                   # group_1, group_2, …
#         names = [df.iloc[i]['name'] for i in sorted(raw[g], key=sort_key)]
#         ordered[f"group_{g+1}"] = names

#     return ordered

from ortools.sat.python import cp_model
import math
from collections import OrderedDict
def choose_group_count(n):
    """
    n명 참석 시 4·3인 그룹 배치를 위한 최소 그룹 수와
    허용되는 3인 그룹 개수 k (k ≤ 2) 를 반환.
    """
    # print(f'{n=}')
    g = math.ceil(n / 4)          # 우선 4명 기준 최소 그룹 수
    while True:
        # print(g)
        three_groups = (4 * g) - n  # 3명 그룹 개수 (4명 그룹 g−three_groups)
        if 0 <= three_groups <= 3:
            # print(f'{g=}')
            # print(f'{three_groups=}')
            return g, three_groups
        g += 1                    # 그룹 수를 늘려서 three_groups 감소

def build_groups_cpsat(df):
    n = len(df)
    G, three_max = choose_group_count(n)
    if three_max > 2: G += 1; three_max = 4*G - n
    I, Gs = range(n), range(G)

    admin  = df.is_admin.astype(int).tolist()
    leader = (df.is_admin | (df.english=='상')).astype(int).tolist()
    new    = df.is_new.astype(int).tolist()
    male   = (df.gender=='M').astype(int).tolist()
    p1,p2,p3  = df.get('prev_group_code1',''), df.get('prev_group_code2',''), df.get('prev_group_code3','')

    mdl = cp_model.CpModel()
    x   = {(i,g): mdl.NewBoolVar(f'x_{i}_{g}') for i in I for g in Gs}

    # ① 모든 인원 한 그룹
    for i in I: mdl.Add(sum(x[i,g] for g in Gs) == 1)

    # ② 그룹 크기
    sizes = {}
    for g in Gs:
        sizes[g] = mdl.NewIntVar(3,4,f'sz_{g}')
        mdl.Add(sizes[g] == sum(x[i,g] for i in I))
    mdl.Add(sum((4-sizes[g]) for g in Gs) <= 2)   # 3명 그룹 ≤2

    # ③ 리더
    for g in Gs:
        mdl.Add(sum(leader[i]*x[i,g] for i in I) >= 1)

    # ④ 신규→운영진
    for g in Gs:
        mdl.Add(sum(admin[i]*x[i,g] for i in I) * n >=
                sum(new[i]*x[i,g] for i in I))

    # ⑤ f-조건
    def same(i,j):
        return (
                (bool(p1.iloc[i]) and p1.iloc[i]==p1.iloc[j]) or
                (bool(p2.iloc[i]) and p2.iloc[i]==p2.iloc[j]) or
                (bool(p3.iloc[i]) and p3.iloc[i]==p3.iloc[j])
                )
    for i in I:
        for j in range(i+1,n):
            if same(i,j):
                for g in Gs:
                    mdl.Add(x[i,g] + x[j,g] <= 1)

    # ---- 목적함수: 4명 맞추기 > 성비 ----
    dev_sz  = [mdl.NewIntVar(0,1,f'dsz_{g}') for g in Gs]
    dev_m   = [mdl.NewIntVar(0,n,f'dm_{g}')  for g in Gs]
    # print(f'sum male: {sum(male)}')
    # print(f'G: {G}')
    avg_m   = sum(male)//G
    for g in Gs:
        mdl.Add(dev_sz[g] == 4 - sizes[g])
        mcnt = mdl.NewIntVar(0,n,f'mcnt_{g}')
        mdl.Add(mcnt == sum(male[i]*x[i,g] for i in I))
        mdl.Add(dev_m[g] >= mcnt - avg_m)
        mdl.Add(dev_m[g] >= avg_m - mcnt)

    mdl.Minimize(
        50_000*sum(dev_sz) +
        5_000*sum(dev_m)
    )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 120
    solver.parameters.num_search_workers = 8
    status = solver.Solve(mdl)

    groups = {g: [] for g in Gs}
    for i in I:
        for g in Gs:
            if solver.Value(x[i,g]): groups[g].append(i)

    def key(idx): return (-admin[idx], -(df.iloc[idx].english=='상'), idx)
    out = OrderedDict()
    for g in sorted(groups):
        names = [df.iloc[i]['name'] for i in sorted(groups[g], key=key)]
        out[f"Group {g+1}"] = names
    return out


def print_output(TODAY, grp_dict):
    print(f"▶ {TODAY} 그룹 편성 결과")
    for k, members in grp_dict.items():
        print(f"{k} ({len(members)}명) : {', '.join(members)}")
    return



def save_updated_attendance(txt):
    member_list = pd.read_excel('member_list.xlsx')
    attendance = pd.read_excel('attendance.xlsx')

    # txt = '''
    # ▶ 2025-06-14 그룹 편성 결과
    # Group 1 (4명) : 상훈, 박은서, 안디, 오현진
    # Group 2 (4명) : 김정주, 이유정, 김, 이원진
    # Group 3 (4명) : 김선우, 김준희, 서경덕, 동호
    # Group 4 (4명) : 천지현, 김수연, 테리, 이다훈
    # Group 5 (3명) : 김태주, 이예지, 석종훈
    # Group 6 (3명) : 류진향, 박제용, 정현구
    # '''

    attendance_temp = attendance.copy()

    cols=['name','gender','attend_date','total_attend','is_new','english','is_admin','group_code']

    attend_date = txt.split('Group ')[0].split(' ')[1]
    for elem in txt.split('Group ')[1:]:
        group_code = int(elem[0])
        for raw_name in elem.split(': ')[1].split(', '):
            if '\n' in raw_name:
                name = raw_name[:-1]
                if '\n' in name:
                    name = name[:-1]
            else:   
                name = raw_name
            gender = member_list['gender'].loc[member_list['name']==name].values[0]
            total_attend = sum(attendance['name'] == name)
            is_new = True if sum(attendance['name'] == name)==0 else False
            english = member_list['english'].loc[member_list['name']==name].values[0]
            is_admin = member_list['is_admin'].loc[member_list['name']==name].values[0]
            new_df = pd.DataFrame([[name,gender,attend_date,total_attend,is_new,english,is_admin,group_code]], columns=cols)
            attendance_temp = pd.concat([attendance_temp, new_df], ignore_index=True)
            print(new_df)
    attendance_temp["attend_date"] = pd.to_datetime(attendance_temp["attend_date"]).dt.date
    drop_cols = [x for x in attendance_temp.columns if 'Unnamed' in x]
    print(drop_cols)
    attendance_temp.drop(columns=drop_cols, inplace=True)
    print(attendance_temp)
    attendance_temp.to_excel('attendance.xlsx')
    return