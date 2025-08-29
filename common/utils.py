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
from ortools.sat.python import cp_model
import math
from google.oauth2 import service_account
import gspread

warnings.filterwarnings("ignore")
load_dotenv()
client = OpenAI()                     # 환경변수 OPENAI_API_KEY 사용

def get_current_all_members(driver):
    '''
    ## 1. 현재 총 멤버 업데이트 (from Somoim)
    '''
    page_url = 'https://www.somoim.co.kr/95381bac-d344-11ec-ba98-0a4d683471cd1'

    # options = Options()
    # options.add_argument("--headless")
    # options.add_argument("--disable-gpu")
    # driver = webdriver.Chrome(options=options)

    # 2) Open the page
    driver.get(page_url)
    # driver.close()
    # driver.quit()
    time.sleep(5)

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
def compare_and_update_member_list_excel(curr_member_name_lst, gc):
    member_list = pd.read_excel('member_list.xlsx',sheet_name="Sheet1")
    new_members = [n for n in curr_member_name_lst if n not in set(member_list["name"])]
    resigned_members = [n for n in member_list["name"] if n not in set(curr_member_name_lst)]
    if len(new_members) == 0:
        new_members = ['없음']
    if len(resigned_members) == 0:
        resigned_members = ['없음']
    for member_name in curr_member_name_lst:
        if member_name not in member_list['name']:
            print(f"{member_list=}")
            member_list.loc[len(member_list)] = [len(member_list), member_name, 'M', 0, '하', False]
    member_list = member_list[member_list["name"].isin(curr_member_name_lst)].reset_index(drop=True)
    member_list.to_excel('member_list.xlsx') #TODO Check for Release
    
    print('>> [UPDATE COMPLETE] member_list.xlsx')
    return new_members, resigned_members, member_list

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
                "중간 중간 비어있는 열이 있더라도 그냥 무시하고 나머지 이름들을 반환해줘.\n"
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
    print(names)
    return names




import math
import random
from collections import Counter, defaultdict
import pandas as pd
import pulp


# -------------------- 이번 주 인원 리스트 불러오기 --------------------


# -------------------- 이번 주 인원 출석 데이터에 병합하기 --------------------
def update_attendance(TODAY, curr_participants, gc):
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
    # """
    # n명 참석 시 4·3인 그룹 배치를 위한 최소 그룹 수와
    # 허용되는 3인 그룹 개수 k (k ≤ 2) 를 반환.
    # """
    # # print(f'{n=}')
    # g = math.ceil(n / 4)          # 우선 4명 기준 최소 그룹 수
    # while True:
    #     # print(g)
    #     three_groups = (4 * g) - n  # 3명 그룹 개수 (4명 그룹 g−three_groups)
    #     if 0 <= three_groups <= 3:
    #         # print(f'{g=}')
    #         # print(f'{three_groups=}')
    #         return g, three_groups
    #     g += 1                    # 그룹 수를 늘려서 three_groups 감소

# def build_groups_cpsat(df):
    # n = len(df)
    # G, three_max = choose_group_count(n)
    # if three_max > 2: G += 1; three_max = 4*G - n
    # I, Gs = range(n), range(G)

    # admin  = df.is_admin.astype(int).tolist()
    # leader = (df.is_admin | (df.english=='상')).astype(int).tolist()
    # new    = df.is_new.astype(int).tolist()
    # male   = (df.gender=='M').astype(int).tolist()
    # p1,p2,p3  = df.get('prev_group_code1',''), df.get('prev_group_code2',''), df.get('prev_group_code3','')

    # mdl = cp_model.CpModel()
    # x   = {(i,g): mdl.NewBoolVar(f'x_{i}_{g}') for i in I for g in Gs}

    # # 0.  ***이름 단위 중복 방지 제약 추가***  ← 새 코드
    # name_to_rows = df.groupby('name').groups      # {'김OO': [0,7], '이OO': [3], ...}
    # for rows in name_to_rows.values():
    #     if len(rows) > 1:                         # 동일 이름이 여러 행
    #         mdl.Add(sum(x[i,g] for i in rows for g in Gs) <= 1)

    # # ① 모든 인원(행) 한 그룹
    # for i in I:
    #     mdl.Add(sum(x[i,g] for g in Gs) == 1)

    # # # ① 모든 인원 한 그룹
    # # for i in I: mdl.Add(sum(x[i,g] for g in Gs) == 1)

    # # ② 그룹 크기
    # sizes = {}
    # for g in Gs:
    #     sizes[g] = mdl.NewIntVar(3,4,f'sz_{g}')
    #     mdl.Add(sizes[g] == sum(x[i,g] for i in I))
    # mdl.Add(sum((4-sizes[g]) for g in Gs) <= 2)   # 3명 그룹 ≤2

    # # ③ 리더
    # for g in Gs:
    #     mdl.Add(sum(leader[i]*x[i,g] for i in I) >= 1)

    # # ④ 신규→운영진
    # for g in Gs:
    #     mdl.Add(sum(admin[i]*x[i,g] for i in I) * n >=
    #             sum(new[i]*x[i,g] for i in I))

    # # ⑤ f-조건
    # def same(i,j):
    #     return (
    #             (bool(p1.iloc[i]) and p1.iloc[i]==p1.iloc[j]) or
    #             (bool(p2.iloc[i]) and p2.iloc[i]==p2.iloc[j]) or
    #             (bool(p3.iloc[i]) and p3.iloc[i]==p3.iloc[j])
    #             )
    # for i in I:
    #     for j in range(i+1,n):
    #         if same(i,j):
    #             for g in Gs:
    #                 mdl.Add(x[i,g] + x[j,g] <= 1)

    # # ---- 목적함수: 4명 맞추기 > 성비 ----
    # dev_sz  = [mdl.NewIntVar(0,1,f'dsz_{g}') for g in Gs]
    # dev_m   = [mdl.NewIntVar(0,n,f'dm_{g}')  for g in Gs]
    # # print(f'sum male: {sum(male)}')
    # # print(f'G: {G}')
    # avg_m   = sum(male)//G
    # for g in Gs:
    #     mdl.Add(dev_sz[g] == 4 - sizes[g])
    #     mcnt = mdl.NewIntVar(0,n,f'mcnt_{g}')
    #     mdl.Add(mcnt == sum(male[i]*x[i,g] for i in I))
    #     mdl.Add(dev_m[g] >= mcnt - avg_m)
    #     mdl.Add(dev_m[g] >= avg_m - mcnt)

    # mdl.Minimize(
    #     50_000*sum(dev_sz) +
    #     5_000*sum(dev_m)
    # )

    # solver = cp_model.CpSolver()
    # solver.parameters.max_time_in_seconds = 120
    # solver.parameters.num_search_workers = 8
    # status = solver.Solve(mdl)

    # groups = {g: [] for g in Gs}
    # for i in I:
    #     for g in Gs:
    #         if solver.Value(x[i,g]): groups[g].append(i)

    # def key(idx): return (-admin[idx], -(df.iloc[idx].english=='상'), idx)
    # out = OrderedDict()
    # for g in sorted(groups):
    #     names = [df.iloc[i]['name'] for i in sorted(groups[g], key=key)]
    #     out[f"Group {g+1}"] = names
    # return out


# ──────────────────────────────────────────────────────────────
# 튜닝 가능한 가중치 (값이 클수록 더 중요한 조건)
W_SIZE   = 10000    # ② 그룹 인원 dev(|size-4|)
W_NEWADM =  3000    # ③ 신규+운영진 slack
W_FCOND  =   800    # ④ f-조건(최근 2회) 충돌
W_LEADER =   200    # ⑤ 리더 부족 slack
# ──────────────────────────────────────────────────────────────


def choose_group_count(n):
    """4명·3명 그룹 중심(3명 그룹 ≤2개)을 만족하는 최소 그룹 수 계산"""
    g = math.ceil(n / 4)
    while (4 * g - n) > 2:   # 3명 그룹이 세 개 이상이면 그룹 하나 늘림
        g += 1
    return g


def build_groups(df, timelimit=60, workers=8):
    n = len(df)
    G = choose_group_count(n)
    I, Gs = range(n), range(G)

    # ===== 특성 =====
    admin  = df['is_admin'].astype(int).tolist()
    leader = (df['is_admin'] | (df['english'] == '상')).astype(int).tolist()
    new    = df['is_new'].astype(int).tolist()
    male   = (df['gender'] == 'M').astype(int).tolist()
    prev1  = df.get('prev_group_code1', '')
    prev2  = df.get('prev_group_code2', '')
    prev3  = df.get('prev_group_code3', '')

    mdl = cp_model.CpModel()
    x   = {(i, g): mdl.NewBoolVar(f'x_{i}_{g}') for i in I for g in Gs}

    # 0) --- 이름 중복 절대 금지 ---
    for rows in df.groupby('name').groups.values():
        if len(rows) > 1:
            mdl.Add(sum(x[i, g] for i in rows for g in Gs) <= 1)

    # 1) --- 각 사람 하나의 그룹 ---
    for i in I:
        mdl.Add(sum(x[i, g] for g in Gs) == 1)

    # 2) --- 그룹 인원 및 dev_size ---
    size      = [mdl.NewIntVar(3, 4, f'size_{g}')  for g in Gs]
    dev_size  = [mdl.NewIntVar(0, 1, f'devsz_{g}') for g in Gs]

    for g in Gs:
        mdl.Add(size[g] == sum(x[i, g] for i in I))
        # dev_size = 4 - size   ⇒  size 4→dev0, size3→dev1
        mdl.Add(size[g] + dev_size[g] == 4)

    # 3명 그룹 ≤ 2개 (dev==1 이 2개 이하)
    mdl.Add(sum(dev_size) <= 2)

    # 3) --- 신규 → 운영진 (slack 허용) ---
    slack_newadm = [mdl.NewIntVar(0, n, f'slack_na_{g}') for g in Gs]
    for g in Gs:
        adm_cnt = mdl.NewIntVar(0, n, f'adm_{g}')
        new_cnt = mdl.NewIntVar(0, n, f'new_{g}')
        mdl.Add(adm_cnt == sum(admin[i] * x[i, g] for i in I))
        mdl.Add(new_cnt == sum(new[i]   * x[i, g] for i in I))
        mdl.Add(adm_cnt + slack_newadm[g] >= new_cnt)   # slack으로 부족분 보완

    # 4) --- f-조건 (최근 2회 같은 그룹 피하기) ---
    f_pairs = []
    def same_recent(i, j):
        return (
            (prev1.iloc[i] and prev1.iloc[i] == prev1.iloc[j]) or
            (prev2.iloc[i] and prev2.iloc[i] == prev2.iloc[j]) or
            (prev3.iloc[i] and prev3.iloc[i] == prev3.iloc[j])
        )
    for i in I:
        for j in range(i + 1, n):
            if same_recent(i, j):
                for g in Gs:
                    p = mdl.NewBoolVar(f'f_{i}_{j}_{g}')
                    mdl.Add(p <= x[i, g])
                    mdl.Add(p <= x[j, g])
                    mdl.Add(p >= x[i, g] + x[j, g] - 1)
                    f_pairs.append(p)

    # ===== 5) 리더 부족 slack =====
    slack_leader = [mdl.NewBoolVar(f'slack_lead_{g}') for g in Gs]
    for g in Gs:
        mdl.Add(sum(leader[i] * x[i, g] for i in I) + slack_leader[g] >= 1)

    # ===== 목적함수 (가중치로 우선순위 반영) =====
    mdl.Minimize(
          W_SIZE   * sum(dev_size) +
          W_NEWADM * sum(slack_newadm) +
          W_FCOND  * sum(f_pairs) +
          W_LEADER * sum(slack_leader)
    )

    # ===== 풀기 =====
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = timelimit
    solver.parameters.num_search_workers = workers
    status = solver.Solve(mdl)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("No feasible solution in time limit.")

    # ===== 결과 정리 =====
    raw_groups = {g: [] for g in Gs}
    for i in I:
        for g in Gs:
            if solver.Value(x[i, g]):
                raw_groups[g].append(i)

    # 그룹 내부 정렬 : 운영진 ▶ 영어 '상' ▶ 기타
    def in_key(idx):
        return (-admin[idx], -(df.iloc[idx]['english'] == '상'), idx)

    ordered = OrderedDict()
    for g in sorted(raw_groups):
        ordered[f"Group {g + 1}"] = [
            df.iloc[i]['name'] for i in sorted(raw_groups[g], key=in_key)
        ]
    return ordered


def print_output(TODAY, grp_dict):
    print(f"▶ {TODAY} 그룹 편성 결과")
    member_count = 0
    member_name_lst = []
    for k, members in grp_dict.items():
        print(f"{k} ({len(members)}명) : {', '.join(members)}")
        member_count += len(members)
        member_name_lst.append(members)
    print(f"총 멤버 수: {member_count}명")
    return [item for sublist in member_name_lst for item in sublist]


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

def check_missing_member(curr_participants, member_name_lst):
    missing_members = [x for x in curr_participants if x not in member_name_lst]
    if len(missing_members)==0:
        missing_members = ['없음']
    return missing_members