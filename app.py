# streamlit run app.py

import numpy as np
import pandas as pd
import os
import time
import streamlit as st
from datetime import datetime
from main import main, update_member_list
from common.utils import save_updated_attendance, compare_and_update_member_list_excel
from PIL import Image

from google.oauth2 import service_account
import gspread

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
# from webdriver_manager.chrome import ChromeDriverManager
# from webdriver_manager.core.os_manager import ChromeType

@st.cache_resource
def get_driver():
    options = Options()
    options.add_argument("--headless")
    # options.add_argument("--disable-gpu")
    options.add_argument("--single-process")
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    driver = webdriver.Chrome(options=options)
    return driver

driver = get_driver()

# Create a connection object.
# credentials = service_account.Credentials.from_service_account_info(
#     st.secrets["gcp_service_account"],
#     scopes=[
#         "https://www.googleapis.com/auth/spreadsheets",
#     ],
# )
# gc = gspread.authorize(credentials)
gc = "testing"


st.markdown("""
<style>
.small-font {
    font-size:10px !important;
}
</style>
""", unsafe_allow_html=True)
today_date = datetime.today().strftime('%Y-%m-%d')
st.markdown(f'<p class="small-font">Last Updated: {today_date}</p>', unsafe_allow_html=True)
st.write("2030 강서 주말 영어회화")
st.write(f"하이강서 @마곡나루")
st.write("> 그룹 편성 Randomizer *by James*")


guide_text = """
* 소모임 어플에서 참여 멤버 리스트를 캡쳐 후, 아래에서 이미지들을 업로드 해주세요. \n \n
* 1부, 2부 종료 시, 반드시 각각 Update Attendance를 실행해 주세요. \n
ex) \n
* [1부] 멤버 리스트 캡쳐 및 업로드 -> 1부 Randomizer 실행 -> 1부 세션 종료 후 -> Update Attendance \n
* [2부] 멤버 리스트 업로드 -> 2부 Randomizer 실행 -> 2부 세션 종료 후 -> Update Attendance \n
"""
def stream_data():
    for word in guide_text.split(" "):
        yield word + " "
        time.sleep(0.02)

if st.button(f" :sunglasses: **[ Randomizer 사용 가이드 *Click Here* ]** :sunglasses:"):
    st.write_stream(stream_data)
    st.write(f" ")
    with st.expander("* 1부만 참여하는 인원 있을경우 "):
        st.write(f" 1) 1부 끝나고 Update Attendance에서 출석 업데이트 후,")
        st.write(f" 2) 캡처 이미지에서 스티커나 그리기 도구로 [1부 only 멤버 이름들] 가리기")
        st.write(f" 3) 마스킹 처리된 캡처 이미지로 Randomizer 실행")
        st.write(f" ")
    with st.expander("* 2부만 참여하는 인원 있을경우 "):
        st.write(f" 1) 캡처 이미지에서 스티커나 그리기 도구로 [2부 only 멤버 이름들] 가리고 나서 Randomizer 실행")
        st.write(f" 2) 1부 끝나고 Update Attendance에서 출석 업데이트 후,")
        st.write(f" 3) 마스킹 안된 원본 이미지로 Randomizer 실행")
        st.write(f" ")
st.divider()


tab1, tab2, tab3 = st.tabs(["Group Randomizer", "Update Attendance", "Current Members"])

# 조 랜덤 편성
with tab1:
    date = st.date_input("Pick a date : **1부는 토요일 날짜, 2부는 일요일 날짜로 입력해주세요**", key='randomizer_date_input')
    uploaded_files = st.file_uploader(
        "Choose image files", accept_multiple_files=True
    )

    file_list = []
    for i, uploaded_file in enumerate(uploaded_files):
        # bytes_data = uploaded_file.read()
        # file_list.append(uploaded_file.name)
        # file_list.append(bytes_data)
        image = Image.open(uploaded_file)
        image.save(f"{i}.jpg")
        file_list.append(f"{i}.jpg")

    left, left1, middle, middle1, right, right1 = st.columns(6)
    if right1.button("Submit", type='secondary', icon="🚨", key='randomizer_button'):
        st.divider()
        st.write(f" ")
        with st.spinner("Wait for it...", show_time=True):
            grp_dict, new_members, resigned_members, missing_members, updated_total_members = main(date, file_list, driver, gc)
            new_member_list = (", ".join(new_members))
            resigned_members = (", ".join(resigned_members))
            missing_members = (", ".join(missing_members))
            st.write(f"▶ {date} 멤버 리스트 업데이트 완료")
            st.write(f"오늘의 신규 멤버: {new_member_list}")
            st.write(f"탈퇴한 기존 멤버: {resigned_members}")
            st.divider()
            st.write(f"▶ {date} 그룹 편성 결과")
            member_count = 0
            for k, members in grp_dict.items():
                st.write(f"{k} ({len(members)}명) : {', '.join(members)}")
                member_count += (len(members))
            st.write(f"총 멤버 수: {member_count}명")
            st.write(f"누락 멤버: {missing_members} - 있다면 그룹 편성에 추가해주세요!")
            st.success('Done!')



# @st.dialog("Update Attendance")
# def confirm():
#     st.write(f"Are you sure? Updates cannot be reverted.")
#     left, left1, right, right1 = st.columns(4)
#     if right.button ("Cancel", key='update_cancel_button'):
#         st.session_state.confirm = False
#         st.rerun()
#     if right1.button("Confirm", key='update_confirm_button'):
#         st.session_state.confirm = True
#         st.rerun()

# 출석체크 업데이트하기기
with tab2:
    # update_date = st.date_input("Pick a date", key='update_date_input')
    txt = st.text_area(label="Copy Group Info:")
    if 'confirm' not in st.session_state:
        st.session_state.confirm = False

    if st.button("Submit", type='secondary', icon="🚨", key='update_submit_button'):
        # confirm()
        st.write(st.session_state.confirm)
    if st.session_state.confirm:
        save_updated_attendance(txt)
        st.success('[Update Complete!')

# 현재 모임 멤버 이름(아이디) 확인
with tab3:
    with st.spinner("Wait for it...", show_time=True):
        sheet_url = st.secrets["memberlist_url"]
        # sheet = gc.open_by_url(sheet_url)
        # st.write(updated_total_members)


        new_members, resigned_members, updated_total_members = update_member_list(date, file_list, driver, gc)
        new_member_list = (", ".join(new_members))
        resigned_members = (", ".join(resigned_members))
        st.write(f"▶ {date} 멤버 리스트 업데이트 완료")
        st.write(f"신규 멤버: {new_member_list}")
        st.write(f"탈퇴 멤버: {resigned_members}")
        st.divider()
        st.write(f"▶ 현재 전체 멤버")
        st.write(updated_total_members)
        df = pd.DataFrame(updated_total_members)
        st.dataframe(df, use_container_width=True)


    
    

