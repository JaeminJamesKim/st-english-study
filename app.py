# streamlit run app.py

import numpy as np
import pandas as pd
import os
import time
import streamlit as st
from main import main
from common.utils import save_updated_attendance
from PIL import Image

st.write("2030 강서 주말 영어회화")
st.write("하이강서 @마곡나루")
st.write("> 그룹 편성 Randomizer")
st.divider()

guide_text = """
* 소모임 어플에서 참여 멤버 리스트를 캡쳐 후, 아래에서 이미지들을 업로드 해주세요. \n \n
* 1부, 2부 종료 시, 반드시 각각 Update Attendance를 실행해 주세요. \n
ex) \n
* [1부] 멤버 리스트 캡쳐 및 업로드 -> 1부 Randomizer 실행 -> 1부 종료 -> Update Attendance \n
* [2부] 멤버 리스트 업로드 -> 2부 Randomizer 실행 -> 2부 종료 --> Update Attendance \n
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
        st.write(f" 1) 캡처 이미지에서 스티커나 그리기 도구로 이름 가리고 나서 Randomizer 실행")
        st.write(f" 2) 1부 끝나고 Update Attendance에서 출석 업데이트 후,")
        st.write(f" 3) 마스킹 안된 원본 이미지로 Randomizer 실행")
        st.write(f" ")



tab1, tab2 = st.tabs(["Group Randomizer", "Update Attendance"])

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
            grp_dict = main(date, file_list)
            st.write(f"▶ {date} 그룹 편성 결과")
            for k, members in grp_dict.items():
                st.write(f"{k} ({len(members)}명) : {', '.join(members)}")
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


    
    

