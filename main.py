'''
group_randomizer.py
- ver. 250609
- by James

## 필요 데이터
1. 현재 총 멤버
2. 이번 모임 출석 멤버
2. 현재까지 출결 데이터

## Workflow
1. 현재 총 멤버 업데이트 (from Somoim)
2. 이번 모임 출석 멤버 input (from captures)
3. 현재까지 출결 데이터 활용하여 필터링 후 이번 모임 조 편성 output (Python conditions)
    a. 운영진+신규회원
    b. 남녀 비중 최대한 분배
    c. 1부-2부 중복되지 않도록
    d. 영어 실력 분배
    e. (optional) 이전 2 세션까지 비교해서 같이 한 사람들과 조 편성 최대한 피하기
'''

from common.utils import *



def main(TODAY, imgs, driver):
    ## 소모임 전체 멤버 리스트 가져오기 (운영진, 전체 멤버)
    board_members, total_members = get_current_all_members(driver)
    ## 전체 멤버 리스트 업데이트하고 신규멤버 반환
    new_members, resigned_members = compare_and_update_member_list_excel(total_members)
    ## 이번 모임 출석 멤버 input (from captures)
    curr_participants = extract_member_names(*imgs)
    # print(f'{curr_participants=}')
    attendance_df = update_attendance(TODAY, curr_participants)
    # print(f'{attendance_df=}')
    today_df = load_today(attendance_df, TODAY)
    # print(f'{today_df=}')
    # grp_dict = build_groups_cpsat(today_df)
    grp_dict = build_groups(today_df)
    # print(f'{grp_dict=}')
    member_name_lst = print_output(TODAY, grp_dict)
    missing_members = [x for x in curr_participants if x not in member_name_lst]

    return grp_dict, new_members, resigned_members, missing_members



if __name__ == "__main__":
    TODAY = "2025-06-14"
    imgs = [
        # "KakaoTalk_20250608_202659088.jpg",
        # "KakaoTalk_20250608_202659088_01.jpg"
        "1.png",
        "2.png"
    ]

    main(TODAY, imgs)
