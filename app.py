import streamlit as st
from google.cloud import vision
import io
import os

# ---------------------------
# 🔹 Google Vision API 설정
# ---------------------------
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = st.secrets["general"]["GOOGLE_APPLICATION_CREDENTIALS"]

client = vision.ImageAnnotatorClient()

st.title("📋 도로주행 근무 자동 배정 도우미")
st.write("이미지를 업로드하면 근무자 이름을 자동으로 인식하고, 교양/1종수동/2종자동 순번에 맞게 결과를 표시합니다.")

# ---------------------------
# 🔹 옵션 설정
# ---------------------------
st.subheader("옵션 선택")
session_type = st.radio("근무 이미지 종류를 선택하세요", ["오전 (1~2교시)", "오후 (3~5교시)"])
manual_count = st.radio("1종 수동 인원 수", [1, 2], index=0)

# ---------------------------
# 🔹 이미지 업로드
# ---------------------------
uploaded_file = st.file_uploader("근무표 이미지를 업로드하세요", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = vision.Image(content=uploaded_file.read())
    response = client.text_detection(image=image)
    texts = response.text_annotations

    if texts:
        detected_text = texts[0].description
        st.text_area("📄 인식된 텍스트", detected_text, height=250)

        # ---------------------------
        # 🔹 예시 로직 (간단 표시)
        # ---------------------------
        st.subheader("📊 자동 배정 결과")

        if session_type.startswith("오전"):
            st.write("🧑‍🏫 **교양 담당** → 1교시, 2교시 표시")
        else:
            st.write("🧑‍🏫 **교양 담당** → 3교시~5교시 표시")

        st.write(f"🚘 **1종 수동 담당자 수**: {manual_count}명")

        st.info("※ 실제 교양·열쇠·차량 순번 로직은 이후 자동화 데이터 연동 예정입니다.")
    else:
        st.error("❌ 텍스트를 인식하지 못했습니다. 이미지 해상도를 확인해주세요.")
