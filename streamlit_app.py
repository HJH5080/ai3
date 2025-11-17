# streamlit_py
import os, re
from io import BytesIO
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from fastai.vision.all import *
import gdown

# ======================
# 페이지/스타일
# ======================
st.set_page_config(page_title="Fastai 이미지 분류기", page_icon="🤖", layout="wide")
st.markdown("""
<style>
h1 { color:#1E88E5; text-align:center; font-weight:800; letter-spacing:-0.5px; }
.prediction-box { background:#E3F2FD; border:2px solid #1E88E5; border-radius:12px; padding:22px; text-align:center; margin:16px 0; box-shadow:0 4px 10px rgba(0,0,0,.06);}
.prediction-box h2 { color:#0D47A1; margin:0; font-size:2.0rem; }
.prob-card { background:#fff; border-radius:10px; padding:12px 14px; margin:10px 0; box-shadow:0 2px 6px rgba(0,0,0,.06); }
.prob-bar-bg { background:#ECEFF1; border-radius:6px; width:100%; height:22px; overflow:hidden; }
.prob-bar-fg { background:#4CAF50; height:100%; border-radius:6px; transition:width .5s; }
.prob-bar-fg.highlight { background:#FF6F00; }
.info-grid { display:grid; grid-template-columns:repeat(12,1fr); gap:14px; }
.card { border:1px solid #e3e6ea; border-radius:12px; padding:14px; background:#fff; box-shadow:0 2px 6px rgba(0,0,0,.05); }
.card h4 { margin:0 0 10px; font-size:1.05rem; color:#0D47A1; }
.thumb { width:100%; height:auto; border-radius:10px; display:block; }
.thumb-wrap { position:relative; display:block; }
.play { position:absolute; top:50%; left:50%; transform:translate(-50%,-50%); width:60px; height:60px; border-radius:50%; background:rgba(0,0,0,.55); }
.play:after{ content:''; border-style:solid; border-width:12px 0 12px 20px; border-color:transparent transparent transparent #fff; position:absolute; top:50%; left:50%; transform:translate(-40%,-50%); }
.helper { color:#607D8B; font-size:.9rem; }
.stFileUploader, .stCameraInput { border:2px dashed #1E88E5; border-radius:12px; padding:16px; background:#f5fafe; }
</style>
""", unsafe_allow_html=True)

st.title("이미지 분류기 (Fastai) — 확률 막대 + 라벨별 고정 콘텐츠")

# ======================
# 세션 상태
# ======================
if "img_bytes" not in st.session_state:
    st.session_state.img_bytes = None
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

# ======================
# 모델 로드
# ======================
FILE_ID = st.secrets.get("GDRIVE_FILE_ID", "1aMWEa_KXv3LXmqECcxkKDubfiVGv4ZnJ")
MODEL_PATH = st.secrets.get("MODEL_PATH", "model.pkl")

@st.cache_resource
def load_model_from_drive(file_id: str, output_path: str):
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)
    return load_learner(output_path, cpu=True)

with st.spinner("🤖 모델 로드 중..."):
    learner = load_model_from_drive(FILE_ID, MODEL_PATH)
st.success("✅ 모델 로드 완료")

labels = [str(x) for x in learner.dls.vocab]
st.write(f"**분류 가능한 항목:** `{', '.join(labels)}`")
st.markdown("---")

# ======================
# 라벨 이름 매핑: 여기를 채우세요!
# 각 라벨당 최대 3개씩 표시됩니다.
# ======================
CONTENT_BY_LABEL: dict[str, dict[str, list[str]]] = {
    # 예)
    # "짬뽕": {
    #   "texts": ["짬뽕의 특징과 유래", "국물 맛 포인트", "지역별 스타일 차이"],
    #   "images": ["https://.../jjampong1.jpg", "https://.../jjampong2.jpg"],
    #   "videos": ["https://youtu.be/XXXXXXXXXXX"]
    # },

    labels[0]:{"texts":["아일릿은 인기가 많은 신인 아이돌이야"],
               "videos":["https://youtu.be/SvAtijkbp4w?si=slRe4jtIGPLL1xnV"]},
    labels[1]:{"texts":["에스파는 가장 남자들한테 인기가 많은 그룹이야"],
               "videos":["https://youtu.be/JvABRVxCoJU?si=HF1vL47-KRt4WRXD"]},
    labels[2]:{"texts":["하츠투하츠는 최근 유입된 팬이 많아"],
               "videos":["https://youtu.be/FGBwQeD2FpY?si=g8iMbVCQh_gM5WfI"]
              "images":["data:image/jpeg;base64,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"]},
}

# ======================
# 유틸
# ======================
def load_pil_from_bytes(b: bytes) -> Image.Image:
    pil = Image.open(BytesIO(b))
    pil = ImageOps.exif_transpose(pil)
    if pil.mode != "RGB": pil = pil.convert("RGB")
    return pil

def yt_id_from_url(url: str) -> str | None:
    if not url: return None
    pats = [r"(?:v=|/)([0-9A-Za-z_-]{11})(?:\?|&|/|$)", r"youtu\.be/([0-9A-Za-z_-]{11})"]
    for p in pats:
        m = re.search(p, url)
        if m: return m.group(1)
    return None

def yt_thumb(url: str) -> str | None:
    vid = yt_id_from_url(url)
    return f"https://img.youtube.com/vi/{vid}/hqdefault.jpg" if vid else None

def pick_top3(lst):
    return [x for x in lst if isinstance(x, str) and x.strip()][:3]

def get_content_for_label(label: str):
    """라벨명으로 콘텐츠 반환 (texts, images, videos). 없으면 빈 리스트."""
    cfg = CONTENT_BY_LABEL.get(label, {})
    return (
        pick_top3(cfg.get("texts", [])),
        pick_top3(cfg.get("images", [])),
        pick_top3(cfg.get("videos", [])),
    )

# ======================
# 입력(카메라/업로드)
# ======================
tab_cam, tab_file = st.tabs(["📷 카메라로 촬영", "📁 파일 업로드"])
new_bytes = None

with tab_cam:
    cam = st.camera_input("카메라 스냅샷", label_visibility="collapsed")
    if cam is not None:
        new_bytes = cam.getvalue()

with tab_file:
    f = st.file_uploader("이미지를 업로드하세요 (jpg, png, jpeg, webp, tiff)",
                         type=["jpg","png","jpeg","webp","tiff"])
    if f is not None:
        new_bytes = f.getvalue()

if new_bytes:
    st.session_state.img_bytes = new_bytes

# ======================
# 예측 & 레이아웃
# ======================
if st.session_state.img_bytes:
    top_l, top_r = st.columns([1, 1], vertical_alignment="center")

    pil_img = load_pil_from_bytes(st.session_state.img_bytes)
    with top_l:
        st.image(pil_img, caption="입력 이미지", use_container_width=True)

    with st.spinner("🧠 분석 중..."):
        pred, pred_idx, probs = learner.predict(PILImage.create(np.array(pil_img)))
        st.session_state.last_prediction = str(pred)

    with top_r:
        st.markdown(
            f"""
            <div class="prediction-box">
                <span style="font-size:1.0rem;color:#555;">예측 결과:</span>
                <h2>{st.session_state.last_prediction}</h2>
                <div class="helper">오른쪽 패널에서 예측 라벨의 콘텐츠가 표시됩니다.</div>
            </div>
            """, unsafe_allow_html=True
        )

    left, right = st.columns([1,1], vertical_alignment="top")

    # 왼쪽: 확률 막대
    with left:
        st.subheader("상세 예측 확률")
        prob_list = sorted(
            [(labels[i], float(probs[i])) for i in range(len(labels))],
            key=lambda x: x[1], reverse=True
        )
        for lbl, p in prob_list:
            pct = p * 100
            hi = "highlight" if lbl == st.session_state.last_prediction else ""
            st.markdown(
                f"""
                <div class="prob-card">
                  <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
                    <strong>{lbl}</strong><span>{pct:.2f}%</span>
                  </div>
                  <div class="prob-bar-bg">
                    <div class="prob-bar-fg {hi}" style="width:{pct:.4f}%;"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True
            )

    # 오른쪽: 정보 패널 (예측 라벨 기본, 다른 라벨로 바꿔보기 가능)
    with right:
        st.subheader("라벨별 고정 콘텐츠")
        default_idx = labels.index(st.session_state.last_prediction) if st.session_state.last_prediction in labels else 0
        info_label = st.selectbox("표시할 라벨 선택", options=labels, index=default_idx)

        texts, images, videos = get_content_for_label(info_label)

        if not any([texts, images, videos]):
            st.info(f"라벨 `{info_label}`에 대한 콘텐츠가 아직 없습니다. 코드의 CONTENT_BY_LABEL에 추가하세요.")
        else:
            # 텍스트
            if texts:
                st.markdown('<div class="info-grid">', unsafe_allow_html=True)
                for t in texts:
                    st.markdown(f"""
                    <div class="card" style="grid-column:span 12;">
                      <h4>텍스트</h4>
                      <div>{t}</div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # 이미지(최대 3, 3열)
            if images:
                st.markdown('<div class="info-grid">', unsafe_allow_html=True)
                for url in images[:3]:
                    st.markdown(f"""
                    <div class="card" style="grid-column:span 4;">
                      <h4>이미지</h4>
                      <img src="{url}" class="thumb" />
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # 동영상(유튜브 썸네일)
            if videos:
                st.markdown('<div class="info-grid">', unsafe_allow_html=True)
                for v in videos[:3]:
                    thumb = yt_thumb(v)
                    if thumb:
                        st.markdown(f"""
                        <div class="card" style="grid-column:span 6;">
                          <h4>동영상</h4>
                          <a href="{v}" target="_blank" class="thumb-wrap">
                            <img src="{thumb}" class="thumb"/>
                            <div class="play"></div>
                          </a>
                          <div class="helper">{v}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="card" style="grid-column:span 6;">
                          <h4>동영상</h4>
                          <a href="{v}" target="_blank">{v}</a>
                        </div>
                        """, unsafe_allow_html=True)
else:
    st.info("카메라로 촬영하거나 파일을 업로드하면 분석 결과와 라벨별 콘텐츠가 표시됩니다.")
