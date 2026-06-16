import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
import os

st.set_page_config(
    page_title="TomatoGuard | AI Plant Health Diagnostics",
    page_icon="🍅",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.hero {
    background: linear-gradient(135deg, #0b1f0e 0%, #163320 40%, #3d0d0d 100%);
    padding: 2.5rem 2.5rem 2rem;
    border-radius: 20px;
    margin-bottom: 1.5rem;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 25px 70px rgba(0,0,0,0.5);
}
.hero h1 { color:#fff; font-size:2.6rem; font-weight:800; margin:0 0 0.4rem; letter-spacing:-0.8px; }
.hero h1 span { color:#ef5350; }
.hero p { color:rgba(255,255,255,0.6); font-size:0.98rem; max-width:600px; line-height:1.6; margin:0; }

.metric-card {
    background: linear-gradient(145deg,#131c14,#111827);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px; padding: 1.4rem 1.2rem; text-align:center; position:relative; overflow:hidden;
}
.metric-card::after {
    content:""; position:absolute; bottom:0; left:0; right:0; height:3px; border-radius:0 0 16px 16px;
}
.metric-card.green::after  { background:linear-gradient(90deg,#4caf50,#8bc34a); }
.metric-card.red::after    { background:linear-gradient(90deg,#ef5350,#ff7043); }
.metric-card.blue::after   { background:linear-gradient(90deg,#42a5f5,#7e57c2); }
.metric-card.orange::after { background:linear-gradient(90deg,#ffa726,#ef5350); }
.metric-value { font-size:2.2rem; font-weight:800; line-height:1; margin-bottom:0.2rem; }
.metric-value.green  { color:#81c784; }
.metric-value.red    { color:#ef9a9a; }
.metric-value.blue   { color:#90caf9; }
.metric-value.orange { color:#ffcc80; }
.metric-sublabel { color:rgba(255,255,255,0.35); font-size:0.7rem; text-transform:uppercase; letter-spacing:0.8px; font-weight:600; margin-bottom:0.3rem; }
.metric-label    { color:rgba(255,255,255,0.6); font-size:0.8rem; font-weight:500; }

.pipeline-stage {
    background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08);
    border-radius:14px; padding:1.2rem 1.3rem;
}
.stage-num {
    display:inline-flex; align-items:center; justify-content:center;
    width:1.8rem; height:1.8rem; border-radius:50%; font-size:0.8rem; font-weight:700; margin-right:0.5rem;
}
.stage-num.blue  { background:rgba(66,165,245,0.2); color:#90caf9; }
.stage-num.green { background:rgba(76,175,80,0.2);  color:#81c784; }
.stage-title { color:#e0e0e0; font-weight:600; font-size:0.92rem; }
.stage-desc  { color:rgba(255,255,255,0.4); font-size:0.8rem; margin-top:0.3rem; }

.section-header {
    font-size:1.2rem; font-weight:700; color:#e8e8e8;
    margin:1.8rem 0 1rem; padding-bottom:0.5rem;
    border-bottom:1px solid rgba(255,255,255,0.08);
}

.arch-layer {
    background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.07);
    border-radius:10px; padding:0.7rem 1.1rem; margin:0.3rem 0;
    display:flex; align-items:center; gap:0.8rem;
}
.arch-icon  { font-size:1rem; }
.arch-name  { color:#e0e0e0; font-size:0.86rem; font-weight:600; }
.arch-shape { color:rgba(255,255,255,0.3); font-size:0.75rem; font-family:monospace; margin-left:auto; }

.warn-box {
    background:rgba(255,152,0,0.08); border:1px solid rgba(255,152,0,0.3);
    border-radius:10px; padding:0.8rem 1rem; color:#ffcc80; font-size:0.86rem;
}
.info-box {
    background:rgba(33,150,243,0.08); border:1px solid rgba(33,150,243,0.3);
    border-radius:10px; padding:0.8rem 1rem; color:#90caf9; font-size:0.86rem;
}

.sidebar-logo {
    background:linear-gradient(135deg,#163320,#3d0d0d); border-radius:14px;
    padding:1.2rem; text-align:center; margin-bottom:1rem;
    border:1px solid rgba(255,255,255,0.07);
}
.sidebar-logo .emoji { font-size:2.2rem; }
.sidebar-logo h2 { color:#fff; font-size:1rem; font-weight:700; margin:0.3rem 0 0; }
.sidebar-logo .tagline { color:rgba(255,255,255,0.4); font-size:0.72rem; }

.status-dot { display:inline-block; width:7px; height:7px; border-radius:50%; margin-right:5px; vertical-align:middle; }
.dot-green  { background:#4caf50; box-shadow:0 0 5px #4caf50; }
.dot-yellow { background:#ffa726; box-shadow:0 0 5px #ffa726; }

#MainMenu {visibility:hidden;} footer {visibility:hidden;} header {visibility:hidden;}
section[data-testid="stSidebar"] { background-color:#0d1117; }
</style>
""", unsafe_allow_html=True)


# ── DATA ───────────────────────────────────────────────────────────────────────

DISEASE_P1_VAL_LOSS = [
    0.6194,0.4216,0.3102,0.2697,0.2496,0.2205,0.2093,0.1919,0.1852,0.1705,
    0.1724,0.1709,0.1723,0.1695,0.1702,0.1676,0.1666,0.1669,0.1643,0.1579,
    0.1551,0.1547,0.1534,0.1517,0.1533,0.1539,0.1482,0.1499,0.1471,0.1488,
]
DISEASE_P1_TRAIN_LOSS = [
    0.832,0.581,0.436,0.374,0.325,0.286,0.255,0.231,0.212,0.196,
    0.183,0.172,0.163,0.155,0.148,0.142,0.136,0.131,0.127,0.123,
    0.119,0.116,0.113,0.110,0.107,0.105,0.103,0.101,0.099,0.097,
]
DISEASE_P1_VAL_ACC = [
    0.782,0.840,0.882,0.895,0.904,0.915,0.921,0.928,0.930,0.936,
    0.935,0.936,0.935,0.937,0.936,0.938,0.939,0.938,0.940,0.943,
    0.944,0.944,0.945,0.945,0.945,0.945,0.947,0.946,0.947,0.947,
]
DISEASE_P1_TRAIN_ACC = [
    0.715,0.787,0.829,0.855,0.872,0.886,0.896,0.905,0.912,0.918,
    0.924,0.928,0.933,0.937,0.940,0.943,0.946,0.948,0.951,0.953,
    0.955,0.957,0.959,0.960,0.962,0.963,0.964,0.965,0.966,0.967,
]
DISEASE_P2_VAL_LOSS = [
    0.1414,0.1390,0.1340,0.1358,0.1208,0.1110,0.1113,0.1056,0.1070,0.1065,
    0.1015,0.1008,0.0930,0.0955,0.0926,0.0882,0.0877,0.0840,0.0816,0.0762,
    0.0802,0.0823,0.0805,0.0800,0.0783,0.0782,0.0777,
]
DISEASE_P2_TRAIN_LOSS = [
    0.131,0.121,0.113,0.107,0.101,0.096,0.092,0.088,0.085,0.082,
    0.079,0.077,0.074,0.072,0.070,0.068,0.066,0.065,0.063,0.062,
    0.060,0.059,0.058,0.057,0.056,0.056,0.055,
]
DISEASE_P2_VAL_ACC = [
    0.949,0.950,0.951,0.950,0.954,0.956,0.956,0.958,0.957,0.958,
    0.959,0.959,0.961,0.960,0.961,0.962,0.963,0.963,0.964,0.965,
    0.964,0.964,0.964,0.964,0.965,0.965,0.965,
]
DISEASE_P2_TRAIN_ACC = [
    0.967,0.969,0.971,0.972,0.973,0.974,0.975,0.976,0.977,0.977,
    0.978,0.979,0.980,0.980,0.981,0.981,0.982,0.982,0.983,0.983,
    0.983,0.984,0.984,0.984,0.984,0.985,0.985,
]

GATE_P1_VAL_LOSS = [
    0.0930,0.0358,0.0295,0.0221,0.0213,0.0182,0.0170,0.0160,0.0142,0.0143,
    0.0136,0.0127,0.0128,0.0126,0.0121,0.0119,0.0127,0.0114,0.0112,0.0110,
    0.0109,0.0104,0.0110,0.0107,0.0105,0.0107,0.0103,0.0103,0.0106,0.0103,
]
GATE_P1_TRAIN_LOSS = [
    0.112,0.054,0.037,0.027,0.022,0.018,0.015,0.013,0.011,0.010,
    0.009,0.009,0.008,0.008,0.007,0.007,0.007,0.006,0.006,0.006,
    0.006,0.006,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,
]
GATE_P1_VAL_ACC = [
    0.966,0.987,0.989,0.992,0.993,0.994,0.994,0.995,0.996,0.996,
    0.996,0.996,0.996,0.996,0.997,0.997,0.996,0.997,0.997,0.998,
    0.998,0.998,0.997,0.997,0.997,0.997,0.998,0.998,0.997,0.998,
]
GATE_P1_TRAIN_ACC = [
    0.962,0.981,0.987,0.991,0.993,0.995,0.996,0.997,0.997,0.998,
    0.998,0.998,0.998,0.999,0.999,0.999,0.999,0.999,0.999,0.999,
    0.999,0.999,0.999,0.999,0.999,0.999,0.999,0.999,0.999,0.999,
]
GATE_P2_VAL_LOSS   = [0.00957,0.00949,0.00964,0.01003,0.00988,0.00974,0.00977,0.00968,0.00978]
GATE_P2_TRAIN_LOSS = [0.00570,0.00670,0.00720,0.00620,0.00480,0.00460,0.00480,0.00470,0.00440]
GATE_P2_VAL_ACC    = [0.9986]*9
GATE_P2_TRAIN_ACC  = [0.9982,0.9977,0.9976,0.9983,0.9985,0.9985,0.9980,0.9988,0.9991]

DISEASE_CM = np.array([[134,6,10],[0,240,0],[3,5,279]])
DISEASE_CLASSES = ["Early Blight","Healthy","Late Blight"]
GATE_CM = np.array([[750,0],[1,674]])
GATE_CLASSES = ["Non-Tomato","Tomato"]

RED  = "#ef5350"
GRN  = "#4caf50"
ORG  = "#ffa726"
BLU  = "#42a5f5"
THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,255,255,0.02)",
    font=dict(color="rgba(255,255,255,0.7)", family="Inter"),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.05)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.05)"),
)


# ── MODEL LOADING ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models(gate_path, disease_path):
    try:
        import onnxruntime as ort
        gate    = ort.InferenceSession(gate_path)
        disease = ort.InferenceSession(disease_path)
        return gate, disease, None
    except ImportError:
        return None, None, "onnxruntime not installed."
    except Exception as e:
        return None, None, str(e)

def run_inference(img, gate_model, disease_model):
    arr = np.expand_dims(np.array(img.resize((224,224)).convert("RGB"), dtype=np.float32), 0)
    gate_input  = gate_model.get_inputs()[0].name
    gate_prob   = float(gate_model.run(None, {gate_input: arr})[0][0][0])
    if gate_prob < 0.5:
        return {"is_tomato": False, "gate_confidence": 1 - gate_prob}
    disease_input = disease_model.get_inputs()[0].name
    probs = disease_model.run(None, {disease_input: arr})[0][0]
    names = ["Early Blight","Healthy","Late Blight"]
    return {
        "is_tomato": True,
        "gate_confidence": gate_prob,
        "disease": names[int(np.argmax(probs))],
        "disease_probs": probs.tolist(),
        "disease_names": names,
    }


# ── PLOT HELPERS ───────────────────────────────────────────────────────────────
def training_curves_fig(p1_tl, p1_vl, p1_ta, p1_va, p2_tl, p2_vl, p2_ta, p2_va, title=""):
    e1 = list(range(1, len(p1_tl)+1))
    e2 = list(range(len(p1_tl)+1, len(p1_tl)+len(p2_tl)+1))
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Loss","Accuracy"), horizontal_spacing=0.1)
    for row, (y1t, y1v, y2t, y2v) in enumerate([
        (p1_tl, p1_vl, p2_tl, p2_vl),
        ([v*100 for v in p1_ta], [v*100 for v in p1_va],
         [v*100 for v in p2_ta], [v*100 for v in p2_va])
    ], start=1):
        fig.add_trace(go.Scatter(x=e1,y=y1t,mode="lines",name="Train",
            line=dict(color=BLU,width=2),legendgroup="train",showlegend=(row==1)), row=1,col=row)
        fig.add_trace(go.Scatter(x=e1,y=y1v,mode="lines",name="Validation",
            line=dict(color=GRN,width=2),legendgroup="val",showlegend=(row==1)), row=1,col=row)
        fig.add_trace(go.Scatter(x=e2,y=y2t,mode="lines",
            line=dict(color=BLU,width=2,dash="dot"),legendgroup="train",showlegend=False), row=1,col=row)
        fig.add_trace(go.Scatter(x=e2,y=y2v,mode="lines",
            line=dict(color=GRN,width=2,dash="dot"),legendgroup="val",showlegend=False), row=1,col=row)
        fig.add_vline(x=len(p1_tl)+0.5, line_dash="dash",
                      line_color="rgba(255,255,255,0.12)",
                      annotation_text="Fine-tuning →",
                      annotation_font_color="rgba(255,255,255,0.3)",
                      annotation_position="top right", row=1, col=row)
    fig.update_layout(title=dict(text=title,font=dict(size=13,color="rgba(255,255,255,0.75)")),
        height=320, margin=dict(l=10,r=10,t=45,b=10),
        legend=dict(orientation="h",y=-0.2,x=0.5,xanchor="center",bgcolor="rgba(0,0,0,0)"),
        **THEME)
    fig.update_xaxes(title_text="Epoch")
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy (%)", row=1, col=2)
    return fig

def confusion_matrix_fig(cm, labels, title="Confusion Matrix"):
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    n = len(labels)
    fig = go.Figure(go.Heatmap(
        z=cm_norm, x=labels, y=labels,
        text=[[f"{cm[i][j]}<br>({cm_norm[i][j]*100:.1f}%)" for j in range(n)] for i in range(n)],
        texttemplate="%{text}", textfont=dict(size=12,color="white"),
        colorscale=[[0,"rgba(239,83,80,0.15)"],[0.5,"rgba(239,83,80,0.5)"],[1,"#4caf50"]],
        showscale=False, xgap=3, ygap=3,
    ))
    fig.update_layout(title=dict(text=title,font=dict(size=12,color="rgba(255,255,255,0.75)")),
        xaxis_title="Predicted", yaxis_title="True",
        height=300, margin=dict(l=10,r=10,t=45,b=10), **THEME)
    return fig

def per_class_bar(labels, precision, recall, f1, title=""):
    fig = go.Figure()
    fig.add_bar(x=labels, y=[p*100 for p in precision], name="Precision", marker_color=BLU, width=0.22)
    fig.add_bar(x=labels, y=[r*100 for r in recall],    name="Recall",    marker_color=GRN, width=0.22)
    fig.add_bar(x=labels, y=[f*100 for f in f1],        name="F1-Score",  marker_color=RED, width=0.22)
    fig.update_layout(barmode="group",
        title=dict(text=title,font=dict(size=12,color="rgba(255,255,255,0.75)")),
        height=280, margin=dict(l=10,r=10,t=45,b=10),
        legend=dict(orientation="h",y=-0.25,x=0.5,xanchor="center",bgcolor="rgba(0,0,0,0)"),
        **THEME)
    fig.update_yaxes(range=[80,102], title_text="Score (%)")
    return fig


# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
      <div class="emoji">🍅</div>
      <h2>TomatoGuard</h2>
      <div class="tagline">AI Plant Health Diagnostics</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Model Paths**")
    gate_path    = st.text_input("Gate Model (.onnx)",    value="models/gate_phase2.onnx")
    disease_path = st.text_input("Disease Model (.onnx)", value="models/detect_phase2.onnx")

    gate_model, disease_model, load_err = load_models(gate_path, disease_path)

    st.markdown("**Model Status**")
    g_dot = "dot-green" if gate_model    else "dot-yellow"
    d_dot = "dot-green" if disease_model else "dot-yellow"
    g_txt = "loaded"    if gate_model    else "not found"
    d_txt = "loaded"    if disease_model else "not found"
    st.markdown(f'<span class="status-dot {g_dot}"></span> Gate Model — {g_txt}',    unsafe_allow_html=True)
    st.markdown(f'<span class="status-dot {d_dot}"></span> Disease Model — {d_txt}', unsafe_allow_html=True)

    if load_err:
        st.markdown(f'<div class="warn-box" style="margin-top:0.5rem;">⚠️ {load_err[:100]}</div>', unsafe_allow_html=True)

    st.markdown(
        '<p style="color:rgba(255,255,255,0.18);font-size:0.7rem;text-align:center;margin-top:2rem;">CapStone Project · 2024</p>',
        unsafe_allow_html=True)


# ── PERMANENT HEADER ───────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>Tomato<span>Guard</span></h1>
  <p>A two-stage deep learning pipeline for tomato plant health diagnostics —
     confirming the image is a tomato, then classifying any disease present.</p>
</div>
""", unsafe_allow_html=True)

hc1, hc2, hc3, hc4 = st.columns(4)
with hc1:
    st.markdown('<div class="metric-card green"><div class="metric-sublabel">Gate Model</div><div class="metric-value green">99.93%</div><div class="metric-label">Tomato Detection Accuracy</div></div>', unsafe_allow_html=True)
with hc2:
    st.markdown('<div class="metric-card red"><div class="metric-sublabel">Disease Classifier</div><div class="metric-value red">96.16%</div><div class="metric-label">Disease Classification Accuracy</div></div>', unsafe_allow_html=True)
with hc3:
    st.markdown('<div class="metric-card blue"><div class="metric-sublabel">Training Images</div><div class="metric-value blue">9,799</div><div class="metric-label">Across Both Models</div></div>', unsafe_allow_html=True)
with hc4:
    st.markdown('<div class="metric-card orange"><div class="metric-sublabel">Backbone</div><div class="metric-value orange">4.2M</div><div class="metric-label">EfficientNetB0 Parameters</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

pa, parr1, pb, parr2, pc = st.columns([5,0.4,5,0.4,5])
with pa:
    st.markdown('<div class="pipeline-stage"><div><span class="stage-num blue">📷</span><span class="stage-title">Input Image</span></div><div class="stage-desc">224×224 px RGB · any plant photo</div></div>', unsafe_allow_html=True)
with parr1:
    st.markdown("<div style='text-align:center;font-size:1.3rem;color:rgba(255,255,255,0.2);padding-top:1rem;'>→</div>", unsafe_allow_html=True)
with pb:
    st.markdown('<div class="pipeline-stage"><div><span class="stage-num blue">1</span><span class="stage-title">Gate Model</span></div><div class="stage-desc">Tomato vs Non-Tomato · Sigmoid · 99.93%</div></div>', unsafe_allow_html=True)
with parr2:
    st.markdown("<div style='text-align:center;font-size:1.3rem;color:rgba(255,255,255,0.2);padding-top:1rem;'>→</div>", unsafe_allow_html=True)
with pc:
    st.markdown('<div class="pipeline-stage"><div><span class="stage-num green">2</span><span class="stage-title">Disease Classifier</span></div><div class="stage-desc">Early Blight · Healthy · Late Blight · 96.16%</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ── TABS ───────────────────────────────────────────────────────────────────────
tab_diagnose, tab_analytics = st.tabs(["🔬  Diagnose","📊  Analytics"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 · DIAGNOSE
# ══════════════════════════════════════════════════════════════════════════════
with tab_diagnose:
    models_ready = gate_model is not None and disease_model is not None

    diag_left, diag_right = st.columns([1,1], gap="large")

    with diag_left:
        st.markdown('<div style="font-size:0.75rem;font-weight:600;color:rgba(255,255,255,0.35);text-transform:uppercase;letter-spacing:0.8px;margin-bottom:0.5rem;">Upload Image</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("upload", type=["jpg","jpeg","png","webp"], label_visibility="collapsed")
        if uploaded:
            img = Image.open(uploaded)
            st.image(img, use_container_width=True)
            st.markdown(f"""
            <div style="display:flex;gap:0.4rem;margin-top:0.4rem;flex-wrap:wrap;">
              <span style="background:rgba(255,255,255,0.06);border-radius:6px;padding:0.2rem 0.6rem;font-size:0.72rem;color:rgba(255,255,255,0.45);">{img.width}×{img.height} px</span>
              <span style="background:rgba(255,255,255,0.06);border-radius:6px;padding:0.2rem 0.6rem;font-size:0.72rem;color:rgba(255,255,255,0.45);">{img.mode}</span>
              <span style="background:rgba(255,255,255,0.06);border-radius:6px;padding:0.2rem 0.6rem;font-size:0.72rem;color:rgba(255,255,255,0.45);">{uploaded.name}</span>
            </div>
            """, unsafe_allow_html=True)

    with diag_right:
        st.markdown('<div style="font-size:0.75rem;font-weight:600;color:rgba(255,255,255,0.35);text-transform:uppercase;letter-spacing:0.8px;margin-bottom:0.5rem;">Diagnosis Result</div>', unsafe_allow_html=True)

        if not uploaded:
            st.markdown("""
            <div style="height:300px;display:flex;flex-direction:column;align-items:center;justify-content:center;
                        background:rgba(255,255,255,0.015);border:1.5px dashed rgba(255,255,255,0.08);border-radius:16px;">
              <div style="font-size:3rem;margin-bottom:1rem;opacity:0.5;">🍅</div>
              <div style="color:rgba(255,255,255,0.35);font-size:0.88rem;text-align:center;line-height:1.7;">
                Drop an image on the left<br>to run the diagnostic pipeline
              </div>
              <div style="margin-top:1rem;display:flex;gap:0.4rem;">
                <span style="background:rgba(239,83,80,0.12);border:1px solid rgba(239,83,80,0.25);border-radius:20px;padding:0.15rem 0.6rem;font-size:0.7rem;color:#ef9a9a;">Early Blight</span>
                <span style="background:rgba(76,175,80,0.12);border:1px solid rgba(76,175,80,0.25);border-radius:20px;padding:0.15rem 0.6rem;font-size:0.7rem;color:#a5d6a7;">Healthy</span>
                <span style="background:rgba(255,167,38,0.12);border:1px solid rgba(255,167,38,0.25);border-radius:20px;padding:0.15rem 0.6rem;font-size:0.7rem;color:#ffcc80;">Late Blight</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

        elif not models_ready:
            st.markdown("""
            <div style="height:300px;display:flex;flex-direction:column;align-items:center;justify-content:center;
                        background:rgba(255,152,0,0.03);border:1.5px dashed rgba(255,152,0,0.15);border-radius:16px;">
              <div style="font-size:2.5rem;margin-bottom:0.8rem;opacity:0.6;">🔌</div>
              <div style="color:rgba(255,200,100,0.55);font-size:0.86rem;text-align:center;line-height:1.7;">
                Add model paths in the sidebar<br>to enable live inference.
              </div>
            </div>
            """, unsafe_allow_html=True)

        else:
            with st.spinner("Running pipeline…"):
                result = run_inference(img, gate_model, disease_model)

            gate_conf = result["gate_confidence"]

            if result["is_tomato"]:
                disease     = result["disease"]
                probs       = result["disease_probs"]
                class_names = result["disease_names"]
                clr         = {"Early Blight": RED, "Healthy": GRN, "Late Blight": ORG}
                lbl_clr     = {"Early Blight":"#ef9a9a","Healthy":"#66bb6a","Late Blight":"#ffcc80"}
                bdr_clr     = {"Early Blight":"#ef5350","Healthy":"#4caf50","Late Blight":"#ffa726"}
                icons       = {"Early Blight":"🔴","Healthy":"🟢","Late Blight":"🟠"}
                severity    = {"Early Blight":"High Risk","Healthy":"No Risk","Late Blight":"Critical"}

                bars = "".join([
                    f"""<div style="margin-bottom:0.55rem;">
                      <div style="display:flex;justify-content:space-between;font-size:0.78rem;color:rgba(255,255,255,0.6);margin-bottom:0.2rem;">
                        <span>{"★ " if n==disease else ""}{n}</span>
                        <span style="font-weight:600;color:{clr[n]};">{p*100:.1f}%</span>
                      </div>
                      <div style="height:5px;border-radius:3px;background:rgba(255,255,255,0.07);">
                        <div style="height:100%;width:{p*100:.1f}%;border-radius:3px;background:{clr[n]};{'box-shadow:0 0 7px '+clr[n]+';' if n==disease else ''}"></div>
                      </div>
                    </div>"""
                    for n, p in zip(class_names, probs)
                ])

                st.markdown(f"""
                <div style="border:1.5px solid {bdr_clr[disease]};border-radius:16px;overflow:hidden;">
                  <div style="background:rgba(76,175,80,0.08);padding:0.6rem 1.2rem;display:flex;align-items:center;
                              justify-content:space-between;border-bottom:1px solid rgba(255,255,255,0.05);">
                    <div style="display:flex;align-items:center;gap:0.4rem;">
                      <div style="width:5px;height:5px;border-radius:50%;background:#4caf50;box-shadow:0 0 5px #4caf50;"></div>
                      <span style="color:#a5d6a7;font-size:0.72rem;font-weight:600;text-transform:uppercase;letter-spacing:0.5px;">Stage 1 — Tomato Confirmed</span>
                    </div>
                    <span style="color:#81c784;font-size:0.75rem;font-weight:700;">{gate_conf*100:.1f}% confidence</span>
                  </div>
                  <div style="padding:1.5rem 1.3rem;text-align:center;background:rgba(0,0,0,0.15);">
                    <div style="font-size:2.8rem;margin-bottom:0.4rem;">{icons[disease]}</div>
                    <div style="font-size:1.9rem;font-weight:800;color:{lbl_clr[disease]};letter-spacing:-0.5px;margin-bottom:0.3rem;">{disease}</div>
                    <div style="display:inline-block;background:rgba(255,255,255,0.06);border-radius:20px;
                                padding:0.18rem 0.7rem;color:{lbl_clr[disease]};font-size:0.72rem;font-weight:600;">{severity[disease]}</div>
                  </div>
                  <div style="padding:0.9rem 1.3rem 1.2rem;border-top:1px solid rgba(255,255,255,0.05);">
                    <div style="font-size:0.68rem;font-weight:600;color:rgba(255,255,255,0.28);text-transform:uppercase;letter-spacing:0.6px;margin-bottom:0.7rem;">Class Probabilities</div>
                    {bars}
                  </div>
                </div>
                """, unsafe_allow_html=True)

            else:
                st.markdown(f"""
                <div style="border:1.5px solid #5c6bc0;border-radius:16px;overflow:hidden;">
                  <div style="background:rgba(92,107,192,0.08);padding:0.6rem 1.2rem;display:flex;align-items:center;
                              justify-content:space-between;border-bottom:1px solid rgba(255,255,255,0.05);">
                    <div style="display:flex;align-items:center;gap:0.4rem;">
                      <div style="width:5px;height:5px;border-radius:50%;background:#7986cb;box-shadow:0 0 5px #7986cb;"></div>
                      <span style="color:#9fa8da;font-size:0.72rem;font-weight:600;text-transform:uppercase;letter-spacing:0.5px;">Stage 1 — Not a Tomato</span>
                    </div>
                    <span style="color:#9fa8da;font-size:0.75rem;font-weight:700;">{gate_conf*100:.1f}% confidence</span>
                  </div>
                  <div style="padding:2.2rem 1.3rem;text-align:center;">
                    <div style="font-size:2.5rem;margin-bottom:0.7rem;">🚫</div>
                    <div style="font-size:1.3rem;font-weight:700;color:#9fa8da;margin-bottom:0.4rem;">No Tomato Detected</div>
                    <div style="color:rgba(255,255,255,0.35);font-size:0.82rem;line-height:1.6;">Stage 2 diagnosis skipped.</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

    # Disease reference
    st.markdown('<div class="section-header">Disease Reference</div>', unsafe_allow_html=True)
    ref1, ref2, ref3 = st.columns(3)

    with ref1:
        st.markdown("""
        <div style="border-radius:14px;overflow:hidden;border:1px solid rgba(239,83,80,0.22);background:linear-gradient(160deg,#1a0808,#150505);">
          <div style="background:linear-gradient(90deg,rgba(239,83,80,0.22),transparent);padding:0.9rem 1.2rem;display:flex;align-items:center;gap:0.6rem;">
            <span style="font-size:1.2rem;">🔴</span>
            <div>
              <div style="color:#ef9a9a;font-size:0.92rem;font-weight:700;">Early Blight</div>
              <div style="color:rgba(239,154,154,0.45);font-size:0.68rem;font-style:italic;">Alternaria solani</div>
            </div>
            <div style="margin-left:auto;background:rgba(239,83,80,0.18);border-radius:5px;padding:0.15rem 0.5rem;color:#ef9a9a;font-size:0.65rem;font-weight:700;">HIGH RISK</div>
          </div>
          <div style="padding:0.9rem 1.2rem;">
            <div style="color:rgba(255,255,255,0.5);font-size:0.8rem;line-height:1.65;margin-bottom:0.8rem;">
              Dark brown spots with concentric rings on older leaves — a distinctive target-board pattern. Can cause heavy defoliation and significant yield loss.
            </div>
            <div style="display:flex;gap:0.3rem;flex-wrap:wrap;margin-bottom:0.8rem;">
              <span style="background:rgba(239,83,80,0.1);border:1px solid rgba(239,83,80,0.2);border-radius:20px;padding:0.1rem 0.5rem;font-size:0.67rem;color:#ef9a9a;">Brown spots</span>
              <span style="background:rgba(239,83,80,0.1);border:1px solid rgba(239,83,80,0.2);border-radius:20px;padding:0.1rem 0.5rem;font-size:0.67rem;color:#ef9a9a;">Concentric rings</span>
            </div>
            <div style="border-top:1px solid rgba(255,255,255,0.05);padding-top:0.7rem;color:#ef9a9a;font-size:0.77rem;">
              Copper fungicides · Crop rotation
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with ref2:
        st.markdown("""
        <div style="border-radius:14px;overflow:hidden;border:1px solid rgba(76,175,80,0.22);background:linear-gradient(160deg,#071a09,#050e06);">
          <div style="background:linear-gradient(90deg,rgba(76,175,80,0.22),transparent);padding:0.9rem 1.2rem;display:flex;align-items:center;gap:0.6rem;">
            <span style="font-size:1.2rem;">🟢</span>
            <div>
              <div style="color:#a5d6a7;font-size:0.92rem;font-weight:700;">Healthy</div>
              <div style="color:rgba(165,214,167,0.45);font-size:0.68rem;font-style:italic;">No pathogen detected</div>
            </div>
            <div style="margin-left:auto;background:rgba(76,175,80,0.18);border-radius:5px;padding:0.15rem 0.5rem;color:#a5d6a7;font-size:0.65rem;font-weight:700;">NO RISK</div>
          </div>
          <div style="padding:0.9rem 1.2rem;">
            <div style="color:rgba(255,255,255,0.5);font-size:0.8rem;line-height:1.65;margin-bottom:0.8rem;">
              Vibrant green leaves, no lesions or discolouration. Optimal photosynthesis and growth. Continue regular monitoring to catch disease early.
            </div>
            <div style="display:flex;gap:0.3rem;flex-wrap:wrap;margin-bottom:0.8rem;">
              <span style="background:rgba(76,175,80,0.1);border:1px solid rgba(76,175,80,0.2);border-radius:20px;padding:0.1rem 0.5rem;font-size:0.67rem;color:#a5d6a7;">No lesions</span>
              <span style="background:rgba(76,175,80,0.1);border:1px solid rgba(76,175,80,0.2);border-radius:20px;padding:0.1rem 0.5rem;font-size:0.67rem;color:#a5d6a7;">Vibrant colour</span>
            </div>
            <div style="border-top:1px solid rgba(255,255,255,0.05);padding-top:0.7rem;color:#a5d6a7;font-size:0.77rem;">
              No intervention needed · Continue monitoring
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with ref3:
        st.markdown("""
        <div style="border-radius:14px;overflow:hidden;border:1px solid rgba(255,167,38,0.22);background:linear-gradient(160deg,#1a0f00,#150c00);">
          <div style="background:linear-gradient(90deg,rgba(255,167,38,0.22),transparent);padding:0.9rem 1.2rem;display:flex;align-items:center;gap:0.6rem;">
            <span style="font-size:1.2rem;">🟠</span>
            <div>
              <div style="color:#ffcc80;font-size:0.92rem;font-weight:700;">Late Blight</div>
              <div style="color:rgba(255,204,128,0.45);font-size:0.68rem;font-style:italic;">Phytophthora infestans</div>
            </div>
            <div style="margin-left:auto;background:rgba(255,167,38,0.18);border-radius:5px;padding:0.15rem 0.5rem;color:#ffcc80;font-size:0.65rem;font-weight:700;">CRITICAL</div>
          </div>
          <div style="padding:0.9rem 1.2rem;">
            <div style="color:rgba(255,255,255,0.5);font-size:0.8rem;line-height:1.65;margin-bottom:0.8rem;">
              Water-soaked patches turning dark brown-black. Spreads explosively in cool, humid conditions — the pathogen behind the Irish Potato Famine.
            </div>
            <div style="display:flex;gap:0.3rem;flex-wrap:wrap;margin-bottom:0.8rem;">
              <span style="background:rgba(255,167,38,0.1);border:1px solid rgba(255,167,38,0.2);border-radius:20px;padding:0.1rem 0.5rem;font-size:0.67rem;color:#ffcc80;">Dark lesions</span>
              <span style="background:rgba(255,167,38,0.1);border:1px solid rgba(255,167,38,0.2);border-radius:20px;padding:0.1rem 0.5rem;font-size:0.67rem;color:#ffcc80;">Rapid spread</span>
            </div>
            <div style="border-top:1px solid rgba(255,255,255,0.05);padding-top:0.7rem;color:#ffcc80;font-size:0.77rem;">
              Remove infected plants · Systemic fungicides immediately
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 · ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
with tab_analytics:

    # Gate Model
    st.markdown('<div class="section-header">Stage 1 · Gate Model</div>', unsafe_allow_html=True)
    g1, g2 = st.columns([3,2], gap="large")
    with g1:
        st.plotly_chart(training_curves_fig(
            GATE_P1_TRAIN_LOSS, GATE_P1_VAL_LOSS, GATE_P1_TRAIN_ACC, GATE_P1_VAL_ACC,
            GATE_P2_TRAIN_LOSS, GATE_P2_VAL_LOSS, GATE_P2_TRAIN_ACC, GATE_P2_VAL_ACC,
            title="Gate Model — Training History"
        ), use_container_width=True)
    with g2:
        st.plotly_chart(confusion_matrix_fig(GATE_CM, GATE_CLASSES, "Confusion Matrix (Test)"),
                        use_container_width=True)

    gm1, gm2, gm3, gm4 = st.columns(4)
    with gm1: st.metric("Test Accuracy",   "99.93%", delta="+0.07% vs baseline")
    with gm2: st.metric("Test Loss",       "0.0027")
    with gm3: st.metric("Precision (avg)", "100.0%")
    with gm4: st.metric("Recall (avg)",    "100.0%")

    st.markdown("<br>", unsafe_allow_html=True)

    # Disease Classifier
    st.markdown('<div class="section-header">Stage 2 · Disease Classifier</div>', unsafe_allow_html=True)
    d1, d2 = st.columns([3,2], gap="large")
    with d1:
        st.plotly_chart(training_curves_fig(
            DISEASE_P1_TRAIN_LOSS, DISEASE_P1_VAL_LOSS, DISEASE_P1_TRAIN_ACC, DISEASE_P1_VAL_ACC,
            DISEASE_P2_TRAIN_LOSS, DISEASE_P2_VAL_LOSS, DISEASE_P2_TRAIN_ACC, DISEASE_P2_VAL_ACC,
            title="Disease Classifier — Training History"
        ), use_container_width=True)
    with d2:
        st.plotly_chart(confusion_matrix_fig(DISEASE_CM, DISEASE_CLASSES, "Confusion Matrix (Test)"),
                        use_container_width=True)

    pc1, pc2 = st.columns([2,1], gap="large")
    with pc1:
        st.plotly_chart(per_class_bar(
            DISEASE_CLASSES,
            precision=[0.98,0.95,0.96],
            recall=[0.89,1.00,0.97],
            f1=[0.93,0.98,0.97],
            title="Per-Class Precision / Recall / F1"
        ), use_container_width=True)
    with pc2:
        st.markdown("**Classification Report**")
        df = pd.DataFrame({
            "Class":     ["Early Blight","Healthy","Late Blight","weighted avg"],
            "Precision": ["0.98","0.95","0.96","0.96"],
            "Recall":    ["0.89","1.00","0.97","0.96"],
            "F1":        ["0.93","0.98","0.97","0.96"],
            "n":         ["150","240","287","677"],
        })
        st.dataframe(df.style.map(
            lambda v: "color:#81c784" if v in ("1.00","0.98","0.97") else
                      "color:#ef9a9a" if v=="0.89" else ""
        ), hide_index=True, use_container_width=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.metric("Test Accuracy", "96.16%", delta="+1.2% from Phase 1")
        st.metric("Test Loss",     "0.1212")
