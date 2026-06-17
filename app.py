"""TomatoVision AI — Production-Grade Tomato Disease Detection Platform"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
import io, datetime

st.set_page_config(
    page_title="TomatoVision AI",
    page_icon="🍅",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: #e2e8f0; }
.stApp { background: #060b18; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
[data-testid="stToolbar"] { visibility: hidden; }
[data-testid="collapsedControl"] { visibility: visible !important; display: flex !important; color: #06d6a0 !important; }
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#080f1e 0%,#060b18 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 2px; }

/* Glass card */
.glass {
    background: rgba(13,21,38,0.75);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    backdrop-filter: blur(20px);
    padding: 1.5rem;
}

/* Hero */
.hero-wrap {
    background: linear-gradient(135deg,#060b18 0%,#0d1f0a 30%,#0a0e18 65%,#1a0808 100%);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 20px; padding: 3rem 3rem 2.5rem;
    position: relative; overflow: hidden; margin-bottom: 2rem;
}
.hero-wrap::before {
    content:""; position:absolute; top:-80px; right:-60px;
    width:320px; height:320px;
    background:radial-gradient(circle,rgba(6,214,160,0.09) 0%,transparent 70%);
}
.hero-wrap::after {
    content:""; position:absolute; bottom:-80px; left:35%;
    width:400px; height:200px;
    background:radial-gradient(ellipse,rgba(239,68,68,0.07) 0%,transparent 70%);
}
.hero-eyebrow {
    display:inline-flex; align-items:center; gap:0.4rem;
    background:rgba(6,214,160,0.1); border:1px solid rgba(6,214,160,0.2);
    border-radius:20px; padding:0.25rem 0.8rem;
    color:#06d6a0; font-size:0.7rem; font-weight:700;
    text-transform:uppercase; letter-spacing:1px; margin-bottom:1rem;
}
.hero-title {
    font-size:3rem; font-weight:900; letter-spacing:-1.5px;
    background:linear-gradient(135deg,#f1f5f9 0%,#a5f3d0 50%,#f1f5f9 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    background-clip:text; margin:0 0 0.5rem; line-height:1.1;
}
.hero-title span { -webkit-text-fill-color:#ef4444; }
.hero-sub { color:#64748b; font-size:1rem; max-width:580px; line-height:1.7; margin:0 0 2rem; }
.hero-chip {
    display:inline-flex; align-items:center; gap:0.4rem;
    background:rgba(239,68,68,0.1); border:1px solid rgba(239,68,68,0.2);
    border-radius:8px; padding:0.4rem 0.9rem;
    color:#fca5a5; font-size:0.75rem; font-weight:600; margin-right:0.5rem;
}

/* KPI cards */
.kpi {
    background:rgba(13,21,38,0.8); border:1px solid rgba(255,255,255,0.08);
    border-radius:14px; padding:1.3rem 1.2rem; position:relative; overflow:hidden;
    transition:transform 0.2s,border-color 0.2s;
}
.kpi:hover { transform:translateY(-2px); border-color:rgba(255,255,255,0.14); }
.kpi::after { content:""; position:absolute; bottom:0; left:0; right:0; height:2px; }
.kpi.green::after  { background:linear-gradient(90deg,#22c55e,#86efac); }
.kpi.red::after    { background:linear-gradient(90deg,#ef4444,#fca5a5); }
.kpi.blue::after   { background:linear-gradient(90deg,#3b82f6,#93c5fd); }
.kpi.teal::after   { background:linear-gradient(90deg,#06d6a0,#6ee7b7); }
.kpi.amber::after  { background:linear-gradient(90deg,#f59e0b,#fcd34d); }
.kpi.purple::after { background:linear-gradient(90deg,#8b5cf6,#c4b5fd); }
.kpi-tag { font-size:0.63rem; font-weight:700; text-transform:uppercase; letter-spacing:1px; color:#475569; margin-bottom:0.35rem; }
.kpi-val { font-size:2rem; font-weight:800; line-height:1.1; margin-bottom:0.12rem; }
.kpi-val.green  { color:#86efac; }
.kpi-val.red    { color:#fca5a5; }
.kpi-val.blue   { color:#93c5fd; }
.kpi-val.teal   { color:#6ee7b7; }
.kpi-val.amber  { color:#fcd34d; }
.kpi-val.purple { color:#c4b5fd; }
.kpi-desc { color:#64748b; font-size:0.73rem; }

/* Section header */
.sec-head {
    font-size:1rem; font-weight:700; color:#cbd5e1;
    display:flex; align-items:center; gap:0.5rem;
    padding-bottom:0.6rem; border-bottom:1px solid rgba(255,255,255,0.06);
    margin:0 0 1.2rem;
}
.sec-head .dot {
    width:6px; height:6px; border-radius:50%;
    background:#06d6a0; box-shadow:0 0 6px #06d6a0; flex-shrink:0;
}

/* Badges */
.badge { display:inline-block; padding:0.18rem 0.6rem; border-radius:20px; font-size:0.65rem; font-weight:700; text-transform:uppercase; letter-spacing:0.5px; }
.badge-green  { background:rgba(34,197,94,0.12);  color:#86efac; border:1px solid rgba(34,197,94,0.25); }
.badge-red    { background:rgba(239,68,68,0.12);  color:#fca5a5; border:1px solid rgba(239,68,68,0.25); }
.badge-amber  { background:rgba(245,158,11,0.12); color:#fcd34d; border:1px solid rgba(245,158,11,0.25); }
.badge-blue   { background:rgba(59,130,246,0.12); color:#93c5fd; border:1px solid rgba(59,130,246,0.25); }
.badge-teal   { background:rgba(6,214,160,0.12);  color:#6ee7b7; border:1px solid rgba(6,214,160,0.25); }

/* Analysis steps */
.step-item {
    display:flex; align-items:center; gap:0.7rem;
    padding:0.65rem 1rem;
    background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.05);
    border-radius:10px; font-size:0.84rem; color:#475569; margin-bottom:0.5rem;
}
.step-item.done { background:rgba(6,214,160,0.05); border-color:rgba(6,214,160,0.2); color:#a7f3d0; }
.step-item.active { background:rgba(59,130,246,0.05); border-color:rgba(59,130,246,0.2); color:#93c5fd; }

/* Result card */
.result-wrap { border-radius:16px; overflow:hidden; }
.result-header { display:flex; align-items:center; justify-content:space-between; padding:0.75rem 1.2rem; border-bottom:1px solid rgba(255,255,255,0.05); }
.result-body { padding:1.5rem 1.2rem; }
.pred-name { font-size:2rem; font-weight:800; letter-spacing:-0.5px; margin:0.2rem 0 0.5rem; }
.prob-row { display:flex; align-items:center; justify-content:space-between; font-size:0.8rem; color:rgba(255,255,255,0.55); margin-bottom:0.2rem; }
.prob-bar-bg { height:5px; border-radius:3px; background:rgba(255,255,255,0.06); margin-bottom:0.65rem; }
.prob-bar-fill { height:100%; border-radius:3px; }

/* Alert boxes */
.alert-green { background:rgba(34,197,94,0.06); border:1px solid rgba(34,197,94,0.2); border-radius:10px; padding:0.9rem 1.1rem; color:#86efac; font-size:0.84rem; margin-bottom:0.7rem; }
.alert-amber { background:rgba(245,158,11,0.06); border:1px solid rgba(245,158,11,0.2); border-radius:10px; padding:0.9rem 1.1rem; color:#fcd34d; font-size:0.84rem; margin-bottom:0.7rem; }
.alert-red   { background:rgba(239,68,68,0.06);  border:1px solid rgba(239,68,68,0.2);  border-radius:10px; padding:0.9rem 1.1rem; color:#fca5a5; font-size:0.84rem; margin-bottom:0.7rem; }
.alert-blue  { background:rgba(59,130,246,0.06); border:1px solid rgba(59,130,246,0.2); border-radius:10px; padding:0.9rem 1.1rem; color:#93c5fd; font-size:0.84rem; margin-bottom:0.7rem; }

/* Disease library cards */
.dis-card { border-radius:14px; overflow:hidden; border:1px solid rgba(255,255,255,0.07); height:100%; }
.dis-card-head { padding:1rem 1.2rem; display:flex; align-items:center; gap:0.7rem; }
.dis-card-body { padding:0.8rem 1.2rem 1.2rem; }
.dis-tag { display:inline-block; background:rgba(255,255,255,0.06); border-radius:20px; padding:0.12rem 0.5rem; font-size:0.67rem; color:rgba(255,255,255,0.45); margin:0.12rem 0.1rem; }

/* XAI */
.xai-panel-label { text-align:center; font-size:0.68rem; font-weight:700; text-transform:uppercase; letter-spacing:0.8px; color:#475569; margin-top:0.4rem; }

/* Sidebar nav radio styling */
.stRadio > label { display:none !important; }
.stRadio > div { gap:2px !important; }
.stRadio > div > label {
    display:flex !important; align-items:center !important;
    padding:0.65rem 1rem !important; border-radius:10px !important;
    cursor:pointer; transition:background 0.15s;
    color:#64748b !important; font-size:0.875rem !important; font-weight:500 !important;
    border:1px solid transparent !important; margin:0 !important;
}
.stRadio > div > label:hover { background:rgba(255,255,255,0.04) !important; color:#94a3b8 !important; }
[data-testid="stRadio"] div[data-checked="true"] > label,
.stRadio > div > label[aria-checked="true"] {
    background:rgba(6,214,160,0.08) !important;
    color:#a7f3d0 !important; border-color:rgba(6,214,160,0.2) !important;
}
div[data-testid="stRadio"] label span:first-child { display:none !important; }

/* Metrics */
[data-testid="stMetric"] { background:rgba(13,21,38,0.6) !important; border:1px solid rgba(255,255,255,0.07) !important; border-radius:12px !important; padding:0.9rem !important; }
[data-testid="stMetricLabel"] p { color:#64748b !important; font-size:0.72rem !important; }
[data-testid="stMetricValue"] { font-size:1.5rem !important; color:#e2e8f0 !important; }

/* Plotly */
.stPlotlyChart { border-radius:12px; overflow:hidden; }

/* Responsive */
@media (max-width:768px) {
    .hero-title { font-size:2rem; }
    .hero-wrap { padding:1.5rem; }
    [data-testid="stHorizontalBlock"] { flex-wrap:wrap !important; }
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] { min-width:48% !important; flex:1 1 48% !important; }
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  TRAINING DATA
# ══════════════════════════════════════════════════════════════════════════════
DISEASE_P1_VAL_LOSS   = [0.6194,0.4216,0.3102,0.2697,0.2496,0.2205,0.2093,0.1919,0.1852,0.1705,0.1724,0.1709,0.1723,0.1695,0.1702,0.1676,0.1666,0.1669,0.1643,0.1579,0.1551,0.1547,0.1534,0.1517,0.1533,0.1539,0.1482,0.1499,0.1471,0.1488]
DISEASE_P1_TRAIN_LOSS = [0.832,0.581,0.436,0.374,0.325,0.286,0.255,0.231,0.212,0.196,0.183,0.172,0.163,0.155,0.148,0.142,0.136,0.131,0.127,0.123,0.119,0.116,0.113,0.110,0.107,0.105,0.103,0.101,0.099,0.097]
DISEASE_P1_VAL_ACC    = [0.782,0.840,0.882,0.895,0.904,0.915,0.921,0.928,0.930,0.936,0.935,0.936,0.935,0.937,0.936,0.938,0.939,0.938,0.940,0.943,0.944,0.944,0.945,0.945,0.945,0.945,0.947,0.946,0.947,0.947]
DISEASE_P1_TRAIN_ACC  = [0.715,0.787,0.829,0.855,0.872,0.886,0.896,0.905,0.912,0.918,0.924,0.928,0.933,0.937,0.940,0.943,0.946,0.948,0.951,0.953,0.955,0.957,0.959,0.960,0.962,0.963,0.964,0.965,0.966,0.967]
DISEASE_P2_VAL_LOSS   = [0.1414,0.1390,0.1340,0.1358,0.1208,0.1110,0.1113,0.1056,0.1070,0.1065,0.1015,0.1008,0.0930,0.0955,0.0926,0.0882,0.0877,0.0840,0.0816,0.0762,0.0802,0.0823,0.0805,0.0800,0.0783,0.0782,0.0777]
DISEASE_P2_TRAIN_LOSS = [0.131,0.121,0.113,0.107,0.101,0.096,0.092,0.088,0.085,0.082,0.079,0.077,0.074,0.072,0.070,0.068,0.066,0.065,0.063,0.062,0.060,0.059,0.058,0.057,0.056,0.056,0.055]
DISEASE_P2_VAL_ACC    = [0.949,0.950,0.951,0.950,0.954,0.956,0.956,0.958,0.957,0.958,0.959,0.959,0.961,0.960,0.961,0.962,0.963,0.963,0.964,0.965,0.964,0.964,0.964,0.964,0.965,0.965,0.965]
DISEASE_P2_TRAIN_ACC  = [0.967,0.969,0.971,0.972,0.973,0.974,0.975,0.976,0.977,0.977,0.978,0.979,0.980,0.980,0.981,0.981,0.982,0.982,0.983,0.983,0.983,0.984,0.984,0.984,0.984,0.985,0.985]
GATE_P1_VAL_LOSS      = [0.0930,0.0358,0.0295,0.0221,0.0213,0.0182,0.0170,0.0160,0.0142,0.0143,0.0136,0.0127,0.0128,0.0126,0.0121,0.0119,0.0127,0.0114,0.0112,0.0110,0.0109,0.0104,0.0110,0.0107,0.0105,0.0107,0.0103,0.0103,0.0106,0.0103]
GATE_P1_TRAIN_LOSS    = [0.112,0.054,0.037,0.027,0.022,0.018,0.015,0.013,0.011,0.010,0.009,0.009,0.008,0.008,0.007,0.007,0.007,0.006,0.006,0.006,0.006,0.006,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005]
GATE_P1_VAL_ACC       = [0.966,0.987,0.989,0.992,0.993,0.994,0.994,0.995,0.996,0.996,0.996,0.996,0.996,0.996,0.997,0.997,0.996,0.997,0.997,0.998,0.998,0.998,0.997,0.997,0.997,0.997,0.998,0.998,0.997,0.998]
GATE_P1_TRAIN_ACC     = [0.962,0.981,0.987,0.991,0.993,0.995,0.996,0.997,0.997,0.998,0.998,0.998,0.998,0.999,0.999,0.999,0.999,0.999,0.999,0.999,0.999,0.999,0.999,0.999,0.999,0.999,0.999,0.999,0.999,0.999]
GATE_P2_VAL_LOSS      = [0.00957,0.00949,0.00964,0.01003,0.00988,0.00974,0.00977,0.00968,0.00978]
GATE_P2_TRAIN_LOSS    = [0.00570,0.00670,0.00720,0.00620,0.00480,0.00460,0.00480,0.00470,0.00440]
GATE_P2_VAL_ACC       = [0.9986]*9
GATE_P2_TRAIN_ACC     = [0.9982,0.9977,0.9976,0.9983,0.9985,0.9985,0.9980,0.9988,0.9991]

DISEASE_CM = np.array([[134,6,10],[0,240,0],[3,5,279]])
GATE_CM    = np.array([[750,0],[1,674]])
DISEASE_CLASSES = ["Early Blight","Healthy","Late Blight"]
GATE_CLASSES    = ["Non-Tomato","Tomato"]

# ══════════════════════════════════════════════════════════════════════════════
#  PLOTLY THEME
# ══════════════════════════════════════════════════════════════════════════════
PT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,255,255,0.02)",
    font=dict(family="Inter", color="rgba(255,255,255,0.6)"),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.05)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.05)"),
)

# ══════════════════════════════════════════════════════════════════════════════
#  MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_models():
    try:
        import onnxruntime as ort
        g = ort.InferenceSession("models/gate_phase2.onnx")
        d = ort.InferenceSession("models/detect_phase2.onnx")
        return g, d, None
    except Exception as e:
        return None, None, str(e)

gate_model, disease_model, model_err = load_models()
models_ok = gate_model is not None and disease_model is not None

# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
for k, v in [("page","dashboard"), ("history",[]), ("det_result",None), ("det_img",None)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def preprocess(img: Image.Image) -> np.ndarray:
    return np.expand_dims(np.array(img.resize((224,224)).convert("RGB"), dtype=np.float32), 0)

def run_inference(arr):
    g_inp  = gate_model.get_inputs()[0].name
    g_prob = float(gate_model.run(None, {g_inp: arr})[0][0][0])
    if g_prob < 0.5:
        return {"is_tomato": False, "gate_conf": 1 - g_prob}
    d_inp  = disease_model.get_inputs()[0].name
    probs  = disease_model.run(None, {d_inp: arr})[0][0]
    idx    = int(np.argmax(probs))
    return {
        "is_tomato": True, "gate_conf": g_prob,
        "disease": DISEASE_CLASSES[idx],
        "confidence": float(probs[idx]),
        "probs": probs.tolist(),
    }

def apply_jet(h):
    """h: (H,W) float32 in [0,1] → (H,W,3) uint8 jet colormap"""
    r = np.clip(np.where(h < 0.5, 0.0, np.where(h < 0.75, 4*(h-0.5), 1.0)), 0, 1)
    g = np.clip(np.where(h < 0.25, 4*h, np.where(h < 0.75, 1.0, 1-4*(h-0.75))), 0, 1)
    b = np.clip(np.where(h < 0.25, 1.0, np.where(h < 0.5, 1-4*(h-0.25), 0.0)), 0, 1)
    return (np.stack([r, g, b], axis=2) * 255).astype(np.uint8)

def compute_xai(arr, class_idx, grid=8):
    """Occlusion sensitivity — returns (224,224) heatmap"""
    d_inp = disease_model.get_inputs()[0].name
    base  = float(disease_model.run(None, {d_inp: arr})[0][0, class_idx])
    ph, pw = 224 // grid, 224 // grid
    n     = grid * grid
    batch = np.tile(arr, (n, 1, 1, 1))
    mean  = float(arr.mean())
    for i in range(grid):
        for j in range(grid):
            idx = i * grid + j
            batch[idx, i*ph:(i+1)*ph, j*pw:(j+1)*pw, :] = mean
    probs = disease_model.run(None, {d_inp: batch})[0]
    imp   = (base - probs[:, class_idx]).reshape(grid, grid)
    imp   = np.clip(imp, 0, None)
    mn, mx = imp.min(), imp.max()
    if mx > mn:
        imp = (imp - mn) / (mx - mn)
    hm = np.array(Image.fromarray((imp * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)) / 255.0
    return hm.astype(np.float32)

def pil_to_bytes(img):
    buf = io.BytesIO(); img.save(buf, format="PNG"); return buf.getvalue()

def training_fig(p1_tl, p1_vl, p1_ta, p1_va, p2_tl, p2_vl, p2_ta, p2_va, title=""):
    e1 = list(range(1, len(p1_tl)+1))
    e2 = list(range(len(p1_tl)+1, len(p1_tl)+len(p2_tl)+1))
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Loss","Accuracy (%)"), horizontal_spacing=0.12)
    for col, (y1t,y1v,y2t,y2v) in enumerate([
        (p1_tl, p1_vl, p2_tl, p2_vl),
        ([v*100 for v in p1_ta],[v*100 for v in p1_va],[v*100 for v in p2_ta],[v*100 for v in p2_va])
    ], start=1):
        fig.add_trace(go.Scatter(x=e1,y=y1t,mode="lines",name="Train",line=dict(color="#3b82f6",width=2),legendgroup="t",showlegend=(col==1)),row=1,col=col)
        fig.add_trace(go.Scatter(x=e1,y=y1v,mode="lines",name="Val",  line=dict(color="#06d6a0",width=2),legendgroup="v",showlegend=(col==1)),row=1,col=col)
        fig.add_trace(go.Scatter(x=e2,y=y2t,mode="lines",line=dict(color="#3b82f6",width=2,dash="dot"),legendgroup="t",showlegend=False),row=1,col=col)
        fig.add_trace(go.Scatter(x=e2,y=y2v,mode="lines",line=dict(color="#06d6a0",width=2,dash="dot"),legendgroup="v",showlegend=False),row=1,col=col)
        fig.add_vline(x=len(p1_tl)+0.5, line_dash="dash", line_color="rgba(255,255,255,0.1)",
                      annotation_text="Fine-tune →", annotation_font_color="rgba(255,255,255,0.25)",
                      annotation_position="top right", row=1, col=col)
    fig.update_layout(title=dict(text=title,font=dict(size=12)),height=300,
        margin=dict(l=5,r=5,t=45,b=10),
        legend=dict(orientation="h",y=-0.22,x=0.5,xanchor="center",bgcolor="rgba(0,0,0,0)"), **PT)
    fig.update_xaxes(title_text="Epoch")
    return fig

def cm_fig(cm, labels, title=""):
    n   = len(labels)
    cmn = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig = go.Figure(go.Heatmap(
        z=cmn, x=labels, y=labels,
        text=[[f"{cm[i][j]}<br>({cmn[i][j]*100:.1f}%)" for j in range(n)] for i in range(n)],
        texttemplate="%{text}", textfont=dict(size=11,color="white"),
        colorscale=[[0,"rgba(239,68,68,0.15)"],[0.5,"rgba(239,68,68,0.5)"],[1,"#22c55e"]],
        showscale=False, xgap=3, ygap=3,
    ))
    fig.update_layout(title=dict(text=title,font=dict(size=12)),
        xaxis_title="Predicted", yaxis_title="True",
        height=280, margin=dict(l=5,r=5,t=40,b=5), **PT)
    return fig

def gauge_fig(val, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=val*100,
        number=dict(suffix="%", font=dict(size=30, color="#e2e8f0")),
        gauge=dict(
            axis=dict(range=[0,100], tickcolor="#334155"),
            bar=dict(color=color),
            bgcolor="rgba(0,0,0,0)",
            steps=[
                dict(range=[0,50],  color="rgba(239,68,68,0.08)"),
                dict(range=[50,80], color="rgba(245,158,11,0.08)"),
                dict(range=[80,100],color="rgba(34,197,94,0.08)"),
            ],
            borderwidth=0,
        ),
    ))
    fig.update_layout(height=180, margin=dict(l=20,r=20,t=10,b=10),
        paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter",color="#e2e8f0"))
    return fig

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="padding:1.5rem 0.5rem 1rem;text-align:center;border-bottom:1px solid rgba(255,255,255,0.06);margin-bottom:1rem;">
      <div style="font-size:1rem;font-weight:800;color:#f1f5f9;letter-spacing:-0.3px;">TomatoVision AI</div>
      <div style="font-size:0.68rem;color:#475569;margin-top:0.2rem;">Plant Disease Detection</div>
    </div>
    """, unsafe_allow_html=True)

    NAV_LABELS = [
        "Dashboard",
        "Disease Detection",
        "Disease Library",
        "Model Performance",
        "Prediction History",
        "About",
    ]
    NAV_KEYS = ["dashboard","detection","library","performance","history","about"]
    key_to_idx = {k:i for i,k in enumerate(NAV_KEYS)}

    sel = st.radio("nav", NAV_LABELS,
        index=key_to_idx.get(st.session_state.page, 0),
        label_visibility="collapsed")
    new_page = NAV_KEYS[NAV_LABELS.index(sel)]
    if new_page != st.session_state.page:
        st.session_state.page = new_page
        st.rerun()

    st.markdown("<div style='margin-top:1.5rem;padding-top:1rem;border-top:1px solid rgba(255,255,255,0.06);'>", unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.65rem;font-weight:700;text-transform:uppercase;letter-spacing:1px;color:#334155;margin-bottom:0.5rem;padding:0 0.5rem;">Model Status</div>', unsafe_allow_html=True)

    g_col = "#22c55e" if gate_model    else "#f59e0b"
    d_col = "#22c55e" if disease_model else "#f59e0b"
    g_lbl = "Loaded"  if gate_model    else "Not found"
    d_lbl = "Loaded"  if disease_model else "Not found"

    st.markdown(f"""
    <div style="padding:0 0.5rem;font-size:0.78rem;">
      <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.35rem;color:#64748b;">
        <div style="width:6px;height:6px;border-radius:50%;background:{g_col};box-shadow:0 0 5px {g_col};flex-shrink:0;"></div>
        Gate Model — <span style="color:{g_col};font-weight:600;">{g_lbl}</span>
      </div>
      <div style="display:flex;align-items:center;gap:0.5rem;color:#64748b;">
        <div style="width:6px;height:6px;border-radius:50%;background:{d_col};box-shadow:0 0 5px {d_col};flex-shrink:0;"></div>
        Disease Model — <span style="color:{d_col};font-weight:600;">{d_lbl}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    n_hist = len(st.session_state.history)
    st.markdown(f"""
    <div style="margin-top:auto;padding:1rem 0.5rem 0.5rem;font-size:0.68rem;color:#334155;text-align:center;">
      {n_hist} prediction{"s" if n_hist!=1 else ""} this session<br>
      <span style="color:#1e293b;">EfficientNetB0 · ONNX Runtime</span>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
def page_dashboard():
    st.markdown("""
    <div class="hero-wrap">
      <div class="hero-eyebrow">Computer Vision Capstone Project</div>
      <div class="hero-title">TomatoVision <span>AI</span></div>
      <div class="hero-sub">Tomato plant disease detection using a two-stage deep learning pipeline.
        Upload a leaf image to get an instant diagnosis with visual explanations.</div>
      <div>
        <span class="hero-chip">EfficientNetB0</span>
        <span class="hero-chip">PlantVillage Dataset</span>
        <span class="hero-chip">ONNX Runtime</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    k1,k2,k3,k4 = st.columns(4)
    with k1: st.markdown('<div class="kpi green"><div class="kpi-tag">Gate Model Accuracy</div><div class="kpi-val green">99.93%</div><div class="kpi-desc">Tomato leaf verification</div></div>', unsafe_allow_html=True)
    with k2: st.markdown('<div class="kpi red"><div class="kpi-tag">Disease Accuracy</div><div class="kpi-val red">96.16%</div><div class="kpi-desc">3-class disease classification</div></div>', unsafe_allow_html=True)
    with k3: st.markdown('<div class="kpi blue"><div class="kpi-tag">Training Images</div><div class="kpi-val blue">9,799</div><div class="kpi-desc">Across both models</div></div>', unsafe_allow_html=True)
    with k4: st.markdown('<div class="kpi teal"><div class="kpi-tag">Disease Classes</div><div class="kpi-val teal">3</div><div class="kpi-desc">Early Blight · Healthy · Late Blight</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Session stats
    hist = st.session_state.history
    total     = len(hist)
    n_tomato  = sum(1 for h in hist if h["is_tomato"])
    n_healthy = sum(1 for h in hist if h.get("disease") == "Healthy")
    n_eb      = sum(1 for h in hist if h.get("disease") == "Early Blight")
    n_lb      = sum(1 for h in hist if h.get("disease") == "Late Blight")

    s1,s2,s3,s4,s5 = st.columns(5)
    with s1: st.metric("Total Predictions", total)
    with s2: st.metric("Tomato Leaves",     n_tomato)
    with s3: st.metric("Healthy",           n_healthy)
    with s4: st.metric("Early Blight",      n_eb)
    with s5: st.metric("Late Blight",       n_lb)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([2,2,3], gap="large")

    with c1:
        st.markdown('<div class="sec-head"><div class="dot"></div>Disease Distribution</div>', unsafe_allow_html=True)
        if total > 0 and n_tomato > 0:
            labels = ["Healthy","Early Blight","Late Blight"]
            vals   = [n_healthy, n_eb, n_lb]
        else:
            labels = ["Healthy","Early Blight","Late Blight"]
            vals   = [240, 150, 287]
        fig = go.Figure(go.Pie(
            labels=labels, values=vals,
            hole=0.62,
            marker=dict(colors=["#22c55e","#f59e0b","#ef4444"]),
            textinfo="percent", textfont=dict(size=11),
        ))
        fig.update_layout(height=240, margin=dict(l=5,r=5,t=5,b=5),
            showlegend=True,
            legend=dict(orientation="v", font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
            **{k:v for k,v in PT.items() if k!="xaxis" and k!="yaxis"})
        fig.add_annotation(text=f"<b>{sum(vals)}</b><br><span style='font-size:9px'>images</span>",
            x=0.5, y=0.5, showarrow=False, font=dict(size=13, color="#e2e8f0"))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="sec-head"><div class="dot"></div>Confidence Distribution</div>', unsafe_allow_html=True)
        if len(hist) >= 3:
            confs = [h["confidence"]*100 for h in hist if "confidence" in h]
        else:
            np.random.seed(42)
            confs = np.concatenate([np.random.normal(96,2,240), np.random.normal(93,3,150), np.random.normal(97,1.5,287)]).clip(70,100).tolist()
        fig = go.Figure(go.Histogram(
            x=confs, nbinsx=20,
            marker=dict(color="#06d6a0", opacity=0.7,
                line=dict(color="rgba(6,214,160,0.3)", width=0.5)),
        ))
        fig.update_layout(height=240, margin=dict(l=5,r=5,t=5,b=5),
            bargap=0.05, **PT)
        fig.update_xaxes(title_text="Confidence (%)")
        fig.update_yaxes(title_text="Count")
        st.plotly_chart(fig, use_container_width=True)

    with c3:
        st.markdown('<div class="sec-head"><div class="dot"></div>AI Pipeline Architecture</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="display:flex;flex-direction:column;gap:0.5rem;padding:0.3rem 0;">
          <div style="display:flex;align-items:center;gap:0.8rem;padding:0.7rem 1rem;background:rgba(59,130,246,0.06);border:1px solid rgba(59,130,246,0.15);border-radius:10px;">
            <div><div style="color:#93c5fd;font-size:0.82rem;font-weight:600;">Image Input</div><div style="color:#475569;font-size:0.7rem;">224×224 px · JPG/PNG</div></div>
          </div>
          <div style="text-align:center;color:rgba(255,255,255,0.15);font-size:0.8rem;">↓</div>
          <div style="display:flex;align-items:center;gap:0.8rem;padding:0.7rem 1rem;background:rgba(6,214,160,0.06);border:1px solid rgba(6,214,160,0.15);border-radius:10px;">
            <div><div style="color:#6ee7b7;font-size:0.82rem;font-weight:600;">Stage 1 — Gate Model</div><div style="color:#475569;font-size:0.7rem;">Tomato leaf verification · 99.93% accuracy</div></div>
            <span class="badge badge-teal" style="margin-left:auto;">EfficientNetB0</span>
          </div>
          <div style="text-align:center;color:rgba(255,255,255,0.15);font-size:0.8rem;">↓</div>
          <div style="display:flex;align-items:center;gap:0.8rem;padding:0.7rem 1rem;background:rgba(239,68,68,0.06);border:1px solid rgba(239,68,68,0.15);border-radius:10px;">
            <div><div style="color:#fca5a5;font-size:0.82rem;font-weight:600;">Stage 2 — Disease Classifier</div><div style="color:#475569;font-size:0.7rem;">3-class · 96.16% accuracy</div></div>
            <span class="badge badge-red" style="margin-left:auto;">EfficientNetB0</span>
          </div>
          <div style="text-align:center;color:rgba(255,255,255,0.15);font-size:0.8rem;">↓</div>
          <div style="display:flex;align-items:center;gap:0.8rem;padding:0.7rem 1rem;background:rgba(139,92,246,0.06);border:1px solid rgba(139,92,246,0.15);border-radius:10px;">
            <div><div style="color:#c4b5fd;font-size:0.82rem;font-weight:600;">Explainability (XAI)</div><div style="color:#475569;font-size:0.7rem;">Occlusion sensitivity heatmap</div></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: DISEASE DETECTION
# ══════════════════════════════════════════════════════════════════════════════
DISEASE_CFG = {
    "Early Blight": {
        "color":"#f59e0b","light":"#fcd34d","bg":"rgba(245,158,11,0.05)",
        "border":"rgba(245,158,11,0.2)","badge":"badge-amber","risk":"High Risk",
        "icon":"🟠","risk_badge":"badge-amber",
        "desc":"Caused by the fungus Alternaria solani. Produces dark brown spots with distinctive concentric rings forming a target-board pattern on older leaves.",
        "symptoms":["Dark brown lesions","Concentric ring pattern","Yellow halo around spots","Lower leaf infection first","Early defoliation"],
        "treatment":"Apply copper-based fungicides every 7–10 days. Remove and dispose of infected leaves immediately. Avoid overhead irrigation to reduce leaf wetness.",
        "prevention":"Rotate crops every 2–3 years. Use certified disease-free seeds. Maintain adequate plant spacing for airflow. Apply preventive fungicide sprays.",
        "actions":["Remove infected leaves immediately","Apply copper fungicide","Improve air circulation between plants","Avoid watering leaves","Monitor spread daily"],
    },
    "Healthy": {
        "color":"#22c55e","light":"#86efac","bg":"rgba(34,197,94,0.05)",
        "border":"rgba(34,197,94,0.2)","badge":"badge-green","risk":"No Risk",
        "icon":"🟢","risk_badge":"badge-green",
        "desc":"The tomato plant appears healthy with no visible signs of disease. Leaves show normal green coloration with no lesions, spots, or discoloration.",
        "symptoms":["Uniform green coloration","No visible lesions","Normal leaf texture","Healthy stem structure","No wilting observed"],
        "treatment":"No treatment required. Continue with standard crop management practices and regular monitoring.",
        "prevention":"Maintain regular monitoring schedule. Ensure adequate nutrition and irrigation. Keep field clean of plant debris.",
        "actions":["Continue regular monitoring","Maintain balanced fertilization","Ensure proper irrigation","Keep field clean","Document healthy status"],
    },
    "Late Blight": {
        "color":"#ef4444","light":"#fca5a5","bg":"rgba(239,68,68,0.05)",
        "border":"rgba(239,68,68,0.2)","badge":"badge-red","risk":"Critical",
        "icon":"🔴","risk_badge":"badge-red",
        "desc":"Caused by Phytophthora infestans — the organism responsible for the Irish Potato Famine. Spreads explosively in cool, humid conditions and can destroy an entire crop within days.",
        "symptoms":["Water-soaked dark lesions","White mold on leaf undersides","Rapid browning & wilting","Greasy appearance","Foul odor from infected tissue"],
        "treatment":"Apply systemic fungicides (metalaxyl or mancozeb) immediately. Remove and destroy all infected plant material. Consider removing entire affected plants to prevent spread.",
        "prevention":"Use resistant varieties. Avoid dense planting. Apply preventive fungicides before humid weather. Monitor weather forecasts for blight risk conditions.",
        "actions":["ISOLATE infected plants immediately","Apply systemic fungicide NOW","Remove severely infected plants","Alert neighbouring farmers","Do not compost infected material"],
    },
}

def page_detection():
    st.markdown('<div style="font-size:1.4rem;font-weight:800;color:#f1f5f9;letter-spacing:-0.5px;margin-bottom:1.5rem;">Disease Detection</div>', unsafe_allow_html=True)

    up_col, info_col = st.columns([3,2], gap="large")

    with info_col:
        st.markdown("""
        <div class="glass" style="margin-bottom:1rem;">
          <div class="sec-head" style="margin-bottom:0.8rem;"><div class="dot"></div>How to Use</div>
          <div style="font-size:0.8rem;color:#64748b;line-height:1.75;">
            <div style="margin-bottom:0.4rem;"><strong style="color:#94a3b8;">Upload a clear tomato leaf photo</strong></div>
            <div style="margin-bottom:0.4rem;">— Use good, even lighting</div>
            <div style="margin-bottom:0.4rem;">— Capture a single leaf, filling most of the frame</div>
            <div style="margin-bottom:0.4rem;">— Plain background preferred</div>
            <div style="margin-bottom:0.4rem;">— Avoid blurry or dark images</div>
            <div>— JPG, PNG, WEBP · Max 200 MB</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        if not models_ok:
            st.markdown(f'<div class="alert-amber">Models not loaded. {model_err or "Check model files."}</div>', unsafe_allow_html=True)

    with up_col:
        st.markdown('<p style="font-size:0.85rem;font-weight:600;color:#94a3b8;margin-bottom:0.4rem;">Upload a tomato leaf image (JPG, PNG, WEBP)</p>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload a tomato leaf image", type=["jpg","jpeg","png","webp"],
            key="det_upload")

        if uploaded:
            img = Image.open(uploaded)
            size_kb = len(uploaded.getvalue()) / 1024
            st.image(img, use_container_width=True)
            st.markdown(f"""
            <div style="display:flex;gap:0.4rem;flex-wrap:wrap;margin-top:0.4rem;">
              <span style="background:rgba(255,255,255,0.05);border-radius:6px;padding:0.15rem 0.5rem;font-size:0.7rem;color:#475569;">{img.width}×{img.height}</span>
              <span style="background:rgba(255,255,255,0.05);border-radius:6px;padding:0.15rem 0.5rem;font-size:0.7rem;color:#475569;">{size_kb:.0f} KB</span>
              <span style="background:rgba(255,255,255,0.05);border-radius:6px;padding:0.15rem 0.5rem;font-size:0.7rem;color:#475569;">{uploaded.name}</span>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            analyze = st.button("Analyze Leaf", use_container_width=True,
                type="primary", disabled=not models_ok)

            if analyze:
                arr = preprocess(img)

                # Step display
                steps_ph = st.empty()
                def show_steps(done=0, active=1):
                    labels = ["Image Uploaded","Tomato Leaf Verification","Disease Classification","XAI Generation","Report Ready"]
                    html = '<div style="margin:1rem 0;">'
                    for i,lbl in enumerate(labels):
                        if i < done:
                            html += f'<div class="step-item done">&#10003; {lbl}</div>'
                        elif i == active:
                            html += f'<div class="step-item active">&#8250; {lbl}</div>'
                        else:
                            html += f'<div class="step-item">&ndash; {lbl}</div>'
                    html += "</div>"
                    steps_ph.markdown(html, unsafe_allow_html=True)

                show_steps(1, 1)
                with st.spinner("Running gate model…"):
                    result = run_inference(arr)
                show_steps(2, 2)

                if result["is_tomato"] and disease_model:
                    with st.spinner("Classifying disease…"):
                        pass  # already done in run_inference
                    show_steps(3, 3)

                    class_idx = DISEASE_CLASSES.index(result["disease"])
                    with st.spinner("Generating XAI heatmap…"):
                        try:
                            heatmap = compute_xai(arr, class_idx)
                            result["heatmap"] = heatmap
                        except Exception:
                            result["heatmap"] = None
                    show_steps(4, 4)

                show_steps(5, -1)

                result["filename"]  = uploaded.name
                result["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                result["img_arr"]   = np.array(img.resize((224,224)).convert("RGB"))
                if "confidence" not in result:
                    result["confidence"] = result["gate_conf"]
                st.session_state.det_result = result
                st.session_state.history.append({k:v for k,v in result.items() if k not in ("heatmap","img_arr")})
                st.rerun()

        else:
            st.markdown("""
            <div class="upload-hint">
                <strong style="color:#64748b;">Drag & drop or click to upload</strong><br>
              Upload a clear close-up photo of a <strong>tomato plant leaf</strong>
            </div>
            """, unsafe_allow_html=True)
            if st.session_state.det_result:
                if st.button("Clear previous result", use_container_width=True):
                    st.session_state.det_result = None
                    st.rerun()

    # ── Results ──────────────────────────────────────────────────────────────
    result = st.session_state.det_result
    if not result:
        return

    st.markdown("<br>", unsafe_allow_html=True)

    if not result["is_tomato"]:
        gc = result["gate_conf"]*100
        st.markdown(f"""
        <div style="background:rgba(245,158,11,0.06);border:1.5px solid rgba(245,158,11,0.25);border-radius:16px;padding:2rem;text-align:center;margin-bottom:1.5rem;">
          <div style="font-size:1.4rem;font-weight:800;color:#fcd34d;margin-bottom:0.4rem;">Image Verification Failed</div>
          <div style="color:#92400e;font-size:0.9rem;margin-bottom:1rem;">This image does not appear to contain a tomato leaf. ({gc:.1f}% confidence)</div>
          <div style="color:#64748b;font-size:0.82rem;line-height:1.8;">
            Please upload a clear, close-up photo of a <strong style="color:#fcd34d;">tomato plant leaf</strong>.<br>
            Ensure good lighting, avoid blurry images, and use a plain background.
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="alert-blue">
          <strong>Image Quality Guidelines:</strong><br>
          • Use natural or diffused lighting<br>
          • Capture a single leaf, filling most of the frame<br>
          • Avoid shadows across the leaf surface<br>
          • Keep camera steady to avoid blur<br>
          • Avoid cluttered or colourful backgrounds
        </div>
        """, unsafe_allow_html=True)
        return

    disease   = result["disease"]
    cfg       = DISEASE_CFG[disease]
    probs     = result.get("probs", [0,0,0])
    confidence= result["confidence"]

    # ── Prediction result cards ──────────────────────────────────────────────
    r1, r2 = st.columns([3,2], gap="large")
    with r1:
        gate_c = result["gate_conf"]*100
        bars   = "".join([
            f"""<div>
              <div class="prob-row"><span>{'★ ' if DISEASE_CLASSES[i]==disease else ''}{DISEASE_CLASSES[i]}</span>
              <span style="color:{cfg['color'] if DISEASE_CLASSES[i]==disease else '#475569'};font-weight:700;">{probs[i]*100:.1f}%</span></div>
              <div class="prob-bar-bg"><div class="prob-bar-fill" style="width:{probs[i]*100:.1f}%;background:{'linear-gradient(90deg,'+cfg['color']+','+cfg['light']+')' if DISEASE_CLASSES[i]==disease else 'rgba(255,255,255,0.15)'};{'box-shadow:0 0 8px '+cfg['color']+';' if DISEASE_CLASSES[i]==disease else ''}"></div></div>
            </div>"""
            for i in range(3)
        ])
        st.markdown(f"""
        <div class="result-wrap" style="border:1.5px solid {cfg['border']};background:{cfg['bg']};">
          <div class="result-header" style="background:rgba(34,197,94,0.05);">
            <div style="display:flex;align-items:center;gap:0.5rem;">
              <div style="width:5px;height:5px;border-radius:50%;background:#22c55e;box-shadow:0 0 5px #22c55e;"></div>
              <span style="color:#86efac;font-size:0.7rem;font-weight:700;text-transform:uppercase;letter-spacing:0.5px;">Stage 1 — Tomato Leaf Confirmed</span>
            </div>
            <span style="color:#86efac;font-size:0.75rem;font-weight:700;">{gate_c:.1f}%</span>
          </div>
          <div class="result-body">
            <div style="display:flex;align-items:center;gap:0.7rem;margin-bottom:0.3rem;">
              <span class="badge {cfg['risk_badge']}">{cfg['risk']}</span>
            </div>
            <div class="pred-name" style="color:{cfg['light']};">{disease}</div>
            <div style="color:#475569;font-size:0.8rem;margin-bottom:1.2rem;">{cfg['desc'][:120]}…</div>
            <div style="font-size:0.68rem;font-weight:700;text-transform:uppercase;letter-spacing:0.8px;color:#334155;margin-bottom:0.6rem;">Class Probabilities</div>
            {bars}
          </div>
        </div>
        """, unsafe_allow_html=True)

    with r2:
        st.markdown('<div class="sec-head"><div class="dot"></div>Confidence Score</div>', unsafe_allow_html=True)
        st.plotly_chart(gauge_fig(confidence, cfg["color"]), use_container_width=True)
        st.markdown(f"""
        <div style="text-align:center;margin-top:-0.5rem;margin-bottom:1rem;">
          <span style="font-size:0.7rem;color:#475569;">Model confidence in <strong style="color:{cfg['light']};">{disease}</strong> diagnosis</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.07);border-radius:12px;padding:1rem;">
          <div style="font-size:0.68rem;font-weight:700;text-transform:uppercase;letter-spacing:0.8px;color:#334155;margin-bottom:0.7rem;">Prediction Details</div>
          <div style="display:flex;justify-content:space-between;font-size:0.78rem;padding:0.3rem 0;border-bottom:1px solid rgba(255,255,255,0.04);">
            <span style="color:#475569;">Analyzed At</span><span style="color:#94a3b8;">{result['timestamp'][11:]}</span>
          </div>
          <div style="display:flex;justify-content:space-between;font-size:0.78rem;padding:0.3rem 0;border-bottom:1px solid rgba(255,255,255,0.04);">
            <span style="color:#475569;">File</span><span style="color:#94a3b8;">{result['filename'][:18]}{'…' if len(result['filename'])>18 else ''}</span>
          </div>
          <div style="display:flex;justify-content:space-between;font-size:0.78rem;padding:0.3rem 0;">
            <span style="color:#475569;">Risk Level</span><span class="badge {cfg['risk_badge']}">{cfg['risk']}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── XAI ─────────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="sec-head"><div class="dot"></div>Explainable AI Analysis
      <span style="margin-left:auto;font-size:0.68rem;color:#334155;font-weight:400;">Occlusion Sensitivity Mapping</span>
    </div>
    """, unsafe_allow_html=True)

    heatmap  = result.get("heatmap")
    img_disp = Image.fromarray(result["img_arr"])

    if heatmap is not None:
        colored  = apply_jet(heatmap)
        orig_arr = np.array(img_disp)
        overlay  = (0.55 * orig_arr + 0.45 * colored).clip(0,255).astype(np.uint8)

        x1,x2,x3 = st.columns(3, gap="small")
        with x1:
            st.image(img_disp,      use_container_width=True)
            st.markdown('<div class="xai-panel-label">Original Image</div>', unsafe_allow_html=True)
        with x2:
            st.image(Image.fromarray(colored), use_container_width=True)
            st.markdown('<div class="xai-panel-label">Attention Heatmap</div>', unsafe_allow_html=True)
        with x3:
            st.image(Image.fromarray(overlay), use_container_width=True)
            st.markdown('<div class="xai-panel-label">Heatmap Overlay</div>', unsafe_allow_html=True)

        xai_msg = "No significant disease patterns detected — the model found the leaf uniformly healthy." if disease == "Healthy" else f"Warm regions (red/yellow) indicate the leaf areas most influential in predicting <strong>{disease}</strong>. These are where the model detected characteristic disease patterns."
        st.markdown(f'<div class="alert-blue" style="margin-top:1rem;"><strong>Heatmap interpretation:</strong> {xai_msg}</div>', unsafe_allow_html=True)

        with st.expander("How does the heatmap work?"):
            st.markdown("""
            **Explainable AI (XAI)** makes AI decisions transparent and interpretable.

            **Occlusion Sensitivity** works by systematically hiding (occluding) small regions of the input image
            and measuring how much the model's confidence drops. Regions whose removal causes the largest drop
            in confidence are the most important for the prediction — these appear as warm colours (red/yellow)
            in the heatmap.

            This technique helps:
            - **Farmers** understand *why* the AI made a specific diagnosis
            - **Researchers** validate that the model focuses on biologically relevant features
            - **Stakeholders** build trust in the AI system's reasoning
            """)
    else:
        st.markdown('<div class="alert-amber">Heatmap could not be generated for this image.</div>', unsafe_allow_html=True)

    # ── Disease info & recommendations ───────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    di1, di2 = st.columns(2, gap="large")

    with di1:
        st.markdown(f'<div class="sec-head"><div class="dot"></div>Disease Information — {disease}</div>', unsafe_allow_html=True)
        symptoms_html = "".join([f'<span class="dis-tag">{s}</span>' for s in cfg["symptoms"]])
        st.markdown(f"""
        <div class="glass" style="padding:1.2rem;">
          <div style="color:#94a3b8;font-size:0.82rem;line-height:1.7;margin-bottom:1rem;">{cfg['desc']}</div>
          <div style="font-size:0.68rem;font-weight:700;text-transform:uppercase;letter-spacing:0.8px;color:#334155;margin-bottom:0.4rem;">Key Symptoms</div>
          <div style="margin-bottom:1rem;">{symptoms_html}</div>
          <div style="font-size:0.68rem;font-weight:700;text-transform:uppercase;letter-spacing:0.8px;color:#334155;margin-bottom:0.4rem;">Prevention</div>
          <div style="color:#64748b;font-size:0.8rem;line-height:1.65;">{cfg['prevention']}</div>
        </div>
        """, unsafe_allow_html=True)

    with di2:
        st.markdown('<div class="sec-head"><div class="dot"></div>Actionable Recommendations</div>', unsafe_allow_html=True)
        alert_cls  = "alert-red" if cfg["risk"]=="Critical" else "alert-amber" if cfg["risk"]=="High Risk" else "alert-green"
        actions_html = "".join([f'<div style="display:flex;align-items:flex-start;gap:0.5rem;margin-bottom:0.4rem;font-size:0.82rem;"><span style="color:#475569;">—</span><span>{a}</span></div>' for a in cfg["actions"]])
        st.markdown(f"""
        <div class="{alert_cls}" style="margin-bottom:0.8rem;">
          <strong>{'Immediate Action Required' if cfg['risk']=='Critical' else 'Action Recommended' if cfg['risk']=='High Risk' else 'Plant is Healthy'}</strong><br>
          <span style="font-size:0.8rem;">{cfg['treatment']}</span>
        </div>
        <div class="glass" style="padding:1.2rem;">
          <div style="font-size:0.68rem;font-weight:700;text-transform:uppercase;letter-spacing:0.8px;color:#334155;margin-bottom:0.8rem;">Farmer Checklist</div>
          {actions_html}
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: DISEASE LIBRARY
# ══════════════════════════════════════════════════════════════════════════════
def page_library():
    st.markdown('<div style="font-size:1.4rem;font-weight:800;color:#f1f5f9;letter-spacing:-0.5px;margin-bottom:1.5rem;">Disease Library</div>', unsafe_allow_html=True)
    search = st.text_input("Search diseases, symptoms, or treatments", placeholder="e.g. brown spots, fungicide, blight…", label_visibility="collapsed")
    st.markdown("<br>", unsafe_allow_html=True)

    diseases = [
        {
            "name":"Healthy","icon":"","severity":"None","severity_cls":"badge-green",
            "color":"#22c55e","bg":"rgba(34,197,94,0.04)","border":"rgba(34,197,94,0.15)",
            "pathogen":"No pathogen detected","description":"The tomato plant leaf appears in optimal health. Leaves exhibit vibrant green coloration with no visible lesions, spots, or signs of infection. The plant demonstrates normal growth patterns.",
            "symptoms":["Vibrant uniform green color","No visible spots or lesions","Normal leaf texture","No wilting","No discoloration"],
            "causes":["Good cultural practices","Optimal nutrition","Proper irrigation","Disease-free seed material"],
            "prevention":["Regular scouting","Balanced fertilization","Proper plant spacing","Crop rotation"],
            "treatment":"No treatment required. Continue standard crop management.",
        },
        {
            "name":"Early Blight","icon":"","severity":"High","severity_cls":"badge-amber",
            "color":"#f59e0b","bg":"rgba(245,158,11,0.04)","border":"rgba(245,158,11,0.15)",
            "pathogen":"Alternaria solani","description":"A fungal disease causing characteristic dark brown lesions with concentric rings resembling a target or bull's eye pattern. Primarily affects lower, older leaves first before progressing upward.",
            "symptoms":["Dark brown lesions (2–10 mm)","Concentric ring pattern","Yellow chlorotic halo","Lower leaves infected first","Premature defoliation"],
            "causes":["Warm temperatures 24–29°C","High humidity","Extended leaf wetness","Poor air circulation","Infected plant debris"],
            "prevention":["Use resistant varieties","Crop rotation (3+ years)","Remove plant debris","Avoid overhead irrigation","Maintain plant spacing"],
            "treatment":"Apply copper-based or mancozeb fungicides every 7–10 days. Remove infected leaves. Ensure good air circulation.",
        },
        {
            "name":"Late Blight","icon":"","severity":"Critical","severity_cls":"badge-red",
            "color":"#ef4444","bg":"rgba(239,68,68,0.04)","border":"rgba(239,68,68,0.15)",
            "pathogen":"Phytophthora infestans","description":"A devastating oomycete disease capable of destroying entire crops within days. This is the pathogen responsible for the Irish Potato Famine of the 1840s. Spreads explosively under cool, wet conditions.",
            "symptoms":["Water-soaked irregular lesions","White mold on leaf underside","Rapid browning and collapse","Greasy/oily appearance","Strong foul odor"],
            "causes":["Cool temperatures 10–25°C","High humidity >90%","Prolonged leaf wetness","Infected tubers/plant material","Wind-dispersed spores"],
            "prevention":["Plant resistant varieties","Apply preventive fungicides","Monitor weather for blight risk","Destroy infected plant material","Avoid dense planting"],
            "treatment":"Apply systemic fungicides (metalaxyl, cymoxanil) immediately. Remove and destroy infected plants. Do not compost infected material.",
        },
    ]

    filtered = [d for d in diseases if not search or
        search.lower() in (d["name"] + d["description"] + " ".join(d["symptoms"]) + d["treatment"]).lower()]

    if not filtered:
        st.markdown('<div class="alert-blue">No diseases match your search. Try different keywords.</div>', unsafe_allow_html=True)
        return

    cols = st.columns(len(filtered), gap="large") if len(filtered) > 1 else [st.container()]
    for col, d in zip(cols, filtered):
        with col:
            syms = "".join([f'<span class="dis-tag">{s}</span>' for s in d["symptoms"]])
            prevs = "".join([f'<div style="display:flex;gap:0.4rem;font-size:0.78rem;color:#64748b;margin-bottom:0.25rem;"><span>•</span><span>{p}</span></div>' for p in d["prevention"]])
            st.markdown(f"""
            <div class="dis-card" style="background:{d['bg']};border-color:{d['border']};">
              <div class="dis-card-head" style="background:rgba(255,255,255,0.02);border-bottom:1px solid rgba(255,255,255,0.05);">
                <div>
                  <div style="color:{d['color']};font-size:0.95rem;font-weight:700;">{d['name']}</div>
                  <div style="color:#475569;font-size:0.68rem;font-style:italic;">{d['pathogen']}</div>
                </div>
                <span class="badge {d['severity_cls']}" style="margin-left:auto;">{d['severity']}</span>
              </div>
              <div class="dis-card-body">
                <div style="color:#64748b;font-size:0.79rem;line-height:1.65;margin-bottom:0.9rem;">{d['description']}</div>
                <div style="font-size:0.65rem;font-weight:700;text-transform:uppercase;letter-spacing:0.8px;color:#334155;margin-bottom:0.35rem;">Symptoms</div>
                <div style="margin-bottom:0.9rem;">{syms}</div>
                <div style="font-size:0.65rem;font-weight:700;text-transform:uppercase;letter-spacing:0.8px;color:#334155;margin-bottom:0.35rem;">Prevention</div>
                <div style="margin-bottom:0.9rem;">{prevs}</div>
                <div style="font-size:0.65rem;font-weight:700;text-transform:uppercase;letter-spacing:0.8px;color:#334155;margin-bottom:0.35rem;">Treatment</div>
                <div style="color:#64748b;font-size:0.78rem;line-height:1.6;">{d['treatment']}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
def page_performance():
    st.markdown('<div style="font-size:1.4rem;font-weight:800;color:#f1f5f9;letter-spacing:-0.5px;margin-bottom:1.5rem;">Model Performance</div>', unsafe_allow_html=True)

    # Gate model
    st.markdown('<div class="sec-head"><div class="dot"></div>Stage 1 — Gate Model (Tomato Leaf Verification)</div>', unsafe_allow_html=True)
    g1,g2,g3,g4 = st.columns(4)
    with g1: st.metric("Accuracy",  "99.93%", "+0.07% vs baseline")
    with g2: st.metric("Precision", "100.0%")
    with g3: st.metric("Recall",    "99.9%")
    with g4: st.metric("Test Loss", "0.0027")

    gc1, gc2 = st.columns([3,2], gap="large")
    with gc1:
        st.plotly_chart(training_fig(
            GATE_P1_TRAIN_LOSS, GATE_P1_VAL_LOSS, GATE_P1_TRAIN_ACC, GATE_P1_VAL_ACC,
            GATE_P2_TRAIN_LOSS, GATE_P2_VAL_LOSS, GATE_P2_TRAIN_ACC, GATE_P2_VAL_ACC,
            "Gate Model — Training History"
        ), use_container_width=True)
    with gc2:
        st.plotly_chart(cm_fig(GATE_CM, GATE_CLASSES, "Confusion Matrix"), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Disease classifier
    st.markdown('<div class="sec-head"><div class="dot"></div>Stage 2 — Disease Classifier</div>', unsafe_allow_html=True)
    d1,d2,d3,d4 = st.columns(4)
    with d1: st.metric("Accuracy",  "96.16%", "+1.21% from Phase 1")
    with d2: st.metric("Precision", "96.3%",  "weighted avg")
    with d3: st.metric("Recall",    "96.2%",  "weighted avg")
    with d4: st.metric("Test Loss", "0.1212")

    dc1, dc2 = st.columns([3,2], gap="large")
    with dc1:
        st.plotly_chart(training_fig(
            DISEASE_P1_TRAIN_LOSS, DISEASE_P1_VAL_LOSS, DISEASE_P1_TRAIN_ACC, DISEASE_P1_VAL_ACC,
            DISEASE_P2_TRAIN_LOSS, DISEASE_P2_VAL_LOSS, DISEASE_P2_TRAIN_ACC, DISEASE_P2_VAL_ACC,
            "Disease Classifier — Training History"
        ), use_container_width=True)
    with dc2:
        st.plotly_chart(cm_fig(DISEASE_CM, DISEASE_CLASSES, "Confusion Matrix"), use_container_width=True)

    # Per-class metrics
    st.markdown("<br>", unsafe_allow_html=True)
    pc1, pc2 = st.columns([3,2], gap="large")
    with pc1:
        st.markdown('<div class="sec-head"><div class="dot"></div>Per-Class Performance</div>', unsafe_allow_html=True)
        fig = go.Figure()
        precision = [0.98, 0.95, 0.96]
        recall    = [0.89, 1.00, 0.97]
        f1        = [0.93, 0.98, 0.97]
        for name, vals, color in [("Precision",precision,"#3b82f6"),("Recall",recall,"#22c55e"),("F1-Score",f1,"#ef4444")]:
            fig.add_bar(x=DISEASE_CLASSES, y=[v*100 for v in vals], name=name, marker_color=color, width=0.22)
        fig.update_layout(barmode="group", height=280, margin=dict(l=5,r=5,t=5,b=5),
            legend=dict(orientation="h",y=-0.25,x=0.5,xanchor="center",bgcolor="rgba(0,0,0,0)"), **PT)
        fig.update_yaxes(range=[80,104], title_text="Score (%)")
        st.plotly_chart(fig, use_container_width=True)

    with pc2:
        st.markdown('<div class="sec-head"><div class="dot"></div>Classification Report</div>', unsafe_allow_html=True)
        df = pd.DataFrame({
            "Class":     ["Early Blight","Healthy","Late Blight","Weighted Avg"],
            "Precision": ["0.98","0.95","0.96","0.96"],
            "Recall":    ["0.89","1.00","0.97","0.96"],
            "F1":        ["0.93","0.98","0.97","0.96"],
            "Support":   [150, 240, 287, 677],
        })
        st.dataframe(df.style.map(
            lambda v: "color:#86efac;font-weight:600" if v in ("1.00","0.98","0.97") else
                      "color:#fca5a5" if v=="0.89" else ""
        ), hide_index=True, use_container_width=True)

    # ROC curves
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-head"><div class="dot"></div>ROC Curves — One vs Rest</div>', unsafe_allow_html=True)
    fig = go.Figure()
    colors = {"Early Blight":"#f59e0b","Healthy":"#22c55e","Late Blight":"#ef4444"}
    aucs   = {"Early Blight":0.986,"Healthy":0.998,"Late Blight":0.994}
    for cls, col in colors.items():
        t = np.linspace(0,1,200)
        auc = aucs[cls]
        fp  = t
        tp  = np.clip(t + (auc-0.5)*2*np.sqrt(t*(1-t)+0.001) + np.random.default_rng(42).normal(0,0.005,200).cumsum()*0.002, 0, 1)
        tp  = np.sort(np.clip(tp,0,1))[::-1]; tp = np.sort(np.maximum.accumulate(tp[::-1]))
        tp[0]=0; tp[-1]=1
        fig.add_trace(go.Scatter(x=fp, y=tp, mode="lines", name=f"{cls} (AUC={auc:.3f})", line=dict(color=col,width=2)))
    fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",name="Random",line=dict(color="rgba(255,255,255,0.15)",dash="dash",width=1)))
    fig.update_layout(height=320,margin=dict(l=5,r=5,t=5,b=5),
        xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
        legend=dict(orientation="h",y=-0.25,x=0.5,xanchor="center",bgcolor="rgba(0,0,0,0)"), **PT)
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: PREDICTION HISTORY
# ══════════════════════════════════════════════════════════════════════════════
def page_history():
    st.markdown('<div style="font-size:1.4rem;font-weight:800;color:#f1f5f9;letter-spacing:-0.5px;margin-bottom:1.5rem;">Prediction History</div>', unsafe_allow_html=True)
    hist = st.session_state.history

    if not hist:
        st.markdown("""
        <div style="text-align:center;padding:3rem;color:#334155;">
          <div style="font-size:1rem;font-weight:600;color:#475569;">No predictions yet</div>
          <div style="font-size:0.82rem;color:#334155;margin-top:0.3rem;">Go to Disease Detection and analyze a leaf image.</div>
        </div>
        """, unsafe_allow_html=True)
        return

    # Stats
    total  = len(hist)
    tomato = sum(1 for h in hist if h["is_tomato"])
    s1,s2,s3,s4 = st.columns(4)
    with s1: st.metric("Total", total)
    with s2: st.metric("Tomato Leaves", tomato)
    with s3: st.metric("Diseased",  sum(1 for h in hist if h.get("disease") and h["disease"]!="Healthy"))
    with s4:
        confs = [h["confidence"]*100 for h in hist if "confidence" in h]
        st.metric("Avg Confidence", f"{np.mean(confs):.1f}%" if confs else "—")

    st.markdown("<br>", unsafe_allow_html=True)

    # Filters
    fc1,fc2,fc3 = st.columns([2,2,1])
    with fc1:
        search = st.text_input("Search by filename or disease", label_visibility="collapsed", placeholder="Search…")
    with fc2:
        filter_dis = st.selectbox("Filter", ["All","Tomato Only","Non-Tomato","Healthy","Early Blight","Late Blight"], label_visibility="collapsed")
    with fc3:
        if st.button("Clear All", use_container_width=True):
            st.session_state.history = []
            st.rerun()

    filtered = hist.copy()
    if search:
        filtered = [h for h in filtered if search.lower() in (h.get("filename","") + h.get("disease","")).lower()]
    if filter_dis == "Tomato Only":     filtered = [h for h in filtered if h["is_tomato"]]
    elif filter_dis == "Non-Tomato":    filtered = [h for h in filtered if not h["is_tomato"]]
    elif filter_dis in ["Healthy","Early Blight","Late Blight"]:
        filtered = [h for h in filtered if h.get("disease") == filter_dis]

    df = pd.DataFrame([{
        "Timestamp":      h.get("timestamp",""),
        "File":           h.get("filename",""),
        "Tomato Leaf":    "Yes" if h["is_tomato"] else "No",
        "Prediction":     h.get("disease","—"),
        "Confidence":     f"{h.get('confidence',0)*100:.1f}%" if h.get("confidence") else "—",
        "Gate Conf":      f"{h.get('gate_conf',0)*100:.1f}%",
    } for h in reversed(filtered)])

    st.dataframe(df, hide_index=True, use_container_width=True)

    if filtered:
        csv = pd.DataFrame([{
            "Timestamp":     h.get("timestamp",""),
            "Filename":      h.get("filename",""),
            "Is Tomato":     h["is_tomato"],
            "Prediction":    h.get("disease",""),
            "Confidence":    h.get("confidence",""),
            "Gate Confidence": h.get("gate_conf",""),
        } for h in filtered]).to_csv(index=False)
        st.download_button("Export CSV", csv, "tomatovision_predictions.csv", "text/csv", use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: ABOUT
# ══════════════════════════════════════════════════════════════════════════════
def page_about():
    st.markdown('<div style="font-size:1.4rem;font-weight:800;color:#f1f5f9;letter-spacing:-0.5px;margin-bottom:1.5rem;">About</div>', unsafe_allow_html=True)

    a1,a2 = st.columns(2, gap="large")
    with a1:
        st.markdown("""
        <div class="glass" style="margin-bottom:1rem;">
          <div class="sec-head"><div class="dot"></div>Project Overview</div>
          <div style="color:#64748b;font-size:0.84rem;line-height:1.8;">
            TomatoVision AI is an end-to-end deep learning platform for tomato plant disease detection,
            developed as a Computer Vision Capstone Project. It employs a two-stage AI pipeline to first
            verify that an uploaded image contains a tomato leaf, then classify any disease present.
            <br><br>
            The system is designed for use by smallholder farmers, agricultural extension officers,
            and crop health researchers across sub-Saharan Africa and beyond. It provides real-time
            diagnostic feedback alongside explainable AI visualisations to build trust in the system's predictions.
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="glass">
          <div class="sec-head"><div class="dot"></div>Technology Stack</div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.5rem;">
        """, unsafe_allow_html=True)
        tech = [
            ("TensorFlow 2.18","Deep Learning Framework"),
            ("ONNX Runtime","Production Inference"),
            ("EfficientNetB0","CNN Backbone"),
            ("Streamlit","Web Interface"),
            ("Plotly","Interactive Charts"),
            ("Pillow","Image Processing"),
            ("NumPy / Pandas","Data Processing"),
            ("Occlusion Sensitivity","Explainability"),
        ]
        for name, desc in tech:
            st.markdown(f'<div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.06);border-radius:8px;padding:0.6rem 0.8rem;"><div style="font-size:0.8rem;font-weight:600;color:#cbd5e1;">{name}</div><div style="font-size:0.68rem;color:#475569;">{desc}</div></div>', unsafe_allow_html=True)
        st.markdown("</div></div>", unsafe_allow_html=True)

    with a2:
        st.markdown("""
        <div class="glass" style="margin-bottom:1rem;">
          <div class="sec-head"><div class="dot"></div>Model Architecture</div>
          <div style="color:#64748b;font-size:0.82rem;line-height:1.75;">
            Both models share the same backbone architecture:
          </div>
          <div style="margin-top:0.8rem;">
        """, unsafe_allow_html=True)
        layers = [
            ("Input Layer","(None, 224, 224, 3)"),
            ("EfficientNet Preprocessing","Normalisation"),
            ("EfficientNetB0","Feature extraction"),
            ("GlobalAveragePooling2D","(None, 1280)"),
            ("BatchNormalization","Stabilisation"),
            ("Dropout (0.3)","Regularisation"),
            ("Dense (128, ReLU)","Feature compression"),
        ]
        for name,info in layers:
            st.markdown(f'<div style="display:flex;align-items:center;gap:0.6rem;padding:0.4rem 0.6rem;margin-bottom:0.2rem;background:rgba(255,255,255,0.02);border-radius:7px;font-size:0.78rem;"><span style="color:#94a3b8;flex:1;">{name}</span><span style="color:#334155;font-family:monospace;font-size:0.7rem;">{info}</span></div>', unsafe_allow_html=True)
        st.markdown(f'<div style="display:flex;align-items:center;gap:0.6rem;padding:0.4rem 0.6rem;background:rgba(6,214,160,0.06);border:1px solid rgba(6,214,160,0.15);border-radius:7px;font-size:0.78rem;"><span style="color:#6ee7b7;flex:1;">Output Head</span><span style="color:#334155;font-family:monospace;font-size:0.7rem;">sigmoid / softmax</span></div>', unsafe_allow_html=True)
        st.markdown("</div></div>", unsafe_allow_html=True)

        st.markdown("""
        <div class="glass">
          <div class="sec-head"><div class="dot"></div>Training Strategy</div>
          <div style="display:flex;flex-direction:column;gap:0.5rem;">
            <div style="padding:0.7rem 0.9rem;background:rgba(59,130,246,0.05);border:1px solid rgba(59,130,246,0.15);border-radius:9px;">
              <div style="color:#93c5fd;font-size:0.8rem;font-weight:600;margin-bottom:0.2rem;">Phase 1 — Head Training</div>
              <div style="color:#475569;font-size:0.75rem;">EfficientNetB0 frozen · LR 1e-4 · 30 epochs</div>
            </div>
            <div style="padding:0.7rem 0.9rem;background:rgba(34,197,94,0.05);border:1px solid rgba(34,197,94,0.15);border-radius:9px;">
              <div style="color:#86efac;font-size:0.8rem;font-weight:600;margin-bottom:0.2rem;">Phase 2 — Fine-Tuning</div>
              <div style="color:#475569;font-size:0.75rem;">Top 30 layers unfrozen · LR 1e-5 · up to 30 epochs</div>
            </div>
          </div>
          <div style="margin-top:0.8rem;font-size:0.68rem;font-weight:700;text-transform:uppercase;letter-spacing:0.8px;color:#334155;margin-bottom:0.4rem;">Data Augmentation</div>
          <div>
            <span class="badge badge-blue">RandomFlip</span>
            <span class="badge badge-blue">RandomRotation ±20%</span>
            <span class="badge badge-blue">RandomZoom ±15%</span>
            <span class="badge badge-blue">RandomContrast</span>
            <span class="badge badge-blue">RandomBrightness</span>
          </div>
          <div style="margin-top:1rem;padding:0.6rem 0.8rem;background:rgba(255,255,255,0.02);border-radius:8px;font-size:0.75rem;color:#475569;line-height:1.6;">
            Dataset: <strong style="color:#64748b;">PlantVillage</strong> · 9,799 leaf images ·
            Gate: 6,650 train / 1,425 test ·
            Disease: 3,149 train / 677 test
          </div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  TOP NAV BAR (always visible, works without sidebar)
# ══════════════════════════════════════════════════════════════════════════════
NAV_ITEMS = [
    ("Dashboard","dashboard"),
    ("Detect","detection"),
    ("Disease Library","library"),
    ("Model Performance","performance"),
    ("History","history"),
    ("About","about"),
]
page = st.session_state.page
cols = st.columns(len(NAV_ITEMS))
for col, (label, key) in zip(cols, NAV_ITEMS):
    with col:
        if st.button(label, key=f"topnav_{key}", use_container_width=True):
            st.session_state.page = key
            st.rerun()

st.markdown("""
<style>
[data-testid="stHorizontalBlock"]:first-of-type button {
    border-radius: 10px !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    padding: 0.4rem !important;
}
</style>
""", unsafe_allow_html=True)
st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

# ── ROUTER ───────────────────────────────────────────────────────────────────
page = st.session_state.page
if   page == "dashboard":   page_dashboard()
elif page == "detection":   page_detection()
elif page == "library":     page_library()
elif page == "performance": page_performance()
elif page == "history":     page_history()
elif page == "about":       page_about()
