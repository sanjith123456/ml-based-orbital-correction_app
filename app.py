import streamlit as st
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import plotly.graph_objects as go

from astropy.coordinates import ITRS, GCRS
from astropy.time import Time
import astropy.units as u

from datetime import datetime

st.set_page_config(
    page_title="ML Orbit Correction",
    layout="wide"
)

device = torch.device("cpu")

# ==================================
# MODEL
# ==================================

class Model(nn.Module):

    def __init__(self, input_dim):

        super().__init__()

        self.fc1 = nn.Linear(input_dim, 64)

        enc = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=8,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True
        )

        self.tr = nn.TransformerEncoder(
            enc,
            num_layers=3
        )

        self.fc2 = nn.Sequential(
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,3)
        )

    def forward(self,x):

        x = self.fc1(x)

        x = self.tr(x)

        x = x[:, -1, :]

        return self.fc2(x)

# ==================================
# LOAD MODEL
# ==================================

@st.cache_resource
def load_model():

    model = Model(9)

    model.load_state_dict(
        torch.load(
            "models/orbit_transformer.pth",
            map_location=device
        )
    )

    model.eval()

    sx = joblib.load("models/sx.pkl")
    sy = joblib.load("models/sy.pkl")

    return model,sx,sy

model,sx,sy = load_model()

# ==================================
# TITLE
# ==================================

st.title("🛰 Physics-Informed Orbit Correction")

st.markdown("""
GMAT + CPF + Transformer Residual Learning

Typical result:

7 km → 50 m RMSE
""")

# ==================================
# SIDEBAR
# ==================================

mode = st.sidebar.radio(
    "Mode",
    ["Demo","Upload"]
)

cpf_file = None
gmat_file = None

if mode == "Upload":

    cpf_file = st.sidebar.file_uploader(
        "CPF File",
        type=["txt"]
    )

    gmat_file = st.sidebar.file_uploader(
        "GMAT File",
        type=["txt"]
    )

run_button = st.sidebar.button(
    "Run Correction"
)
# ==================================
# CPF LOADER
# ==================================

def load_cpf_multi(file_obj):

    t_all = []
    r_all = []

    for line in file_obj:

        line = line.decode("utf-8")

        if not line.startswith("10"):
            continue

        p = line.split()

        mjd = float(p[2])
        sod = float(p[3])

        t = Time(
            mjd,
            format="mjd",
            scale="utc"
        ) + sod*u.s

        x = float(p[5])/1000.0
        y = float(p[6])/1000.0
        z = float(p[7])/1000.0

        t_all.append(t.unix)
        r_all.append([x,y,z])

    t_all = np.array(t_all)
    r_all = np.array(r_all)

    idx = np.argsort(t_all)

    t_all = t_all[idx]
    r_all = r_all[idx]

    keep = np.insert(
        np.diff(t_all) > 0,
        0,
        True
    )

    return (
        Time(
            t_all[keep],
            format="unix",
            scale="utc"
        ),
        r_all[keep]
    )


# ==================================
# GMAT LOADER
# ==================================

def load_gmat(file_obj):

    t = []
    r = []

    for line in file_obj:

        line = line.decode("utf-8")

        if "=" in line:
            continue

        if "META" in line:
            continue

        p = line.split()

        if len(p) < 4:
            continue

        try:

            tt = Time(
                p[0],
                format="isot",
                scale="utc"
            )

        except:

            try:

                tt = Time(
                    datetime.strptime(
                        p[0],
                        "%Y-%m-%dT%H:%M:%S.%f"
                    ),
                    scale="utc"
                )

            except:
                continue

        x = float(p[1])
        y = float(p[2])
        z = float(p[3])

        t.append(tt)

        r.append([x,y,z])

    return Time(t,scale="utc"), np.array(r)


# ==================================
# VELOCITY
# ==================================

def compute_velocity(r,t):

    dt = np.gradient(t)

    v = (
        np.gradient(
            r,
            axis=0
        )
        /
        dt[:,None]
    )

    return v


# ==================================
# RTN BASIS
# ==================================

def make_rtn_basis(r,v):

    eps = 1e-12

    R = (
        r
        /
        (
            np.linalg.norm(
                r,
                axis=1,
                keepdims=True
            )
            +
            eps
        )
    )

    N = np.cross(r,v)

    N = (
        N
        /
        (
            np.linalg.norm(
                N,
                axis=1,
                keepdims=True
            )
            +
            eps
        )
    )

    T = np.cross(
        N,
        R
    )

    T = (
        T
        /
        (
            np.linalg.norm(
                T,
                axis=1,
                keepdims=True
            )
            +
            eps
        )
    )

    return R,T,N


# ==================================
# MATCH CPF TO GMAT
# ==================================

def match_data(
    t_cpf,
    r_cpf,
    t_gmat,
    r_gmat
):

    t_cpf_u = t_cpf.unix
    t_gmat_u = t_gmat.unix

    idx = np.searchsorted(
        t_gmat_u,
        t_cpf_u
    )

    valid = (
        (idx > 0)
        &
        (idx < len(t_gmat_u))
    )

    idx = idx[valid]

    t_cpf_valid = t_cpf_u[valid]

    r_cpf_valid = r_cpf[valid]

    left = np.abs(
        t_gmat_u[idx-1]
        -
        t_cpf_valid
    )

    right = np.abs(
        t_gmat_u[idx]
        -
        t_cpf_valid
    )

    idx[left < right] -= 1

    dt = np.abs(
        t_gmat_u[idx]
        -
        t_cpf_valid
    )

    good = dt < 0.1

    idx = idx[good]

    t_match = t_gmat_u[idx]

    r_cpf_match = r_cpf_valid[good]

    r_gmat_match = r_gmat[idx]

    return (
        t_match,
        r_cpf_match,
        r_gmat_match
    )
# ==================================
# CPF -> GCRS
# ==================================

def cpf_to_gcrs(
    t_cpf,
    r_cpf
):

    itrs = ITRS(
        x=r_cpf[:,0] * u.km,
        y=r_cpf[:,1] * u.km,
        z=r_cpf[:,2] * u.km,
        obstime=t_cpf
    )

    gcrs = itrs.transform_to(
        GCRS(obstime=t_cpf)
    )

    r_gcrs = np.vstack([
        gcrs.cartesian.x.to(u.km).value,
        gcrs.cartesian.y.to(u.km).value,
        gcrs.cartesian.z.to(u.km).value
    ]).T

    return r_gcrs


# ==================================
# SEQUENCE BUILDER
# ==================================

SEQ = 100

def make_seq(X, seq):

    xs = []

    for i in range(len(X) - seq):

        xs.append(
            X[i:i+seq]
        )

    return np.array(xs)


# ==================================
# FEATURE GENERATION
# ==================================

def build_features(
    r_gmat,
    t_match
):

    v = compute_velocity(
        r_gmat,
        t_match
    )

    theta = np.arctan2(
        r_gmat[:,1],
        r_gmat[:,0]
    ).reshape(-1,1)

    time_norm = (
        (
            t_match
            -
            t_match[0]
        )
        /
        (
            t_match[-1]
            -
            t_match[0]
        )
    ).reshape(-1,1)

    X_raw = np.hstack([

        r_gmat,

        v,

        np.sin(theta),

        np.cos(theta),

        time_norm

    ])

    return X_raw,v


# ==================================
# MODEL INFERENCE
# ==================================

def run_model(
    X_raw,
    sx,
    sy,
    model
):

    X_scaled = sx.transform(
        X_raw
    )

    X_seq = make_seq(
        X_scaled,
        SEQ
    )

    preds = []

    with torch.no_grad():

        for i in range(len(X_seq)):

            xb = torch.tensor(
                X_seq[i:i+1],
                dtype=torch.float32
            )

            out = model(
                xb
            ).cpu().numpy()[0]

            preds.append(out)

    preds = np.array(preds)

    preds = sy.inverse_transform(
        preds
    )

    return preds


# ==================================
# APPLY RTN CORRECTION
# ==================================

def apply_correction(
    r_gmat,
    v,
    pred
):

    r_gmat = r_gmat[SEQ:]
    v = v[SEQ:]

    R,T,N = make_rtn_basis(
        r_gmat,
        v
    )

    r_corr = (

        r_gmat

        + pred[:,0:1] * R

        + pred[:,1:2] * T

        + pred[:,2:3] * N

    )

    return (
        r_corr,
        R,
        T,
        N
    )


# ==================================
# RMSE
# ==================================

def rmse(x):

    return np.sqrt(
        np.mean(
            x**2
        )
    )
# ==================================
# DEMO FILES
# ==================================
if mode == "Demo":
    st.info("Using sample data")

    cpf_files = sorted(
        glob.glob(
            "sample_data/cpf/*.hts.txt"
        )
    )

    gmat_path = "sample_data/sample_gmat.txt"

    if run_button:

        cpf_lines = []

        for f in cpf_files:

            with open(f, "rb") as fp:

                cpf_lines.extend(
                    fp.read().splitlines()
                )

        with open(gmat_path, "rb") as fp:

            gmat_lines = (
                fp.read().splitlines()
            )

# ==================================
# RUN
# ==================================

if run_button:

    if mode == "Upload":

        if cpf_file is None or gmat_file is None:
            st.error("Upload CPF and GMAT files")
            st.stop()

        cpf_lines = cpf_file.readlines()
        gmat_lines = gmat_file.readlines()

    

    with st.spinner("Loading data..."):

        t_cpf, r_cpf = load_cpf_multi(cpf_lines)

        t_gmat, r_gmat = load_gmat(gmat_lines)

        r_cpf_gcrs = cpf_to_gcrs(
            t_cpf,
            r_cpf
        )

        (
            t_match,
            r_cpf_match,
            r_gmat_match
        ) = match_data(
            t_cpf,
            r_cpf_gcrs,
            t_gmat,
            r_gmat
        )

    st.success(
        f"Matched {len(r_cpf_match)} points"
    )

    # ============================
    # FEATURES
    # ============================

    X_raw, v = build_features(
        r_gmat_match,
        t_match
    )

    # ============================
    # PREDICTION
    # ============================

    with st.spinner("Running transformer..."):

        pred = run_model(
            X_raw,
            sx,
            sy,
            model
        )

    r_corr, R, T, N = apply_correction(
        r_gmat_match,
        v,
        pred
    )

    r_cpf_t = r_cpf_match[SEQ:]
    r_gmat_t = r_gmat_match[SEQ:]

    # ============================
    # ERRORS
    # ============================

    err_before_vec = (
        r_cpf_t
        -
        r_gmat_t
    )

    err_after_vec = (
        r_cpf_t
        -
        r_corr
    )

    err_before = np.linalg.norm(
        err_before_vec,
        axis=1
    )

    err_after = np.linalg.norm(
        err_after_vec,
        axis=1
    )

    rmse_before = rmse(
        err_before
    )

    rmse_after = rmse(
        err_after
    )

    # ============================
    # METRICS
    # ============================

    st.header("Results")

    c1,c2 = st.columns(2)

    c1.metric(
        "RMSE Before",
        f"{rmse_before:.3f} km"
    )

    c2.metric(
        "RMSE After",
        f"{rmse_after:.3f} km"
    )

    # ============================
    # XYZ RMSE
    # ============================

    x_before = rmse(
        err_before_vec[:,0]
    )

    y_before = rmse(
        err_before_vec[:,1]
    )

    z_before = rmse(
        err_before_vec[:,2]
    )

    x_after = rmse(
        err_after_vec[:,0]
    )

    y_after = rmse(
        err_after_vec[:,1]
    )

    z_after = rmse(
        err_after_vec[:,2]
    )

    st.subheader("Axis RMSE")

    df = pd.DataFrame({

        "Axis":["X","Y","Z"],

        "Before":[
            x_before,
            y_before,
            z_before
        ],

        "After":[
            x_after,
            y_after,
            z_after
        ]

    })

    st.dataframe(df)

    # ============================
    # TOTAL ERROR PLOT
    # ============================

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            y=err_before,
            name="GMAT"
        )
    )

    fig.add_trace(
        go.Scatter(
            y=err_after,
            name="Corrected"
        )
    )

    fig.update_layout(
        title="Total Position Error",
        yaxis_title="km"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ============================
    # RTN ERRORS
    # ============================

    err_rtn_before = np.column_stack([

        np.sum(
            err_before_vec*R,
            axis=1
        ),

        np.sum(
            err_before_vec*T,
            axis=1
        ),

        np.sum(
            err_before_vec*N,
            axis=1
        )

    ])

    err_rtn_after = np.column_stack([

        np.sum(
            err_after_vec*R,
            axis=1
        ),

        np.sum(
            err_after_vec*T,
            axis=1
        ),

        np.sum(
            err_after_vec*N,
            axis=1
        )

    ])

    fig_rtn = go.Figure()

    fig_rtn.add_trace(
        go.Scatter(
            y=err_rtn_before[:,1],
            name="Along Track Before"
        )
    )

    fig_rtn.add_trace(
        go.Scatter(
            y=err_rtn_after[:,1],
            name="Along Track After"
        )
    )

    fig_rtn.update_layout(
        title="Along Track Error"
    )

    st.plotly_chart(
        fig_rtn,
        use_container_width=True
    )

    # ============================
    # 3D ORBIT
    # ============================

    st.subheader(
        "3D Orbit Comparison"
    )

    fig3d = go.Figure()

    fig3d.add_trace(
        go.Scatter3d(
            x=r_cpf_t[:,0],
            y=r_cpf_t[:,1],
            z=r_cpf_t[:,2],
            mode="lines",
            name="CPF"
        )
    )

    fig3d.add_trace(
        go.Scatter3d(
            x=r_gmat_t[:,0],
            y=r_gmat_t[:,1],
            z=r_gmat_t[:,2],
            mode="lines",
            name="GMAT"
        )
    )

    fig3d.add_trace(
        go.Scatter3d(
            x=r_corr[:,0],
            y=r_corr[:,1],
            z=r_corr[:,2],
            mode="lines",
            name="Corrected"
        )
    )

    fig3d.update_layout(
        scene=dict(
            aspectmode="data"
        ),
        height=700
    )

    st.plotly_chart(
        fig3d,
        use_container_width=True
    )

    # ============================
    # DOWNLOAD
    # ============================

    corr_df = pd.DataFrame(
        r_corr,
        columns=[
            "X_km",
            "Y_km",
            "Z_km"
        ]
    )

    csv = corr_df.to_csv(
        index=False
    )

    st.download_button(
        "Download Corrected Orbit",
        csv,
        "corrected_orbit.csv",
        "text/csv"
    )
