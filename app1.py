from pathlib import Path

app_code = r'''import streamlit as st
import glob
import io
import ftplib
import re
from datetime import datetime, date

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import plotly.graph_objects as go

from astropy.coordinates import ITRS, GCRS
from astropy.time import Time
import astropy.units as u


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="ML Orbit Correction",
    page_icon="🛰️",
    layout="wide"
)

device = torch.device("cpu")


# ============================================================
# MODEL
# Same architecture as the original repository app.py
# ============================================================

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
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.tr(x)
        x = x[:, -1, :]
        return self.fc2(x)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

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

    return model, sx, sy


try:
    model, sx, sy = load_model()
except Exception as e:
    st.error(
        "Could not load the trained model/scalers. "
        "Make sure models/orbit_transformer.pth, "
        "models/sx.pkl and models/sy.pkl exist."
    )
    st.exception(e)
    st.stop()


# ============================================================
# TITLE
# ============================================================

st.title("🛰️ Physics-Informed Orbit Correction")

st.markdown(
    """
**GMAT + CPF + Transformer Residual Learning**

The application compares the GMAT trajectory with an ILRS/EDC
CPF prediction, predicts RTN residual corrections with the trained
Transformer, and visualizes the corrected orbit.
"""
)


# ============================================================
# EDC CPF CONFIGURATION
# ============================================================

EDC_HOST = "edc.dgfi.tum.de"

EDC_CPF_ROOT = "/pub/slr/cpf_predicts_v2"

SATELLITES = {
    "Ajisai": {
        "directory": "ajisai",
        "prefix": "ajisai_cpf_"
    }
}


# ============================================================
# CPF HEADER PARSER
# ============================================================

def parse_cpf_header_dates(raw_bytes):
    """
    Parse CPF H2 record.

    Returns
    -------
    start_dt, end_dt : datetime or (None, None)
    """

    if isinstance(raw_bytes, bytes):
        text = raw_bytes.decode(
            "utf-8",
            errors="ignore"
        )
    else:
        text = raw_bytes

    for line in text.splitlines():

        if not line.startswith("H2"):
            continue

        parts = line.split()

        # Standard CPF H2 contains:
        # H2 ID SIC NORAD YYYY MM DD HH MM SS
        #    YYYY MM DD HH MM SS ...

        if len(parts) < 16:
            continue

        try:

            # For CPFv2 files observed at EDC, these are:
            # p[5:11] = start date/time
            # p[11:17] = end date/time

            start = datetime(
                int(parts[5]),
                int(parts[6]),
                int(parts[7]),
                int(parts[8]),
                int(parts[9]),
                int(float(parts[10]))
            )

            end = datetime(
                int(parts[11]),
                int(parts[12]),
                int(parts[13]),
                int(parts[14]),
                int(parts[15]),
                int(float(parts[16]))
            )

            return start, end

        except (ValueError, IndexError):
            continue

    return None, None


# ============================================================
# AUTOMATIC EDC CPF DOWNLOAD
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def download_cpf_for_date(
    satellite,
    target_date
):
    """
    Search the EDC Ajisai CPF directory and download the
    first valid CPF prediction whose header covers target_date.

    Returns
    -------
    raw_bytes, filename, start, end
    """

    if satellite not in SATELLITES:
        raise ValueError(
            f"Unsupported satellite: {satellite}"
        )

    sat = SATELLITES[satellite]

    directory = (
        f"{EDC_CPF_ROOT}/"
        f"{target_date.year}/"
        f"{sat['directory']}"
    )

    target_dt = datetime.combine(
        target_date,
        datetime.min.time()
    )

    with ftplib.FTP(
        EDC_HOST,
        timeout=30
    ) as ftp:

        ftp.login()

        files = ftp.nlst(directory)

        candidates = []

        for path in files:

            filename = path.rsplit("/", 1)[-1]

            if not filename.startswith(
                sat["prefix"]
            ):
                continue

            if filename.lower().endswith(
                (".dgf", ".hts", ".sgf")
            ):
                candidates.append(filename)

        # Most recent incoming prediction first.
        candidates.sort(reverse=True)

        for filename in candidates:

            remote_path = (
                f"{directory}/{filename}"
            )

            buffer = io.BytesIO()

            try:

                ftp.retrbinary(
                    f"RETR {remote_path}",
                    buffer.write
                )

                raw = buffer.getvalue()

                start, end = parse_cpf_header_dates(raw)

                if start is None or end is None:
                    continue

                if start <= target_dt <= end:

                    return (
                        raw,
                        filename,
                        start,
                        end
                    )

            except Exception:
                continue

    raise FileNotFoundError(
        f"No valid CPF prediction covering "
        f"{target_date.isoformat()} was found "
        f"for {satellite}."
    )


# ============================================================
# CPF LOADER
# ============================================================

def load_cpf_multi(file_obj):

    t_all = []
    r_all = []

    for line in file_obj:

        if isinstance(line, bytes):
            line = line.decode(
                "utf-8",
                errors="ignore"
            )

        line = line.strip()

        if not line.startswith("10"):
            continue

        p = line.split()

        if len(p) < 8:
            continue

        try:

            mjd = float(p[2])
            sod = float(p[3])

            t = (
                Time(
                    mjd,
                    format="mjd",
                    scale="utc"
                )
                + sod * u.s
            )

            x = float(p[5]) / 1000.0
            y = float(p[6]) / 1000.0
            z = float(p[7]) / 1000.0

        except (ValueError, IndexError):
            continue

        t_all.append(t.unix)
        r_all.append([x, y, z])

    if len(t_all) == 0:
        raise ValueError(
            "No CPF type-10 position records were found."
        )

    t_all = np.asarray(t_all)
    r_all = np.asarray(r_all)

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


# ============================================================
# GMAT LOADER
# ============================================================

def load_gmat(file_obj):

    t = []
    r = []

    for line in file_obj:

        if isinstance(line, bytes):
            line = line.decode(
                "utf-8",
                errors="ignore"
            )

        line = line.strip()

        if not line:
            continue

        if "=" in line:
            continue

        if "META" in line:
            continue

        p = line.split()

        if len(p) < 4:
            continue

        try:

            try:
                tt = Time(
                    p[0],
                    format="isot",
                    scale="utc"
                )

            except Exception:

                tt = Time(
                    datetime.strptime(
                        p[0],
                        "%Y-%m-%dT%H:%M:%S.%f"
                    ),
                    scale="utc"
                )

            x = float(p[1])
            y = float(p[2])
            z = float(p[3])

        except (ValueError, IndexError):
            continue

        t.append(tt)
        r.append([x, y, z])

    if len(t) == 0:
        raise ValueError(
            "No valid GMAT position records were found."
        )

    return Time(t, scale="utc"), np.asarray(r)


# ============================================================
# VELOCITY
# ============================================================

def compute_velocity(r, t):

    dt = np.gradient(t)

    if np.any(dt == 0):
        raise ValueError(
            "Duplicate GMAT timestamps prevent "
            "velocity calculation."
        )

    return (
        np.gradient(
            r,
            axis=0
        )
        / dt[:, None]
    )


# ============================================================
# RTN BASIS
# ============================================================

def make_rtn_basis(r, v):

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
            + eps
        )
    )

    N = np.cross(r, v)

    N = (
        N
        /
        (
            np.linalg.norm(
                N,
                axis=1,
                keepdims=True
            )
            + eps
        )
    )

    T = np.cross(N, R)

    T = (
        T
        /
        (
            np.linalg.norm(
                T,
                axis=1,
                keepdims=True
            )
            + eps
        )
    )

    return R, T, N


# ============================================================
# MATCH CPF TO GMAT
# ============================================================

def match_data(
    t_cpf,
    r_cpf,
    t_gmat,
    r_gmat,
    tolerance=0.1
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
        t_gmat_u[idx - 1]
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

    good = dt < tolerance

    idx = idx[good]

    t_match = t_gmat_u[idx]

    r_cpf_match = r_cpf_valid[good]
    r_gmat_match = r_gmat[idx]

    return (
        t_match,
        r_cpf_match,
        r_gmat_match
    )


# ============================================================
# CPF -> GCRS
# ============================================================

def cpf_to_gcrs(
    t_cpf,
    r_cpf
):

    itrs = ITRS(
        x=r_cpf[:, 0] * u.km,
        y=r_cpf[:, 1] * u.km,
        z=r_cpf[:, 2] * u.km,
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


# ============================================================
# SEQUENCE
# ============================================================

SEQ = 100


def make_seq(X, seq):

    xs = []

    for i in range(
        len(X) - seq
    ):

        xs.append(
            X[i:i + seq]
        )

    if len(xs) == 0:
        return np.empty(
            (0, seq, X.shape[1]),
            dtype=X.dtype
        )

    return np.asarray(xs)


# ============================================================
# FEATURES
# ============================================================

def build_features(
    r_gmat,
    t_match
):

    if len(r_gmat) < 3:
        raise ValueError(
            "At least 3 matched GMAT points are needed."
        )

    v = compute_velocity(
        r_gmat,
        t_match
    )

    theta = np.arctan2(
        r_gmat[:, 1],
        r_gmat[:, 0]
    ).reshape(-1, 1)

    duration = (
        t_match[-1]
        -
        t_match[0]
    )

    if duration <= 0:
        raise ValueError(
            "Matched time interval must be positive."
        )

    time_norm = (
        (
            t_match
            -
            t_match[0]
        )
        /
        duration
    ).reshape(-1, 1)

    X_raw = np.hstack([
        r_gmat,
        v,
        np.sin(theta),
        np.cos(theta),
        time_norm
    ])

    return X_raw, v


# ============================================================
# MODEL INFERENCE
# ============================================================

def run_model(
    X_raw,
    sx,
    sy,
    model
):

    X_scaled = sx.transform(X_raw)

    X_seq = make_seq(
        X_scaled,
        SEQ
    )

    if len(X_seq) == 0:
        raise ValueError(
            f"Only {len(X_raw)} matched points are "
            f"available. At least {SEQ + 1} are "
            f"required for a sequence length of {SEQ}."
        )

    preds = []

    with torch.no_grad():

        for i in range(
            len(X_seq)
        ):

            xb = torch.tensor(
                X_seq[i:i + 1],
                dtype=torch.float32
            )

            out = model(
                xb
            ).cpu().numpy()[0]

            preds.append(out)

    preds = np.asarray(preds)

    preds = sy.inverse_transform(
        preds
    )

    return preds


# ============================================================
# APPLY RTN CORRECTION
# ============================================================

def apply_correction(
    r_gmat,
    v,
    pred
):

    r_gmat_trim = r_gmat[SEQ:]
    v_trim = v[SEQ:]

    R, T, N = make_rtn_basis(
        r_gmat_trim,
        v_trim
    )

    r_corr = (
        r_gmat_trim
        + pred[:, 0:1] * R
        + pred[:, 1:2] * T
        + pred[:, 2:3] * N
    )

    return (
        r_corr,
        R,
        T,
        N
    )


# ============================================================
# RMSE
# ============================================================

def rmse(x):

    return np.sqrt(
        np.mean(
            np.asarray(x) ** 2
        )
    )


# ============================================================
# SIDEBAR
# ============================================================

mode = st.sidebar.radio(
    "Mode",
    [
        "Demo",
        "Upload",
        "Automatic CPF"
    ]
)

cpf_file = None
gmat_file = None

run_button = False

if mode == "Upload":

    cpf_file = st.sidebar.file_uploader(
        "CPF File",
        type=[
            "txt",
            "dgf",
            "hts",
            "sgf"
        ]
    )

    gmat_file = st.sidebar.file_uploader(
        "GMAT File",
        type=["txt", "csv"]
    )

    run_button = st.sidebar.button(
        "Run Correction",
        type="primary"
    )


elif mode == "Automatic CPF":

    satellite = st.sidebar.selectbox(
        "Satellite",
        list(SATELLITES.keys())
    )

    cpf_date = st.sidebar.date_input(
        "CPF coverage date",
        value=date.today()
    )

    gmat_file = st.sidebar.file_uploader(
        "GMAT File",
        type=["txt", "csv"]
    )

    run_button = st.sidebar.button(
        "Fetch CPF + Run Correction",
        type="primary"
    )

else:

    st.sidebar.info(
        "Demo uses the files bundled in "
        "`sample_data/`."
    )

    run_button = st.sidebar.button(
        "Run Demo",
        type="primary"
    )


# ============================================================
# PREPARE INPUT FILES
# ============================================================

if run_button:

    try:

        # ----------------------------------------------------
        # DEMO
        # ----------------------------------------------------

        if mode == "Demo":

            cpf_files = sorted(
                glob.glob(
                    "sample_data/*.hts.txt"
                )
            )

            gmat_path = (
                "sample_data/hope_atm_240.txt"
            )

            if not cpf_files:
                raise FileNotFoundError(
                    "No sample CPF files were found "
                    "in sample_data/."
                )

            if not glob.glob(gmat_path):
                raise FileNotFoundError(
                    f"Demo GMAT file not found: {gmat_path}"
                )

            cpf_lines = []

            for f in cpf_files:

                with open(
                    f,
                    "rb"
                ) as fp:

                    cpf_lines.extend(
                        fp.read().splitlines()
                    )

            with open(
                gmat_path,
                "rb"
            ) as fp:

                gmat_lines = (
                    fp.read().splitlines()
                )

            st.info(
                "Using bundled sample data."
            )


        # ----------------------------------------------------
        # UPLOAD
        # ----------------------------------------------------

        elif mode == "Upload":

            if (
                cpf_file is None
                or gmat_file is None
            ):

                st.error(
                    "Please upload both CPF and GMAT files."
                )

                st.stop()

            cpf_lines = cpf_file.readlines()
            gmat_lines = gmat_file.readlines()


        # ----------------------------------------------------
        # AUTOMATIC CPF
        # ----------------------------------------------------

        elif mode == "Automatic CPF":

            if gmat_file is None:

                st.error(
                    "Please upload the GMAT trajectory. "
                    "The current trained Transformer was "
                    "trained using GMAT-derived features."
                )

                st.stop()

            with st.spinner(
                f"Searching EDC for {satellite} CPF..."
            ):

                (
                    cpf_bytes,
                    cpf_filename,
                    cpf_start,
                    cpf_end
                ) = download_cpf_for_date(
                    satellite,
                    cpf_date
                )

            cpf_lines = (
                cpf_bytes.splitlines()
            )

            gmat_lines = (
                gmat_file.readlines()
            )

            st.success(
                f"Downloaded CPF: {cpf_filename}"
            )

            st.caption(
                "CPF coverage: "
                f"{cpf_start:%Y-%m-%d %H:%M} UTC "
                "→ "
                f"{cpf_end:%Y-%m-%d %H:%M} UTC"
            )


        # ====================================================
        # PROCESS
        # ====================================================

        with st.spinner(
            "Loading CPF and GMAT data..."
        ):

            t_cpf, r_cpf = load_cpf_multi(
                cpf_lines
            )

            t_gmat, r_gmat = load_gmat(
                gmat_lines
            )

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

        if len(r_cpf_match) <= SEQ:

            raise ValueError(
                f"Only {len(r_cpf_match)} points matched. "
                f"The Transformer needs more than {SEQ} "
                "matched points because its sequence "
                "length is 100."
            )

        st.success(
            f"Matched {len(r_cpf_match)} CPF/GMAT points."
        )


        # ====================================================
        # FEATURES
        # ====================================================

        X_raw, v = build_features(
            r_gmat_match,
            t_match
        )


        # ====================================================
        # PREDICTION
        # ====================================================

        with st.spinner(
            "Running trained Transformer..."
        ):

            pred = run_model(
                X_raw,
                sx,
                sy,
                model
            )


        # ====================================================
        # CORRECTION
        # ====================================================

        (
            r_corr,
            R,
            T,
            N
        ) = apply_correction(
            r_gmat_match,
            v,
            pred
        )

        r_cpf_t = r_cpf_match[SEQ:]
        r_gmat_t = r_gmat_match[SEQ:]


        # ====================================================
        # ERRORS
        # ====================================================

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


        # ====================================================
        # RESULTS
        # ====================================================

        st.header("Results")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "RMSE Before",
            f"{rmse_before:.3f} km"
        )

        c2.metric(
            "RMSE After",
            f"{rmse_after:.3f} km"
        )

        improvement = (
            100.0
            *
            (
                1.0
                -
                rmse_after / rmse_before
            )
            if rmse_before != 0
            else 0.0
        )

        c3.metric(
            "RMSE Change",
            f"{improvement:.1f}%"
        )


        # ====================================================
        # XYZ RMSE
        # ====================================================

        x_before = rmse(
            err_before_vec[:, 0]
        )

        y_before = rmse(
            err_before_vec[:, 1]
        )

        z_before = rmse(
            err_before_vec[:, 2]
        )

        x_after = rmse(
            err_after_vec[:, 0]
        )

        y_after = rmse(
            err_after_vec[:, 1]
        )

        z_after = rmse(
            err_after_vec[:, 2]
        )

        st.subheader(
            "Axis RMSE"
        )

        df_axis = pd.DataFrame({
            "Axis": ["X", "Y", "Z"],
            "Before (km)": [
                x_before,
                y_before,
                z_before
            ],
            "After (km)": [
                x_after,
                y_after,
                z_after
            ]
        })

        st.dataframe(
            df_axis,
            use_container_width=True,
            hide_index=True
        )


        # ====================================================
        # TOTAL ERROR
        # ====================================================

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
            xaxis_title="Matched sample",
            yaxis_title="Error (km)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


        # ====================================================
        # RTN ERROR
        # ====================================================

        err_rtn_before = np.column_stack([
            np.sum(
                err_before_vec * R,
                axis=1
            ),
            np.sum(
                err_before_vec * T,
                axis=1
            ),
            np.sum(
                err_before_vec * N,
                axis=1
            )
        ])

        err_rtn_after = np.column_stack([
            np.sum(
                err_after_vec * R,
                axis=1
            ),
            np.sum(
                err_after_vec * T,
                axis=1
            ),
            np.sum(
                err_after_vec * N,
                axis=1
            )
        ])


        st.subheader(
            "RTN Error"
        )

        fig_rtn = go.Figure()

        for i, name in enumerate(
            ["Radial", "Along Track", "Cross Track"]
        ):

            fig_rtn.add_trace(
                go.Scatter(
                    y=err_rtn_before[:, i],
                    name=f"{name} Before"
                )
            )

            fig_rtn.add_trace(
                go.Scatter(
                    y=err_rtn_after[:, i],
                    name=f"{name} After"
                )
            )

        fig_rtn.update_layout(
            title="RTN Position Error",
            xaxis_title="Matched sample",
            yaxis_title="Error (km)"
        )

        st.plotly_chart(
            fig_rtn,
            use_container_width=True
        )


        # ====================================================
        # 3D ORBIT
        # ====================================================

        st.subheader(
            "3D Orbit Comparison"
        )

        fig3d = go.Figure()

        fig3d.add_trace(
            go.Scatter3d(
                x=r_cpf_t[:, 0],
                y=r_cpf_t[:, 1],
                z=r_cpf_t[:, 2],
                mode="lines",
                name="CPF"
            )
        )

        fig3d.add_trace(
            go.Scatter3d(
                x=r_gmat_t[:, 0],
                y=r_gmat_t[:, 1],
                z=r_gmat_t[:, 2],
                mode="lines",
                name="GMAT"
            )
        )

        fig3d.add_trace(
            go.Scatter3d(
                x=r_corr[:, 0],
                y=r_corr[:, 1],
                z=r_corr[:, 2],
                mode="lines",
                name="Corrected"
            )
        )

        fig3d.update_layout(
            scene=dict(
                aspectmode="data",
                xaxis_title="X (km)",
                yaxis_title="Y (km)",
                zaxis_title="Z (km)"
            ),
            height=700
        )

        st.plotly_chart(
            fig3d,
            use_container_width=True
        )


        # ====================================================
        # RESULTS TABLE
        # ====================================================

        result_df = pd.DataFrame({
            "Time_unix": t_match[SEQ:].unix,
            "CPF_X_km": r_cpf_t[:, 0],
            "CPF_Y_km": r_cpf_t[:, 1],
            "CPF_Z_km": r_cpf_t[:, 2],
            "GMAT_X_km": r_gmat_t[:, 0],
            "GMAT_Y_km": r_gmat_t[:, 1],
            "GMAT_Z_km": r_gmat_t[:, 2],
            "Corrected_X_km": r_corr[:, 0],
            "Corrected_Y_km": r_corr[:, 1],
            "Corrected_Z_km": r_corr[:, 2],
            "Error_Before_km": err_before,
            "Error_After_km": err_after,
            "Radial_Before_km": err_rtn_before[:, 0],
            "Along_Before_km": err_rtn_before[:, 1],
            "Cross_Before_km": err_rtn_before[:, 2],
            "Radial_After_km": err_rtn_after[:, 0],
            "Along_After_km": err_rtn_after[:, 1],
            "Cross_After_km": err_rtn_after[:, 2]
        })

        csv = result_df.to_csv(
            index=False
        )

        st.download_button(
            "⬇️ Download Corrected Orbit CSV",
            csv,
            "corrected_orbit.csv",
            "text/csv"
        )


    except Exception as e:

        st.error(
            "The orbit-correction pipeline failed."
        )

        st.exception(e)
'''

path = Path("/mnt/data/app_automatic_cpf.py")
path.write_text(app_code, encoding="utf-8")

print(f"Created: {path}")
print(f"Lines: {len(app_code.splitlines())}")
