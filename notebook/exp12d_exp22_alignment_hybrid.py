# %% [markdown]
# # 00. EXP12D — EXP22 Alignment Fix, EXP15-First Selection, dan Real Hybrid Candidates
#
# Penjelasan bagian:
# Script ini merebase langsung dari EXP22/EXP12C, lalu memperbaiki alignment output dan candidate selection. Modeling tetap boleh mengadaptasi notebook teman, tetapi struktur folder, naming artifact, guardrail, dan validasi submission mengikuti standar project kita.
#
# Output yang perlu dilihat:
# Lihat `PROJECT_ROOT`, `OUT_DIR`, dan file submission/summaries di akhir. Output harus berada di `PROJECT_ROOT/outputs/exp12d_exp22_alignment_hybrid/`, bukan di `notebook/outputs/`.

# %% [markdown]
# # 01. Error Guardrails dari Eksperimen Sebelumnya
#
# Penjelasan bagian:
# Bagian ini mencatat error yang sudah pernah terjadi dan guardrail yang wajib dipertahankan: sample submission finder robust, strict/non-strict validation, output root hardening, array-safe outcome, dan final decision yang tidak mengklaim best audit.
#
# Output yang perlu dilihat:
# Di bagian akhir, cek `submission_check.csv`, `final_decision.csv`, dan log `[CHECK]`. Intermediate artifact boleh `strict=False`, sedangkan `submission_exp12d_best_safe.csv` wajib strict dan pair-consistent.

# %% [markdown]
# # 02. EXP22 Source Pipeline yang Dipreserve
#
# Penjelasan bagian:
# Bagian berikutnya mempertahankan core EXP22: data loading, country-aware features, missing imputer, last-known strength, EXP15 ordinal PMF, EXP17 W-specialist, dan submission generation. Kode modeling boleh mengadaptasi notebook teman, tetapi output path dan artifact mengikuti struktur project user.
#
# Output yang perlu dilihat:
# Cek apakah output EXP15, EXP17, dan final candidate terbentuk. D0/D1/D2 akan menyimpan semua kandidat ini supaya bisa diaudit lokal tanpa memakai GT di notebook.
# %%
# ============================================================
# 00. Setup
# ============================================================

import os
import gc
import json
import math
import random
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

from IPython.display import display

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 200)
pd.set_option("display.max_rows", 120)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

# Project root dibuat eksplisit supaya output tidak nyasar ke notebook/outputs.
def infer_project_root() -> Path:
    cwd = Path.cwd().resolve()
    if cwd.name.lower() in {"notebook", "notebooks"}:
        return cwd.parent
    for candidate in [cwd, cwd.parent]:
        if (candidate / "data").exists() or (candidate / "dataset").exists():
            return candidate
    return cwd

PROJECT_ROOT = infer_project_root()
OUTPUT_ROOT = PROJECT_ROOT / "outputs"

# Path input dibuat fleksibel, tetapi tetap menghindari outputs/submission lama.
CANDIDATE_DIRS = [
    PROJECT_ROOT,
    PROJECT_ROOT / "data",
    PROJECT_ROOT / "dataset",
    PROJECT_ROOT.parent / "data",
    PROJECT_ROOT.parent / "dataset",
    Path("."),
    Path("./data"),
    Path("../dataset"),
    Path("/mnt/data"),
]

def find_file(filename):
    """Cari file secara fleksibel.

    Menerima nama file biasa seperti `train.csv` maupun path relatif.
    Ini sengaja dibuat lebih robust supaya aman di lokal, Colab, JupyterHub,
    atau environment ChatGPT `/mnt/data`.
    """
    raw = Path(filename)
    candidates = []

    if raw.exists():
        return raw

    # Kalau user memberi "../dataset/train.csv", tetap coba path itu
    # dan juga basename-nya di semua candidate dir.
    candidate_names = list(dict.fromkeys([str(raw), raw.name]))

    for d in CANDIDATE_DIRS:
        for name in candidate_names:
            p = (d / name).resolve()
            candidates.append(str(p))
            if "outputs" in p.parts:
                continue
            if p.exists():
                return p

    raise FileNotFoundError(
        f"{filename} tidak ditemukan. Coba taruh file di folder notebook, ./data, ../dataset, atau /mnt/data. "
        f"Paths yang dicek: {candidates[:12]}"
    )

TRAIN_PATH = find_file("train.csv")
TEST_PATH = find_file("test.csv")

# ============================================================
# EXP12D selector — EXP22 alignment and candidate switchboard
# ============================================================
# EXP12D tetap memakai EXP22 sebagai source modeling, tetapi output path,
# candidate selection, dan final decision mengikuti standar project user.
# Variant bisa diubah lewat environment variable:
#   EXP12D_VARIANT=d1_exp15_first
#   EXP12D_VARIANT=d3_exp15_bias_calibrated
EXP12D_VARIANT = os.environ.get("EXP12D_VARIANT", "d1_exp15_first")

EXP12D_CONFIGS = {
    "d0_exp22_reproduction": {
        "base_ablation": "exp22_no_pseudo_no_overlay_strength",
        "strength_mode": "full",
        "candidate_policy": "d0_all_exp22_outputs",
        "description": "D0 EXP22 reproduction: save EXP15, EXP17, and final candidate.",
    },
    "d1_exp15_first": {
        "base_ablation": "exp22_no_pseudo_no_overlay_strength",
        "strength_mode": "full",
        "candidate_policy": "exp15_first",
        "description": "D1: EXP15 ordinal is treated as primary candidate.",
    },
    "d2_exp17_original": {
        "base_ablation": "exp22_no_pseudo_no_overlay_strength",
        "strength_mode": "full",
        "candidate_policy": "exp17_original",
        "description": "D2: EXP17 original final candidate for comparison.",
    },
    "d3_exp15_bias_calibrated": {
        "base_ablation": "exp22_no_pseudo_no_overlay_strength",
        "strength_mode": "full",
        "candidate_policy": "bias_calibration",
        "description": "D3: Conservative EXP15 bias/scoreline calibration.",
    },
    "d4_exp15_outcome_calibrated": {
        "base_ablation": "exp22_no_pseudo_no_overlay_strength",
        "strength_mode": "full",
        "candidate_policy": "outcome_calibration",
        "description": "D4: Outcome-aware candidate using EXP17 agreement where safe.",
    },
    "d5_agreement_hybrid": {
        "base_ablation": "exp22_no_pseudo_no_overlay_strength",
        "strength_mode": "full",
        "candidate_policy": "agreement_hybrid",
        "description": "D5: Real agreement hybrid if an EXP12B candidate is reproduced/provided.",
    },
    "d6_exact_booster": {
        "base_ablation": "exp22_no_pseudo_no_overlay_strength",
        "strength_mode": "full",
        "candidate_policy": "exact_booster",
        "description": "D6: Conservative scoreline transition booster.",
    },
    "d7_final_selected_safe": {
        "base_ablation": "exp22_no_pseudo_no_overlay_strength",
        "strength_mode": "full",
        "candidate_policy": "final_selected",
        "description": "D7: Save best-safe with clear recommended local audit list.",
    },
}
if EXP12D_VARIANT not in EXP12D_CONFIGS:
    raise ValueError(f"Unknown EXP12D_VARIANT={EXP12D_VARIANT}. Valid options: {list(EXP12D_CONFIGS)}")

EXP12D_CONFIG = EXP12D_CONFIGS[EXP12D_VARIANT]
EXP18_ABLATION_VARIANT = EXP12D_CONFIG["base_ablation"]
STRENGTH_FEATURE_MODE = EXP12D_CONFIG["strength_mode"]
EXP12D_CANDIDATE_POLICY = EXP12D_CONFIG["candidate_policy"]

# Keep these flags for compatibility with imported EXP22 code.
DECODER_CALIBRATION = False
AGREEMENT_HYBRID = False
CONSERVATIVE_EXACT_BOOSTER = False
RUN_GT_AUDIT = False

ABLATION_ROOT_DIR = OUTPUT_ROOT / "exp12d_exp22_alignment_hybrid"
OUT_DIR = ABLATION_ROOT_DIR / EXP12D_VARIANT
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"[INFO] PROJECT_ROOT = {PROJECT_ROOT}", flush=True)
print(f"[INFO] OUTPUT_ROOT = {OUTPUT_ROOT}", flush=True)
print(f"[INFO] OUT_DIR = {OUT_DIR}", flush=True)
print(f"[INFO] EXP12D_VARIANT = {EXP12D_VARIANT}", flush=True)
print(f"[INFO] EXP18_ABLATION_VARIANT = {EXP18_ABLATION_VARIANT}", flush=True)

# Backward-compatible aliases for reused EXP22/EXP12C cells.
EXP12C_VARIANT = EXP12D_VARIANT
EXP12C_CONFIG = EXP12D_CONFIG

# Konfigurasi utama.
TARGETS = ["team_goals", "opp_goals"]
ID_COL = "Id"
MATCH_COL = "match_id"

# Validasi dibuat dekat dengan test karena test dimulai 2011-08-06.
# Train asli berakhir 2011-08-04, jadi holdout 2008-2011 cukup masuk akal untuk simulasi future block.
VALID_START_DATE = "2008-01-01"

# Data sangat lama dapat membantu history, tapi untuk fitting model final biasanya modern era lebih relevan.
MODEL_TRAIN_START_DATE = "1990-01-01"  # User-requested cutoff for train data

# Kandidat start date untuk eksperimen opsional.
# Default 1995 dipilih agar fitting model lebih condong ke era modern,
# sementara fitur history tetap dibangun dari data train penuh.
MODEL_TRAIN_START_CANDIDATES = [
    "1985-01-01",
    "1990-01-01",
    "1995-01-01",
    "2000-01-01",
    "2002-01-01",
]

# Multi-fold temporal validation opsional.
# Set True kalau mau validasi lebih kuat, tapi runtime akan naik banyak karena model dilatih ulang per fold.
RUN_EXPENSIVE_MULTI_FOLD_SEARCH = False
TEMPORAL_FOLD_START_DATES = [
    "2002-01-01",
    "2005-01-01",
    "2008-01-01",
]

# Model config.
USE_CATBOOST = True
USE_LIGHTGBM = True
USE_GENDER_SEGMENT = True

# Patch 1-7.
# 1) source policy pengganti symmetrization mentah
# 2) decoder per segment
# 3) calibration per segment
# 4) auxiliary target: total goals, goal difference, outcome
# 5) cold-start features
# 6) recency-aware sample weight
# 7) optional multi-fold/start-date search
USE_ADVANCED_POSTPROCESSOR = True
USE_AUXILIARY_TARGETS = True
USE_RECENCY_SAMPLE_WEIGHT = True
RECENCY_HALFLIFE_YEARS = 10.0
RECENCY_WEIGHT_MIN = 0.60
RECENCY_WEIGHT_MAX = 1.75

# Kalau mau running lebih cepat untuk debug, ubah True.
FAST_MODE = os.environ.get(
    "EXP12D_FAST_MODE",
    os.environ.get("EXP12C_FAST_MODE", "0")
).strip() in {"1", "true", "True", "YES", "yes"}

# Optional stacking signal. Default True karena sering membantu, tapi dibuat leakage-safe.
USE_SUPERVISED_SIGNAL = True

# EXP22C: fitur last-known rank/Elo dari train-only columns.
# Default False untuk semua ablation lama; variant EXP22 menyalakannya via config.
USE_LAST_KNOWN_STRENGTH_FEATURES = False


# ------------------------------------------------------------
# Extensions added: country features, missing imputer, extreme
# clustering, pseudo-labelling, extreme rule overlay.
# ------------------------------------------------------------
USE_COUNTRY_FEATURES = True
USE_MISSING_IMPUTER = True
USE_EXTREME_CLUSTERS = True
USE_PSEUDO_LABELLING = True
USE_EXTREME_OVERLAY = True

# Ablation config. Flag akan dioverride sesuai EXP18_ABLATION_VARIANT.
EXP18_ABLATION_CONFIGS = {
    "full": {
        "USE_COUNTRY_FEATURES": True,
        "USE_MISSING_IMPUTER": True,
        "USE_EXTREME_CLUSTERS": True,
        "USE_PSEUDO_LABELLING": True,
        "USE_EXTREME_OVERLAY": True,
    },
    "no_pseudo": {
        "USE_COUNTRY_FEATURES": True,
        "USE_MISSING_IMPUTER": True,
        "USE_EXTREME_CLUSTERS": True,
        "USE_PSEUDO_LABELLING": False,
        "USE_EXTREME_OVERLAY": True,
    },
    "no_extreme_overlay": {
        "USE_COUNTRY_FEATURES": True,
        "USE_MISSING_IMPUTER": True,
        "USE_EXTREME_CLUSTERS": True,
        "USE_PSEUDO_LABELLING": True,
        "USE_EXTREME_OVERLAY": False,
    },
    "no_country_prior": {
        "USE_COUNTRY_FEATURES": False,
        "USE_MISSING_IMPUTER": True,
        "USE_EXTREME_CLUSTERS": True,
        "USE_PSEUDO_LABELLING": True,
        "USE_EXTREME_OVERLAY": False,  # overlay butuh country-prior feature seperti tier_gap
    },
    "no_extreme_cluster": {
        "USE_COUNTRY_FEATURES": True,
        "USE_MISSING_IMPUTER": True,
        "USE_EXTREME_CLUSTERS": False,
        "USE_PSEUDO_LABELLING": True,
        "USE_EXTREME_OVERLAY": False,
    },
    "no_pseudo_no_overlay": {
        "USE_COUNTRY_FEATURES": True,
        "USE_MISSING_IMPUTER": True,
        "USE_EXTREME_CLUSTERS": True,
        "USE_PSEUDO_LABELLING": False,
        "USE_EXTREME_OVERLAY": False,
    },
    "exp22_no_pseudo_no_overlay_strength": {
        # EXP22C default: champion-safe direction.
        # Pseudo OFF karena EXP21 naik ke ~3.1.
        # Extreme overlay OFF karena di validation no_pseudo overlay trigger 0 baris
        # tetapi tetap mengubah sebagian kecil test.
        # Last-known Elo/rank ON sebagai tambahan sinyal strength tanpa pseudo-label.
        "USE_COUNTRY_FEATURES": True,
        "USE_MISSING_IMPUTER": True,
        "USE_EXTREME_CLUSTERS": True,
        "USE_PSEUDO_LABELLING": False,
        "USE_EXTREME_OVERLAY": False,
        "USE_LAST_KNOWN_STRENGTH_FEATURES": True,
    },
    "minimal_safe": {
        "USE_COUNTRY_FEATURES": False,
        "USE_MISSING_IMPUTER": False,
        "USE_EXTREME_CLUSTERS": False,
        "USE_PSEUDO_LABELLING": False,
        "USE_EXTREME_OVERLAY": False,
    },
}

if EXP18_ABLATION_VARIANT not in EXP18_ABLATION_CONFIGS:
    raise ValueError(f"Unknown EXP18_ABLATION_VARIANT={EXP18_ABLATION_VARIANT}. Pilih salah satu: {list(EXP18_ABLATION_CONFIGS)}")

for _k, _v in EXP18_ABLATION_CONFIGS[EXP18_ABLATION_VARIANT].items():
    globals()[_k] = _v

# Safety dependency: extreme overlay hanya valid kalau country feature dan cluster aktif.
USE_EXTREME_OVERLAY = bool(USE_EXTREME_OVERLAY and USE_COUNTRY_FEATURES and USE_EXTREME_CLUSTERS)

print("=" * 80)
print("EXP18_ABLATION_VARIANT:", EXP18_ABLATION_VARIANT)
print("Ablation flags:")
for _k in [
    "USE_COUNTRY_FEATURES", "USE_MISSING_IMPUTER", "USE_EXTREME_CLUSTERS",
    "USE_PSEUDO_LABELLING", "USE_EXTREME_OVERLAY", "USE_LAST_KNOWN_STRENGTH_FEATURES"
]:
    print(f"  {_k}: {globals()[_k]}")
print("=" * 80)

print("TRAIN_PATH:", TRAIN_PATH.resolve())
print("TEST_PATH :", TEST_PATH.resolve())
print("OUT_DIR   :", OUT_DIR.resolve())
print("EXP12D_VARIANT:", EXP12D_VARIANT)
print("STRENGTH_FEATURE_MODE:", STRENGTH_FEATURE_MODE)
print("DECODER_CALIBRATION:", DECODER_CALIBRATION)
print("AGREEMENT_HYBRID:", AGREEMENT_HYBRID)
print("CONSERVATIVE_EXACT_BOOSTER:", CONSERVATIVE_EXACT_BOOSTER)
print("RUN_GT_AUDIT:", RUN_GT_AUDIT)

# %%
# ============================================================
# 01. Load data dan basic audit
# ============================================================

train_raw = pd.read_csv(TRAIN_PATH)
test_raw = pd.read_csv(TEST_PATH)

print("train shape:", train_raw.shape)
print("test shape :", test_raw.shape)

required_train = [ID_COL, MATCH_COL, "date", "gender", "team", "opponent", "tournament", "team_goals", "opp_goals"]
required_test = [ID_COL, MATCH_COL, "date", "gender", "team", "opponent", "tournament"]

missing_train = [c for c in required_train if c not in train_raw.columns]
missing_test = [c for c in required_test if c not in test_raw.columns]
assert not missing_train, f"Kolom wajib hilang di train: {missing_train}"
assert not missing_test, f"Kolom wajib hilang di test: {missing_test}"

train_raw["date"] = pd.to_datetime(train_raw["date"])
test_raw["date"] = pd.to_datetime(test_raw["date"])

overview = pd.DataFrame({
    "dataset": ["train", "test"],
    "rows": [len(train_raw), len(test_raw)],
    "matches": [train_raw[MATCH_COL].nunique(), test_raw[MATCH_COL].nunique()],
    "date_min": [train_raw["date"].min(), test_raw["date"].min()],
    "date_max": [train_raw["date"].max(), test_raw["date"].max()],
    "columns": [train_raw.shape[1], test_raw.shape[1]],
})
display(overview)

print("Kolom train-only yang TIDAK boleh langsung dipakai model karena tidak ada di test:")
train_only_cols = sorted(set(train_raw.columns) - set(test_raw.columns) - set(TARGETS))
print(train_only_cols)

# %% [markdown]
# ## 02. Metric AW-MAE
# 
# Cell ini memakai rumus metric sesuai yang kamu kirim. Metric ini dipakai untuk validasi lokal dan pemilihan decoder skor integer.

# %%
# ============================================================
# 02. Metric AW-MAE
# ============================================================

def get_tournament_weight(tournament: str) -> float:
    t = str(tournament).lower().strip()
    if "fifa world cup" in t or t == "world cup":
        return 2.00
    if "afc championship" in t or "afc asian cup" in t or "asian cup" in t:
        return 1.80
    if "friendly" in t:
        return 0.96
    return 1.20

EXACT_PENALTY            = 0.30
OUTCOME_PENALTY          = 0.25
GD_PENALTY               = 0.15
WRONG_OUTCOME_MULTIPLIER = 1.50
NONLINEAR_POWER          = 1.50

def _outcome(a, b):
    if a > b:
        return 1
    if a < b:
        return -1
    return 0

def official_match_loss(y_team_true, y_opp_true, y_team_pred, y_opp_pred):
    y_team_true = np.asarray(y_team_true).astype(int)
    y_opp_true  = np.asarray(y_opp_true).astype(int)
    y_team_pred = np.asarray(y_team_pred).astype(int)
    y_opp_pred  = np.asarray(y_opp_pred).astype(int)

    base_mae = (np.abs(y_team_true - y_team_pred) + np.abs(y_opp_true - y_opp_pred)) / 2.0

    exact_hit = (y_team_true == y_team_pred) & (y_opp_true == y_opp_pred)

    true_outcome = np.vectorize(_outcome)(y_team_true, y_opp_true)
    pred_outcome = np.vectorize(_outcome)(y_team_pred, y_opp_pred)
    outcome_hit = (true_outcome == pred_outcome)

    gd_hit = ((y_team_true - y_opp_true) == (y_team_pred - y_opp_pred))

    penalty = (
        (~exact_hit).astype(float) * EXACT_PENALTY
        + (~outcome_hit).astype(float) * OUTCOME_PENALTY
        + (~gd_hit).astype(float) * GD_PENALTY
    )

    raw_error = base_mae + penalty
    raw_error = np.where(outcome_hit, raw_error, raw_error * WRONG_OUTCOME_MULTIPLIER)

    return raw_error ** NONLINEAR_POWER

def awmae_score(y_team_true, y_opp_true, y_team_pred, y_opp_pred, weights=None) -> float:
    losses = official_match_loss(y_team_true, y_opp_true, y_team_pred, y_opp_pred)
    if weights is None:
        weights = np.ones(len(losses), dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)
    return float(np.average(losses, weights=weights))

print("AW-MAE metric defined ✓")

# %% [markdown]
# ## 02b. Country features, funding, fanbase
# 
# Hand-curated country priors plus a missing-data imputer fit dari train_raw. Dipakai oleh extended `make_feature_frames` di section 03b.

# %%
# ============================================================
# 02b. Country knowledge: football tier, funding proxy, fanbase
# ============================================================
#
# Tabel domain knowledge ini dipakai sebagai prior tambahan di samping
# fitur numerik mentah seperti population dan gdp_per_capita yang sering
# missing atau noisy di test block 2011-2026.
#
# Tier dipisah men vs women karena distribusi kekuatan dan funding sangat
# berbeda lintas gender. Contoh: USA W ada di tier 1 sejak lama tetapi USA M
# baru tier 2-3. Brazil M tier 1 tapi Brazil W relatif tier 2 karena
# infrastruktur liga women baru berkembang.
#
# Skala: 1 = elite, 5 = developing/early-stage. Default 4 untuk negara
# yang tidak listed agar bias condong ke "biasa-biasa saja", bukan ke elite.

# Top-tier men senior football, kasar tapi konsisten dengan FIFA ranking 2010-2024
COUNTRY_TIER_M = {
    # AFC
    "Japan": 2, "South Korea": 2, "Iran": 2, "Australia": 2,
    "Saudi Arabia": 3, "Qatar": 3, "Iraq": 3, "United Arab Emirates": 3,
    "China PR": 3, "Uzbekistan": 3, "Jordan": 3, "Oman": 4,
    "Bahrain": 4, "Vietnam": 4, "Thailand": 4, "Kuwait": 4,
    "India": 5, "Indonesia": 5, "Malaysia": 5, "Philippines": 5,
    # CONMEBOL
    "Brazil": 1, "Argentina": 1, "Uruguay": 2, "Colombia": 2,
    "Chile": 2, "Peru": 3, "Ecuador": 3, "Paraguay": 3,
    "Venezuela": 3, "Bolivia": 4,
    # UEFA top
    "Germany": 1, "Spain": 1, "France": 1, "Italy": 1, "England": 1,
    "Netherlands": 1, "Portugal": 1, "Belgium": 1,
    "Croatia": 2, "Denmark": 2, "Switzerland": 2, "Sweden": 2,
    "Poland": 2, "Czech Republic": 3, "Czechia": 3, "Austria": 3,
    "Serbia": 3, "Ukraine": 3, "Russia": 2, "Turkey": 3,
    "Wales": 3, "Scotland": 3, "Republic of Ireland": 3, "Ireland": 3,
    "Northern Ireland": 4, "Norway": 3, "Finland": 4, "Iceland": 4,
    "Greece": 3, "Hungary": 3, "Romania": 3, "Bulgaria": 4,
    "Slovakia": 4, "Slovenia": 4, "Albania": 4, "North Macedonia": 4,
    "Bosnia and Herzegovina": 4, "Montenegro": 4, "Estonia": 5,
    "Latvia": 5, "Lithuania": 5, "Cyprus": 5, "Malta": 5,
    "Luxembourg": 5, "Liechtenstein": 5, "Andorra": 5, "San Marino": 5,
    "Faroe Islands": 5, "Gibraltar": 5,
    # CAF
    "Senegal": 2, "Morocco": 2, "Tunisia": 3, "Algeria": 3, "Egypt": 3,
    "Nigeria": 2, "Cameroon": 3, "Ghana": 3, "Ivory Coast": 3,
    "Côte d'Ivoire": 3, "Mali": 3, "Burkina Faso": 4, "South Africa": 3,
    "Kenya": 5, "Tanzania": 5, "Uganda": 5, "Sudan": 5, "Ethiopia": 5,
    "Zambia": 5, "Zimbabwe": 5, "Madagascar": 5, "Mozambique": 5,
    "Angola": 5, "DR Congo": 4, "Congo DR": 4, "Cape Verde": 4,
    # CONCACAF
    "United States": 2, "Mexico": 2, "Canada": 3, "Costa Rica": 3,
    "Jamaica": 4, "Honduras": 4, "Panama": 4, "El Salvador": 5,
    "Guatemala": 5, "Trinidad and Tobago": 5, "Haiti": 5, "Cuba": 5,
    # OFC
    "New Zealand": 4, "Fiji": 5, "Papua New Guinea": 5,
    "New Caledonia": 5, "Solomon Islands": 5, "Tahiti": 5, "Vanuatu": 5,
}

# Women senior football tier
COUNTRY_TIER_W = {
    # Top tier W
    "United States": 1, "Germany": 1, "Sweden": 1, "Japan": 1,
    "England": 1, "Spain": 1, "France": 2, "Netherlands": 2,
    "Norway": 2, "Brazil": 2, "Canada": 2, "Australia": 2,
    "China PR": 3, "China": 3, "Denmark": 2, "Iceland": 3,
    "Italy": 3, "Portugal": 3, "Belgium": 3, "Switzerland": 3,
    "Austria": 3, "Republic of Ireland": 3, "Ireland": 3, "Scotland": 3,
    "Wales": 4, "Northern Ireland": 4, "Finland": 3, "Russia": 3,
    "Ukraine": 4, "Poland": 4, "Czech Republic": 4, "Czechia": 4,
    # Mid
    "Argentina": 3, "Colombia": 3, "Chile": 4, "Mexico": 3,
    "Costa Rica": 4, "Jamaica": 4, "Nigeria": 3, "South Africa": 3,
    "Cameroon": 4, "Morocco": 4, "Ghana": 4, "Zambia": 4,
    "South Korea": 3, "North Korea": 2, "Korea DPR": 2, "Vietnam": 4,
    "Thailand": 4, "Philippines": 4, "New Zealand": 3,
    # Developing W programs
    "Saudi Arabia": 5, "Iran": 5, "Iraq": 5, "Qatar": 5,
    "United Arab Emirates": 5, "Egypt": 5, "Algeria": 5, "Tunisia": 5,
    "Indonesia": 5, "India": 5, "Malaysia": 5, "Pakistan": 5,
    "Senegal": 5, "Kenya": 5, "Tanzania": 5, "Uganda": 5,
    "Bolivia": 5, "Venezuela": 4, "Paraguay": 5, "Peru": 5,
    "Ecuador": 5, "Uruguay": 4,
    "Hungary": 4, "Romania": 4, "Bulgaria": 5, "Greece": 5,
    "Croatia": 4, "Serbia": 4, "Turkey": 5, "Israel": 5,
    "Slovakia": 5, "Slovenia": 5, "Albania": 5, "Bosnia and Herzegovina": 5,
    "Estonia": 5, "Latvia": 5, "Lithuania": 5, "Cyprus": 5,
    "Malta": 5, "Luxembourg": 5, "Faroe Islands": 5, "Kazakhstan": 5,
    "Belarus": 5,
}

# Fanbase intensity / popularity domestic
COUNTRY_FANBASE = {
    "Brazil": 1, "Argentina": 1, "England": 1, "Germany": 1, "Italy": 1,
    "Spain": 1, "Mexico": 1, "France": 2, "Netherlands": 2, "Portugal": 2,
    "Turkey": 1, "Egypt": 2, "Senegal": 2, "Nigeria": 1, "Ghana": 2,
    "Ivory Coast": 2, "Côte d'Ivoire": 2, "Morocco": 2, "Tunisia": 2,
    "Algeria": 2, "Cameroon": 2, "Iran": 2, "Iraq": 2, "Saudi Arabia": 2,
    "Japan": 3, "South Korea": 3, "Australia": 3, "Uruguay": 2, "Chile": 2,
    "Colombia": 2, "Peru": 2, "Paraguay": 3, "Ecuador": 3,
    "United States": 4, "Canada": 4, "Costa Rica": 3, "Jamaica": 3,
    "Honduras": 3, "Panama": 3, "Croatia": 2, "Serbia": 2,
    "Russia": 2, "Ukraine": 3, "Poland": 2, "Czech Republic": 3,
    "Czechia": 3, "Austria": 3, "Switzerland": 3, "Belgium": 2,
    "Norway": 3, "Sweden": 3, "Denmark": 3, "Finland": 4, "Iceland": 4,
    "Republic of Ireland": 2, "Ireland": 2, "Scotland": 2, "Wales": 3,
    "China PR": 4, "China": 4, "Vietnam": 3, "Thailand": 3, "Indonesia": 2,
    "Malaysia": 4, "Philippines": 4, "India": 3, "New Zealand": 4,
    "Bolivia": 3, "Venezuela": 4, "South Africa": 3,
}

# Confederation-level fallback popularity (loose proxy)
CONFED_FANBASE = {
    "UEFA": 2, "CONMEBOL": 1, "CONCACAF": 3, "AFC": 3, "CAF": 2, "OFC": 4,
}

DEFAULT_TIER_M = 4
DEFAULT_TIER_W = 4
DEFAULT_FANBASE = 4
DEFAULT_TIER_DIFF = 0  # untuk pasangan yang sama-sama unknown


def lookup_country_tier(country, gender):
    """Tier 1-5 untuk negara per gender. NaN dan negara unknown jadi default."""
    if country is None or (isinstance(country, float) and np.isnan(country)):
        return DEFAULT_TIER_W if str(gender).upper() == "W" else DEFAULT_TIER_M
    name = str(country).strip()
    if str(gender).upper() == "W":
        return COUNTRY_TIER_W.get(name, DEFAULT_TIER_W)
    return COUNTRY_TIER_M.get(name, DEFAULT_TIER_M)


def lookup_country_fanbase(country, confederation=None):
    """Fanbase intensity 1-5. Kalau negara unknown, fallback ke confederation."""
    if country is not None and not (isinstance(country, float) and np.isnan(country)):
        name = str(country).strip()
        if name in COUNTRY_FANBASE:
            return COUNTRY_FANBASE[name]
    if confederation is not None and not (isinstance(confederation, float) and np.isnan(confederation)):
        conf = str(confederation).strip()
        return CONFED_FANBASE.get(conf, DEFAULT_FANBASE)
    return DEFAULT_FANBASE


def federation_funding_proxy(population, gdp_per_capita, tier):
    """Proxy funding pakai GDP total negara, dipotong oleh tier (sebagai
    representasi share dari GDP yang masuk ke federasi sepak bola).

    Hasil di-log untuk meredam outlier negara raksasa seperti China/India.
    """
    pop = pd.to_numeric(pd.Series([population]), errors="coerce").iloc[0]
    gdp = pd.to_numeric(pd.Series([gdp_per_capita]), errors="coerce").iloc[0]
    if pd.isna(pop) or pd.isna(gdp) or pop <= 0 or gdp <= 0:
        return np.nan
    # Tier 1 dapat share lebih besar; tier 5 share lebih kecil.
    tier_share = {1: 1.0, 2: 0.7, 3: 0.45, 4: 0.25, 5: 0.12}.get(int(tier), 0.25)
    total_gdp = pop * gdp
    return float(np.log1p(total_gdp * tier_share))


print(
    "Country priors loaded ✓ |",
    "M tiers:", len(COUNTRY_TIER_M),
    "| W tiers:", len(COUNTRY_TIER_W),
    "| Fanbase:", len(COUNTRY_FANBASE),
)

# %% [markdown]
# ## 02c. Missing-data imputer
# 
# Country-decade median fallback untuk kolom yang sering kosong di test era 2011-2026 (population, gdp_per_capita, temperature_venue, altitude_venue, distance_travel). Tambah missing-indicator flags.

# %%
# ============================================================
# 02c. Missing data strategy untuk test era 2011-2026
# ============================================================
#
# Beberapa kolom test sangat sparse di tahun-tahun belakangan, terutama:
#   - temperature_venue, altitude_venue (sensor data tidak lengkap)
#   - population_team / population_opp (kumpulan source berbeda per era)
#   - gdp_per_capita_team / gdp_per_capita_opp (negara baru / nama berubah)
#   - distance_travel_team / distance_travel_opp (venue_country missing)
#
# Strategi:
# 1) Country-level fill: median per (country, decade) dari train.
# 2) Decade-level fallback ketika country tidak ada di train.
# 3) Global median fallback paling akhir.
# 4) Sentinel cleanup (-9999 -> NaN).
# 5) Tambah missing-indicator flag agar model bisa belajar pola "missingness".
#
# Imputer harus di-fit hanya pada train, lalu di-apply ke train dan test.
# Kalau di-fit di test juga, leakage temporal akan masuk ke validation.

# Kolom yang akan diberi country-level + decade fill
NUMERIC_COUNTRY_COLS = [
    "population_team", "population_opp",
    "gdp_per_capita_team", "gdp_per_capita_opp",
    "distance_travel_team", "distance_travel_opp",
]

# Kolom venue-level
NUMERIC_VENUE_COLS = ["temperature_venue", "altitude_venue"]

SENTINEL_VALUES = {
    "altitude_venue": [-9999, -999],
    "temperature_venue": [-9999, -999],
}


def _to_numeric_clean(series, sentinels=None):
    s = pd.to_numeric(series, errors="coerce")
    if sentinels:
        for sv in sentinels:
            s = s.where(s != sv, np.nan)
    return s


def fit_missing_imputer(train_df):
    """Belajar median per (country, decade) dari train.

    Yang dibangun:
      - country_decade_median: dict[(col, country, decade)] -> value
      - country_median:        dict[(col, country)]         -> value
      - decade_median:         dict[(col, decade)]          -> value
      - global_median:         dict[col]                    -> value
    """
    tr = train_df.copy()
    tr["date"] = pd.to_datetime(tr["date"], errors="coerce")
    tr["__decade"] = ((tr["date"].dt.year // 10) * 10).astype("Int64")

    # Setiap kolom punya kolom country pasangannya (team-side vs opp-side vs venue)
    country_lookup = {
        "population_team": "team",
        "gdp_per_capita_team": "team",
        "distance_travel_team": "team",
        "population_opp": "opponent",
        "gdp_per_capita_opp": "opponent",
        "distance_travel_opp": "opponent",
        "temperature_venue": "venue_country",
        "altitude_venue": "venue_country",
    }

    country_decade_median = {}
    country_median = {}
    decade_median = {}
    global_median = {}

    all_cols = NUMERIC_COUNTRY_COLS + NUMERIC_VENUE_COLS
    for col in all_cols:
        if col not in tr.columns:
            continue
        country_col = country_lookup.get(col)
        sentinels = SENTINEL_VALUES.get(col)
        clean = _to_numeric_clean(tr[col], sentinels)

        gm = float(clean.median()) if clean.notna().any() else np.nan
        global_median[col] = gm

        if country_col is not None and country_col in tr.columns:
            for (cn, dec), grp in tr.groupby([country_col, "__decade"], dropna=False):
                v = clean.loc[grp.index]
                if v.notna().any():
                    country_decade_median[(col, str(cn), int(dec) if pd.notna(dec) else -1)] = float(v.median())
            for cn, grp in tr.groupby(country_col, dropna=False):
                v = clean.loc[grp.index]
                if v.notna().any():
                    country_median[(col, str(cn))] = float(v.median())

        for dec, grp in tr.groupby("__decade", dropna=False):
            v = clean.loc[grp.index]
            if v.notna().any():
                decade_median[(col, int(dec) if pd.notna(dec) else -1)] = float(v.median())

    return {
        "country_decade_median": country_decade_median,
        "country_median": country_median,
        "decade_median": decade_median,
        "global_median": global_median,
        "country_lookup": country_lookup,
    }


def apply_missing_imputer(df, imputer):
    """Apply imputer hasil fit_missing_imputer ke df (train atau test).

    Tambah juga kolom flag __was_missing untuk tiap kolom imputed.
    """
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["__decade"] = ((out["date"].dt.year // 10) * 10).astype("Int64")

    country_lookup = imputer["country_lookup"]
    cd_med = imputer["country_decade_median"]
    c_med = imputer["country_median"]
    d_med = imputer["decade_median"]
    g_med = imputer["global_median"]

    for col in NUMERIC_COUNTRY_COLS + NUMERIC_VENUE_COLS:
        if col not in out.columns:
            continue
        sentinels = SENTINEL_VALUES.get(col)
        original = _to_numeric_clean(out[col], sentinels)
        miss_mask = original.isna()
        out[f"{col}__was_missing"] = miss_mask.astype(int)

        if not miss_mask.any():
            out[col] = original.values
            continue

        country_col = country_lookup.get(col)
        decade_arr = out["__decade"].values

        if country_col is not None and country_col in out.columns:
            country_arr = out[country_col].astype(str).values
            keys_cd = list(zip([col] * len(out), country_arr, [int(d) if pd.notna(d) else -1 for d in decade_arr]))
            keys_c = list(zip([col] * len(out), country_arr))
        else:
            keys_cd = [None] * len(out)
            keys_c = [None] * len(out)
        keys_d = list(zip([col] * len(out), [int(d) if pd.notna(d) else -1 for d in decade_arr]))

        filled = original.values.astype(float).copy()
        gm = g_med.get(col, np.nan)

        miss_idx = np.where(miss_mask.values)[0]
        for i in miss_idx:
            v = np.nan
            if keys_cd[i] is not None:
                v = cd_med.get(keys_cd[i], np.nan)
            if pd.isna(v) and keys_c[i] is not None:
                v = c_med.get(keys_c[i], np.nan)
            if pd.isna(v):
                v = d_med.get(keys_d[i], np.nan)
            if pd.isna(v):
                v = gm
            if pd.isna(v):
                v = 0.0
            filled[i] = v

        out[col] = filled

    out = out.drop(columns="__decade", errors="ignore")
    return out


# Fit imputer dari train_raw (gunakan semua train sebelum cutoff diterapkan,
# karena imputer ini level fitur dasar, bukan model fitting).
missing_imputer = fit_missing_imputer(train_raw)

print("Missing-data imputer fit ✓")
print("Global medians sample:", {k: round(v, 2) if not pd.isna(v) else None
                                  for k, v in list(missing_imputer["global_median"].items())[:6]})

# %% [markdown]
# ## 03. Feature engineering leakage-safe
# 
# Prinsip yang dipakai:
# 
# - Model hanya memakai kolom yang tersedia juga di `test.csv`.
# - Fitur historis untuk sebuah match hanya memakai match sebelum blok prediksi.
# - Dalam blok validasi/test, label tidak diketahui, jadi fitur gol/form tidak di-update memakai target hidden.
# - Namun jadwal di blok prediksi sudah diketahui, jadi fitur seperti `days_since_prev_seen` boleh diperbarui berdasarkan match sebelumnya di blok yang sama.
# - Riwayat tim dipisah berdasarkan `gender`, supaya tim pria dan wanita tidak saling tercampur.

# %%
# ============================================================
# 03. Feature engineering helpers
# ============================================================

COMMON_INPUT_COLS = [c for c in test_raw.columns if c in train_raw.columns]
TRAIN_INPUT_COLS = COMMON_INPUT_COLS + TARGETS

print("Jumlah common input cols:", len(COMMON_INPUT_COLS))
print(COMMON_INPUT_COLS)

def points_from_score(gf, ga):
    if gf > ga:
        return 3.0
    if gf == ga:
        return 1.0
    return 0.0

def outcome_from_score(gf, ga):
    if gf > ga:
        return 1.0
    if gf < ga:
        return -1.0
    return 0.0

def add_base_features(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    df["year"] = df["date"].dt.year.astype(int)
    df["month"] = df["date"].dt.month.astype(int)
    df["dayofyear"] = df["date"].dt.dayofyear.astype(int)
    df["decade"] = ((df["year"] // 10) * 10).astype(int)

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dayofyear_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 366)
    df["dayofyear_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 366)

    df["is_women_match"] = df["gender"].astype(str).str.upper().eq("W").astype(int)

    if {"confederation_team", "confederation_opp"}.issubset(df.columns):
        team_conf = df["confederation_team"].astype("string").fillna("__MISSING__")
        opp_conf = df["confederation_opp"].astype("string").fillna("__MISSING__")
        df["same_confederation"] = (team_conf == opp_conf).astype(int)
        df["confederation_pair"] = team_conf.astype(str) + "_vs_" + opp_conf.astype(str)
        df["confederation_pair_sorted"] = np.where(
            team_conf.astype(str) <= opp_conf.astype(str),
            team_conf.astype(str) + "_vs_" + opp_conf.astype(str),
            opp_conf.astype(str) + "_vs_" + team_conf.astype(str),
        )
    else:
        df["same_confederation"] = 0
        df["confederation_pair"] = "__MISSING__"
        df["confederation_pair_sorted"] = "__MISSING__"

    if {"team", "venue_country"}.issubset(df.columns):
        df["is_host_team"] = (df["team"].astype(str) == df["venue_country"].astype(str)).astype(int)
    else:
        df["is_host_team"] = 0

    tour = df["tournament"].astype(str).str.lower()
    df["tournament_weight"] = df["tournament"].map(get_tournament_weight).astype(float)
    df["is_friendly"] = tour.str.contains("friendly", na=False).astype(int)
    df["is_world_cup"] = tour.str.contains("fifa world cup|world cup", regex=True, na=False).astype(int)
    df["is_asian_cup"] = tour.str.contains("afc championship|afc asian cup|asian cup", regex=True, na=False).astype(int)
    df["is_qualifier"] = tour.str.contains("qualification|qualifier|qualifying", regex=True, na=False).astype(int)
    df["is_cup_or_championship"] = tour.str.contains(
        "cup|championship|league|games|nations|tournament|copa|euro|afcon|olympic",
        regex=True,
        na=False,
    ).astype(int)
    df["is_nations_league"] = tour.str.contains("nations league", regex=False, na=False).astype(int)
    df["is_olympic"] = tour.str.contains("olympic", regex=False, na=False).astype(int)
    df["is_regional_cup"] = tour.str.contains(
        "asian cup|afc championship|afcon|copa|euro|gold cup|concacaf|ofc|caf|copa america",
        regex=True,
        na=False,
    ).astype(int)

    def tier_name(t):
        t = str(t).lower()
        if "friendly" in t:
            return "friendly"
        if "fifa world cup" in t or t == "world cup":
            return "world_cup"
        if "asian cup" in t or "afc championship" in t:
            return "asian_cup"
        if "qualification" in t or "qualifier" in t or "qualifying" in t:
            return "qualifier"
        if any(k in t for k in ["cup", "championship", "league", "games", "nations", "tournament", "euro", "copa", "afcon"]):
            return "competition"
        return "other"

    df["tournament_tier"] = df["tournament"].map(tier_name)

    pair_cols = [
        ("population_team", "population_opp", "population"),
        ("gdp_per_capita_team", "gdp_per_capita_opp", "gdp"),
        ("distance_travel_team", "distance_travel_opp", "distance"),
    ]
    for a, b, name in pair_cols:
        if a in df.columns and b in df.columns:
            av = pd.to_numeric(df[a], errors="coerce").where(pd.to_numeric(df[a], errors="coerce") >= 0)
            bv = pd.to_numeric(df[b], errors="coerce").where(pd.to_numeric(df[b], errors="coerce") >= 0)
            df[f"{name}_diff"] = av - bv
            df[f"{name}_ratio"] = av / bv.replace(0, np.nan)
            df[f"log_{name}_team"] = np.log1p(av)
            df[f"log_{name}_opp"] = np.log1p(bv)
            df[f"log_{name}_diff"] = np.log1p(av) - np.log1p(bv)

    if "altitude_venue" in df.columns:
        df["altitude_venue_clean"] = pd.to_numeric(df["altitude_venue"], errors="coerce").replace(-9999, np.nan)

    return df

def build_temporal_history_features(train_df, pred_df, windows=(3, 5, 10, 20)):
    """Buat fitur historis dari train_df untuk pred_df.

    train_df harus punya target.
    pred_df boleh train validation atau test, tapi targetnya tidak dipakai.
    """
    train_work = train_df.copy()
    pred_work = pred_df.copy()

    train_work["__dataset"] = "train"
    pred_work["__dataset"] = "pred"

    if "team_goals" not in pred_work.columns:
        pred_work["team_goals"] = np.nan
    if "opp_goals" not in pred_work.columns:
        pred_work["opp_goals"] = np.nan

    # Input yang masuk ke fungsi ini sudah dibatasi ke common/test-feasible columns.
    # Setelah add_base_features, ada kolom engineered baru seperti tournament_tier yang juga perlu ikut.
    base_cols = list(dict.fromkeys(list(train_work.columns) + list(pred_work.columns)))
    for col in base_cols:
        if col not in train_work.columns:
            train_work[col] = np.nan
        if col not in pred_work.columns:
            pred_work[col] = np.nan

    work = pd.concat([train_work[base_cols], pred_work[base_cols]], ignore_index=True)
    work["date"] = pd.to_datetime(work["date"])
    work["__orig_idx"] = np.arange(len(work))
    work["__dataset_order"] = work["__dataset"].map({"train": 0, "pred": 1}).astype(int)
    work = work.sort_values(["date", MATCH_COL, "__dataset_order", ID_COL]).reset_index(drop=True)

    max_w = int(max(windows))

    def new_hist():
        return {
            "gf": [], "ga": [], "gd": [], "pts": [], "win": [], "outcome": [],
            "known_n": 0,
            "seen_n": 0,
            "gf_sum": 0.0, "ga_sum": 0.0, "gd_sum": 0.0, "pts_sum": 0.0,
            "last_known_date": None,
            "last_seen_date": None,
            "ewm_gf_035": np.nan, "ewm_ga_035": np.nan, "ewm_gd_035": np.nan, "ewm_pts_035": np.nan,
            "ewm_gf_015": np.nan, "ewm_ga_015": np.nan, "ewm_gd_015": np.nan, "ewm_pts_015": np.nan,
        }

    team_hist = defaultdict(new_hist)
    h2h_hist = defaultdict(new_hist)
    global_hist = defaultdict(new_hist)

    feat_rows = []
    pending = []
    current_match = None

    def team_key(gender, team):
        return (str(gender), str(team))

    def pair_key(gender, team, opponent):
        return (str(gender), str(team), str(opponent))

    def global_key(gender, tier):
        return (str(gender), str(tier))

    def mean_last(vals, w):
        if len(vals) == 0:
            return np.nan
        arr = vals[-w:]
        return float(sum(arr) / len(arr))

    def ewm_update(old, val, alpha):
        return val if pd.isna(old) else alpha * val + (1 - alpha) * old

    def add_hist_features(f, prefix, H):
        f[f"{prefix}_known_n"] = H["known_n"]
        f[f"{prefix}_seen_n"] = H["seen_n"]
        for w in windows:
            for metric in ("gf", "ga", "gd", "pts", "win", "outcome"):
                f[f"{prefix}_{metric}_mean_l{w}"] = mean_last(H[metric], w)

        n = H["known_n"]
        f[f"{prefix}_gf_mean_exp"] = H["gf_sum"] / n if n else np.nan
        f[f"{prefix}_ga_mean_exp"] = H["ga_sum"] / n if n else np.nan
        f[f"{prefix}_gd_mean_exp"] = H["gd_sum"] / n if n else np.nan
        f[f"{prefix}_pts_mean_exp"] = H["pts_sum"] / n if n else np.nan

        f[f"{prefix}_gf_ewm_035"] = H["ewm_gf_035"]
        f[f"{prefix}_ga_ewm_035"] = H["ewm_ga_035"]
        f[f"{prefix}_gd_ewm_035"] = H["ewm_gd_035"]
        f[f"{prefix}_pts_ewm_035"] = H["ewm_pts_035"]
        f[f"{prefix}_gf_ewm_015"] = H["ewm_gf_015"]
        f[f"{prefix}_ga_ewm_015"] = H["ewm_ga_015"]
        f[f"{prefix}_gd_ewm_015"] = H["ewm_gd_015"]
        f[f"{prefix}_pts_ewm_015"] = H["ewm_pts_015"]

    def update_schedule_only(H, date):
        H["seen_n"] += 1
        H["last_seen_date"] = date

    def update_known_result(H, date, gf, ga):
        gd = gf - ga
        pts = points_from_score(gf, ga)
        win = 1.0 if gf > ga else 0.0
        out = outcome_from_score(gf, ga)

        for key, val in (("gf", gf), ("ga", ga), ("gd", gd), ("pts", pts), ("win", win), ("outcome", out)):
            H[key].append(float(val))
            if len(H[key]) > max_w:
                H[key].pop(0)

        H["known_n"] += 1
        H["gf_sum"] += float(gf)
        H["ga_sum"] += float(ga)
        H["gd_sum"] += float(gd)
        H["pts_sum"] += float(pts)
        H["last_known_date"] = date

        for alpha, tag in ((0.35, "035"), (0.15, "015")):
            for metric, val in (("gf", gf), ("ga", ga), ("gd", gd), ("pts", pts)):
                k = f"ewm_{metric}_{tag}"
                H[k] = ewm_update(H[k], float(val), alpha)

    def process_match(rows):
        # compute features first for every row in this match
        for row in rows:
            idx, match_id, date, gender, team, opponent, tier, gf_cur, ga_cur, dataset = row

            tk = team_key(gender, team)
            ok = team_key(gender, opponent)
            hk = pair_key(gender, team, opponent)
            gk = global_key(gender, tier)

            h = team_hist[tk]
            oh = team_hist[ok]
            hh = h2h_hist[hk]
            gh = global_hist[gk]

            f = {"__sort_idx": idx}

            f["days_since_team_prev_known"] = np.nan if h["last_known_date"] is None else (date - h["last_known_date"]).days
            f["days_since_opp_prev_known"] = np.nan if oh["last_known_date"] is None else (date - oh["last_known_date"]).days
            f["days_since_team_prev_seen"] = np.nan if h["last_seen_date"] is None else (date - h["last_seen_date"]).days
            f["days_since_opp_prev_seen"] = np.nan if oh["last_seen_date"] is None else (date - oh["last_seen_date"]).days

            add_hist_features(f, "team", h)
            add_hist_features(f, "opp", oh)
            add_hist_features(f, "h2h", hh)
            add_hist_features(f, "global_gender_tier", gh)

            feat_rows.append(f)

        # update histories after features are computed
        for row in rows:
            idx, match_id, date, gender, team, opponent, tier, gf_cur, ga_cur, dataset = row

            tk = team_key(gender, team)
            hk = pair_key(gender, team, opponent)
            gk = global_key(gender, tier)

            # Schedule info is known for train, validation, and test block.
            update_schedule_only(team_hist[tk], date)
            update_schedule_only(h2h_hist[hk], date)
            update_schedule_only(global_hist[gk], date)

            # Goal/result info only from train block.
            if dataset != "train" or pd.isna(gf_cur) or pd.isna(ga_cur):
                continue

            gf = float(gf_cur)
            ga = float(ga_cur)

            update_known_result(team_hist[tk], date, gf, ga)
            update_known_result(h2h_hist[hk], date, gf, ga)
            update_known_result(global_hist[gk], date, gf, ga)

    needed = [MATCH_COL, "date", "gender", "team", "opponent", "tournament_tier", "team_goals", "opp_goals", "__dataset"]
    records = work[needed].itertuples(index=True, name=None)

    for rec in records:
        idx, match_id, date, gender, team, opponent, tier, gf_cur, ga_cur, dataset = rec
        if current_match is None:
            current_match = match_id
        if match_id != current_match:
            process_match(pending)
            pending = []
            current_match = match_id
        pending.append(rec)

    if pending:
        process_match(pending)

    feat_df = pd.DataFrame(feat_rows).sort_values("__sort_idx").drop(columns="__sort_idx").reset_index(drop=True)
    out = pd.concat([work.reset_index(drop=True), feat_df], axis=1)

    for w in windows:
        out[f"attack_defense_goal_proxy_l{w}"] = 0.55 * out[f"team_gf_mean_l{w}"] + 0.45 * out[f"opp_ga_mean_l{w}"]
        out[f"opp_attack_defense_goal_proxy_l{w}"] = 0.55 * out[f"opp_gf_mean_l{w}"] + 0.45 * out[f"team_ga_mean_l{w}"]
        out[f"goal_pressure_diff_l{w}"] = out[f"attack_defense_goal_proxy_l{w}"] - out[f"opp_attack_defense_goal_proxy_l{w}"]
        out[f"form_gd_diff_l{w}"] = out[f"team_gd_mean_l{w}"] - out[f"opp_gd_mean_l{w}"]
        out[f"points_diff_l{w}"] = out[f"team_pts_mean_l{w}"] - out[f"opp_pts_mean_l{w}"]
        out[f"winrate_diff_l{w}"] = out[f"team_win_mean_l{w}"] - out[f"opp_win_mean_l{w}"]
        out[f"h2h_goal_pressure_l{w}"] = out[f"h2h_gf_mean_l{w}"] - out[f"h2h_ga_mean_l{w}"]
        out[f"global_goal_env_l{w}"] = out[f"global_gender_tier_gf_mean_l{w}"] + out[f"global_gender_tier_ga_mean_l{w}"]

    out["attack_defense_goal_proxy_ewm035"] = 0.55 * out["team_gf_ewm_035"] + 0.45 * out["opp_ga_ewm_035"]
    out["opp_attack_defense_goal_proxy_ewm035"] = 0.55 * out["opp_gf_ewm_035"] + 0.45 * out["team_ga_ewm_035"]
    out["goal_pressure_diff_ewm035"] = out["attack_defense_goal_proxy_ewm035"] - out["opp_attack_defense_goal_proxy_ewm035"]

    out["attack_defense_goal_proxy_ewm015"] = 0.55 * out["team_gf_ewm_015"] + 0.45 * out["opp_ga_ewm_015"]
    out["opp_attack_defense_goal_proxy_ewm015"] = 0.55 * out["opp_gf_ewm_015"] + 0.45 * out["team_ga_ewm_015"]
    out["goal_pressure_diff_ewm015"] = out["attack_defense_goal_proxy_ewm015"] - out["opp_attack_defense_goal_proxy_ewm015"]

    out["attack_defense_goal_proxy_exp"] = 0.55 * out["team_gf_mean_exp"] + 0.45 * out["opp_ga_mean_exp"]
    out["opp_attack_defense_goal_proxy_exp"] = 0.55 * out["opp_gf_mean_exp"] + 0.45 * out["team_ga_mean_exp"]
    out["strength_diff_exp"] = (out["team_gf_mean_exp"] - out["team_ga_mean_exp"]) - (out["opp_gf_mean_exp"] - out["opp_ga_mean_exp"])

    # ------------------------------------------------------------
    # Patch 5: cold-start dan history sufficiency features.
    # Ini penting untuk test era modern, terutama W, karena banyak tim punya history tipis.
    # ------------------------------------------------------------
    for prefix in ["team", "opp", "h2h"]:
        known_col = f"{prefix}_known_n"
        seen_col = f"{prefix}_seen_n"
        out[f"{prefix}_known_log1p"] = np.log1p(pd.to_numeric(out[known_col], errors="coerce").fillna(0))
        out[f"{prefix}_seen_log1p"] = np.log1p(pd.to_numeric(out[seen_col], errors="coerce").fillna(0))
        out[f"{prefix}_is_cold_0"] = (pd.to_numeric(out[known_col], errors="coerce").fillna(0) == 0).astype(int)
        out[f"{prefix}_is_cold_lte2"] = (pd.to_numeric(out[known_col], errors="coerce").fillna(0) <= 2).astype(int)
        out[f"{prefix}_is_cold_lte5"] = (pd.to_numeric(out[known_col], errors="coerce").fillna(0) <= 5).astype(int)
        out[f"{prefix}_history_bucket"] = pd.cut(
            pd.to_numeric(out[known_col], errors="coerce").fillna(0),
            bins=[-1, 0, 2, 5, 10, 25, 10**9],
            labels=["0", "1_2", "3_5", "6_10", "11_25", "26_plus"],
        ).astype(str)

    out["both_team_opp_cold_lte2"] = ((out["team_known_n"] <= 2) & (out["opp_known_n"] <= 2)).astype(int)
    out["either_team_opp_cold_lte2"] = ((out["team_known_n"] <= 2) | (out["opp_known_n"] <= 2)).astype(int)
    out["known_n_diff"] = pd.to_numeric(out["team_known_n"], errors="coerce").fillna(0) - pd.to_numeric(out["opp_known_n"], errors="coerce").fillna(0)
    out["known_n_ratio"] = (pd.to_numeric(out["team_known_n"], errors="coerce").fillna(0) + 1) / (pd.to_numeric(out["opp_known_n"], errors="coerce").fillna(0) + 1)
    out["known_log_diff"] = out["team_known_log1p"] - out["opp_known_log1p"]

    # Interaction categorical yang membantu model fallback ketika team-level history kosong.
    out["gender_tier_history_bucket"] = (
        out["gender"].astype(str) + "_" +
        out["tournament_tier"].astype(str) + "_" +
        out["team_history_bucket"].astype(str) + "_vs_" +
        out["opp_history_bucket"].astype(str)
    )
    out["gender_confed_tier"] = (
        out["gender"].astype(str) + "_" +
        out["confederation_pair_sorted"].astype(str) + "_" +
        out["tournament_tier"].astype(str)
    )

    train_out = (
        out[out["__dataset"].eq("train")]
        .sort_values("__orig_idx")
        .drop(columns=["__dataset", "__dataset_order", "__orig_idx"], errors="ignore")
        .reset_index(drop=True)
    )

    pred_out = (
        out[out["__dataset"].eq("pred")]
        .sort_values("__orig_idx")
        .drop(columns=["__dataset", "__dataset_order", "__orig_idx", "team_goals", "opp_goals"], errors="ignore")
        .reset_index(drop=True)
    )

    return train_out, pred_out

def add_online_elo_features(train_df, pred_df, k=22.0, home_adv=60.0):
    """Online Elo yang dipisah berdasarkan gender.

    Elo update hanya memakai train block. Untuk pred block, rating tidak di-update memakai target hidden.
    """
    train_work = train_df.copy()
    pred_work = pred_df.copy()

    train_work["__dataset"] = "train"
    pred_work["__dataset"] = "pred"

    if "team_goals" not in pred_work.columns:
        pred_work["team_goals"] = np.nan
    if "opp_goals" not in pred_work.columns:
        pred_work["opp_goals"] = np.nan

    base_cols = list(dict.fromkeys(list(train_work.columns) + list(pred_work.columns)))
    for col in base_cols:
        if col not in train_work.columns:
            train_work[col] = np.nan
        if col not in pred_work.columns:
            pred_work[col] = np.nan

    work = pd.concat([train_work[base_cols], pred_work[base_cols]], ignore_index=True)
    work["date"] = pd.to_datetime(work["date"])
    work["__orig_idx"] = np.arange(len(work))
    work["__dataset_order"] = work["__dataset"].map({"train": 0, "pred": 1}).astype(int)
    work = work.sort_values(["date", MATCH_COL, "__dataset_order", ID_COL]).reset_index(drop=True)

    ratings = defaultdict(lambda: 1500.0)
    feat_rows = []

    def key(gender, team):
        return (str(gender), str(team))

    for match_id, g in work.groupby(MATCH_COL, sort=False):
        for idx, row in g.iterrows():
            rt = ratings[key(row["gender"], row["team"])]
            ro = ratings[key(row["gender"], row["opponent"])]
            ha = home_adv if int(row.get("is_home", 0)) == 1 and int(row.get("neutral", 0)) == 0 else 0.0
            diff = rt + ha - ro
            feat_rows.append({
                "__sort_idx": idx,
                "online_elo_team": rt,
                "online_elo_opp": ro,
                "online_elo_diff": diff,
                "online_elo_expected": 1 / (1 + 10 ** (-diff / 400)),
            })

        # update only if match belongs to train block
        if g["__dataset"].iloc[0] != "train":
            continue
        if len(g) < 2:
            continue

        row0 = g.iloc[0]
        if pd.isna(row0["team_goals"]) or pd.isna(row0["opp_goals"]):
            continue

        gender = row0["gender"]
        team_a = row0["team"]
        team_b = row0["opponent"]
        gf = float(row0["team_goals"])
        ga = float(row0["opp_goals"])

        actual_a = 1.0 if gf > ga else 0.5 if gf == ga else 0.0
        ha = home_adv if int(row0.get("is_home", 0)) == 1 and int(row0.get("neutral", 0)) == 0 else 0.0

        ra = ratings[key(gender, team_a)]
        rb = ratings[key(gender, team_b)]
        diff = ra + ha - rb
        expected_a = 1 / (1 + 10 ** (-diff / 400))

        gd = abs(gf - ga)
        goal_mult = 1.0 if gd <= 1 else 1.5 if gd == 2 else 1.75 + (gd - 3) / 8
        change = k * goal_mult * (actual_a - expected_a)

        ratings[key(gender, team_a)] += change
        ratings[key(gender, team_b)] -= change

    feat_df = pd.DataFrame(feat_rows).sort_values("__sort_idx").drop(columns="__sort_idx").reset_index(drop=True)
    out = pd.concat([work.reset_index(drop=True), feat_df], axis=1)

    cols = ["online_elo_team", "online_elo_opp", "online_elo_diff", "online_elo_expected"]
    train_elo = out[out["__dataset"].eq("train")].sort_values("__orig_idx")[cols].reset_index(drop=True)
    pred_elo = out[out["__dataset"].eq("pred")].sort_values("__orig_idx")[cols].reset_index(drop=True)

    return train_elo, pred_elo

def make_feature_frames(train_source, pred_source, windows=(3, 5, 10, 20)):
    train_input = train_source[TRAIN_INPUT_COLS].copy()
    pred_input = pred_source[COMMON_INPUT_COLS].copy()

    train_base = add_base_features(train_input)
    pred_base = add_base_features(pred_input)

    train_fe, pred_fe = build_temporal_history_features(train_base, pred_base, windows=windows)
    train_elo, pred_elo = add_online_elo_features(train_base, pred_base)

    for col in train_elo.columns:
        train_fe[col] = train_elo[col].values
        pred_fe[col] = pred_elo[col].values

    return train_fe, pred_fe

print("Feature engineering helpers ready ✓")

# %% [markdown]
# ## 03b. Country-aware feature wrap
# 
# Wrap `make_feature_frames` agar memanggil missing imputer dan country-feature helper sebelum pipeline temporal+Elo original.

# %%
# ============================================================
# 03b. Country-aware feature engineering (extends section 03)
# ============================================================
#
# Helper di sini menambah:
#   - country_tier_team / country_tier_opp                   (1-5, dependent on gender)
#   - country_fanbase_team / country_fanbase_opp             (1-5)
#   - federation_funding_proxy_team / federation_funding_proxy_opp (log-scale)
#   - tier_diff, fanbase_diff, funding_diff                  (sinyal mismatch utama)
#   - women_x_country_tier_diff                              (interaksi dengan W segment)
#   - extreme_mismatch_score                                 (heuristik kasar 0-1)
#
# extreme_mismatch_score TIDAK dipakai untuk override prediction; itu hanya
# fitur tambahan yang kemudian dipakai oleh rule-based overlay di akhir.

EXTREME_MISMATCH_TIER_GAP_THRESHOLD = 2  # tier_diff >= 2 = mismatch besar


def add_country_features(df):
    """Tambah fitur country tier, fanbase, funding proxy, plus interaksi.

    Asumsi df sudah punya kolom: gender, team, opponent, confederation_team,
    confederation_opp, population_team, population_opp, gdp_per_capita_team,
    gdp_per_capita_opp.
    """
    df = df.copy()

    gender_arr = df["gender"].astype(str).values
    team_arr = df["team"].astype(str).values if "team" in df.columns else np.array([""] * len(df))
    opp_arr = df["opponent"].astype(str).values if "opponent" in df.columns else np.array([""] * len(df))
    conf_t_arr = df["confederation_team"].astype(str).values if "confederation_team" in df.columns else np.array([""] * len(df))
    conf_o_arr = df["confederation_opp"].astype(str).values if "confederation_opp" in df.columns else np.array([""] * len(df))

    tier_t = np.array([lookup_country_tier(team_arr[i], gender_arr[i]) for i in range(len(df))], dtype=float)
    tier_o = np.array([lookup_country_tier(opp_arr[i], gender_arr[i]) for i in range(len(df))], dtype=float)

    fan_t = np.array([lookup_country_fanbase(team_arr[i], conf_t_arr[i]) for i in range(len(df))], dtype=float)
    fan_o = np.array([lookup_country_fanbase(opp_arr[i], conf_o_arr[i]) for i in range(len(df))], dtype=float)

    pop_t = pd.to_numeric(df.get("population_team", pd.Series([np.nan] * len(df))), errors="coerce").values
    pop_o = pd.to_numeric(df.get("population_opp", pd.Series([np.nan] * len(df))), errors="coerce").values
    gdp_t = pd.to_numeric(df.get("gdp_per_capita_team", pd.Series([np.nan] * len(df))), errors="coerce").values
    gdp_o = pd.to_numeric(df.get("gdp_per_capita_opp", pd.Series([np.nan] * len(df))), errors="coerce").values

    fund_t = np.array([federation_funding_proxy(pop_t[i], gdp_t[i], tier_t[i]) for i in range(len(df))], dtype=float)
    fund_o = np.array([federation_funding_proxy(pop_o[i], gdp_o[i], tier_o[i]) for i in range(len(df))], dtype=float)

    df["country_tier_team"] = tier_t
    df["country_tier_opp"] = tier_o
    df["country_tier_diff"] = tier_o - tier_t  # positif = team lebih kuat
    df["country_tier_min"] = np.minimum(tier_t, tier_o)
    df["country_tier_sum"] = tier_t + tier_o

    df["country_fanbase_team"] = fan_t
    df["country_fanbase_opp"] = fan_o
    df["country_fanbase_diff"] = fan_o - fan_t

    df["federation_funding_team"] = fund_t
    df["federation_funding_opp"] = fund_o
    df["federation_funding_diff"] = fund_t - fund_o
    df["federation_funding_min"] = np.fmin(fund_t, fund_o)

    is_w = df.get("is_women_match", pd.Series([0] * len(df))).astype(float).values
    df["women_x_tier_diff"] = is_w * df["country_tier_diff"].values
    df["women_x_funding_diff"] = is_w * df["federation_funding_diff"].values

    # Heuristik mismatch: tier opponent jauh lebih lemah dari team + funding gap besar.
    tier_gap = (tier_o - tier_t).clip(min=0)
    funding_gap = np.where(np.isfinite(fund_t - fund_o), (fund_t - fund_o).clip(min=0), 0.0)
    funding_gap_norm = funding_gap / (np.nanstd(funding_gap[np.isfinite(funding_gap)]) + 1e-6)

    score = 0.5 * (tier_gap / 4.0) + 0.5 * np.clip(funding_gap_norm, 0, 3) / 3.0
    df["extreme_mismatch_score"] = np.clip(score, 0.0, 1.0)
    df["extreme_mismatch_high"] = (df["extreme_mismatch_score"] > 0.55).astype(int)
    df["tier_gap_at_least_2"] = (tier_gap >= EXTREME_MISMATCH_TIER_GAP_THRESHOLD).astype(int)

    return df


# ============================================================
# Wrap make_feature_frames untuk include country + missing handler
# ============================================================
#
# Fungsi original di cell 03 tetap ada. Kita tinggal redefine di sini agar
# alur pipeline lain (cell 04, 12, dll.) langsung pakai versi baru.

_make_feature_frames_orig = make_feature_frames


def make_feature_frames(train_source, pred_source, windows=(3, 5, 10, 20)):
    """Versi extended:
       1) Apply missing-data imputer (sudah di-fit dari train_raw lengkap).
       2) Add country features (tier, fanbase, funding).
       3) Lalu jalankan pipeline original (base + temporal history + Elo).
    """
    train_input = train_source[TRAIN_INPUT_COLS].copy()
    pred_input = pred_source[COMMON_INPUT_COLS].copy()

    # 1) imputasi missing data
    if USE_MISSING_IMPUTER:
        train_input = apply_missing_imputer(train_input, missing_imputer)
        pred_input = apply_missing_imputer(pred_input, missing_imputer)

    # 2) base features dari pipeline original
    train_base = add_base_features(train_input)
    pred_base = add_base_features(pred_input)

    # 3) country priors
    if USE_COUNTRY_FEATURES:
        train_base = add_country_features(train_base)
        pred_base = add_country_features(pred_base)

    # 4) temporal history (persis seperti original)
    train_fe, pred_fe = build_temporal_history_features(train_base, pred_base, windows=windows)
    train_elo, pred_elo = add_online_elo_features(train_base, pred_base)

    for col in train_elo.columns:
        train_fe[col] = train_elo[col].values
        pred_fe[col] = pred_elo[col].values

    return train_fe, pred_fe


print("Country features helper ready ✓")
print("make_feature_frames extended. Flags:")
print("  USE_MISSING_IMPUTER:", USE_MISSING_IMPUTER)
print("  USE_COUNTRY_FEATURES:", USE_COUNTRY_FEATURES)

# %% [markdown]
# ## 03c. EXP22C — Last-known Elo/rank carry-forward
# 
# Bagian ini menambahkan fitur strength dari kolom train-only seperti `elo_team`, `elo_opponent`, `rank_team`, dan `rank_opponent`.
# 
# Prinsip leakage-safe:
# - Untuk train/model block, setiap baris hanya melihat nilai strength terakhir dari match sebelumnya.
# - Untuk validation/test block, nilai strength hanya diambil dari history train block, bukan dari label atau pseudo-label block prediksi.
# - Jika tim belum punya history, fitur diisi median per gender/global plus indikator missing.

# %%

# ============================================================
# 03c. EXP22C: last-known Elo/rank carry-forward features
# ============================================================

def _safe_numeric_series(s, default=np.nan):
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)


FULL_STRENGTH_FEATURES = [
    "lk_team_last_elo", "lk_opp_last_elo", "lk_last_elo_diff", "lk_last_elo_absdiff",
    "lk_team_last_rank", "lk_opp_last_rank", "lk_last_rank_diff", "lk_last_rank_absdiff",
    "lk_rank_advantage", "lk_team_rank_missing_flag", "lk_opp_rank_missing_flag",
    "lk_team_strength_age_days", "lk_opp_strength_age_days", "lk_strength_age_diff",
    "lk_team_strength_age_log1p", "lk_opp_strength_age_log1p",
    "lk_team_has_strength_history", "lk_opp_has_strength_history",
    "lk_both_have_strength_history", "lk_one_or_more_cold_strength",
    "lk_max_strength_age_days", "lk_min_strength_age_days", "lk_max_strength_age_log1p",
]


def get_strength_feature_columns(mode: str) -> list:
    """Return selected last-known strength columns for EXP12C ablation."""
    mode = str(mode or "full").lower()
    if mode == "none":
        return []
    if mode == "elo_only":
        return [
            "lk_team_last_elo", "lk_opp_last_elo", "lk_last_elo_diff", "lk_last_elo_absdiff",
            "lk_team_has_strength_history", "lk_opp_has_strength_history",
            "lk_both_have_strength_history", "lk_one_or_more_cold_strength",
        ]
    if mode == "rank_only":
        return [
            "lk_team_last_rank", "lk_opp_last_rank", "lk_last_rank_diff", "lk_last_rank_absdiff",
            "lk_rank_advantage", "lk_team_rank_missing_flag", "lk_opp_rank_missing_flag",
            "lk_team_last_rank_log1p", "lk_opp_last_rank_log1p",
            "lk_team_has_strength_history", "lk_opp_has_strength_history",
            "lk_both_have_strength_history", "lk_one_or_more_cold_strength",
        ]
    if mode == "freshness_only":
        return [
            "lk_team_strength_age_days", "lk_opp_strength_age_days", "lk_strength_age_diff",
            "lk_team_strength_age_log1p", "lk_opp_strength_age_log1p",
            "lk_team_has_strength_history", "lk_opp_has_strength_history",
            "lk_both_have_strength_history", "lk_one_or_more_cold_strength",
            "lk_max_strength_age_days", "lk_min_strength_age_days", "lk_max_strength_age_log1p",
        ]
    return FULL_STRENGTH_FEATURES


def _filter_strength_columns_by_mode(train_out: pd.DataFrame, pred_out: pd.DataFrame, mode: str):
    """Drop last-known strength columns outside the current ablation mode."""
    keep = set(get_strength_feature_columns(mode))
    train_strength_cols = [c for c in train_out.columns if c.startswith("lk_")]
    drop_cols = [c for c in train_strength_cols if c not in keep]
    train_out = train_out.drop(columns=drop_cols, errors="ignore")
    pred_out = pred_out.drop(columns=drop_cols, errors="ignore")
    kept_cols = [c for c in train_out.columns if c.startswith("lk_")]
    print(f"[EXP12C] STRENGTH_FEATURE_MODE={mode} | kept={len(kept_cols)} | dropped={len(drop_cols)}")
    return train_out, pred_out


def _build_strength_history_records(train_source):
    """Bangun history strength per (gender, team) dari kolom train-only.

    Hanya memakai sisi `team` karena dataset mirror biasanya punya satu baris
    untuk tiap tim di setiap match. Jika kolom tidak tersedia, return None.
    """
    required = {"date", "gender", "team"}
    if not required.issubset(train_source.columns):
        return None

    has_elo = "elo_team" in train_source.columns
    has_rank = "rank_team" in train_source.columns
    if not has_elo and not has_rank:
        return None

    hist = train_source[["date", "gender", "team"]].copy()
    hist["date"] = pd.to_datetime(hist["date"])
    hist["gender"] = hist["gender"].astype(str)
    hist["entity"] = hist["team"].astype(str)

    if has_elo:
        hist["hist_elo"] = pd.to_numeric(train_source["elo_team"], errors="coerce")
    else:
        hist["hist_elo"] = np.nan

    if has_rank:
        hist["hist_rank"] = pd.to_numeric(train_source["rank_team"], errors="coerce")
    else:
        hist["hist_rank"] = np.nan

    if "rank_missing_team" in train_source.columns:
        hist["hist_rank_missing"] = pd.to_numeric(train_source["rank_missing_team"], errors="coerce")
    else:
        hist["hist_rank_missing"] = hist["hist_rank"].isna().astype(float)

    hist = hist.sort_values(["gender", "entity", "date"]).reset_index(drop=True)
    return hist


def _gender_global_defaults(history):
    defaults = {}
    if history is None or len(history) == 0:
        defaults["global_elo"] = 1500.0
        defaults["global_rank"] = 150.0
        defaults["elo_by_gender"] = {}
        defaults["rank_by_gender"] = {}
        return defaults

    defaults["global_elo"] = float(history["hist_elo"].median()) if history["hist_elo"].notna().any() else 1500.0
    defaults["global_rank"] = float(history["hist_rank"].median()) if history["hist_rank"].notna().any() else 150.0

    elo_by_gender = history.groupby("gender")["hist_elo"].median().dropna().to_dict()
    rank_by_gender = history.groupby("gender")["hist_rank"].median().dropna().to_dict()
    defaults["elo_by_gender"] = {str(k): float(v) for k, v in elo_by_gender.items()}
    defaults["rank_by_gender"] = {str(k): float(v) for k, v in rank_by_gender.items()}
    return defaults


def _lookup_last_strength_for_side(history, target_df, side_col, prefix, defaults):
    """Lookup strict previous strength untuk `side_col` di target_df."""
    n = len(target_df)
    out = pd.DataFrame(index=np.arange(n))

    out[f"{prefix}_last_elo"] = np.nan
    out[f"{prefix}_last_rank"] = np.nan
    out[f"{prefix}_rank_missing_flag"] = 1.0
    out[f"{prefix}_strength_age_days"] = 9999.0
    out[f"{prefix}_has_strength_history"] = 0.0

    if history is None or side_col not in target_df.columns:
        return out

    target_work = target_df[["date", "gender", side_col]].copy()
    target_work["date"] = pd.to_datetime(target_work["date"])
    target_work["gender"] = target_work["gender"].astype(str)
    target_work["entity"] = target_work[side_col].astype(str)
    target_work["__row_id"] = np.arange(n)

    # Group lookup manual dengan searchsorted supaya allow_exact_matches=False.
    hist_groups = {}
    for key, g in history.groupby(["gender", "entity"], sort=False):
        g = g.sort_values("date")
        hist_groups[key] = {
            "dates": g["date"].values.astype("datetime64[ns]"),
            "elo": g["hist_elo"].to_numpy(dtype=float),
            "rank": g["hist_rank"].to_numpy(dtype=float),
            "rank_missing": g["hist_rank_missing"].to_numpy(dtype=float),
        }

    for key, idxs in target_work.groupby(["gender", "entity"]).groups.items():
        pack = hist_groups.get(key)
        if pack is None:
            continue

        idx_arr = np.asarray(list(idxs), dtype=int)
        tdates = target_work.loc[idx_arr, "date"].values.astype("datetime64[ns]")

        # strict previous: posisi terakhir dengan hist_date < target_date
        pos = np.searchsorted(pack["dates"], tdates, side="left") - 1
        ok = pos >= 0
        if not np.any(ok):
            continue

        row_ids = target_work.loc[idx_arr[ok], "__row_id"].to_numpy(dtype=int)
        hist_pos = pos[ok]

        out.loc[row_ids, f"{prefix}_last_elo"] = pack["elo"][hist_pos]
        out.loc[row_ids, f"{prefix}_last_rank"] = pack["rank"][hist_pos]
        out.loc[row_ids, f"{prefix}_rank_missing_flag"] = pack["rank_missing"][hist_pos]

        age_days = (
            tdates[ok].astype("datetime64[D]") -
            pack["dates"][hist_pos].astype("datetime64[D]")
        ).astype("timedelta64[D]").astype(float)

        out.loc[row_ids, f"{prefix}_strength_age_days"] = np.clip(age_days, 0, 9999)
        out.loc[row_ids, f"{prefix}_has_strength_history"] = 1.0

    # Fill missing with gender median, then global fallback.
    gender_values = target_work["gender"].astype(str).values
    for i in range(n):
        g = gender_values[i]
        if pd.isna(out.loc[i, f"{prefix}_last_elo"]):
            out.loc[i, f"{prefix}_last_elo"] = defaults["elo_by_gender"].get(g, defaults["global_elo"])
        if pd.isna(out.loc[i, f"{prefix}_last_rank"]):
            out.loc[i, f"{prefix}_last_rank"] = defaults["rank_by_gender"].get(g, defaults["global_rank"])

    out[f"{prefix}_rank_missing_flag"] = out[f"{prefix}_rank_missing_flag"].fillna(1.0).astype(float)
    out[f"{prefix}_strength_age_days"] = out[f"{prefix}_strength_age_days"].fillna(9999.0).clip(lower=0, upper=9999)
    out[f"{prefix}_has_strength_history"] = out[f"{prefix}_has_strength_history"].fillna(0.0).astype(float)
    out[f"{prefix}_strength_age_log1p"] = np.log1p(out[f"{prefix}_strength_age_days"].astype(float))
    out[f"{prefix}_last_rank_log1p"] = np.log1p(out[f"{prefix}_last_rank"].clip(lower=1).astype(float))
    return out.reset_index(drop=True)


def add_last_known_strength_features(train_source, pred_source, train_fe, pred_fe):
    """Tambahkan fitur last-known Elo/rank ke train_fe dan pred_fe."""
    history = _build_strength_history_records(train_source)

    if history is None:
        print("[EXP22C] Kolom elo/rank train-only tidak tersedia. Skip last-known strength features.")
        return train_fe, pred_fe

    defaults = _gender_global_defaults(history)

    train_team = _lookup_last_strength_for_side(history, train_source, "team", "team", defaults)
    train_opp = _lookup_last_strength_for_side(history, train_source, "opponent", "opp", defaults)

    pred_team = _lookup_last_strength_for_side(history, pred_source, "team", "team", defaults)
    pred_opp = _lookup_last_strength_for_side(history, pred_source, "opponent", "opp", defaults)

    def attach(base, a, b):
        out = base.copy()
        for col in a.columns:
            out[f"lk_{col}"] = a[col].values
        for col in b.columns:
            out[f"lk_{col}"] = b[col].values

        out["lk_last_elo_diff"] = out["lk_team_last_elo"] - out["lk_opp_last_elo"]
        out["lk_last_elo_absdiff"] = np.abs(out["lk_last_elo_diff"])

        # rank lebih kecil berarti lebih kuat.
        out["lk_last_rank_diff"] = out["lk_team_last_rank"] - out["lk_opp_last_rank"]
        out["lk_rank_advantage"] = out["lk_opp_last_rank"] - out["lk_team_last_rank"]
        out["lk_last_rank_absdiff"] = np.abs(out["lk_last_rank_diff"])

        out["lk_both_have_strength_history"] = (
            (out["lk_team_has_strength_history"] > 0) &
            (out["lk_opp_has_strength_history"] > 0)
        ).astype(int)
        out["lk_one_or_more_cold_strength"] = 1 - out["lk_both_have_strength_history"]
        out["lk_max_strength_age_days"] = np.maximum(
            out["lk_team_strength_age_days"].astype(float),
            out["lk_opp_strength_age_days"].astype(float),
        )
        out["lk_min_strength_age_days"] = np.minimum(
            out["lk_team_strength_age_days"].astype(float),
            out["lk_opp_strength_age_days"].astype(float),
        )
        out["lk_max_strength_age_log1p"] = np.log1p(out["lk_max_strength_age_days"])
        out["lk_strength_age_diff"] = out["lk_team_strength_age_days"].astype(float) - out["lk_opp_strength_age_days"].astype(float)
        return out

    train_out = attach(train_fe, train_team, train_opp)
    pred_out = attach(pred_fe, pred_team, pred_opp)

    train_out, pred_out = _filter_strength_columns_by_mode(
        train_out,
        pred_out,
        globals().get("STRENGTH_FEATURE_MODE", "full"),
    )
    added_cols = [c for c in train_out.columns if c.startswith("lk_") and c not in train_fe.columns]
    print(f"[EXP22C] Added last-known strength features after mode filter: {len(added_cols)}")
    print(added_cols[:40])
    return train_out, pred_out


# Wrap make_feature_frames setelah country-aware wrapper.
_base_make_feature_frames_before_exp22 = make_feature_frames

def make_feature_frames(train_source, pred_source, windows=(3, 5, 10, 20)):
    train_fe, pred_fe = _base_make_feature_frames_before_exp22(train_source, pred_source, windows=windows)

    if USE_LAST_KNOWN_STRENGTH_FEATURES:
        train_fe, pred_fe = add_last_known_strength_features(
            train_source=train_source,
            pred_source=pred_source,
            train_fe=train_fe,
            pred_fe=pred_fe,
        )
    else:
        print("[EXP22C] USE_LAST_KNOWN_STRENGTH_FEATURES=False; skip carry-forward rank/Elo.")

    return train_fe, pred_fe

print("EXP22C last-known strength wrapper ready ✓")

# %% [markdown]
# ## 04. Build validation block
# 
# Validasi dibuat sebagai future block. Ini lebih aman daripada random split karena test memang berada setelah train secara waktu.

# %%
# ============================================================
# 04. Build temporal validation data
# ============================================================

valid_start = pd.Timestamp(VALID_START_DATE)
model_start = pd.Timestamp(MODEL_TRAIN_START_DATE)

train_hist_raw = train_raw[train_raw["date"] < valid_start].copy()
valid_raw = train_raw[train_raw["date"] >= valid_start].copy()

assert len(train_hist_raw) > 0 and len(valid_raw) > 0, "Split validasi kosong. Cek VALID_START_DATE."

print("Train history rows:", train_hist_raw.shape)
print("Valid rows        :", valid_raw.shape)
print("Train history date:", train_hist_raw["date"].min(), "to", train_hist_raw["date"].max())
print("Valid date        :", valid_raw["date"].min(), "to", valid_raw["date"].max())

train_fe_valid, valid_fe = make_feature_frames(train_hist_raw, valid_raw)

# model hanya dilatih pada era modern, tapi fitur history tetap memanfaatkan era lama sebelum 1990.
model_train_mask = pd.to_datetime(train_fe_valid["date"]) >= model_start
train_model_fe = train_fe_valid[model_train_mask].reset_index(drop=True).copy()

print("train_fe_valid shape:", train_fe_valid.shape)
print("train_model_fe shape:", train_model_fe.shape)
print("valid_fe shape      :", valid_fe.shape)
display(train_model_fe.head(3))
display(valid_fe.head(3))

# %% [markdown]
# ## 05. Feature column selection
# 
# Kolom yang dipakai model dipilih dari hasil feature engineering yang benar-benar ada di blok prediksi. Kolom ID, tanggal, dan target tidak dipakai sebagai fitur numerik mentah.

# %%
# ============================================================
# 05. Feature column selection
# ============================================================

AUX_TARGETS = ["goal_diff_target", "total_goals_target", "outcome_value_target"]

DROP_FEATURES = {
    ID_COL, MATCH_COL, "date",
    "team_goals", "opp_goals",
    "goal_diff_target", "total_goals_target", "outcome_value_target",
}

def add_auxiliary_training_targets(df):
    """Target tambahan untuk decoder-aware modeling.

    Kolom ini tidak boleh masuk sebagai fitur. Isinya hanya dibuat pada train/model block.
    """
    df = df.copy()
    df["goal_diff_target"] = df["team_goals"].astype(float) - df["opp_goals"].astype(float)
    df["total_goals_target"] = df["team_goals"].astype(float) + df["opp_goals"].astype(float)
    df["outcome_value_target"] = np.sign(df["goal_diff_target"]).astype(float)
    return df

def select_feature_columns(train_fe, pred_fe):
    candidate_cols = [
        c for c in train_fe.columns
        if c in pred_fe.columns and c not in DROP_FEATURES
    ]

    selected = []
    removed_all_missing_pred = []
    removed_constant_train = []

    for c in candidate_cols:
        # Hapus kolom yang kosong total di pred block.
        if pred_fe[c].isna().all():
            removed_all_missing_pred.append(c)
            continue

        # Hapus kolom konstan di train.
        nunique = train_fe[c].nunique(dropna=True)
        if nunique <= 1:
            removed_constant_train.append(c)
            continue

        selected.append(c)

    return selected, removed_all_missing_pred, removed_constant_train

base_feature_cols, removed_missing, removed_constant = select_feature_columns(train_model_fe, valid_fe)

num_cols = [c for c in base_feature_cols if pd.api.types.is_numeric_dtype(train_model_fe[c])]
cat_cols = [c for c in base_feature_cols if c not in num_cols]

print("Jumlah fitur:", len(base_feature_cols))
print("Numeric fitur:", len(num_cols))
print("Categorical fitur:", len(cat_cols))
print("Contoh numeric:", num_cols[:25])
print("Contoh categorical:", cat_cols[:25])
print("Kolom dihapus karena pred block missing total:", removed_missing[:30])
print("Kolom dihapus karena konstan di train:", removed_constant[:30])

# %% [markdown]
# ## 05b. Extreme-prone country clustering
# 
# KMeans per gender pada statistik per-negara (mean GD, std GD, tail probabilities). Hasil cluster labelnya jadi categorical feature untuk model dan jadi gating untuk extreme overlay nanti.

# %%
# ============================================================
# 05b. Identify extreme-prone countries via clustering
# ============================================================
#
# Tujuannya menandai negara-negara yang historis sering muncul di pertandingan
# dengan margin sangat besar (baik sebagai pemenang ekstrem maupun sebagai
# yang dihancurkan).
#
# Sinyal ini dipakai untuk:
#   1) Fitur kategorikal extra (cluster_team, cluster_opp).
#   2) Gating untuk extreme rule-based overlay di section 13c.
#
# Catatan domain knowledge sports prediction extremes:
#   - Match dengan goal_diff >= 8 hampir tidak pernah terjadi di pertemuan
#     tim tier 1 vs tier 1. Mayoritas terjadi di qualifier kontinental
#     (CAF, AFC, OFC) ataupun saat federasi muda baru bertanding.
#   - Pada women football era 2000-an, gap antar tim ekstrem karena banyak
#     federasi baru memulai program W. Ini "domain shift" yang berbeda dari
#     era 2010-an dan jauh berbeda dari era 2020-an.
#   - Karena itu kita compute statistik per gender dan per (country, gender)
#     supaya tidak salah label satu negara karena agregasi M+W.

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

EXTREME_GD_THRESHOLDS = (5, 8, 10, 15)
EXTREME_CLUSTER_K = 4
EXTREME_CLUSTER_MIN_MATCHES = 5  # negara dengan riwayat <5 match diberi default cluster

EXTREME_CLUSTER_LABELS = {
    0: "developing_underdog",  # cluster default; akan di-relabel dari centroid
    1: "developing",
    2: "competitive",
    3: "powerhouse",
}


def compute_country_extreme_stats(matches_df, gender):
    """Hitung statistik per (country, gender) sebagai team:
       gf_mean, ga_mean, gd_mean, gd_std, max_gd_for, max_gd_against,
       p_gd_ge_5, p_gd_ge_8, p_gd_ge_10, n_matches.

    matches_df harus punya: gender, team, opponent, team_goals, opp_goals.
    """
    seg = matches_df[matches_df["gender"].astype(str).str.upper() == str(gender).upper()].copy()
    seg["team_goals"] = pd.to_numeric(seg["team_goals"], errors="coerce")
    seg["opp_goals"] = pd.to_numeric(seg["opp_goals"], errors="coerce")
    seg = seg.dropna(subset=["team_goals", "opp_goals"])
    seg["gd"] = seg["team_goals"] - seg["opp_goals"]

    rows = []
    for country, grp in seg.groupby("team", dropna=True):
        if len(grp) < 1:
            continue
        gd = grp["gd"].values
        rec = {
            "country": str(country),
            "gender": str(gender).upper(),
            "n_matches": int(len(grp)),
            "gf_mean": float(grp["team_goals"].mean()),
            "ga_mean": float(grp["opp_goals"].mean()),
            "gd_mean": float(gd.mean()),
            "gd_std": float(gd.std(ddof=0)) if len(gd) > 1 else 0.0,
            "max_gd_for": float(gd.max()) if len(gd) else 0.0,
            "max_gd_against": float((-gd).max()) if len(gd) else 0.0,
        }
        for thr in EXTREME_GD_THRESHOLDS:
            rec[f"p_gd_for_ge_{thr}"] = float((gd >= thr).mean())
            rec[f"p_gd_against_ge_{thr}"] = float((gd <= -thr).mean())
        rows.append(rec)
    return pd.DataFrame(rows)


def fit_extreme_country_clusters(stats_df, k=EXTREME_CLUSTER_K, seed=SEED):
    """Cluster negara per gender. Return dict[(gender, country)] -> cluster_label."""
    if stats_df.empty:
        return {}, {}, None

    feature_cols = [
        "gf_mean", "ga_mean", "gd_mean", "gd_std",
        "max_gd_for", "max_gd_against",
        "p_gd_for_ge_5", "p_gd_for_ge_8",
        "p_gd_against_ge_5", "p_gd_against_ge_8",
    ]
    work = stats_df.copy()

    # Filter negara dengan jumlah match cukup, sisa dapat cluster default
    enough = work["n_matches"] >= EXTREME_CLUSTER_MIN_MATCHES
    if enough.sum() < k * 2:
        # Tidak cukup negara untuk clustering bermakna -> return default
        labels = {(row["gender"], row["country"]): "default_unknown" for _, row in work.iterrows()}
        return labels, {}, None

    X = work.loc[enough, feature_cols].fillna(0.0).values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    cluster_ids = km.fit_predict(Xs)

    # Relabel cluster berdasarkan rata-rata gd_mean (positif = powerhouse)
    centroid_strength = []
    for cid in range(k):
        mask = cluster_ids == cid
        avg_gd = float(work.loc[enough].iloc[mask]["gd_mean"].mean()) if mask.any() else 0.0
        centroid_strength.append((cid, avg_gd))
    centroid_strength.sort(key=lambda x: x[1])  # ascending: lemah -> kuat
    sorted_labels = ["developing_underdog", "developing", "competitive", "powerhouse"]
    if k != len(sorted_labels):
        sorted_labels = [f"cluster_{i}" for i in range(k)]
    cid_to_label = {cid: sorted_labels[i] for i, (cid, _) in enumerate(centroid_strength)}

    labels = {}
    work_idx = work.loc[enough].index.tolist()
    for i, idx in enumerate(work_idx):
        row = work.loc[idx]
        labels[(row["gender"], row["country"])] = cid_to_label[int(cluster_ids[i])]
    for _, row in work.loc[~enough].iterrows():
        labels[(row["gender"], row["country"])] = "default_unknown"

    artifact = {
        "scaler": scaler,
        "kmeans": km,
        "feature_cols": feature_cols,
        "cid_to_label": cid_to_label,
    }
    return labels, artifact, work


def attach_country_cluster_features(df, country_cluster_labels):
    """Tambah cluster_team / cluster_opp ke df."""
    df = df.copy()
    gender_arr = df["gender"].astype(str).str.upper().values
    team_arr = df["team"].astype(str).values
    opp_arr = df["opponent"].astype(str).values

    cluster_team = np.array([
        country_cluster_labels.get((gender_arr[i], team_arr[i]), "default_unknown")
        for i in range(len(df))
    ])
    cluster_opp = np.array([
        country_cluster_labels.get((gender_arr[i], opp_arr[i]), "default_unknown")
        for i in range(len(df))
    ])

    df["country_cluster_team"] = cluster_team
    df["country_cluster_opp"] = cluster_opp
    df["country_cluster_pair"] = np.array([
        f"{cluster_team[i]}__vs__{cluster_opp[i]}" for i in range(len(df))
    ])

    rank = {
        "powerhouse": 4, "competitive": 3, "developing": 2,
        "developing_underdog": 1, "default_unknown": 2,
    }
    rank_t = np.array([rank.get(c, 2) for c in cluster_team], dtype=float)
    rank_o = np.array([rank.get(c, 2) for c in cluster_opp], dtype=float)
    df["country_cluster_rank_team"] = rank_t
    df["country_cluster_rank_opp"] = rank_o
    df["country_cluster_rank_diff"] = rank_t - rank_o
    df["is_extreme_mismatch_pair"] = (
        ((cluster_team == "powerhouse") & (cluster_opp == "developing_underdog"))
        | ((cluster_team == "developing_underdog") & (cluster_opp == "powerhouse"))
    ).astype(int)

    return df


if USE_EXTREME_CLUSTERS:
    # Build cluster dari train_hist_raw (data train sebelum validasi).
    # Tetap pisah per gender karena distribusi sangat beda.
    country_extreme_stats_M = compute_country_extreme_stats(train_hist_raw, "M")
    country_extreme_stats_W = compute_country_extreme_stats(train_hist_raw, "W")

    cluster_labels_M, cluster_artifact_M, _ = fit_extreme_country_clusters(country_extreme_stats_M)
    cluster_labels_W, cluster_artifact_W, _ = fit_extreme_country_clusters(country_extreme_stats_W)

    country_cluster_labels = {**cluster_labels_M, **cluster_labels_W}

    print("Cluster fit ✓")
    print("M countries clustered:", sum(1 for k in country_cluster_labels if k[0] == "M"))
    print("W countries clustered:", sum(1 for k in country_cluster_labels if k[0] == "W"))

    # Apply cluster features ke train_model_fe dan valid_fe (validation pipeline).
    train_model_fe = attach_country_cluster_features(train_model_fe, country_cluster_labels)
    valid_fe = attach_country_cluster_features(valid_fe, country_cluster_labels)

    # Re-run feature column selection agar fitur cluster terpilih
    base_feature_cols, removed_missing, removed_constant = select_feature_columns(train_model_fe, valid_fe)
    num_cols = [c for c in base_feature_cols if pd.api.types.is_numeric_dtype(train_model_fe[c])]
    cat_cols = [c for c in base_feature_cols if c not in num_cols]
    feature_cols = base_feature_cols  # akan di-overwrite cell 06 kalau supervised signal aktif

    print("Updated feature count:", len(base_feature_cols))
    print("Cluster-related cat features:", [c for c in cat_cols if "cluster" in c])

    # Print top extreme-prone countries untuk audit
    print("\n--- Top powerhouse countries (M) ---")
    ph_M = sorted([k for k, v in cluster_labels_M.items() if v == "powerhouse"])
    print(", ".join(c for _, c in ph_M[:25]))
    print("\n--- Top extreme-underdog countries (M) ---")
    und_M = sorted([k for k, v in cluster_labels_M.items() if v == "developing_underdog"])
    print(", ".join(c for _, c in und_M[:25]))
    print("\n--- Top powerhouse countries (W) ---")
    ph_W = sorted([k for k, v in cluster_labels_W.items() if v == "powerhouse"])
    print(", ".join(c for _, c in ph_W[:25]))
    print("\n--- Top extreme-underdog countries (W) ---")
    und_W = sorted([k for k, v in cluster_labels_W.items() if v == "developing_underdog"])
    print(", ".join(c for _, c in und_W[:25]))

else:
    print("Extreme country clustering dimatikan untuk ablation variant:", EXP18_ABLATION_VARIANT)
    country_extreme_stats_M = pd.DataFrame()
    country_extreme_stats_W = pd.DataFrame()
    cluster_labels_M, cluster_labels_W = {}, {}
    cluster_artifact_M, cluster_artifact_W = {}, {}
    country_cluster_labels = {}

    # Pakai feature selection sebelumnya tanpa cluster feature.
    base_feature_cols, removed_missing, removed_constant = select_feature_columns(train_model_fe, valid_fe)
    num_cols = [c for c in base_feature_cols if pd.api.types.is_numeric_dtype(train_model_fe[c])]
    cat_cols = [c for c in base_feature_cols if c not in num_cols]
    feature_cols = base_feature_cols

    print("Updated feature count without cluster:", len(base_feature_cols))

# %% [markdown]
# ## 06. Optional supervised signal features
# 
# Bagian ini membuat fitur sinyal prediksi awal secara OOF di train block dan full-model prediction di valid/test block.
# 
# Kenapa aman:
# - Untuk baris train, sinyal dibuat dengan time-based OOF.
# - Untuk valid/test, sinyal dibuat dari model yang hanya melihat train block.
# - Tidak ada target valid/test yang dipakai.

# %%
# ============================================================
# 06. Optional supervised signal features
# ============================================================

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def add_supervised_signal_features(train_fe, pred_fe, feature_cols, target_cols=TARGETS):
    train_fe = train_fe.copy()
    pred_fe = pred_fe.copy()

    signal_feature_cols = [c for c in feature_cols if c in train_fe.columns and c in pred_fe.columns]
    num_signal_cols = [c for c in signal_feature_cols if pd.api.types.is_numeric_dtype(train_fe[c])]
    cat_signal_cols = [c for c in signal_feature_cols if c not in num_signal_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler(with_mean=False)), num_signal_cols),
            ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore", min_frequency=25)), cat_signal_cols),
        ],
        remainder="drop",
        sparse_threshold=0.35,
    )

    base_signal_model = make_pipeline(
        preprocessor,
        Ridge(alpha=10.0, solver="lsqr", random_state=SEED)
    )

    years = pd.to_datetime(train_fe["date"]).dt.year.values
    unique_years = np.array(sorted(pd.Series(years).dropna().unique()))

    # Expanding folds berdasarkan tahun.
    # Fold awal sengaja dimulai ketika data historis sudah cukup.
    fold_boundaries = []
    for start_year in [1998, 2002, 2006]:
        if (years >= start_year).sum() > 0 and (years < start_year).sum() >= 1000:
            fold_boundaries.append(start_year)

    for target in target_cols:
        oof = np.full(len(train_fe), np.nan, dtype=float)

        for start_year in fold_boundaries:
            tr_idx = np.where(years < start_year)[0]
            va_idx = np.where(years >= start_year)[0]

            # Agar fold tidak saling menimpa terlalu parah, valid fold dibatasi sampai boundary berikutnya.
            next_boundaries = [b for b in fold_boundaries if b > start_year]
            if next_boundaries:
                end_year = min(next_boundaries)
                va_idx = np.where((years >= start_year) & (years < end_year))[0]

            if len(tr_idx) < 1000 or len(va_idx) == 0:
                continue

            model = clone(base_signal_model)
            model.fit(train_fe.iloc[tr_idx][signal_feature_cols], train_fe.iloc[tr_idx][target])
            oof[va_idx] = model.predict(train_fe.iloc[va_idx][signal_feature_cols])

        # Fill bagian awal yang belum punya OOF dengan prior, bukan full-model, supaya tidak leakage.
        prior = float(train_fe[target].median())
        oof = np.where(np.isnan(oof), prior, oof)

        full_model = clone(base_signal_model)
        full_model.fit(train_fe[signal_feature_cols], train_fe[target])
        pred_signal = full_model.predict(pred_fe[signal_feature_cols])

        train_fe[f"oof_signal_{target}"] = np.clip(oof, 0, 12)
        pred_fe[f"oof_signal_{target}"] = np.clip(pred_signal, 0, 12)

    train_fe["oof_signal_goal_diff"] = train_fe["oof_signal_team_goals"] - train_fe["oof_signal_opp_goals"]
    pred_fe["oof_signal_goal_diff"] = pred_fe["oof_signal_team_goals"] - pred_fe["oof_signal_opp_goals"]

    train_fe["oof_signal_total_goals"] = train_fe["oof_signal_team_goals"] + train_fe["oof_signal_opp_goals"]
    pred_fe["oof_signal_total_goals"] = pred_fe["oof_signal_team_goals"] + pred_fe["oof_signal_opp_goals"]

    signal_cols = [
        "oof_signal_team_goals", "oof_signal_opp_goals",
        "oof_signal_goal_diff", "oof_signal_total_goals",
    ]

    return train_fe, pred_fe, signal_cols

if USE_SUPERVISED_SIGNAL:
    train_model_fe, valid_fe, signal_cols = add_supervised_signal_features(train_model_fe, valid_fe, base_feature_cols)
    feature_cols, removed_missing, removed_constant = select_feature_columns(train_model_fe, valid_fe)
    print("Supervised signal ON")
    print("Signal cols:", signal_cols)
else:
    feature_cols = base_feature_cols
    print("Supervised signal OFF")

num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train_model_fe[c])]
cat_cols = [c for c in feature_cols if c not in num_cols]

print("Final jumlah fitur:", len(feature_cols))
print("Numeric:", len(num_cols), "| Categorical:", len(cat_cols))

# %% [markdown]
# ## 07. Model training helpers
# 
# Default model:
# - CatBoost untuk data campuran numeric + categorical.
# - LightGBM sebagai ensemble tambahan.
# - Training dipisah per `gender` karena distribusi skor M dan W cukup berbeda.
# 
# Kalau ingin lebih cepat, set `USE_LIGHTGBM = False` atau `FAST_MODE = True` di awal notebook.

# %%
# ============================================================
# 07. Model helpers
# ============================================================

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except Exception as e:
    CATBOOST_AVAILABLE = False
    print("CatBoost tidak tersedia:", repr(e))

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except Exception as e:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM tidak tersedia:", repr(e))

from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import HistGradientBoostingRegressor

def compute_sample_weight(df):
    """Tournament weight + optional recency weight.

    Patch 6:
    - tournament weight tetap dipakai karena sesuai metric,
    - recency weight membuat model lebih condong ke distribusi era modern.
    """
    w = df["tournament"].map(get_tournament_weight).astype(float).values

    if USE_RECENCY_SAMPLE_WEIGHT and "date" in df.columns:
        d = pd.to_datetime(df["date"], errors="coerce")
        max_date = d.max()
        age_years = ((max_date - d).dt.days / 365.25).fillna(0).clip(lower=0)
        rec = np.power(0.5, age_years / float(RECENCY_HALFLIFE_YEARS))
        rec = rec / np.nanmedian(rec)
        rec = np.clip(rec, RECENCY_WEIGHT_MIN, RECENCY_WEIGHT_MAX)
        w = w * rec.values

    return np.asarray(w, dtype=float)

def get_xyw(df, feature_cols, target):
    X = df[feature_cols].copy()
    y = df[target].astype(float).values
    w = compute_sample_weight(df)
    return X, y, w

def split_feature_types(df, feature_cols):
    num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in feature_cols if c not in num_cols]
    return num_cols, cat_cols

def prepare_catboost_X(X, cat_cols):
    X = X.copy()
    for c in cat_cols:
        X[c] = X[c].astype("string").fillna("__MISSING__")
    return X

def make_catboost_model(target):
    # Target utama dibuat lebih kuat; auxiliary dibuat sedikit lebih ringan agar runtime tidak meledak.
    is_aux = target in AUX_TARGETS
    iterations = (450 if FAST_MODE else 1200) if is_aux else (650 if FAST_MODE else 1800)
    learning_rate = (0.060 if FAST_MODE else 0.040) if is_aux else (0.055 if FAST_MODE else 0.035)

    return CatBoostRegressor(
        loss_function="MAE",
        eval_metric="MAE",
        iterations=iterations,
        learning_rate=learning_rate,
        depth=6,
        l2_leaf_reg=8.0,
        random_strength=0.8,
        bootstrap_type="Bernoulli",
        subsample=0.85,
        min_data_in_leaf=25,
        random_seed=SEED,
        allow_writing_files=False,
        thread_count=-1,
        verbose=250,
        task_type="CPU",
    )

def make_lgbm_model(target):
    is_aux = target in AUX_TARGETS
    n_estimators = (350 if FAST_MODE else 900) if is_aux else (500 if FAST_MODE else 1400)
    learning_rate = (0.060 if FAST_MODE else 0.035) if is_aux else (0.050 if FAST_MODE else 0.025)

    return LGBMRegressor(
        objective="mae",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=48,
        max_depth=-1,
        min_child_samples=35,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=1.0,
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
    )

def make_fallback_model(target):
    # Fallback sklearn kalau CatBoost/LightGBM tidak ada.
    return HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.045,
        max_iter=350 if FAST_MODE else 700,
        max_leaf_nodes=31,
        l2_regularization=0.05,
        random_state=SEED,
    )

def build_lgbm_pipeline(num_cols, cat_cols, target):
    preprocess = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            ("cat", make_pipeline(
                SimpleImputer(strategy="most_frequent"),
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            ), cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    return make_pipeline(preprocess, make_lgbm_model(target))

def build_fallback_pipeline(num_cols, cat_cols, target):
    preprocess = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            ("cat", make_pipeline(
                SimpleImputer(strategy="most_frequent"),
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            ), cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    return make_pipeline(preprocess, make_fallback_model(target))

def fit_backend_model(backend, train_df, feature_cols, target):
    X, y, w = get_xyw(train_df, feature_cols, target)
    num_cols, cat_cols = split_feature_types(train_df, feature_cols)

    if backend == "catboost":
        X_cb = prepare_catboost_X(X, cat_cols)
        cat_indices = [X_cb.columns.get_loc(c) for c in cat_cols]
        model = make_catboost_model(target)
        model.fit(X_cb, y, sample_weight=w, cat_features=cat_indices)
        return {"backend": backend, "model": model, "cat_cols": cat_cols, "feature_cols": feature_cols}

    if backend == "lightgbm":
        model = build_lgbm_pipeline(num_cols, cat_cols, target)
        # sample_weight untuk pipeline dikirim ke step terakhir.
        model.fit(X, y, lgbmregressor__sample_weight=w)
        return {"backend": backend, "model": model, "cat_cols": cat_cols, "feature_cols": feature_cols}

    if backend == "fallback":
        model = build_fallback_pipeline(num_cols, cat_cols, target)
        model.fit(X, y, histgradientboostingregressor__sample_weight=w)
        return {"backend": backend, "model": model, "cat_cols": cat_cols, "feature_cols": feature_cols}

    raise ValueError(f"backend tidak dikenal: {backend}")

def predict_backend_model(bundle, df):
    feature_cols = bundle["feature_cols"]
    X = df[feature_cols].copy()
    if bundle["backend"] == "catboost":
        X = prepare_catboost_X(X, bundle["cat_cols"])
    pred = bundle["model"].predict(X)
    return np.asarray(pred, dtype=float)

def available_backends():
    backends = []
    if USE_CATBOOST and CATBOOST_AVAILABLE:
        backends.append("catboost")
    if USE_LIGHTGBM and LIGHTGBM_AVAILABLE:
        backends.append("lightgbm")
    if not backends:
        backends.append("fallback")
    return backends

def backends_for_target(target):
    """Auxiliary target dibatasi supaya runtime tidak naik terlalu ekstrem."""
    backends = available_backends()
    if target in TARGETS:
        return backends

    if LIGHTGBM_AVAILABLE and USE_LIGHTGBM:
        return ["lightgbm"]
    return [backends[0]]

def model_target_list(train_df):
    targets = list(TARGETS)
    if USE_AUXILIARY_TARGETS:
        for t in AUX_TARGETS:
            if t in train_df.columns:
                targets.append(t)
    return targets

def fit_segmented_models(train_df, feature_cols):
    print("Backends utama:", available_backends())

    model_store = {}
    segments = sorted(train_df["gender"].dropna().astype(str).unique()) if USE_GENDER_SEGMENT else ["__all__"]
    all_targets = model_target_list(train_df)
    print("Targets yang dilatih:", all_targets)

    for seg in segments:
        if USE_GENDER_SEGMENT:
            seg_df = train_df[train_df["gender"].astype(str).eq(seg)].copy()
        else:
            seg_df = train_df.copy()

        if len(seg_df) < 1000:
            print(f"Skip segment {seg}, terlalu sedikit row:", len(seg_df))
            continue

        model_store[seg] = {}
        print(f"\n=== Training segment {seg} | rows={len(seg_df):,} ===")

        for target in all_targets:
            if target not in seg_df.columns:
                continue

            model_store[seg][target] = []
            for backend in backends_for_target(target):
                print(f"Training {backend} | segment={seg} | target={target}")
                bundle = fit_backend_model(backend, seg_df, feature_cols, target)
                model_store[seg][target].append(bundle)
                gc.collect()

    return model_store

def _predict_one_target_from_store(model_store, pred_df, target):
    out = np.full(len(pred_df), np.nan, dtype=float)

    for seg, seg_models in model_store.items():
        if target not in seg_models:
            continue

        if USE_GENDER_SEGMENT:
            idx = pred_df["gender"].astype(str).eq(seg).values
        else:
            idx = np.ones(len(pred_df), dtype=bool)

        if idx.sum() == 0:
            continue

        seg_pred_df = pred_df.loc[idx].copy()
        preds = []
        for bundle in seg_models[target]:
            p = predict_backend_model(bundle, seg_pred_df)
            preds.append(p)
        out[idx] = np.mean(preds, axis=0)

    return out

def predict_segmented_models(model_store, pred_df):
    pred_team = _predict_one_target_from_store(model_store, pred_df, "team_goals")
    pred_opp = _predict_one_target_from_store(model_store, pred_df, "opp_goals")

    # fallback kalau ada row yang belum terisi
    if np.isnan(pred_team).any():
        fill = np.nanmedian(pred_team)
        if not np.isfinite(fill):
            fill = 1.25
        pred_team = np.where(np.isnan(pred_team), fill, pred_team)
    if np.isnan(pred_opp).any():
        fill = np.nanmedian(pred_opp)
        if not np.isfinite(fill):
            fill = 1.10
        pred_opp = np.where(np.isnan(pred_opp), fill, pred_opp)

    return np.clip(pred_team, 0, 12), np.clip(pred_opp, 0, 12)

def predict_auxiliary_models(model_store, pred_df, fallback_team_mu=None, fallback_opp_mu=None):
    """Prediksi auxiliary target untuk decoder metric-aware.

    Kalau auxiliary model tidak tersedia, fallback dihitung dari mu team/opp.
    """
    if fallback_team_mu is None:
        fallback_team_mu = np.zeros(len(pred_df), dtype=float)
    if fallback_opp_mu is None:
        fallback_opp_mu = np.zeros(len(pred_df), dtype=float)

    aux = {}
    for target in AUX_TARGETS:
        if any(target in seg_models for seg_models in model_store.values()):
            aux[target] = _predict_one_target_from_store(model_store, pred_df, target)
        else:
            aux[target] = np.full(len(pred_df), np.nan, dtype=float)

    default_goal_diff = np.asarray(fallback_team_mu) - np.asarray(fallback_opp_mu)
    default_total_goals = np.asarray(fallback_team_mu) + np.asarray(fallback_opp_mu)
    default_outcome = np.sign(default_goal_diff)

    aux["goal_diff_target"] = np.where(np.isnan(aux.get("goal_diff_target", np.nan)), default_goal_diff, aux.get("goal_diff_target", default_goal_diff))
    aux["total_goals_target"] = np.where(np.isnan(aux.get("total_goals_target", np.nan)), default_total_goals, aux.get("total_goals_target", default_total_goals))
    aux["outcome_value_target"] = np.where(np.isnan(aux.get("outcome_value_target", np.nan)), default_outcome, aux.get("outcome_value_target", default_outcome))

    aux["goal_diff_target"] = np.clip(aux["goal_diff_target"], -12, 12)
    aux["total_goals_target"] = np.clip(aux["total_goals_target"], 0, 16)
    aux["outcome_value_target"] = np.clip(aux["outcome_value_target"], -1, 1)

    return aux

print("Model helpers ready ✓")

# %% [markdown]
# ## 08. Validasi awal: train model dan prediksi raw
# 
# Cell ini melatih model pada train block dan mengevaluasi raw prediction yang sudah dibulatkan. Setelah itu baru dilakukan symmetrization dan decoder tuning.

# %%
# ============================================================
# 08. Train validation models + raw prediction
# ============================================================

# Patch 4: auxiliary target dibuat setelah feature selection agar tidak bisa ikut jadi fitur.
train_model_fe = add_auxiliary_training_targets(train_model_fe)

valid_model_store = fit_segmented_models(train_model_fe, feature_cols)

valid_pred_team_raw, valid_pred_opp_raw = predict_segmented_models(valid_model_store, valid_fe)
valid_aux_pred_raw = predict_auxiliary_models(
    valid_model_store,
    valid_fe,
    fallback_team_mu=valid_pred_team_raw,
    fallback_opp_mu=valid_pred_opp_raw,
)

valid_weights = valid_raw["tournament"].map(get_tournament_weight).astype(float).values
valid_y_team = valid_raw["team_goals"].astype(int).values
valid_y_opp = valid_raw["opp_goals"].astype(int).values

valid_round_team = np.rint(valid_pred_team_raw).clip(0, 12).astype(int)
valid_round_opp = np.rint(valid_pred_opp_raw).clip(0, 12).astype(int)

score_raw_round = awmae_score(
    valid_y_team, valid_y_opp,
    valid_round_team, valid_round_opp,
    weights=valid_weights,
)

print("Valid AW-MAE raw + round:", score_raw_round)
display(pd.DataFrame({
    "Id": valid_raw[ID_COL].values[:10],
    "true_team_goals": valid_y_team[:10],
    "true_opp_goals": valid_y_opp[:10],
    "pred_team_raw": valid_pred_team_raw[:10],
    "pred_opp_raw": valid_pred_opp_raw[:10],
    "pred_goal_diff_aux": valid_aux_pred_raw["goal_diff_target"][:10],
    "pred_total_aux": valid_aux_pred_raw["total_goals_target"][:10],
    "round_team": valid_round_team[:10],
    "round_opp": valid_round_opp[:10],
}))

# %% [markdown]
# ## 09. Symmetrization per match
# 
# Karena tiap match punya dua baris yang saling mirror, prediksi harus dibuat konsisten:
# 
# - prediksi goal team di baris A harus sama dengan prediksi goal opponent di baris B,
# - prediksi goal opponent di baris A harus sama dengan prediksi goal team di baris B.

# %%
# ============================================================
# 09. Match-level source helpers
# ============================================================

def symmetrize_match_predictions(df, pred_team, pred_opp):
    """Baseline symmetrization rata-rata dari dua baris mirror."""
    return build_match_level_mu(df, pred_team, pred_opp, source_policy="sym_avg")

def build_match_level_mu(df, pred_team, pred_opp, source_policy="sym_avg"):
    """Buat mu yang konsisten di level match.

    Patch 1:
    - tidak selalu memakai average mentah,
    - notebook bisa memilih source policy terbaik berdasarkan validasi.

    source_policy:
    - raw: biarkan prediksi per row apa adanya
    - sym_avg: rata-rata dua baris mirror
    - first_raw: pakai baris pertama match sebagai anchor
    - second_raw: pakai baris kedua match sebagai anchor
    - home_anchor: pakai row yang home jika tersedia, fallback sym_avg
    - higher_total / lower_total: pilih anchor dengan total prediksi lebih tinggi/rendah
    """
    df_temp = df[[ID_COL, MATCH_COL, "team", "opponent"]].copy()
    if "is_home" in df.columns:
        df_temp["is_home"] = pd.to_numeric(df["is_home"], errors="coerce").fillna(0).astype(int)
    else:
        df_temp["is_home"] = 0

    df_temp["_pred_team"] = np.asarray(pred_team, dtype=float)
    df_temp["_pred_opp"] = np.asarray(pred_opp, dtype=float)

    out_team = df_temp["_pred_team"].values.copy()
    out_opp = df_temp["_pred_opp"].values.copy()

    for match_id, idxs in df_temp.groupby(MATCH_COL, sort=False).groups.items():
        idxs = list(idxs)
        if len(idxs) != 2:
            continue

        i, j = idxs[0], idxs[1]

        cand = {}

        # Kandidat 1: rata-rata mirror.
        avg_i = np.nanmean([df_temp.at[i, "_pred_team"], df_temp.at[j, "_pred_opp"]])
        avg_j = np.nanmean([df_temp.at[i, "_pred_opp"], df_temp.at[j, "_pred_team"]])
        cand["sym_avg"] = (avg_i, avg_j)

        # Kandidat 2/3: percaya salah satu orientasi match.
        cand["first_raw"] = (df_temp.at[i, "_pred_team"], df_temp.at[i, "_pred_opp"])
        cand["second_raw"] = (df_temp.at[j, "_pred_opp"], df_temp.at[j, "_pred_team"])

        # Kandidat 4: anchor ke home row kalau tersedia.
        if int(df_temp.at[i, "is_home"]) == 1 and int(df_temp.at[j, "is_home"]) == 0:
            cand["home_anchor"] = cand["first_raw"]
        elif int(df_temp.at[j, "is_home"]) == 1 and int(df_temp.at[i, "is_home"]) == 0:
            cand["home_anchor"] = cand["second_raw"]
        else:
            cand["home_anchor"] = cand["sym_avg"]

        total_first = cand["first_raw"][0] + cand["first_raw"][1]
        total_second = cand["second_raw"][0] + cand["second_raw"][1]
        cand["higher_total"] = cand["first_raw"] if total_first >= total_second else cand["second_raw"]
        cand["lower_total"] = cand["first_raw"] if total_first <= total_second else cand["second_raw"]

        score_i, score_j = cand.get(source_policy, cand["sym_avg"])

        out_team[i] = score_i
        out_opp[i] = score_j
        out_team[j] = score_j
        out_opp[j] = score_i

    return np.clip(out_team, 0, 12), np.clip(out_opp, 0, 12)

def transform_aux_to_match_source(df, aux_pred, source_policy="sym_avg"):
    """Mirror auxiliary predictions supaya konsisten dengan source policy.

    goal_diff_target bersifat signed terhadap perspektif row.
    total_goals_target simetris.
    outcome_value_target signed terhadap perspektif row.
    """
    if aux_pred is None:
        return None

    gd_raw = np.asarray(aux_pred["goal_diff_target"], dtype=float)
    total_raw = np.asarray(aux_pred["total_goals_target"], dtype=float)
    out_raw = np.asarray(aux_pred["outcome_value_target"], dtype=float)

    # Untuk goal diff dan outcome, row lawan harus dibalik sign-nya.
    gd_team, gd_opp = build_match_level_mu(df, gd_raw, -gd_raw, source_policy=source_policy)
    out_team, out_opp = build_match_level_mu(df, out_raw, -out_raw, source_policy=source_policy)

    # Untuk total goals, dua orientasi punya nilai yang sama.
    total_team, _ = build_match_level_mu(df, total_raw, total_raw, source_policy=source_policy)

    return {
        "goal_diff_target": np.clip(gd_team, -12, 12),
        "total_goals_target": np.clip(total_team, 0, 16),
        "outcome_value_target": np.clip(out_team, -1, 1),
    }

valid_pred_team_sym, valid_pred_opp_sym = build_match_level_mu(
    valid_raw.reset_index(drop=True),
    valid_pred_team_raw,
    valid_pred_opp_raw,
    source_policy="sym_avg",
)

valid_sym_round_team = np.rint(valid_pred_team_sym).clip(0, 12).astype(int)
valid_sym_round_opp = np.rint(valid_pred_opp_sym).clip(0, 12).astype(int)

score_sym_round = awmae_score(
    valid_y_team, valid_y_opp,
    valid_sym_round_team, valid_sym_round_opp,
    weights=valid_weights,
)

print("Valid AW-MAE raw round :", score_raw_round)
print("Valid AW-MAE sym round :", score_sym_round)

# %% [markdown]
# ## 10. Decoder tuning
# 
# Model mengeluarkan expected goals berbentuk float. Karena metric menghukum exact score, outcome, dan goal difference, kita coba beberapa decoder:
# 
# 1. `round`: pembulatan biasa.
# 2. `floor`: cenderung konservatif.
# 3. `poisson_mbr`: memilih skor integer dengan expected AW-MAE terkecil berdasarkan distribusi Poisson dari expected goals.
# 
# Decoder terbaik dipilih dari validation AW-MAE.

# %%
# ============================================================
# 10. Advanced decoder, calibration, dan segment postprocessor
# ============================================================

_LOSS_CACHE = {}

def poisson_pmf_matrix(mu, max_goal=12):
    """Return shape (n, max_goal+1). Tail probability dimasukkan ke bin max_goal."""
    mu = np.asarray(mu, dtype=float)
    mu = np.clip(mu, 1e-4, 12.0)

    n = len(mu)
    pmf = np.zeros((n, max_goal + 1), dtype=float)
    pmf[:, 0] = np.exp(-mu)

    for k in range(1, max_goal + 1):
        pmf[:, k] = pmf[:, k - 1] * mu / k

    # Fold tail into last bin supaya total prob = 1.
    sums = pmf.sum(axis=1)
    pmf[:, -1] += np.maximum(0.0, 1.0 - sums)
    pmf = pmf / pmf.sum(axis=1, keepdims=True)
    return pmf

def make_loss_matrix(max_true=12, max_pred=10):
    key = (int(max_true), int(max_pred))
    if key in _LOSS_CACHE:
        return _LOSS_CACHE[key]

    true_pairs = [(a, b) for a in range(max_true + 1) for b in range(max_true + 1)]
    pred_pairs = [(a, b) for a in range(max_pred + 1) for b in range(max_pred + 1)]

    loss = np.zeros((len(pred_pairs), len(true_pairs)), dtype=float)

    yta = np.array([p[0] for p in true_pairs])
    ytb = np.array([p[1] for p in true_pairs])

    for i, (pa, pb) in enumerate(pred_pairs):
        loss[i] = official_match_loss(
            yta,
            ytb,
            np.full_like(yta, pa),
            np.full_like(ytb, pb),
        )

    result = (np.asarray(pred_pairs, dtype=int), np.asarray(true_pairs, dtype=int), loss)
    _LOSS_CACHE[key] = result
    return result

def calibrate_mu(mu, scale=1.0, offset=0.0):
    return np.clip(np.asarray(mu, dtype=float) * float(scale) + float(offset), 0.02, 12.0)

def poisson_mbr_decode(
    mu_team,
    mu_opp,
    max_true=12,
    max_pred=10,
    scale=1.0,
    offset=0.0,
    aux_pred=None,
    lambda_gd=0.0,
    lambda_total=0.0,
    lambda_outcome=0.0,
    batch_size=12000,
):
    """MBR decoder yang bisa ditambah auxiliary penalty.

    Patch 4:
    candidate scoreline tidak hanya mengikuti expected goals, tapi juga diarahkan
    oleh prediksi goal difference, total goals, dan outcome.
    """
    mu_team = calibrate_mu(mu_team, scale=scale, offset=offset)
    mu_opp = calibrate_mu(mu_opp, scale=scale, offset=offset)

    pred_pairs, true_pairs, loss_matrix = make_loss_matrix(max_true=max_true, max_pred=max_pred)
    pair_team = pred_pairs[:, 0]
    pair_opp = pred_pairs[:, 1]
    pair_gd = pair_team - pair_opp
    pair_total = pair_team + pair_opp
    pair_outcome = np.sign(pair_gd)

    out_team = np.zeros(len(mu_team), dtype=int)
    out_opp = np.zeros(len(mu_team), dtype=int)

    use_aux = aux_pred is not None and (lambda_gd > 0 or lambda_total > 0 or lambda_outcome > 0)
    if use_aux:
        aux_gd = np.asarray(aux_pred.get("goal_diff_target", mu_team - mu_opp), dtype=float)
        aux_total = np.asarray(aux_pred.get("total_goals_target", mu_team + mu_opp), dtype=float)
        aux_outcome = np.sign(np.asarray(aux_pred.get("outcome_value_target", mu_team - mu_opp), dtype=float))

    for start in range(0, len(mu_team), batch_size):
        end = min(start + batch_size, len(mu_team))
        pmf_t = poisson_pmf_matrix(mu_team[start:end], max_goal=max_true)
        pmf_o = poisson_pmf_matrix(mu_opp[start:end], max_goal=max_true)

        # shape: batch x true_pair
        true_probs = np.einsum("bi,bj->bij", pmf_t, pmf_o).reshape(end - start, -1)

        # expected loss: batch x pred_pair
        exp_loss = true_probs @ loss_matrix.T

        if use_aux:
            aux_pen = np.zeros_like(exp_loss)
            if lambda_gd > 0:
                aux_pen += float(lambda_gd) * np.abs(pair_gd[None, :] - aux_gd[start:end, None])
            if lambda_total > 0:
                aux_pen += float(lambda_total) * np.abs(pair_total[None, :] - aux_total[start:end, None])
            if lambda_outcome > 0:
                aux_pen += float(lambda_outcome) * (pair_outcome[None, :] != aux_outcome[start:end, None]).astype(float)
            exp_loss = exp_loss + aux_pen

        best_idx = np.argmin(exp_loss, axis=1)
        best_pairs = pred_pairs[best_idx]

        out_team[start:end] = best_pairs[:, 0]
        out_opp[start:end] = best_pairs[:, 1]

    return out_team, out_opp

def apply_decoder_config(mu_team, mu_opp, config, aux_pred=None):
    decoder = config["decoder"]
    scale = float(config.get("scale", 1.0))
    offset = float(config.get("offset", 0.0))

    mu_team_cal = calibrate_mu(mu_team, scale=scale, offset=offset)
    mu_opp_cal = calibrate_mu(mu_opp, scale=scale, offset=offset)

    if decoder == "round":
        return np.rint(mu_team_cal).clip(0, 12).astype(int), np.rint(mu_opp_cal).clip(0, 12).astype(int)

    if decoder == "floor":
        return np.floor(mu_team_cal).clip(0, 12).astype(int), np.floor(mu_opp_cal).clip(0, 12).astype(int)

    if decoder == "ceil":
        return np.ceil(mu_team_cal).clip(0, 12).astype(int), np.ceil(mu_opp_cal).clip(0, 12).astype(int)

    if decoder == "poisson_mbr":
        return poisson_mbr_decode(
            mu_team,
            mu_opp,
            max_true=int(config.get("max_true", 12)),
            max_pred=int(config.get("max_pred", 8)),
            scale=scale,
            offset=offset,
            aux_pred=aux_pred,
            lambda_gd=float(config.get("lambda_gd", 0.0)),
            lambda_total=float(config.get("lambda_total", 0.0)),
            lambda_outcome=float(config.get("lambda_outcome", 0.0)),
        )

    raise ValueError(f"decoder tidak dikenal: {decoder}")

def evaluate_decoder_grid(y_team, y_opp, mu_team, mu_opp, weights, aux_pred=None, compact=False):
    rows = []

    scale_grid = [0.95, 1.00, 1.05, 1.10, 1.15] if compact else [0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20]
    offset_grid = [-0.10, 0.00, 0.10] if compact else [-0.15, -0.10, -0.05, 0.00, 0.05, 0.10, 0.15]

    # Baseline round/floor/ceil dengan calibration.
    for scale in scale_grid:
        for offset in offset_grid:
            for name in ["round", "floor", "ceil"]:
                config = {
                    "decoder": name,
                    "scale": scale,
                    "offset": offset,
                    "max_true": None,
                    "max_pred": None,
                    "lambda_gd": 0.0,
                    "lambda_total": 0.0,
                    "lambda_outcome": 0.0,
                }
                pt, po = apply_decoder_config(mu_team, mu_opp, config, aux_pred=aux_pred)
                rows.append({
                    **config,
                    "awmae": awmae_score(y_team, y_opp, pt, po, weights=weights),
                })

    # MBR grid. Dibuat compact agar masih reasonable untuk dijalankan.
    aux_lambda_grid = [
        (0.00, 0.00, 0.00),
        (0.03, 0.02, 0.10),
        (0.05, 0.03, 0.15),
    ]
    max_pred_grid = [7, 8, 10] if not compact else [7, 8]

    for scale in scale_grid:
        for offset in offset_grid:
            for max_pred in max_pred_grid:
                for lambda_gd, lambda_total, lambda_outcome in aux_lambda_grid:
                    config = {
                        "decoder": "poisson_mbr",
                        "scale": scale,
                        "offset": offset,
                        "max_true": 12,
                        "max_pred": max_pred,
                        "lambda_gd": lambda_gd,
                        "lambda_total": lambda_total,
                        "lambda_outcome": lambda_outcome,
                    }
                    pt, po = apply_decoder_config(mu_team, mu_opp, config, aux_pred=aux_pred)
                    rows.append({
                        **config,
                        "awmae": awmae_score(y_team, y_opp, pt, po, weights=weights),
                    })

    report = pd.DataFrame(rows).sort_values("awmae").reset_index(drop=True)
    return report

def tune_postprocessor_for_subset(
    df,
    y_team,
    y_opp,
    raw_team_mu,
    raw_opp_mu,
    weights,
    aux_raw=None,
    source_policies=None,
    compact=False,
):
    """Tune source policy + calibration + decoder untuk satu segment."""
    if source_policies is None:
        # first_raw / second_raw adalah match-level candidate selection yang valid untuk test,
        # karena policy-nya dipilih dari validation, bukan dari target test.
        source_policies = ["sym_avg", "home_anchor", "first_raw", "second_raw"]

    reports = []
    for source_policy in source_policies:
        mu_team, mu_opp = build_match_level_mu(
            df.reset_index(drop=True),
            raw_team_mu,
            raw_opp_mu,
            source_policy=source_policy,
        )
        aux_src = transform_aux_to_match_source(df.reset_index(drop=True), aux_raw, source_policy=source_policy)

        rep = evaluate_decoder_grid(
            y_team,
            y_opp,
            mu_team,
            mu_opp,
            weights,
            aux_pred=aux_src,
            compact=compact,
        )
        rep["source_policy"] = source_policy
        reports.append(rep)

    report = pd.concat(reports, ignore_index=True).sort_values("awmae").reset_index(drop=True)
    best = report.iloc[0].to_dict()
    return best, report

def fit_segment_postprocessors(
    valid_df,
    y_team,
    y_opp,
    raw_team_mu,
    raw_opp_mu,
    weights,
    aux_raw=None,
    segment_col="gender",
):
    """Patch 2 dan 3: decoder + calibration per segment.

    Global config tetap disimpan sebagai fallback. Jika segment terlalu kecil,
    dia akan memakai global config agar tidak overfit.
    """
    valid_df = valid_df.reset_index(drop=True).copy()
    y_team = np.asarray(y_team)
    y_opp = np.asarray(y_opp)
    weights = np.asarray(weights, dtype=float)

    post = {
        "segment_col": segment_col,
        "global": None,
        "by_segment": {},
    }

    print("Tuning global postprocessor...")
    global_best, global_report = tune_postprocessor_for_subset(
        valid_df,
        y_team,
        y_opp,
        raw_team_mu,
        raw_opp_mu,
        weights,
        aux_raw=aux_raw,
        compact=False,
    )
    post["global"] = global_best
    all_reports = [global_report.assign(segment="__global__", n_rows=len(valid_df))]

    if segment_col in valid_df.columns:
        for seg in sorted(valid_df[segment_col].dropna().astype(str).unique()):
            idx = valid_df[segment_col].astype(str).eq(seg).values
            n = int(idx.sum())
            if n < 200:
                print(f"Skip postprocessor segment {seg}, row terlalu sedikit: {n}")
                continue

            print(f"Tuning postprocessor segment={seg} | rows={n:,}")
            best, rep = tune_postprocessor_for_subset(
                valid_df.loc[idx].reset_index(drop=True),
                y_team[idx],
                y_opp[idx],
                raw_team_mu[idx],
                raw_opp_mu[idx],
                weights[idx],
                aux_raw={k: np.asarray(v)[idx] for k, v in aux_raw.items()} if aux_raw is not None else None,
                compact=True,
            )
            post["by_segment"][str(seg)] = best
            all_reports.append(rep.assign(segment=str(seg), n_rows=n))

    full_report = pd.concat(all_reports, ignore_index=True)
    full_report = full_report.sort_values(["segment", "awmae"]).reset_index(drop=True)
    return post, full_report

def enforce_integer_mirror(df, pred_team, pred_opp):
    """Pastikan output integer konsisten untuk dua row pada match yang sama."""
    df_temp = df[[MATCH_COL]].reset_index(drop=True).copy()
    pt = np.asarray(pred_team, dtype=int).copy()
    po = np.asarray(pred_opp, dtype=int).copy()

    for match_id, idxs in df_temp.groupby(MATCH_COL, sort=False).groups.items():
        idxs = list(idxs)
        if len(idxs) != 2:
            continue
        i, j = idxs[0], idxs[1]
        score_i = int(np.rint(np.mean([pt[i], po[j]])))
        score_j = int(np.rint(np.mean([po[i], pt[j]])))
        pt[i], po[i] = score_i, score_j
        pt[j], po[j] = score_j, score_i

    return np.clip(pt, 0, 12).astype(int), np.clip(po, 0, 12).astype(int)

def apply_segment_postprocessors(df, raw_team_mu, raw_opp_mu, postprocessor, aux_raw=None):
    """Apply config global/per segment ke valid/test."""
    df = df.reset_index(drop=True).copy()
    out_team = np.zeros(len(df), dtype=int)
    out_opp = np.zeros(len(df), dtype=int)

    segment_col = postprocessor.get("segment_col", "gender")
    if segment_col in df.columns:
        segments = df[segment_col].astype(str).fillna("__MISSING__").values
    else:
        segments = np.array(["__global__"] * len(df))

    unique_sources = set()
    unique_sources.add(str(postprocessor["global"].get("source_policy", "sym_avg")))
    for cfg in postprocessor.get("by_segment", {}).values():
        unique_sources.add(str(cfg.get("source_policy", "sym_avg")))

    mu_cache = {}
    aux_cache = {}
    for source_policy in unique_sources:
        mu_cache[source_policy] = build_match_level_mu(df, raw_team_mu, raw_opp_mu, source_policy=source_policy)
        aux_cache[source_policy] = transform_aux_to_match_source(df, aux_raw, source_policy=source_policy)

    for seg in sorted(set(segments)):
        idx = segments == seg
        config = postprocessor.get("by_segment", {}).get(str(seg), postprocessor["global"])
        source_policy = str(config.get("source_policy", "sym_avg"))

        mu_team, mu_opp = mu_cache[source_policy]
        aux_src = aux_cache[source_policy]
        aux_subset = {k: np.asarray(v)[idx] for k, v in aux_src.items()} if aux_src is not None else None

        pt, po = apply_decoder_config(
            mu_team[idx],
            mu_opp[idx],
            config,
            aux_pred=aux_subset,
        )
        out_team[idx] = pt
        out_opp[idx] = po

    return enforce_integer_mirror(df, out_team, out_opp)

def json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        if np.isnan(obj):
            return None
        return float(obj)
    try:
        if obj is None or (not isinstance(obj, (str, bytes)) and pd.isna(obj)):
            return None
    except Exception:
        pass
    return obj

print("Advanced decoder helpers ready ✓")

# %%
# ============================================================
# 11. Validation final score dengan advanced postprocessor
# ============================================================

if USE_ADVANCED_POSTPROCESSOR:
    segment_postprocessor, postprocessor_report = fit_segment_postprocessors(
        valid_raw.reset_index(drop=True),
        valid_y_team,
        valid_y_opp,
        valid_pred_team_raw,
        valid_pred_opp_raw,
        valid_weights,
        aux_raw=valid_aux_pred_raw,
        segment_col="gender",
    )

    valid_pred_team_final, valid_pred_opp_final = apply_segment_postprocessors(
        valid_raw.reset_index(drop=True),
        valid_pred_team_raw,
        valid_pred_opp_raw,
        segment_postprocessor,
        aux_raw=valid_aux_pred_raw,
    )

    best_decoder = segment_postprocessor["global"]
    decoder_report = postprocessor_report.copy()
else:
    # Fallback lama: global decoder dari sym_avg.
    decoder_report = evaluate_decoder_grid(
        valid_y_team,
        valid_y_opp,
        valid_pred_team_sym,
        valid_pred_opp_sym,
        valid_weights,
        aux_pred=transform_aux_to_match_source(valid_raw.reset_index(drop=True), valid_aux_pred_raw, source_policy="sym_avg"),
    )
    best_decoder = decoder_report.iloc[0].to_dict()
    segment_postprocessor = {
        "segment_col": "gender",
        "global": best_decoder,
        "by_segment": {},
    }
    valid_pred_team_final, valid_pred_opp_final = apply_segment_postprocessors(
        valid_raw.reset_index(drop=True),
        valid_pred_team_raw,
        valid_pred_opp_raw,
        segment_postprocessor,
        aux_raw=valid_aux_pred_raw,
    )

valid_awmae_final = awmae_score(
    valid_y_team,
    valid_y_opp,
    valid_pred_team_final,
    valid_pred_opp_final,
    weights=valid_weights,
)

print("=" * 80)
print("VALIDATION SUMMARY")
print("=" * 80)
print("Raw + round AW-MAE :", score_raw_round)
print("Sym + round AW-MAE :", score_sym_round)
print("Advanced AW-MAE    :", valid_awmae_final)
print("Global decoder config:", best_decoder)
print("Segment configs:", segment_postprocessor.get("by_segment", {}))

decoder_report.to_csv(OUT_DIR / "advanced_decoder_report.csv", index=False)
with open(OUT_DIR / "segment_postprocessor.json", "w", encoding="utf-8") as f:
    json.dump(json_safe(segment_postprocessor), f, ensure_ascii=False, indent=2)

valid_pred_df = valid_raw[[ID_COL, MATCH_COL, "date", "gender", "team", "opponent", "tournament", "team_goals", "opp_goals"]].copy()
valid_pred_df["pred_team_raw"] = valid_pred_team_raw
valid_pred_df["pred_opp_raw"] = valid_pred_opp_raw
valid_pred_df["pred_team_sym"] = valid_pred_team_sym
valid_pred_df["pred_opp_sym"] = valid_pred_opp_sym
valid_pred_df["pred_goal_diff_aux"] = valid_aux_pred_raw["goal_diff_target"]
valid_pred_df["pred_total_aux"] = valid_aux_pred_raw["total_goals_target"]
valid_pred_df["pred_outcome_aux"] = valid_aux_pred_raw["outcome_value_target"]
valid_pred_df["pred_team_goals"] = valid_pred_team_final
valid_pred_df["pred_opp_goals"] = valid_pred_opp_final
valid_pred_df["row_loss"] = official_match_loss(
    valid_pred_df["team_goals"],
    valid_pred_df["opp_goals"],
    valid_pred_df["pred_team_goals"],
    valid_pred_df["pred_opp_goals"],
)
valid_pred_df["weight"] = valid_pred_df["tournament"].map(get_tournament_weight)

valid_pred_df.to_csv(OUT_DIR / "validation_predictions_advanced.csv", index=False)

display(decoder_report.groupby("segment").head(5).reset_index(drop=True))
display(valid_pred_df.head(10))
display(valid_pred_df.groupby("gender").apply(
    lambda g: awmae_score(
        g["team_goals"],
        g["opp_goals"],
        g["pred_team_goals"],
        g["pred_opp_goals"],
        weights=g["weight"],
    )
).rename("awmae_by_gender").reset_index())

# %% [markdown]
# ## 11B. Optional multi-fold dan start-date search
# 
# Bagian ini mengimplementasikan patch nomor 7. Default-nya **tidak langsung dijalankan** karena akan melatih ulang model berkali-kali dan runtime bisa sangat panjang. Kalau mau benar-benar memilih `MODEL_TRAIN_START_DATE` secara lebih kuat, ubah `RUN_EXPENSIVE_MULTI_FOLD_SEARCH = True` di cell setup, lalu run dari awal.
# 
# Validasi default notebook tetap memakai holdout 2008–2011 supaya workflow utama tidak terlalu berat.

# %%
# ============================================================
# 11B. Optional multi-fold temporal validation + model start search
# ============================================================

def build_fold_ranges(fold_start_dates, max_date):
    starts = [pd.Timestamp(x) for x in fold_start_dates]
    ranges = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else pd.Timestamp(max_date) + pd.Timedelta(days=1)
        ranges.append((start, end))
    return ranges

def run_single_temporal_fold(fold_start, fold_end, model_start_date):
    fold_start = pd.Timestamp(fold_start)
    fold_end = pd.Timestamp(fold_end)
    model_start_date = pd.Timestamp(model_start_date)

    tr_raw = train_raw[pd.to_datetime(train_raw["date"]) < fold_start].reset_index(drop=True).copy()
    va_raw = train_raw[
        (pd.to_datetime(train_raw["date"]) >= fold_start) &
        (pd.to_datetime(train_raw["date"]) < fold_end)
    ].reset_index(drop=True).copy()

    if len(tr_raw) < 3000 or len(va_raw) < 200:
        return None

    tr_fe, va_fe = make_feature_frames(tr_raw, va_raw)
    tr_model = tr_fe[pd.to_datetime(tr_fe["date"]) >= model_start_date].reset_index(drop=True).copy()

    base_cols, _, _ = select_feature_columns(tr_model, va_fe)

    if USE_SUPERVISED_SIGNAL:
        tr_model, va_fe, _ = add_supervised_signal_features(tr_model, va_fe, base_cols)
        feat_cols, _, _ = select_feature_columns(tr_model, va_fe)
    else:
        feat_cols = base_cols

    tr_model = add_auxiliary_training_targets(tr_model)
    store = fit_segmented_models(tr_model, feat_cols)

    raw_t, raw_o = predict_segmented_models(store, va_fe)
    aux = predict_auxiliary_models(store, va_fe, fallback_team_mu=raw_t, fallback_opp_mu=raw_o)

    y_t = va_raw["team_goals"].astype(int).values
    y_o = va_raw["opp_goals"].astype(int).values
    w = va_raw["tournament"].map(get_tournament_weight).astype(float).values

    post, rep = fit_segment_postprocessors(
        va_raw.reset_index(drop=True),
        y_t,
        y_o,
        raw_t,
        raw_o,
        w,
        aux_raw=aux,
        segment_col="gender",
    )
    pt, po = apply_segment_postprocessors(
        va_raw.reset_index(drop=True),
        raw_t,
        raw_o,
        post,
        aux_raw=aux,
    )

    score = awmae_score(y_t, y_o, pt, po, weights=w)
    return {
        "fold_start": fold_start.date().isoformat(),
        "fold_end": (fold_end - pd.Timedelta(days=1)).date().isoformat(),
        "model_start_date": model_start_date.date().isoformat(),
        "n_train": len(tr_model),
        "n_valid": len(va_raw),
        "awmae": score,
        "global_source": post["global"].get("source_policy"),
        "global_decoder": post["global"].get("decoder"),
        "global_scale": post["global"].get("scale"),
        "global_offset": post["global"].get("offset"),
    }

if RUN_EXPENSIVE_MULTI_FOLD_SEARCH:
    max_train_date = pd.to_datetime(train_raw["date"]).max()
    fold_ranges = build_fold_ranges(TEMPORAL_FOLD_START_DATES, max_train_date)

    rows = []
    for model_start_date in MODEL_TRAIN_START_CANDIDATES:
        print("=" * 90)
        print("MODEL_START_DATE:", model_start_date)
        print("=" * 90)

        for fold_start, fold_end in fold_ranges:
            print(f"Fold {fold_start.date()} -> {(fold_end - pd.Timedelta(days=1)).date()}")
            result = run_single_temporal_fold(fold_start, fold_end, model_start_date)
            if result is not None:
                rows.append(result)
                display(pd.DataFrame(rows).tail(1))
            gc.collect()

    multifold_report = pd.DataFrame(rows)
    multifold_report.to_csv(OUT_DIR / "multifold_start_date_search.csv", index=False)

    if len(multifold_report):
        display(multifold_report)
        display(
            multifold_report
            .groupby("model_start_date")["awmae"]
            .agg(["mean", "std", "count"])
            .sort_values("mean")
            .reset_index()
        )
else:
    print("Multi-fold/start-date search dilewati.")
    print("Untuk menjalankannya, ubah RUN_EXPENSIVE_MULTI_FOLD_SEARCH = True di cell setup lalu run ulang dari awal.")

# %% [markdown]
# ## 12. Final training untuk test
# 
# Setelah decoder dipilih dari validasi, model final dilatih ulang memakai seluruh train sampai 2011. Prediction dibuat untuk `test.csv`, lalu disimpan sebagai `submission_awmae_pipeline.csv`.

# %%
# ============================================================
# 12. Build final train/test features
# ============================================================

final_train_fe, test_fe = make_feature_frames(train_raw, test_raw)

final_model_mask = pd.to_datetime(final_train_fe["date"]) >= model_start
final_model_fe = final_train_fe[final_model_mask].reset_index(drop=True).copy()

final_base_feature_cols, final_removed_missing, final_removed_constant = select_feature_columns(final_model_fe, test_fe)

if USE_SUPERVISED_SIGNAL:
    final_model_fe, test_fe, final_signal_cols = add_supervised_signal_features(
        final_model_fe,
        test_fe,
        final_base_feature_cols,
    )
    final_feature_cols, final_removed_missing, final_removed_constant = select_feature_columns(final_model_fe, test_fe)
else:
    final_feature_cols = final_base_feature_cols

print("final_train_fe shape :", final_train_fe.shape)
print("final_model_fe shape :", final_model_fe.shape)
print("test_fe shape        :", test_fe.shape)
print("final features       :", len(final_feature_cols))

with open(OUT_DIR / "feature_columns.json", "w", encoding="utf-8") as f:
    json.dump(final_feature_cols, f, ensure_ascii=False, indent=2)

# %% [markdown]
# ## 12b. Cluster features untuk final pipeline
# 
# Refit cluster pakai train_raw lengkap, attach ke final_model_fe dan test_fe, lalu re-run feature selection + supervised signal.

# %%
# ============================================================
# 12b. Attach extreme-cluster features to final pipeline
# ============================================================

if USE_EXTREME_CLUSTERS:
    #
    # Untuk final block kita refit cluster pakai train_raw lengkap (sampai
    # 2011-08-04) supaya cluster aware terhadap era yang sedikit lebih dekat
    # dengan test block. Fungsi yang dipakai sama dengan section 05b.

    country_extreme_stats_final_M = compute_country_extreme_stats(train_raw, "M")
    country_extreme_stats_final_W = compute_country_extreme_stats(train_raw, "W")

    cluster_labels_final_M, cluster_artifact_final_M, _ = fit_extreme_country_clusters(country_extreme_stats_final_M)
    cluster_labels_final_W, cluster_artifact_final_W, _ = fit_extreme_country_clusters(country_extreme_stats_final_W)

    country_cluster_labels_final = {**cluster_labels_final_M, **cluster_labels_final_W}

    final_model_fe = attach_country_cluster_features(final_model_fe, country_cluster_labels_final)
    test_fe = attach_country_cluster_features(test_fe, country_cluster_labels_final)

    final_base_feature_cols, _, _ = select_feature_columns(final_model_fe, test_fe)

    if USE_SUPERVISED_SIGNAL:
        final_model_fe, test_fe, final_signal_cols = add_supervised_signal_features(
            final_model_fe,
            test_fe,
            final_base_feature_cols,
        )
        final_feature_cols, _, _ = select_feature_columns(final_model_fe, test_fe)
    else:
        final_feature_cols = final_base_feature_cols

    print("Final cluster attach ✓")
    print("Final feature count:", len(final_feature_cols))
    print("Cluster cat features in final:",
          [c for c in final_feature_cols if "cluster" in c])

    # Re-save feature columns
    with open(OUT_DIR / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(final_feature_cols, f, ensure_ascii=False, indent=2)

else:
    print("Final extreme-cluster attach dilewati untuk ablation variant:", EXP18_ABLATION_VARIANT)
    country_cluster_labels_final = {}
    # final_feature_cols sudah dibuat di section 12. Re-save untuk folder variant ini.
    with open(OUT_DIR / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(final_feature_cols, f, ensure_ascii=False, indent=2)
    print("Final feature count:", len(final_feature_cols))

# %%
# ============================================================
# 13. Train final models and predict test
# ============================================================

# Patch 4: auxiliary targets juga dibuat pada final train block, tapi tidak masuk feature_cols.
final_model_fe = add_auxiliary_training_targets(final_model_fe)

final_model_store = fit_segmented_models(final_model_fe, final_feature_cols)

test_pred_team_raw, test_pred_opp_raw = predict_segmented_models(final_model_store, test_fe)
test_aux_pred_raw = predict_auxiliary_models(
    final_model_store,
    test_fe,
    fallback_team_mu=test_pred_team_raw,
    fallback_opp_mu=test_pred_opp_raw,
)

test_pred_team_sym, test_pred_opp_sym = build_match_level_mu(
    test_raw.reset_index(drop=True),
    test_pred_team_raw,
    test_pred_opp_raw,
    source_policy="sym_avg",
)

if USE_ADVANCED_POSTPROCESSOR:
    test_pred_team_final, test_pred_opp_final = apply_segment_postprocessors(
        test_raw.reset_index(drop=True),
        test_pred_team_raw,
        test_pred_opp_raw,
        segment_postprocessor,
        aux_raw=test_aux_pred_raw,
    )
else:
    test_pred_team_final, test_pred_opp_final = apply_segment_postprocessors(
        test_raw.reset_index(drop=True),
        test_pred_team_raw,
        test_pred_opp_raw,
        {
            "segment_col": "gender",
            "global": best_decoder,
            "by_segment": {},
        },
        aux_raw=test_aux_pred_raw,
    )

submission = pd.DataFrame({
    ID_COL: test_raw[ID_COL].values,
    "team_goals": test_pred_team_final.astype(int),
    "opp_goals": test_pred_opp_final.astype(int),
})

assert len(submission) == len(test_raw)
assert submission[ID_COL].equals(test_raw[ID_COL].reset_index(drop=True))
assert submission["team_goals"].between(0, 12).all()
assert submission["opp_goals"].between(0, 12).all()

submission_path = OUT_DIR / "submission_awmae_pipeline_advanced.csv"
submission.to_csv(submission_path, index=False)

# Simpan juga raw/diagnostic prediction untuk audit.
test_prediction_audit = test_raw[[ID_COL, MATCH_COL, "date", "gender", "team", "opponent", "tournament"]].copy()
test_prediction_audit["pred_team_raw"] = test_pred_team_raw
test_prediction_audit["pred_opp_raw"] = test_pred_opp_raw
test_prediction_audit["pred_team_sym"] = test_pred_team_sym
test_prediction_audit["pred_opp_sym"] = test_pred_opp_sym
test_prediction_audit["pred_goal_diff_aux"] = test_aux_pred_raw["goal_diff_target"]
test_prediction_audit["pred_total_aux"] = test_aux_pred_raw["total_goals_target"]
test_prediction_audit["pred_team_goals"] = submission["team_goals"]
test_prediction_audit["pred_opp_goals"] = submission["opp_goals"]
test_prediction_audit.to_csv(OUT_DIR / "test_prediction_audit_advanced.csv", index=False)

print("Submission saved to:", submission_path.resolve())
display(submission.head(20))
display(submission[["team_goals", "opp_goals"]].describe())

# %% [markdown]
# ## 13b. Pseudo-labelling untuk handle domain shift 2011-2026
# 
# Ambil prediksi test paling confident sebagai label tambahan, refit final_model_store dengan sample_weight lebih kecil untuk pseudo-rows. Bonus weight untuk W segment karena distribusinya shift paling drastis dari 2010-an ke 2020-an.

# %%
# ============================================================
# 13b. Pseudo-labelling untuk handle domain shift 2011-2026
# ============================================================

if USE_PSEUDO_LABELLING:
    #
    # Motif:
    #   - Test block (2011-2026) jauh lebih besar dari train (1872-2011 → 1990-2011
    #     setelah cutoff). Distribusi modern era cukup berbeda, terutama untuk W
    #     football yang dari 2010-an ke 2020-an pengalami profesionalisasi besar.
    #   - Pseudo-labelling: ambil prediksi test paling confident, jadikan label
    #     tambahan, refit model, predict ulang.
    #
    # Cara pilih confident sample:
    #   1) Symmetrization stable: |pred_team_raw - pred_team_sym| < eps dan sama
    #      untuk opp. Artinya dua orientasi match setuju.
    #   2) Auxiliary head konsisten: |goal_diff_aux - (mu_team - mu_opp)| < eps_gd.
    #   3) Total goals aux konsisten: |total_aux - (mu_team + mu_opp)| < eps_total.
    #   4) Tidak terlalu banyak score nol-nol; harus ada signal goal.
    #   5) Bobot lebih kecil daripada match train asli (0.5 default).
    #
    # Pseudo-label yang dipilih MASUK sebagai data train tambahan via concat ke
    # final_model_fe + recompute auxiliary target. Lalu refit final_model_store.

    PSEUDO_LABEL_ITERATIONS = 1     # cukup 1 iter; lebih bisa overfit ke prediksi sendiri
    PSEUDO_LABEL_CONF_EPS_MU = 0.35
    PSEUDO_LABEL_CONF_EPS_GD = 0.55
    PSEUDO_LABEL_CONF_EPS_TOTAL = 0.70
    PSEUDO_LABEL_WEIGHT = 0.5       # multiplier bobot pseudo-label
    PSEUDO_LABEL_W_PRIORITY_WEIGHT = 0.7  # W dapat sample weight lebih tinggi
    PSEUDO_LABEL_MIN_FRACTION = 0.15
    PSEUDO_LABEL_MAX_FRACTION = 0.55


    def select_confident_pseudo_labels(
        test_meta_df,
        pred_team_raw,
        pred_opp_raw,
        pred_team_sym,
        pred_opp_sym,
        aux_pred,
        eps_mu=PSEUDO_LABEL_CONF_EPS_MU,
        eps_gd=PSEUDO_LABEL_CONF_EPS_GD,
        eps_total=PSEUDO_LABEL_CONF_EPS_TOTAL,
    ):
        """Return boolean mask of test rows yang predictionsnya confident."""
        pred_team_raw = np.asarray(pred_team_raw, dtype=float)
        pred_opp_raw = np.asarray(pred_opp_raw, dtype=float)
        pred_team_sym = np.asarray(pred_team_sym, dtype=float)
        pred_opp_sym = np.asarray(pred_opp_sym, dtype=float)

        diff_team = np.abs(pred_team_raw - pred_team_sym)
        diff_opp = np.abs(pred_opp_raw - pred_opp_sym)
        sym_consistent = (diff_team <= eps_mu) & (diff_opp <= eps_mu)

        if aux_pred is not None and "goal_diff_target" in aux_pred:
            gd_aux = np.asarray(aux_pred["goal_diff_target"], dtype=float)
            gd_mu = pred_team_sym - pred_opp_sym
            gd_consistent = np.abs(gd_aux - gd_mu) <= eps_gd
        else:
            gd_consistent = np.ones(len(pred_team_raw), dtype=bool)

        if aux_pred is not None and "total_goals_target" in aux_pred:
            total_aux = np.asarray(aux_pred["total_goals_target"], dtype=float)
            total_mu = pred_team_sym + pred_opp_sym
            total_consistent = np.abs(total_aux - total_mu) <= eps_total
        else:
            total_consistent = np.ones(len(pred_team_raw), dtype=bool)

        return sym_consistent & gd_consistent & total_consistent


    def build_pseudo_labelled_train(
        base_train_fe,
        test_fe,
        test_raw_df,
        pred_team_sym,
        pred_opp_sym,
        pseudo_mask,
    ):
        """Bangun extended train_fe dengan pseudo-labels masuk sebagai rows.

        Pseudo-label score di-round ke integer (model belajar di domain integer).
        Tambah kolom __is_pseudo_label untuk audit dan untuk weighting nanti.
        """
        base_train_fe = base_train_fe.copy()
        test_with_target = test_fe.copy()

        pseudo_team = np.rint(np.asarray(pred_team_sym, dtype=float)).clip(0, 25).astype(int)
        pseudo_opp = np.rint(np.asarray(pred_opp_sym, dtype=float)).clip(0, 25).astype(int)

        test_with_target["team_goals"] = pseudo_team
        test_with_target["opp_goals"] = pseudo_opp

        test_with_target = test_with_target.loc[pseudo_mask].reset_index(drop=True)

        # Restrict ke kolom intersection dengan base_train_fe
        common_cols = [c for c in base_train_fe.columns if c in test_with_target.columns]
        extended = pd.concat(
            [base_train_fe[common_cols], test_with_target[common_cols]],
            ignore_index=True,
        )

        extended["__is_pseudo_label"] = (
            [0] * len(base_train_fe) + [1] * len(test_with_target)
        )

        return extended


    def fit_segmented_models_with_pseudo_weight(
        train_df,
        feature_cols,
        pseudo_label_weight=PSEUDO_LABEL_WEIGHT,
        w_pseudo_priority=PSEUDO_LABEL_W_PRIORITY_WEIGHT,
    ):
        """Sama dengan fit_segmented_models tapi multiplier weight untuk
        rows pseudo-label, dengan bonus untuk W segment.

        Implementasinya: kita modify sample weight via override compute_sample_weight
        via temporary monkeypatch.
        """
        if "__is_pseudo_label" not in train_df.columns:
            return fit_segmented_models(train_df, feature_cols)

        is_pseudo = train_df["__is_pseudo_label"].astype(int).values == 1
        is_w = train_df["gender"].astype(str).str.upper().eq("W").values

        pseudo_mult = np.ones(len(train_df), dtype=float)
        pseudo_mult[is_pseudo] = pseudo_label_weight
        pseudo_mult[is_pseudo & is_w] = w_pseudo_priority

        # Temporarily monkeypatch compute_sample_weight untuk kalikan multiplier
        global compute_sample_weight
        original_compute = compute_sample_weight

        def compute_sample_weight_with_pseudo(df):
            base_w = original_compute(df)
            if "__is_pseudo_label" not in df.columns:
                return base_w
            local_pseudo = df["__is_pseudo_label"].astype(int).values == 1
            local_w_seg = df["gender"].astype(str).str.upper().eq("W").values
            mult = np.ones(len(df), dtype=float)
            mult[local_pseudo] = pseudo_label_weight
            mult[local_pseudo & local_w_seg] = w_pseudo_priority
            return base_w * mult

        try:
            compute_sample_weight = compute_sample_weight_with_pseudo
            store = fit_segmented_models(train_df, feature_cols)
        finally:
            compute_sample_weight = original_compute

        return store


    # ------------------------------------------------------------------
    # Eksekusi pseudo-label loop
    # ------------------------------------------------------------------

    if PSEUDO_LABEL_ITERATIONS > 0:
        pseudo_team_sym = test_pred_team_sym.copy()
        pseudo_opp_sym = test_pred_opp_sym.copy()
        pseudo_aux = test_aux_pred_raw

        for iteration in range(1, PSEUDO_LABEL_ITERATIONS + 1):
            pseudo_mask = select_confident_pseudo_labels(
                test_raw,
                test_pred_team_raw,
                test_pred_opp_raw,
                pseudo_team_sym,
                pseudo_opp_sym,
                pseudo_aux,
            )

            frac = float(pseudo_mask.mean()) if len(pseudo_mask) else 0.0
            print(f"\n=== Pseudo-label iteration {iteration} ===")
            print(f"Confident fraction: {frac:.3f}")

            if frac < PSEUDO_LABEL_MIN_FRACTION:
                print("Terlalu sedikit confident sample. Stop pseudo-label loop.")
                break
            if frac > PSEUDO_LABEL_MAX_FRACTION:
                # Kalau terlalu banyak yang dianggap confident, tighten threshold dulu
                print("Terlalu banyak confident; tighten threshold dulu sebelum train.")
                pseudo_mask = select_confident_pseudo_labels(
                    test_raw,
                    test_pred_team_raw,
                    test_pred_opp_raw,
                    pseudo_team_sym,
                    pseudo_opp_sym,
                    pseudo_aux,
                    eps_mu=PSEUDO_LABEL_CONF_EPS_MU * 0.7,
                    eps_gd=PSEUDO_LABEL_CONF_EPS_GD * 0.7,
                    eps_total=PSEUDO_LABEL_CONF_EPS_TOTAL * 0.7,
                )
                print(f"After tightening: {pseudo_mask.mean():.3f}")

            # Tambahkan pseudo-label ke train, refit
            extended_train = build_pseudo_labelled_train(
                final_model_fe,
                test_fe,
                test_raw,
                pseudo_team_sym,
                pseudo_opp_sym,
                pseudo_mask,
            )

            # Pastikan auxiliary target dihitung untuk pseudo-rows juga
            extended_train = add_auxiliary_training_targets(extended_train)

            print(f"Extended train rows: {len(extended_train):,} "
                  f"(original {len(final_model_fe):,} + pseudo {pseudo_mask.sum():,})")

            # Refit model
            pseudo_model_store = fit_segmented_models_with_pseudo_weight(
                extended_train,
                final_feature_cols,
            )

            # Predict ulang
            new_pred_team_raw, new_pred_opp_raw = predict_segmented_models(pseudo_model_store, test_fe)
            new_aux = predict_auxiliary_models(
                pseudo_model_store,
                test_fe,
                fallback_team_mu=new_pred_team_raw,
                fallback_opp_mu=new_pred_opp_raw,
            )

            new_team_sym, new_opp_sym = build_match_level_mu(
                test_raw.reset_index(drop=True),
                new_pred_team_raw,
                new_pred_opp_raw,
                source_policy="sym_avg",
            )

            # Update pointer untuk iterasi selanjutnya
            test_pred_team_raw = new_pred_team_raw
            test_pred_opp_raw = new_pred_opp_raw
            test_aux_pred_raw = new_aux
            pseudo_team_sym = new_team_sym
            pseudo_opp_sym = new_opp_sym
            pseudo_aux = new_aux
            final_model_store = pseudo_model_store

        # Update test_pred_team_sym untuk downstream (cell 27 segment postprocessor)
        test_pred_team_sym = pseudo_team_sym
        test_pred_opp_sym = pseudo_opp_sym

        # Re-apply postprocessor + re-build submission supaya overlay dan
        # downstream cells pakai prediksi pseudo-label.
        if USE_ADVANCED_POSTPROCESSOR:
            post_team, post_opp = apply_segment_postprocessors(
                test_raw.reset_index(drop=True),
                test_pred_team_raw,
                test_pred_opp_raw,
                segment_postprocessor,
                aux_raw=test_aux_pred_raw,
            )
        else:
            post_team, post_opp = apply_segment_postprocessors(
                test_raw.reset_index(drop=True),
                test_pred_team_raw,
                test_pred_opp_raw,
                {"segment_col": "gender", "global": best_decoder, "by_segment": {}},
                aux_raw=test_aux_pred_raw,
            )

        # Backup submission lama untuk audit, lalu overwrite dengan versi pseudo
        submission_pre_pseudo = submission.copy()
        submission = pd.DataFrame({
            ID_COL: test_raw[ID_COL].values,
            "team_goals": post_team.astype(int),
            "opp_goals": post_opp.astype(int),
        })

        submission_pseudo_path = OUT_DIR / "submission_awmae_pseudo_label.csv"
        submission.to_csv(submission_pseudo_path, index=False)

        n_changed = int(((submission["team_goals"].values != submission_pre_pseudo["team_goals"].values)
                         | (submission["opp_goals"].values != submission_pre_pseudo["opp_goals"].values)).sum())
        print(f"Pseudo-label submission saved ke: {submission_pseudo_path.resolve()}")
        print(f"Rows changed by pseudo-label: {n_changed}")
        print("Pseudo-label loop selesai. final_model_store dan submission sudah di-update.")
    else:
        print("Pseudo-label loop dimatikan (PSEUDO_LABEL_ITERATIONS=0).")

else:
    print("Pseudo-labelling dimatikan untuk ablation variant:", EXP18_ABLATION_VARIANT)
    submission_pre_pseudo = submission.copy()
    pseudo_mask = np.zeros(len(test_raw), dtype=bool)
    pseudo_audit = pd.DataFrame({ID_COL: test_raw[ID_COL].values, "pseudo_selected": pseudo_mask})
    pseudo_audit.to_csv(OUT_DIR / "pseudo_label_audit.csv", index=False)

# %% [markdown]
# ## 13c. Extreme blowout rule-based overlay
# 
# Untuk match yang masuk extreme bucket (powerhouse vs developing-underdog di qualifier kontinental), blend prediksi model dengan history P75 dari pasangan cluster sejenis di train. Gated by validation: hanya apply kalau improve AW-MAE di valid block.

# %%
# ============================================================
# 13c. Extreme blowout rule-based overlay
# ============================================================

if USE_EXTREME_OVERLAY:
    #
    # Model boosted tree biasanya konservatif di tail; mereka jarang predict
    # 13-0, 17-1 atau 21-0 walaupun match-up jelas mismatch ekstrem (powerhouse
    # vs developing-underdog di kualifikasi awal CAF/AFC/OFC).
    #
    # Strategi rule-based:
    #   1) Identifikasi match yang masuk extreme bucket berdasarkan:
    #      - cluster_team in {powerhouse, competitive} dan cluster_opp in
    #        {developing_underdog} (atau sebaliknya untuk perspektif sebaliknya)
    #      - is_extreme_mismatch_pair == 1
    #      - tier_gap_at_least_2 == 1
    #      - tournament termasuk qualifier / friendly / regional cup awal
    #      - Model sudah predict goal_diff searah, hanya kurang besar
    #   2) Bangun "history score table": untuk tiap pasangan (cluster_strong,
    #      cluster_weak, gender, tier_gap), ambil dari train_raw historical mean
    #      goals dan p90/p95 dari pemenang.
    #   3) Override prediksi: max(model_pred, blend(model, history_p75/p90)).
    #   4) Gate: hitung AW-MAE di validation block dengan vs tanpa overlay.
    #      Hanya apply di test kalau improvement.
    #
    # CATATAN PENTING: kita TIDAK pernah override match yang model predict draw
    # atau bahkan menang sebaliknya. Override hanya menyelaraskan magnitude
    # kemenangan kalau direction sudah konsisten.

    EXTREME_OVERLAY_MIN_HISTORY_MATCHES = 6
    EXTREME_OVERLAY_PERCENTILE = 0.80   # ambil P80 untuk margin kemenangan
    EXTREME_OVERLAY_BLEND_RATIO = 0.55  # 55% bobot history, 45% model
    EXTREME_OVERLAY_MIN_GOAL_DIFF_TO_TRIGGER = 1.5
    EXTREME_OVERLAY_TOURNAMENT_PATTERNS = [
        "qualification", "qualifier", "qualifying",
        "friendly", "asian cup", "afc", "caf", "afcon",
        "concacaf", "ofc", "cosafa", "gulf", "south asian",
    ]


    def _is_extreme_overlay_tournament(tournament_str):
        t = str(tournament_str).lower()
        return any(p in t for p in EXTREME_OVERLAY_TOURNAMENT_PATTERNS)


    def build_extreme_history_score_table(train_df, country_cluster_labels):
        """Per (gender, cluster_strong, cluster_weak), kumpulkan:
           - n match historis
           - mean goals strong, mean goals weak
           - P75 goals strong, P75 goals weak
           - max gd
        """
        df = train_df.copy()
        df["gender_norm"] = df["gender"].astype(str).str.upper()
        df = df.dropna(subset=["team_goals", "opp_goals"]).copy()
        df["team_goals"] = pd.to_numeric(df["team_goals"], errors="coerce").astype(int)
        df["opp_goals"] = pd.to_numeric(df["opp_goals"], errors="coerce").astype(int)

        df["cluster_team"] = [
            country_cluster_labels.get((g, str(t)), "default_unknown")
            for g, t in zip(df["gender_norm"], df["team"].astype(str))
        ]
        df["cluster_opp"] = [
            country_cluster_labels.get((g, str(o)), "default_unknown")
            for g, o in zip(df["gender_norm"], df["opponent"].astype(str))
        ]

        rows = []
        for (g, ct, co), grp in df.groupby(["gender_norm", "cluster_team", "cluster_opp"], dropna=False):
            if len(grp) < EXTREME_OVERLAY_MIN_HISTORY_MATCHES:
                continue
            rows.append({
                "gender": g,
                "cluster_team": ct,
                "cluster_opp": co,
                "n": int(len(grp)),
                "team_goals_mean": float(grp["team_goals"].mean()),
                "team_goals_p75": float(grp["team_goals"].quantile(0.75)),
                "team_goals_p90": float(grp["team_goals"].quantile(0.90)),
                "opp_goals_mean": float(grp["opp_goals"].mean()),
                "opp_goals_p25": float(grp["opp_goals"].quantile(0.25)),
                "opp_goals_p10": float(grp["opp_goals"].quantile(0.10)),
                "max_team_goals": float(grp["team_goals"].max()),
                "max_gd": float((grp["team_goals"] - grp["opp_goals"]).max()),
            })
        table = pd.DataFrame(rows)

        # Convert ke dict[(gender, cluster_team, cluster_opp)] -> row
        lookup = {(r["gender"], r["cluster_team"], r["cluster_opp"]): r for _, r in table.iterrows()}
        return table, lookup


    def apply_extreme_overlay(
        pred_team,
        pred_opp,
        feature_df,
        country_cluster_labels,
        history_lookup,
        overlay_gate_enabled=True,
    ):
        """Apply rule-based overlay ke pred_team/pred_opp untuk match yang masuk
        extreme bucket. feature_df harus punya kolom: gender, team, opponent,
        tournament, is_extreme_mismatch_pair, tier_gap_at_least_2, country_tier_team,
        country_tier_opp.

        Return (new_team, new_opp, overlay_mask) sebagai array integer.
        """
        pred_team = np.asarray(pred_team, dtype=float).copy()
        pred_opp = np.asarray(pred_opp, dtype=float).copy()
        n = len(pred_team)

        if not overlay_gate_enabled:
            return (
                np.rint(pred_team).clip(0, 25).astype(int),
                np.rint(pred_opp).clip(0, 25).astype(int),
                np.zeros(n, dtype=bool),
            )

        gender_arr = feature_df["gender"].astype(str).str.upper().values
        team_arr = feature_df["team"].astype(str).values
        opp_arr = feature_df["opponent"].astype(str).values
        tour_arr = feature_df["tournament"].astype(str).values

        cluster_team = np.array([
            country_cluster_labels.get((gender_arr[i], team_arr[i]), "default_unknown")
            for i in range(n)
        ])
        cluster_opp = np.array([
            country_cluster_labels.get((gender_arr[i], opp_arr[i]), "default_unknown")
            for i in range(n)
        ])

        is_mismatch = feature_df.get("is_extreme_mismatch_pair", pd.Series([0] * n)).astype(int).values == 1
        tier_gap2 = feature_df.get("tier_gap_at_least_2", pd.Series([0] * n)).astype(int).values == 1
        tier_t = feature_df.get("country_tier_team", pd.Series([4] * n)).astype(float).values
        tier_o = feature_df.get("country_tier_opp", pd.Series([4] * n)).astype(float).values

        is_overlay_tournament = np.array([_is_extreme_overlay_tournament(t) for t in tour_arr])

        model_gd = pred_team - pred_opp

        # Bucket A: team strong, opponent weak. Direction: team menang besar.
        A = (
            is_mismatch & tier_gap2 & is_overlay_tournament
            & (tier_t < tier_o)
            & (model_gd >= EXTREME_OVERLAY_MIN_GOAL_DIFF_TO_TRIGGER)
        )
        # Bucket B: team weak, opponent strong. Direction: opponent menang besar.
        B = (
            is_mismatch & tier_gap2 & is_overlay_tournament
            & (tier_t > tier_o)
            & (model_gd <= -EXTREME_OVERLAY_MIN_GOAL_DIFF_TO_TRIGGER)
        )

        new_team = pred_team.copy()
        new_opp = pred_opp.copy()
        overlay_mask = np.zeros(n, dtype=bool)

        for i in np.where(A)[0]:
            key = (gender_arr[i], cluster_team[i], cluster_opp[i])
            h = history_lookup.get(key)
            if h is None:
                continue
            target_team = h["team_goals_p75"]
            target_opp = h["opp_goals_p25"]
            new_team[i] = (1 - EXTREME_OVERLAY_BLEND_RATIO) * pred_team[i] + EXTREME_OVERLAY_BLEND_RATIO * target_team
            new_opp[i] = (1 - EXTREME_OVERLAY_BLEND_RATIO) * pred_opp[i] + EXTREME_OVERLAY_BLEND_RATIO * target_opp
            overlay_mask[i] = True

        for i in np.where(B)[0]:
            # Mirror perspective: cluster_team weak, cluster_opp strong. Lookup
            # untuk perspektif sebaliknya.
            key = (gender_arr[i], cluster_opp[i], cluster_team[i])
            h = history_lookup.get(key)
            if h is None:
                continue
            target_winning = h["team_goals_p75"]   # kemenangan opponent
            target_losing = h["opp_goals_p25"]    # kekalahan team
            new_team[i] = (1 - EXTREME_OVERLAY_BLEND_RATIO) * pred_team[i] + EXTREME_OVERLAY_BLEND_RATIO * target_losing
            new_opp[i] = (1 - EXTREME_OVERLAY_BLEND_RATIO) * pred_opp[i] + EXTREME_OVERLAY_BLEND_RATIO * target_winning
            overlay_mask[i] = True

        return (
            np.rint(new_team).clip(0, 25).astype(int),
            np.rint(new_opp).clip(0, 25).astype(int),
            overlay_mask,
        )


    # ------------------------------------------------------------------
    # Build history table dari train_raw + cluster labels final
    # ------------------------------------------------------------------

    extreme_history_table, extreme_history_lookup = build_extreme_history_score_table(
        train_raw, country_cluster_labels_final
    )

    print("Extreme history table built ✓")
    display(extreme_history_table.head(20))

    # ------------------------------------------------------------------
    # Validate overlay on validation block sebelum apply ke test
    # ------------------------------------------------------------------

    # Re-compute valid pred sym sudah dihitung sebelumnya. Cluster validasi sudah
    # di-attach di section 05b. Build history lookup yang sesuai (pakai train_hist).
    extreme_history_table_valid, extreme_history_lookup_valid = build_extreme_history_score_table(
        train_hist_raw, country_cluster_labels
    )

    valid_team_no_overlay = np.rint(np.asarray(valid_pred_team_sym)).clip(0, 25).astype(int)
    valid_opp_no_overlay = np.rint(np.asarray(valid_pred_opp_sym)).clip(0, 25).astype(int)

    # Untuk validasi kita perlu valid_fe sudah punya kolom country_tier dll. Cell
    # 05b sudah attach country_cluster ke valid_fe; country_tier dst sudah dari
    # add_country_features di make_feature_frames.
    valid_team_with_overlay, valid_opp_with_overlay, valid_overlay_mask = apply_extreme_overlay(
        valid_pred_team_sym,
        valid_pred_opp_sym,
        valid_fe,
        country_cluster_labels,
        extreme_history_lookup_valid,
    )

    awmae_no_overlay = awmae_score(
        valid_y_team, valid_y_opp,
        valid_team_no_overlay, valid_opp_no_overlay,
        weights=valid_weights,
    )
    awmae_with_overlay = awmae_score(
        valid_y_team, valid_y_opp,
        valid_team_with_overlay, valid_opp_with_overlay,
        weights=valid_weights,
    )

    print(f"\nValid AW-MAE no overlay  : {awmae_no_overlay:.6f}")
    print(f"Valid AW-MAE with overlay: {awmae_with_overlay:.6f}")
    print(f"Overlay rows in valid    : {int(valid_overlay_mask.sum())}/{len(valid_overlay_mask)}")

    EXTREME_OVERLAY_ENABLED = bool(awmae_with_overlay <= awmae_no_overlay)
    print(f"Extreme overlay enabled  : {EXTREME_OVERLAY_ENABLED}")

    # ------------------------------------------------------------------
    # Apply ke submission akhir
    # ------------------------------------------------------------------

    base_team_for_overlay = submission["team_goals"].values.astype(float).copy()
    base_opp_for_overlay = submission["opp_goals"].values.astype(float).copy()

    final_team_overlay, final_opp_overlay, test_overlay_mask = apply_extreme_overlay(
        base_team_for_overlay,
        base_opp_for_overlay,
        test_fe,
        country_cluster_labels_final,
        extreme_history_lookup,
        overlay_gate_enabled=EXTREME_OVERLAY_ENABLED,
    )

    submission_extreme = pd.DataFrame({
        ID_COL: test_raw[ID_COL].values,
        "team_goals": final_team_overlay,
        "opp_goals": final_opp_overlay,
    })

    # Mirror-consistency check: kalau overlay melanggar mirror, restore dari base
    sub_check_extreme = test_raw[[ID_COL, MATCH_COL]].copy().merge(submission_extreme, on=ID_COL, how="left")
    mirror_violations = []
    for match_id, g in sub_check_extreme.groupby(MATCH_COL, sort=False):
        if len(g) != 2:
            continue
        a, b = g.iloc[0], g.iloc[1]
        if int(a["team_goals"]) != int(b["opp_goals"]) or int(a["opp_goals"]) != int(b["team_goals"]):
            mirror_violations.append(match_id)

    if mirror_violations:
        # Re-symmetrize using row 1 as anchor for violations
        fix_mask = sub_check_extreme[MATCH_COL].isin(mirror_violations)
        print(f"Mirror violations after overlay: {len(mirror_violations)} matches → re-symmetrizing")

        fix_team, fix_opp = build_match_level_mu(
            test_raw.reset_index(drop=True),
            submission_extreme["team_goals"].values.astype(float),
            submission_extreme["opp_goals"].values.astype(float),
            source_policy="sym_avg",
        )
        submission_extreme["team_goals"] = np.rint(fix_team).clip(0, 25).astype(int)
        submission_extreme["opp_goals"] = np.rint(fix_opp).clip(0, 25).astype(int)

    submission_extreme_path = OUT_DIR / "submission_awmae_extreme_overlay.csv"
    submission_extreme.to_csv(submission_extreme_path, index=False)

    n_changed = int(((submission_extreme["team_goals"].values != submission["team_goals"].values)
                     | (submission_extreme["opp_goals"].values != submission["opp_goals"].values)).sum())

    print(f"\nExtreme overlay submission saved ke: {submission_extreme_path.resolve()}")
    print(f"Test rows triggered overlay : {int(test_overlay_mask.sum())}")
    print(f"Test rows actually changed  : {n_changed}")
    display(submission_extreme.head(10))

    # Distribusi score baru
    display(
        submission_extreme.groupby(["team_goals", "opp_goals"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(20)
    )

else:
    print("Extreme blowout overlay dimatikan untuk ablation variant:", EXP18_ABLATION_VARIANT)
    EXTREME_OVERLAY_ENABLED = False
    test_overlay_mask = np.zeros(len(test_raw), dtype=bool)
    submission_extreme = submission.copy()
    submission_extreme_path = OUT_DIR / "submission_awmae_extreme_overlay.csv"
    submission_extreme.to_csv(submission_extreme_path, index=False)
    print("Extreme overlay fallback submission saved ke:", submission_extreme_path.resolve())

# %% [markdown]
# ## 14. Sanity check submission
# 
# Bagian ini mengecek konsistensi mirror per match pada submission. Tidak semua kompetisi mewajibkan mirror sempurna, tapi untuk format 2-row-per-match, ini biasanya lebih aman.

# %%
# ============================================================
# 14. Sanity check mirror consistency
# ============================================================

sub_check = test_raw[[ID_COL, MATCH_COL, "team", "opponent"]].copy()
sub_check = sub_check.merge(submission, on=ID_COL, how="left")

bad_matches = []
for match_id, g in sub_check.groupby(MATCH_COL, sort=False):
    if len(g) != 2:
        continue
    a = g.iloc[0]
    b = g.iloc[1]
    cond1 = int(a["team_goals"]) == int(b["opp_goals"])
    cond2 = int(a["opp_goals"]) == int(b["team_goals"])
    if not (cond1 and cond2):
        bad_matches.append(match_id)

print("Jumlah match tidak mirror:", len(bad_matches))
if bad_matches[:5]:
    display(sub_check[sub_check[MATCH_COL].isin(bad_matches[:5])])

# Distribusi prediksi supaya kelihatan tidak collapse jadi semua 1 atau semua 2.
score_dist = (
    submission.groupby(["team_goals", "opp_goals"])
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
)
display(score_dist.head(20))

print("core EXP22 pipeline selesai, lanjut EXP12D output hardening...")

# %% [markdown]
# ## 15. EXP15 — Ordinal goal distribution
# 
# Eksperimen sebelumnya menunjukkan model sering collapse ke scoreline tengah seperti `1-1`, `1-2`, dan `2-1`.  
# Di EXP15, prediksi tidak langsung dipaksa jadi exact scoreline. Model tambahan mempelajari peluang bertingkat:
# 
# ```text
# P(goal >= 1), P(goal >= 2), ..., P(goal >= K)
# ```
# 
# Dari peluang bertingkat ini, notebook membentuk distribusi goal marginal:
# 
# ```text
# P(goal = 0), P(goal = 1), ..., P(goal = K/tail)
# ```
# 
# Lalu decoder memilih scoreline dengan expected AW-MAE paling rendah. Konfigurasinya tetap dipilih dari validation, bukan dari `ground_truth_bersih.csv`.

# %%

# ============================================================
# 15. EXP15 setup
# ============================================================

EXP15_ENABLED = True

# Max ordinal goal. Semakin besar, semakin banyak binary classifier yang dilatih.
# Untuk iterasi awal, 10 cukup ringan. Kalau hasilnya menjanjikan, coba 12.
EXP15_MAX_GOAL = 8 if FAST_MODE else 10

# Kandidat scoreline maksimum saat decoder memilih prediksi final.
EXP15_MAX_PRED_GOAL = 10 if FAST_MODE else 12

# Runtime control.
EXP15_FAST_MODE = FAST_MODE
EXP15_USE_GENDER_SEGMENT = True
EXP15_MIN_SEGMENT_ROWS = 1800

# Kalau True, GT hanya dipakai di akhir untuk audit lokal.
# Tidak dipakai untuk training, tuning, ataupun pemilihan config.
RUN_EXP15_GT_AUDIT = bool(globals().get("RUN_GT_AUDIT", False))

EXP15_SUBDIR = OUT_DIR / "exp15_ordinal"
EXP15_SUBDIR.mkdir(parents=True, exist_ok=True)

print("EXP15_SUBDIR:", EXP15_SUBDIR.resolve())
print("EXP15_MAX_GOAL:", EXP15_MAX_GOAL)
print("EXP15_MAX_PRED_GOAL:", EXP15_MAX_PRED_GOAL)

# %%

# ============================================================
# 15A. Ordinal model helpers
# ============================================================

try:
    from lightgbm import LGBMClassifier
    EXP15_LGBM_CLASSIFIER_AVAILABLE = True
except Exception as e:
    EXP15_LGBM_CLASSIFIER_AVAILABLE = False
    print("LightGBM classifier tidak tersedia:", repr(e))

from sklearn.ensemble import HistGradientBoostingClassifier


def exp15_make_classifier(threshold):
    """Classifier binary untuk target goal >= threshold."""
    if EXP15_LGBM_CLASSIFIER_AVAILABLE:
        return LGBMClassifier(
            objective="binary",
            n_estimators=250 if EXP15_FAST_MODE else 520,
            learning_rate=0.050 if EXP15_FAST_MODE else 0.030,
            num_leaves=40,
            max_depth=-1,
            min_child_samples=35,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.05,
            reg_lambda=1.0,
            random_state=SEED + int(threshold),
            n_jobs=-1,
            verbose=-1,
        )

    return HistGradientBoostingClassifier(
        learning_rate=0.055 if EXP15_FAST_MODE else 0.035,
        max_iter=180 if EXP15_FAST_MODE else 360,
        max_leaf_nodes=31,
        l2_regularization=0.05,
        random_state=SEED + int(threshold),
    )


def exp15_build_classifier_pipeline(train_df, feature_cols, threshold):
    num_cols, cat_cols = split_feature_types(train_df, feature_cols)
    preprocess = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            ("cat", make_pipeline(
                SimpleImputer(strategy="most_frequent"),
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            ), cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    return make_pipeline(preprocess, exp15_make_classifier(threshold))


def exp15_fit_ordinal_models(train_df, feature_cols, max_goal=10, label_prefix="global"):
    """Fit 2 x max_goal model:
    - team_goals >= k
    - opp_goals >= k

    Kalau target threshold hanya punya satu kelas, model tidak dilatih dan prediksi memakai prior.
    """
    models = {"team_goals": {}, "opp_goals": {}, "priors": {"team_goals": {}, "opp_goals": {}}}
    weights = compute_sample_weight(train_df)

    for target in ["team_goals", "opp_goals"]:
        y_raw = train_df[target].astype(float).values
        for k in range(1, max_goal + 1):
            y_bin = (y_raw >= k).astype(int)
            prior = float(np.mean(y_bin))
            models["priors"][target][k] = prior

            if y_bin.min() == y_bin.max():
                models[target][k] = None
                print(f"[EXP15][{label_prefix}] {target} >= {k}: single class, prior={prior:.4f}")
                continue

            clf = exp15_build_classifier_pipeline(train_df, feature_cols, threshold=k)
            clf.fit(train_df[feature_cols], y_bin, **{"lgbmclassifier__sample_weight": weights} if EXP15_LGBM_CLASSIFIER_AVAILABLE else {"histgradientboostingclassifier__sample_weight": weights})
            models[target][k] = clf
            if k in [1, 3, 5, max_goal]:
                print(f"[EXP15][{label_prefix}] fitted {target} >= {k} | prior={prior:.4f}")

    models["max_goal"] = int(max_goal)
    models["label_prefix"] = label_prefix
    return models


def exp15_predict_threshold_probs(models, pred_df, feature_cols):
    max_goal = int(models["max_goal"])
    out = {}

    for target in ["team_goals", "opp_goals"]:
        probs = np.zeros((len(pred_df), max_goal), dtype=float)

        for k in range(1, max_goal + 1):
            model = models[target].get(k)
            if model is None:
                probs[:, k - 1] = float(models["priors"][target][k])
                continue

            pred = model.predict_proba(pred_df[feature_cols])
            if pred.shape[1] == 1:
                # Defensive fallback kalau estimator melihat satu kelas.
                cls = getattr(model[-1], "classes_", np.array([0]))
                probs[:, k - 1] = 1.0 if int(cls[0]) == 1 else 0.0
            else:
                # Ambil probabilitas kelas 1.
                probs[:, k - 1] = pred[:, 1]

        # enforce monotonic: P(>=1) >= P(>=2) >= ...
        probs = np.clip(probs, 0.0, 1.0)
        probs = np.minimum.accumulate(probs, axis=1)
        out[target] = probs

    return out


def exp15_threshold_to_pmf(threshold_probs):
    """Convert P(goal >= k) ke PMF P(goal = 0..K), dengan bin K sebagai tail."""
    threshold_probs = np.asarray(threshold_probs, dtype=float)
    n, max_goal = threshold_probs.shape

    pmf = np.zeros((n, max_goal + 1), dtype=float)
    pmf[:, 0] = 1.0 - threshold_probs[:, 0]

    for k in range(1, max_goal):
        pmf[:, k] = threshold_probs[:, k - 1] - threshold_probs[:, k]

    pmf[:, max_goal] = threshold_probs[:, max_goal - 1]
    pmf = np.clip(pmf, 0.0, 1.0)
    pmf = pmf / np.maximum(pmf.sum(axis=1, keepdims=True), 1e-12)
    return pmf


def exp15_predict_pmf(models, pred_df, feature_cols):
    probs = exp15_predict_threshold_probs(models, pred_df, feature_cols)
    return {
        "team_goals": exp15_threshold_to_pmf(probs["team_goals"]),
        "opp_goals": exp15_threshold_to_pmf(probs["opp_goals"]),
    }


def exp15_expected_goal_from_pmf(pmf):
    goals = np.arange(pmf.shape[1], dtype=float)
    return pmf @ goals


def exp15_symmetrize_pmf(df, pmf_team, pmf_opp, source_policy="sym_avg", anchor_mu_team=None, anchor_mu_opp=None):
    """Mirror-aware PMF source policy.

    Untuk row i dan mirror row j:
    - team PMF row i sebanding dengan opp PMF row j
    - opp PMF row i sebanding dengan team PMF row j
    """
    df_temp = df[[ID_COL, MATCH_COL]].copy()
    if "is_home" in df.columns:
        df_temp["is_home"] = pd.to_numeric(df["is_home"], errors="coerce").fillna(0).astype(int)
    else:
        df_temp["is_home"] = 0

    pmf_team = np.asarray(pmf_team, dtype=float)
    pmf_opp = np.asarray(pmf_opp, dtype=float)

    out_team = pmf_team.copy()
    out_opp = pmf_opp.copy()

    if anchor_mu_team is None:
        anchor_mu_team = exp15_expected_goal_from_pmf(pmf_team)
    if anchor_mu_opp is None:
        anchor_mu_opp = exp15_expected_goal_from_pmf(pmf_opp)

    for match_id, idxs in df_temp.groupby(MATCH_COL, sort=False).groups.items():
        idxs = list(idxs)
        if len(idxs) != 2:
            continue

        i, j = idxs[0], idxs[1]

        # sym average
        sym_i_team = 0.5 * (pmf_team[i] + pmf_opp[j])
        sym_i_opp = 0.5 * (pmf_opp[i] + pmf_team[j])

        first_i_team = pmf_team[i]
        first_i_opp = pmf_opp[i]

        second_i_team = pmf_opp[j]
        second_i_opp = pmf_team[j]

        if source_policy == "raw":
            use_i_team, use_i_opp = first_i_team, first_i_opp
        elif source_policy == "first_raw":
            use_i_team, use_i_opp = first_i_team, first_i_opp
        elif source_policy == "second_raw":
            use_i_team, use_i_opp = second_i_team, second_i_opp
        elif source_policy == "home_anchor":
            if int(df_temp.at[i, "is_home"]) == 1 and int(df_temp.at[j, "is_home"]) == 0:
                use_i_team, use_i_opp = first_i_team, first_i_opp
            elif int(df_temp.at[j, "is_home"]) == 1 and int(df_temp.at[i, "is_home"]) == 0:
                use_i_team, use_i_opp = second_i_team, second_i_opp
            else:
                use_i_team, use_i_opp = sym_i_team, sym_i_opp
        elif source_policy in ["higher_total", "lower_total"]:
            total_first = float(anchor_mu_team[i] + anchor_mu_opp[i])
            total_second = float(anchor_mu_opp[j] + anchor_mu_team[j])
            take_first = total_first >= total_second if source_policy == "higher_total" else total_first <= total_second
            if take_first:
                use_i_team, use_i_opp = first_i_team, first_i_opp
            else:
                use_i_team, use_i_opp = second_i_team, second_i_opp
        else:
            use_i_team, use_i_opp = sym_i_team, sym_i_opp

        use_i_team = np.clip(use_i_team, 0, 1)
        use_i_opp = np.clip(use_i_opp, 0, 1)
        use_i_team = use_i_team / np.maximum(use_i_team.sum(), 1e-12)
        use_i_opp = use_i_opp / np.maximum(use_i_opp.sum(), 1e-12)

        out_team[i] = use_i_team
        out_opp[i] = use_i_opp
        out_team[j] = use_i_opp
        out_opp[j] = use_i_team

    return out_team, out_opp


def exp15_fit_ordinal_store(train_df, feature_cols, max_goal=10, segment_col="gender"):
    store = {
        "global": exp15_fit_ordinal_models(train_df, feature_cols, max_goal=max_goal, label_prefix="global"),
        "by_segment": {},
        "segment_col": segment_col,
        "feature_cols": list(feature_cols),
        "max_goal": int(max_goal),
    }

    if EXP15_USE_GENDER_SEGMENT and segment_col in train_df.columns:
        for seg, g in train_df.groupby(segment_col):
            if len(g) < EXP15_MIN_SEGMENT_ROWS:
                print(f"[EXP15] skip segment {seg}: n={len(g)}")
                continue
            print("=" * 90)
            print(f"[EXP15] Fit ordinal segment {segment_col}={seg} | n={len(g)}")
            print("=" * 90)
            store["by_segment"][str(seg)] = exp15_fit_ordinal_models(
                g.reset_index(drop=True),
                feature_cols,
                max_goal=max_goal,
                label_prefix=f"{segment_col}={seg}",
            )

    return store


def exp15_predict_ordinal_store(store, pred_df, feature_cols):
    global_pmf = exp15_predict_pmf(store["global"], pred_df, feature_cols)
    out_team = global_pmf["team_goals"].copy()
    out_opp = global_pmf["opp_goals"].copy()

    segment_col = store.get("segment_col", "gender")
    if EXP15_USE_GENDER_SEGMENT and segment_col in pred_df.columns:
        for seg, model in store.get("by_segment", {}).items():
            mask = pred_df[segment_col].astype(str).values == str(seg)
            if not np.any(mask):
                continue
            seg_pmf = exp15_predict_pmf(model, pred_df.loc[mask].reset_index(drop=True), feature_cols)
            out_team[mask] = seg_pmf["team_goals"]
            out_opp[mask] = seg_pmf["opp_goals"]

    return {"team_goals": out_team, "opp_goals": out_opp}

# %%

# ============================================================
# 15B. Ordinal distribution MBR decoder
# ============================================================

_EXP15_LOSS_CACHE = {}

def exp15_make_loss_matrix(max_true=10, max_pred=12):
    key = (int(max_true), int(max_pred))
    if key in _EXP15_LOSS_CACHE:
        return _EXP15_LOSS_CACHE[key]

    true_pairs = [(a, b) for a in range(max_true + 1) for b in range(max_true + 1)]
    pred_pairs = [(a, b) for a in range(max_pred + 1) for b in range(max_pred + 1)]

    loss = np.zeros((len(pred_pairs), len(true_pairs)), dtype=float)
    yta = np.array([p[0] for p in true_pairs])
    ytb = np.array([p[1] for p in true_pairs])

    for i, (pa, pb) in enumerate(pred_pairs):
        loss[i] = official_match_loss(
            yta,
            ytb,
            np.full_like(yta, pa),
            np.full_like(ytb, pb),
        )

    result = (np.asarray(pred_pairs, dtype=int), np.asarray(true_pairs, dtype=int), loss)
    _EXP15_LOSS_CACHE[key] = result
    return result


def exp15_poisson_pmf_for_blend(mu, max_goal, scale=1.0, offset=0.0):
    mu = calibrate_mu(mu, scale=scale, offset=offset)
    return poisson_pmf_matrix(mu, max_goal=max_goal)


def exp15_decode_distribution(
    pmf_team_ord,
    pmf_opp_ord,
    mu_team,
    mu_opp,
    config,
    aux_pred=None,
    batch_size=9000,
):
    """Decode scoreline dari distribusi ordinal yang diblend dengan Poisson baseline."""
    max_true = int(config.get("max_true", pmf_team_ord.shape[1] - 1))
    max_pred = int(config.get("max_pred", EXP15_MAX_PRED_GOAL))
    blend_ordinal = float(config.get("blend_ordinal", 0.75))
    scale = float(config.get("scale", 1.0))
    offset = float(config.get("offset", 0.0))

    # Pastikan dimensi sesuai max_true.
    pmf_team_ord = np.asarray(pmf_team_ord, dtype=float)[:, : max_true + 1]
    pmf_opp_ord = np.asarray(pmf_opp_ord, dtype=float)[:, : max_true + 1]
    pmf_team_ord = pmf_team_ord / np.maximum(pmf_team_ord.sum(axis=1, keepdims=True), 1e-12)
    pmf_opp_ord = pmf_opp_ord / np.maximum(pmf_opp_ord.sum(axis=1, keepdims=True), 1e-12)

    pmf_team_poi = exp15_poisson_pmf_for_blend(mu_team, max_goal=max_true, scale=scale, offset=offset)
    pmf_opp_poi = exp15_poisson_pmf_for_blend(mu_opp, max_goal=max_true, scale=scale, offset=offset)

    pmf_team = blend_ordinal * pmf_team_ord + (1.0 - blend_ordinal) * pmf_team_poi
    pmf_opp = blend_ordinal * pmf_opp_ord + (1.0 - blend_ordinal) * pmf_opp_poi
    pmf_team = pmf_team / np.maximum(pmf_team.sum(axis=1, keepdims=True), 1e-12)
    pmf_opp = pmf_opp / np.maximum(pmf_opp.sum(axis=1, keepdims=True), 1e-12)

    pred_pairs, true_pairs, loss_matrix = exp15_make_loss_matrix(max_true=max_true, max_pred=max_pred)
    pair_team = pred_pairs[:, 0]
    pair_opp = pred_pairs[:, 1]
    pair_total = pair_team + pair_opp
    pair_gd = pair_team - pair_opp
    pair_outcome = np.sign(pair_gd)

    out_team = np.zeros(len(pmf_team), dtype=int)
    out_opp = np.zeros(len(pmf_team), dtype=int)

    lambda_mu = float(config.get("lambda_mu", 0.0))
    lambda_gd = float(config.get("lambda_gd", 0.0))
    lambda_total = float(config.get("lambda_total", 0.0))
    lambda_outcome = float(config.get("lambda_outcome", 0.0))

    mu_total = np.asarray(mu_team, dtype=float) + np.asarray(mu_opp, dtype=float)
    mu_gd = np.asarray(mu_team, dtype=float) - np.asarray(mu_opp, dtype=float)

    if aux_pred is not None:
        aux_total = np.asarray(aux_pred.get("total_goals_target", mu_total), dtype=float)
        aux_gd = np.asarray(aux_pred.get("goal_diff_target", mu_gd), dtype=float)
        aux_outcome = np.sign(np.asarray(aux_pred.get("outcome_value_target", mu_gd), dtype=float))
    else:
        aux_total = mu_total
        aux_gd = mu_gd
        aux_outcome = np.sign(mu_gd)

    for start in range(0, len(pmf_team), batch_size):
        end = min(start + batch_size, len(pmf_team))
        true_probs = np.einsum(
            "bi,bj->bij",
            pmf_team[start:end],
            pmf_opp[start:end],
        ).reshape(end - start, -1)

        exp_loss = true_probs @ loss_matrix.T

        if lambda_mu > 0:
            exp_loss += lambda_mu * (
                np.abs(pair_total[None, :] - mu_total[start:end, None]) * 0.5
                + np.abs(pair_gd[None, :] - mu_gd[start:end, None]) * 0.25
            )

        if lambda_total > 0:
            exp_loss += lambda_total * np.abs(pair_total[None, :] - aux_total[start:end, None])
        if lambda_gd > 0:
            exp_loss += lambda_gd * np.abs(pair_gd[None, :] - aux_gd[start:end, None])
        if lambda_outcome > 0:
            exp_loss += lambda_outcome * (pair_outcome[None, :] != aux_outcome[start:end, None]).astype(float)

        best = np.argmin(exp_loss, axis=1)
        chosen = pred_pairs[best]
        out_team[start:end] = chosen[:, 0]
        out_opp[start:end] = chosen[:, 1]

    return out_team, out_opp


def exp15_apply_config(df, pmf_raw, mu_team_raw, mu_opp_raw, config, aux_raw=None):
    source_policy = config.get("source_policy", "sym_avg")

    # Apply source policy ke mu dan aux agar konsisten dengan baseline.
    mu_team, mu_opp = build_match_level_mu(
        df,
        mu_team_raw,
        mu_opp_raw,
        source_policy=source_policy,
    )
    aux = transform_aux_to_match_source(df, aux_raw, source_policy=source_policy) if aux_raw is not None else None

    # Apply source policy ke ordinal PMF.
    pmf_team, pmf_opp = exp15_symmetrize_pmf(
        df,
        pmf_raw["team_goals"],
        pmf_raw["opp_goals"],
        source_policy=source_policy,
        anchor_mu_team=mu_team_raw,
        anchor_mu_opp=mu_opp_raw,
    )

    return exp15_decode_distribution(
        pmf_team,
        pmf_opp,
        mu_team,
        mu_opp,
        config,
        aux_pred=aux,
    )


def exp15_decoder_grid(compact=True):
    if compact:
        source_policies = ["sym_avg", "raw", "higher_total"]
        blend_grid = [0.45, 0.65, 0.85]
        scale_grid = [0.95, 1.00, 1.05]
        offset_grid = [0.00, 0.05]
        aux_grid = [
            (0.00, 0.00, 0.00, 0.00),
            (0.03, 0.02, 0.08, 0.00),
            (0.02, 0.02, 0.06, 0.05),
        ]
    else:
        source_policies = ["sym_avg", "raw", "first_raw", "second_raw", "higher_total"]
        blend_grid = [0.35, 0.50, 0.65, 0.80, 0.95]
        scale_grid = [0.92, 0.97, 1.00, 1.05, 1.10]
        offset_grid = [0.00, 0.05, 0.10]
        aux_grid = [
            (0.00, 0.00, 0.00, 0.00),
            (0.03, 0.02, 0.08, 0.00),
            (0.04, 0.03, 0.10, 0.00),
            (0.02, 0.02, 0.06, 0.05),
        ]

    for source_policy in source_policies:
        for blend_ordinal in blend_grid:
            for scale in scale_grid:
                for offset in offset_grid:
                    for lambda_gd, lambda_total, lambda_outcome, lambda_mu in aux_grid:
                        yield {
                            "decoder": "ordinal_mbr",
                            "source_policy": source_policy,
                            "blend_ordinal": blend_ordinal,
                            "scale": scale,
                            "offset": offset,
                            "lambda_gd": lambda_gd,
                            "lambda_total": lambda_total,
                            "lambda_outcome": lambda_outcome,
                            "lambda_mu": lambda_mu,
                            "max_true": EXP15_MAX_GOAL,
                            "max_pred": EXP15_MAX_PRED_GOAL,
                        }


def exp15_tune_subset(df, y_team, y_opp, weights, pmf_raw, mu_team_raw, mu_opp_raw, aux_raw=None, compact=True, label="global"):
    rows = []
    configs = list(exp15_decoder_grid(compact=compact))
    print(f"[EXP15] Tuning {label}: {len(configs)} configs")

    for idx, cfg in enumerate(configs, start=1):
        pt, po = exp15_apply_config(df, pmf_raw, mu_team_raw, mu_opp_raw, cfg, aux_raw=aux_raw)
        score = awmae_score(y_team, y_opp, pt, po, weights=weights)
        rows.append({**cfg, "awmae": score, "label": label})
        if idx % 25 == 0 or idx == len(configs):
            print(f"[EXP15] {label}: {idx}/{len(configs)} | current={score:.5f}")

    report = pd.DataFrame(rows).sort_values("awmae").reset_index(drop=True)
    return report


def exp15_fit_segment_postprocessor(df, y_team, y_opp, weights, pmf_raw, mu_team_raw, mu_opp_raw, aux_raw=None):
    df = df.reset_index(drop=True).copy()

    global_report = exp15_tune_subset(
        df, y_team, y_opp, weights,
        pmf_raw, mu_team_raw, mu_opp_raw,
        aux_raw=aux_raw,
        compact=True,
        label="global",
    )
    global_cfg = global_report.iloc[0].to_dict()

    by_segment = {}
    reports = [global_report]

    if EXP15_USE_GENDER_SEGMENT and "gender" in df.columns:
        for seg, idx in df.groupby("gender").groups.items():
            idx = np.asarray(list(idx), dtype=int)
            if len(idx) < 300:
                continue

            seg_pmf = {
                "team_goals": pmf_raw["team_goals"][idx],
                "opp_goals": pmf_raw["opp_goals"][idx],
            }
            seg_aux = {k: np.asarray(v)[idx] for k, v in aux_raw.items()} if aux_raw is not None else None

            seg_report = exp15_tune_subset(
                df.iloc[idx].reset_index(drop=True),
                np.asarray(y_team)[idx],
                np.asarray(y_opp)[idx],
                np.asarray(weights)[idx],
                seg_pmf,
                np.asarray(mu_team_raw)[idx],
                np.asarray(mu_opp_raw)[idx],
                aux_raw=seg_aux,
                compact=True,
                label=f"gender={seg}",
            )

            best_seg = seg_report.iloc[0].to_dict()
            # Segment config hanya dipakai kalau memang mengalahkan global pada subset tersebut.
            pt_g, po_g = exp15_apply_config(
                df.iloc[idx].reset_index(drop=True),
                seg_pmf,
                np.asarray(mu_team_raw)[idx],
                np.asarray(mu_opp_raw)[idx],
                global_cfg,
                aux_raw=seg_aux,
            )
            global_on_seg = awmae_score(
                np.asarray(y_team)[idx],
                np.asarray(y_opp)[idx],
                pt_g,
                po_g,
                weights=np.asarray(weights)[idx],
            )

            if float(best_seg["awmae"]) <= float(global_on_seg) - 1e-6:
                by_segment[str(seg)] = best_seg
                print(f"[EXP15] pakai config khusus gender={seg}: {best_seg['awmae']:.5f} < {global_on_seg:.5f}")
            else:
                print(f"[EXP15] gender={seg} fallback global: {global_on_seg:.5f} <= {best_seg['awmae']:.5f}")

            reports.append(seg_report)

    post = {
        "global": global_cfg,
        "by_segment": by_segment,
        "segment_col": "gender",
        "enabled": True,
    }

    return post, pd.concat(reports, ignore_index=True)


def exp15_apply_segment_postprocessor(df, pmf_raw, mu_team_raw, mu_opp_raw, post, aux_raw=None):
    df = df.reset_index(drop=True).copy()
    pt, po = exp15_apply_config(df, pmf_raw, mu_team_raw, mu_opp_raw, post["global"], aux_raw=aux_raw)

    segment_col = post.get("segment_col", "gender")
    for seg, cfg in post.get("by_segment", {}).items():
        if segment_col not in df.columns:
            continue
        mask = df[segment_col].astype(str).values == str(seg)
        if not np.any(mask):
            continue

        sub_pmf = {
            "team_goals": pmf_raw["team_goals"][mask],
            "opp_goals": pmf_raw["opp_goals"][mask],
        }
        sub_aux = {k: np.asarray(v)[mask] for k, v in aux_raw.items()} if aux_raw is not None else None

        sub_pt, sub_po = exp15_apply_config(
            df.loc[mask].reset_index(drop=True),
            sub_pmf,
            np.asarray(mu_team_raw)[mask],
            np.asarray(mu_opp_raw)[mask],
            cfg,
            aux_raw=sub_aux,
        )
        pt[mask] = sub_pt
        po[mask] = sub_po

    return pt.astype(int), po.astype(int)

# %%

# ============================================================
# 15C. Fit ordinal models on validation block and tune EXP15 decoder
# ============================================================

if EXP15_ENABLED:
    print("=" * 90)
    print("[EXP15] Fit ordinal models for validation block")
    print("=" * 90)

    exp15_valid_store = exp15_fit_ordinal_store(
        train_model_fe.reset_index(drop=True),
        feature_cols,
        max_goal=EXP15_MAX_GOAL,
        segment_col="gender",
    )

    exp15_valid_pmf_raw = exp15_predict_ordinal_store(
        exp15_valid_store,
        valid_fe.reset_index(drop=True),
        feature_cols,
    )

    # Diagnostik distribusi marginal dari ordinal model sebelum decoder.
    exp15_valid_diag = pd.DataFrame({
        "true_team_goals": valid_y_team,
        "true_opp_goals": valid_y_opp,
        "ord_mu_team": exp15_expected_goal_from_pmf(exp15_valid_pmf_raw["team_goals"]),
        "ord_mu_opp": exp15_expected_goal_from_pmf(exp15_valid_pmf_raw["opp_goals"]),
        "base_mu_team": valid_pred_team_raw,
        "base_mu_opp": valid_pred_opp_raw,
        "gender": valid_raw["gender"].values if "gender" in valid_raw.columns else "__NA__",
    })
    display(exp15_valid_diag[["true_team_goals", "true_opp_goals", "ord_mu_team", "ord_mu_opp", "base_mu_team", "base_mu_opp"]].describe())

    exp15_postprocessor, exp15_report = exp15_fit_segment_postprocessor(
        valid_raw.reset_index(drop=True),
        valid_y_team,
        valid_y_opp,
        valid_weights,
        exp15_valid_pmf_raw,
        valid_pred_team_raw,
        valid_pred_opp_raw,
        aux_raw=valid_aux_pred_raw,
    )

    exp15_valid_team, exp15_valid_opp = exp15_apply_segment_postprocessor(
        valid_raw.reset_index(drop=True),
        exp15_valid_pmf_raw,
        valid_pred_team_raw,
        valid_pred_opp_raw,
        exp15_postprocessor,
        aux_raw=valid_aux_pred_raw,
    )

    exp15_valid_awmae = awmae_score(
        valid_y_team,
        valid_y_opp,
        exp15_valid_team,
        exp15_valid_opp,
        weights=valid_weights,
    )

    print("=" * 90)
    print("EXP15 validation AW-MAE:", exp15_valid_awmae)
    print("Baseline advanced validation AW-MAE:", valid_awmae_final)
    print("EXP15 selected global:", exp15_postprocessor["global"])
    print("EXP15 segment configs:", exp15_postprocessor.get("by_segment", {}))
    print("=" * 90)

    exp15_report.to_csv(EXP15_SUBDIR / "exp15_decoder_validation_report.csv", index=False)
    with open(EXP15_SUBDIR / "exp15_postprocessor.json", "w", encoding="utf-8") as f:
        json.dump(json_safe(exp15_postprocessor), f, ensure_ascii=False, indent=2)

    exp15_valid_pred_df = valid_raw[[ID_COL, MATCH_COL, "date", "gender", "team", "opponent", "tournament", "team_goals", "opp_goals"]].copy()
    exp15_valid_pred_df["pred_team_baseline"] = valid_pred_team_final
    exp15_valid_pred_df["pred_opp_baseline"] = valid_pred_opp_final
    exp15_valid_pred_df["pred_team_exp15"] = exp15_valid_team
    exp15_valid_pred_df["pred_opp_exp15"] = exp15_valid_opp
    exp15_valid_pred_df["ord_mu_team"] = exp15_valid_diag["ord_mu_team"].values
    exp15_valid_pred_df["ord_mu_opp"] = exp15_valid_diag["ord_mu_opp"].values
    exp15_valid_pred_df["weight"] = valid_weights
    exp15_valid_pred_df.to_csv(EXP15_SUBDIR / "exp15_validation_predictions.csv", index=False)

    display(exp15_report.groupby("label").head(5).reset_index(drop=True))
    display(exp15_valid_pred_df.head(10))
else:
    print("EXP15 dilewati karena EXP15_ENABLED=False")

# %%

# ============================================================
# 15D. Fit final ordinal models and predict test
# ============================================================

if EXP15_ENABLED:
    print("=" * 90)
    print("[EXP15] Fit final ordinal models on final_model_fe")
    print("=" * 90)

    exp15_final_store = exp15_fit_ordinal_store(
        final_model_fe.reset_index(drop=True),
        final_feature_cols,
        max_goal=EXP15_MAX_GOAL,
        segment_col="gender",
    )

    exp15_test_pmf_raw = exp15_predict_ordinal_store(
        exp15_final_store,
        test_fe.reset_index(drop=True),
        final_feature_cols,
    )

    exp15_test_team, exp15_test_opp = exp15_apply_segment_postprocessor(
        test_raw.reset_index(drop=True),
        exp15_test_pmf_raw,
        test_pred_team_raw,
        test_pred_opp_raw,
        exp15_postprocessor,
        aux_raw=test_aux_pred_raw,
    )

    submission_exp15 = pd.DataFrame({
        ID_COL: test_raw[ID_COL].values,
        "team_goals": exp15_test_team.astype(int),
        "opp_goals": exp15_test_opp.astype(int),
    })

    assert len(submission_exp15) == len(test_raw)
    assert submission_exp15[ID_COL].equals(test_raw[ID_COL].reset_index(drop=True))
    assert submission_exp15["team_goals"].between(0, EXP15_MAX_PRED_GOAL).all()
    assert submission_exp15["opp_goals"].between(0, EXP15_MAX_PRED_GOAL).all()

    exp15_submission_path = EXP15_SUBDIR / "submission_awmae_exp15_ordinal.csv"
    submission_exp15.to_csv(exp15_submission_path, index=False)

    exp15_test_audit = test_raw[[ID_COL, MATCH_COL, "date", "gender", "team", "opponent", "tournament"]].copy()
    exp15_test_audit["pred_team_baseline"] = submission["team_goals"].values
    exp15_test_audit["pred_opp_baseline"] = submission["opp_goals"].values
    exp15_test_audit["pred_team_exp15"] = submission_exp15["team_goals"].values
    exp15_test_audit["pred_opp_exp15"] = submission_exp15["opp_goals"].values
    exp15_test_audit["ord_mu_team"] = exp15_expected_goal_from_pmf(exp15_test_pmf_raw["team_goals"])
    exp15_test_audit["ord_mu_opp"] = exp15_expected_goal_from_pmf(exp15_test_pmf_raw["opp_goals"])
    exp15_test_audit["raw_mu_team"] = test_pred_team_raw
    exp15_test_audit["raw_mu_opp"] = test_pred_opp_raw
    exp15_test_audit.to_csv(EXP15_SUBDIR / "test_prediction_audit_exp15.csv", index=False)

    print("EXP15 submission saved to:", exp15_submission_path.resolve())
    display(submission_exp15.head(20))
    display(submission_exp15[["team_goals", "opp_goals"]].describe())

    exp15_score_dist = (
        submission_exp15.groupby(["team_goals", "opp_goals"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    display(exp15_score_dist.head(20))
else:
    print("EXP15 final prediction dilewati.")

# %%

# ============================================================
# 15E. Optional GT audit untuk diagnosis lokal
# ============================================================

def exp15_find_optional_file(filename):
    for d in CANDIDATE_DIRS:
        p = d / filename
        if p.exists():
            return p
    return None


def exp15_make_scoreline_distribution_compare(gt_df, pred_df, prefix="exp15"):
    temp = gt_df[[ID_COL, "team_goals", "opp_goals"]].merge(
        pred_df[[ID_COL, "team_goals", "opp_goals"]],
        on=ID_COL,
        how="inner",
        suffixes=("_true", "_pred"),
    )
    temp["true_scoreline"] = temp["team_goals_true"].astype(int).astype(str) + "-" + temp["opp_goals_true"].astype(int).astype(str)
    temp["pred_scoreline"] = temp["team_goals_pred"].astype(int).astype(str) + "-" + temp["opp_goals_pred"].astype(int).astype(str)

    true_dist = temp["true_scoreline"].value_counts().rename("true_count")
    pred_dist = temp["pred_scoreline"].value_counts().rename("pred_count")
    comp = pd.concat([true_dist, pred_dist], axis=1).fillna(0).astype(int).reset_index()
    comp = comp.rename(columns={"index": "scoreline"})
    comp["diff_pred_minus_true"] = comp["pred_count"] - comp["true_count"]
    comp = comp.sort_values("diff_pred_minus_true", ascending=False).reset_index(drop=True)
    return comp


def exp15_total_goal_bucket_report(gt_df, pred_df):
    temp = gt_df[[ID_COL, "team_goals", "opp_goals"]].merge(
        pred_df[[ID_COL, "team_goals", "opp_goals"]],
        on=ID_COL,
        how="inner",
        suffixes=("_true", "_pred"),
    )
    temp["true_total"] = temp["team_goals_true"] + temp["opp_goals_true"]
    temp["pred_total"] = temp["team_goals_pred"] + temp["opp_goals_pred"]

    def bucket(x):
        x = int(x)
        if x <= 0:
            return "0"
        if x == 1:
            return "1"
        if x == 2:
            return "2"
        if x == 3:
            return "3"
        if x == 4:
            return "4"
        if x <= 6:
            return "5-6"
        return "7+"

    temp["true_total_bucket"] = temp["true_total"].map(bucket)
    rows = []
    for b, g in temp.groupby("true_total_bucket"):
        rows.append({
            "true_total_bucket": b,
            "rows": len(g),
            "true_total_mean": g["true_total"].mean(),
            "pred_total_mean": g["pred_total"].mean(),
            "abs_total_error_mean": np.abs(g["true_total"] - g["pred_total"]).mean(),
        })
    order = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5-6": 5, "7+": 6}
    out = pd.DataFrame(rows)
    if len(out):
        out["_order"] = out["true_total_bucket"].map(order)
        out = out.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    return out


if EXP15_ENABLED and RUN_EXP15_GT_AUDIT:
    gt_path = exp15_find_optional_file("./dataset/ground_truth_bersih.csv")
    if gt_path is None:
        print("ground_truth_bersih.csv tidak ditemukan. GT audit dilewati.")
    else:
        gt = pd.read_csv(gt_path)
        gt = gt[[ID_COL, "team_goals", "opp_goals"]].copy()

        eval_df = gt.merge(submission_exp15, on=ID_COL, how="inner", suffixes=("_true", "_pred"))
        eval_df = eval_df.merge(test_raw[[ID_COL, "gender", "tournament"]], on=ID_COL, how="left")

        gt_weights = eval_df["tournament"].map(get_tournament_weight).astype(float).values
        exp15_gt_awmae = awmae_score(
            eval_df["team_goals_true"],
            eval_df["opp_goals_true"],
            eval_df["team_goals_pred"],
            eval_df["opp_goals_pred"],
            weights=gt_weights,
        )

        baseline_eval_df = gt.merge(submission, on=ID_COL, how="inner", suffixes=("_true", "_pred"))
        baseline_eval_df = baseline_eval_df.merge(test_raw[[ID_COL, "gender", "tournament"]], on=ID_COL, how="left")
        baseline_weights = baseline_eval_df["tournament"].map(get_tournament_weight).astype(float).values
        baseline_gt_awmae = awmae_score(
            baseline_eval_df["team_goals_true"],
            baseline_eval_df["opp_goals_true"],
            baseline_eval_df["team_goals_pred"],
            baseline_eval_df["opp_goals_pred"],
            weights=baseline_weights,
        )

        summary = pd.DataFrame([{
            "baseline_gt_awmae": baseline_gt_awmae,
            "exp15_gt_awmae": exp15_gt_awmae,
            "delta_exp15_minus_baseline": exp15_gt_awmae - baseline_gt_awmae,
            "true_mean_goal": pd.concat([eval_df["team_goals_true"], eval_df["opp_goals_true"]]).mean(),
            "pred_mean_goal": pd.concat([eval_df["team_goals_pred"], eval_df["opp_goals_pred"]]).mean(),
            "true_std_goal": pd.concat([eval_df["team_goals_true"], eval_df["opp_goals_true"]]).std(),
            "pred_std_goal": pd.concat([eval_df["team_goals_pred"], eval_df["opp_goals_pred"]]).std(),
        }])
        summary.to_csv(EXP15_SUBDIR / "gt_audit_exp15_summary.csv", index=False)

        seg_rows = []
        for gender, g in eval_df.groupby("gender"):
            w = g["tournament"].map(get_tournament_weight).astype(float).values
            seg_rows.append({
                "gender": gender,
                "rows": len(g),
                "exp15_awmae": awmae_score(g["team_goals_true"], g["opp_goals_true"], g["team_goals_pred"], g["opp_goals_pred"], weights=w),
                "true_mean_goal": pd.concat([g["team_goals_true"], g["opp_goals_true"]]).mean(),
                "pred_mean_goal": pd.concat([g["team_goals_pred"], g["opp_goals_pred"]]).mean(),
            })
        seg_report = pd.DataFrame(seg_rows)
        seg_report.to_csv(EXP15_SUBDIR / "gt_audit_exp15_segment_report.csv", index=False)

        scoreline_compare = exp15_make_scoreline_distribution_compare(gt, submission_exp15)
        scoreline_compare.to_csv(EXP15_SUBDIR / "gt_audit_exp15_scoreline_distribution_compare.csv", index=False)

        total_bucket = exp15_total_goal_bucket_report(gt, submission_exp15)
        total_bucket.to_csv(EXP15_SUBDIR / "gt_audit_exp15_total_goal_bucket_report.csv", index=False)

        print("=" * 90)
        print("GT AUDIT EXP15")
        print("=" * 90)
        display(summary)
        display(seg_report)
        display(scoreline_compare.head(20))
        display(total_bucket)
else:
    print("EXP15 GT audit dilewati.")

# %% [markdown]
# ## 16. EXP17 — W-specialist overlay dari EXP15
# 
# Dari audit sebelumnya, bottleneck utama ada di segment `W`. EXP17 tidak mengubah prediksi `M`; prediksi `M` tetap memakai EXP15. Bagian `W` diproses ulang dengan model ordinal khusus women, lalu hasilnya hanya dipakai jika validasi W membaik.
# 
# Prinsipnya:
# 
# ```text
# M rows → tetap EXP15
# W rows → W-only ordinal model + W-only decoder config
# ```
# 
# Config dipilih dari validation block, bukan dari `ground_truth_bersih.csv`. GT audit di akhir hanya untuk diagnosis lokal.

# %%

# ============================================================
# 16. EXP17 setup
# ============================================================

EXP17_ENABLED = True

# W-only ordinal biasanya butuh ruang tail lebih besar dibanding M.
# Kalau runtime berat, turunkan EXP17_W_MAX_GOAL ke 10 dan EXP17_W_MAX_PRED_GOAL ke 14.
EXP17_W_MAX_GOAL = 10 if FAST_MODE else 12
EXP17_W_MAX_PRED_GOAL = 12 if FAST_MODE else 16
EXP17_W_MIN_TRAIN_ROWS = 700
EXP17_W_MIN_VALID_ROWS = 100

# Kalau True, M benar-benar dipertahankan dari EXP15.
EXP17_FREEZE_M_FROM_EXP15 = True

# Kalau True, GT hanya dipakai untuk audit akhir. Jangan dipakai untuk memilih config final.
RUN_EXP17_GT_AUDIT = bool(globals().get("RUN_GT_AUDIT", False))

EXP17_SUBDIR = OUT_DIR / "exp17_w_specialist_overlay"
EXP17_SUBDIR.mkdir(parents=True, exist_ok=True)

print("EXP17_SUBDIR:", EXP17_SUBDIR.resolve())
print("EXP17_W_MAX_GOAL:", EXP17_W_MAX_GOAL)
print("EXP17_W_MAX_PRED_GOAL:", EXP17_W_MAX_PRED_GOAL)
print("EXP17_FREEZE_M_FROM_EXP15:", EXP17_FREEZE_M_FROM_EXP15)

# %%

# ============================================================
# 16A. EXP17 W-only helper functions
# ============================================================

def exp17_array_subset(obj, idx):
    """Subset dict/list/array secara defensive."""
    idx = np.asarray(idx, dtype=int)
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {k: np.asarray(v)[idx] for k, v in obj.items()}
    return np.asarray(obj)[idx]


def exp17_pmf_subset(pmf, idx):
    idx = np.asarray(idx, dtype=int)
    return {
        "team_goals": np.asarray(pmf["team_goals"])[idx],
        "opp_goals": np.asarray(pmf["opp_goals"])[idx],
    }


def exp17_fit_w_only_ordinal(train_df, feature_cols, max_goal=12):
    """Fit ordinal model khusus W saja tanpa segment nested lagi."""
    train_df = train_df.reset_index(drop=True).copy()
    if len(train_df) < EXP17_W_MIN_TRAIN_ROWS:
        raise ValueError(f"Data W terlalu sedikit untuk specialist model: n={len(train_df)}")

    print("=" * 90)
    print(f"[EXP17] Fit W-only ordinal model | n={len(train_df)} | max_goal={max_goal}")
    print("=" * 90)
    model = exp15_fit_ordinal_models(
        train_df,
        feature_cols,
        max_goal=max_goal,
        label_prefix="EXP17_W_ONLY",
    )
    return {
        "global": model,
        "feature_cols": list(feature_cols),
        "max_goal": int(max_goal),
        "label_prefix": "EXP17_W_ONLY",
    }


def exp17_predict_w_only_ordinal(store, pred_df, feature_cols):
    return exp15_predict_pmf(store["global"], pred_df.reset_index(drop=True), feature_cols)


def exp17_w_decoder_grid():
    """Grid kecil tapi W-aware.

    Dibuat tidak terlalu agresif supaya tidak mengulang kegagalan EXP10/EXP12.
    Fokusnya menaikkan ruang tail W dan sedikit menaikkan scale/offset.
    """
    source_policies = ["sym_avg", "raw", "higher_total", "second_raw"]
    blend_grid = [0.45, 0.60, 0.75, 0.90]
    scale_grid = [1.00, 1.05, 1.10, 1.15, 1.22]
    offset_grid = [0.00, 0.05, 0.10, 0.15]
    aux_grid = [
        (0.00, 0.00, 0.00, 0.00),
        (0.02, 0.02, 0.06, 0.00),
        (0.03, 0.03, 0.08, 0.03),
        (0.04, 0.04, 0.10, 0.05),
    ]

    for source_policy in source_policies:
        for blend_ordinal in blend_grid:
            for scale in scale_grid:
                for offset in offset_grid:
                    for lambda_gd, lambda_total, lambda_outcome, lambda_mu in aux_grid:
                        yield {
                            "decoder": "exp17_w_ordinal_mbr",
                            "source_policy": source_policy,
                            "blend_ordinal": blend_ordinal,
                            "scale": scale,
                            "offset": offset,
                            "lambda_gd": lambda_gd,
                            "lambda_total": lambda_total,
                            "lambda_outcome": lambda_outcome,
                            "lambda_mu": lambda_mu,
                            "max_true": EXP17_W_MAX_GOAL,
                            "max_pred": EXP17_W_MAX_PRED_GOAL,
                        }


def exp17_score_predictions(y_team, y_opp, pred_team, pred_opp, weights):
    return awmae_score(
        np.asarray(y_team),
        np.asarray(y_opp),
        np.asarray(pred_team),
        np.asarray(pred_opp),
        weights=np.asarray(weights) if weights is not None else None,
    )


def exp17_round_blend(pred_a_team, pred_a_opp, pred_b_team, pred_b_opp, alpha):
    """alpha=1 berarti pakai pred_a penuh, alpha=0 berarti pakai pred_b penuh."""
    alpha = float(alpha)
    team = np.rint(alpha * np.asarray(pred_a_team) + (1.0 - alpha) * np.asarray(pred_b_team)).astype(int)
    opp = np.rint(alpha * np.asarray(pred_a_opp) + (1.0 - alpha) * np.asarray(pred_b_opp)).astype(int)
    return np.clip(team, 0, EXP17_W_MAX_PRED_GOAL), np.clip(opp, 0, EXP17_W_MAX_PRED_GOAL)


def exp17_tune_w_overlay(
    valid_w_df,
    y_team_w,
    y_opp_w,
    weights_w,
    w_pmf_raw,
    mu_team_raw_w,
    mu_opp_raw_w,
    exp15_team_w,
    exp15_opp_w,
    aux_raw_w=None,
):
    """Tune W-only overlay dari validation W.

    Selalu memasukkan no-op EXP15 sebagai kandidat. Kalau tidak ada kandidat yang menang,
    overlay otomatis disabled.
    """
    rows = []

    base_score = exp17_score_predictions(y_team_w, y_opp_w, exp15_team_w, exp15_opp_w, weights_w)
    rows.append({
        "enabled": False,
        "blend_with_specialist": 0.0,
        "awmae": base_score,
        "source_policy": "exp15_noop",
        "decoder": "exp15_noop",
        "scale": np.nan,
        "offset": np.nan,
        "blend_ordinal": np.nan,
        "lambda_gd": np.nan,
        "lambda_total": np.nan,
        "lambda_outcome": np.nan,
        "lambda_mu": np.nan,
        "max_true": np.nan,
        "max_pred": np.nan,
    })

    blend_alphas = [1.00, 0.75]

    for cfg in exp17_w_decoder_grid():
        try:
            spec_team, spec_opp = exp15_apply_config(
                valid_w_df.reset_index(drop=True),
                w_pmf_raw,
                mu_team_raw_w,
                mu_opp_raw_w,
                cfg,
                aux_raw=aux_raw_w,
            )
        except Exception as e:
            print("[EXP17] skip config karena error:", repr(e), cfg)
            continue

        for alpha in blend_alphas:
            pred_team, pred_opp = exp17_round_blend(
                spec_team,
                spec_opp,
                exp15_team_w,
                exp15_opp_w,
                alpha=alpha,
            )
            score = exp17_score_predictions(y_team_w, y_opp_w, pred_team, pred_opp, weights_w)
            row = dict(cfg)
            row.update({
                "enabled": True,
                "blend_with_specialist": float(alpha),
                "awmae": float(score),
            })
            rows.append(row)

    report = pd.DataFrame(rows).sort_values("awmae").reset_index(drop=True)
    best = report.iloc[0].to_dict()
    best["base_w_awmae"] = float(base_score)
    best["selected_w_awmae"] = float(report.iloc[0]["awmae"])
    best["improvement_vs_exp15_w"] = float(base_score - report.iloc[0]["awmae"])

    return best, report


def exp17_apply_w_overlay(
    df,
    base_pred_team,
    base_pred_opp,
    w_mask,
    w_pmf_raw,
    mu_team_raw_w,
    mu_opp_raw_w,
    config,
    aux_raw_w=None,
):
    """Apply overlay hanya ke W rows. M rows tetap dari EXP15."""
    out_team = np.asarray(base_pred_team).astype(int).copy()
    out_opp = np.asarray(base_pred_opp).astype(int).copy()
    w_mask = np.asarray(w_mask, dtype=bool)

    if not bool(config.get("enabled", False)):
        return out_team, out_opp

    w_df = df.loc[w_mask].reset_index(drop=True)
    spec_team, spec_opp = exp15_apply_config(
        w_df,
        w_pmf_raw,
        mu_team_raw_w,
        mu_opp_raw_w,
        config,
        aux_raw=aux_raw_w,
    )

    alpha = float(config.get("blend_with_specialist", 1.0))
    final_w_team, final_w_opp = exp17_round_blend(
        spec_team,
        spec_opp,
        out_team[w_mask],
        out_opp[w_mask],
        alpha=alpha,
    )

    out_team[w_mask] = final_w_team
    out_opp[w_mask] = final_w_opp
    return out_team.astype(int), out_opp.astype(int)

# %%

# ============================================================
# 16B. Tune W-specialist overlay on validation block
# ============================================================

if EXP17_ENABLED:
    if not EXP15_ENABLED:
        raise RuntimeError("EXP17 membutuhkan EXP15 aktif karena M rows dibekukan dari EXP15.")

    if "gender" not in train_model_fe.columns or "gender" not in valid_raw.columns:
        print("Kolom gender tidak tersedia. EXP17 fallback ke EXP15.")
        exp17_enabled_runtime = False
    else:
        exp17_enabled_runtime = True

    if exp17_enabled_runtime:
        valid_w_mask = valid_raw["gender"].astype(str).values == "W"
        train_w_mask = train_model_fe["gender"].astype(str).values == "W"

        print("EXP17 train W rows:", int(train_w_mask.sum()))
        print("EXP17 valid W rows:", int(valid_w_mask.sum()))

        if int(train_w_mask.sum()) < EXP17_W_MIN_TRAIN_ROWS or int(valid_w_mask.sum()) < EXP17_W_MIN_VALID_ROWS:
            print("Data W kurang untuk specialist overlay. EXP17 fallback ke EXP15.")
            exp17_enabled_runtime = False

    if exp17_enabled_runtime:
        train_w = train_model_fe.loc[train_w_mask].reset_index(drop=True).copy()
        valid_w = valid_raw.loc[valid_w_mask].reset_index(drop=True).copy()
        valid_w_fe = valid_fe.loc[valid_w_mask].reset_index(drop=True).copy()

        exp17_w_valid_store = exp17_fit_w_only_ordinal(
            train_w,
            feature_cols,
            max_goal=EXP17_W_MAX_GOAL,
        )

        exp17_w_valid_pmf = exp17_predict_w_only_ordinal(
            exp17_w_valid_store,
            valid_w_fe,
            feature_cols,
        )

        valid_w_aux = exp17_array_subset(valid_aux_pred_raw, np.where(valid_w_mask)[0]) if "valid_aux_pred_raw" in globals() else None

        exp17_w_best_config, exp17_w_report = exp17_tune_w_overlay(
            valid_w,
            np.asarray(valid_y_team)[valid_w_mask],
            np.asarray(valid_y_opp)[valid_w_mask],
            np.asarray(valid_weights)[valid_w_mask],
            exp17_w_valid_pmf,
            np.asarray(valid_pred_team_raw)[valid_w_mask],
            np.asarray(valid_pred_opp_raw)[valid_w_mask],
            np.asarray(exp15_valid_team)[valid_w_mask],
            np.asarray(exp15_valid_opp)[valid_w_mask],
            aux_raw_w=valid_w_aux,
        )

        exp17_w_report.to_csv(EXP17_SUBDIR / "exp17_w_overlay_validation_grid.csv", index=False)
        with open(EXP17_SUBDIR / "exp17_w_overlay_selected_config.json", "w", encoding="utf-8") as f:
            json.dump(json_safe(exp17_w_best_config), f, ensure_ascii=False, indent=2)

        print("=" * 90)
        print("EXP17 W base AW-MAE    :", exp17_w_best_config.get("base_w_awmae"))
        print("EXP17 W selected AW-MAE:", exp17_w_best_config.get("selected_w_awmae"))
        print("EXP17 W improvement    :", exp17_w_best_config.get("improvement_vs_exp15_w"))
        print("EXP17 enabled selected :", exp17_w_best_config.get("enabled"))
        print("=" * 90)
        display(exp17_w_report.head(20))
    else:
        exp17_w_best_config = {
            "enabled": False,
            "reason": "W specialist skipped; fallback to EXP15",
            "base_w_awmae": None,
            "selected_w_awmae": None,
            "improvement_vs_exp15_w": 0.0,
        }
        exp17_w_report = pd.DataFrame([exp17_w_best_config])
        with open(EXP17_SUBDIR / "exp17_w_overlay_selected_config.json", "w", encoding="utf-8") as f:
            json.dump(json_safe(exp17_w_best_config), f, ensure_ascii=False, indent=2)
else:
    print("EXP17 dilewati karena EXP17_ENABLED=False")

# %%

# ============================================================
# 16C. Fit final W-specialist and create EXP17 submission
# ============================================================

if EXP17_ENABLED:
    # Base prediction selalu EXP15. M rows akan tetap EXP15, W rows hanya dioverwrite jika config W validasi enabled.
    base_exp17_team = submission_exp15["team_goals"].values.astype(int).copy()
    base_exp17_opp = submission_exp15["opp_goals"].values.astype(int).copy()

    test_w_mask = test_raw["gender"].astype(str).values == "W" if "gender" in test_raw.columns else np.zeros(len(test_raw), dtype=bool)
    final_w_mask = final_model_fe["gender"].astype(str).values == "W" if "gender" in final_model_fe.columns else np.zeros(len(final_model_fe), dtype=bool)

    print("EXP17 final train W rows:", int(final_w_mask.sum()))
    print("EXP17 test W rows       :", int(test_w_mask.sum()))

    if bool(exp17_w_best_config.get("enabled", False)) and int(final_w_mask.sum()) >= EXP17_W_MIN_TRAIN_ROWS and int(test_w_mask.sum()) > 0:
        final_w = final_model_fe.loc[final_w_mask].reset_index(drop=True).copy()
        test_w_fe = test_fe.loc[test_w_mask].reset_index(drop=True).copy()

        exp17_w_final_store = exp17_fit_w_only_ordinal(
            final_w,
            final_feature_cols,
            max_goal=EXP17_W_MAX_GOAL,
        )

        exp17_w_test_pmf = exp17_predict_w_only_ordinal(
            exp17_w_final_store,
            test_w_fe,
            final_feature_cols,
        )

        test_w_aux = exp17_array_subset(test_aux_pred_raw, np.where(test_w_mask)[0]) if "test_aux_pred_raw" in globals() else None

        exp17_test_team, exp17_test_opp = exp17_apply_w_overlay(
            test_raw.reset_index(drop=True),
            base_exp17_team,
            base_exp17_opp,
            test_w_mask,
            exp17_w_test_pmf,
            np.asarray(test_pred_team_raw)[test_w_mask],
            np.asarray(test_pred_opp_raw)[test_w_mask],
            exp17_w_best_config,
            aux_raw_w=test_w_aux,
        )
        exp17_runtime_note = "W overlay applied"
    else:
        exp17_test_team, exp17_test_opp = base_exp17_team, base_exp17_opp
        exp17_runtime_note = "W overlay disabled; using EXP15 predictions"

    submission_exp17 = pd.DataFrame({
        ID_COL: test_raw[ID_COL].values,
        "team_goals": exp17_test_team.astype(int),
        "opp_goals": exp17_test_opp.astype(int),
    })

    assert len(submission_exp17) == len(test_raw)
    assert submission_exp17[ID_COL].equals(test_raw[ID_COL].reset_index(drop=True))
    assert (submission_exp17[["team_goals", "opp_goals"]] >= 0).all().all()
    assert submission_exp17["team_goals"].max() <= max(EXP15_MAX_PRED_GOAL, EXP17_W_MAX_PRED_GOAL)
    assert submission_exp17["opp_goals"].max() <= max(EXP15_MAX_PRED_GOAL, EXP17_W_MAX_PRED_GOAL)

    exp17_submission_path = EXP17_SUBDIR / "submission_awmae_exp17_w_specialist_overlay.csv"
    submission_exp17.to_csv(exp17_submission_path, index=False)

    exp17_test_audit = test_raw[[ID_COL, MATCH_COL, "date", "gender", "team", "opponent", "tournament"]].copy()
    exp17_test_audit["pred_team_exp15"] = submission_exp15["team_goals"].values
    exp17_test_audit["pred_opp_exp15"] = submission_exp15["opp_goals"].values
    exp17_test_audit["pred_team_exp17"] = submission_exp17["team_goals"].values
    exp17_test_audit["pred_opp_exp17"] = submission_exp17["opp_goals"].values
    exp17_test_audit["changed_from_exp15"] = (
        (exp17_test_audit["pred_team_exp15"] != exp17_test_audit["pred_team_exp17"])
        | (exp17_test_audit["pred_opp_exp15"] != exp17_test_audit["pred_opp_exp17"])
    )
    exp17_test_audit.to_csv(EXP17_SUBDIR / "test_prediction_audit_exp17.csv", index=False)

    print(exp17_runtime_note)
    print("EXP17 submission saved to:", exp17_submission_path.resolve())
    print("Rows changed from EXP15:", int(exp17_test_audit["changed_from_exp15"].sum()))
    display(submission_exp17.head(20))
    display(submission_exp17[["team_goals", "opp_goals"]].describe())
    display(
        submission_exp17.groupby(["team_goals", "opp_goals"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(20)
    )
else:
    print("EXP17 final prediction dilewati.")

# %%

# ============================================================
# 16D. Optional GT audit EXP17 untuk diagnosis lokal
# ============================================================

if EXP17_ENABLED and RUN_EXP17_GT_AUDIT:
    gt_path = exp15_find_optional_file("../dataset/ground_truth_bersih.csv")
    if gt_path is None:
        print("ground_truth_bersih.csv tidak ditemukan. GT audit EXP17 dilewati.")
    else:
        gt = pd.read_csv(gt_path)[[ID_COL, "team_goals", "opp_goals"]].copy()

        eval_exp17 = gt.merge(submission_exp17, on=ID_COL, how="inner", suffixes=("_true", "_pred"))
        eval_exp17 = eval_exp17.merge(test_raw[[ID_COL, "gender", "tournament"]], on=ID_COL, how="left")
        w_exp17 = eval_exp17["tournament"].map(get_tournament_weight).astype(float).values

        exp17_gt_awmae = awmae_score(
            eval_exp17["team_goals_true"],
            eval_exp17["opp_goals_true"],
            eval_exp17["team_goals_pred"],
            eval_exp17["opp_goals_pred"],
            weights=w_exp17,
        )

        eval_exp15 = gt.merge(submission_exp15, on=ID_COL, how="inner", suffixes=("_true", "_pred"))
        eval_exp15 = eval_exp15.merge(test_raw[[ID_COL, "gender", "tournament"]], on=ID_COL, how="left")
        w_exp15 = eval_exp15["tournament"].map(get_tournament_weight).astype(float).values
        exp15_gt_awmae_again = awmae_score(
            eval_exp15["team_goals_true"],
            eval_exp15["opp_goals_true"],
            eval_exp15["team_goals_pred"],
            eval_exp15["opp_goals_pred"],
            weights=w_exp15,
        )

        rows = []
        for label, df_eval in [("EXP15", eval_exp15), ("EXP17", eval_exp17)]:
            for seg, g in df_eval.groupby("gender", dropna=False):
                wg = g["tournament"].map(get_tournament_weight).astype(float).values
                rows.append({
                    "experiment": label,
                    "segment_col": "gender",
                    "segment": str(seg),
                    "n": len(g),
                    "awmae": awmae_score(
                        g["team_goals_true"],
                        g["opp_goals_true"],
                        g["team_goals_pred"],
                        g["opp_goals_pred"],
                        weights=wg,
                    ),
                    "true_mean_goal": float(pd.concat([g["team_goals_true"], g["opp_goals_true"]]).mean()),
                    "pred_mean_goal": float(pd.concat([g["team_goals_pred"], g["opp_goals_pred"]]).mean()),
                })
        exp17_segment_report = pd.DataFrame(rows)

        exp17_summary = pd.DataFrame([{
            "experiment": "EXP17_w_specialist_overlay",
            "awmae": exp17_gt_awmae,
            "exp15_awmae_reference": exp15_gt_awmae_again,
            "delta_vs_exp15": exp17_gt_awmae - exp15_gt_awmae_again,
            "n_rows": len(eval_exp17),
            "true_mean_goal": float(pd.concat([eval_exp17["team_goals_true"], eval_exp17["opp_goals_true"]]).mean()),
            "pred_mean_goal": float(pd.concat([eval_exp17["team_goals_pred"], eval_exp17["opp_goals_pred"]]).mean()),
            "true_std_goal": float(pd.concat([eval_exp17["team_goals_true"], eval_exp17["opp_goals_true"]]).std()),
            "pred_std_goal": float(pd.concat([eval_exp17["team_goals_pred"], eval_exp17["opp_goals_pred"]]).std()),
            "exact_acc": float(((eval_exp17["team_goals_true"] == eval_exp17["team_goals_pred"]) & (eval_exp17["opp_goals_true"] == eval_exp17["opp_goals_pred"])).mean()),
            "w_overlay_enabled_by_validation": bool(exp17_w_best_config.get("enabled", False)),
            "w_validation_improvement": exp17_w_best_config.get("improvement_vs_exp15_w"),
        }])

        exp17_scoreline_compare = exp15_make_scoreline_distribution_compare(gt, submission_exp17)
        exp17_total_bucket = exp15_total_goal_bucket_report(gt, submission_exp17)

        exp17_summary.to_csv(EXP17_SUBDIR / "gt_audit_exp17_summary.csv", index=False)
        exp17_segment_report.to_csv(EXP17_SUBDIR / "gt_audit_exp17_segment_report.csv", index=False)
        exp17_scoreline_compare.to_csv(EXP17_SUBDIR / "gt_audit_exp17_scoreline_distribution_compare.csv", index=False)
        exp17_total_bucket.to_csv(EXP17_SUBDIR / "gt_audit_exp17_total_goal_bucket_report.csv", index=False)

        print("EXP15 GT AW-MAE reference:", exp15_gt_awmae_again)
        print("EXP17 GT AW-MAE          :", exp17_gt_awmae)
        print("Delta EXP17 - EXP15      :", exp17_gt_awmae - exp15_gt_awmae_again)
        display(exp17_summary)
        display(exp17_segment_report)
        display(exp17_scoreline_compare.head(20))
        display(exp17_total_bucket)
else:
    print("GT audit EXP17 dilewati.")

# %% [markdown]
# ## 16E. EXP22 final candidate exports
# 
# Cell ini menyimpan kandidat submit yang paling aman untuk dibandingkan:
# - EXP15 frozen ordinal.
# - EXP17 result jika overlay W lolos validasi.
# - EXP22 default candidate sesuai variant aktif.
# 
# Untuk variant default `exp22_no_pseudo_no_overlay_strength`, file final candidate berada di `exp22_final_candidates/submission_exp22_final_candidate.csv`.

# %%

# ============================================================
# 16E. EXP22 final candidate exports
# ============================================================

EXP22_SUBDIR = OUT_DIR / "exp22_final_candidates"
EXP22_SUBDIR.mkdir(parents=True, exist_ok=True)

candidate_rows = []

def _save_candidate_submission(name, df):
    path = EXP22_SUBDIR / f"submission_{name}.csv"
    df = df[[ID_COL, "team_goals", "opp_goals"]].copy()
    assert len(df) == len(test_raw)
    assert df[ID_COL].equals(test_raw[ID_COL].reset_index(drop=True))
    assert df["team_goals"].notna().all() and df["opp_goals"].notna().all()
    assert (df[["team_goals", "opp_goals"]] >= 0).all().all()
    df.to_csv(path, index=False)
    candidate_rows.append({
        "candidate": name,
        "path": str(path),
        "n_rows": len(df),
        "mean_team_goals": float(df["team_goals"].mean()),
        "mean_opp_goals": float(df["opp_goals"].mean()),
        "max_goal": int(max(df["team_goals"].max(), df["opp_goals"].max())),
    })
    print("[SAVED]", path.resolve())
    return path

if "submission_exp15" in globals():
    _save_candidate_submission("exp15_frozen_ordinal", submission_exp15)
else:
    print("[WARN] submission_exp15 belum ada.")

if "submission_exp17" in globals():
    _save_candidate_submission("exp17_w_overlay_or_exp15_fallback", submission_exp17)

# Default final candidate:
# - Jika EXP17 tersedia dan benar-benar mengubah prediksi, gunakan EXP17.
# - Jika tidak, gunakan EXP15 frozen.
# - Fitur EXP22C sudah memengaruhi model/EXP15 jika flag USE_LAST_KNOWN_STRENGTH_FEATURES=True.
if "submission_exp17" in globals() and "submission_exp15" in globals():
    changed = int(
        (submission_exp17["team_goals"].values != submission_exp15["team_goals"].values).sum() +
        (submission_exp17["opp_goals"].values != submission_exp15["opp_goals"].values).sum()
    )
    print("EXP17 changed cells vs EXP15:", changed)
    final_candidate = submission_exp17.copy() if changed > 0 else submission_exp15.copy()
elif "submission_exp15" in globals():
    final_candidate = submission_exp15.copy()
else:
    final_candidate = submission.copy()

final_candidate_path = _save_candidate_submission("exp22_final_candidate", final_candidate)

candidate_summary = pd.DataFrame(candidate_rows)
candidate_summary.to_csv(EXP22_SUBDIR / "candidate_summary.csv", index=False)
display(candidate_summary)

# Optional local GT audit. Ini hanya untuk diagnosis setelah submit/prediksi jadi.
# Tidak dipakai di training/tuning.
def _find_optional_gt_file_exp22():
    candidates = [
        Path("ground_truth_bersih.csv"),
        Path("./data/ground_truth_bersih.csv"),
        Path("../dataset/ground_truth_bersih.csv"),
        Path("/mnt/data/ground_truth_bersih.csv"),
        Path("test_with_groundtruth.csv"),
        Path("./data/test_with_groundtruth.csv"),
        Path("../dataset/test_with_groundtruth.csv"),
        Path("/mnt/data/test_with_groundtruth.csv"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None

gt_path_exp22 = _find_optional_gt_file_exp22() if bool(globals().get("RUN_GT_AUDIT", False)) else None
if gt_path_exp22 is not None:
    print("[EXP22 optional audit] Found GT file:", gt_path_exp22)
    gt_df = pd.read_csv(gt_path_exp22)

    gt_cols = set(gt_df.columns)
    if {ID_COL, "team_goals", "opp_goals"}.issubset(gt_cols):
        audit = final_candidate.merge(
            gt_df[[ID_COL, "team_goals", "opp_goals"]].rename(columns={
                "team_goals": "true_team_goals",
                "opp_goals": "true_opp_goals",
            }),
            on=ID_COL,
            how="left",
        )
    elif {ID_COL, "actual_team_goals", "actual_opp_goals"}.issubset(gt_cols):
        audit = final_candidate.merge(
            gt_df[[ID_COL, "actual_team_goals", "actual_opp_goals"]].rename(columns={
                "actual_team_goals": "true_team_goals",
                "actual_opp_goals": "true_opp_goals",
            }),
            on=ID_COL,
            how="left",
        )
    else:
        audit = None
        print("[EXP22 optional audit] GT columns tidak dikenali:", list(gt_df.columns))

    if audit is not None and audit[["true_team_goals", "true_opp_goals"]].notna().all().all():
        # Convert tournament name -> numeric weight dulu
        audit_weights = np.array(
            [get_tournament_weight(t) for t in test_raw["tournament"].values],
            dtype=float
        )

        score = awmae_score(
            audit["true_team_goals"].values,
            audit["true_opp_goals"].values,
            audit["team_goals"].values,
            audit["opp_goals"].values,
            audit_weights,
        )
        pd.DataFrame([{
            "variant": globals().get("EXP12C_VARIANT", EXP18_ABLATION_VARIANT),
            "candidate": "exp22_final_candidate",
            "awmae": float(score),
            "path": str(final_candidate_path),
        }]).to_csv(EXP22_SUBDIR / "gt_audit_exp22_final_candidate.csv", index=False)
        print("[EXP22 optional audit] AW-MAE:", score)
else:
    print("[EXP22 optional audit] GT file tidak ditemukan; audit dilewati.")

# %%
# ============================================================
# 17. Ablation run summary
# ============================================================

summary_files = []
for p in OUT_DIR.rglob("gt_audit_*_summary.csv"):
    try:
        df = pd.read_csv(p)
        df.insert(0, "variant", EXP18_ABLATION_VARIANT)
        df.insert(1, "file", str(p.relative_to(OUT_DIR)))
        summary_files.append(df)
    except Exception as e:
        print("Failed reading", p, e)

if summary_files:
    ablation_summary = pd.concat(summary_files, ignore_index=True)
    ablation_summary.to_csv(OUT_DIR / "ablation_summary_collected.csv", index=False)
    print("Ablation summary saved:", (OUT_DIR / "ablation_summary_collected.csv").resolve())
    display(ablation_summary)
else:
    print("Belum ada GT summary file yang terkumpul di", OUT_DIR)

print("Variant selesai:", EXP18_ABLATION_VARIANT)
print("Output folder:", OUT_DIR.resolve())

# %%
print("EXP12D selesai ✓")


# %% [markdown]
# # 17. EXP12D Output Hardening, Candidate Generation, dan Final Decision
#
# Penjelasan bagian:
# Bagian ini mengambil output EXP22 yang sudah dibangun oleh source pipeline, lalu menyimpannya dengan struktur project kita. Di sini D0/D1/D2 dibuat dari kandidat EXP22, sedangkan D3/D4/D5/D6 membuat candidate tambahan secara eksplisit tanpa memakai GT.
#
# Output yang perlu dilihat:
# Lihat `submission_catalog.csv`, `submission_check.csv`, `variant_metrics.csv`, `changed_prediction_analysis.csv`, dan `final_decision.csv`. Intermediate artifact memakai `strict=False`, tetapi `submission_exp12d_best_safe.csv` memakai `strict=True`.

# %%
# ============================================================
# 17. EXP12D output hardening, candidate generation, final decision
# ============================================================

SUM_DIR = OUT_DIR / "summaries"
PRED_DIR = OUT_DIR / "predictions"
SUB_DIR = OUT_DIR / "submissions"
FIG_DIR = OUT_DIR / "figures"
for _d in [SUM_DIR, PRED_DIR, SUB_DIR, FIG_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

def log_section(title: str):
    print("\n" + "=" * 100, flush=True)
    print(f"[SECTION] {title}", flush=True)
    print("=" * 100, flush=True)

def log_info(msg: str):
    print(f"[INFO] {msg}", flush=True)

def log_check(name: str, passed: bool, detail: str = ""):
    status = "PASS" if bool(passed) else "FAIL"
    suffix = f" | {detail}" if detail else ""
    print(f"[CHECK] {name}: {status}{suffix}", flush=True)

def log_result(msg: str):
    print(f"[RESULT] {msg}", flush=True)

def log_saved(path):
    print(f"[SAVED] {path}", flush=True)

def outcome_array(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return np.where(a > b, 1, np.where(a < b, -1, 0))

# Robust sample finder. Do not search outputs or old submissions as sample.
def find_sample_submission_exp12d() -> Path:
    names = [
        "sample_submission.csv",
        "sample submission.csv",
        "samplesubmission.csv",
        "sample-submission.csv",
        "submission_sample.csv",
    ]
    dirs = []
    for d in [
        PROJECT_ROOT / "data",
        PROJECT_ROOT / "dataset",
        PROJECT_ROOT,
        PROJECT_ROOT.parent / "data",
        PROJECT_ROOT.parent / "dataset",
        TRAIN_PATH.parent if "TRAIN_PATH" in globals() else None,
        TEST_PATH.parent if "TEST_PATH" in globals() else None,
        Path("."),
        Path("./data"),
        Path("../dataset"),
        Path("/mnt/data"),
    ]:
        if d is not None:
            dirs.append(Path(d).resolve())

    checked = []
    for d in list(dict.fromkeys(dirs)):
        for name in names:
            p = (d / name).resolve()
            checked.append(str(p))
            if "outputs" in p.parts:
                continue
            if p.exists():
                return p

    # fallback recursive search, but exclude outputs and submission_exp*
    for root in [PROJECT_ROOT, PROJECT_ROOT.parent, Path("/mnt/data")]:
        root = Path(root).resolve()
        if not root.exists():
            continue
        for p in root.rglob("*.csv"):
            low = p.name.lower().replace("_", "").replace("-", "").replace(" ", "")
            if "outputs" in p.parts:
                continue
            if p.name.lower().startswith("submission_exp"):
                continue
            if low in {"samplesubmission.csv", "submissionsample.csv"}:
                return p.resolve()

    raise FileNotFoundError("sample_submission file not found. Checked examples: " + str(checked[:20]))

SAMPLE_SUBMISSION_PATH = find_sample_submission_exp12d()
sample_submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)
sample_id_col = ID_COL if ID_COL in sample_submission.columns else ("id" if "id" in sample_submission.columns else sample_submission.columns[0])
log_info(f"SAMPLE_SUBMISSION_PATH = {SAMPLE_SUBMISSION_PATH}")
log_info(f"sample_id_col = {sample_id_col}")

def _normalize_submission_columns_exp12d(sub_df: pd.DataFrame) -> pd.DataFrame:
    out = sub_df.copy()
    if ID_COL not in out.columns and "id" in out.columns:
        out = out.rename(columns={"id": ID_COL})
    if ID_COL not in out.columns and sample_id_col in out.columns:
        out = out.rename(columns={sample_id_col: ID_COL})
    required = [ID_COL, "team_goals", "opp_goals"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise KeyError(f"Submission missing columns: {missing}")
    out = out[required].copy()
    out["team_goals"] = pd.to_numeric(out["team_goals"], errors="raise").round().astype(int).clip(lower=0)
    out["opp_goals"] = pd.to_numeric(out["opp_goals"], errors="raise").round().astype(int).clip(lower=0)
    return out

def _pair_consistency_exp12d(sub_df: pd.DataFrame, row_df: pd.DataFrame) -> tuple[bool, int]:
    sub = _normalize_submission_columns_exp12d(sub_df)
    ids = row_df[[ID_COL, MATCH_COL]].copy()
    ids["_id_key"] = ids[ID_COL].astype(str)
    sub["_id_key"] = sub[ID_COL].astype(str)
    merged = sub.merge(ids[["_id_key", MATCH_COL]], on="_id_key", how="left", validate="one_to_one")
    if merged[MATCH_COL].isna().any():
        return False, int(merged[MATCH_COL].isna().sum())
    bad = 0
    for _, g in merged.groupby(MATCH_COL, sort=False):
        if len(g) != 2:
            bad += 1
            continue
        r0 = g.iloc[0]
        r1 = g.iloc[1]
        ok = (int(r0["team_goals"]) == int(r1["opp_goals"])) and (int(r0["opp_goals"]) == int(r1["team_goals"]))
        if not ok:
            bad += 1
    return bad == 0, bad

def align_to_sample_exp12d(sub_df: pd.DataFrame) -> pd.DataFrame:
    sub = _normalize_submission_columns_exp12d(sub_df)
    sample_aligned = sample_submission[[sample_id_col]].copy()
    sample_aligned_internal = sample_aligned.rename(columns={sample_id_col: ID_COL})
    sample_aligned_internal["_id_key"] = sample_aligned_internal[ID_COL].astype(str)

    sub["_id_key"] = sub[ID_COL].astype(str)
    aligned = sample_aligned_internal.merge(
        sub[["_id_key", "team_goals", "opp_goals"]],
        on="_id_key",
        how="left",
        validate="one_to_one",
    ).drop(columns=["_id_key"])
    return aligned[[ID_COL, "team_goals", "opp_goals"]].copy()

def validate_and_save_submission_exp12d(sub_df: pd.DataFrame, label: str, strict: bool = True):
    aligned = align_to_sample_exp12d(sub_df)
    pair_pass, n_bad_pairs = _pair_consistency_exp12d(aligned, test_raw)
    checks = [
        {"variant": label, "check": "shape_matches_sample", "passed": tuple(aligned.shape) == (len(sample_submission), 3), "detail": str(tuple(aligned.shape))},
        {"variant": label, "check": "id_order_matches_sample", "passed": aligned[ID_COL].astype(str).tolist() == sample_submission[sample_id_col].astype(str).tolist(), "detail": ""},
        {"variant": label, "check": "no_missing", "passed": not aligned[["team_goals", "opp_goals"]].isna().any().any(), "detail": ""},
        {"variant": label, "check": "no_duplicate_id", "passed": not aligned[ID_COL].duplicated().any(), "detail": ""},
        {"variant": label, "check": "non_negative", "passed": bool((aligned[["team_goals", "opp_goals"]] >= 0).all().all()), "detail": ""},
        {"variant": label, "check": "pair_consistency", "passed": bool(pair_pass), "detail": f"n_bad_pairs={n_bad_pairs}"},
        {"variant": label, "check": "max_goal_sanity_le_40", "passed": bool((aligned[["team_goals", "opp_goals"]] <= 40).all().all()), "detail": ""},
    ]
    check_df = pd.DataFrame(checks)

    save_df = aligned.rename(columns={ID_COL: sample_id_col})[[sample_id_col, "team_goals", "opp_goals"]].copy()
    path = SUB_DIR / f"submission_exp12d_{label}.csv"
    save_df.to_csv(path, index=False)
    log_saved(path)

    if strict and not bool(check_df["passed"].all()):
        display(check_df)
        raise RuntimeError(f"Submission validation failed for {label}")

    return aligned, path, check_df

def submission_to_match_df(sub_df: pd.DataFrame, row_df: pd.DataFrame) -> pd.DataFrame:
    sub = align_to_sample_exp12d(sub_df)
    row_info = row_df[[ID_COL, MATCH_COL, "gender", "tournament"]].copy()
    row_info["_id_key"] = row_info[ID_COL].astype(str)
    sub["_id_key"] = sub[ID_COL].astype(str)
    merged = sub.merge(row_info[["_id_key", MATCH_COL, "gender", "tournament"]], on="_id_key", how="left", validate="one_to_one")
    records = []
    for match_id, g in merged.groupby(MATCH_COL, sort=False):
        if len(g) != 2:
            continue
        r0 = g.iloc[0]
        r1 = g.iloc[1]
        records.append({
            MATCH_COL: match_id,
            "row_id_a": r0[ID_COL],
            "row_id_b": r1[ID_COL],
            "gender": r0.get("gender", np.nan),
            "tournament": r0.get("tournament", np.nan),
            "pred_team_a_goals": int(r0["team_goals"]),
            "pred_team_b_goals": int(r0["opp_goals"]),
        })
    return pd.DataFrame(records)

def match_df_to_submission_exp12d(match_df: pd.DataFrame) -> pd.DataFrame:
    rows_a = pd.DataFrame({
        ID_COL: match_df["row_id_a"],
        "team_goals": match_df["pred_team_a_goals"].astype(int),
        "opp_goals": match_df["pred_team_b_goals"].astype(int),
    })
    rows_b = pd.DataFrame({
        ID_COL: match_df["row_id_b"],
        "team_goals": match_df["pred_team_b_goals"].astype(int),
        "opp_goals": match_df["pred_team_a_goals"].astype(int),
    })
    return pd.concat([rows_a, rows_b], ignore_index=True)

def outcome_from_scores(a: pd.Series, b: pd.Series) -> np.ndarray:
    return outcome_array(pd.to_numeric(a), pd.to_numeric(b))

def make_scoreline_distribution(sub_df: pd.DataFrame, label: str) -> pd.DataFrame:
    sub = align_to_sample_exp12d(sub_df)
    d = (
        sub.assign(scoreline=lambda x: x["team_goals"].astype(int).astype(str) + "-" + x["opp_goals"].astype(int).astype(str))
        ["scoreline"].value_counts(normalize=True).reset_index()
    )
    d.columns = ["scoreline", "share"]
    d.insert(0, "variant", label)
    return d

def changed_analysis(base_sub: pd.DataFrame, cand_sub: pd.DataFrame, label: str) -> pd.DataFrame:
    base = align_to_sample_exp12d(base_sub).rename(columns={"team_goals": "base_team_goals", "opp_goals": "base_opp_goals"})
    cand = align_to_sample_exp12d(cand_sub).rename(columns={"team_goals": "cand_team_goals", "opp_goals": "cand_opp_goals"})
    m = base.merge(cand, on=ID_COL, how="inner", validate="one_to_one")
    changed = (m["base_team_goals"] != m["cand_team_goals"]) | (m["base_opp_goals"] != m["cand_opp_goals"])
    base_out = outcome_from_scores(m["base_team_goals"], m["base_opp_goals"])
    cand_out = outcome_from_scores(m["cand_team_goals"], m["cand_opp_goals"])
    base_gd = m["base_team_goals"] - m["base_opp_goals"]
    cand_gd = m["cand_team_goals"] - m["cand_opp_goals"]
    trans = (
        m.loc[changed]
        .assign(
            transition=lambda x: x["base_team_goals"].astype(str) + "-" + x["base_opp_goals"].astype(str) + " -> " + x["cand_team_goals"].astype(str) + "-" + x["cand_opp_goals"].astype(str)
        )["transition"].value_counts().head(10).reset_index()
    )
    if len(trans):
        trans.columns = ["top_transition", "transition_count"]
        top_transition = "; ".join((trans["top_transition"] + ":" + trans["transition_count"].astype(str)).tolist())
    else:
        top_transition = ""
    return pd.DataFrame([{
        "variant": label,
        "n_rows": len(m),
        "n_changed": int(changed.sum()),
        "changed_rate": float(changed.mean()) if len(m) else 0.0,
        "same_outcome_changed": int(((base_out == cand_out) & changed).sum()),
        "same_gd_changed": int(((base_gd == cand_gd) & changed).sum()),
        "mean_total_before": float((m["base_team_goals"] + m["base_opp_goals"]).mean()),
        "mean_total_after": float((m["cand_team_goals"] + m["cand_opp_goals"]).mean()),
        "top_transitions": top_transition,
    }])

def gt_free_metrics(sub_df: pd.DataFrame, label: str, base_sub: pd.DataFrame | None = None) -> pd.DataFrame:
    aligned = align_to_sample_exp12d(sub_df)
    pair_pass, n_bad = _pair_consistency_exp12d(aligned, test_raw)
    dist = make_scoreline_distribution(aligned, label)
    rec = {
        "variant": label,
        "n_rows": len(aligned),
        "mean_pred_total": float((aligned["team_goals"] + aligned["opp_goals"]).mean()),
        "max_pred_goal": int(aligned[["team_goals", "opp_goals"]].max().max()),
        "top1_scoreline_share": float(dist["share"].iloc[0]) if len(dist) else np.nan,
        "top3_scoreline_share": float(dist["share"].head(3).sum()) if len(dist) else np.nan,
        "pair_consistency": bool(pair_pass),
        "n_bad_pairs": int(n_bad),
    }
    if base_sub is not None:
        ch = changed_analysis(base_sub, aligned, label)
        rec["changed_rate_vs_exp15_base"] = float(ch["changed_rate"].iloc[0])
    else:
        rec["changed_rate_vs_exp15_base"] = 0.0
    return pd.DataFrame([rec])

# Candidate source outputs from EXP22.
candidate_sources = {}
if "submission_exp15" in globals():
    candidate_sources["d1_exp15_first"] = submission_exp15
if "submission_exp17" in globals():
    candidate_sources["d2_exp17_original"] = submission_exp17
if "final_candidate" in globals():
    candidate_sources["d0_final_candidate"] = final_candidate
elif "submission_exp17" in globals():
    candidate_sources["d0_final_candidate"] = submission_exp17
elif "submission_exp15" in globals():
    candidate_sources["d0_final_candidate"] = submission_exp15
if "submission" in globals():
    candidate_sources["advanced_base"] = submission

if "d1_exp15_first" not in candidate_sources:
    raise RuntimeError("submission_exp15 is required for EXP12D D1/D3/D4/D5/D6 candidates.")

exp15_base_sub = candidate_sources["d1_exp15_first"]

# D3: bias/scoreline calibration. Conservative same-outcome reductions.
def apply_bias_calibration(sub_df: pd.DataFrame, strength: str = "light") -> pd.DataFrame:
    match = submission_to_match_df(sub_df, test_raw)
    a = match["pred_team_a_goals"].astype(int).copy()
    b = match["pred_team_b_goals"].astype(int).copy()

    if strength in {"light", "medium", "strong"}:
        # Same-outcome / same-GD lower-total transitions.
        mask = (a == 2) & (b == 1)
        a.loc[mask], b.loc[mask] = 1, 0
        mask = (a == 1) & (b == 2)
        a.loc[mask], b.loc[mask] = 0, 1
        mask = (a == 2) & (b == 2)
        a.loc[mask], b.loc[mask] = 1, 1

    if strength in {"medium", "strong"}:
        mask = (a == 3) & (b == 1)
        a.loc[mask], b.loc[mask] = 2, 0
        mask = (a == 1) & (b == 3)
        a.loc[mask], b.loc[mask] = 0, 2
        mask = (a == 3) & (b == 2)
        a.loc[mask], b.loc[mask] = 2, 1
        mask = (a == 2) & (b == 3)
        a.loc[mask], b.loc[mask] = 1, 2

    if strength == "strong":
        mask = (a == 1) & (b == 1)
        a.loc[mask], b.loc[mask] = 0, 0

    out_match = match.copy()
    out_match["pred_team_a_goals"] = a.clip(lower=0).astype(int)
    out_match["pred_team_b_goals"] = b.clip(lower=0).astype(int)
    return match_df_to_submission_exp12d(out_match)

d3_grid_records = []
d3_candidates = {}
for strength in ["light", "medium", "strong"]:
    cand = apply_bias_calibration(exp15_base_sub, strength=strength)
    label = f"d3_bias_{strength}"
    d3_candidates[label] = cand
    m = gt_free_metrics(cand, label, exp15_base_sub)
    ch = changed_analysis(exp15_base_sub, cand, label)
    d3_grid_records.append({**m.iloc[0].to_dict(), **{f"changed_{k}": v for k, v in ch.iloc[0].to_dict().items() if k != "variant"}})
decoder_bias_calibration_grid = pd.DataFrame(d3_grid_records)
decoder_bias_calibration_grid.to_csv(SUM_DIR / "decoder_bias_calibration_grid.csv", index=False)
log_saved(SUM_DIR / "decoder_bias_calibration_grid.csv")
display(decoder_bias_calibration_grid)

# Select D3 with changed rate not too high; prefer light if all close.
d3_selected_label = "d3_bias_light"
candidate_sources["d3_exp15_bias_calibrated"] = d3_candidates[d3_selected_label]

# D4: outcome-aware calibration using EXP17 where it agrees tightly with EXP15.
def make_outcome_aware_from_exp17(exp15_sub: pd.DataFrame, exp17_sub: pd.DataFrame) -> pd.DataFrame:
    base = submission_to_match_df(exp15_sub, test_raw)
    alt = submission_to_match_df(exp17_sub, test_raw).rename(columns={
        "pred_team_a_goals": "alt_a",
        "pred_team_b_goals": "alt_b",
    })
    m = base.merge(alt[[MATCH_COL, "alt_a", "alt_b"]], on=MATCH_COL, how="left", validate="one_to_one")
    base_out = outcome_array(m["pred_team_a_goals"], m["pred_team_b_goals"])
    alt_out = outcome_array(m["alt_a"], m["alt_b"])
    base_gd = m["pred_team_a_goals"] - m["pred_team_b_goals"]
    alt_gd = m["alt_a"] - m["alt_b"]
    base_total = m["pred_team_a_goals"] + m["pred_team_b_goals"]
    alt_total = m["alt_a"] + m["alt_b"]
    take_alt = (
        (base_out == alt_out)
        & ((base_gd - alt_gd).abs() <= 1)
        & ((base_total - alt_total).abs() <= 1)
        & m["alt_a"].notna()
        & m["alt_b"].notna()
    )
    m.loc[take_alt, "pred_team_a_goals"] = m.loc[take_alt, "alt_a"].astype(int)
    m.loc[take_alt, "pred_team_b_goals"] = m.loc[take_alt, "alt_b"].astype(int)
    return match_df_to_submission_exp12d(m[["match_id", "row_id_a", "row_id_b", "gender", "tournament", "pred_team_a_goals", "pred_team_b_goals"]])

if "d2_exp17_original" in candidate_sources:
    d4_candidate = make_outcome_aware_from_exp17(exp15_base_sub, candidate_sources["d2_exp17_original"])
else:
    d4_candidate = exp15_base_sub
candidate_sources["d4_exp15_outcome_calibrated"] = d4_candidate
pd.DataFrame([{
    "variant": "d4_exp15_outcome_calibrated",
    "source": "exp17_agreement" if "d2_exp17_original" in candidate_sources else "fallback_exp15",
}]).to_csv(SUM_DIR / "outcome_calibration_grid.csv", index=False)
log_saved(SUM_DIR / "outcome_calibration_grid.csv")

# D5: real agreement hybrid with EXP12B candidate if provided. It is never used for training.
def load_exp12b_candidate_if_allowed():
    allow = os.environ.get("EXP12D_ALLOW_EXP12B_CSV_HYBRID", "0") == "1"
    explicit = os.environ.get("EXP12B_CANDIDATE_PATH", "")
    if not allow and not explicit:
        return None, "disabled_no_inline_exp12b"
    paths = []
    if explicit:
        paths.append(Path(explicit))
    for p in (PROJECT_ROOT / "outputs").rglob("submission_exp12b_b5_exp15_calibrated.csv"):
        paths.append(p)
    for p in paths:
        if p.exists():
            return pd.read_csv(p), str(p)
    return None, "not_found"

exp12b_candidate, exp12b_source = load_exp12b_candidate_if_allowed()

def make_agreement_hybrid(base_sub: pd.DataFrame, alt_sub: pd.DataFrame) -> pd.DataFrame:
    base = submission_to_match_df(base_sub, test_raw)
    alt = submission_to_match_df(alt_sub, test_raw).rename(columns={
        "pred_team_a_goals": "alt_a",
        "pred_team_b_goals": "alt_b",
    })
    m = base.merge(alt[[MATCH_COL, "alt_a", "alt_b"]], on=MATCH_COL, how="left", validate="one_to_one")
    base_out = outcome_array(m["pred_team_a_goals"], m["pred_team_b_goals"])
    alt_out = outcome_array(m["alt_a"], m["alt_b"])
    base_gd = m["pred_team_a_goals"] - m["pred_team_b_goals"]
    alt_gd = m["alt_a"] - m["alt_b"]
    base_total = m["pred_team_a_goals"] + m["pred_team_b_goals"]
    alt_total = m["alt_a"] + m["alt_b"]
    high_risk_w = m["gender"].astype(str).str.upper().eq("W") & ((base_total >= 5) | (alt_total >= 5))
    take_alt = (
        (base_out == alt_out)
        & ((base_gd - alt_gd).abs() <= 1)
        & ((base_total - alt_total).abs() <= 1)
        & (~high_risk_w)
        & m["alt_a"].notna()
        & m["alt_b"].notna()
    )
    m.loc[take_alt, "pred_team_a_goals"] = m.loc[take_alt, "alt_a"].astype(int)
    m.loc[take_alt, "pred_team_b_goals"] = m.loc[take_alt, "alt_b"].astype(int)
    return match_df_to_submission_exp12d(m[["match_id", "row_id_a", "row_id_b", "gender", "tournament", "pred_team_a_goals", "pred_team_b_goals"]])

if exp12b_candidate is not None:
    d5_candidate = make_agreement_hybrid(exp15_base_sub, exp12b_candidate)
    d5_status = "active"
else:
    d5_candidate = exp15_base_sub
    d5_status = f"skipped_{exp12b_source}"
candidate_sources["d5_agreement_hybrid"] = d5_candidate
hybrid_analysis = changed_analysis(exp15_base_sub, d5_candidate, "d5_agreement_hybrid")
hybrid_analysis.insert(1, "status", d5_status)
hybrid_analysis.insert(2, "exp12b_source", exp12b_source)
hybrid_analysis.to_csv(SUM_DIR / "hybrid_analysis.csv", index=False)
log_saved(SUM_DIR / "hybrid_analysis.csv")
display(hybrid_analysis)

# D6: conservative exact booster with limited same-outcome transitions.
def apply_exact_booster(sub_df: pd.DataFrame) -> pd.DataFrame:
    match = submission_to_match_df(sub_df, test_raw)
    a = match["pred_team_a_goals"].astype(int).copy()
    b = match["pred_team_b_goals"].astype(int).copy()

    # Conservative transitions: keep outcome, avoid extreme/high total.
    total = a + b
    mask = (a == 2) & (b == 0) & (total <= 3)
    a.loc[mask], b.loc[mask] = 2, 1
    mask = (a == 0) & (b == 2) & (total <= 3)
    a.loc[mask], b.loc[mask] = 1, 2
    mask = (a == 2) & (b == 1) & (total <= 3)
    a.loc[mask], b.loc[mask] = 1, 0
    mask = (a == 1) & (b == 2) & (total <= 3)
    a.loc[mask], b.loc[mask] = 0, 1
    mask = (a == 2) & (b == 2) & (total <= 4)
    a.loc[mask], b.loc[mask] = 1, 1

    out_match = match.copy()
    out_match["pred_team_a_goals"] = a.clip(lower=0).astype(int)
    out_match["pred_team_b_goals"] = b.clip(lower=0).astype(int)
    return match_df_to_submission_exp12d(out_match)

d6_candidate = apply_exact_booster(exp15_base_sub)
candidate_sources["d6_exact_booster"] = d6_candidate
exact_booster_grid = changed_analysis(exp15_base_sub, d6_candidate, "d6_exact_booster")
exact_booster_grid.to_csv(SUM_DIR / "exact_booster_grid.csv", index=False)
log_saved(SUM_DIR / "exact_booster_grid.csv")
display(exact_booster_grid)

# Save candidates. Intermediates are non-strict; final is strict.
submission_catalog_rows = []
submission_check_frames = []
candidate_save_map = {
    "d0_final_candidate": "d0_exp22_reproduction_final_candidate",
    "d1_exp15_first": "d1_exp15_first",
    "d2_exp17_original": "d2_exp17_original",
    "d3_exp15_bias_calibrated": "d3_bias_calibrated",
    "d4_exp15_outcome_calibrated": "d4_outcome_calibrated",
    "d5_agreement_hybrid": "d5_agreement_hybrid",
    "d6_exact_booster": "d6_exact_booster",
}
if "advanced_base" in candidate_sources:
    candidate_save_map["advanced_base"] = "comparison_advanced_base"

for key, label in candidate_save_map.items():
    if key not in candidate_sources:
        continue
    strict = False
    aligned, path, check_df = validate_and_save_submission_exp12d(candidate_sources[key], label, strict=strict)
    submission_catalog_rows.append({"key": key, "label": label, "path": str(path), "strict": strict})
    submission_check_frames.append(check_df)

# Also save EXP15/EXP17 explicitly for D0 reproduction artifacts if available.
if "d1_exp15_first" in candidate_sources:
    aligned, path, check_df = validate_and_save_submission_exp12d(candidate_sources["d1_exp15_first"], "d0_exp22_reproduction_exp15", strict=False)
    submission_catalog_rows.append({"key": "d0_exp15_artifact", "label": "d0_exp22_reproduction_exp15", "path": str(path), "strict": False})
    submission_check_frames.append(check_df)
if "d2_exp17_original" in candidate_sources:
    aligned, path, check_df = validate_and_save_submission_exp12d(candidate_sources["d2_exp17_original"], "d0_exp22_reproduction_exp17", strict=False)
    submission_catalog_rows.append({"key": "d0_exp17_artifact", "label": "d0_exp22_reproduction_exp17", "path": str(path), "strict": False})
    submission_check_frames.append(check_df)

# Select best-safe according to requested policy, without claiming audit-best.
policy_to_key = {
    "d0_all_exp22_outputs": "d0_final_candidate",
    "exp15_first": "d1_exp15_first",
    "exp17_original": "d2_exp17_original",
    "bias_calibration": "d3_exp15_bias_calibrated",
    "outcome_calibration": "d4_exp15_outcome_calibrated",
    "agreement_hybrid": "d5_agreement_hybrid" if d5_status == "active" else "d1_exp15_first",
    "exact_booster": "d6_exact_booster",
    "final_selected": "d1_exp15_first",
}
selected_key = policy_to_key.get(EXP12D_CANDIDATE_POLICY, "d1_exp15_first")
if selected_key not in candidate_sources:
    selected_key = "d1_exp15_first"

best_aligned, best_path, best_check_df = validate_and_save_submission_exp12d(
    candidate_sources[selected_key],
    "best_safe",
    strict=True,
)
submission_catalog_rows.append({"key": selected_key, "label": "best_safe", "path": str(best_path), "strict": True})
submission_check_frames.append(best_check_df)

submission_catalog_df = pd.DataFrame(submission_catalog_rows).drop_duplicates().reset_index(drop=True)
submission_catalog_df.to_csv(SUM_DIR / "submission_catalog.csv", index=False)
log_saved(SUM_DIR / "submission_catalog.csv")
display(submission_catalog_df)

submission_check_df = pd.concat(submission_check_frames, ignore_index=True).drop_duplicates().reset_index(drop=True)
submission_check_df.to_csv(SUM_DIR / "submission_check.csv", index=False)
log_saved(SUM_DIR / "submission_check.csv")
display(submission_check_df)

# Variant metrics and scoreline distribution.
metrics_frames = []
scoreline_frames = []
for key, sub in candidate_sources.items():
    label = candidate_save_map.get(key, key)
    metrics_frames.append(gt_free_metrics(sub, label, exp15_base_sub))
    scoreline_frames.append(make_scoreline_distribution(sub, label).head(20))
variant_metrics_df = pd.concat(metrics_frames, ignore_index=True)
variant_metrics_df.to_csv(SUM_DIR / "variant_metrics.csv", index=False)
log_saved(SUM_DIR / "variant_metrics.csv")
display(variant_metrics_df)

scoreline_distribution = pd.concat(scoreline_frames, ignore_index=True)
scoreline_distribution.to_csv(SUM_DIR / "scoreline_distribution.csv", index=False)
log_saved(SUM_DIR / "scoreline_distribution.csv")
display(scoreline_distribution.head(50))

changed_frames = []
for key, sub in candidate_sources.items():
    if key == "d1_exp15_first":
        continue
    label = candidate_save_map.get(key, key)
    changed_frames.append(changed_analysis(exp15_base_sub, sub, label))
changed_prediction_analysis = pd.concat(changed_frames, ignore_index=True) if changed_frames else pd.DataFrame()
changed_prediction_analysis.to_csv(SUM_DIR / "changed_prediction_analysis.csv", index=False)
log_saved(SUM_DIR / "changed_prediction_analysis.csv")
display(changed_prediction_analysis)

# EXP15 vs EXP17 comparison.
if "d2_exp17_original" in candidate_sources:
    exp15_vs_exp17 = changed_analysis(candidate_sources["d1_exp15_first"], candidate_sources["d2_exp17_original"], "exp17_vs_exp15")
    exp15_vs_exp17.to_csv(SUM_DIR / "exp15_vs_exp17_comparison.csv", index=False)
    log_saved(SUM_DIR / "exp15_vs_exp17_comparison.csv")
    display(exp15_vs_exp17)
else:
    exp15_vs_exp17 = pd.DataFrame([{"variant": "exp17_vs_exp15", "note": "EXP17 candidate not available"}])
    exp15_vs_exp17.to_csv(SUM_DIR / "exp15_vs_exp17_comparison.csv", index=False)

# Strength / leakage artifact compatibility.
if "strength_cols_train" in globals():
    n_strength = len(strength_cols_train)
else:
    n_strength = len([c for c in globals().get("final_model_fe", pd.DataFrame()).columns if str(c).startswith("lk_")]) if "final_model_fe" in globals() else 0
pd.DataFrame([{
    "variant": EXP12D_VARIANT,
    "n_strength_features": n_strength,
    "strength_mode": globals().get("STRENGTH_FEATURE_MODE", "full"),
    "note": "EXP12D preserves EXP22 strength feature construction. See source pipeline cells above.",
}]).to_csv(SUM_DIR / "strength_feature_audit.csv", index=False)
pd.DataFrame([
    {"check": "RUN_GT_AUDIT_false", "passed": bool(not RUN_GT_AUDIT), "detail": "GT audit disabled by default."},
    {"check": "output_root_project_outputs", "passed": bool(str(OUT_DIR.resolve()).startswith(str((OUTPUT_ROOT).resolve()))), "detail": str(OUT_DIR.resolve())},
    {"check": "sample_not_from_outputs", "passed": bool("outputs" not in SAMPLE_SUBMISSION_PATH.parts), "detail": str(SAMPLE_SUBMISSION_PATH)},
]).to_csv(SUM_DIR / "strength_no_leakage_check.csv", index=False)

# Final decision table: do not claim best audit.
recommended = [
    "submission_exp12d_d1_exp15_first.csv",
    "submission_exp12d_d3_bias_calibrated.csv",
    "submission_exp12d_d4_outcome_calibrated.csv",
    "submission_exp12d_d5_agreement_hybrid.csv" if d5_status == "active" else "d5_skipped_no_exp12b_candidate",
    "submission_exp12d_d6_exact_booster.csv",
]
final_decision = {
    "experiment": "EXP12D",
    "variant": EXP12D_VARIANT,
    "selected_by_pipeline": selected_key,
    "selected_by_validation": "not_available_in_notebook_gt_free_only",
    "recommended_for_local_audit": ", ".join(recommended),
    "main_submission_path": str(best_path),
    "project_root": str(PROJECT_ROOT),
    "out_dir": str(OUT_DIR),
    "exp22_reference_awmae": 2.996820,
    "exp12c_exp15_reference_awmae": 3.003729,
    "note": "Notebook does not use GT for selection. Audit candidates locally after CSV generation.",
}
with open(SUM_DIR / "final_decision.json", "w", encoding="utf-8") as f:
    json.dump(final_decision, f, indent=2)
final_decision_df = pd.DataFrame([{"key": k, "value": str(v)} for k, v in final_decision.items()])
final_decision_df.to_csv(SUM_DIR / "final_decision.csv", index=False)
log_saved(SUM_DIR / "final_decision.json")
log_saved(SUM_DIR / "final_decision.csv")
display(final_decision_df)

print("[EXP12D SUMMARY]", flush=True)
log_info(f"EXP12D_VARIANT: {EXP12D_VARIANT}")
log_info(f"selected_key: {selected_key}")
log_info(f"best_safe_path: {best_path}")
log_info(f"OUT_DIR: {OUT_DIR.resolve()}")
log_info(f"mean_pred_total: {float((best_aligned['team_goals'] + best_aligned['opp_goals']).mean()):.4f}")
log_info(f"max_pred_goal: {int(best_aligned[['team_goals', 'opp_goals']].max().max())}")
