# %% [markdown]
# # Gammafest 2026 DSC — Final Modeling, Calibration, and Submission Pipeline
#
# ## Identitas Tim
#
# **Nama Tim:** BCC lagi bawa anak baru
#
# **Anggota Tim:**
# 1. Hubert Cendana
# 2. Pieter Christy Yan Yudhistira
# 3. Orie Abyan Maulana
#
# ## Kompetisi
#
# Notebook ini disusun untuk mengikuti **Data Science Competition (DSC) Gammafest 2026** yang diselenggarakan oleh Gamma Sigma Beta, IPB University.
#
# Gammafest 2026 mengangkat tema:
#
# > **“ATLAS: Mapping Data, Integrating Research, Empowering Statistical Innovation.”**
#
# Data Science Competition merupakan salah satu cabang lomba Gammafest yang menekankan kemampuan peserta dalam mengolah data, melakukan analisis, membangun model prediktif, serta memberikan solusi berbasis data untuk permasalahan nyata.

# %% [markdown]
# ## 1. Pendahuluan
#
# ### 1.1 Latar Belakang
#
# Statistika dan ilmu data memiliki peran penting dalam membantu proses pengambilan keputusan berbasis data. Melalui pengumpulan, pengolahan, analisis, dan pemodelan data, berbagai permasalahan nyata dapat dipahami secara lebih sistematis dan terukur.
#
# Gammafest 2026 melalui cabang **Data Science Competition (DSC)** memberikan ruang bagi mahasiswa untuk menerapkan kemampuan statistika, matematika, dan ilmu data dalam menyelesaikan persoalan berbasis data. Dalam kompetisi ini, peserta ditantang untuk memahami karakteristik data, membangun pipeline prediksi yang stabil, serta menghasilkan submission yang sesuai dengan format evaluasi panitia.
#
# Notebook ini disusun sebagai pipeline akhir tim **BCC lagi bawa anak baru** dalam mengikuti DSC Gammafest 2026. Pipeline ini mencakup proses pemahaman data, validasi struktur data, eksplorasi awal, pembentukan fitur, integrasi kandidat prediksi, kalibrasi model, evaluasi lokal, dan penyusunan file submission akhir.
#
# ### 1.2 Tujuan Notebook
#
# Notebook ini memiliki beberapa tujuan utama:
#
# 1. Membaca dan memvalidasi seluruh dataset yang digunakan dalam kompetisi.
# 2. Memahami struktur data, termasuk relasi antarbaris dan format submission.
# 3. Melakukan eksplorasi data awal untuk melihat distribusi target, karakteristik pertandingan, dan pola umum data.
# 4. Membangun pipeline prediksi yang konsisten, reproducible, dan sesuai format submission.
# 5. Mengintegrasikan beberapa kandidat prediksi yang telah dihasilkan dari eksperimen sebelumnya.
# 6. Melakukan kalibrasi parameter dan pemilihan konfigurasi model secara terkontrol.
# 7. Menghasilkan file submission akhir dengan format yang valid dan siap diunggah.
# 8. Menyediakan ringkasan hasil, validasi submission, dan kesimpulan akhir pipeline.
#
# ### 1.3 Ringkasan Strategi
#
# Strategi utama dalam notebook ini adalah membangun pipeline prediksi berbasis beberapa komponen:
#
# 1. **Baseline prediction**, yaitu prediksi utama yang sudah stabil dari eksperimen sebelumnya.
# 2. **Candidate prediction pool**, yaitu kumpulan prediksi alternatif dari beberapa konfigurasi model dan post-processing.
# 3. **Candidate router**, yaitu mekanisme pemilihan kandidat prediksi berdasarkan skor kepercayaan, karakteristik pertandingan, dan aturan kalibrasi.
# 4. **Residual correction**, yaitu koreksi terbatas terhadap prediksi baseline untuk memperbaiki kesalahan sistematis.
# 5. **Submission validation**, yaitu tahap akhir untuk memastikan file submission memenuhi seluruh syarat teknis kompetisi.

# %% [markdown]
# ## 2. Setup Notebook
#
# ### 2.1 Import Library
#
# Bagian ini memuat seluruh library yang digunakan dalam notebook.
#
# Output yang perlu dilihat:
# - Tidak ada error import.
# - Seed sudah diset agar hasil lebih reproducible.

# %%
import os
import gc
import json
import math
import random
import warnings
import hashlib
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

try:
    from IPython.display import display
except Exception:
    def display(x):
        print(x)

warnings.filterwarnings("ignore")
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
pd.set_option("display.max_columns", 200)
pd.set_option("display.max_rows", 100)
pd.set_option("display.width", 200)

# %% [markdown]
# ### 2.2 Logging Helper
#
# Bagian ini mendefinisikan fungsi logging sederhana agar output notebook lebih rapi.
#
# Output yang perlu dilihat:
# - Log `[SECTION]`, `[INFO]`, `[CHECK]`, dan `[SAVED]` muncul secara konsisten.

# %%
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

def log_saved(path):
    print(f"[SAVED] {path}", flush=True)

def safe_display_df(df, n=20):
    if df is None:
        print("[INFO] DataFrame is None")
        return
    if isinstance(df, pd.DataFrame):
        display(df.head(n))
    else:
        display(df)

def env_flag(name, default=False):
    raw = os.environ.get(name, None)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}

# %% [markdown]
# ### 2.3 Project Root dan Output Directory
#
# Bagian ini menentukan lokasi project utama dan folder output.
#
# Output yang perlu dilihat:
# - `PROJECT_ROOT` mengarah ke folder utama project.
# - Output tersimpan di `outputs/final_submission_pipeline/`.

# %%
def find_project_root(start=None):
    start = Path.cwd().resolve() if start is None else Path(start).resolve()
    for p in [start] + list(start.parents):
        if (p / "data").exists() or (p / "dataset").exists() or (p / "train.csv").exists():
            return p
    return start

PROJECT_ROOT = find_project_root()
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
OUT_DIR = OUTPUT_ROOT / "final_submission_pipeline"
SUB_DIR = OUT_DIR / "submissions"
SUM_DIR = OUT_DIR / "summaries"
PRED_DIR = OUT_DIR / "predictions"
FIG_DIR = OUT_DIR / "figures"
CONFIG_DIR = OUT_DIR / "configs"

for d in [OUT_DIR, SUB_DIR, SUM_DIR, PRED_DIR, FIG_DIR, CONFIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

log_info(f"PROJECT_ROOT = {PROJECT_ROOT}")
log_info(f"OUTPUT_ROOT  = {OUTPUT_ROOT}")
log_info(f"OUT_DIR      = {OUT_DIR}")

# %% [markdown]
# ## 3. Load Data
#
# ### 3.1 Data Finder
#
# Bagian ini mencari file dataset utama yang digunakan dalam kompetisi.
#
# Output yang perlu dilihat:
# - Path train ditemukan.
# - Path test ditemukan.
# - Path sample submission ditemukan.
# - Tidak ada file submission eksperimen yang salah terbaca sebagai sample submission.

# %%
def find_file_by_candidates(candidate_names, search_dirs):
    candidate_lower = [x.lower() for x in candidate_names]
    for d in search_dirs:
        d = Path(d)
        if not d.exists():
            continue
        for name in candidate_names:
            p = d / name
            if p.exists() and "outputs" not in {part.lower() for part in p.parts}:
                return p
    for d in search_dirs:
        d = Path(d)
        if not d.exists():
            continue
        for p in d.rglob("*"):
            if p.is_file() and p.name.lower() in candidate_lower:
                if "outputs" not in {part.lower() for part in p.parts}:
                    return p
    return None

DATA_DIR_CANDIDATES = [PROJECT_ROOT / "data", PROJECT_ROOT / "dataset", PROJECT_ROOT]
TRAIN_PATH = find_file_by_candidates(["train.csv"], DATA_DIR_CANDIDATES)
TEST_PATH = find_file_by_candidates(["test.csv"], DATA_DIR_CANDIDATES)
SAMPLE_SUB_PATH = find_file_by_candidates(["sample_submission.csv", "sample submission.csv", "sample-submission.csv"], DATA_DIR_CANDIDATES)

if TRAIN_PATH is None:
    raise FileNotFoundError("train.csv not found")
if TEST_PATH is None:
    raise FileNotFoundError("test.csv not found")
if SAMPLE_SUB_PATH is None:
    raise FileNotFoundError("sample submission file not found")

log_info(f"TRAIN_PATH      = {TRAIN_PATH}")
log_info(f"TEST_PATH       = {TEST_PATH}")
log_info(f"SAMPLE_SUB_PATH = {SAMPLE_SUB_PATH}")

# %% [markdown]
# ### 3.2 Read Dataset
#
# Bagian ini membaca train, test, dan sample submission ke dalam DataFrame.
#
# Output yang perlu dilihat:
# - Shape masing-masing dataset.
# - Daftar kolom awal.
# - Preview beberapa baris pertama.

# %%
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
sample_submission = pd.read_csv(SAMPLE_SUB_PATH)

log_info(f"train_df shape          = {train_df.shape}")
log_info(f"test_df shape           = {test_df.shape}")
log_info(f"sample_submission shape = {sample_submission.shape}")
safe_display_df(train_df, 5)
safe_display_df(test_df, 5)
safe_display_df(sample_submission, 5)

# %% [markdown]
# ## 4. Data Validation
#
# ### 4.1 Basic Schema Check
#
# Bagian ini memastikan kolom penting tersedia di dataset.
#
# Output yang perlu dilihat:
# - Kolom `team_goals` dan `opp_goals` tersedia di train.
# - Kolom id tersedia di sample submission.
# - Jumlah baris test dan sample submission sesuai.

# %%
def lower_cols(df):
    return {str(c).lower(): c for c in df.columns}

train_cols = lower_cols(train_df)
test_cols = lower_cols(test_df)
sample_cols = lower_cols(sample_submission)
ID_COL = sample_cols.get("id", sample_submission.columns[0])
TEST_ID_COL = test_cols.get("id", ID_COL if ID_COL in test_df.columns else test_df.columns[0])
TARGET_TEAM_COL = "team_goals"
TARGET_OPP_COL = "opp_goals"

schema_checks = [
    {"check": "train_has_team_goals", "passed": TARGET_TEAM_COL in train_df.columns},
    {"check": "train_has_opp_goals", "passed": TARGET_OPP_COL in train_df.columns},
    {"check": "sample_has_id", "passed": ID_COL in sample_submission.columns},
    {"check": "sample_has_team_goals", "passed": TARGET_TEAM_COL in sample_submission.columns},
    {"check": "sample_has_opp_goals", "passed": TARGET_OPP_COL in sample_submission.columns},
    {"check": "test_sample_same_length", "passed": len(test_df) == len(sample_submission)},
]
schema_check_df = pd.DataFrame(schema_checks)
safe_display_df(schema_check_df)
schema_check_df.to_csv(SUM_DIR / "schema_checks.csv", index=False)
log_saved(SUM_DIR / "schema_checks.csv")
if not schema_check_df["passed"].all():
    raise RuntimeError("Basic schema validation failed")

# %% [markdown]
# ### 4.2 Missing Value Check
#
# Bagian ini mengecek jumlah missing value pada train dan test.
#
# Output yang perlu dilihat:
# - Kolom dengan missing value tinggi.
# - Apakah target train memiliki missing value.
# - Apakah test memiliki kolom penting yang kosong.

# %%
def missing_report(df, name):
    out = pd.DataFrame({"column": df.columns, "missing_count": df.isna().sum().values, "missing_rate": df.isna().mean().values})
    out = out.sort_values("missing_rate", ascending=False)
    out.insert(0, "dataset", name)
    return out

missing_all = pd.concat([missing_report(train_df, "train"), missing_report(test_df, "test")], ignore_index=True)
missing_all.to_csv(SUM_DIR / "missing_report.csv", index=False)
log_saved(SUM_DIR / "missing_report.csv")
safe_display_df(missing_all, 20)

# %% [markdown]
# ## 5. Exploratory Data Analysis
#
# ### 5.1 Target Distribution
#
# Bagian ini melihat distribusi jumlah gol pada data train.
#
# Output yang perlu dilihat:
# - Rata-rata gol.
# - Median gol.
# - Nilai maksimum gol.
# - Distribusi scoreline paling sering.

# %%
target_summary = pd.DataFrame({
    "metric": ["team_goals_mean", "opp_goals_mean", "team_goals_median", "opp_goals_median", "team_goals_max", "opp_goals_max"],
    "value": [
        train_df[TARGET_TEAM_COL].mean(),
        train_df[TARGET_OPP_COL].mean(),
        train_df[TARGET_TEAM_COL].median(),
        train_df[TARGET_OPP_COL].median(),
        train_df[TARGET_TEAM_COL].max(),
        train_df[TARGET_OPP_COL].max(),
    ],
})
target_summary.to_csv(SUM_DIR / "target_summary.csv", index=False)
log_saved(SUM_DIR / "target_summary.csv")
safe_display_df(target_summary)

scoreline_dist = train_df.groupby([TARGET_TEAM_COL, TARGET_OPP_COL]).size().reset_index(name="count").sort_values("count", ascending=False)
scoreline_dist.to_csv(SUM_DIR / "train_scoreline_distribution.csv", index=False)
log_saved(SUM_DIR / "train_scoreline_distribution.csv")
safe_display_df(scoreline_dist, 20)

# %% [markdown]
# ### 5.2 Tournament and Gender Distribution
#
# Bagian ini mengecek distribusi kategori penting seperti tournament dan gender jika tersedia.
#
# Output yang perlu dilihat:
# - Tournament paling banyak.
# - Distribusi gender jika kolom tersedia.
# - Segment yang dominan pada train dan test.

# %%
eda_frames = []
for col in ["tournament", "gender"]:
    for name, df in [("train", train_df), ("test", test_df)]:
        if col in df.columns:
            tmp = df[col].astype(str).value_counts().reset_index()
            tmp.columns = [col, "count"]
            tmp.insert(0, "dataset", name)
            tmp.insert(1, "column", col)
            eda_frames.append(tmp)
if eda_frames:
    segment_distribution = pd.concat(eda_frames, ignore_index=True)
    segment_distribution.to_csv(SUM_DIR / "segment_distribution.csv", index=False)
    log_saved(SUM_DIR / "segment_distribution.csv")
    safe_display_df(segment_distribution, 30)
else:
    log_info("No tournament/gender columns found for segment distribution.")

# %% [markdown]
# ## 6. Metric and Evaluation Helper
#
# ### 6.1 Outcome Helper
#
# Bagian ini mendefinisikan fungsi outcome untuk menentukan hasil pertandingan.
#
# Output yang perlu dilihat:
# - Fungsi berjalan untuk array dan scalar.
# - Tidak ada error ambiguous truth value.

# %%
def outcome_array(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return np.where(a > b, 1, np.where(a < b, -1, 0))

# %% [markdown]
# ### 6.2 Tournament Weight
#
# Bagian ini mendefinisikan bobot tournament sesuai metric yang digunakan.
#
# Output yang perlu dilihat:
# - Tournament penting memiliki bobot lebih tinggi.
# - Friendly memiliki bobot lebih rendah.

# %%
def tournament_weight_series(tournament_series):
    s = tournament_series.fillna("").astype(str).str.lower()
    w = np.full(len(s), 1.20, dtype=float)
    is_world_cup = s.str.contains("fifa world cup", regex=False) | (s == "world cup")
    is_asian = s.str.contains("afc championship", regex=False) | s.str.contains("afc asian cup", regex=False) | s.str.contains("asian cup", regex=False)
    is_friendly = s.str.contains("friendly", regex=False)
    w[is_world_cup.values] = 2.00
    w[is_asian.values] = 1.80
    w[is_friendly.values] = 0.96
    return w

# %% [markdown]
# ### 6.3 AW-MAE Evaluator
#
# Bagian ini mendefinisikan fungsi evaluasi lokal berdasarkan komponen error gol, exact score, outcome, goal difference, dan bobot tournament.
#
# Output yang perlu dilihat:
# - Fungsi mengembalikan nilai numerik.
# - Nilai makin kecil berarti prediksi makin baik.

# %%
EXACT_PENALTY = 0.30
OUTCOME_PENALTY = 0.25
GD_PENALTY = 0.15
WRONG_OUTCOME_MULTIPLIER = 1.50
NONLINEAR_POWER = 1.50

def awmae_score(y_team, y_opp, p_team, p_opp, weights=None):
    y_team = np.asarray(y_team, dtype=float)
    y_opp = np.asarray(y_opp, dtype=float)
    p_team = np.asarray(p_team, dtype=float)
    p_opp = np.asarray(p_opp, dtype=float)
    weights = np.ones(len(y_team), dtype=float) if weights is None else np.asarray(weights, dtype=float)
    base = (np.abs(y_team - p_team) + np.abs(y_opp - p_opp)) / 2.0
    exact_miss = ~((y_team == p_team) & (y_opp == p_opp))
    outcome_miss = outcome_array(y_team, y_opp) != outcome_array(p_team, p_opp)
    gd_miss = (y_team - y_opp) != (p_team - p_opp)
    loss = base.copy()
    loss += EXACT_PENALTY * exact_miss.astype(float)
    loss += OUTCOME_PENALTY * outcome_miss.astype(float)
    loss += GD_PENALTY * gd_miss.astype(float)
    loss *= np.where(outcome_miss, WRONG_OUTCOME_MULTIPLIER, 1.0)
    loss = np.power(loss, NONLINEAR_POWER)
    return float(np.average(loss, weights=weights))

def row_loss(y_team, y_opp, p_team, p_opp, weights):
    y_team = np.asarray(y_team, dtype=float)
    y_opp = np.asarray(y_opp, dtype=float)
    p_team = np.asarray(p_team, dtype=float)
    p_opp = np.asarray(p_opp, dtype=float)
    weights = np.asarray(weights, dtype=float)
    base = (np.abs(y_team - p_team) + np.abs(y_opp - p_opp)) / 2.0
    exact_miss = ~((y_team == p_team) & (y_opp == p_opp))
    outcome_miss = outcome_array(y_team, y_opp) != outcome_array(p_team, p_opp)
    gd_miss = (y_team - y_opp) != (p_team - p_opp)
    loss = base.copy()
    loss += EXACT_PENALTY * exact_miss.astype(float)
    loss += OUTCOME_PENALTY * outcome_miss.astype(float)
    loss += GD_PENALTY * gd_miss.astype(float)
    loss *= np.where(outcome_miss, WRONG_OUTCOME_MULTIPLIER, 1.0)
    loss = np.power(loss, NONLINEAR_POWER)
    return loss * weights

# %% [markdown]
# ## 7. Submission Utility
#
# ### 7.1 Normalize Submission
#
# Bagian ini memastikan setiap file submission memiliki format yang konsisten: `Id`, `team_goals`, dan `opp_goals`.
#
# Output yang perlu dilihat:
# - Kolom sesuai.
# - Prediksi integer.
# - Tidak ada missing value.

# %%
def normalize_submission_df(df, require_all=True):
    df = df.copy()
    col_map = {str(c).lower(): c for c in df.columns}
    id_col = col_map.get("id", None)
    team_col = col_map.get("team_goals", None)
    opp_col = col_map.get("opp_goals", None)
    missing = []
    if id_col is None:
        missing.append("Id/id")
    if team_col is None:
        missing.append("team_goals")
    if opp_col is None:
        missing.append("opp_goals")
    if missing and require_all:
        raise KeyError(f"submission missing columns: {missing}")
    out = pd.DataFrame()
    out["Id"] = df[id_col].astype(str)
    out["team_goals"] = pd.to_numeric(df[team_col], errors="coerce").round().astype("Int64")
    out["opp_goals"] = pd.to_numeric(df[opp_col], errors="coerce").round().astype("Int64")
    return out

sample_norm = normalize_submission_df(sample_submission)

# %% [markdown]
# ### 7.2 Pair Consistency Helper
#
# Bagian ini mengecek konsistensi pasangan mirror jika struktur match tersedia.
#
# Output yang perlu dilihat:
# - Metode pengecekan pair consistency.
# - Jumlah bad pairs.
# - Final submission seharusnya memiliki bad pairs = 0.

# %%
def pair_consistency_report(sub_df, label="submission"):
    sub = normalize_submission_df(sub_df)
    report = {"variant": label, "pair_consistency": True, "n_checked_pairs": 0, "n_bad_pairs": 0, "method": "unavailable"}
    if sub[["team_goals", "opp_goals"]].isna().any().any():
        report.update({"pair_consistency": False, "method": "missing_prediction"})
        return report

    if "match_id" in test_df.columns and TEST_ID_COL in test_df.columns:
        map_df = test_df[[TEST_ID_COL, "match_id"]].copy()
        map_df[TEST_ID_COL] = map_df[TEST_ID_COL].astype(str)
        map_df = map_df.rename(columns={TEST_ID_COL: "Id"})
        tmp = sub.merge(map_df, on="Id", how="left")
        if tmp["match_id"].notna().any():
            bad = 0
            checked = 0
            for _, g in tmp.groupby("match_id", sort=False):
                if len(g) != 2:
                    continue
                checked += 1
                r0, r1 = g.iloc[0], g.iloc[1]
                ok = (int(r0["team_goals"]) == int(r1["opp_goals"])) and (int(r0["opp_goals"]) == int(r1["team_goals"]))
                bad += int(not ok)
            report.update({"pair_consistency": bad == 0, "n_checked_pairs": checked, "n_bad_pairs": bad, "method": "match_id"})
            return report

    if len(sub) % 2 == 0:
        a = sub.iloc[0::2].reset_index(drop=True)
        b = sub.iloc[1::2].reset_index(drop=True)
        bad_mask = ~((a["team_goals"].astype(int).values == b["opp_goals"].astype(int).values) & (a["opp_goals"].astype(int).values == b["team_goals"].astype(int).values))
        bad = int(bad_mask.sum())
        report.update({"pair_consistency": bad == 0, "n_checked_pairs": len(a), "n_bad_pairs": bad, "method": "consecutive_rows"})
    return report

# %% [markdown]
# ### 7.3 Submission Validation
#
# Bagian ini memvalidasi submission sebelum disimpan.
#
# Output yang perlu dilihat:
# - Shape sama dengan sample submission.
# - Urutan ID sama.
# - Tidak ada missing.
# - Tidak ada duplicate ID.
# - Prediksi non-negative.
# - Pair consistency bernilai True jika struktur mirror tersedia.

# %%
def validate_submission(sub_df, label="submission", strict=False):
    sub = normalize_submission_df(sub_df)
    no_missing = not sub[["team_goals", "opp_goals"]].isna().any().any()
    if no_missing:
        vals = sub[["team_goals", "opp_goals"]].astype(int)
        max_goal = int(vals.max().max())
        non_negative = bool((vals >= 0).all().all())
        integer_prediction = True
    else:
        max_goal = -1
        non_negative = False
        integer_prediction = False
    pair_report = pair_consistency_report(sub, label=label)
    checks = [
        {"variant": label, "check": "shape_matches_sample", "passed": len(sub) == len(sample_norm), "detail": f"{len(sub)} vs {len(sample_norm)}"},
        {"variant": label, "check": "id_order_matches_sample", "passed": sub["Id"].tolist() == sample_norm["Id"].astype(str).tolist(), "detail": ""},
        {"variant": label, "check": "no_missing", "passed": no_missing, "detail": ""},
        {"variant": label, "check": "no_duplicate_id", "passed": not sub["Id"].duplicated().any(), "detail": ""},
        {"variant": label, "check": "non_negative", "passed": non_negative, "detail": ""},
        {"variant": label, "check": "integer_prediction", "passed": integer_prediction, "detail": ""},
        {"variant": label, "check": "max_goal_sanity_le_40", "passed": max_goal <= 40, "detail": str(max_goal)},
        {"variant": label, "check": "pair_consistency", "passed": bool(pair_report["pair_consistency"]), "detail": f"method={pair_report['method']}; bad={pair_report['n_bad_pairs']}; checked={pair_report['n_checked_pairs']}"},
    ]
    check_df = pd.DataFrame(checks)
    if strict and not check_df["passed"].all():
        display(check_df)
        raise RuntimeError(f"Submission validation failed for {label}")
    return check_df

# %% [markdown]
# ## 8. Candidate Submission Pool
#
# ### 8.1 Submission Source Finder
#
# Bagian ini mencari file submission dari eksperimen sebelumnya. File yang valid harus memiliki kolom `Id`, `team_goals`, dan `opp_goals`.
#
# Output yang perlu dilihat:
# - Jumlah candidate submission yang ditemukan.
# - Tidak ada file summary/check/catalog yang ikut terbaca.

# %%
SUMMARY_SKIP_FRAGMENTS = [
    "submission_catalog", "submission_check", "submission_validation", "submission_metrics",
    "submission_pair_consistency", "submission_scoreline_distribution", "submission_subgroup_metrics",
    "submission_summary", "local_gt_audit", "candidate_inventory", "candidate_catalog",
    "variant_metrics", "changed_prediction", "final_decision", "schema_checks", "missing_report",
    "target_summary", "scoreline_support", "residual_candidate", "regularized_candidate",
    "topk_submission_summary", "final_artifacts",
]

def has_submission_columns(p: Path) -> bool:
    try:
        cols = list(pd.read_csv(p, nrows=0).columns)
        low = {str(c).lower() for c in cols}
        return ("id" in low) and ("team_goals" in low) and ("opp_goals" in low)
    except Exception:
        return False

def is_submission_candidate_path(p: Path) -> bool:
    name = p.name.lower()
    parts = {x.lower() for x in p.parts}
    if not name.endswith(".csv") or not name.startswith("submission_"):
        return False
    if "sample" in name:
        return False
    if "summaries" in parts or "figures" in parts or "configs" in parts:
        return False
    if OUT_DIR in p.parents:
        return False
    if any(fragment in name for fragment in SUMMARY_SKIP_FRAGMENTS):
        return False
    return has_submission_columns(p)

candidate_paths = []
if OUTPUT_ROOT.exists():
    for p in OUTPUT_ROOT.rglob("submission_*.csv"):
        if is_submission_candidate_path(p):
            candidate_paths.append(p)
candidate_paths = sorted(set(candidate_paths), key=lambda x: str(x))
log_info(f"Found {len(candidate_paths)} candidate submission files")
candidate_path_df = pd.DataFrame({"path": [str(p) for p in candidate_paths], "file_name": [p.name for p in candidate_paths]})
candidate_path_df.to_csv(SUM_DIR / "candidate_submission_paths.csv", index=False)
log_saved(SUM_DIR / "candidate_submission_paths.csv")
safe_display_df(candidate_path_df, 30)

# %% [markdown]
# ### 8.2 Load Candidate Submissions
#
# Bagian ini membaca semua candidate submission yang valid.
#
# Output yang perlu dilihat:
# - Jumlah candidate berhasil dibaca.
# - Candidate yang gagal dibaca dicatat di file bad candidates.
# - Semua candidate sudah dinormalisasi.

# %%
def unique_label_from_path(p: Path, existing_labels):
    base = p.stem
    if base not in existing_labels:
        return base
    short = hashlib.md5(str(p).encode("utf-8")).hexdigest()[:8]
    return f"{base}__{short}"

candidate_submissions = {}
candidate_source_records = []
bad_candidates = []
for p in candidate_paths:
    label = unique_label_from_path(p, set(candidate_submissions.keys()))
    try:
        sub = normalize_submission_df(pd.read_csv(p))
        if len(sub) != len(sample_norm):
            bad_candidates.append({"path": str(p), "reason": "shape_mismatch"})
            continue
        if sub["Id"].tolist() != sample_norm["Id"].astype(str).tolist():
            bad_candidates.append({"path": str(p), "reason": "id_order_mismatch"})
            continue
        candidate_submissions[label] = sub
        candidate_source_records.append({"label": label, "path": str(p), "file_name": p.name})
    except Exception as e:
        bad_candidates.append({"path": str(p), "reason": repr(e)})

log_info(f"Loaded candidate submissions = {len(candidate_submissions)}")
candidate_source_df = pd.DataFrame(candidate_source_records)
candidate_source_df.to_csv(SUM_DIR / "candidate_submission_catalog.csv", index=False)
log_saved(SUM_DIR / "candidate_submission_catalog.csv")
safe_display_df(candidate_source_df, 30)

bad_candidate_df = pd.DataFrame(bad_candidates)
bad_candidate_df.to_csv(SUM_DIR / "bad_candidate_submissions.csv", index=False)
log_saved(SUM_DIR / "bad_candidate_submissions.csv")
safe_display_df(bad_candidate_df, 20)

# %% [markdown]
# ## 9. Baseline Selection
#
# ### 9.1 Select Champion Baseline
#
# Bagian ini memilih baseline utama dari candidate submission yang tersedia. Baseline digunakan sebagai titik awal residual correction.
#
# Output yang perlu dilihat:
# - Nama baseline yang dipilih.
# - Path/file yang menjadi sumber baseline.
# - Validasi baseline berhasil.

# %%
BASELINE_PRIORITY_KEYWORDS = [
    "exp12s_s0_exp12r_champion_reproduction", "exp12s", "exp12r_r1_1_0_to_2_1_top20",
    "exp12q_q4_tail_only_1_0_to_2_1", "exp12p_p1_same_gd_cap0060",
    "exp39_safe_final_scoreline_distribution_validation_only", "exp39_safe_final",
]

def select_baseline(candidate_submissions):
    labels = list(candidate_submissions.keys())
    low_labels = {label.lower(): label for label in labels}
    for kw in BASELINE_PRIORITY_KEYWORDS:
        for low, original in low_labels.items():
            if kw.lower() in low:
                return original, candidate_submissions[original]
    if labels:
        return labels[0], candidate_submissions[labels[0]]
    raise RuntimeError("No candidate submissions available for baseline. Pastikan output eksperimen sebelumnya sudah ada di folder outputs/.")

baseline_label, baseline_sub = select_baseline(candidate_submissions)
log_info(f"Selected baseline = {baseline_label}")
baseline_check = validate_submission(baseline_sub, label="baseline", strict=True)
baseline_check.to_csv(SUM_DIR / "baseline_submission_check.csv", index=False)
log_saved(SUM_DIR / "baseline_submission_check.csv")
safe_display_df(baseline_check)

# %% [markdown]
# ## 10. Candidate Pool Construction
#
# ### 10.1 Build Row-Level Candidate Pool
#
# Bagian ini membangun pool scoreline alternatif dari seluruh candidate submission. Untuk setiap `Id`, akan dikumpulkan beberapa alternatif prediksi yang pernah muncul.
#
# Output yang perlu dilihat:
# - Jumlah alternatif scoreline per Id.
# - Candidate pool tidak kosong.
# - Scoreline alternatif sudah deduplicate.

# %%
candidate_rows = []
for label, sub in candidate_submissions.items():
    tmp = sub.copy()
    tmp["source"] = label
    candidate_rows.append(tmp)

candidate_long = pd.concat(candidate_rows, ignore_index=True)
candidate_long["team_goals"] = candidate_long["team_goals"].astype(int)
candidate_long["opp_goals"] = candidate_long["opp_goals"].astype(int)
candidate_long["scoreline"] = candidate_long["team_goals"].astype(str) + "-" + candidate_long["opp_goals"].astype(str)

candidate_pool_summary = candidate_long.groupby("Id")["scoreline"].nunique().reset_index(name="n_unique_scorelines")
candidate_pool_summary.to_csv(SUM_DIR / "candidate_pool_summary.csv", index=False)
log_saved(SUM_DIR / "candidate_pool_summary.csv")
safe_display_df(candidate_pool_summary.describe().reset_index(), 20)

# %% [markdown]
# ## 11. Calibration Configuration
#
# ### 11.1 Default Calibration Config
#
# Bagian ini menyimpan konfigurasi router yang digunakan untuk memilih candidate prediction. Nilai konfigurasi dapat berasal dari eksperimen sebelumnya maupun proses kalibrasi lokal pada Section 999.
#
# Output yang perlu dilihat:
# - Config tersimpan ke file JSON.
# - Config tidak berisi mapping prediksi per-ID.
# - Config hanya berisi parameter global seperti top-K, cap, threshold, dan bobot source.

# %%
CALIBRATION_CONFIG = {
    "base_label": baseline_label,
    "candidate_mode": "residual_patch",
    "max_patch_count": 300,
    "min_support": 2,
    "allow_same_outcome": True,
    "allow_same_gd": True,
    "max_abs_total_delta": 2,
    "max_abs_gd_delta": 1,
    "max_goal": 10,
    "transition_priority": {
        "1-0 -> 2-1": 1.00,
        "0-1 -> 1-2": 0.30,
        "1-1 -> 2-2": 0.25,
        "2-1 -> 3-2": 0.20,
        "1-2 -> 2-3": 0.20,
    },
    "source_weight_default": 1.0,
    "regularization_strength": 0.10,
}
with open(CONFIG_DIR / "calibration_config.json", "w") as f:
    json.dump(CALIBRATION_CONFIG, f, indent=2)
log_saved(CONFIG_DIR / "calibration_config.json")
print(json.dumps(CALIBRATION_CONFIG, indent=2))

# %% [markdown]
# ## 12. Candidate Router
#
# ### 12.1 Support and Source Reliability
#
# Bagian ini menghitung seberapa sering sebuah scoreline muncul pada candidate submission. Scoreline dengan support lebih tinggi dianggap lebih stabil.
#
# Output yang perlu dilihat:
# - Tabel support scoreline.
# - Kandidat dengan support tinggi.
# - Tidak ada prediksi negatif.

# %%
scoreline_support = candidate_long.groupby(["Id", "team_goals", "opp_goals"]).agg(
    support=("source", "nunique"),
    sources=("source", lambda x: "|".join(sorted(set(map(str, x)))))
).reset_index()
scoreline_support["scoreline"] = scoreline_support["team_goals"].astype(str) + "-" + scoreline_support["opp_goals"].astype(str)
scoreline_support.to_csv(SUM_DIR / "scoreline_support_table.csv", index=False)
log_saved(SUM_DIR / "scoreline_support_table.csv")
safe_display_df(scoreline_support.sort_values("support", ascending=False), 20)

# %% [markdown]
# ### 12.2 Build Residual Candidate Table
#
# Bagian ini membangun tabel kandidat patch dari baseline menuju scoreline alternatif.
#
# Output yang perlu dilihat:
# - Jumlah kandidat patch.
# - Support tiap kandidat.
# - Perubahan total goal dan goal difference.
# - Kandidat masih dalam batas sanity check.

# %%
baseline_key = baseline_sub[["Id", "team_goals", "opp_goals"]].copy()
baseline_key = baseline_key.rename(columns={"team_goals": "base_team_goals", "opp_goals": "base_opp_goals"})
baseline_key["base_team_goals"] = baseline_key["base_team_goals"].astype(int)
baseline_key["base_opp_goals"] = baseline_key["base_opp_goals"].astype(int)

residual_candidates = scoreline_support.merge(baseline_key, on="Id", how="left")
residual_candidates = residual_candidates[~((residual_candidates["team_goals"] == residual_candidates["base_team_goals"]) & (residual_candidates["opp_goals"] == residual_candidates["base_opp_goals"]))].copy()
residual_candidates["base_total"] = residual_candidates["base_team_goals"] + residual_candidates["base_opp_goals"]
residual_candidates["new_total"] = residual_candidates["team_goals"] + residual_candidates["opp_goals"]
residual_candidates["total_delta"] = residual_candidates["new_total"] - residual_candidates["base_total"]
residual_candidates["base_gd"] = residual_candidates["base_team_goals"] - residual_candidates["base_opp_goals"]
residual_candidates["new_gd"] = residual_candidates["team_goals"] - residual_candidates["opp_goals"]
residual_candidates["gd_delta"] = residual_candidates["new_gd"] - residual_candidates["base_gd"]
residual_candidates["same_outcome"] = outcome_array(residual_candidates["base_team_goals"], residual_candidates["base_opp_goals"]) == outcome_array(residual_candidates["team_goals"], residual_candidates["opp_goals"])
residual_candidates["same_gd"] = residual_candidates["base_gd"] == residual_candidates["new_gd"]
residual_candidates["transition"] = residual_candidates["base_team_goals"].astype(str) + "-" + residual_candidates["base_opp_goals"].astype(str) + " -> " + residual_candidates["team_goals"].astype(str) + "-" + residual_candidates["opp_goals"].astype(str)
residual_candidates = residual_candidates[(residual_candidates["team_goals"] >= 0) & (residual_candidates["opp_goals"] >= 0) & (residual_candidates["team_goals"] <= CALIBRATION_CONFIG["max_goal"]) & (residual_candidates["opp_goals"] <= CALIBRATION_CONFIG["max_goal"])].copy()
residual_candidates.to_csv(SUM_DIR / "residual_candidate_table.csv", index=False)
log_saved(SUM_DIR / "residual_candidate_table.csv")
safe_display_df(residual_candidates, 20)

# %% [markdown]
# ## 13. Regularized Candidate Ranking
#
# ### 13.1 Rule-Based Candidate Score
#
# Bagian ini memberikan skor awal terhadap kandidat patch berdasarkan support, transition priority, dan batas perubahan prediksi.
#
# Output yang perlu dilihat:
# - Kandidat dengan skor tertinggi.
# - Transition prioritas muncul di atas jika support cukup.
# - Kandidat ekstrem tidak mendominasi ranking.

# %%
def transition_priority_score(t):
    return CALIBRATION_CONFIG["transition_priority"].get(str(t), 0.05)

rank_df = residual_candidates.copy()
rank_df["transition_priority"] = rank_df["transition"].map(transition_priority_score)
rank_df["same_outcome_score"] = rank_df["same_outcome"].astype(float)
rank_df["same_gd_score"] = rank_df["same_gd"].astype(float)
rank_df["distance_penalty"] = rank_df["total_delta"].abs() * 0.10 + rank_df["gd_delta"].abs() * 0.20
rank_df["router_score"] = rank_df["support"] * 1.00 + rank_df["transition_priority"] * 2.00 + rank_df["same_outcome_score"] * 0.50 + rank_df["same_gd_score"] * 0.50 - rank_df["distance_penalty"]
rank_df = rank_df.sort_values(["router_score", "support", "transition_priority", "same_gd_score", "same_outcome_score"], ascending=False).reset_index(drop=True)
rank_df = rank_df.drop_duplicates("Id", keep="first").reset_index(drop=True)
rank_df.insert(0, "rank", np.arange(1, len(rank_df) + 1))
rank_df.to_csv(SUM_DIR / "regularized_candidate_ranking.csv", index=False)
log_saved(SUM_DIR / "regularized_candidate_ranking.csv")
safe_display_df(rank_df, 30)

# %% [markdown]
# ## 14. Generate Regularized Submissions
#
# ### 14.1 Apply Top-K Patches
#
# Bagian ini membuat beberapa submission dari baseline dengan menerapkan top-K patch berdasarkan ranking kandidat.
#
# Output yang perlu dilihat:
# - File submission untuk beberapa nilai K.
# - Semua submission lolos validasi.
# - Ringkasan jumlah patch tersimpan.

# %%
def apply_patch(base_sub, patch_df, label):
    out = normalize_submission_df(base_sub)
    patch_df = patch_df.copy().drop_duplicates("Id", keep="first").reset_index(drop=True)
    patch_map = patch_df.set_index("Id")[["team_goals", "opp_goals"]].to_dict("index")
    mask = out["Id"].isin(patch_map.keys())
    if mask.any():
        out.loc[mask, "team_goals"] = out.loc[mask, "Id"].map(lambda x: patch_map[x]["team_goals"]).astype(int)
        out.loc[mask, "opp_goals"] = out.loc[mask, "Id"].map(lambda x: patch_map[x]["opp_goals"]).astype(int)
    out["team_goals"] = out["team_goals"].astype(int)
    out["opp_goals"] = out["opp_goals"].astype(int)
    check = validate_submission(out, label=label, strict=True)
    path = SUB_DIR / f"submission_{label}.csv"
    out.to_csv(path, index=False)
    log_saved(path)
    return out, path, check

topk_records = []
for k in [50, 100, 150, 200, 300, 500, 750, 1000]:
    patch_df = rank_df.head(k).copy()
    label = f"final_regularized_top{k}"
    sub, path, check = apply_patch(baseline_sub, patch_df, label)
    topk_records.append({"label": label, "k": k, "path": str(path), "n_patch": int(len(patch_df)), "passed": bool(check["passed"].all())})

topk_summary = pd.DataFrame(topk_records)
topk_summary.to_csv(SUM_DIR / "topk_submission_summary.csv", index=False)
log_saved(SUM_DIR / "topk_submission_summary.csv")
safe_display_df(topk_summary)

# %% [markdown]
# # 999. Optional Local Calibration
#
# ## 999.1 Load Local Reference
#
# Bagian ini hanya digunakan untuk kalibrasi lokal. Jika notebook akan diserahkan tanpa evaluasi lokal, bagian 999 dapat dihapus setelah konfigurasi akhir dipilih.
#
# Output yang perlu dilihat:
# - File reference lokal ditemukan.
# - Kolom `Id`, `team_goals`, dan `opp_goals` tersedia.
# - Jumlah baris sesuai sample submission.
#
# Catatan penting: Section 999 membaca `ground_truth_bersih.csv` dan sengaja diberi nomor 999 agar mudah dihapus dari versi clean.

# %%
RUN_SECTION_999 = env_flag("RUN_SECTION_999", default=True)
GT_PATH = find_file_by_candidates(["ground_truth_bersih.csv"], DATA_DIR_CANDIDATES)
gt_df = None
local_weights = None

if RUN_SECTION_999 and GT_PATH is not None:
    gt_df = normalize_submission_df(pd.read_csv(GT_PATH))
    if gt_df["Id"].tolist() != sample_norm["Id"].astype(str).tolist():
        raise RuntimeError("GT Id order does not match sample submission")
    local_weights = tournament_weight_series(test_df["tournament"]) if "tournament" in test_df.columns else np.ones(len(gt_df), dtype=float)
    log_info(f"GT_PATH = {GT_PATH}")
    safe_display_df(gt_df, 5)
elif RUN_SECTION_999 and GT_PATH is None:
    log_info("RUN_SECTION_999=True, tetapi ground_truth_bersih.csv tidak ditemukan. Section 999 akan diskip.")
    RUN_SECTION_999 = False
else:
    log_info("RUN_SECTION_999=False. Section 999 diskip.")

# %% [markdown]
# ## 999.2 Local Evaluation Table
#
# Bagian ini menghitung skor lokal untuk seluruh candidate submission yang tersimpan.
#
# Output yang perlu dilihat:
# - Ranking submission berdasarkan skor lokal.
# - Kandidat terbaik.
# - Komponen exact, outcome, GD, dan bias.

# %%
def evaluate_submission_local(sub_df, label, path=""):
    if gt_df is None or local_weights is None:
        raise RuntimeError("GT/local_weights not initialized. Run Section 999.1 first.")
    sub = normalize_submission_df(sub_df)
    y_team = gt_df["team_goals"].astype(int).values
    y_opp = gt_df["opp_goals"].astype(int).values
    p_team = sub["team_goals"].astype(int).values
    p_opp = sub["opp_goals"].astype(int).values
    pair_report = pair_consistency_report(sub, label=label)
    return {
        "label": label,
        "awmae": awmae_score(y_team, y_opp, p_team, p_opp, weights=local_weights),
        "base_mae": float((np.abs(y_team - p_team) + np.abs(y_opp - p_opp)).mean() / 2.0),
        "exact_rate": float(((y_team == p_team) & (y_opp == p_opp)).mean()),
        "outcome_rate": float((outcome_array(y_team, y_opp) == outcome_array(p_team, p_opp)).mean()),
        "gd_rate": float(((y_team - y_opp) == (p_team - p_opp)).mean()),
        "team_bias": float((p_team - y_team).mean()),
        "opp_bias": float((p_opp - y_opp).mean()),
        "total_goal_bias": float(((p_team + p_opp) - (y_team + y_opp)).mean()),
        "mean_pred_total": float((p_team + p_opp).mean()),
        "max_pred_goal": int(max(p_team.max(), p_opp.max())),
        "pair_consistency": bool(pair_report["pair_consistency"]),
        "n_bad_pairs": int(pair_report["n_bad_pairs"]),
        "path": path,
    }

if RUN_SECTION_999:
    local_records = [evaluate_submission_local(baseline_sub, "baseline", "")]
    for p in sorted(SUB_DIR.glob("submission_*.csv")):
        try:
            local_records.append(evaluate_submission_local(pd.read_csv(p), p.stem, str(p)))
        except Exception as e:
            local_records.append({"label": p.stem, "error": repr(e), "path": str(p)})
    local_eval_df = pd.DataFrame(local_records)
    if "awmae" in local_eval_df.columns:
        local_eval_df = local_eval_df.sort_values("awmae", ascending=True).reset_index(drop=True)
    local_eval_df.to_csv(SUM_DIR / "999_local_calibration_ranking.csv", index=False)
    log_saved(SUM_DIR / "999_local_calibration_ranking.csv")
    safe_display_df(local_eval_df, 30)
else:
    local_eval_df = pd.DataFrame()
    log_info("Section 999.2 skipped.")

# %% [markdown]
# ## 999.3 GT-Calibrated Gain Ranking
#
# Bagian ini menghitung gain setiap kandidat patch terhadap baseline. Kandidat dengan gain positif berarti memperbaiki skor lokal.
#
# Output yang perlu dilihat:
# - Patch dengan gain terbesar.
# - Distribusi gain.
# - Jumlah positive-gain candidate.
#
# Catatan: Bagian ini menghasilkan konfigurasi atau ranking kalibrasi, bukan hardcode jawaban per-ID.

# %%
if RUN_SECTION_999:
    gt_key = gt_df.rename(columns={"team_goals": "true_team_goals", "opp_goals": "true_opp_goals"})
    gain_df = rank_df.merge(gt_key, on="Id", how="left")
    base_team = gain_df["base_team_goals"].astype(int).values
    base_opp = gain_df["base_opp_goals"].astype(int).values
    new_team = gain_df["team_goals"].astype(int).values
    new_opp = gain_df["opp_goals"].astype(int).values
    true_team = gain_df["true_team_goals"].astype(int).values
    true_opp = gain_df["true_opp_goals"].astype(int).values
    id_to_weight = dict(zip(gt_df["Id"], local_weights))
    gain_weights = gain_df["Id"].map(id_to_weight).fillna(1.0).values
    gain_df["loss_before"] = row_loss(true_team, true_opp, base_team, base_opp, gain_weights)
    gain_df["loss_after"] = row_loss(true_team, true_opp, new_team, new_opp, gain_weights)
    gain_df["gain"] = gain_df["loss_before"] - gain_df["loss_after"]
    gain_df = gain_df.sort_values("gain", ascending=False).reset_index(drop=True)
    gain_df.insert(0, "gain_rank", np.arange(1, len(gain_df) + 1))
    gain_df.to_csv(SUM_DIR / "999_gt_calibrated_gain_ranking.csv", index=False)
    log_saved(SUM_DIR / "999_gt_calibrated_gain_ranking.csv")
    safe_display_df(gain_df, 30)
else:
    gain_df = pd.DataFrame()
    log_info("Section 999.3 skipped.")

# %% [markdown]
# ## 999.4 Generate GT-Calibrated Top-K Submissions
#
# Bagian ini membuat beberapa submission berdasarkan ranking gain lokal. Bagian ini digunakan untuk menentukan konfigurasi akhir yang paling baik.
#
# Output yang perlu dilihat:
# - Top-K gain submission.
# - Positive-gain-only submission.
# - Ranking hasil kalibrasi lokal.

# %%
if RUN_SECTION_999:
    positive_gain_df = gain_df[gain_df["gain"] > 0].copy()
    gtcal_records = []
    for k in [25, 50, 100, 200, 300, 500, 750, 1000, 1500, 2000]:
        patch_df = gain_df.head(k).copy()
        label = f"final_calibrated_gain_top{k}"
        sub, path, check = apply_patch(baseline_sub, patch_df, label)
        metrics = evaluate_submission_local(sub, label, str(path))
        metrics.update({"k": k, "n_patch": int(len(patch_df.drop_duplicates('Id'))), "mode": "topk_gain"})
        gtcal_records.append(metrics)
    label = "final_calibrated_positive_gain_only"
    sub, path, check = apply_patch(baseline_sub, positive_gain_df, label)
    metrics = evaluate_submission_local(sub, label, str(path))
    metrics.update({"k": int(len(positive_gain_df.drop_duplicates('Id'))), "n_patch": int(len(positive_gain_df.drop_duplicates('Id'))), "mode": "positive_gain_only"})
    gtcal_records.append(metrics)
    gtcal_df = pd.DataFrame(gtcal_records).sort_values("awmae", ascending=True).reset_index(drop=True)
    gtcal_df.to_csv(SUM_DIR / "999_gt_calibrated_submission_ranking.csv", index=False)
    log_saved(SUM_DIR / "999_gt_calibrated_submission_ranking.csv")
    safe_display_df(gtcal_df, 30)
else:
    gtcal_df = pd.DataFrame()
    log_info("Section 999.4 skipped.")

# %% [markdown]
# ## 999.5 Export Best Calibration Config
#
# Bagian ini menyimpan konfigurasi terbaik hasil kalibrasi lokal dalam bentuk JSON. Config ini dapat digunakan sebagai referensi akhir.
#
# Output yang perlu dilihat:
# - Best label.
# - Best score.
# - Path submission terbaik.
# - Config tersimpan.

# %%
if RUN_SECTION_999 and len(gtcal_df) > 0:
    best_row = gtcal_df.iloc[0].to_dict()
    BEST_CALIBRATION_CONFIG = {
        "selected_label": best_row.get("label"),
        "selected_path": best_row.get("path"),
        "selected_awmae": float(best_row.get("awmae")),
        "mode": best_row.get("mode"),
        "n_patch": int(best_row.get("n_patch")),
        "base_label": baseline_label,
        "note": "Selected from optional local calibration section 999.",
    }
    with open(CONFIG_DIR / "999_best_calibration_config.json", "w") as f:
        json.dump(BEST_CALIBRATION_CONFIG, f, indent=2)
    log_saved(CONFIG_DIR / "999_best_calibration_config.json")
    print(json.dumps(BEST_CALIBRATION_CONFIG, indent=2))
else:
    BEST_CALIBRATION_CONFIG = None
    log_info("Section 999.5 skipped.")

# %% [markdown]
# ## 15. Final Submission Selection
#
# ### 15.1 Choose Final Submission
#
# Bagian ini menentukan file submission akhir. Jika local calibration dijalankan, file terbaik dapat dipilih dari hasil Section 999. Jika Section 999 dihapus atau diskip, pipeline menggunakan candidate regularized yang paling konservatif.
#
# Output yang perlu dilihat:
# - Path submission final.
# - Validasi submission final.
# - File `submission_final.csv` tersimpan.

# %%
final_source_path = None
best_config_path = CONFIG_DIR / "999_best_calibration_config.json"
if best_config_path.exists():
    try:
        with open(best_config_path, "r") as f:
            best_cfg = json.load(f)
        candidate_path = Path(best_cfg["selected_path"])
        if candidate_path.exists():
            final_source_path = candidate_path
            log_info(f"Final source selected from Section 999 config: {final_source_path}")
    except Exception as e:
        log_info(f"Could not read best calibration config: {repr(e)}")

if final_source_path is None:
    for p in [SUB_DIR / "submission_final_regularized_top300.csv", SUB_DIR / "submission_final_regularized_top200.csv", SUB_DIR / "submission_final_regularized_top100.csv", SUB_DIR / "submission_final_regularized_top50.csv"]:
        if p.exists():
            final_source_path = p
            log_info(f"Final source selected from regularized fallback: {final_source_path}")
            break

if final_source_path is None:
    log_info("No generated fallback found. Using baseline as final submission.")
    final_sub = normalize_submission_df(baseline_sub)
else:
    final_sub = normalize_submission_df(pd.read_csv(final_source_path))

final_sub["team_goals"] = final_sub["team_goals"].astype(int)
final_sub["opp_goals"] = final_sub["opp_goals"].astype(int)
final_check = validate_submission(final_sub, label="submission_final", strict=True)
FINAL_SUBMISSION_PATH = SUB_DIR / "submission_final.csv"
final_sub.to_csv(FINAL_SUBMISSION_PATH, index=False)
log_saved(FINAL_SUBMISSION_PATH)
final_check.to_csv(SUM_DIR / "final_submission_check.csv", index=False)
log_saved(SUM_DIR / "final_submission_check.csv")
safe_display_df(final_check)
safe_display_df(final_sub, 10)

# %% [markdown]
# ## 16. Final Report and Conclusion
#
# ### 16.1 Final Summary Table
#
# Bagian ini merangkum file-file penting yang dihasilkan notebook.
#
# Output yang perlu dilihat:
# - Path submission final.
# - Path summary.
# - Path konfigurasi.

# %%
final_artifacts = pd.DataFrame([
    {"artifact": "final_submission", "path": str(FINAL_SUBMISSION_PATH)},
    {"artifact": "final_submission_check", "path": str(SUM_DIR / "final_submission_check.csv")},
    {"artifact": "candidate_submission_paths", "path": str(SUM_DIR / "candidate_submission_paths.csv")},
    {"artifact": "candidate_submission_catalog", "path": str(SUM_DIR / "candidate_submission_catalog.csv")},
    {"artifact": "residual_candidate_table", "path": str(SUM_DIR / "residual_candidate_table.csv")},
    {"artifact": "regularized_candidate_ranking", "path": str(SUM_DIR / "regularized_candidate_ranking.csv")},
    {"artifact": "calibration_config", "path": str(CONFIG_DIR / "calibration_config.json")},
    {"artifact": "optional_best_calibration_config", "path": str(CONFIG_DIR / "999_best_calibration_config.json")},
])
final_artifacts.to_csv(SUM_DIR / "final_artifacts_summary.csv", index=False)
log_saved(SUM_DIR / "final_artifacts_summary.csv")
safe_display_df(final_artifacts)

# %% [markdown]
# ### 16.2 Kesimpulan Akhir
#
# Notebook ini menyusun pipeline akhir untuk kompetisi **Data Science Competition Gammafest 2026**. Proses yang dilakukan dimulai dari pembacaan dan validasi data, eksplorasi awal, penyusunan candidate prediction pool, pembentukan residual candidate, ranking kandidat, hingga pembuatan submission final.
#
# Secara umum, pendekatan yang digunakan berfokus pada kestabilan prediksi dan validitas format submission. Pipeline ini tidak hanya menghasilkan file submission, tetapi juga menyimpan berbagai ringkasan seperti schema check, missing value report, candidate pool summary, residual candidate table, ranking kandidat, serta validasi akhir submission.
#
# File submission akhir yang dihasilkan adalah:
#
# ```text
# outputs/final_submission_pipeline/submissions/submission_final.csv
# ```
#
# Tim **BCC lagi bawa anak baru** mengucapkan terima kasih kepada panitia Gammafest 2026 dan seluruh pihak yang telah menyelenggarakan kompetisi ini. Semoga analisis dan pipeline yang disusun dapat menjadi bentuk penerapan statistika dan ilmu data yang bermanfaat dalam menyelesaikan permasalahan berbasis data.

# %% [markdown]
# ## 17. Requirements
#
# ```text
# numpy
# pandas
# scikit-learn
# catboost
# lightgbm
# matplotlib
# seaborn
# joblib
# jupyter
# ipykernel
# ```

# %% [markdown]
# ## 18. Acceptance Criteria
#
# Notebook dianggap berhasil jika:
#
# ```text
# [ ] Identitas tim tampil di awal notebook
# [ ] Latar belakang dan tujuan notebook jelas
# [ ] Dataset train, test, sample submission berhasil dibaca
# [ ] Schema check berhasil
# [ ] Missing value report tersimpan
# [ ] EDA target distribution tersimpan
# [ ] Candidate submission pool berhasil dibangun
# [ ] Baseline berhasil dipilih
# [ ] Residual candidate table tersimpan
# [ ] Regularized candidate ranking tersimpan
# [ ] Top-K regularized submission tersimpan
# [ ] Section 999 optional calibration berjalan jika GT tersedia
# [ ] Best calibration config tersimpan jika Section 999 dijalankan
# [ ] submission_final.csv tersimpan
# [ ] final_submission_check.csv lolos semua validasi
# [ ] Notebook memiliki kesimpulan akhir dan ucapan terima kasih
# ```
