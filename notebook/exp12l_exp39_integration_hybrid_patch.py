# %% [markdown]
# # 00. EXP12L — EXP39 Integration, Export Audit, dan Conservative Hybrid Patch
#
# Penjelasan bagian:
# EXP12L mengintegrasikan EXP39 sebagai backbone baru, lalu melakukan audit export candidate dan hybrid patch kecil di atas EXP39. Notebook ini adalah post-processing/integration experiment, bukan training model baru.
#
# Output yang perlu dilihat:
# Pastikan EXP39 final berhasil diload sebagai L0, semua candidate tersimpan di `outputs/exp12l_exp39_integration_hybrid_patch/<variant>/submissions/`, dan Section 99 local audit menampilkan ranking AW-MAE tanpa memengaruhi pipeline utama.

# %% [markdown]
# # 01. Setup, Project Root, Logging, dan Guardrail
#
# Penjelasan bagian:
# Bagian ini menyiapkan import, seed, logging, project root resolver, serta output folder standar project. Resolver ini mencegah output masuk ke `notebook/outputs/`.
#
# Output yang perlu dilihat:
# Log harus menunjukkan `PROJECT_ROOT`, `OUTPUT_ROOT`, `OUT_DIR`, `SUB_DIR`, dan `SUM_DIR`. Ground truth tidak dibaca di section ini.

# %%
import os
import re
import shutil
import json
import math
import random
import warnings
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
pd.set_option("display.max_columns", 240)
pd.set_option("display.max_rows", 120)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)


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


def safe_display_df(df, n=30):
    if isinstance(df, pd.DataFrame):
        display(df.head(n))
    else:
        display(df)


def infer_project_root() -> Path:
    cwd = Path.cwd().resolve()
    if cwd.name.lower() in {"notebook", "notebooks"}:
        return cwd.parent
    for candidate in [cwd, cwd.parent, cwd.parent.parent]:
        if (candidate / "data").exists() or (candidate / "dataset").exists():
            return candidate
    if Path("/mnt/data").exists() and ((Path("/mnt/data") / "test.csv").exists() or (Path("/mnt/data") / "sample submission.csv").exists()):
        return Path("/mnt/data")
    return cwd


PROJECT_ROOT = infer_project_root()
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
EXP12L_VARIANT = os.environ.get("EXP12L_VARIANT", "l1_exp39_export_audit")
OUT_DIR = OUTPUT_ROOT / "exp12l_exp39_integration_hybrid_patch" / EXP12L_VARIANT
PRED_DIR = OUT_DIR / "predictions"
SUB_DIR = OUT_DIR / "submissions"
SUM_DIR = OUT_DIR / "summaries"
FIG_DIR = OUT_DIR / "figures"

# Optional hard-clean untuk mencegah partial output EXP12L lama ikut kebaca sebagai source.
# Aktifkan saat full run dengan: EXP12L_CLEAN_OUTDIR=1
EXP12L_CLEAN_OUTDIR = os.environ.get("EXP12L_CLEAN_OUTDIR", "0").strip().lower() in {"1", "true", "yes", "y"}
if EXP12L_CLEAN_OUTDIR and OUT_DIR.exists():
    print(f"[INFO] Cleaning previous EXP12L output folder: {OUT_DIR}", flush=True)
    shutil.rmtree(OUT_DIR)

for d in [PRED_DIR, SUB_DIR, SUM_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RUN_LOCAL_GT_AUDIT_DEFAULT = os.environ.get("EXP12L_RUN_LOCAL_GT_AUDIT", "1").strip().lower() in {"1", "true", "yes", "y"}
STRICT_FINAL = True
MAX_GOAL_SANITY = 40

log_section("Setup")
log_info(f"PROJECT_ROOT = {PROJECT_ROOT}")
log_info(f"OUTPUT_ROOT = {OUTPUT_ROOT}")
log_info(f"OUT_DIR = {OUT_DIR}")
log_info(f"SUB_DIR = {SUB_DIR}")
log_info(f"SUM_DIR = {SUM_DIR}")
log_info(f"EXP12L_VARIANT = {EXP12L_VARIANT}")
log_info(f"EXP12L_CLEAN_OUTDIR = {EXP12L_CLEAN_OUTDIR}")
log_info("GT guardrail: ground_truth_bersih.csv hanya dibaca di Section 99 optional local audit")

# %% [markdown]
# # 02. Robust Data Finder dan Basic Audit
#
# Penjelasan bagian:
# Bagian ini mencari `train.csv`, `test.csv`, dan sample submission dengan robust. Finder sengaja menghindari folder outputs agar tidak salah mengambil submission eksperimen sebagai sample.
#
# Output yang perlu dilihat:
# Pastikan path train/test/sample mengarah ke dataset asli. Row count sample harus sama dengan test.

# %%
SAMPLE_NAMES = [
    "sample_submission.csv",
    "sample submission.csv",
    "samplesubmission.csv",
    "sample-submission.csv",
    "submission_sample.csv",
]


def is_forbidden_data_path(p: Path) -> bool:
    parts = {x.lower() for x in p.parts}
    name = p.name.lower()
    if "outputs" in parts:
        return True
    if name.startswith("submission_exp"):
        return True
    if "leaderboard" in name:
        return True
    return False


def candidate_data_dirs():
    dirs = [
        PROJECT_ROOT / "data",
        PROJECT_ROOT / "dataset",
        PROJECT_ROOT,
        PROJECT_ROOT.parent / "data",
        PROJECT_ROOT.parent / "dataset",
        Path.cwd(),
        Path.cwd() / "data",
        Path.cwd() / "dataset",
        Path.cwd().parent / "data",
        Path.cwd().parent / "dataset",
        Path("/mnt/data"),
    ]
    out = []
    for d in dirs:
        try:
            d = d.resolve()
        except Exception:
            continue
        if d.exists() and d not in out:
            out.append(d)
    return out


def find_data_file(names, recursive: bool = True) -> Path:
    if isinstance(names, str):
        names = [names]
    lower_names = {n.lower() for n in names}
    for d in candidate_data_dirs():
        for n in names:
            p = d / n
            if p.exists() and not is_forbidden_data_path(p):
                return p
    if recursive:
        for d in candidate_data_dirs():
            try:
                for p in d.rglob("*.csv"):
                    if is_forbidden_data_path(p):
                        continue
                    if p.name.lower() in lower_names:
                        return p
            except Exception:
                pass
    raise FileNotFoundError(f"Tidak menemukan file: {names}")


TRAIN_PATH = find_data_file("train.csv")
TEST_PATH = find_data_file("test.csv")
SAMPLE_PATH = find_data_file(SAMPLE_NAMES)

train_raw = pd.read_csv(TRAIN_PATH)
test_raw = pd.read_csv(TEST_PATH)
sample_submission = pd.read_csv(SAMPLE_PATH)

SAMPLE_ID_COL_RAW = "Id" if "Id" in sample_submission.columns else ("id" if "id" in sample_submission.columns else sample_submission.columns[0])
TEST_ID_COL_RAW = "Id" if "Id" in test_raw.columns else ("id" if "id" in test_raw.columns else SAMPLE_ID_COL_RAW)

sample_submission = sample_submission.rename(columns={SAMPLE_ID_COL_RAW: "Id"}).copy()
test_raw = test_raw.rename(columns={TEST_ID_COL_RAW: "Id"}).copy()

if "match_id" not in test_raw.columns:
    test_raw = test_raw.copy()
    test_raw["match_id"] = np.arange(len(test_raw)) // 2
    log_info("test.csv tidak punya match_id; fallback membuat match_id dari pasangan dua row berurutan")

for c in ["tournament", "gender"]:
    if c not in test_raw.columns:
        test_raw[c] = "unknown"

log_section("Input Audit")
log_info(f"TRAIN_PATH = {TRAIN_PATH}")
log_info(f"TEST_PATH = {TEST_PATH}")
log_info(f"SAMPLE_PATH = {SAMPLE_PATH}")
log_info(f"train shape = {train_raw.shape}")
log_info(f"test shape = {test_raw.shape}")
log_info(f"sample shape = {sample_submission.shape}")
log_check("sample has Id", "Id" in sample_submission.columns)
log_check("test has Id", "Id" in test_raw.columns)
log_check("test has match_id", "match_id" in test_raw.columns)
log_check("sample row count equals test row count", len(sample_submission) == len(test_raw))

input_audit_df = pd.DataFrame([
    {"name": "TRAIN_PATH", "value": str(TRAIN_PATH)},
    {"name": "TEST_PATH", "value": str(TEST_PATH)},
    {"name": "SAMPLE_PATH", "value": str(SAMPLE_PATH)},
    {"name": "train_shape", "value": str(train_raw.shape)},
    {"name": "test_shape", "value": str(test_raw.shape)},
    {"name": "sample_shape", "value": str(sample_submission.shape)},
])
safe_display_df(input_audit_df)

# %% [markdown]
# # 03. Core Helpers: Outcome, Weights, Submission Alignment, dan Pair Consistency
#
# Penjelasan bagian:
# Bagian ini berisi helper utama untuk metric proxy, tournament weight, alignment submission ke sample, validasi pair consistency, dan konversi row-level ke match-level. Semua hybrid di EXP12L bekerja di match-level agar pair-consistent.
#
# Output yang perlu dilihat:
# Tidak ada output besar. Jika ada submission invalid, `validate_and_save_submission` akan mencatat check dan bisa raise untuk final strict.

# %%
EXACT_PENALTY = 0.30
OUTCOME_PENALTY = 0.25
GD_PENALTY = 0.15
WRONG_OUTCOME_MULTIPLIER = 1.50
NONLINEAR_POWER = 1.50


def outcome_scalar(a, b):
    a = int(a)
    b = int(b)
    if a > b:
        return 1
    if a < b:
        return -1
    return 0


def outcome_array(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return np.where(a > b, 1, np.where(a < b, -1, 0))


def scoreline(a, b) -> str:
    return f"{int(a)}-{int(b)}"


def get_tournament_weight(tournament: str) -> float:
    t = str(tournament).lower().strip()
    if "fifa world cup" in t or t == "world cup":
        return 2.00
    if "afc championship" in t or "afc asian cup" in t or "asian cup" in t:
        return 1.80
    if "friendly" in t:
        return 0.96
    return 1.20


def normalize_submission_df(sub_df: pd.DataFrame, sample_df: pd.DataFrame = None, require_all: bool = True) -> pd.DataFrame:
    if sample_df is None:
        sample_df = sample_submission
    df = sub_df.copy()
    id_col = "Id" if "Id" in df.columns else ("id" if "id" in df.columns else None)
    if id_col is None:
        raise KeyError("submission missing Id/id column")
    df = df.rename(columns={id_col: "Id"})
    required = ["Id", "team_goals", "opp_goals"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"submission missing columns: {missing}")
    df = df[required].copy()
    df["Id"] = df["Id"].astype(str)
    df["team_goals"] = pd.to_numeric(df["team_goals"], errors="raise").round().astype(int).clip(lower=0)
    df["opp_goals"] = pd.to_numeric(df["opp_goals"], errors="raise").round().astype(int).clip(lower=0)

    sample_ids = sample_df[["Id"]].copy()
    sample_ids["Id"] = sample_ids["Id"].astype(str)
    aligned = sample_ids.merge(df, on="Id", how="left", validate="one_to_one")
    if require_all and aligned[["team_goals", "opp_goals"]].isna().any().any():
        missing_ids = aligned.loc[aligned[["team_goals", "opp_goals"]].isna().any(axis=1), "Id"].head(10).tolist()
        raise RuntimeError(f"missing predictions after align, examples: {missing_ids}")
    aligned["team_goals"] = pd.to_numeric(aligned["team_goals"], errors="coerce").round().astype("Int64")
    aligned["opp_goals"] = pd.to_numeric(aligned["opp_goals"], errors="coerce").round().astype("Int64")
    if require_all:
        aligned["team_goals"] = aligned["team_goals"].astype(int).clip(lower=0)
        aligned["opp_goals"] = aligned["opp_goals"].astype(int).clip(lower=0)
    return aligned[["Id", "team_goals", "opp_goals"]]


# Match metadata based on test row order. Each match should have exactly two rows.
def build_match_meta(test_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for match_id, g in test_df.reset_index(drop=True).groupby("match_id", sort=False):
        if len(g) != 2:
            # Fallback keeps first two rows if data malformed; validation later will catch it.
            if len(g) < 2:
                continue
            g = g.iloc[:2].copy()
        r0 = g.iloc[0]
        r1 = g.iloc[1]
        rows.append({
            "match_id": match_id,
            "row_id_a": str(r0["Id"]),
            "row_id_b": str(r1["Id"]),
            "gender": str(r0.get("gender", "unknown")),
            "tournament": str(r0.get("tournament", "unknown")),
            "tournament_weight": float(get_tournament_weight(r0.get("tournament", "unknown"))),
        })
    meta = pd.DataFrame(rows)
    return meta


match_meta = build_match_meta(test_raw)
MATCH_ID_TO_META = match_meta.set_index("match_id", drop=False).to_dict("index")
ID_TO_MATCH = {}
for _, r in match_meta.iterrows():
    ID_TO_MATCH[str(r["row_id_a"])] = {"match_id": r["match_id"], "side": "a"}
    ID_TO_MATCH[str(r["row_id_b"])] = {"match_id": r["match_id"], "side": "b"}


def submission_to_match(sub_df: pd.DataFrame, label: str = "submission") -> pd.DataFrame:
    sub = normalize_submission_df(sub_df, require_all=True)
    sub_idx = sub.set_index(sub["Id"].astype(str), drop=False)
    rows = []
    for _, m in match_meta.iterrows():
        id_a = str(m["row_id_a"])
        id_b = str(m["row_id_b"])
        if id_a not in sub_idx.index or id_b not in sub_idx.index:
            continue
        ra = sub_idx.loc[id_a]
        rb = sub_idx.loc[id_b]
        if isinstance(ra, pd.DataFrame):
            ra = ra.iloc[0]
        if isinstance(rb, pd.DataFrame):
            rb = rb.iloc[0]
        rows.append({
            "match_id": m["match_id"],
            "pred_a": int(ra["team_goals"]),
            "pred_b": int(ra["opp_goals"]),
            "mirror_pred_a": int(rb["opp_goals"]),
            "mirror_pred_b": int(rb["team_goals"]),
            "pair_consistent_row": bool(int(ra["team_goals"]) == int(rb["opp_goals"]) and int(ra["opp_goals"]) == int(rb["team_goals"])),
            "gender": m["gender"],
            "tournament": m["tournament"],
            "tournament_weight": m["tournament_weight"],
            "label": label,
        })
    out = pd.DataFrame(rows)
    return out


def match_to_submission(match_pred: pd.DataFrame) -> pd.DataFrame:
    required = ["match_id", "pred_a", "pred_b"]
    missing = [c for c in required if c not in match_pred.columns]
    if missing:
        raise KeyError(f"match_pred missing columns: {missing}")
    pred = match_pred[required].copy()
    pred["pred_a"] = pd.to_numeric(pred["pred_a"], errors="raise").round().astype(int).clip(lower=0)
    pred["pred_b"] = pd.to_numeric(pred["pred_b"], errors="raise").round().astype(int).clip(lower=0)
    merged = match_meta[["match_id", "row_id_a", "row_id_b"]].merge(pred, on="match_id", how="left", validate="one_to_one")
    if merged[["pred_a", "pred_b"]].isna().any().any():
        bad = merged.loc[merged[["pred_a", "pred_b"]].isna().any(axis=1), "match_id"].head().tolist()
        raise RuntimeError(f"missing match predictions examples: {bad}")
    rows_a = pd.DataFrame({"Id": merged["row_id_a"].astype(str), "team_goals": merged["pred_a"].astype(int), "opp_goals": merged["pred_b"].astype(int)})
    rows_b = pd.DataFrame({"Id": merged["row_id_b"].astype(str), "team_goals": merged["pred_b"].astype(int), "opp_goals": merged["pred_a"].astype(int)})
    long_pred = pd.concat([rows_a, rows_b], ignore_index=True)
    return normalize_submission_df(long_pred, require_all=True)


def pair_consistency_report(sub_df: pd.DataFrame) -> dict:
    try:
        sub = normalize_submission_df(sub_df, require_all=True)
    except Exception as e:
        return {"pair_consistency": False, "n_bad_pairs": -1, "error": repr(e)}
    m = submission_to_match(sub)
    if len(m) != len(match_meta):
        return {"pair_consistency": False, "n_bad_pairs": int(len(match_meta) - len(m)), "error": "match_count_mismatch"}
    n_bad = int((~m["pair_consistent_row"].astype(bool)).sum())
    return {"pair_consistency": n_bad == 0, "n_bad_pairs": n_bad, "error": ""}


def validation_checks(sub_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    try:
        sub = normalize_submission_df(sub_df, require_all=True)
        rows.append({"check": "shape_matches_sample", "passed": tuple(sub.shape) == tuple(sample_submission[["Id"]].assign(team_goals=0, opp_goals=0).shape)})
        rows.append({"check": "id_order_matches_sample", "passed": sub["Id"].astype(str).tolist() == sample_submission["Id"].astype(str).tolist()})
        rows.append({"check": "no_missing", "passed": not sub[["team_goals", "opp_goals"]].isna().any().any()})
        rows.append({"check": "no_duplicate_id", "passed": not sub["Id"].duplicated().any()})
        rows.append({"check": "non_negative", "passed": bool((sub[["team_goals", "opp_goals"]] >= 0).all().all())})
        rows.append({"check": "integer_prediction", "passed": bool(np.allclose(sub["team_goals"], sub["team_goals"].round()) and np.allclose(sub["opp_goals"], sub["opp_goals"].round()))})
        pc = pair_consistency_report(sub)
        rows.append({"check": "pair_consistency", "passed": bool(pc["pair_consistency"]), "detail": f"n_bad_pairs={pc['n_bad_pairs']}"})
        rows.append({"check": "max_goal_sanity_le_40", "passed": int(sub[["team_goals", "opp_goals"]].max().max()) <= MAX_GOAL_SANITY})
    except Exception as e:
        rows.append({"check": "exception", "passed": False, "detail": repr(e)})
    out = pd.DataFrame(rows)
    if "detail" not in out.columns:
        out["detail"] = ""
    return out


def basic_submission_metrics(sub_df: pd.DataFrame, label: str) -> dict:
    sub = normalize_submission_df(sub_df, require_all=True)
    m = submission_to_match(sub, label=label)
    scores = (m["pred_a"].astype(int).astype(str) + "-" + m["pred_b"].astype(int).astype(str)).value_counts(normalize=True)
    top1 = float(scores.iloc[0]) if len(scores) else np.nan
    top3 = float(scores.head(3).sum()) if len(scores) else np.nan
    pc = pair_consistency_report(sub)
    return {
        "label": label,
        "n_rows": int(len(sub)),
        "n_matches": int(len(m)),
        "mean_pred_total": float((m["pred_a"] + m["pred_b"]).mean()),
        "max_pred_goal": int(m[["pred_a", "pred_b"]].max().max()),
        "top1_scoreline_share": top1,
        "top3_scoreline_share": top3,
        "pair_consistency": bool(pc["pair_consistency"]),
        "n_bad_pairs": int(pc["n_bad_pairs"]),
    }


log_section("Match Meta Audit")
match_audit_df = pd.DataFrame([
    {"key": "n_test_rows", "value": len(test_raw)},
    {"key": "n_match_meta", "value": len(match_meta)},
    {"key": "n_ids_in_id_to_match", "value": len(ID_TO_MATCH)},
])
safe_display_df(match_audit_df)

# %% [markdown]
# # 04. Candidate Source Finder dan Loader
#
# Penjelasan bagian:
# Bagian ini mencari submission artifact dari outputs project. EXP39 final, semua export EXP39, dan champion user EXP12K/J0 dicari dengan priority keyword, bukan sorting path biasa.
#
# Output yang perlu dilihat:
# Lihat `source_candidate_inventory.csv`, `exp39_final_source`, dan `user_champion_source`. Jika L0 tidak memakai file EXP39 yang benar, stop dulu.

# %%
SUMMARY_SKIP_NAMES = {
    "submission_catalog.csv",
    "submission_check.csv",
    "submission_validation.csv",
    "submission_metrics.csv",
    "submission_pair_consistency.csv",
    "submission_scoreline_distribution.csv",
    "submission_subgroup_metrics.csv",
    "submission_summary.csv",
}


def is_submission_candidate_path(p: Path) -> bool:
    name = p.name.lower()
    if not name.endswith(".csv"):
        return False
    if not name.startswith("submission_"):
        return False
    if name in SUMMARY_SKIP_NAMES:
        return False
    if "sample" in name:
        return False
    return True


def all_submission_paths() -> list:
    search_roots = [OUTPUT_ROOT, PROJECT_ROOT, Path.cwd(), Path("/mnt/data")]
    seen = set()
    paths = []
    for root in search_roots:
        if not root.exists():
            continue
        try:
            for p in root.rglob("submission_*.csv"):
                p = p.resolve()
                if p in seen:
                    continue
                seen.add(p)

                # Source finder jangan mengambil output EXP12L sendiri, terutama partial run lama.
                # Local audit Section 99 tetap mencari EXP12L candidates secara terpisah.
                p_text = str(p).lower().replace("\\", "/")
                if "/exp12l_exp39_integration_hybrid_patch/" in p_text:
                    continue

                if is_submission_candidate_path(p):
                    paths.append(p)
        except Exception:
            pass
    return sorted(paths, key=lambda x: str(x))


ALL_SUBMISSION_PATHS = all_submission_paths()


def slugify_label(s: str, max_len: int = 120) -> str:
    s = str(s).lower()
    s = re.sub(r"\.csv$", "", s)
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    if len(s) > max_len:
        s = s[:max_len].rstrip("_")
    return s or "candidate"


def keyword_score(path: Path, priority_keywords: list) -> tuple:
    hay = (str(path.parent) + "/" + path.name).lower().replace("-", "_")
    for i, kws in enumerate(priority_keywords):
        if isinstance(kws, str):
            kws = [kws]
        ok = True
        for kw in kws:
            kw_norm = str(kw).lower().replace("-", "_")
            if kw_norm not in hay:
                ok = False
                break
        if ok:
            return (i, len(str(path)), str(path))
    return (10_000, len(str(path)), str(path))


def select_priority_path(paths: list, priority_keywords: list, required_substrings=None) -> Path:
    if required_substrings:
        req = [x.lower().replace("-", "_") for x in required_substrings]
        filtered = []
        for p in paths:
            hay = str(p).lower().replace("-", "_")
            if all(x in hay for x in req):
                filtered.append(p)
    else:
        filtered = list(paths)
    if not filtered:
        return None
    filtered = sorted(filtered, key=lambda p: keyword_score(p, priority_keywords))
    if keyword_score(filtered[0], priority_keywords)[0] >= 10_000:
        return None
    return filtered[0]




def normalize_file_key(x) -> str:
    s = str(x).lower().replace("\\", "/")
    s = s.replace("-", "_").replace(" ", "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s


def exp39_priority_score(path: Path) -> tuple:
    """Priority khusus EXP39 final agar tidak kalah oleh sorting path biasa."""
    key = normalize_file_key(path.name)
    full = normalize_file_key(str(path))

    priority_patterns = [
        "submission_exp39_safe_final_scoreline_distribution_validation_only",
        "exp39_safe_final_scoreline_distribution_validation_only",
        "submission_exp39_safe_final_scoreline_distribution",
        "exp39_safe_final_scoreline_distribution",
        "submission_exp39_safe_final",
        "exp39_safe_final",
        "submission_exp39_scoreline_distribution",
        "exp39_scoreline_distribution",
        "submission_exp39_router",
        "exp39_router",
    ]
    for i, pat in enumerate(priority_patterns):
        if pat in key or pat in full:
            return (i, len(str(path)), str(path))
    return (9999, len(str(path)), str(path))


def find_exp39_final_submission_csv(paths: list) -> tuple:
    """Cari CSV EXP39 final. Kalau hanya notebook yang ada, error message dibuat eksplisit."""
    exp39_paths = []
    for p in paths:
        key = normalize_file_key(p.name)
        full = normalize_file_key(str(p))
        if "exp39" not in key and "exp39" not in full:
            continue
        if not any(x in key or x in full for x in ["safe_final", "scoreline", "router"]):
            continue
        exp39_paths.append(p)

    exp39_paths = sorted(set(exp39_paths), key=exp39_priority_score)

    exp39_source_catalog_df = pd.DataFrame([
        {
            "rank": i + 1,
            "file_name": p.name,
            "path": str(p),
            "priority_score": exp39_priority_score(p)[0],
        }
        for i, p in enumerate(exp39_paths)
    ])
    exp39_source_catalog_df.to_csv(SUM_DIR / "exp39_source_candidate_search.csv", index=False)
    log_saved(SUM_DIR / "exp39_source_candidate_search.csv")
    if len(exp39_source_catalog_df):
        safe_display_df(exp39_source_catalog_df.head(30))

    if not exp39_paths:
        notebook_hits = []
        for root in [PROJECT_ROOT, PROJECT_ROOT / "outputs", PROJECT_ROOT / "data", PROJECT_ROOT / "dataset", Path("/mnt/data")]:
            if root.exists() and root.is_dir():
                try:
                    notebook_hits.extend(root.rglob("*exp39*.ipynb"))
                except Exception:
                    pass
        notebook_hits = sorted(set(notebook_hits), key=lambda x: str(x))
        notebook_hint_df = pd.DataFrame([
            {"notebook_name": p.name, "path": str(p)} for p in notebook_hits
        ])
        notebook_hint_df.to_csv(SUM_DIR / "exp39_notebook_hits_without_csv.csv", index=False)
        if len(notebook_hint_df):
            safe_display_df(notebook_hint_df.head(20))
        raise FileNotFoundError(
            "Tidak menemukan CSV submission EXP39 final. "
            "EXP12L tidak membaca output langsung dari .ipynb. "
            "Run notebook EXP39 dulu sampai menghasilkan CSV final, atau copy file seperti "
            "'submission_exp39_SAFE_FINAL_scoreline_distribution_validation_only.csv' "
            "ke PROJECT_ROOT, data/, dataset/, outputs/, atau /mnt/data."
        )

    selected = exp39_paths[0]
    selected_text = str(selected).lower().replace("\\", "/")
    if "/exp12l_exp39_integration_hybrid_patch/" in selected_text:
        raise RuntimeError(
            f"EXP39 source mengarah ke output EXP12L sendiri: {selected}. "
            "Bersihkan OUT_DIR lama atau pindahkan CSV EXP39 final asli ke folder project/data/outputs."
        )

    return selected, exp39_source_catalog_df

EXP39_PRIORITY = [
    ["exp39", "safe", "final", "scoreline", "distribution", "validation", "only"],
    ["exp39", "safe", "final"],
    ["exp39", "scoreline", "distribution"],
    ["exp39", "router"],
]
USER_CHAMPION_PRIORITY = [
    ["exp12k", "k0", "reproduction"],
    ["exp12j", "j0", "only_1_1", "reproduction"],
    ["exp12i", "i7", "cap0290", "only_1_1"],
    ["exp12i", "i7", "cap0290", "no_2_2"],
    ["exp12i", "i7", "cap0290", "only_0_0_1_1"],
    ["exp12i", "i7", "cap0290", "no_0_0_and_2_2"],
]

exp39_final_path, exp39_source_catalog_df = find_exp39_final_submission_csv(ALL_SUBMISSION_PATHS)
user_champion_path = select_priority_path(ALL_SUBMISSION_PATHS, USER_CHAMPION_PRIORITY)

inventory_rows = []
for p in ALL_SUBMISSION_PATHS:
    low = str(p).lower()
    inventory_rows.append({
        "file_name": p.name,
        "path": str(p),
        "is_exp39": "exp39" in low,
        "is_exp38": "exp38" in low,
        "is_user_old": any(x in low for x in ["exp12k", "exp12j", "exp12i", "exp12h", "exp12e", "exp12f", "exp12c", "exp12b"]),
    })
source_inventory_df = pd.DataFrame(inventory_rows)
source_inventory_df.to_csv(SUM_DIR / "source_candidate_inventory.csv", index=False)
log_saved(SUM_DIR / "source_candidate_inventory.csv")

log_section("Source Finder")
log_info(f"n_submission_paths_found = {len(ALL_SUBMISSION_PATHS)}")
log_info(f"exp39_final_path = {exp39_final_path}")
log_info(f"user_champion_path = {user_champion_path}")
safe_display_df(source_inventory_df.sort_values(["is_exp39", "is_user_old"], ascending=False).head(30))


def read_submission_path(path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    out = normalize_submission_df(df, require_all=True)
    out.attrs["label"] = label
    out.attrs["path"] = str(path)
    return out


def save_candidate(sub_df: pd.DataFrame, label: str, strict: bool = False):
    safe_label = slugify_label(label)
    out = normalize_submission_df(sub_df, require_all=True)
    path = SUB_DIR / f"submission_exp12l_{safe_label}.csv"
    out.to_csv(path, index=False)
    check = validation_checks(out)
    check.insert(0, "label", label)
    check.to_csv(SUM_DIR / f"submission_check_{safe_label}.csv", index=False)
    log_saved(path)
    if strict and not bool(check["passed"].all()):
        display(check)
        raise RuntimeError(f"Submission validation failed for {label}")
    return out, path, check

# %% [markdown]
# # 05. Load EXP39 Final dan User Champion Candidate
#
# Penjelasan bagian:
# Bagian ini memuat EXP39 final sebagai base baru dan pipeline user EXP12K/J0 sebagai comparison artifact. Jika EXP39 final tidak ditemukan, notebook berhenti karena EXP12L memang bertumpu pada EXP39.
#
# Output yang perlu dilihat:
# Cek `l0_exp39_reproduction_summary.csv`. `pair_consistency` harus True dan mean_pred_total harus wajar, idealnya sekitar 2.52-an sesuai karakter EXP39.

# %%
if exp39_final_path is None:
    raise FileNotFoundError("Tidak menemukan submission EXP39 final CSV. Pastikan output EXP39 tersedia sebagai CSV, bukan hanya notebook.")

exp39_final_sub = read_submission_path(exp39_final_path, "exp39_final_source")
exp39_final_match = submission_to_match(exp39_final_sub, label="exp39_final")

if user_champion_path is not None:
    user_champion_sub = read_submission_path(user_champion_path, "user_champion_source")
    user_champion_match = submission_to_match(user_champion_sub, label="user_champion")
else:
    user_champion_sub = None
    user_champion_match = None
    log_info("WARNING: user champion EXP12K/J0 tidak ditemukan. L3/L5 hybrid akan terbatas.")

exp39_l0_sub, exp39_l0_path, exp39_l0_check = save_candidate(exp39_final_sub, "l0_exp39_final_reproduction", strict=False)
l0_summary = pd.DataFrame([{
    "exp39_source_path": str(exp39_final_path),
    **basic_submission_metrics(exp39_l0_sub, "l0_exp39_final_reproduction"),
}])
l0_summary.to_csv(SUM_DIR / "l0_exp39_reproduction_summary.csv", index=False)
log_saved(SUM_DIR / "l0_exp39_reproduction_summary.csv")
safe_display_df(l0_summary)

if not bool(l0_summary["pair_consistency"].iloc[0]):
    log_info("WARNING: L0 EXP39 final tidak pair-consistent. Jangan submit sebelum diperbaiki.")

# EXP39 final biasanya low-total. Jika mean total terlalu tinggi, kemungkinan source CSV salah.
l0_mean_total = float(l0_summary["mean_pred_total"].iloc[0])
if l0_mean_total > 2.60:
    log_check(
        "l0_mean_total_expected_exp39_low_total",
        False,
        f"mean_pred_total={l0_mean_total:.4f}, expected sekitar 2.52-an untuk EXP39 final",
    )
    raise RuntimeError(
        "L0 EXP39 mean_pred_total terlalu tinggi. "
        "Kemungkinan file yang kebaca bukan EXP39 SAFE FINAL scoreline distribution yang benar."
    )
log_check("l0_mean_total_expected_exp39_low_total", True, f"mean_pred_total={l0_mean_total:.4f}")

# %% [markdown]
# # 06. L1 — Audit Semua Export Candidate EXP39
#
# Penjelasan bagian:
# Bagian ini mencari semua export candidate EXP39/EXP38/scoreline/router, mengalign ke sample, menghitung sanity metric, lalu menyimpan ulang ke folder EXP12L. Ini memberi peluang menemukan export EXP39 lain yang lebih baik dari SAFE_FINAL pada local audit.
#
# Output yang perlu dilihat:
# Cek `exp39_export_candidate_catalog.csv` dan `exp39_export_candidate_sanity.csv`. Candidate yang tidak pair-consistent jangan dijadikan final tanpa repair.

# %%
EXP39_EXPORT_PATTERNS = ["exp39", "exp38", "scoreline_distribution", "router"]
exp39_export_paths = []
for p in ALL_SUBMISSION_PATHS:
    hay = str(p).lower()
    if any(pattern in hay for pattern in EXP39_EXPORT_PATTERNS):
        exp39_export_paths.append(p)
exp39_export_paths = sorted(set(exp39_export_paths), key=lambda p: str(p))

EXP39_EXPORT_SUBS = {}
EXP39_EXPORT_MATCH = {}
export_catalog_rows = []
export_sanity_rows = []

for path in exp39_export_paths:
    label = "l1_" + slugify_label(path.stem, max_len=100)
    try:
        sub = read_submission_path(path, label)
        saved_sub, saved_path, check = save_candidate(sub, label, strict=False)
        met = basic_submission_metrics(saved_sub, label)
        met.update({"source_path": str(path), "saved_path": str(saved_path)})
        export_catalog_rows.append(met)
        for _, r in check.iterrows():
            export_sanity_rows.append({"label": label, "check": r["check"], "passed": bool(r["passed"]), "detail": r.get("detail", "")})
        EXP39_EXPORT_SUBS[label] = saved_sub
        EXP39_EXPORT_MATCH[label] = submission_to_match(saved_sub, label=label)
    except Exception as e:
        export_catalog_rows.append({"label": label, "source_path": str(path), "error": repr(e)})

exp39_export_catalog_df = pd.DataFrame(export_catalog_rows)
exp39_export_sanity_df = pd.DataFrame(export_sanity_rows)
exp39_export_catalog_df.to_csv(SUM_DIR / "exp39_export_candidate_catalog.csv", index=False)
exp39_export_sanity_df.to_csv(SUM_DIR / "exp39_export_candidate_sanity.csv", index=False)
log_saved(SUM_DIR / "exp39_export_candidate_catalog.csv")
log_saved(SUM_DIR / "exp39_export_candidate_sanity.csv")
safe_display_df(exp39_export_catalog_df.sort_values("mean_pred_total" if "mean_pred_total" in exp39_export_catalog_df.columns else exp39_export_catalog_df.columns[0]).head(30))

# %% [markdown]
# # 07. Candidate Pool Index dan Reliability Weight
#
# Penjelasan bagian:
# Bagian ini membuat candidate pool match-level dari EXP39 exports dan pipeline user lama. Cache index dipakai supaya router tidak melakukan `set_index()` berulang di loop.
#
# Output yang perlu dilihat:
# Cek `candidate_pool_loaded.csv`. EXP39 final dan user champion sebaiknya ada di pool. Pair unsafe raw candidate boleh tercatat, tetapi final candidate harus tetap pair-consistent.

# %%
def candidate_reliability_weight(label: str) -> float:
    lab = str(label).lower()
    if "exp39" in lab and "safe" in lab and "final" in lab:
        return 3.0
    if "exp39" in lab:
        return 2.5
    if "exp38" in lab:
        return 2.0
    if "exp12k" in lab or "exp12j" in lab or "exp12i" in lab:
        return 1.6
    if "exp12h" in lab or "exp12g" in lab or "exp12f" in lab or "exp12e" in lab:
        return 1.2
    if "exp17" in lab:
        return 0.7
    return 1.0


candidate_match_pool = {}
candidate_match_pool["__exp39_base__"] = exp39_final_match.copy()
if user_champion_match is not None:
    candidate_match_pool["__user_champion__"] = user_champion_match.copy()

# Add EXP39 exports.
for lab, mdf in EXP39_EXPORT_MATCH.items():
    candidate_match_pool[lab] = mdf.copy()

# Add selected older candidates as comparison artifacts.
OLD_CANDIDATE_KEYWORDS = [
    ["exp12k"], ["exp12j"], ["exp12i"], ["exp12h"], ["exp12g"], ["exp12f"], ["exp12e"], ["exp12c"], ["exp12b"]
]
old_added = 0
for p in ALL_SUBMISSION_PATHS:
    hay = str(p).lower()
    if not any(all(k in hay for k in kws) for kws in OLD_CANDIDATE_KEYWORDS):
        continue
    lab = "old_" + slugify_label(p.stem, max_len=100)
    if lab in candidate_match_pool:
        continue
    try:
        sub = read_submission_path(p, lab)
        mdf = submission_to_match(sub, label=lab)
        candidate_match_pool[lab] = mdf
        old_added += 1
    except Exception:
        pass

CANDIDATE_MATCH_POOL_INDEX = {
    lab: df.set_index("match_id", drop=False)
    for lab, df in candidate_match_pool.items()
}

pool_rows = []
for lab, df in candidate_match_pool.items():
    pc_rows = int((~df["pair_consistent_row"].astype(bool)).sum()) if "pair_consistent_row" in df.columns else 0
    pool_rows.append({
        "label": lab,
        "n_matches": int(len(df)),
        "weight": float(candidate_reliability_weight(lab)),
        "n_bad_pair_rows_source": pc_rows,
        "mean_total": float((df["pred_a"] + df["pred_b"]).mean()),
        "max_goal": int(df[["pred_a", "pred_b"]].max().max()),
    })
candidate_pool_loaded_df = pd.DataFrame(pool_rows).sort_values(["weight", "label"], ascending=[False, True])
candidate_pool_loaded_df.to_csv(SUM_DIR / "candidate_pool_loaded.csv", index=False)
log_saved(SUM_DIR / "candidate_pool_loaded.csv")
safe_display_df(candidate_pool_loaded_df.head(50))

# %% [markdown]
# # 08. L2 — EXP39 vs EXP12K/J0 Delta Analysis
#
# Penjelasan bagian:
# Bagian ini membandingkan EXP39 final dengan champion user lama. Delta analysis dipakai untuk mendesain patch konservatif, bukan untuk memilih rule dengan GT.
#
# Output yang perlu dilihat:
# Cek `exp39_vs_exp12k_delta_table.csv`, transition summary, dan segment summary. Lihat perbedaan total goal, outcome, GD, gender, dan tournament weight.

# %%
def merge_match_predictions(left: pd.DataFrame, right: pd.DataFrame, left_prefix: str, right_prefix: str) -> pd.DataFrame:
    l = left[["match_id", "pred_a", "pred_b", "gender", "tournament", "tournament_weight"]].rename(columns={"pred_a": f"{left_prefix}_a", "pred_b": f"{left_prefix}_b"})
    r = right[["match_id", "pred_a", "pred_b"]].rename(columns={"pred_a": f"{right_prefix}_a", "pred_b": f"{right_prefix}_b"})
    return l.merge(r, on="match_id", how="inner", validate="one_to_one")

if user_champion_match is not None:
    delta = merge_match_predictions(exp39_final_match, user_champion_match, "exp39", "exp12k")
    delta["exp39_score"] = delta.apply(lambda r: scoreline(r["exp39_a"], r["exp39_b"]), axis=1)
    delta["exp12k_score"] = delta.apply(lambda r: scoreline(r["exp12k_a"], r["exp12k_b"]), axis=1)
    delta["transition"] = delta["exp39_score"] + " -> " + delta["exp12k_score"]
    delta["same_score"] = delta["exp39_score"].eq(delta["exp12k_score"])
    delta["exp39_outcome"] = outcome_array(delta["exp39_a"], delta["exp39_b"])
    delta["exp12k_outcome"] = outcome_array(delta["exp12k_a"], delta["exp12k_b"])
    delta["same_outcome"] = delta["exp39_outcome"].eq(delta["exp12k_outcome"])
    delta["exp39_gd"] = delta["exp39_a"] - delta["exp39_b"]
    delta["exp12k_gd"] = delta["exp12k_a"] - delta["exp12k_b"]
    delta["same_gd"] = delta["exp39_gd"].eq(delta["exp12k_gd"])
    delta["gd_diff"] = (delta["exp12k_gd"] - delta["exp39_gd"]).abs()
    delta["exp39_total"] = delta["exp39_a"] + delta["exp39_b"]
    delta["exp12k_total"] = delta["exp12k_a"] + delta["exp12k_b"]
    delta["total_diff"] = delta["exp12k_total"] - delta["exp39_total"]
    delta["is_high_weight"] = delta["tournament_weight"] >= 1.8
    delta["is_low_weight"] = delta["tournament_weight"] <= 0.96
    delta["is_world_asian"] = delta["tournament"].astype(str).str.lower().str.contains("world|asian|afc", regex=True)
    exp39_vs_exp12k_delta_table = delta.copy()
else:
    exp39_vs_exp12k_delta_table = pd.DataFrame()

exp39_vs_exp12k_delta_table.to_csv(SUM_DIR / "exp39_vs_exp12k_delta_table.csv", index=False)
log_saved(SUM_DIR / "exp39_vs_exp12k_delta_table.csv")

if len(exp39_vs_exp12k_delta_table):
    transition_summary = (
        exp39_vs_exp12k_delta_table
        .assign(changed=lambda d: ~d["same_score"])
        .groupby("transition", dropna=False)
        .agg(n=("match_id", "size"), n_changed=("changed", "sum"), mean_weight=("tournament_weight", "mean"), mean_total_diff=("total_diff", "mean"))
        .reset_index()
        .sort_values("n", ascending=False)
    )
    segment_summary = (
        exp39_vs_exp12k_delta_table
        .assign(weight_bucket=lambda d: np.where(d["is_high_weight"], "high", np.where(d["is_low_weight"], "low", "default")))
        .groupby(["gender", "weight_bucket"], dropna=False)
        .agg(n=("match_id", "size"), changed_score=("same_score", lambda s: int((~s).sum())), same_outcome_rate=("same_outcome", "mean"), same_gd_rate=("same_gd", "mean"), mean_total_diff=("total_diff", "mean"))
        .reset_index()
        .sort_values("n", ascending=False)
    )
else:
    transition_summary = pd.DataFrame()
    segment_summary = pd.DataFrame()

transition_summary.to_csv(SUM_DIR / "exp39_vs_exp12k_transition_summary.csv", index=False)
segment_summary.to_csv(SUM_DIR / "exp39_vs_exp12k_segment_summary.csv", index=False)
log_saved(SUM_DIR / "exp39_vs_exp12k_transition_summary.csv")
log_saved(SUM_DIR / "exp39_vs_exp12k_segment_summary.csv")
safe_display_df(transition_summary.head(20))
safe_display_df(segment_summary.head(20))

# %% [markdown]
# # 09. Patch Candidate Builder dan Support Helpers
#
# Penjelasan bagian:
# Bagian ini membangun helper untuk mengambil dukungan dari candidate pool, membuat ranking patch, menerapkan perubahan kecil di match-level, dan menyimpan candidate. Patch di atas EXP39 harus kecil dan konservatif.
#
# Output yang perlu dilihat:
# Tidak ada output besar. Summary patch muncul di section L3-L7.

# %%
def candidate_scores_for_match(match_id, exclude_labels=None):
    exclude_labels = set(exclude_labels or [])
    rows = []
    for lab, idx in CANDIDATE_MATCH_POOL_INDEX.items():
        if lab in exclude_labels:
            continue
        if match_id not in idx.index:
            continue
        r = idx.loc[match_id]
        if isinstance(r, pd.DataFrame):
            r = r.iloc[0]
        a = int(r["pred_a"])
        b = int(r["pred_b"])
        rows.append({
            "label": lab,
            "weight": float(candidate_reliability_weight(lab)),
            "a": a,
            "b": b,
            "score": scoreline(a, b),
            "outcome": outcome_scalar(a, b),
            "gd": a - b,
            "total": a + b,
        })
    return rows


def support_for_score(match_id, target_a, target_b, exclude_labels=None):
    rows = candidate_scores_for_match(match_id, exclude_labels=exclude_labels)
    target_score = scoreline(target_a, target_b)
    hit = [r for r in rows if r["score"] == target_score]
    return {
        "support": int(len(hit)),
        "weighted_support": float(sum(r["weight"] for r in hit)),
        "source_labels": ";".join([r["label"] for r in hit[:8]]),
        "source_label_top": hit[0]["label"] if hit else "",
        "source_weight_top": float(hit[0]["weight"]) if hit else 0.0,
    }


def support_for_outcome(match_id, target_outcome, exclude_labels=None):
    rows = candidate_scores_for_match(match_id, exclude_labels=exclude_labels)
    hit = [r for r in rows if int(r["outcome"]) == int(target_outcome)]
    return {
        "outcome_support": int(len(hit)),
        "outcome_weighted_support": float(sum(r["weight"] for r in hit)),
        "outcome_source_labels": ";".join([r["label"] for r in hit[:8]]),
        "outcome_source_label_top": hit[0]["label"] if hit else "",
        "outcome_source_weight_top": float(hit[0]["weight"]) if hit else 0.0,
    }


def apply_match_changes(base_match: pd.DataFrame, changes: pd.DataFrame, label: str):
    out = base_match.copy()
    if changes is None or len(changes) == 0:
        out["variant_label"] = label
        return out
    idx = out.set_index("match_id", drop=False)
    for _, r in changes.iterrows():
        mid = r["match_id"]
        if mid not in idx.index:
            continue
        idx.loc[mid, "pred_a"] = int(r["new_a"])
        idx.loc[mid, "pred_b"] = int(r["new_b"])
    out = idx.reset_index(drop=True)
    out["variant_label"] = label
    return out


def compare_match_to_base(candidate_match: pd.DataFrame, base_match: pd.DataFrame, label: str):
    m = merge_match_predictions(base_match, candidate_match, "base", "cand")
    changed = (m["base_a"] != m["cand_a"]) | (m["base_b"] != m["cand_b"])
    m["base_outcome"] = outcome_array(m["base_a"], m["base_b"])
    m["cand_outcome"] = outcome_array(m["cand_a"], m["cand_b"])
    m["base_gd"] = m["base_a"] - m["base_b"]
    m["cand_gd"] = m["cand_a"] - m["cand_b"]
    return {
        "label": label,
        "n_matches": int(len(m)),
        "n_changed_vs_exp39": int(changed.sum()),
        "changed_rate_vs_exp39": float(changed.mean()) if len(m) else 0.0,
        "same_outcome_changed_rate": float((m.loc[changed, "base_outcome"].values == m.loc[changed, "cand_outcome"].values).mean()) if changed.any() else np.nan,
        "same_gd_changed_rate": float((m.loc[changed, "base_gd"].values == m.loc[changed, "cand_gd"].values).mean()) if changed.any() else np.nan,
        "mean_pred_total": float((candidate_match["pred_a"] + candidate_match["pred_b"]).mean()),
        "mean_total_delta_vs_exp39": float(((candidate_match["pred_a"] + candidate_match["pred_b"]).mean()) - ((base_match["pred_a"] + base_match["pred_b"]).mean())),
        "max_pred_goal": int(candidate_match[["pred_a", "pred_b"]].max().max()),
    }


def transition_changes_df(base_match: pd.DataFrame, candidate_match: pd.DataFrame, label: str) -> pd.DataFrame:
    m = merge_match_predictions(base_match, candidate_match, "base", "cand")
    changed = (m["base_a"] != m["cand_a"]) | (m["base_b"] != m["cand_b"])
    m = m.loc[changed].copy()
    if len(m) == 0:
        return pd.DataFrame()
    m["old_a"] = m["base_a"].astype(int)
    m["old_b"] = m["base_b"].astype(int)
    m["new_a"] = m["cand_a"].astype(int)
    m["new_b"] = m["cand_b"].astype(int)
    m["old_score"] = m.apply(lambda r: scoreline(r["old_a"], r["old_b"]), axis=1)
    m["new_score"] = m.apply(lambda r: scoreline(r["new_a"], r["new_b"]), axis=1)
    m["transition"] = m["old_score"] + " -> " + m["new_score"]
    m["label"] = label
    return m[["label", "match_id", "old_a", "old_b", "new_a", "new_b", "old_score", "new_score", "transition", "gender", "tournament", "tournament_weight"]]


ALL_VARIANT_METRICS = []
ALL_SUBMISSION_CHECKS = []
ALL_SUBMISSION_CATALOG = []
ALL_CHANGED_DETAILS = []
ALL_CANDIDATE_MATCH = {"l0_exp39_final_reproduction": exp39_final_match.copy()}
ALL_CANDIDATE_SUB = {"l0_exp39_final_reproduction": exp39_l0_sub.copy()}


def register_match_candidate(label: str, match_df: pd.DataFrame, changes_df: pd.DataFrame = None, strict: bool = False):
    sub = match_to_submission(match_df)
    saved_sub, path, check = save_candidate(sub, label, strict=strict)
    rec = compare_match_to_base(submission_to_match(saved_sub, label=label), exp39_final_match, label)
    pc = pair_consistency_report(saved_sub)
    rec.update({
        "path": str(path),
        "pair_consistency": bool(pc["pair_consistency"]),
        "n_bad_pairs": int(pc["n_bad_pairs"]),
        "n_selected": int(len(changes_df)) if isinstance(changes_df, pd.DataFrame) else 0,
    })
    ALL_VARIANT_METRICS.append(rec)
    check2 = check.copy()
    check2.insert(0, "candidate_label", label)
    ALL_SUBMISSION_CHECKS.append(check2)
    ALL_SUBMISSION_CATALOG.append({"label": label, "path": str(path), "pair_consistency": bool(pc["pair_consistency"]), "n_bad_pairs": int(pc["n_bad_pairs"])})
    ALL_CANDIDATE_MATCH[label] = submission_to_match(saved_sub, label=label)
    ALL_CANDIDATE_SUB[label] = saved_sub
    if isinstance(changes_df, pd.DataFrame) and len(changes_df):
        tmp = changes_df.copy()
        tmp.insert(0, "candidate_label", label)
        ALL_CHANGED_DETAILS.append(tmp)
    return rec

# Register L0 metrics.
register_match_candidate("l0_exp39_final_reproduction", exp39_final_match, changes_df=pd.DataFrame(), strict=False)

# %% [markdown]
# # 10. L3 — EXP39 + Tiny EXP12K Outcome Patch
#
# Penjelasan bagian:
# Bagian ini memakai EXP39 sebagai base dan mengambil sinyal outcome dari EXP12K/user champion secara sangat kecil. Patch hanya berlaku jika EXP39 draw dan user champion decisive dengan support cukup.
#
# Output yang perlu dilihat:
# Cek `exp39_outcome_patch_grid.csv` dan changes file. Changed rate harus kecil, terutama cap 0.25%, 0.5%, dan 1%.

# %%
def build_exp39_user_outcome_patch_candidates(min_support=1, high_weight_support=3, segment_filter=None):
    if user_champion_match is None:
        return pd.DataFrame()
    m = merge_match_predictions(exp39_final_match, user_champion_match, "exp39", "user")
    m["exp39_outcome"] = outcome_array(m["exp39_a"], m["exp39_b"])
    m["user_outcome"] = outcome_array(m["user_a"], m["user_b"])
    m["exp39_total"] = m["exp39_a"] + m["exp39_b"]
    m["user_total"] = m["user_a"] + m["user_b"]
    m["total_delta"] = m["user_total"] - m["exp39_total"]
    m["exp39_gd"] = m["exp39_a"] - m["exp39_b"]
    m["user_gd"] = m["user_a"] - m["user_b"]
    m["gd_delta_abs"] = (m["user_gd"] - m["exp39_gd"]).abs()
    m["is_high_weight"] = m["tournament_weight"] >= 1.8
    m["is_low_weight"] = m["tournament_weight"] <= 0.96
    m["is_world_asian"] = m["tournament"].astype(str).str.lower().str.contains("world|asian|afc", regex=True)

    # Base condition: EXP39 predicts draw, user predicts decisive, total not too aggressive.
    cond = (m["exp39_outcome"] == 0) & (m["user_outcome"] != 0) & (m["total_delta"].abs() <= 1) & (m["gd_delta_abs"] <= 2)
    m = m.loc[cond].copy()
    if len(m) == 0:
        return pd.DataFrame()

    support_records = []
    for _, r in m.iterrows():
        s_score = support_for_score(r["match_id"], int(r["user_a"]), int(r["user_b"]), exclude_labels={"__exp39_base__"})
        s_out = support_for_outcome(r["match_id"], int(r["user_outcome"]), exclude_labels={"__exp39_base__"})
        rec = {**s_score, **s_out}
        support_records.append(rec)
    supp = pd.DataFrame(support_records, index=m.index)
    m = pd.concat([m, supp], axis=1)

    # Outcome support is more relevant than exact-score support for L3.
    m["support_effective"] = m["outcome_support"].astype(float)
    m["weighted_support_effective"] = m["outcome_weighted_support"].astype(float)
    m["old_a"] = m["exp39_a"].astype(int)
    m["old_b"] = m["exp39_b"].astype(int)
    m["new_a"] = m["user_a"].astype(int)
    m["new_b"] = m["user_b"].astype(int)
    m["old_score"] = m.apply(lambda r: scoreline(r["old_a"], r["old_b"]), axis=1)
    m["new_score"] = m.apply(lambda r: scoreline(r["new_a"], r["new_b"]), axis=1)
    m["transition"] = m["old_score"] + " -> " + m["new_score"]

    # Support gating.
    support_ok = m["support_effective"] >= float(min_support)
    high_ok = (~m["is_high_weight"]) | (m["support_effective"] >= float(high_weight_support))
    m = m.loc[support_ok & high_ok].copy()

    if segment_filter is not None and len(m):
        m = segment_filter(m).copy()

    if len(m) == 0:
        return pd.DataFrame()

    # Conservative ranking: support, weighted support, non-high-weight first, small total/gd delta.
    m["rank_high_weight_penalty"] = m["is_high_weight"].astype(int)
    m["rank_total_abs"] = m["total_delta"].abs()
    m = m.sort_values(
        ["support_effective", "weighted_support_effective", "rank_high_weight_penalty", "rank_total_abs", "gd_delta_abs", "tournament_weight"],
        ascending=[False, False, True, True, True, True],
    ).reset_index(drop=True)
    m["rank"] = np.arange(1, len(m) + 1)
    return m


def select_top_by_cap(candidate_table: pd.DataFrame, cap_rate: float):
    if candidate_table is None or len(candidate_table) == 0:
        return pd.DataFrame()
    n = int(math.floor(len(match_meta) * float(cap_rate)))
    n = max(0, min(n, len(candidate_table)))
    return candidate_table.head(n).copy()


patch_grid_rows = []
patch_change_frames = []

L3_CONFIGS = [
    ("l3_exp39_plus_exp12k_outcome_patch_cap0025", 0.0025, 1, 3, None),
    ("l3_exp39_plus_exp12k_outcome_patch_cap0050", 0.0050, 1, 3, None),
    ("l3_exp39_plus_exp12k_outcome_patch_cap0100", 0.0100, 1, 3, None),
    ("l3_exp39_plus_exp12k_outcome_patch_cap0150", 0.0150, 1, 3, None),
    ("l3_exp39_plus_exp12k_support2_cap0050", 0.0050, 2, 3, None),
    ("l3_exp39_plus_exp12k_support3_cap0050", 0.0050, 3, 3, None),
    ("l3_exp39_plus_exp12k_high_weight_support3_cap0050", 0.0050, 1, 3, None),
]

for label, cap, support, high_support, seg_filter in L3_CONFIGS:
    cand_table = build_exp39_user_outcome_patch_candidates(min_support=support, high_weight_support=high_support, segment_filter=seg_filter)
    selected = select_top_by_cap(cand_table, cap)
    patched_match = apply_match_changes(exp39_final_match, selected, label)
    rec = register_match_candidate(label, patched_match, selected, strict=False)
    rec.update({"cap_rate": cap, "support_threshold": support, "high_weight_support": high_support, "n_candidates_before_cap": int(len(cand_table))})
    patch_grid_rows.append(rec)
    if len(selected):
        tmp = selected.copy()
        tmp.insert(0, "variant", label)
        patch_change_frames.append(tmp)

exp39_outcome_patch_grid_df = pd.DataFrame(patch_grid_rows)
exp39_outcome_patch_grid_df.to_csv(SUM_DIR / "exp39_outcome_patch_grid.csv", index=False)
if patch_change_frames:
    exp39_outcome_patch_changes_df = pd.concat(patch_change_frames, ignore_index=True)
else:
    exp39_outcome_patch_changes_df = pd.DataFrame()
exp39_outcome_patch_changes_df.to_csv(SUM_DIR / "exp39_outcome_patch_changes.csv", index=False)
log_saved(SUM_DIR / "exp39_outcome_patch_grid.csv")
log_saved(SUM_DIR / "exp39_outcome_patch_changes.csv")
safe_display_df(exp39_outcome_patch_grid_df.sort_values("changed_rate_vs_exp39").head(20))

# %% [markdown]
# # 11. L4 — EXP39 + Same-Outcome Exact/GD Patch
#
# Penjelasan bagian:
# Bagian ini mencoba patch kecil yang tidak mengubah outcome. Tujuannya memperbaiki exact/GD tanpa mengganggu kekuatan outcome EXP39.
#
# Output yang perlu dilihat:
# Cek `exp39_same_outcome_patch_grid.csv`. Candidate harus mayoritas same-outcome dan changed rate sangat kecil.

# %%
def build_same_outcome_candidate_table(mode="exact", min_support=2):
    rows = []
    base_idx = exp39_final_match.set_index("match_id", drop=False)
    for mid, base_r in base_idx.iterrows():
        base_a = int(base_r["pred_a"])
        base_b = int(base_r["pred_b"])
        base_out = outcome_scalar(base_a, base_b)
        base_total = base_a + base_b
        base_gd = base_a - base_b
        scores = candidate_scores_for_match(mid, exclude_labels={"__exp39_base__"})
        if not scores:
            continue
        # Aggregate candidate scorelines.
        agg = defaultdict(lambda: {"support": 0, "weighted_support": 0.0, "labels": []})
        for s in scores:
            if s["outcome"] != base_out:
                continue
            if s["a"] == base_a and s["b"] == base_b:
                continue
            if abs(s["total"] - base_total) > 1:
                continue
            if abs(s["gd"] - base_gd) > 1:
                continue
            key = (s["a"], s["b"])
            agg[key]["support"] += 1
            agg[key]["weighted_support"] += float(s["weight"])
            agg[key]["labels"].append(s["label"])
        if not agg:
            continue
        # Choose highest support exact candidate.
        best_key, best_info = sorted(agg.items(), key=lambda kv: (kv[1]["support"], kv[1]["weighted_support"], -abs((kv[0][0] + kv[0][1]) - base_total), -abs((kv[0][0] - kv[0][1]) - base_gd)), reverse=True)[0]
        if best_info["support"] < min_support:
            continue
        new_a, new_b = best_key
        rows.append({
            "match_id": mid,
            "old_a": base_a,
            "old_b": base_b,
            "new_a": int(new_a),
            "new_b": int(new_b),
            "old_score": scoreline(base_a, base_b),
            "new_score": scoreline(new_a, new_b),
            "transition": scoreline(base_a, base_b) + " -> " + scoreline(new_a, new_b),
            "support": int(best_info["support"]),
            "weighted_support": float(best_info["weighted_support"]),
            "source_labels": ";".join(best_info["labels"][:8]),
            "gender": base_r["gender"],
            "tournament": base_r["tournament"],
            "tournament_weight": float(base_r["tournament_weight"]),
            "total_delta": int(new_a + new_b - base_total),
            "gd_delta": int((new_a - new_b) - base_gd),
        })
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["support", "weighted_support", "tournament_weight"], ascending=[False, False, True]).reset_index(drop=True)
        out["rank"] = np.arange(1, len(out) + 1)
    return out

same_outcome_table = build_same_outcome_candidate_table(min_support=2)
L4_CONFIGS = [
    ("l4_exp39_plus_same_outcome_exact_cap0025", 0.0025, same_outcome_table),
    ("l4_exp39_plus_same_outcome_exact_cap0050", 0.0050, same_outcome_table),
    ("l4_exp39_plus_gd_patch_cap0025", 0.0025, same_outcome_table.loc[same_outcome_table["gd_delta"].abs() > 0].copy() if len(same_outcome_table) else same_outcome_table),
    ("l4_exp39_plus_gd_patch_cap0050", 0.0050, same_outcome_table.loc[same_outcome_table["gd_delta"].abs() > 0].copy() if len(same_outcome_table) else same_outcome_table),
]

l4_rows = []
l4_changes = []
for label, cap, table in L4_CONFIGS:
    selected = select_top_by_cap(table, cap)
    patched_match = apply_match_changes(exp39_final_match, selected, label)
    rec = register_match_candidate(label, patched_match, selected, strict=False)
    rec.update({"cap_rate": cap, "n_candidates_before_cap": int(len(table)) if isinstance(table, pd.DataFrame) else 0})
    l4_rows.append(rec)
    if len(selected):
        tmp = selected.copy()
        tmp.insert(0, "variant", label)
        l4_changes.append(tmp)

exp39_same_outcome_patch_grid_df = pd.DataFrame(l4_rows)
exp39_same_outcome_patch_grid_df.to_csv(SUM_DIR / "exp39_same_outcome_patch_grid.csv", index=False)
exp39_same_outcome_patch_changes_df = pd.concat(l4_changes, ignore_index=True) if l4_changes else pd.DataFrame()
exp39_same_outcome_patch_changes_df.to_csv(SUM_DIR / "exp39_same_outcome_patch_changes.csv", index=False)
log_saved(SUM_DIR / "exp39_same_outcome_patch_grid.csv")
log_saved(SUM_DIR / "exp39_same_outcome_patch_changes.csv")
safe_display_df(exp39_same_outcome_patch_grid_df)

# %% [markdown]
# # 12. L5 — Segment-Aware Hybrid Patch
#
# Penjelasan bagian:
# Bagian ini menerapkan patch outcome EXP12K di atas EXP39 hanya untuk segmen tertentu seperti low-weight, M-only, W-only, atau high-weight support ketat. Ini penting karena EXP39 punya segment router.
#
# Output yang perlu dilihat:
# Cek `segment_aware_hybrid_grid.csv` dan changes. Lihat changed_M/W, changed_high_weight, dan mean total setelah patch.

# %%
def filter_low_weight(df):
    return df.loc[df["is_low_weight"]].copy()


def filter_non_high_weight(df):
    return df.loc[~df["is_high_weight"]].copy()


def filter_friendly(df):
    return df.loc[df["tournament"].astype(str).str.lower().str.contains("friendly", na=False)].copy()


def filter_m_only(df):
    return df.loc[df["gender"].astype(str).str.upper().eq("M")].copy()


def filter_w_only(df):
    return df.loc[df["gender"].astype(str).str.upper().eq("W")].copy()


def filter_world_asian(df):
    return df.loc[df["is_world_asian"]].copy()


def filter_default_weight(df):
    return df.loc[(~df["is_high_weight"]) & (~df["is_low_weight"])].copy()

SEGMENT_CONFIGS = [
    ("l5_exp39_plus_patch_low_weight_cap0050", 0.0050, 1, 3, filter_low_weight),
    ("l5_exp39_plus_patch_non_high_weight_cap0050", 0.0050, 1, 3, filter_non_high_weight),
    ("l5_exp39_plus_patch_friendly_cap0050", 0.0050, 1, 3, filter_friendly),
    ("l5_exp39_plus_patch_M_only_cap0050", 0.0050, 1, 3, filter_m_only),
    ("l5_exp39_plus_patch_W_only_cap0025", 0.0025, 1, 3, filter_w_only),
    ("l5_exp39_plus_patch_world_asian_support3_cap0025", 0.0025, 3, 3, filter_world_asian),
    ("l5_exp39_plus_patch_default_weight_cap0050", 0.0050, 1, 3, filter_default_weight),
]

l5_rows = []
l5_changes = []
for label, cap, support, high_support, seg_filter in SEGMENT_CONFIGS:
    table = build_exp39_user_outcome_patch_candidates(min_support=support, high_weight_support=high_support, segment_filter=seg_filter)
    selected = select_top_by_cap(table, cap)
    patched_match = apply_match_changes(exp39_final_match, selected, label)
    rec = register_match_candidate(label, patched_match, selected, strict=False)
    rec.update({
        "cap_rate": cap,
        "segment_name": label,
        "n_candidates_before_cap": int(len(table)),
        "changed_M": int((selected["gender"].astype(str).str.upper().eq("M")).sum()) if len(selected) else 0,
        "changed_W": int((selected["gender"].astype(str).str.upper().eq("W")).sum()) if len(selected) else 0,
        "changed_high_weight": int((selected["is_high_weight"]).sum()) if len(selected) and "is_high_weight" in selected.columns else 0,
        "changed_low_weight": int((selected["is_low_weight"]).sum()) if len(selected) and "is_low_weight" in selected.columns else 0,
    })
    l5_rows.append(rec)
    if len(selected):
        tmp = selected.copy()
        tmp.insert(0, "variant", label)
        l5_changes.append(tmp)

segment_aware_hybrid_grid_df = pd.DataFrame(l5_rows)
segment_aware_hybrid_grid_df.to_csv(SUM_DIR / "segment_aware_hybrid_grid.csv", index=False)
segment_aware_hybrid_changes_df = pd.concat(l5_changes, ignore_index=True) if l5_changes else pd.DataFrame()
segment_aware_hybrid_changes_df.to_csv(SUM_DIR / "segment_aware_hybrid_changes.csv", index=False)
log_saved(SUM_DIR / "segment_aware_hybrid_grid.csv")
log_saved(SUM_DIR / "segment_aware_hybrid_changes.csv")
safe_display_df(segment_aware_hybrid_grid_df)

# %% [markdown]
# # 13. L6 — EXP39 Export Candidate Router
#
# Penjelasan bagian:
# Bagian ini mencoba router internal antar export EXP39, tanpa memakai EXP12K. Base tetap SAFE_FINAL, lalu scoreline diganti sedikit jika beberapa export EXP39 sepakat pada scoreline berbeda yang low-risk.
#
# Output yang perlu dilihat:
# Cek `exp39_export_router_grid.csv`, `exp39_export_agreement_matrix.csv`, dan changes. Jika L6 menang, improvement datang dari internal EXP39 candidates.

# %%
def build_exp39_export_agreement_table(mode="exact", min_support=2, same_outcome_only=True):
    export_labels = [lab for lab in EXP39_EXPORT_MATCH.keys() if lab in candidate_match_pool]
    base_idx = exp39_final_match.set_index("match_id", drop=False)
    rows = []
    agreement_rows = []
    for mid, base_r in base_idx.iterrows():
        base_a = int(base_r["pred_a"])
        base_b = int(base_r["pred_b"])
        base_out = outcome_scalar(base_a, base_b)
        base_total = base_a + base_b
        base_gd = base_a - base_b
        counts = defaultdict(lambda: {"n": 0, "w": 0.0, "labels": []})
        for lab in export_labels:
            idx = CANDIDATE_MATCH_POOL_INDEX.get(lab)
            if idx is None or mid not in idx.index:
                continue
            r = idx.loc[mid]
            if isinstance(r, pd.DataFrame):
                r = r.iloc[0]
            a = int(r["pred_a"])
            b = int(r["pred_b"])
            if a == base_a and b == base_b:
                continue
            out = outcome_scalar(a, b)
            total = a + b
            gd = a - b
            if same_outcome_only and out != base_out:
                continue
            if abs(total - base_total) > 1:
                continue
            if abs(gd - base_gd) > 1:
                continue
            key = (a, b)
            counts[key]["n"] += 1
            counts[key]["w"] += candidate_reliability_weight(lab)
            counts[key]["labels"].append(lab)
        if counts:
            best_key, info = sorted(counts.items(), key=lambda kv: (kv[1]["n"], kv[1]["w"]), reverse=True)[0]
            agreement_rows.append({"match_id": mid, "best_score": scoreline(*best_key), "support": info["n"], "weighted_support": info["w"], "labels": ";".join(info["labels"][:8])})
            if info["n"] >= min_support:
                new_a, new_b = best_key
                rows.append({
                    "match_id": mid,
                    "old_a": base_a,
                    "old_b": base_b,
                    "new_a": int(new_a),
                    "new_b": int(new_b),
                    "old_score": scoreline(base_a, base_b),
                    "new_score": scoreline(new_a, new_b),
                    "transition": scoreline(base_a, base_b) + " -> " + scoreline(new_a, new_b),
                    "support": int(info["n"]),
                    "weighted_support": float(info["w"]),
                    "source_labels": ";".join(info["labels"][:8]),
                    "gender": base_r["gender"],
                    "tournament": base_r["tournament"],
                    "tournament_weight": float(base_r["tournament_weight"]),
                    "total_delta": int(new_a + new_b - base_total),
                    "gd_delta": int((new_a - new_b) - base_gd),
                })
    table = pd.DataFrame(rows)
    if len(table):
        table = table.sort_values(["support", "weighted_support", "tournament_weight"], ascending=[False, False, True]).reset_index(drop=True)
        table["rank"] = np.arange(1, len(table) + 1)
    agreement = pd.DataFrame(agreement_rows)
    return table, agreement

l6_rows = []
l6_changes = []
agreement_frames = []
L6_CONFIGS = [
    ("l6_exp39_export_exact_majority_cap0025", 0.0025, 2, True),
    ("l6_exp39_export_exact_majority_cap0050", 0.0050, 2, True),
    ("l6_exp39_export_same_outcome_majority_cap0050", 0.0050, 2, True),
    ("l6_exp39_export_low_total_router_cap0050", 0.0050, 2, True),
]
for label, cap, min_support, same_out in L6_CONFIGS:
    table, agree = build_exp39_export_agreement_table(min_support=min_support, same_outcome_only=same_out)
    if "low_total" in label and len(table):
        table = table.loc[table["total_delta"] <= 0].copy()
    selected = select_top_by_cap(table, cap)
    patched_match = apply_match_changes(exp39_final_match, selected, label)
    rec = register_match_candidate(label, patched_match, selected, strict=False)
    rec.update({"cap_rate": cap, "min_support": min_support, "n_candidates_before_cap": int(len(table)), "n_exp39_candidates_used": len(EXP39_EXPORT_MATCH)})
    l6_rows.append(rec)
    if len(selected):
        tmp = selected.copy()
        tmp.insert(0, "variant", label)
        l6_changes.append(tmp)
    if len(agree):
        agree = agree.copy()
        agree.insert(0, "variant", label)
        agreement_frames.append(agree)

exp39_export_router_grid_df = pd.DataFrame(l6_rows)
exp39_export_router_changes_df = pd.concat(l6_changes, ignore_index=True) if l6_changes else pd.DataFrame()
exp39_export_agreement_matrix_df = pd.concat(agreement_frames, ignore_index=True) if agreement_frames else pd.DataFrame()
exp39_export_router_grid_df.to_csv(SUM_DIR / "exp39_export_router_grid.csv", index=False)
exp39_export_router_changes_df.to_csv(SUM_DIR / "exp39_export_router_changes.csv", index=False)
exp39_export_agreement_matrix_df.to_csv(SUM_DIR / "exp39_export_agreement_matrix.csv", index=False)
log_saved(SUM_DIR / "exp39_export_router_grid.csv")
log_saved(SUM_DIR / "exp39_export_router_changes.csv")
log_saved(SUM_DIR / "exp39_export_agreement_matrix.csv")
safe_display_df(exp39_export_router_grid_df)

# %% [markdown]
# # 14. L7 — Low-Total Guard / Anti-Damage Check
#
# Penjelasan bagian:
# Bagian ini membuat guarded variant agar hybrid tidak menaikkan mean total terlalu jauh dari EXP39. EXP39 kuat kemungkinan karena low-total calibration, jadi guard ini penting.
#
# Output yang perlu dilihat:
# Cek `low_total_guard_grid.csv`: mean_total_before/after, n_changed_kept, dan n_changed_dropped.

# %%
def guarded_changes_by_mean_total(base_match, candidate_changes, guard_threshold):
    if candidate_changes is None or len(candidate_changes) == 0:
        return pd.DataFrame(), 0
    base_mean = float((base_match["pred_a"] + base_match["pred_b"]).mean())
    kept = []
    current = base_match.copy()
    for _, r in candidate_changes.iterrows():
        trial_change = pd.DataFrame([r])
        trial = apply_match_changes(current, trial_change, "trial")
        trial_mean = float((trial["pred_a"] + trial["pred_b"]).mean())
        if trial_mean <= base_mean + float(guard_threshold):
            kept.append(r)
            current = trial
    kept_df = pd.DataFrame(kept)
    n_dropped = int(len(candidate_changes) - len(kept_df))
    return kept_df, n_dropped

# Use L3 cap0100 as base changes for guard, if available.
base_l7_table = build_exp39_user_outcome_patch_candidates(min_support=1, high_weight_support=3)
base_l7_selected = select_top_by_cap(base_l7_table, 0.0100)
L7_CONFIGS = [
    ("l7_exp39_patch_total_guard_005", 0.005),
    ("l7_exp39_patch_total_guard_010", 0.010),
    ("l7_exp39_patch_total_guard_020", 0.020),
]

l7_rows = []
for label, guard in L7_CONFIGS:
    kept, dropped = guarded_changes_by_mean_total(exp39_final_match, base_l7_selected, guard)
    patched_match = apply_match_changes(exp39_final_match, kept, label)
    rec = register_match_candidate(label, patched_match, kept, strict=False)
    rec.update({
        "total_guard_threshold": guard,
        "mean_total_before": float((exp39_final_match["pred_a"] + exp39_final_match["pred_b"]).mean()),
        "mean_total_after": float((patched_match["pred_a"] + patched_match["pred_b"]).mean()),
        "n_changed_kept": int(len(kept)),
        "n_changed_dropped": int(dropped),
    })
    l7_rows.append(rec)

low_total_guard_grid_df = pd.DataFrame(l7_rows)
low_total_guard_grid_df.to_csv(SUM_DIR / "low_total_guard_grid.csv", index=False)
log_saved(SUM_DIR / "low_total_guard_grid.csv")
safe_display_df(low_total_guard_grid_df)

# %% [markdown]
# # 15. Variant Metrics, Changed Prediction Analysis, dan Final Decision GT-Free
#
# Penjelasan bagian:
# Bagian ini menyimpan semua summary GT-free: variant metrics, changed prediction analysis, submission check, catalog, dan final decision. `best_safe` dipilih berdasarkan sanity, bukan local GT.
#
# Output yang perlu dilihat:
# Cek `final_decision.csv`: selected_by_pipeline dan recommended_for_local_audit harus jelas. Champion sebenarnya dibaca dari Section 99 local audit.

# %%
variant_metrics_df = pd.DataFrame(ALL_VARIANT_METRICS)
if len(variant_metrics_df):
    variant_metrics_df = variant_metrics_df.drop_duplicates(subset=["label"], keep="last")
variant_metrics_df.to_csv(SUM_DIR / "variant_metrics.csv", index=False)

changed_prediction_analysis_df = pd.concat(ALL_CHANGED_DETAILS, ignore_index=True) if ALL_CHANGED_DETAILS else pd.DataFrame()
changed_prediction_analysis_df.to_csv(SUM_DIR / "changed_prediction_analysis.csv", index=False)

submission_check_df = pd.concat(ALL_SUBMISSION_CHECKS, ignore_index=True) if ALL_SUBMISSION_CHECKS else pd.DataFrame()
submission_check_df.to_csv(SUM_DIR / "submission_check.csv", index=False)

submission_catalog_df = pd.DataFrame(ALL_SUBMISSION_CATALOG)
submission_catalog_df.to_csv(SUM_DIR / "submission_catalog.csv", index=False)

log_saved(SUM_DIR / "variant_metrics.csv")
log_saved(SUM_DIR / "changed_prediction_analysis.csv")
log_saved(SUM_DIR / "submission_check.csv")
log_saved(SUM_DIR / "submission_catalog.csv")

safe_display_df(variant_metrics_df.sort_values(["pair_consistency", "changed_rate_vs_exp39"], ascending=[False, True]).head(30))

# Conservative best_safe order. This is GT-free and intentionally not claiming local-audit champion.
SAFE_ORDER = [
    "l0_exp39_final_reproduction",
    "l3_exp39_plus_exp12k_outcome_patch_cap0025",
    "l6_exp39_export_exact_majority_cap0025",
    "l4_exp39_plus_same_outcome_exact_cap0025",
]
selected_label = None
for lab in SAFE_ORDER:
    if lab in ALL_CANDIDATE_SUB:
        chk = validation_checks(ALL_CANDIDATE_SUB[lab])
        if bool(chk["passed"].all()):
            selected_label = lab
            break
if selected_label is None:
    selected_label = "l0_exp39_final_reproduction"

best_safe_sub = ALL_CANDIDATE_SUB[selected_label]
best_safe_match = ALL_CANDIDATE_MATCH[selected_label]
best_safe_saved, best_safe_path, best_safe_check = save_candidate(best_safe_sub, "best_safe", strict=STRICT_FINAL)

final_decision = {
    "selected_by_pipeline": selected_label,
    "selected_by_sanity": selected_label,
    "recommended_for_local_audit": "Audit all L0/L1/L3/L4/L5/L6/L7 candidates in Section 99; best_safe is not a GT claim.",
    "main_submission_path": str(best_safe_path),
    "exp39_source_path": str(exp39_final_path),
    "user_champion_source_path": str(user_champion_path) if user_champion_path is not None else "NOT_FOUND",
    "notes": "GT-free selection defaults to EXP39 final or the most conservative valid patch. Champion is determined manually from local_gt_audit_awmae_rank.csv.",
}
final_decision_df = pd.DataFrame([{"key": k, "value": str(v)} for k, v in final_decision.items()])
final_decision_df.to_csv(SUM_DIR / "final_decision.csv", index=False)
with open(SUM_DIR / "final_decision.json", "w", encoding="utf-8") as f:
    json.dump(final_decision, f, indent=2)
log_saved(SUM_DIR / "final_decision.csv")
log_saved(SUM_DIR / "final_decision.json")
safe_display_df(final_decision_df)

# %% [markdown]
# # 99. OPTIONAL LOCAL GT AUDIT — DELETE BEFORE CLEAN SUBMISSION NOTEBOOK
#
# Penjelasan bagian:
# Cell ini membaca `ground_truth_bersih.csv` dan semua `submission_*.csv` yang sudah dibuat, lalu menghitung AW-MAE weighted local audit. Cell ini hanya untuk audit lokal setelah semua submission selesai dibuat.
#
# Output yang perlu dilihat:
# Lihat `local_gt_audit_awmae_rank.csv`. Ranking ini membantu membaca apakah EXP39 final sekitar 2.959616 dan apakah candidate EXP12L mengalahkannya. Cell ini tidak boleh dipakai untuk training, feature engineering, atau automatic selection di pipeline utama.

# %%
RUN_LOCAL_GT_AUDIT = RUN_LOCAL_GT_AUDIT_DEFAULT

if RUN_LOCAL_GT_AUDIT:
    log_section("Section 99 Local GT Audit")

    def find_gt_path():
        candidates = [
            PROJECT_ROOT / "data" / "ground_truth_bersih.csv",
            PROJECT_ROOT / "dataset" / "ground_truth_bersih.csv",
            PROJECT_ROOT / "ground_truth_bersih.csv",
            Path.cwd() / "data" / "ground_truth_bersih.csv",
            Path.cwd().parent / "data" / "ground_truth_bersih.csv",
            Path("/mnt/data") / "ground_truth_bersih.csv",
        ]
        for p in candidates:
            if p.exists():
                return p
        return None

    GT_PATH = find_gt_path()
    if GT_PATH is None:
        raise FileNotFoundError("ground_truth_bersih.csv not found. Put it in data/ or dataset/ for local audit.")

    log_info(f"GT_PATH = {GT_PATH}")
    gt = pd.read_csv(GT_PATH)
    gt_id_col = "Id" if "Id" in gt.columns else ("id" if "id" in gt.columns else None)
    if gt_id_col is None:
        raise KeyError("ground_truth_bersih.csv missing Id/id column")
    gt = gt.rename(columns={gt_id_col: "Id"}).copy()
    required_gt = ["Id", "team_goals", "opp_goals"]
    missing_gt = [c for c in required_gt if c not in gt.columns]
    if missing_gt:
        raise KeyError(f"ground_truth_bersih.csv missing columns: {missing_gt}")
    gt = gt[required_gt].copy()
    gt["Id"] = gt["Id"].astype(str)
    gt = gt.rename(columns={"team_goals": "true_team_goals", "opp_goals": "true_opp_goals"})

    test_weight_df = test_raw[["Id", "tournament"]].copy()
    test_weight_df["Id"] = test_weight_df["Id"].astype(str)
    test_weight_df["weight"] = test_weight_df["tournament"].apply(get_tournament_weight).astype(float)
    gt = gt.merge(test_weight_df[["Id", "weight"]], on="Id", how="left")
    gt["weight"] = gt["weight"].fillna(1.20).astype(float)

    def awmae_components(y_team, y_opp, p_team, p_opp, weights):
        y_team = np.asarray(y_team, dtype=int)
        y_opp = np.asarray(y_opp, dtype=int)
        p_team = np.asarray(p_team, dtype=int)
        p_opp = np.asarray(p_opp, dtype=int)
        weights = np.asarray(weights, dtype=float)

        base = (np.abs(y_team - p_team) + np.abs(y_opp - p_opp)) / 2.0
        exact_miss = ~((y_team == p_team) & (y_opp == p_opp))
        true_out = outcome_array(y_team, y_opp)
        pred_out = outcome_array(p_team, p_opp)
        outcome_miss = true_out != pred_out
        gd_miss = (y_team - y_opp) != (p_team - p_opp)
        raw = (
            base
            + EXACT_PENALTY * exact_miss.astype(float)
            + OUTCOME_PENALTY * outcome_miss.astype(float)
            + GD_PENALTY * gd_miss.astype(float)
        )
        mult = np.where(outcome_miss, WRONG_OUTCOME_MULTIPLIER, 1.0)
        loss = ((raw * mult) ** NONLINEAR_POWER) * weights
        return {
            "awmae": float(loss.sum() / weights.sum()),
            "base_mae": float(base.mean()),
            "exact_rate": float((~exact_miss).mean()),
            "outcome_rate": float((~outcome_miss).mean()),
            "gd_rate": float((~gd_miss).mean()),
            "team_bias": float((p_team - y_team).mean()),
            "opp_bias": float((p_opp - y_opp).mean()),
            "total_goal_bias": float(((p_team + p_opp) - (y_team + y_opp)).mean()),
            "mean_pred_total": float((p_team + p_opp).mean()),
            "max_pred_goal": int(max(p_team.max(), p_opp.max())),
        }

    # Search EXP12L plus relevant previous outputs. Avoid summary files.
    search_dirs = [SUB_DIR, OUT_DIR / "submissions", OUTPUT_ROOT / "exp12l_exp39_integration_hybrid_patch"]
    for exp_name in [
        "exp12k_champion_delta_surgery",
        "exp12j_one_one_draw_to_win_refinement",
        "exp12i_cap0290_marginal_band_surgery",
        "exp12h_outcome_ranking_damage_control",
        "exp12g_outcome_booster_refinement",
        "exp12f_weighted_pair_router_micro_booster",
        "exp12e_pair_repair_micro_audit",
        "exp39",
    ]:
        p = OUTPUT_ROOT / exp_name
        if p.exists():
            search_dirs.append(p)

    # Also include /mnt/data and project root for exported EXP39 final if copied there.
    search_dirs += [PROJECT_ROOT, Path("/mnt/data")]
    search_dirs = [p for p in search_dirs if p.exists()]

    submission_paths = []
    seen = set()
    for d in search_dirs:
        try:
            for p in d.rglob("submission_*.csv"):
                p = p.resolve()
                if p in seen:
                    continue
                seen.add(p)
                if not is_submission_candidate_path(p):
                    continue
                submission_paths.append(p)
        except Exception:
            pass
    submission_paths = sorted(submission_paths, key=lambda x: str(x))
    log_info(f"Found {len(submission_paths)} submission files for local audit")

    rows = []
    bad_rows = []
    for path in submission_paths:
        try:
            sub = pd.read_csv(path)
            sub = normalize_submission_df(sub, require_all=True)
            merged = gt.merge(
                sub.rename(columns={"team_goals": "pred_team_goals", "opp_goals": "pred_opp_goals"}),
                on="Id",
                how="left",
                validate="one_to_one",
            )
            if merged[["pred_team_goals", "pred_opp_goals"]].isna().any().any():
                bad_rows.append({"file_name": path.name, "path": str(path), "reason": "missing predictions after merge"})
                continue
            metric = awmae_components(
                merged["true_team_goals"].values,
                merged["true_opp_goals"].values,
                merged["pred_team_goals"].values,
                merged["pred_opp_goals"].values,
                merged["weight"].values,
            )
            pc = pair_consistency_report(sub)
            metric.update({
                "pair_consistency": bool(pc["pair_consistency"]),
                "n_bad_pairs": int(pc["n_bad_pairs"]),
                "file_name": path.name,
                "path": str(path),
                "n_rows": len(merged),
            })
            rows.append(metric)
        except Exception as e:
            bad_rows.append({"file_name": path.name, "path": str(path), "reason": repr(e)})

    local_gt_audit_rank_df = pd.DataFrame(rows)
    if len(local_gt_audit_rank_df) == 0:
        raise RuntimeError("No valid submission could be audited.")
    local_gt_audit_rank_df = local_gt_audit_rank_df.sort_values("awmae").reset_index(drop=True)
    local_gt_audit_rank_df.to_csv(SUM_DIR / "local_gt_audit_awmae_rank.csv", index=False)
    log_saved(SUM_DIR / "local_gt_audit_awmae_rank.csv")
    display(local_gt_audit_rank_df.head(30))

    if bad_rows:
        bad_df = pd.DataFrame(bad_rows)
        bad_df.to_csv(SUM_DIR / "local_gt_audit_bad_files.csv", index=False)
        log_saved(SUM_DIR / "local_gt_audit_bad_files.csv")
        display(bad_df.head(30))
else:
    log_info("RUN_LOCAL_GT_AUDIT=False; Section 99 skipped.")
