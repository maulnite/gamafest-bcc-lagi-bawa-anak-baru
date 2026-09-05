# %% [markdown]
# # 00. EXP12N — Total-Plus Ranked Micro Patch Router
#
# Penjelasan bagian:
# EXP12N adalah post-processing refinement di atas EXP12N. Fokusnya membuat total-plus same-outcome micro router di atas EXP39 final, bukan training model baru.
#
# Output yang perlu dilihat:
# Pastikan N0 mereproduce EXP12N champion, `total_plus_candidate_table.csv` tersimpan, semua candidate pair-consistent, dan Section 99 local audit menjadi satu-satunya bagian yang membaca `ground_truth_bersih.csv`.

# %% [markdown]
# # 01. Setup, Project Root, Logging, dan Guardrail
#
# Penjelasan bagian:
# Bagian ini menyiapkan import, seed, logging, output folder, dan project root resolver. Output dipaksa ke `PROJECT_ROOT/outputs/exp12n...` supaya tidak nyasar ke `notebook/outputs/`.
#
# Output yang perlu dilihat:
# Cek `PROJECT_ROOT`, `OUT_DIR`, `SUB_DIR`, dan `SUM_DIR`. Ground truth belum dibaca di sini.

# %%
import os, re, json, math, random, warnings, shutil
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

try:
    from IPython.display import display
except Exception:
    def display(x): print(x)

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
    for cand in [cwd, cwd.parent, cwd.parent.parent]:
        if (cand / "data").exists() or (cand / "dataset").exists():
            return cand
    if Path("/mnt/data").exists() and ((Path("/mnt/data") / "test.csv").exists() or (Path("/mnt/data") / "sample submission.csv").exists()):
        return Path("/mnt/data")
    return cwd

PROJECT_ROOT = infer_project_root()
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
EXP12N_VARIANT = os.environ.get("EXP12N_VARIANT", "n1_total_plus_segment_router")
OUT_DIR = OUTPUT_ROOT / "exp12n_total_plus_ranked_micro_patch_router" / EXP12N_VARIANT
PRED_DIR = OUT_DIR / "predictions"
SUB_DIR = OUT_DIR / "submissions"
SUM_DIR = OUT_DIR / "summaries"
FIG_DIR = OUT_DIR / "figures"

EXP12N_CLEAN_OUTDIR = os.environ.get("EXP12N_CLEAN_OUTDIR", "0").strip().lower() in {"1", "true", "yes", "y"}
if EXP12N_CLEAN_OUTDIR and OUT_DIR.exists():
    log_info(f"Cleaning previous EXP12N output folder: {OUT_DIR}")
    shutil.rmtree(OUT_DIR)

for d in [PRED_DIR, SUB_DIR, SUM_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RUN_LOCAL_GT_AUDIT_DEFAULT = os.environ.get("EXP12N_RUN_LOCAL_GT_AUDIT", "1").strip().lower() in {"1", "true", "yes", "y"}
MAX_GOAL_SANITY = 40

log_section("Setup")
log_info(f"PROJECT_ROOT = {PROJECT_ROOT}")
log_info(f"OUTPUT_ROOT = {OUTPUT_ROOT}")
log_info(f"OUT_DIR = {OUT_DIR}")
log_info(f"SUB_DIR = {SUB_DIR}")
log_info(f"SUM_DIR = {SUM_DIR}")
log_info(f"EXP12N_VARIANT = {EXP12N_VARIANT}")
log_info("GT guardrail: ground_truth_bersih.csv hanya dibaca di Section 99 optional local audit")

# %% [markdown]
# # 02. Robust Data Finder dan Basic Audit
#
# Penjelasan bagian:
# Bagian ini mencari `train.csv`, `test.csv`, dan sample submission dengan robust. Finder menghindari folder outputs agar tidak salah mengambil submission eksperimen sebagai sample.
#
# Output yang perlu dilihat:
# Path train/test/sample harus menunjuk dataset asli. Row count sample harus sama dengan test.

# %%
SAMPLE_NAMES = ["sample_submission.csv", "sample submission.csv", "samplesubmission.csv", "sample-submission.csv", "submission_sample.csv"]

def is_forbidden_data_path(p: Path) -> bool:
    parts = {x.lower() for x in p.parts}
    name = p.name.lower()
    return ("outputs" in parts) or name.startswith("submission_exp") or ("leaderboard" in name)

def candidate_data_dirs():
    dirs = [
        PROJECT_ROOT / "data", PROJECT_ROOT / "dataset", PROJECT_ROOT,
        PROJECT_ROOT.parent / "data", PROJECT_ROOT.parent / "dataset",
        Path.cwd(), Path.cwd() / "data", Path.cwd() / "dataset",
        Path.cwd().parent / "data", Path.cwd().parent / "dataset",
        Path("/mnt/data"),
    ]
    out = []
    for d in dirs:
        try: d = d.resolve()
        except Exception: continue
        if d.exists() and d not in out: out.append(d)
    return out

def find_data_file(names, recursive=True) -> Path:
    if isinstance(names, str): names = [names]
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
                    if (not is_forbidden_data_path(p)) and p.name.lower() in lower_names:
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

sample_id_col = "Id" if "Id" in sample_submission.columns else ("id" if "id" in sample_submission.columns else sample_submission.columns[0])
test_id_col = "Id" if "Id" in test_raw.columns else ("id" if "id" in test_raw.columns else sample_id_col)
sample_submission = sample_submission.rename(columns={sample_id_col: "Id"}).copy()
test_raw = test_raw.rename(columns={test_id_col: "Id"}).copy()

if "match_id" not in test_raw.columns:
    test_raw = test_raw.copy()
    test_raw["match_id"] = np.arange(len(test_raw)) // 2
    log_info("test.csv tidak punya match_id; fallback membuat match_id dari pasangan dua row berurutan")

for c in ["tournament", "gender"]:
    if c not in test_raw.columns:
        test_raw[c] = "unknown"

input_audit_df = pd.DataFrame([
    {"name": "TRAIN_PATH", "value": str(TRAIN_PATH)},
    {"name": "TEST_PATH", "value": str(TEST_PATH)},
    {"name": "SAMPLE_PATH", "value": str(SAMPLE_PATH)},
    {"name": "train_shape", "value": str(train_raw.shape)},
    {"name": "test_shape", "value": str(test_raw.shape)},
    {"name": "sample_shape", "value": str(sample_submission.shape)},
])
input_audit_df.to_csv(SUM_DIR / "input_audit.csv", index=False)
log_section("Input Audit")
safe_display_df(input_audit_df)
log_check("sample row count equals test row count", len(sample_submission) == len(test_raw))

# %% [markdown]
# # 03. Metric Helper, Match Mapping, dan Submission Validation
#
# Penjelasan bagian:
# Bagian ini berisi helper tournament weight, outcome, row-level to match-level conversion, match-level to row-level submission mapping, pair consistency, dan validation check.
#
# Output yang perlu dilihat:
# Jumlah match meta harus sekitar setengah test rows. Final candidate wajib pair-consistent.

# %%
EXACT_PENALTY = 0.30
OUTCOME_PENALTY = 0.25
GD_PENALTY = 0.15
WRONG_OUTCOME_MULTIPLIER = 1.50
NONLINEAR_POWER = 1.50

def get_tournament_weight(tournament: str) -> float:
    t = str(tournament).lower().strip()
    if "fifa world cup" in t or t == "world cup":
        return 2.00
    if "afc championship" in t or "afc asian cup" in t or "asian cup" in t:
        return 1.80
    if "friendly" in t:
        return 0.96
    return 1.20

def outcome_scalar(a, b) -> int:
    a, b = int(a), int(b)
    if a > b: return 1
    if a < b: return -1
    return 0

def outcome_array(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return np.where(a > b, 1, np.where(a < b, -1, 0))

def slugify_label(label: str, max_len=140) -> str:
    s = str(label).strip().lower().replace("/", "_").replace("\\", "_")
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    if len(s) > max_len: s = s[:max_len].rstrip("_")
    return s or "candidate"

def normalize_submission_df(df: pd.DataFrame, require_all=True) -> pd.DataFrame:
    sub = df.copy()
    id_col = "Id" if "Id" in sub.columns else ("id" if "id" in sub.columns else sub.columns[0])
    sub = sub.rename(columns={id_col: "Id"}).copy()
    required = ["Id", "team_goals", "opp_goals"]
    missing = [c for c in required if c not in sub.columns]
    if missing: raise KeyError(f"submission missing columns: {missing}")
    sub = sub[required].copy()
    sub["Id"] = sub["Id"].astype(str)
    sub["team_goals"] = pd.to_numeric(sub["team_goals"], errors="raise").round().astype(int).clip(lower=0)
    sub["opp_goals"] = pd.to_numeric(sub["opp_goals"], errors="raise").round().astype(int).clip(lower=0)

    sample_ids = sample_submission[["Id"]].copy()
    sample_ids["Id"] = sample_ids["Id"].astype(str)
    aligned = sample_ids.merge(sub, on="Id", how="left", validate="one_to_one")
    if require_all and aligned[["team_goals", "opp_goals"]].isna().any().any():
        missing_ids = aligned.loc[aligned[["team_goals", "opp_goals"]].isna().any(axis=1), "Id"].head(10).tolist()
        raise RuntimeError(f"missing predictions after align, examples: {missing_ids}")
    aligned["team_goals"] = pd.to_numeric(aligned["team_goals"], errors="coerce").round().astype("Int64")
    aligned["opp_goals"] = pd.to_numeric(aligned["opp_goals"], errors="coerce").round().astype("Int64")
    if require_all:
        aligned["team_goals"] = aligned["team_goals"].astype(int).clip(lower=0)
        aligned["opp_goals"] = aligned["opp_goals"].astype(int).clip(lower=0)
    return aligned[["Id", "team_goals", "opp_goals"]]

def build_match_meta(test_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for match_id, g in test_df.reset_index(drop=True).groupby("match_id", sort=False):
        if len(g) != 2:
            if len(g) < 2: continue
            g = g.iloc[:2].copy()
        r0, r1 = g.iloc[0], g.iloc[1]
        rows.append({
            "match_id": match_id,
            "row_id_a": str(r0["Id"]),
            "row_id_b": str(r1["Id"]),
            "gender": str(r0.get("gender", "unknown")),
            "tournament": str(r0.get("tournament", "unknown")),
            "tournament_weight": float(get_tournament_weight(r0.get("tournament", "unknown"))),
        })
    return pd.DataFrame(rows)

match_meta = build_match_meta(test_raw)

def submission_to_match(sub_df: pd.DataFrame, label="submission") -> pd.DataFrame:
    sub = normalize_submission_df(sub_df, require_all=True)
    sub_idx = sub.set_index(sub["Id"].astype(str), drop=False)
    rows = []
    for _, m in match_meta.iterrows():
        id_a, id_b = str(m["row_id_a"]), str(m["row_id_b"])
        if id_a not in sub_idx.index or id_b not in sub_idx.index: continue
        ra, rb = sub_idx.loc[id_a], sub_idx.loc[id_b]
        if isinstance(ra, pd.DataFrame): ra = ra.iloc[0]
        if isinstance(rb, pd.DataFrame): rb = rb.iloc[0]
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
    return pd.DataFrame(rows)

def match_to_submission(match_pred: pd.DataFrame) -> pd.DataFrame:
    pred = match_pred[["match_id", "pred_a", "pred_b"]].copy()
    pred["pred_a"] = pd.to_numeric(pred["pred_a"], errors="raise").round().astype(int).clip(lower=0)
    pred["pred_b"] = pd.to_numeric(pred["pred_b"], errors="raise").round().astype(int).clip(lower=0)
    merged = match_meta[["match_id", "row_id_a", "row_id_b"]].merge(pred, on="match_id", how="left", validate="one_to_one")
    if merged[["pred_a", "pred_b"]].isna().any().any():
        bad = merged.loc[merged[["pred_a", "pred_b"]].isna().any(axis=1), "match_id"].head().tolist()
        raise RuntimeError(f"missing match predictions examples: {bad}")
    rows_a = pd.DataFrame({"Id": merged["row_id_a"].astype(str), "team_goals": merged["pred_a"].astype(int), "opp_goals": merged["pred_b"].astype(int)})
    rows_b = pd.DataFrame({"Id": merged["row_id_b"].astype(str), "team_goals": merged["pred_b"].astype(int), "opp_goals": merged["pred_a"].astype(int)})
    return normalize_submission_df(pd.concat([rows_a, rows_b], ignore_index=True), require_all=True)

def pair_consistency_report(sub_df: pd.DataFrame) -> dict:
    try:
        m = submission_to_match(sub_df)
    except Exception as e:
        return {"pair_consistency": False, "n_bad_pairs": -1, "error": repr(e)}
    if len(m) != len(match_meta):
        return {"pair_consistency": False, "n_bad_pairs": abs(len(match_meta) - len(m)), "error": "match_count_mismatch"}
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
    if "detail" not in out.columns: out["detail"] = ""
    return out

def basic_submission_metrics(sub_df: pd.DataFrame, label: str) -> dict:
    sub = normalize_submission_df(sub_df, require_all=True)
    m = submission_to_match(sub, label=label)
    scores = (m["pred_a"].astype(int).astype(str) + "-" + m["pred_b"].astype(int).astype(str)).value_counts(normalize=True)
    pc = pair_consistency_report(sub)
    return {
        "label": label,
        "n_rows": int(len(sub)),
        "n_matches": int(len(m)),
        "mean_pred_total": float((m["pred_a"] + m["pred_b"]).mean()),
        "max_pred_goal": int(m[["pred_a", "pred_b"]].max().max()),
        "top1_scoreline_share": float(scores.iloc[0]) if len(scores) else np.nan,
        "top3_scoreline_share": float(scores.head(3).sum()) if len(scores) else np.nan,
        "pair_consistency": bool(pc["pair_consistency"]),
        "n_bad_pairs": int(pc["n_bad_pairs"]),
    }

def transition_changes_df(base_match, candidate_match, label):
    base = base_match[["match_id", "pred_a", "pred_b", "gender", "tournament", "tournament_weight"]].rename(columns={"pred_a": "old_a", "pred_b": "old_b"})
    cand = candidate_match[["match_id", "pred_a", "pred_b"]].rename(columns={"pred_a": "new_a", "pred_b": "new_b"})
    ch = base.merge(cand, on="match_id", how="inner", validate="one_to_one")
    ch = ch[(ch["old_a"] != ch["new_a"]) | (ch["old_b"] != ch["new_b"])].copy()
    if len(ch):
        ch["old_score"] = ch["old_a"].astype(int).astype(str) + "-" + ch["old_b"].astype(int).astype(str)
        ch["new_score"] = ch["new_a"].astype(int).astype(str) + "-" + ch["new_b"].astype(int).astype(str)
        ch["transition"] = ch["old_score"] + " -> " + ch["new_score"]
        ch["old_total"] = ch["old_a"] + ch["old_b"]
        ch["new_total"] = ch["new_a"] + ch["new_b"]
        ch["total_delta"] = ch["new_total"] - ch["old_total"]
        ch["old_gd"] = ch["old_a"] - ch["old_b"]
        ch["new_gd"] = ch["new_a"] - ch["new_b"]
        ch["gd_delta"] = ch["new_gd"] - ch["old_gd"]
        ch["old_outcome"] = [outcome_scalar(a,b) for a,b in zip(ch["old_a"], ch["old_b"])]
        ch["new_outcome"] = [outcome_scalar(a,b) for a,b in zip(ch["new_a"], ch["new_b"])]
        ch["same_outcome"] = ch["old_outcome"] == ch["new_outcome"]
        ch["label"] = label
    return ch

log_section("Match Meta Audit")
safe_display_df(pd.DataFrame([{"key": "n_test_rows", "value": len(test_raw)}, {"key": "n_match_meta", "value": len(match_meta)}]))

# %% [markdown]# %% [markdown]
# # 04. Candidate Source Finder dan Valid Submission Loader
#
# Penjelasan bagian:
# Bagian ini mencari EXP39 final, EXP12M champion, L4/L6 reference, dan candidate submission lain. Finder membaca header CSV supaya tidak salah mengambil file summary/check/catalog.
#
# Output yang perlu dilihat:
# Cek `exp39_final_path`, `exp12m_champion_path`, dan candidate inventory. File source wajib punya kolom Id/id, team_goals, dan opp_goals.

# %%
SUMMARY_SKIP_NAMES = {
    "submission_catalog.csv", "submission_check.csv", "submission_validation.csv",
    "submission_metrics.csv", "submission_pair_consistency.csv", "submission_scoreline_distribution.csv",
    "submission_subgroup_metrics.csv", "submission_summary.csv",
}
SUMMARY_SKIP_FRAGMENTS = [
    "submission_catalog", "submission_check", "submission_validation", "submission_metrics",
    "submission_pair_consistency", "submission_scoreline_distribution", "submission_subgroup_metrics",
    "submission_summary", "local_gt_audit", "variant_metrics", "changed_prediction_analysis",
]

def file_has_submission_columns(p: Path) -> bool:
    try:
        cols = list(pd.read_csv(p, nrows=0).columns)
        low = {str(c).lower() for c in cols}
        return (("id" in low) or ("id" in {c.lower() for c in cols}) or ("Id" in cols)) and {"team_goals", "opp_goals"}.issubset(low)
    except Exception:
        return False

def is_submission_candidate_path(p: Path) -> bool:
    name = p.name.lower()
    parts = {x.lower() for x in p.parts}
    if not name.endswith(".csv") or not name.startswith("submission_"):
        return False
    if "sample" in name:
        return False
    if "summaries" in parts:
        return False
    if name in SUMMARY_SKIP_NAMES:
        return False
    if any(x in name for x in SUMMARY_SKIP_FRAGMENTS):
        return False
    return file_has_submission_columns(p)

def normalize_file_key(x) -> str:
    s = str(x).lower().replace("\\", "/").replace("-", "_").replace(" ", "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s

def all_submission_paths():
    roots = [OUTPUT_ROOT, PROJECT_ROOT, Path.cwd(), Path("/mnt/data")]
    seen, paths = set(), []
    for root in roots:
        if not root.exists():
            continue
        try:
            for p in root.rglob("submission_*.csv"):
                p = p.resolve()
                if p in seen:
                    continue
                seen.add(p)
                key = normalize_file_key(str(p))
                # Hindari partial output EXP12N sendiri sebagai source.
                if "/exp12n_total_plus_ranked_micro_patch_router/" in key:
                    continue
                if is_submission_candidate_path(p):
                    paths.append(p)
        except Exception:
            pass
    return sorted(paths, key=lambda p: str(p))

def priority_score(path: Path, patterns: list) -> tuple:
    key = normalize_file_key(path.name)
    full = normalize_file_key(str(path))
    for i, pat in enumerate(patterns):
        if isinstance(pat, str):
            pat = [pat]
        ok = all(normalize_file_key(tok) in key or normalize_file_key(tok) in full for tok in pat)
        if ok:
            return (i, len(str(path)), str(path))
    return (9999, len(str(path)), str(path))

def select_priority_path(paths: list, patterns: list):
    cand = sorted(paths, key=lambda p: priority_score(p, patterns))
    if not cand or priority_score(cand[0], patterns)[0] >= 9999:
        return None
    return cand[0]

EXP39_PATTERNS = [
    ["exp39", "safe", "final", "scoreline", "distribution", "validation", "only"],
    ["exp39", "safe", "final"],
    ["exp39", "scoreline", "distribution"],
    ["exp39", "router"],
]
EXP12M_PATTERNS = [
    ["exp12m", "m3", "total", "plus", "only", "cap0025"],
    ["exp12m", "m3", "total", "plus"],
    ["exp12m", "best", "safe"],
    ["exp12m", "m1", "l4", "cap0025"],
    ["exp12m", "m0", "l4", "cap0025"],
]
L4_PATTERNS = [
    ["exp12l", "l4", "exp39", "plus", "gd", "patch", "cap0025"],
    ["exp12l", "l4", "exp39", "plus", "same", "outcome", "exact", "cap0025"],
]
L6_PATTERNS = [
    ["exp12l", "l6", "exp39", "export", "exact", "majority", "cap0025"],
    ["exp12l", "l6", "exp39", "export", "exact", "majority", "cap0050"],
    ["exp12l", "l6", "exp39", "export", "same", "outcome"],
    ["exp12l", "l6"],
]

ALL_SUBMISSION_PATHS = all_submission_paths()
exp39_final_path = select_priority_path(ALL_SUBMISSION_PATHS, EXP39_PATTERNS)
exp12m_champion_path = select_priority_path(ALL_SUBMISSION_PATHS, EXP12M_PATTERNS)
l4_reference_path = select_priority_path(ALL_SUBMISSION_PATHS, L4_PATTERNS)
l6_best_path = select_priority_path(ALL_SUBMISSION_PATHS, L6_PATTERNS)

inventory = []
for p in ALL_SUBMISSION_PATHS:
    low = normalize_file_key(str(p))
    inventory.append({
        "file_name": p.name,
        "path": str(p),
        "is_exp39": "exp39" in low,
        "is_exp12m": "exp12m" in low,
        "is_exp12l": "exp12l" in low,
        "is_l4": "l4" in low,
        "is_l6": "l6" in low,
        "priority_exp39": priority_score(p, EXP39_PATTERNS)[0],
        "priority_exp12m": priority_score(p, EXP12M_PATTERNS)[0],
        "priority_l4": priority_score(p, L4_PATTERNS)[0],
        "priority_l6": priority_score(p, L6_PATTERNS)[0],
    })
source_inventory_df = pd.DataFrame(inventory)
source_inventory_df.to_csv(SUM_DIR / "source_candidate_inventory.csv", index=False)
log_saved(SUM_DIR / "source_candidate_inventory.csv")

log_section("Source Finder")
log_info(f"n_submission_paths_found = {len(ALL_SUBMISSION_PATHS)}")
log_info(f"exp39_final_path = {exp39_final_path}")
log_info(f"exp12m_champion_path = {exp12m_champion_path}")
log_info(f"l4_reference_path = {l4_reference_path}")
log_info(f"l6_best_path = {l6_best_path}")
safe_display_df(source_inventory_df.sort_values(["priority_exp12m", "priority_exp39", "priority_l4"]).head(40) if len(source_inventory_df) else source_inventory_df)

if exp39_final_path is None:
    raise FileNotFoundError("Tidak menemukan EXP39 final CSV. Pastikan submission_exp39_SAFE_FINAL_scoreline_distribution_validation_only.csv tersedia.")
if exp12m_champion_path is None:
    raise FileNotFoundError("Tidak menemukan EXP12M champion CSV. Pastikan submission_exp12m_best_safe.csv atau candidate M3 total_plus tersedia.")

def read_submission_path(path: Path, label: str) -> pd.DataFrame:
    out = normalize_submission_df(pd.read_csv(path), require_all=True)
    out.attrs["label"] = label
    out.attrs["path"] = str(path)
    return out

def save_candidate(sub_df: pd.DataFrame, label: str, strict=False):
    safe_label = slugify_label(label)
    out = normalize_submission_df(sub_df, require_all=True)
    path = SUB_DIR / f"submission_exp12n_{safe_label}.csv"
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
# # 05. N0 — Reproduce EXP12M Champion
#
# Penjelasan bagian:
# Bagian ini memuat EXP39 final sebagai base dan EXP12M champion sebagai reference. N0 disimpan ulang ke folder EXP12N agar bisa diaudit bersama candidate baru.
#
# Output yang perlu dilihat:
# `n0_exp12m_reproduction_summary.csv` harus menunjukkan source benar, pair consistency True, dan mean_pred_total sekitar 2.526-an.

# %%
exp39_final_sub = read_submission_path(exp39_final_path, "exp39_final_source")
exp39_final_match = submission_to_match(exp39_final_sub, label="exp39_final")

exp12m_champion_sub = read_submission_path(exp12m_champion_path, "exp12m_champion_source")
exp12m_champion_match = submission_to_match(exp12m_champion_sub, label="exp12m_champion")

l4_reference_match = None
if l4_reference_path is not None:
    l4_reference_match = submission_to_match(read_submission_path(l4_reference_path, "l4_reference"), label="l4_reference")

l6_best_sub = None
l6_best_match = None
if l6_best_path is not None:
    l6_best_sub = read_submission_path(l6_best_path, "l6_source")
    l6_best_match = submission_to_match(l6_best_sub, label="l6_source")

n0_sub, n0_path, n0_check = save_candidate(exp12m_champion_sub, "n0_exp12m_champion_reproduction", strict=False)
n0_summary = pd.DataFrame([{
    "exp12m_source_path": str(exp12m_champion_path),
    "exp39_source_path": str(exp39_final_path),
    **basic_submission_metrics(n0_sub, "n0_exp12m_champion_reproduction"),
}])
n0_summary.to_csv(SUM_DIR / "n0_exp12m_reproduction_summary.csv", index=False)
log_saved(SUM_DIR / "n0_exp12m_reproduction_summary.csv")
safe_display_df(n0_summary)
log_check("N0 pair consistency", bool(n0_summary["pair_consistency"].iloc[0]))

# %% [markdown]
# # 06. Candidate Pool dan Cache
#
# Penjelasan bagian:
# Bagian ini memuat candidate submission lama sebagai comparison/post-processing artifact, bukan fitur training. Candidate pool dipakai untuk menghitung support, ranking, transition confidence, dan L6 combo.
#
# Output yang perlu dilihat:
# `candidate_pool_loaded.csv` menampilkan label, path, pair consistency, dan mean_pred_total. Candidate table nanti harus exclude EXP12L/EXP12M self-reference.

# %%
def candidate_reliability_weight(label: str) -> float:
    s = str(label).lower()
    if "exp39" in s and ("safe_final" in s or "final" in s): return 3.0
    if "exp12m" in s and "best" in s: return 2.5
    if "exp12m" in s and "m3" in s: return 2.2
    if "exp12l" in s and "l6" in s: return 2.0
    if "exp39" in s and ("scoreline" in s or "router" in s): return 2.0
    if "exp12l" in s and "l4" in s: return 0.0  # self-reference excluded anyway
    if any(x in s for x in ["exp12k", "exp12j", "exp12i", "exp12h"]): return 1.2
    if "exp12" in s: return 1.0
    return 0.8

def label_from_path(p: Path) -> str:
    return slugify_label(p.stem.replace("submission_", ""), max_len=120)

pool_paths = []
for p in ALL_SUBMISSION_PATHS:
    low = normalize_file_key(str(p))
    include = False
    if "exp39" in low or "scoreline_distribution" in low or "router" in low:
        include = True
    if "exp12l" in low and ("l4" in low or "l6" in low or "best_safe" in low):
        include = True
    if "exp12m" in low:
        include = True
    if any(x in low for x in ["exp12k", "exp12j", "exp12i", "exp12h"]):
        include = True
    if include:
        pool_paths.append(p)
pool_paths = sorted(set(pool_paths), key=lambda p: str(p))

candidate_match_pool = {}
pool_rows = []
for p in pool_paths:
    label = label_from_path(p)
    try:
        sub = read_submission_path(p, label)
        m = submission_to_match(sub, label=label)
        candidate_match_pool[label] = m
        met = basic_submission_metrics(sub, label)
        met.update({"path": str(p), "source_weight": candidate_reliability_weight(str(p) + " " + label)})
        pool_rows.append(met)
    except Exception as e:
        pool_rows.append({"label": label, "path": str(p), "error": repr(e)})

candidate_match_pool["exp39_final_base"] = exp39_final_match.copy()
candidate_match_pool["exp12m_champion_reference"] = exp12m_champion_match.copy()
if l4_reference_match is not None:
    candidate_match_pool["l4_reference_cap0025"] = l4_reference_match.copy()
if l6_best_match is not None:
    candidate_match_pool["l6_best_candidate"] = l6_best_match.copy()

CANDIDATE_MATCH_POOL_INDEX = {lab: df.set_index("match_id", drop=False) for lab, df in candidate_match_pool.items() if "match_id" in df.columns}
candidate_pool_loaded_df = pd.DataFrame(pool_rows)
candidate_pool_loaded_df.to_csv(SUM_DIR / "candidate_pool_loaded.csv", index=False)
log_saved(SUM_DIR / "candidate_pool_loaded.csv")
safe_display_df(candidate_pool_loaded_df.sort_values("label").head(50) if len(candidate_pool_loaded_df) else candidate_pool_loaded_df)
log_info(f"n_candidate_match_pool = {len(candidate_match_pool)}")

# %% [markdown]
# # 07. Core Requirement — Total-Plus Candidate Table
#
# Penjelasan bagian:
# Bagian ini membangun `total_plus_candidate_table.csv`, yaitu subset same-outcome patch di atas EXP39 dengan `total_delta = +1`. Self-reference EXP12L/EXP12M patch diblok dan satu match hanya boleh punya satu kandidat.
#
# Output yang perlu dilihat:
# Cek jumlah kandidat, support distribution, transition summary, default/non-high-weight share, dan unique match count.

# %%
def should_exclude_total_plus_self_label(label: str) -> bool:
    s = str(label).lower()
    return (
        s in {"exp12m_champion_reference", "l4_reference_cap0025"}
        or ("exp12l" in s and "l4" in s)
        or "l4_cap0025" in s
        or "exp12m_m0" in s
        or "exp12m_m1" in s
        or "exp12m_m2" in s
        or "exp12m_m3" in s
        or "exp12m_m4" in s
        or "exp12m_m5" in s
        or "exp12m_m6" in s
        or "exp12m_m7" in s
        or "exp12m_best" in s
        or "exp12n" in s
    )

def build_score_support_map(match_id, exclude_labels=None):
    exclude_labels = set(exclude_labels or [])
    score_to_labels = defaultdict(list)
    score_to_weight = defaultdict(float)
    for lab, idx in CANDIDATE_MATCH_POOL_INDEX.items():
        if lab in exclude_labels:
            continue
        if should_exclude_total_plus_self_label(lab):
            continue
        if match_id not in idx.index:
            continue
        r = idx.loc[match_id]
        if isinstance(r, pd.DataFrame):
            r = r.iloc[0]
        score = (int(r["pred_a"]), int(r["pred_b"]))
        w = candidate_reliability_weight(lab)
        if w <= 0:
            continue
        score_to_labels[score].append(lab)
        score_to_weight[score] += w
    return score_to_labels, score_to_weight

def build_total_plus_candidate_table(min_support=2, max_abs_gd_delta=1, exclude_labels=None):
    rows = []
    exclude = set(exclude_labels or []) | {"exp39_final_base", "exp12m_champion_reference", "l4_reference_cap0025"}
    for _, base in exp39_final_match.iterrows():
        match_id = base["match_id"]
        old_a, old_b = int(base["pred_a"]), int(base["pred_b"])
        old_out = outcome_scalar(old_a, old_b)
        score_to_labels, score_to_weight = build_score_support_map(match_id, exclude_labels=exclude)
        for (new_a, new_b), labels in score_to_labels.items():
            new_a, new_b = int(new_a), int(new_b)
            if new_a == old_a and new_b == old_b:
                continue
            if outcome_scalar(new_a, new_b) != old_out:
                continue
            old_total, new_total = old_a + old_b, new_a + new_b
            total_delta = int(new_total - old_total)
            if total_delta != 1:
                continue
            old_gd, new_gd = old_a - old_b, new_a - new_b
            gd_delta = int(new_gd - old_gd)
            if abs(gd_delta) > max_abs_gd_delta:
                continue
            support = len(labels)
            weighted_support = float(score_to_weight[(new_a, new_b)])
            if support < min_support:
                continue
            labels_sorted = sorted(labels, key=lambda x: (-candidate_reliability_weight(x), x))
            top = labels_sorted[0] if labels_sorted else ""
            tournament = str(base.get("tournament", "unknown"))
            tw = float(base.get("tournament_weight", 1.20))
            rows.append({
                "match_id": match_id,
                "old_a": old_a, "old_b": old_b,
                "new_a": new_a, "new_b": new_b,
                "old_score": f"{old_a}-{old_b}", "new_score": f"{new_a}-{new_b}",
                "transition": f"{old_a}-{old_b} -> {new_a}-{new_b}",
                "support": int(support), "weighted_support": float(weighted_support),
                "source_labels": "|".join(labels_sorted),
                "source_label_top": top,
                "source_weight_top": candidate_reliability_weight(top) if top else 0.0,
                "gender": base.get("gender", "unknown"),
                "tournament": tournament,
                "tournament_weight": tw,
                "old_total": int(old_total), "new_total": int(new_total), "total_delta": int(total_delta),
                "old_gd": int(old_gd), "new_gd": int(new_gd), "gd_delta": int(gd_delta),
                "abs_gd_delta": int(abs(gd_delta)),
                "same_outcome": True,
                "is_high_weight": bool(tw >= 1.8),
                "is_low_weight": bool(tw <= 0.96),
                "is_default_weight": bool(abs(tw - 1.20) < 1e-9),
                "is_world_asian": bool(any(x in tournament.lower() for x in ["world", "asian", "afc"])),
                "is_friendly": bool("friendly" in tournament.lower()),
            })
    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out
    out["non_high_weight_rank"] = (~out["is_high_weight"].astype(bool)).astype(int)
    out["default_weight_rank"] = out["is_default_weight"].astype(int)
    out = out.sort_values(
        ["support", "weighted_support", "source_weight_top", "default_weight_rank", "non_high_weight_rank", "abs_gd_delta", "match_id"],
        ascending=[False, False, False, False, False, True, True],
    ).reset_index(drop=True)
    out = out.drop_duplicates("match_id", keep="first").reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out

total_plus_candidate_table = build_total_plus_candidate_table(min_support=2, max_abs_gd_delta=1)
total_plus_candidate_table.to_csv(SUM_DIR / "total_plus_candidate_table.csv", index=False)
log_saved(SUM_DIR / "total_plus_candidate_table.csv")

if len(total_plus_candidate_table):
    total_plus_candidate_summary = pd.DataFrame([
        {"metric": "n_candidates", "value": len(total_plus_candidate_table)},
        {"metric": "n_match_unique", "value": total_plus_candidate_table["match_id"].nunique()},
        {"metric": "support_mean", "value": float(total_plus_candidate_table["support"].mean())},
        {"metric": "weighted_support_mean", "value": float(total_plus_candidate_table["weighted_support"].mean())},
        {"metric": "default_weight_share", "value": float(total_plus_candidate_table["is_default_weight"].mean())},
        {"metric": "non_high_weight_share", "value": float((~total_plus_candidate_table["is_high_weight"].astype(bool)).mean())},
        {"metric": "high_weight_share", "value": float(total_plus_candidate_table["is_high_weight"].mean())},
    ])
else:
    total_plus_candidate_summary = pd.DataFrame([{"metric": "n_candidates", "value": 0}])
total_plus_candidate_summary.to_csv(SUM_DIR / "total_plus_candidate_summary.csv", index=False)

if len(total_plus_candidate_table):
    total_plus_transition_summary = total_plus_candidate_table.groupby("transition").agg(
        n_candidates=("match_id", "count"),
        mean_support=("support", "mean"),
        mean_weighted_support=("weighted_support", "mean"),
        share_default_weight=("is_default_weight", "mean"),
        share_non_high_weight=("is_high_weight", lambda s: float((~s.astype(bool)).mean())),
    ).reset_index().sort_values(["n_candidates", "mean_weighted_support"], ascending=[False, False])
    total_plus_segment_summary = total_plus_candidate_table.groupby(["gender", "is_default_weight", "is_high_weight", "is_low_weight"]).agg(
        n_candidates=("match_id", "count"),
        mean_support=("support", "mean"),
        mean_weighted_support=("weighted_support", "mean"),
    ).reset_index().sort_values("n_candidates", ascending=False)
else:
    total_plus_transition_summary = pd.DataFrame()
    total_plus_segment_summary = pd.DataFrame()
total_plus_transition_summary.to_csv(SUM_DIR / "total_plus_transition_summary.csv", index=False)
total_plus_segment_summary.to_csv(SUM_DIR / "total_plus_segment_summary.csv", index=False)
log_saved(SUM_DIR / "total_plus_candidate_summary.csv")
log_saved(SUM_DIR / "total_plus_transition_summary.csv")
log_saved(SUM_DIR / "total_plus_segment_summary.csv")
safe_display_df(total_plus_candidate_summary)
safe_display_df(total_plus_transition_summary.head(20) if len(total_plus_transition_summary) else total_plus_transition_summary)

# %% [markdown]
# # 08. Candidate Application Helper dan Ranking Router
#
# Penjelasan bagian:
# Bagian ini menyiapkan helper untuk memilih top-N berdasarkan cap, menerapkan perubahan ke EXP39, menyimpan submission, dan mengurutkan total-plus candidate table dengan beberapa ranking rule.
#
# Output yang perlu dilihat:
# Helper ini tidak menghasilkan output besar, tetapi dipakai semua variant N1-N8.

# %%
def select_top_by_cap(table: pd.DataFrame, cap_rate: float, base_n_matches=None):
    if table is None or len(table) == 0:
        return pd.DataFrame(columns=[])
    if base_n_matches is None:
        base_n_matches = len(match_meta)
    n = max(int(math.floor(float(cap_rate) * base_n_matches)), 0)
    return table.head(min(n, len(table))).copy()

def select_top_by_count(table: pd.DataFrame, n: int):
    if table is None or len(table) == 0:
        return pd.DataFrame(columns=[])
    return table.head(min(max(int(n), 0), len(table))).copy()

def apply_total_plus_changes(changes: pd.DataFrame, label: str, base_match=None):
    if base_match is None:
        base_match = exp39_final_match
    out = base_match[["match_id", "pred_a", "pred_b", "gender", "tournament", "tournament_weight"]].copy()
    if changes is not None and len(changes) > 0:
        ch = changes.drop_duplicates("match_id", keep="first").set_index("match_id")
        out_idx = out.set_index("match_id", drop=False)
        for mid in ch.index:
            if mid in out_idx.index:
                out_idx.loc[mid, "pred_a"] = int(ch.loc[mid, "new_a"])
                out_idx.loc[mid, "pred_b"] = int(ch.loc[mid, "new_b"])
        out = out_idx.reset_index(drop=True)
    out["label"] = label
    return out

def save_match_candidate(match_df: pd.DataFrame, label: str, strict=False, changes=None):
    sub = match_to_submission(match_df)
    sub, path, check = save_candidate(sub, label, strict=strict)
    metrics = basic_submission_metrics(sub, label)
    metrics["path"] = str(path)
    if changes is None:
        changes = transition_changes_df(exp39_final_match, submission_to_match(sub, label), label)
    metrics["n_changed_vs_exp39"] = int(len(changes)) if changes is not None else 0
    metrics["changed_rate_vs_exp39"] = float(len(changes) / max(len(match_meta), 1)) if changes is not None else 0.0
    if changes is not None and len(changes):
        metrics["mean_total_delta_vs_exp39"] = float(changes["total_delta"].mean()) if "total_delta" in changes.columns else np.nan
        metrics["changed_high_weight"] = int((changes["tournament_weight"] >= 1.8).sum()) if "tournament_weight" in changes.columns else 0
        metrics["changed_low_weight"] = int((changes["tournament_weight"] <= 0.96).sum()) if "tournament_weight" in changes.columns else 0
        metrics["changed_M"] = int(changes["gender"].astype(str).str.upper().eq("M").sum()) if "gender" in changes.columns else 0
        metrics["changed_W"] = int(changes["gender"].astype(str).str.upper().eq("W").sum()) if "gender" in changes.columns else 0
    else:
        metrics["mean_total_delta_vs_exp39"] = 0.0
        metrics["changed_high_weight"] = metrics["changed_low_weight"] = metrics["changed_M"] = metrics["changed_W"] = 0
    return sub, path, check, metrics, changes

def sort_total_plus_candidates(table: pd.DataFrame, rule: str) -> pd.DataFrame:
    if table is None or len(table) == 0:
        return table.copy()
    df = table.copy()
    df["non_high_weight_rank"] = (~df["is_high_weight"].astype(bool)).astype(int)
    df["default_weight_rank"] = df["is_default_weight"].astype(int)
    df["low_gd_rank"] = (df["abs_gd_delta"] == 0).astype(int)
    if len(total_plus_transition_summary):
        conf_map = total_plus_transition_summary.set_index("transition")["mean_weighted_support"].to_dict()
        n_map = total_plus_transition_summary.set_index("transition")["n_candidates"].to_dict()
        df["transition_confidence"] = df["transition"].map(conf_map).fillna(0.0)
        df["transition_count"] = df["transition"].map(n_map).fillna(0).astype(int)
    else:
        df["transition_confidence"] = 0.0
        df["transition_count"] = 0
    if rule == "support_then_weighted":
        cols, asc = ["support", "weighted_support", "source_weight_top", "default_weight_rank", "abs_gd_delta", "match_id"], [False, False, False, False, True, True]
    elif rule == "weighted_then_support":
        cols, asc = ["weighted_support", "support", "source_weight_top", "default_weight_rank", "abs_gd_delta", "match_id"], [False, False, False, False, True, True]
    elif rule == "default_first_then_support":
        cols, asc = ["default_weight_rank", "support", "weighted_support", "source_weight_top", "abs_gd_delta", "match_id"], [False, False, False, False, True, True]
    elif rule == "non_high_first_then_weighted":
        cols, asc = ["non_high_weight_rank", "weighted_support", "support", "source_weight_top", "abs_gd_delta", "match_id"], [False, False, False, False, True, True]
    elif rule == "transition_confidence_first":
        cols, asc = ["transition_confidence", "support", "weighted_support", "default_weight_rank", "match_id"], [False, False, False, False, True]
    elif rule == "source_reliability_first":
        cols, asc = ["source_weight_top", "weighted_support", "support", "default_weight_rank", "match_id"], [False, False, False, False, True]
    elif rule == "low_abs_gd_first":
        cols, asc = ["abs_gd_delta", "support", "weighted_support", "default_weight_rank", "match_id"], [True, False, False, False, True]
    elif rule == "high_support_default_first":
        df["high_support_default_rank"] = ((df["support"] >= 3) & df["is_default_weight"].astype(bool)).astype(int)
        cols, asc = ["high_support_default_rank", "support", "weighted_support", "match_id"], [False, False, False, True]
    elif rule == "high_support_non_high_first":
        df["high_support_non_high_rank"] = ((df["support"] >= 3) & (~df["is_high_weight"].astype(bool))).astype(int)
        cols, asc = ["high_support_non_high_rank", "support", "weighted_support", "match_id"], [False, False, False, True]
    else:
        cols, asc = (["rank"], [True]) if "rank" in df.columns else (["support", "weighted_support", "match_id"], [False, False, True])
    out = df.sort_values(cols, ascending=asc).reset_index(drop=True)
    if "rank" in out.columns:
        out = out.drop(columns=["rank"])
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out

def register_grid_record(records, label, changes, metrics, extra=None):
    rec = {"label": label}
    if extra:
        rec.update(extra)
    rec.update({
        "n_selected": int(len(changes)) if changes is not None else 0,
        "changed_rate": float(len(changes) / max(len(match_meta), 1)) if changes is not None else 0.0,
        "mean_pred_total": metrics.get("mean_pred_total", np.nan),
        "mean_total_delta_vs_exp39": metrics.get("mean_total_delta_vs_exp39", 0.0),
        "changed_high_weight": metrics.get("changed_high_weight", 0),
        "changed_low_weight": metrics.get("changed_low_weight", 0),
        "changed_M": metrics.get("changed_M", 0),
        "changed_W": metrics.get("changed_W", 0),
        "pair_consistency": metrics.get("pair_consistency", False),
        "n_bad_pairs": metrics.get("n_bad_pairs", -1),
        "path": metrics.get("path", ""),
    })
    records.append(rec)

all_variant_metrics, all_submission_checks, submission_catalog, changed_analysis_frames = [], [], [], []

# Register N0 in catalog/metrics.
n0_metrics = basic_submission_metrics(n0_sub, "n0_exp12m_champion_reproduction")
n0_metrics["path"] = str(n0_path)
n0_metrics["n_changed_vs_exp39"] = len(transition_changes_df(exp39_final_match, exp12m_champion_match, "n0"))
n0_metrics["changed_rate_vs_exp39"] = n0_metrics["n_changed_vs_exp39"] / max(len(match_meta), 1)
all_variant_metrics.append(n0_metrics)
all_submission_checks.append(n0_check)
submission_catalog.append({"label": "n0_exp12m_champion_reproduction", "path": str(n0_path), "source": "loaded_exp12m"})

# %% [markdown]
# # 09. N1 — Total Plus + Default Weight Gate
#
# Penjelasan bagian:
# N1 memilih kandidat total-plus hanya pada tournament default-weight 1.20. Ini menggabungkan sinyal total_plus dan segment yang lebih aman.
#
# Output yang perlu dilihat:
# `n1_total_plus_default_grid.csv` menunjukkan cap, jumlah selected, mean total, dan pair consistency.

# %%
n1_table = sort_total_plus_candidates(total_plus_candidate_table[total_plus_candidate_table["is_default_weight"].astype(bool)].copy() if len(total_plus_candidate_table) else total_plus_candidate_table, "default_first_then_support")
n1_caps = [0.0010, 0.0015, 0.0020, 0.0025, 0.0030, 0.0035, 0.0040]
n1_records = []
for cap in n1_caps:
    changes = select_top_by_cap(n1_table, cap)
    label = f"n1_total_plus_default_cap{int(round(cap * 10000)):04d}"
    match_df = apply_total_plus_changes(changes, label)
    sub, path, check, metrics, ch = save_match_candidate(match_df, label, strict=False, changes=changes)
    register_grid_record(n1_records, label, changes, metrics, {"cap_rate": cap, "filter": "default_weight"})
    all_variant_metrics.append(metrics); all_submission_checks.append(check)
    submission_catalog.append({"label": label, "path": str(path), "source": "n1_default"})
    if len(ch): changed_analysis_frames.append(ch.assign(variant_label=label))
n1_grid = pd.DataFrame(n1_records)
n1_grid.to_csv(SUM_DIR / "n1_total_plus_default_grid.csv", index=False)
log_saved(SUM_DIR / "n1_total_plus_default_grid.csv")
safe_display_df(n1_grid)

# %% [markdown]
# # 10. N2 — Total Plus + Non-High-Weight Gate
#
# Penjelasan bagian:
# N2 memilih kandidat total-plus pada tournament non-high-weight, yaitu weight di bawah 1.80. Variant support3 dan weighted support juga disediakan.
#
# Output yang perlu dilihat:
# `n2_total_plus_non_high_grid.csv` membantu membaca apakah high-weight sebaiknya dihindari.

# %%
n2_base = total_plus_candidate_table[~total_plus_candidate_table["is_high_weight"].astype(bool)].copy() if len(total_plus_candidate_table) else total_plus_candidate_table
n2_table = sort_total_plus_candidates(n2_base, "non_high_first_then_weighted")
n2_caps = [0.0010, 0.0015, 0.0020, 0.0025, 0.0030, 0.0035, 0.0040]
n2_records = []
for cap in n2_caps:
    changes = select_top_by_cap(n2_table, cap)
    label = f"n2_total_plus_non_high_cap{int(round(cap * 10000)):04d}"
    match_df = apply_total_plus_changes(changes, label)
    sub, path, check, metrics, ch = save_match_candidate(match_df, label, strict=False, changes=changes)
    register_grid_record(n2_records, label, changes, metrics, {"cap_rate": cap, "filter": "non_high_weight"})
    all_variant_metrics.append(metrics); all_submission_checks.append(check)
    submission_catalog.append({"label": label, "path": str(path), "source": "n2_non_high"})
    if len(ch): changed_analysis_frames.append(ch.assign(variant_label=label))

for name, table in [
    ("support3_cap0030", n2_table[n2_table["support"] >= 3].copy() if len(n2_table) else n2_table),
    ("weighted_support_cap0030", n2_table[n2_table["weighted_support"] >= n2_table["weighted_support"].quantile(0.60)].copy() if len(n2_table) else n2_table),
]:
    changes = select_top_by_cap(table, 0.0030)
    label = f"n2_total_plus_non_high_{name}"
    match_df = apply_total_plus_changes(changes, label)
    sub, path, check, metrics, ch = save_match_candidate(match_df, label, strict=False, changes=changes)
    register_grid_record(n2_records, label, changes, metrics, {"cap_rate": 0.0030, "filter": name})
    all_variant_metrics.append(metrics); all_submission_checks.append(check)
    submission_catalog.append({"label": label, "path": str(path), "source": "n2_non_high_extra"})
    if len(ch): changed_analysis_frames.append(ch.assign(variant_label=label))

n2_grid = pd.DataFrame(n2_records)
n2_grid.to_csv(SUM_DIR / "n2_total_plus_non_high_grid.csv", index=False)
log_saved(SUM_DIR / "n2_total_plus_non_high_grid.csv")
safe_display_df(n2_grid)

# %% [markdown]
# # 11. N3 — Total Plus + Transition Confidence
#
# Penjelasan bagian:
# N3 membuat transition confidence table dan mencoba hanya top transition, top2, top3, exclude top1, serta kombinasi default/non-high top2.
#
# Output yang perlu dilihat:
# `transition_confidence_table.csv` dan `n3_transition_confidence_grid.csv` menunjukkan transition mana yang dominan dan aman.

# %%
transition_confidence_table = total_plus_transition_summary.copy()
transition_confidence_table.to_csv(SUM_DIR / "transition_confidence_table.csv", index=False)
log_saved(SUM_DIR / "transition_confidence_table.csv")
n3_records = []

def n3_make(table, label_suffix, cap=0.0025):
    table = sort_total_plus_candidates(table.copy(), "transition_confidence_first") if len(table) else table
    changes = select_top_by_cap(table, cap)
    label = f"n3_{label_suffix}_cap{int(round(cap * 10000)):04d}"
    match_df = apply_total_plus_changes(changes, label)
    sub, path, check, metrics, ch = save_match_candidate(match_df, label, strict=False, changes=changes)
    register_grid_record(n3_records, label, changes, metrics, {"cap_rate": cap, "filter": label_suffix})
    all_variant_metrics.append(metrics); all_submission_checks.append(check)
    submission_catalog.append({"label": label, "path": str(path), "source": "n3_transition"})
    if len(ch): changed_analysis_frames.append(ch.assign(variant_label=label))

if len(total_plus_candidate_table) and len(transition_confidence_table):
    top_transitions = transition_confidence_table.sort_values(["n_candidates", "mean_weighted_support"], ascending=[False, False])["transition"].tolist()
    for k in [1, 2, 3]:
        trs = set(top_transitions[:k])
        n3_make(total_plus_candidate_table[total_plus_candidate_table["transition"].isin(trs)].copy(), f"total_plus_only_top{k}_transition")
    if top_transitions:
        n3_make(total_plus_candidate_table[~total_plus_candidate_table["transition"].eq(top_transitions[0])].copy(), "total_plus_exclude_top1_transition")
    trs2 = set(top_transitions[:2])
    n3_make(total_plus_candidate_table[total_plus_candidate_table["is_default_weight"].astype(bool) & total_plus_candidate_table["transition"].isin(trs2)].copy(), "total_plus_default_top2_transition")
    n3_make(total_plus_candidate_table[(~total_plus_candidate_table["is_high_weight"].astype(bool)) & total_plus_candidate_table["transition"].isin(trs2)].copy(), "total_plus_non_high_top2_transition")
    explicit_map = {
        "0_0_to_1_1": "0-0 -> 1-1",
        "1_0_to_2_0": "1-0 -> 2-0",
        "0_1_to_0_2": "0-1 -> 0-2",
    }
    for key, tr in explicit_map.items():
        if tr in set(total_plus_candidate_table["transition"]):
            n3_make(total_plus_candidate_table[total_plus_candidate_table["transition"].eq(tr)].copy(), f"only_{key}")
    # Handle either 1-1 -> 2-1 or 1-1 -> 1-2.
    one_one_trs = [tr for tr in ["1-1 -> 2-1", "1-1 -> 1-2"] if tr in set(total_plus_candidate_table["transition"])]
    if one_one_trs:
        n3_make(total_plus_candidate_table[total_plus_candidate_table["transition"].isin(one_one_trs)].copy(), "only_1_1_to_2_1_or_1_2")

n3_grid = pd.DataFrame(n3_records)
n3_grid.to_csv(SUM_DIR / "n3_transition_confidence_grid.csv", index=False)
log_saved(SUM_DIR / "n3_transition_confidence_grid.csv")
safe_display_df(n3_grid)

# %% [markdown]
# # 12. N4 — Ranking Refinement Khusus Total Plus
#
# Penjelasan bagian:
# N4 mencoba ranking khusus total-plus. Karena cap kecil, urutan kandidat lebih penting daripada jumlah kandidat mentah.
#
# Output yang perlu dilihat:
# `n4_total_plus_ranking_grid.csv` menunjukkan ranking rule dan cap 0.20/0.25/0.30%.

# %%
n4_rules = ["support_then_weighted", "weighted_then_support", "default_first_then_support", "non_high_first_then_weighted", "transition_confidence_first", "source_reliability_first", "low_abs_gd_first", "high_support_default_first", "high_support_non_high_first"]
n4_caps = [0.0020, 0.0025, 0.0030]
n4_records, n4_change_frames = [], []
for rule in n4_rules:
    sorted_table = sort_total_plus_candidates(total_plus_candidate_table, rule)
    for cap in n4_caps:
        changes = select_top_by_cap(sorted_table, cap)
        label = f"n4_{rule}_cap{int(round(cap * 10000)):04d}"
        match_df = apply_total_plus_changes(changes, label)
        sub, path, check, metrics, ch = save_match_candidate(match_df, label, strict=False, changes=changes)
        register_grid_record(n4_records, label, changes, metrics, {"ranking_rule": rule, "cap_rate": cap})
        all_variant_metrics.append(metrics); all_submission_checks.append(check)
        submission_catalog.append({"label": label, "path": str(path), "source": "n4_ranking"})
        if len(ch):
            n4_change_frames.append(ch.assign(variant_label=label, ranking_rule=rule, cap_rate=cap))
            changed_analysis_frames.append(ch.assign(variant_label=label))
n4_grid = pd.DataFrame(n4_records)
n4_grid.to_csv(SUM_DIR / "n4_total_plus_ranking_grid.csv", index=False)
log_saved(SUM_DIR / "n4_total_plus_ranking_grid.csv")
safe_display_df(n4_grid.head(60))
n4_change_analysis = pd.concat(n4_change_frames, ignore_index=True) if n4_change_frames else pd.DataFrame()
n4_change_analysis.to_csv(SUM_DIR / "n4_total_plus_ranking_change_analysis.csv", index=False)
log_saved(SUM_DIR / "n4_total_plus_ranking_change_analysis.csv")

# %% [markdown]
# # 13. N5 — Ultra-Fine Cap Sweep on Best Filtered Pool
#
# Penjelasan bagian:
# N5 melakukan sweep cap lebih rapat pada filtered pool GT-free default: total_plus + non-high-weight + support >= 2.
#
# Output yang perlu dilihat:
# `n5_ultrafine_filtered_cap_grid.csv` menampilkan apakah cap lebih kecil/besar dari 0.25% lebih aman.

# %%
n5_pool = sort_total_plus_candidates(n2_base.copy(), "non_high_first_then_weighted") if len(n2_base) else n2_base
n5_caps = [0.0005, 0.0010, 0.0015, 0.0020, 0.0025, 0.0030, 0.0035, 0.0040, 0.0045, 0.0050]
n5_records = []
for cap in n5_caps:
    changes = select_top_by_cap(n5_pool, cap)
    label = f"n5_best_filtered_cap{int(round(cap * 10000)):04d}"
    match_df = apply_total_plus_changes(changes, label)
    sub, path, check, metrics, ch = save_match_candidate(match_df, label, strict=False, changes=changes)
    register_grid_record(n5_records, label, changes, metrics, {"cap_rate": cap, "filtered_pool": "total_plus_non_high_support2"})
    all_variant_metrics.append(metrics); all_submission_checks.append(check)
    submission_catalog.append({"label": label, "path": str(path), "source": "n5_filtered_cap"})
    if len(ch): changed_analysis_frames.append(ch.assign(variant_label=label))
n5_grid = pd.DataFrame(n5_records)
n5_grid.to_csv(SUM_DIR / "n5_ultrafine_filtered_cap_grid.csv", index=False)
log_saved(SUM_DIR / "n5_ultrafine_filtered_cap_grid.csv")
safe_display_df(n5_grid)

# %% [markdown]
# # 14. N6 — Prefix Count Sweep
#
# Penjelasan bagian:
# N6 memilih berdasarkan jumlah match unik langsung dari ranking GT-free total-plus. Ini membantu menemukan golden band tanpa bergantung pada cap rate.
#
# Output yang perlu dilihat:
# `n6_prefix_count_sweep_grid.csv` berisi prefix 20 sampai 100.

# %%
n6_counts = [20, 30, 40, 50, 60, 70, 80, 90, 100]
n6_table = sort_total_plus_candidates(total_plus_candidate_table, "default_first_then_support")
n6_records = []
for n in n6_counts:
    changes = select_top_by_count(n6_table, n)
    label = f"n6_prefix_{n:03d}"
    match_df = apply_total_plus_changes(changes, label)
    sub, path, check, metrics, ch = save_match_candidate(match_df, label, strict=False, changes=changes)
    register_grid_record(n6_records, label, changes, metrics, {"prefix_count": n})
    all_variant_metrics.append(metrics); all_submission_checks.append(check)
    submission_catalog.append({"label": label, "path": str(path), "source": "n6_prefix"})
    if len(ch): changed_analysis_frames.append(ch.assign(variant_label=label))
n6_grid = pd.DataFrame(n6_records)
n6_grid.to_csv(SUM_DIR / "n6_prefix_count_sweep_grid.csv", index=False)
log_saved(SUM_DIR / "n6_prefix_count_sweep_grid.csv")
safe_display_df(n6_grid)

# %% [markdown]
# # 15. N7 — Tiny L6 / EXP39 Export-Router Combo
#
# Penjelasan bagian:
# N7 menambahkan patch sangat kecil dari L6/export-router di atas candidate total-plus filtered. Match yang sudah diubah total-plus tidak boleh diubah lagi.
#
# Output yang perlu dilihat:
# `n7_l6_combo_grid.csv` dan `n7_l6_combo_changes.csv` menunjukkan tambahan kecil dari L6.

# %%
n7_records, n7_frames = [], []
# Base GT-free: N2 cap0025 if available, otherwise N0.
n7_base_changes = select_top_by_cap(n2_table, 0.0025)
n7_base_match = apply_total_plus_changes(n7_base_changes, "n7_base_total_plus")
changed_base_ids = set(n7_base_changes["match_id"].tolist()) if len(n7_base_changes) else set()

def build_l6_add_table():
    if l6_best_match is None:
        return pd.DataFrame()
    base = n7_base_match[["match_id", "pred_a", "pred_b", "gender", "tournament", "tournament_weight"]].rename(columns={"pred_a": "old_a", "pred_b": "old_b"})
    cand = l6_best_match[["match_id", "pred_a", "pred_b"]].rename(columns={"pred_a": "new_a", "pred_b": "new_b"})
    t = base.merge(cand, on="match_id", how="inner", validate="one_to_one")
    t = t[(t["old_a"] != t["new_a"]) | (t["old_b"] != t["new_b"])].copy()
    if len(t) == 0:
        return t
    t = t[~t["match_id"].isin(changed_base_ids)].copy()
    t["old_outcome"] = [outcome_scalar(a,b) for a,b in zip(t["old_a"], t["old_b"])]
    t["new_outcome"] = [outcome_scalar(a,b) for a,b in zip(t["new_a"], t["new_b"])]
    t["same_outcome"] = t["old_outcome"] == t["new_outcome"]
    t["old_total"] = t["old_a"] + t["old_b"]
    t["new_total"] = t["new_a"] + t["new_b"]
    t["total_delta"] = t["new_total"] - t["old_total"]
    t["old_gd"] = t["old_a"] - t["old_b"]
    t["new_gd"] = t["new_a"] - t["new_b"]
    t["gd_delta"] = t["new_gd"] - t["old_gd"]
    t = t[t["same_outcome"] & (t["total_delta"].abs() <= 1) & (t["gd_delta"].abs() <= 1)].copy()
    t["support"] = 2
    t["weighted_support"] = 2.0
    t = t.sort_values(["total_delta", "tournament_weight", "match_id"], ascending=[False, True, True]).reset_index(drop=True)
    t.insert(0, "rank", np.arange(1, len(t)+1))
    return t

l6_add_table = build_l6_add_table()
for name, cap, table_filter in [
    ("total_plus_plus_l6_exact_cap0005", 0.0005, lambda d: d),
    ("total_plus_plus_l6_exact_cap0010", 0.0010, lambda d: d),
    ("total_plus_plus_l6_same_outcome_cap0005", 0.0005, lambda d: d[d["same_outcome"]]),
    ("total_plus_plus_l6_low_total_router_cap0005", 0.0005, lambda d: d[d["total_delta"] <= 0]),
]:
    extra = table_filter(l6_add_table.copy()) if len(l6_add_table) else l6_add_table
    extra = select_top_by_cap(extra, cap)
    merged_changes = pd.concat([n7_base_changes, extra], ignore_index=True) if len(extra) else n7_base_changes.copy()
    label = f"n7_{name}"
    match_df = apply_total_plus_changes(merged_changes, label)
    sub, path, check, metrics, ch = save_match_candidate(match_df, label, strict=False, changes=merged_changes)
    register_grid_record(n7_records, label, merged_changes, metrics, {"base_changed": len(n7_base_changes), "additional_changed": len(extra), "cap_extra": cap})
    all_variant_metrics.append(metrics); all_submission_checks.append(check)
    submission_catalog.append({"label": label, "path": str(path), "source": "n7_l6_combo"})
    if len(ch): changed_analysis_frames.append(ch.assign(variant_label=label))
    if len(extra): n7_frames.append(extra.assign(variant_label=label))
n7_grid = pd.DataFrame(n7_records)
n7_grid.to_csv(SUM_DIR / "n7_l6_combo_grid.csv", index=False)
log_saved(SUM_DIR / "n7_l6_combo_grid.csv")
safe_display_df(n7_grid)
n7_changes = pd.concat(n7_frames, ignore_index=True) if n7_frames else pd.DataFrame()
n7_changes.to_csv(SUM_DIR / "n7_l6_combo_changes.csv", index=False)
log_saved(SUM_DIR / "n7_l6_combo_changes.csv")

# %% [markdown]
# # 16. N8 — Toxic-Tail Surgery from Total-Plus Patch
#
# Penjelasan bagian:
# N8 mengambil total-plus cap0040/cap0050 lalu menghapus tail atau segment toxic seperti high-weight, low-support, atau noisy transition.
#
# Output yang perlu dilihat:
# `n8_toxic_tail_surgery_grid.csv` dan `n8_removed_changes.csv` menunjukkan removed count dan pair consistency.

# %%
n8_records, n8_removed_frames = [], []

def n8_make(base_table, remove_mask_func, name):
    if len(base_table) == 0:
        kept = base_table.copy(); removed = base_table.copy()
    else:
        rm_mask = remove_mask_func(base_table)
        removed = base_table.loc[rm_mask].copy()
        kept = base_table.loc[~rm_mask].copy()
    label = f"n8_{name}"
    match_df = apply_total_plus_changes(kept, label)
    sub, path, check, metrics, ch = save_match_candidate(match_df, label, strict=False, changes=kept)
    register_grid_record(n8_records, label, kept, metrics, {"n_removed": len(removed), "n_kept": len(kept), "remove_rule": name})
    all_variant_metrics.append(metrics); all_submission_checks.append(check)
    submission_catalog.append({"label": label, "path": str(path), "source": "n8_tail_surgery"})
    if len(ch): changed_analysis_frames.append(ch.assign(variant_label=label))
    if len(removed): n8_removed_frames.append(removed.assign(remove_rule=name))

cap0050_total_plus = select_top_by_cap(n6_table, 0.0050)
cap0040_total_plus = select_top_by_cap(n6_table, 0.0040)
if len(cap0050_total_plus):
    n8_make(cap0050_total_plus, lambda d: d["rank"] > max(d["rank"].max() - 10, 0), "total_plus_cap0050_minus_last_10")
    n8_make(cap0050_total_plus, lambda d: d["rank"] > max(d["rank"].max() - 20, 0), "total_plus_cap0050_minus_last_20")
    n8_make(cap0050_total_plus, lambda d: d["is_high_weight"].astype(bool), "total_plus_cap0050_minus_high_weight")
    n8_make(cap0050_total_plus, lambda d: d["support"] <= 2, "total_plus_cap0050_minus_low_support")
    noisy_tr = cap0050_total_plus.tail(max(1, len(cap0050_total_plus)//2))["transition"].value_counts().index[0]
    n8_make(cap0050_total_plus, lambda d, tr=noisy_tr: d["transition"].eq(tr), "total_plus_cap0050_minus_noisy_transition")
if len(cap0040_total_plus):
    n8_make(cap0040_total_plus, lambda d: d["rank"] > max(d["rank"].max() - 10, 0), "total_plus_cap0040_minus_last_10")
    noisy_tr2 = cap0040_total_plus.tail(max(1, len(cap0040_total_plus)//2))["transition"].value_counts().index[0]
    n8_make(cap0040_total_plus, lambda d, tr=noisy_tr2: d["transition"].eq(tr), "total_plus_cap0040_minus_noisy_transition")

n8_grid = pd.DataFrame(n8_records)
n8_grid.to_csv(SUM_DIR / "n8_toxic_tail_surgery_grid.csv", index=False)
log_saved(SUM_DIR / "n8_toxic_tail_surgery_grid.csv")
safe_display_df(n8_grid)
n8_removed = pd.concat(n8_removed_frames, ignore_index=True) if n8_removed_frames else pd.DataFrame()
n8_removed.to_csv(SUM_DIR / "n8_removed_changes.csv", index=False)
log_saved(SUM_DIR / "n8_removed_changes.csv")

# %% [markdown]
# # 17. N9 — Final Selected Safe, Catalog, dan Summary Metrics
#
# Penjelasan bagian:
# Main pipeline tidak memakai GT untuk automatic selection. `best_safe` dipilih berdasarkan sanity dan urutan konservatif: N0, N1/N2/N4 cap0025, lalu EXP39 fallback.
#
# Output yang perlu dilihat:
# `submission_exp12n_best_safe.csv`, `submission_catalog.csv`, `submission_check.csv`, dan `final_decision.csv`. Best_safe bukan klaim best local audit.

# %%
variant_metrics_df = pd.DataFrame(all_variant_metrics).drop_duplicates(subset=["label"], keep="last") if all_variant_metrics else pd.DataFrame()
variant_metrics_df.to_csv(SUM_DIR / "variant_metrics.csv", index=False)
log_saved(SUM_DIR / "variant_metrics.csv")
safe_display_df(variant_metrics_df.sort_values(["pair_consistency", "changed_rate_vs_exp39"], ascending=[False, True]).head(60) if len(variant_metrics_df) else variant_metrics_df)

changed_prediction_analysis = pd.concat([x for x in changed_analysis_frames if isinstance(x, pd.DataFrame) and len(x)], ignore_index=True) if changed_analysis_frames else pd.DataFrame()
changed_prediction_analysis.to_csv(SUM_DIR / "changed_prediction_analysis.csv", index=False)
log_saved(SUM_DIR / "changed_prediction_analysis.csv")

catalog_df = pd.DataFrame(submission_catalog)
if len(catalog_df) == 0:
    catalog_df = pd.DataFrame(columns=["label", "path", "source"])
catalog_df.to_csv(SUM_DIR / "submission_catalog.csv", index=False)
log_saved(SUM_DIR / "submission_catalog.csv")

submission_checks_df = pd.concat(all_submission_checks, ignore_index=True) if all_submission_checks else pd.DataFrame()
submission_checks_df.to_csv(SUM_DIR / "submission_check.csv", index=False)
log_saved(SUM_DIR / "submission_check.csv")

safe_order = [
    "n0_exp12m_champion_reproduction",
    "n1_total_plus_default_cap0025",
    "n2_total_plus_non_high_cap0025",
    "n4_default_first_then_support_cap0025",
]
sub_files = {p.stem.replace("submission_exp12n_", ""): p for p in SUB_DIR.glob("submission_exp12n_*.csv")}
selected_label, selected_path = None, None
for key in safe_order:
    skey = slugify_label(key)
    if skey in sub_files:
        selected_label, selected_path = key, sub_files[skey]
        break
if selected_path is None:
    selected_label, selected_path = "n0_exp12m_champion_reproduction", n0_path

best_safe_sub = normalize_submission_df(pd.read_csv(selected_path), require_all=True)
best_safe_path = SUB_DIR / "submission_exp12n_best_safe.csv"
best_safe_sub.to_csv(best_safe_path, index=False)
best_check = validation_checks(best_safe_sub)
best_check.insert(0, "label", "exp12n_best_safe")
best_check.to_csv(SUM_DIR / "submission_check_best_safe.csv", index=False)
if not bool(best_check["passed"].all()):
    display(best_check)
    raise RuntimeError("EXP12N best_safe validation failed")

final_decision = {
    "selected_by_pipeline": selected_label,
    "selected_by_sanity": selected_label,
    "recommended_for_local_audit": "audit all candidates in Section 99; best_safe is not GT-best claim",
    "main_submission_path": str(best_safe_path),
    "notes": "GT-free selection. Champion dibaca dari local_gt_audit_awmae_rank.csv.",
}
final_decision_df = pd.DataFrame([{"key": k, "value": str(v)} for k, v in final_decision.items()])
final_decision_df.to_csv(SUM_DIR / "final_decision.csv", index=False)
with open(SUM_DIR / "final_decision.json", "w", encoding="utf-8") as f:
    json.dump(final_decision, f, indent=2)
log_saved(best_safe_path)
log_saved(SUM_DIR / "final_decision.csv")
safe_display_df(final_decision_df)

# %% [markdown]
# # 99. OPTIONAL LOCAL GT AUDIT — DELETE BEFORE CLEAN SUBMISSION NOTEBOOK
#
# Penjelasan bagian:
# Cell ini membaca `ground_truth_bersih.csv` dan semua submission candidate, lalu menghitung AW-MAE weighted local audit. Cell ini hanya untuk audit lokal setelah semua submission dibuat.
#
# Output yang perlu dilihat:
# Lihat ranking `local_gt_audit_awmae_rank.csv` berisi AW-MAE, base_mae, exact_rate, outcome_rate, gd_rate, bias, pair consistency, dan path. Jangan gunakan cell ini untuk training, feature engineering, atau selection otomatis di pipeline utama.

# %%
RUN_LOCAL_GT_AUDIT = RUN_LOCAL_GT_AUDIT_DEFAULT

if RUN_LOCAL_GT_AUDIT:
    def audit_log(msg):
        print(f"[LOCAL GT AUDIT] {msg}", flush=True)

    gt_candidates = [
        PROJECT_ROOT / "data" / "ground_truth_bersih.csv",
        PROJECT_ROOT / "dataset" / "ground_truth_bersih.csv",
        PROJECT_ROOT / "ground_truth_bersih.csv",
        Path("data/ground_truth_bersih.csv"),
        Path("../data/ground_truth_bersih.csv"),
        Path("/mnt/data/ground_truth_bersih.csv"),
    ]
    GT_PATH = None
    for p in gt_candidates:
        if Path(p).exists():
            GT_PATH = Path(p)
            break
    if GT_PATH is None:
        raise FileNotFoundError("ground_truth_bersih.csv not found in data/ or dataset/")

    audit_log(f"GT_PATH = {GT_PATH}")
    gt = pd.read_csv(GT_PATH)
    gt_id_col = "Id" if "Id" in gt.columns else ("id" if "id" in gt.columns else gt.columns[0])
    gt = gt.rename(columns={gt_id_col: "Id"}).copy()
    required_gt_cols = ["Id", "team_goals", "opp_goals"]
    missing_gt_cols = [c for c in required_gt_cols if c not in gt.columns]
    if missing_gt_cols:
        raise KeyError(f"ground_truth_bersih.csv missing columns: {missing_gt_cols}")
    gt = gt[required_gt_cols].copy()
    gt["Id"] = gt["Id"].astype(str)
    gt = gt.rename(columns={"team_goals": "true_team_goals", "opp_goals": "true_opp_goals"})

    test_weight_df = test_raw[["Id"]].copy()
    test_weight_df["Id"] = test_weight_df["Id"].astype(str)
    test_weight_df["weight"] = test_raw["tournament"].apply(get_tournament_weight).astype(float) if "tournament" in test_raw.columns else 1.20
    gt = gt.merge(test_weight_df, on="Id", how="left")
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
        raw = base + EXACT_PENALTY * exact_miss.astype(float) + OUTCOME_PENALTY * outcome_miss.astype(float) + GD_PENALTY * gd_miss.astype(float)
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

    def audit_is_submission_file(p: Path) -> bool:
        name = p.name.lower()
        if not name.startswith("submission_") or not name.endswith(".csv"):
            return False
        if "sample" in name:
            return False
        if name in SUMMARY_SKIP_NAMES:
            return False
        if any(x in name for x in SUMMARY_SKIP_FRAGMENTS):
            return False
        return file_has_submission_columns(p)

    search_dirs = []
    for p in [
        SUB_DIR,
        OUTPUT_ROOT / "exp12n_total_plus_ranked_micro_patch_router",
        OUTPUT_ROOT / "exp12m_exp39_same_outcome_micro_patch_refinement",
        OUTPUT_ROOT / "exp12l_exp39_integration_hybrid_patch",
    ]:
        if Path(p).exists():
            search_dirs.append(Path(p))

    submission_paths = []
    for d in search_dirs:
        for p in d.rglob("*.csv"):
            if audit_is_submission_file(p):
                submission_paths.append(p.resolve())
    submission_paths = sorted(set(submission_paths), key=lambda x: str(x))
    if not submission_paths:
        raise FileNotFoundError("No submission_*.csv files found in expected output folders.")
    audit_log(f"Found {len(submission_paths)} submission files")

    rows, bad_rows = [], []
    for path in submission_paths:
        try:
            sub = normalize_submission_df(pd.read_csv(path), require_all=True)
            merged = gt.merge(sub.rename(columns={"team_goals": "pred_team_goals", "opp_goals": "pred_opp_goals"}), on="Id", how="left", validate="one_to_one")
            if merged[["pred_team_goals", "pred_opp_goals"]].isna().any().any():
                bad_rows.append({"file_name": path.name, "path": str(path), "reason": "missing predictions after merge"})
                continue
            metric = awmae_components(
                merged["true_team_goals"].values, merged["true_opp_goals"].values,
                merged["pred_team_goals"].values, merged["pred_opp_goals"].values,
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
    display(local_gt_audit_rank_df.head(30))
    local_gt_audit_rank_df.to_csv(SUM_DIR / "local_gt_audit_awmae_rank.csv", index=False)
    audit_log(f"Saved {SUM_DIR / 'local_gt_audit_awmae_rank.csv'}")
    if bad_rows:
        bad_df = pd.DataFrame(bad_rows)
        display(bad_df.head(30))
        bad_df.to_csv(SUM_DIR / "local_gt_audit_bad_files.csv", index=False)
        audit_log(f"Saved {SUM_DIR / 'local_gt_audit_bad_files.csv'}")
else:
    log_info("RUN_LOCAL_GT_AUDIT=False, skipping Section 99 local audit")
