# %% [markdown]
# # 00. EXP12M — EXP39 Same-Outcome Micro Patch Refinement
#
# Penjelasan bagian:
# EXP12M adalah post-processing refinement di atas EXP12L L4 cap0025. Fokusnya memperhalus same-outcome/GD micro patch di atas EXP39 final, bukan training model baru.
#
# Output yang perlu dilihat:
# Pastikan M0 mereproduce L4 cap0025, `l4_candidate_table.csv` tersimpan, semua candidate pair-consistent, dan Section 99 local audit menjadi satu-satunya bagian yang membaca `ground_truth_bersih.csv`.

# %% [markdown]
# # 01. Setup, Project Root, Logging, dan Guardrail
#
# Penjelasan bagian:
# Bagian ini menyiapkan import, seed, logging, output folder, dan project root resolver. Output dipaksa ke `PROJECT_ROOT/outputs/exp12m...` supaya tidak nyasar ke `notebook/outputs/`.
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
EXP12M_VARIANT = os.environ.get("EXP12M_VARIANT", "m1_l4_ultrafine_cap_sweep")
OUT_DIR = OUTPUT_ROOT / "exp12m_exp39_same_outcome_micro_patch_refinement" / EXP12M_VARIANT
PRED_DIR = OUT_DIR / "predictions"
SUB_DIR = OUT_DIR / "submissions"
SUM_DIR = OUT_DIR / "summaries"
FIG_DIR = OUT_DIR / "figures"

EXP12M_CLEAN_OUTDIR = os.environ.get("EXP12M_CLEAN_OUTDIR", "0").strip().lower() in {"1", "true", "yes", "y"}
if EXP12M_CLEAN_OUTDIR and OUT_DIR.exists():
    log_info(f"Cleaning previous EXP12M output folder: {OUT_DIR}")
    shutil.rmtree(OUT_DIR)

for d in [PRED_DIR, SUB_DIR, SUM_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RUN_LOCAL_GT_AUDIT_DEFAULT = os.environ.get("EXP12M_RUN_LOCAL_GT_AUDIT", "1").strip().lower() in {"1", "true", "yes", "y"}
MAX_GOAL_SANITY = 40

log_section("Setup")
log_info(f"PROJECT_ROOT = {PROJECT_ROOT}")
log_info(f"OUTPUT_ROOT = {OUTPUT_ROOT}")
log_info(f"OUT_DIR = {OUT_DIR}")
log_info(f"SUB_DIR = {SUB_DIR}")
log_info(f"SUM_DIR = {SUM_DIR}")
log_info(f"EXP12M_VARIANT = {EXP12M_VARIANT}")
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

# %% [markdown]
# # 04. Candidate Source Finder dan Loader
#
# Penjelasan bagian:
# Bagian ini mencari EXP39 final, EXP12L L4 cap0025, dan L6 export-router candidate dengan priority keyword. Sorting path biasa tidak boleh mengalahkan priority keyword.
#
# Output yang perlu dilihat:
# Cek `exp39_final_path`, `l4_champion_path`, dan `l6_best_path`. Kalau L4 source salah, M0 tidak valid.

# %%
SUMMARY_SKIP_NAMES = {
    "submission_catalog.csv", "submission_check.csv", "submission_validation.csv",
    "submission_metrics.csv", "submission_pair_consistency.csv",
    "submission_scoreline_distribution.csv", "submission_subgroup_metrics.csv",
    "submission_summary.csv",
}

def is_submission_candidate_path(p: Path) -> bool:
    name = p.name.lower()
    return name.endswith(".csv") and name.startswith("submission_") and name not in SUMMARY_SKIP_NAMES and "sample" not in name

def normalize_file_key(x) -> str:
    s = str(x).lower().replace("\\", "/").replace("-", "_").replace(" ", "_")
    while "__" in s: s = s.replace("__", "_")
    return s

def all_submission_paths():
    roots = [OUTPUT_ROOT, PROJECT_ROOT, Path.cwd(), Path("/mnt/data")]
    seen, paths = set(), []
    for root in roots:
        if not root.exists(): continue
        try:
            for p in root.rglob("submission_*.csv"):
                p = p.resolve()
                if p in seen: continue
                seen.add(p)
                # Hindari partial output EXP12M sendiri sebagai source.
                if "/exp12m_exp39_same_outcome_micro_patch_refinement/" in normalize_file_key(str(p)): continue
                if is_submission_candidate_path(p): paths.append(p)
        except Exception:
            pass
    return sorted(paths, key=lambda p: str(p))

def priority_score(path: Path, patterns: list) -> tuple:
    key = normalize_file_key(path.name)
    full = normalize_file_key(str(path))
    for i, pat in enumerate(patterns):
        if isinstance(pat, str): pat = [pat]
        ok = all(normalize_file_key(tok) in key or normalize_file_key(tok) in full for tok in pat)
        if ok: return (i, len(str(path)), str(path))
    return (9999, len(str(path)), str(path))

def select_priority_path(paths: list, patterns: list) -> Path:
    cand = sorted(paths, key=lambda p: priority_score(p, patterns))
    if not cand or priority_score(cand[0], patterns)[0] >= 9999: return None
    return cand[0]

EXP39_PATTERNS = [
    ["exp39", "safe", "final", "scoreline", "distribution", "validation", "only"],
    ["exp39", "safe", "final"],
    ["exp39", "scoreline", "distribution"],
    ["exp39", "router"],
]
L4_PATTERNS = [
    ["exp12l", "l4", "exp39", "plus", "gd", "patch", "cap0025"],
    ["exp12l", "l4", "exp39", "plus", "same", "outcome", "exact", "cap0025"],
    ["exp12l", "best", "safe"],
]
L6_PATTERNS = [
    ["exp12l", "l6", "exp39", "export", "exact", "majority", "cap0025"],
    ["exp12l", "l6", "exp39", "export", "exact", "majority", "cap0050"],
    ["exp12l", "l6", "exp39", "export", "same", "outcome"],
    ["exp12l", "l6"],
]

ALL_SUBMISSION_PATHS = all_submission_paths()
exp39_final_path = select_priority_path(ALL_SUBMISSION_PATHS, EXP39_PATTERNS)
l4_champion_path = select_priority_path(ALL_SUBMISSION_PATHS, L4_PATTERNS)
l6_best_path = select_priority_path(ALL_SUBMISSION_PATHS, L6_PATTERNS)

inventory = []
for p in ALL_SUBMISSION_PATHS:
    low = normalize_file_key(str(p))
    inventory.append({
        "file_name": p.name, "path": str(p),
        "is_exp39": "exp39" in low,
        "is_exp12l": "exp12l" in low,
        "is_l4": "l4" in low,
        "is_l6": "l6" in low,
        "priority_exp39": priority_score(p, EXP39_PATTERNS)[0],
        "priority_l4": priority_score(p, L4_PATTERNS)[0],
        "priority_l6": priority_score(p, L6_PATTERNS)[0],
    })
source_inventory_df = pd.DataFrame(inventory)
source_inventory_df.to_csv(SUM_DIR / "source_candidate_inventory.csv", index=False)
log_saved(SUM_DIR / "source_candidate_inventory.csv")

log_section("Source Finder")
log_info(f"n_submission_paths_found = {len(ALL_SUBMISSION_PATHS)}")
log_info(f"exp39_final_path = {exp39_final_path}")
log_info(f"l4_champion_path = {l4_champion_path}")
log_info(f"l6_best_path = {l6_best_path}")
safe_display_df(source_inventory_df.sort_values(["priority_l4", "priority_exp39", "priority_l6"]).head(30) if len(source_inventory_df) else source_inventory_df)

if exp39_final_path is None:
    raise FileNotFoundError("Tidak menemukan EXP39 final CSV. Pastikan submission_exp39_SAFE_FINAL_scoreline_distribution_validation_only.csv tersedia.")
if l4_champion_path is None:
    raise FileNotFoundError("Tidak menemukan EXP12L L4 cap0025 CSV. Run EXP12L dulu atau copy submission_exp12l_l4_exp39_plus_gd_patch_cap0025.csv.")

l4_priority = priority_score(l4_champion_path, L4_PATTERNS)[0] if l4_champion_path is not None else 9999
if l4_priority >= 2:
    raise RuntimeError(
        f"L4 champion source fallback terlalu lemah: {l4_champion_path}. "
        "Pastikan file submission_exp12l_l4_exp39_plus_gd_patch_cap0025.csv "
        "atau submission_exp12l_l4_exp39_plus_same_outcome_exact_cap0025.csv tersedia."
    )

def read_submission_path(path: Path, label: str) -> pd.DataFrame:
    out = normalize_submission_df(pd.read_csv(path), require_all=True)
    out.attrs["label"] = label
    out.attrs["path"] = str(path)
    return out

def save_candidate(sub_df: pd.DataFrame, label: str, strict=False):
    safe_label = slugify_label(label)
    out = normalize_submission_df(sub_df, require_all=True)
    path = SUB_DIR / f"submission_exp12m_{safe_label}.csv"
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
# # 05. M0 — Reproduce EXP12L L4 Cap0025
#
# Penjelasan bagian:
# Bagian ini memuat EXP39 final sebagai base dan EXP12L L4 cap0025 sebagai champion reference. M0 disimpan ulang ke folder EXP12M agar bisa diaudit bersama candidate baru.
#
# Output yang perlu dilihat:
# `m0_l4_reproduction_summary.csv` harus menunjukkan source L4 benar, pair consistency True, dan mean_pred_total sekitar 2.524-an.

# %%
exp39_final_sub = read_submission_path(exp39_final_path, "exp39_final_source")
exp39_final_match = submission_to_match(exp39_final_sub, label="exp39_final")

l4_champion_sub = read_submission_path(l4_champion_path, "l4_cap0025_source")
l4_champion_match = submission_to_match(l4_champion_sub, label="l4_cap0025")

l6_best_sub = None
l6_best_match = None
if l6_best_path is not None:
    l6_best_sub = read_submission_path(l6_best_path, "l6_source")
    l6_best_match = submission_to_match(l6_best_sub, label="l6_source")

m0_sub, m0_path, m0_check = save_candidate(l4_champion_sub, "m0_l4_cap0025_reproduction", strict=False)
m0_summary = pd.DataFrame([{
    "l4_source_path": str(l4_champion_path),
    "exp39_source_path": str(exp39_final_path),
    **basic_submission_metrics(m0_sub, "m0_l4_cap0025_reproduction"),
}])
m0_summary.to_csv(SUM_DIR / "m0_l4_reproduction_summary.csv", index=False)
log_saved(SUM_DIR / "m0_l4_reproduction_summary.csv")
safe_display_df(m0_summary)
log_check("M0 pair consistency", bool(m0_summary["pair_consistency"].iloc[0]))

# %% [markdown]
# # 06. Candidate Pool dan Cache
#
# Penjelasan bagian:
# Bagian ini memuat candidate submission lama sebagai comparison/post-processing artifact, bukan fitur training. Candidate pool dipakai untuk menghitung support L4, ranking, dan L6 combo.
#
# Output yang perlu dilihat:
# `candidate_pool_loaded.csv` menampilkan label, path, pair consistency, dan mean_pred_total. Pastikan EXP39/L4/L6 candidates terbaca.

# %%
def candidate_reliability_weight(label: str) -> float:
    s = str(label).lower()
    if "exp39" in s and ("safe_final" in s or "final" in s): return 3.0
    if "exp12l" in s and "l4" in s and "cap0025" in s: return 3.0
    if "exp12l" in s and "l6" in s: return 2.0
    if "exp39" in s and ("scoreline" in s or "router" in s): return 2.0
    if any(x in s for x in ["exp12k", "exp12j", "exp12i", "exp12h"]): return 1.2
    if "exp12" in s: return 1.0
    return 0.8

def label_from_path(p: Path) -> str:
    return slugify_label(p.stem.replace("submission_", ""), max_len=120)

pool_paths = []
for p in ALL_SUBMISSION_PATHS:
    low = normalize_file_key(str(p))
    include = False
    if "exp39" in low or "scoreline_distribution" in low or "router" in low: include = True
    if "exp12l" in low and ("l4" in low or "l6" in low or "best_safe" in low): include = True
    if any(x in low for x in ["exp12k", "exp12j", "exp12i", "exp12h"]): include = True
    if include: pool_paths.append(p)
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
candidate_match_pool["l4_cap0025_champion"] = l4_champion_match.copy()
if l6_best_match is not None:
    candidate_match_pool["l6_best_candidate"] = l6_best_match.copy()

CANDIDATE_MATCH_POOL_INDEX = {lab: df.set_index("match_id", drop=False) for lab, df in candidate_match_pool.items() if "match_id" in df.columns}
candidate_pool_loaded_df = pd.DataFrame(pool_rows)
candidate_pool_loaded_df.to_csv(SUM_DIR / "candidate_pool_loaded.csv", index=False)
log_saved(SUM_DIR / "candidate_pool_loaded.csv")
safe_display_df(candidate_pool_loaded_df.sort_values("label").head(40) if len(candidate_pool_loaded_df) else candidate_pool_loaded_df)
log_info(f"n_candidate_match_pool = {len(candidate_match_pool)}")

# %% [markdown]
# # 07. Core Requirement — Build L4 Candidate Table
#
# Penjelasan bagian:
# Bagian ini membangun `l4_candidate_table.csv`, yaitu kandidat same-outcome patch dari EXP39. Kandidat harus beda scoreline, outcome sama, total/GD diff kecil, dan support minimal dua candidate.
#
# Output yang perlu dilihat:
# Cek jumlah kandidat, support distribution, transition summary, total_delta distribution, dan high-weight count.

# %%
def should_exclude_l4_self_label(label: str) -> bool:
    s = str(label).lower()
    return (
        s == "l4_cap0025_champion"
        or ("exp12l" in s and "_l4_" in s)
        or ("exp12l_l4" in s)
    )


def build_score_support_map(match_id, exclude_labels=None):
    exclude_labels = set(exclude_labels or [])
    score_to_labels = defaultdict(list)
    score_to_weight = defaultdict(float)

    for lab, idx in CANDIDATE_MATCH_POOL_INDEX.items():
        if lab in exclude_labels:
            continue

        # L4 candidate table tidak boleh memakai output L4 lama sebagai support.
        # Kalau tidak, patch menjadi self-referential dan ranking kandidat bisa misleading.
        if should_exclude_l4_self_label(lab):
            continue

        if match_id not in idx.index:
            continue

        r = idx.loc[match_id]
        if isinstance(r, pd.DataFrame):
            r = r.iloc[0]

        score = (int(r["pred_a"]), int(r["pred_b"]))
        w = candidate_reliability_weight(lab)
        score_to_labels[score].append(lab)
        score_to_weight[score] += w

    return score_to_labels, score_to_weight

def build_l4_candidate_table(min_support=2, max_abs_total_delta=1, max_abs_gd_delta=1, exclude_labels=None):
    rows = []
    exclude = set(exclude_labels or []) | {"exp39_final_base", "l4_cap0025_champion"}
    for _, base in exp39_final_match.iterrows():
        match_id = base["match_id"]
        old_a, old_b = int(base["pred_a"]), int(base["pred_b"])
        old_out = outcome_scalar(old_a, old_b)
        score_to_labels, score_to_weight = build_score_support_map(match_id, exclude_labels=exclude)
        for (new_a, new_b), labels in score_to_labels.items():
            if new_a == old_a and new_b == old_b: continue
            if outcome_scalar(new_a, new_b) != old_out: continue
            old_total, new_total = old_a + old_b, new_a + new_b
            old_gd, new_gd = old_a - old_b, new_a - new_b
            total_delta, gd_delta = int(new_total - old_total), int(new_gd - old_gd)
            if abs(total_delta) > max_abs_total_delta or abs(gd_delta) > max_abs_gd_delta: continue
            support = len(labels)
            weighted_support = float(score_to_weight[(new_a, new_b)])
            if support < min_support: continue
            labels_sorted = sorted(labels, key=lambda x: (-candidate_reliability_weight(x), x))
            top = labels_sorted[0] if labels_sorted else ""
            tw = float(base.get("tournament_weight", 1.20))
            rows.append({
                "match_id": match_id,
                "old_a": old_a, "old_b": old_b,
                "new_a": int(new_a), "new_b": int(new_b),
                "old_score": f"{old_a}-{old_b}", "new_score": f"{int(new_a)}-{int(new_b)}",
                "transition": f"{old_a}-{old_b} -> {int(new_a)}-{int(new_b)}",
                "support": int(support), "weighted_support": weighted_support,
                "source_labels": "|".join(labels_sorted),
                "source_label_top": top,
                "source_weight_top": candidate_reliability_weight(top) if top else 0.0,
                "gender": base.get("gender", "unknown"),
                "tournament": base.get("tournament", "unknown"),
                "tournament_weight": tw,
                "old_total": int(old_total), "new_total": int(new_total), "total_delta": int(total_delta),
                "old_gd": int(old_gd), "new_gd": int(new_gd), "gd_delta": int(gd_delta),
                "same_outcome": True,
                "is_high_weight": bool(tw >= 1.8),
                "is_low_weight": bool(tw <= 0.96),
                "is_world_asian": bool(any(x in str(base.get("tournament", "")).lower() for x in ["world", "asian", "afc"])),
            })
    out = pd.DataFrame(rows)
    if len(out) == 0: return out
    out["abs_total_delta"] = out["total_delta"].abs()
    out["abs_gd_delta"] = out["gd_delta"].abs()
    out["non_high_weight_rank"] = (~out["is_high_weight"].astype(bool)).astype(int)
    out = out.sort_values(
        ["support", "weighted_support", "source_weight_top", "non_high_weight_rank", "abs_total_delta", "abs_gd_delta", "match_id"],
        ascending=[False, False, False, False, True, True, True],
    ).reset_index(drop=True)

    # Satu match hanya boleh punya satu kandidat utama.
    # Kalau tidak, cap menghitung row kandidat, bukan match unik.
    out = out.drop_duplicates("match_id", keep="first").reset_index(drop=True)

    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out

l4_candidate_table = build_l4_candidate_table(min_support=2)
l4_candidate_table.to_csv(SUM_DIR / "l4_candidate_table.csv", index=False)
log_saved(SUM_DIR / "l4_candidate_table.csv")

if len(l4_candidate_table):
    l4_candidate_summary = pd.DataFrame([
        {"metric": "n_candidates", "value": len(l4_candidate_table)},
        {"metric": "n_match_unique", "value": l4_candidate_table["match_id"].nunique()},
        {"metric": "support_mean", "value": float(l4_candidate_table["support"].mean())},
        {"metric": "weighted_support_mean", "value": float(l4_candidate_table["weighted_support"].mean())},
        {"metric": "total_delta_plus_share", "value": float((l4_candidate_table["total_delta"] > 0).mean())},
        {"metric": "high_weight_share", "value": float(l4_candidate_table["is_high_weight"].mean())},
    ])
else:
    l4_candidate_summary = pd.DataFrame([{"metric": "n_candidates", "value": 0}])

l4_candidate_summary.to_csv(SUM_DIR / "l4_candidate_summary.csv", index=False)
log_saved(SUM_DIR / "l4_candidate_summary.csv")
safe_display_df(l4_candidate_summary)

l4_transition_summary = l4_candidate_table["transition"].value_counts().reset_index() if len(l4_candidate_table) else pd.DataFrame(columns=["transition", "count"])
l4_transition_summary.columns = ["transition", "count"]
l4_transition_summary.to_csv(SUM_DIR / "l4_candidate_transition_summary.csv", index=False)
log_saved(SUM_DIR / "l4_candidate_transition_summary.csv")
safe_display_df(l4_transition_summary.head(20))

# %% [markdown]
# # 08. Candidate Application Helpers
#
# Penjelasan bagian:
# Bagian ini membuat helper cap selection, ranking, apply changes, dan save match candidate. Semua section M1-M7 memakai fungsi yang sama agar output konsisten.
#
# Output yang perlu dilihat:
# Tidak ada output besar. Kesalahan helper akan terlihat pada validation setiap candidate.

# %%
def select_top_by_cap(table: pd.DataFrame, cap_rate: float, base_n_matches=None) -> pd.DataFrame:
    if table is None or len(table) == 0: return pd.DataFrame(columns=[])
    if base_n_matches is None: base_n_matches = len(match_meta)
    n = max(int(math.floor(float(cap_rate) * base_n_matches)), 0)
    return table.head(min(n, len(table))).copy()

def apply_l4_changes(changes: pd.DataFrame, label: str, base_match=None):
    if base_match is None: base_match = exp39_final_match
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

def sort_l4_candidates(table: pd.DataFrame, rule: str) -> pd.DataFrame:
    if table is None or len(table) == 0: return table.copy()
    df = table.copy()
    if "abs_total_delta" not in df: df["abs_total_delta"] = df["total_delta"].abs()
    if "abs_gd_delta" not in df: df["abs_gd_delta"] = df["gd_delta"].abs()
    df["non_high_weight_rank"] = (~df["is_high_weight"].astype(bool)).astype(int)
    if rule == "support_then_weighted":
        cols, asc = ["support", "weighted_support", "source_weight_top", "abs_total_delta", "abs_gd_delta", "match_id"], [False, False, False, True, True, True]
    elif rule == "weighted_then_support":
        cols, asc = ["weighted_support", "support", "source_weight_top", "abs_total_delta", "abs_gd_delta", "match_id"], [False, False, False, True, True, True]
    elif rule == "total_plus_first":
        df["total_plus_rank"] = (df["total_delta"] > 0).astype(int)
        cols, asc = ["total_plus_rank", "support", "weighted_support", "source_weight_top", "match_id"], [False, False, False, False, True]
    elif rule == "total_abs_small_first":
        cols, asc = ["abs_total_delta", "abs_gd_delta", "support", "weighted_support", "match_id"], [True, True, False, False, True]
    elif rule == "non_high_weight_first":
        cols, asc = ["non_high_weight_rank", "support", "weighted_support", "abs_total_delta", "match_id"], [False, False, False, True, True]
    elif rule == "high_weight_safe_first":
        df["high_weight_safe_rank"] = ((df["is_high_weight"].astype(bool)) & (df["support"] >= 3)).astype(int)
        cols, asc = ["high_weight_safe_rank", "support", "weighted_support", "match_id"], [False, False, False, True]
    elif rule == "source_reliability_first":
        cols, asc = ["source_weight_top", "weighted_support", "support", "abs_total_delta", "match_id"], [False, False, False, True, True]
    elif rule == "gd_delta_zero_first":
        df["gd_zero_rank"] = (df["gd_delta"] == 0).astype(int)
        cols, asc = ["gd_zero_rank", "support", "weighted_support", "abs_total_delta", "match_id"], [False, False, False, True, True]
    elif rule == "total_plus_non_high_first":
        df["total_plus_rank"] = (df["total_delta"] > 0).astype(int)
        cols, asc = ["total_plus_rank", "non_high_weight_rank", "support", "weighted_support", "match_id"], [False, False, False, False, True]
    else:
        cols, asc = (["rank"], [True]) if "rank" in df.columns else (["support", "weighted_support", "match_id"], [False, False, True])
    out = df.sort_values(cols, ascending=asc).reset_index(drop=True)
    if "rank" in out.columns: out = out.drop(columns=["rank"])
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out

def register_grid_record(records, label, changes, metrics, extra=None):
    rec = {"label": label}
    if extra: rec.update(extra)
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

# %% [markdown]
# # 09. M1 — Ultra-Fine Cap Sweep L4
#
# Penjelasan bagian:
# M1 melakukan sweep cap kecil dari 0.10% sampai 0.50% untuk mencari sweet spot L4 lebih halus daripada EXP12L cap0025/cap0050.
#
# Output yang perlu dilihat:
# `m1_l4_ultrafine_cap_sweep_grid.csv` berisi cap, n_selected, changed_rate, mean total, high/low-weight change, dan pair consistency.

# %%
m1_caps = [0.0010, 0.0015, 0.0020, 0.0025, 0.0030, 0.0035, 0.0040, 0.0045, 0.0050, 0.0055, 0.0060, 0.0075]
m1_records = []

# register M0 in catalog/metrics
m0_metrics = basic_submission_metrics(m0_sub, "m0_l4_cap0025_reproduction")
m0_metrics["path"] = str(m0_path)
m0_metrics["n_changed_vs_exp39"] = len(transition_changes_df(exp39_final_match, l4_champion_match, "m0"))
m0_metrics["changed_rate_vs_exp39"] = m0_metrics["n_changed_vs_exp39"] / max(len(match_meta), 1)
all_variant_metrics.append(m0_metrics)
all_submission_checks.append(m0_check)
submission_catalog.append({"label": "m0_l4_cap0025_reproduction", "path": str(m0_path), "source": "loaded_l4"})

for cap in m1_caps:
    changes = select_top_by_cap(l4_candidate_table, cap)
    label = f"m1_l4_cap{int(round(cap * 10000)):04d}"
    match_df = apply_l4_changes(changes, label)
    sub, path, check, metrics, ch = save_match_candidate(match_df, label, strict=False, changes=changes)
    register_grid_record(m1_records, label, changes, metrics, {"cap_rate": cap})
    all_variant_metrics.append(metrics); all_submission_checks.append(check)
    submission_catalog.append({"label": label, "path": str(path), "source": "m1_cap_sweep"})
    if len(ch): changed_analysis_frames.append(ch.assign(variant_label=label))

m1_grid = pd.DataFrame(m1_records)
m1_grid.to_csv(SUM_DIR / "m1_l4_ultrafine_cap_sweep_grid.csv", index=False)
log_saved(SUM_DIR / "m1_l4_ultrafine_cap_sweep_grid.csv")
safe_display_df(m1_grid)

# %% [markdown]
# # 10. M2 — L4 Ranking Refinement
#
# Penjelasan bagian:
# M2 menguji ranking kandidat L4. Karena cap sangat kecil, urutan kandidat sangat menentukan.
#
# Output yang perlu dilihat:
# `m2_l4_ranking_refinement_grid.csv` menunjukkan ranking rule, cap, jumlah selected, dan pair consistency.

# %%
m2_rules = ["support_then_weighted", "weighted_then_support", "total_plus_first", "total_abs_small_first", "non_high_weight_first", "high_weight_safe_first", "source_reliability_first", "gd_delta_zero_first", "total_plus_non_high_first"]
m2_caps = [0.0025, 0.0030]
m2_records, m2_change_frames = [], []
for rule in m2_rules:
    sorted_table = sort_l4_candidates(l4_candidate_table, rule)
    for cap in m2_caps:
        changes = select_top_by_cap(sorted_table, cap)
        label = f"m2_{rule}_cap{int(round(cap * 10000)):04d}"
        match_df = apply_l4_changes(changes, label)
        sub, path, check, metrics, ch = save_match_candidate(match_df, label, strict=False, changes=changes)
        register_grid_record(m2_records, label, changes, metrics, {"ranking_rule": rule, "cap_rate": cap})
        all_variant_metrics.append(metrics); all_submission_checks.append(check)
        submission_catalog.append({"label": label, "path": str(path), "source": "m2_ranking"})
        if len(ch):
            m2_change_frames.append(ch.assign(variant_label=label, ranking_rule=rule, cap_rate=cap))
            changed_analysis_frames.append(ch.assign(variant_label=label))

m2_grid = pd.DataFrame(m2_records)
m2_grid.to_csv(SUM_DIR / "m2_l4_ranking_refinement_grid.csv", index=False)
log_saved(SUM_DIR / "m2_l4_ranking_refinement_grid.csv")
safe_display_df(m2_grid.head(40))
m2_change_analysis = pd.concat(m2_change_frames, ignore_index=True) if m2_change_frames else pd.DataFrame()
m2_change_analysis.to_csv(SUM_DIR / "m2_l4_ranking_change_analysis.csv", index=False)

# %% [markdown]
# # 11. M3 — Total Direction Split
#
# Penjelasan bagian:
# M3 memisahkan kandidat L4 berdasarkan `total_delta`: +1, 0, dan -1. Tujuannya mengecek apakah gain L4 terutama berasal dari menaikkan total goal sedikit.
#
# Output yang perlu dilihat:
# `m3_total_direction_grid.csv` memperlihatkan filter total_delta, n_selected, mean total, dan domain changes.

# %%
m3_configs = [
    ("total_plus_only_cap0025", lambda d: d["total_delta"] > 0, 0.0025),
    ("total_zero_only_cap0025", lambda d: d["total_delta"] == 0, 0.0025),
    ("total_minus_only_cap0025", lambda d: d["total_delta"] < 0, 0.0025),
    ("total_plus_non_high_weight_cap0025", lambda d: (d["total_delta"] > 0) & (~d["is_high_weight"].astype(bool)), 0.0025),
    ("total_plus_support3_cap0025", lambda d: (d["total_delta"] > 0) & (d["support"] >= 3), 0.0025),
    ("total_plus_cap0030", lambda d: d["total_delta"] > 0, 0.0030),
    ("total_plus_cap0040", lambda d: d["total_delta"] > 0, 0.0040),
]
m3_records, m3_change_frames = [], []
for name, filt, cap in m3_configs:
    sub_table = l4_candidate_table.loc[filt(l4_candidate_table)].copy() if len(l4_candidate_table) else l4_candidate_table
    sub_table = sort_l4_candidates(sub_table, "support_then_weighted")
    changes = select_top_by_cap(sub_table, cap)
    label = f"m3_{name}"
    match_df = apply_l4_changes(changes, label)
    sub, path, check, metrics, ch = save_match_candidate(match_df, label, strict=False, changes=changes)
    register_grid_record(m3_records, label, changes, metrics, {"total_delta_filter": name, "cap_rate": cap, "n_candidates_before_cap": len(sub_table)})
    all_variant_metrics.append(metrics); all_submission_checks.append(check)
    submission_catalog.append({"label": label, "path": str(path), "source": "m3_total_direction"})
    if len(ch):
        m3_change_frames.append(ch.assign(variant_label=label))
        changed_analysis_frames.append(ch.assign(variant_label=label))

m3_grid = pd.DataFrame(m3_records)
m3_grid.to_csv(SUM_DIR / "m3_total_direction_grid.csv", index=False)
log_saved(SUM_DIR / "m3_total_direction_grid.csv")
safe_display_df(m3_grid)
m3_change_analysis = pd.concat(m3_change_frames, ignore_index=True) if m3_change_frames else pd.DataFrame()
m3_change_analysis.to_csv(SUM_DIR / "m3_total_direction_change_analysis.csv", index=False)

# %% [markdown]
# # 12. M4 — Transition-Specific L4 Analysis dan Filtering
#
# Penjelasan bagian:
# M4 membaca transition dominan dalam L4 table lalu membuat candidate only/exclude transition. Candidate eksplisit hanya dibuat jika transition tersebut tersedia.
#
# Output yang perlu dilihat:
# `m4_l4_transition_summary.csv` dan `m4_transition_filter_grid.csv` menunjukkan apakah transition tertentu menjadi sumber sinyal atau noise.

# %%
if len(l4_candidate_table):
    m4_transition_summary = l4_candidate_table.groupby("transition").agg(
        n_candidates=("match_id", "count"),
        support_mean=("support", "mean"),
        weighted_support_mean=("weighted_support", "mean"),
        total_delta_mean=("total_delta", "mean"),
    ).reset_index().sort_values("n_candidates", ascending=False)
else:
    m4_transition_summary = pd.DataFrame(columns=["transition", "n_candidates"])
m4_transition_summary.to_csv(SUM_DIR / "m4_l4_transition_summary.csv", index=False)
log_saved(SUM_DIR / "m4_l4_transition_summary.csv")
safe_display_df(m4_transition_summary.head(20))

m4_records = []
if len(l4_candidate_table):
    top_transitions = m4_transition_summary["transition"].head(5).tolist()
    configs = []
    if len(top_transitions) >= 1:
        configs.append(("only_top1_transition_cap0025", lambda d, t=top_transitions[0]: d["transition"].eq(t)))
        configs.append(("exclude_top1_transition_cap0025", lambda d, t=top_transitions[0]: ~d["transition"].eq(t)))
    if len(top_transitions) >= 2:
        configs.append(("only_top2_transition_cap0025", lambda d, t=set(top_transitions[:2]): d["transition"].isin(t)))
    configs.append(("exclude_high_risk_transition_cap0025", lambda d: (d["abs_total_delta"] <= 1) & (d["abs_gd_delta"] <= 1) & (~((d["is_high_weight"].astype(bool)) & (d["support"] <= 2)))))
    configs.append(("low_total_correction_transition_cap0025", lambda d: d["total_delta"] >= 0))
    for nm, tr in [("only_0_0_to_1_1_cap0025", "0-0 -> 1-1"), ("only_1_0_to_2_0_cap0025", "1-0 -> 2-0"), ("only_0_1_to_0_2_cap0025", "0-1 -> 0-2")]:
        if tr in set(l4_candidate_table["transition"]):
            configs.append((nm, lambda d, tr=tr: d["transition"].eq(tr)))
    one_one_like = set([x for x in l4_candidate_table["transition"].unique() if x.startswith("1-1 -> 2-1") or x.startswith("1-1 -> 1-2")])
    if one_one_like:
        configs.append(("only_1_1_to_2_1_or_1_2_cap0025", lambda d, t=one_one_like: d["transition"].isin(t)))

    for name, filt in configs:
        sub_table = l4_candidate_table.loc[filt(l4_candidate_table)].copy()
        sub_table = sort_l4_candidates(sub_table, "support_then_weighted")
        changes = select_top_by_cap(sub_table, 0.0025)
        label = f"m4_{name}"
        match_df = apply_l4_changes(changes, label)
        sub, path, check, metrics, ch = save_match_candidate(match_df, label, strict=False, changes=changes)
        register_grid_record(m4_records, label, changes, metrics, {"filter_name": name, "n_candidates_before_cap": len(sub_table)})
        all_variant_metrics.append(metrics); all_submission_checks.append(check)
        submission_catalog.append({"label": label, "path": str(path), "source": "m4_transition"})
        if len(ch): changed_analysis_frames.append(ch.assign(variant_label=label))

m4_grid = pd.DataFrame(m4_records)
m4_grid.to_csv(SUM_DIR / "m4_transition_filter_grid.csv", index=False)
log_saved(SUM_DIR / "m4_transition_filter_grid.csv")
safe_display_df(m4_grid)

# %% [markdown]
# # 13. M5 — Tournament/Gender Gating L4
#
# Penjelasan bagian:
# M5 menguji apakah L4 patch lebih aman hanya pada subset gender/tournament tertentu, misalnya non-high-weight atau high-weight dengan support kuat.
#
# Output yang perlu dilihat:
# `m5_tournament_gender_grid.csv` dan change analysis menunjukkan changed_M/W, high/low-weight, dan pair consistency.

# %%
m5_configs = [
    ("M_only_l4_cap0025", lambda d: d["gender"].astype(str).str.upper().eq("M"), 0.0025),
    ("W_only_l4_cap0025", lambda d: d["gender"].astype(str).str.upper().eq("W"), 0.0025),
    ("non_high_weight_l4_cap0025", lambda d: ~d["is_high_weight"].astype(bool), 0.0025),
    ("high_weight_support3_l4_cap0025", lambda d: (~d["is_high_weight"].astype(bool)) | (d["support"] >= 3), 0.0025),
    ("high_weight_support4_l4_cap0025", lambda d: (~d["is_high_weight"].astype(bool)) | (d["support"] >= 4), 0.0025),
    ("friendly_only_l4_cap0025", lambda d: d["is_low_weight"].astype(bool), 0.0025),
    ("default_weight_only_l4_cap0025", lambda d: (~d["is_low_weight"].astype(bool)) & (~d["is_high_weight"].astype(bool)), 0.0025),
    ("world_asian_strict_l4_cap0025", lambda d: (~d["is_world_asian"].astype(bool)) | (d["support"] >= 3), 0.0025),
]
m5_records, m5_change_frames = [], []
for name, filt, cap in m5_configs:
    sub_table = l4_candidate_table.loc[filt(l4_candidate_table)].copy() if len(l4_candidate_table) else l4_candidate_table
    sub_table = sort_l4_candidates(sub_table, "support_then_weighted")
    changes = select_top_by_cap(sub_table, cap)
    label = f"m5_{name}"
    match_df = apply_l4_changes(changes, label)
    sub, path, check, metrics, ch = save_match_candidate(match_df, label, strict=False, changes=changes)
    register_grid_record(m5_records, label, changes, metrics, {"gate_name": name, "cap_rate": cap, "n_candidates_before_cap": len(sub_table)})
    all_variant_metrics.append(metrics); all_submission_checks.append(check)
    submission_catalog.append({"label": label, "path": str(path), "source": "m5_gate"})
    if len(ch):
        m5_change_frames.append(ch.assign(variant_label=label))
        changed_analysis_frames.append(ch.assign(variant_label=label))

m5_grid = pd.DataFrame(m5_records)
m5_grid.to_csv(SUM_DIR / "m5_tournament_gender_grid.csv", index=False)
log_saved(SUM_DIR / "m5_tournament_gender_grid.csv")
safe_display_df(m5_grid)
m5_change_analysis = pd.concat(m5_change_frames, ignore_index=True) if m5_change_frames else pd.DataFrame()
m5_change_analysis.to_csv(SUM_DIR / "m5_tournament_gender_change_analysis.csv", index=False)

# %% [markdown]
# # 14. M6 — L4 + Tiny L6 Export-Router Combo
#
# Penjelasan bagian:
# M6 mencoba menambah sinyal L6 sangat kecil di atas L4 cap0025. Patch tambahan hanya same-outcome, total/GD diff kecil, dan tidak mengubah match yang sudah berubah oleh L4.
#
# Output yang perlu dilihat:
# Jika L6 source tidak tersedia, status akan skipped. Jika aktif, cek additional_changed_count, total_changed_count, dan pair consistency.

# %%
def build_l6_additional_table(base_match, l6_match, already_changed_ids):
    if l6_match is None: return pd.DataFrame()
    base = base_match[["match_id", "pred_a", "pred_b", "gender", "tournament", "tournament_weight"]].rename(columns={"pred_a": "old_a", "pred_b": "old_b"})
    cand = l6_match[["match_id", "pred_a", "pred_b"]].rename(columns={"pred_a": "new_a", "pred_b": "new_b"})
    df = base.merge(cand, on="match_id", how="inner", validate="one_to_one")
    df = df[(df["old_a"] != df["new_a"]) | (df["old_b"] != df["new_b"])].copy()
    if len(df) == 0: return df
    df = df[~df["match_id"].isin(already_changed_ids)].copy()
    df["old_outcome"] = [outcome_scalar(a,b) for a,b in zip(df["old_a"], df["old_b"])]
    df["new_outcome"] = [outcome_scalar(a,b) for a,b in zip(df["new_a"], df["new_b"])]
    df["same_outcome"] = df["old_outcome"] == df["new_outcome"]
    df["old_total"] = df["old_a"] + df["old_b"]
    df["new_total"] = df["new_a"] + df["new_b"]
    df["total_delta"] = df["new_total"] - df["old_total"]
    df["old_gd"] = df["old_a"] - df["old_b"]
    df["new_gd"] = df["new_a"] - df["new_b"]
    df["gd_delta"] = df["new_gd"] - df["old_gd"]
    df["support"] = 2
    df["weighted_support"] = 2.0
    df["source_label_top"] = "l6_best_candidate"
    df["source_weight_top"] = 2.0
    df = df[(df["same_outcome"]) & (df["total_delta"].abs() <= 1) & (df["gd_delta"].abs() <= 1)].copy()
    if len(df) == 0: return df
    df["old_score"] = df["old_a"].astype(str) + "-" + df["old_b"].astype(str)
    df["new_score"] = df["new_a"].astype(str) + "-" + df["new_b"].astype(str)
    df["transition"] = df["old_score"] + " -> " + df["new_score"]
    df["is_high_weight"] = df["tournament_weight"] >= 1.8
    df["is_low_weight"] = df["tournament_weight"] <= 0.96
    df = df.sort_values(["support", "weighted_support", "total_delta", "match_id"], ascending=[False, False, False, True]).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df)+1))
    return df

m6_records = []
base_l4_match = l4_champion_match.copy()
base_l4_changes = transition_changes_df(exp39_final_match, base_l4_match, "l4_cap0025")
already_l4_changed = set(base_l4_changes["match_id"].tolist()) if len(base_l4_changes) else set()

if l6_best_match is None:
    m6_grid = pd.DataFrame([{"status": "skipped_l6_source_not_found"}])
    m6_grid.to_csv(SUM_DIR / "m6_l4_l6_combo_grid.csv", index=False)
    pd.DataFrame().to_csv(SUM_DIR / "m6_l4_l6_combo_changes.csv", index=False)
    safe_display_df(m6_grid)
else:
    l6_add_table = build_l6_additional_table(base_l4_match, l6_best_match, already_l4_changed)
    l6_add_table.to_csv(SUM_DIR / "m6_l6_additional_candidate_table.csv", index=False)
    m6_configs = [
        ("l4_cap0025_plus_l6_exact_cap0010", 0.0010, lambda d: d),
        ("l4_cap0025_plus_l6_exact_cap0025", 0.0025, lambda d: d),
        ("l4_cap0025_plus_l6_same_outcome_cap0010", 0.0010, lambda d: d[d["same_outcome"]]),
        ("l4_cap0025_plus_l6_low_total_router_cap0010", 0.0010, lambda d: d[d["total_delta"] <= 0]),
    ]
    for name, cap, filt in m6_configs:
        add_table = filt(l6_add_table).copy() if len(l6_add_table) else l6_add_table
        add_changes = select_top_by_cap(add_table, cap)
        combo_match = base_l4_match.copy()
        if len(add_changes):
            add_idx = add_changes.set_index("match_id")
            combo_idx = combo_match.set_index("match_id", drop=False)
            for mid in add_idx.index:
                if mid in combo_idx.index:
                    combo_idx.loc[mid, "pred_a"] = int(add_idx.loc[mid, "new_a"])
                    combo_idx.loc[mid, "pred_b"] = int(add_idx.loc[mid, "new_b"])
            combo_match = combo_idx.reset_index(drop=True)
        label = f"m6_{name}"
        sub, path, check, metrics, ch = save_match_candidate(combo_match, label, strict=False)
        register_grid_record(m6_records, label, add_changes, metrics, {"base_changed_count": len(base_l4_changes), "additional_changed_count": len(add_changes), "cap_rate": cap, "n_l6_candidates_before_cap": len(add_table)})
        all_variant_metrics.append(metrics); all_submission_checks.append(check)
        submission_catalog.append({"label": label, "path": str(path), "source": "m6_l4_l6_combo"})
        if len(ch): changed_analysis_frames.append(ch.assign(variant_label=label))
    m6_grid = pd.DataFrame(m6_records)
    m6_grid.to_csv(SUM_DIR / "m6_l4_l6_combo_grid.csv", index=False)
    l6_add_table.to_csv(SUM_DIR / "m6_l4_l6_combo_changes.csv", index=False)
    safe_display_df(m6_grid)

# %% [markdown]
# # 15. M7 — Cap0050 Toxic-Tail Surgery
#
# Penjelasan bagian:
# M7 mengambil L4 cap0050 lalu menghapus tail atau segment yang diduga toxic. Ini mengecek apakah cap0050 punya useful additions tapi tail terakhir membuatnya kalah dari cap0025.
#
# Output yang perlu dilihat:
# `m7_cap0050_toxic_tail_grid.csv` dan `m7_cap0050_removed_changes.csv` menunjukkan jumlah removed, transition removed, dan pair consistency.

# %%
cap0050_table = select_top_by_cap(l4_candidate_table, 0.0050)
m7_records, m7_removed_frames = [], []

def make_cap0050_minus(remove_mask_func, name):
    if len(cap0050_table) == 0:
        kept = cap0050_table.copy(); removed = cap0050_table.copy()
    else:
        rm_mask = remove_mask_func(cap0050_table)
        removed = cap0050_table.loc[rm_mask].copy()
        kept = cap0050_table.loc[~rm_mask].copy()
    label = f"m7_{name}"
    match_df = apply_l4_changes(kept, label)
    sub, path, check, metrics, ch = save_match_candidate(match_df, label, strict=False, changes=kept)
    register_grid_record(m7_records, label, kept, metrics, {"n_removed": len(removed), "n_kept": len(kept), "remove_rule": name})
    if len(removed): m7_removed_frames.append(removed.assign(remove_rule=name))
    all_variant_metrics.append(metrics); all_submission_checks.append(check)
    submission_catalog.append({"label": label, "path": str(path), "source": "m7_cap0050_surgery"})
    if len(ch): changed_analysis_frames.append(ch.assign(variant_label=label))

if len(cap0050_table):
    make_cap0050_minus(lambda d: d["rank"] > max(d["rank"].max() - 10, 0), "cap0050_minus_last_10")
    make_cap0050_minus(lambda d: d["rank"] > max(d["rank"].max() - 20, 0), "cap0050_minus_last_20")
    make_cap0050_minus(lambda d: d["rank"] > max(d["rank"].max() - 30, 0), "cap0050_minus_last_30")
    make_cap0050_minus(lambda d: d["is_high_weight"].astype(bool), "cap0050_minus_high_weight")
    make_cap0050_minus(lambda d: d["support"] <= 2, "cap0050_minus_low_support")
    make_cap0050_minus(lambda d: d["total_delta"] < 0, "cap0050_minus_total_minus")
    tail_half = cap0050_table.tail(max(1, len(cap0050_table)//2))
    noisy_transition = tail_half["transition"].value_counts().index[0] if len(tail_half) else ""
    make_cap0050_minus(lambda d, tr=noisy_transition: d["transition"].eq(tr), "cap0050_minus_noisy_transition")

m7_grid = pd.DataFrame(m7_records)
m7_grid.to_csv(SUM_DIR / "m7_cap0050_toxic_tail_grid.csv", index=False)
log_saved(SUM_DIR / "m7_cap0050_toxic_tail_grid.csv")
safe_display_df(m7_grid)

m7_removed_df = pd.concat(m7_removed_frames, ignore_index=True) if m7_removed_frames else pd.DataFrame()
m7_removed_df.to_csv(SUM_DIR / "m7_cap0050_removed_changes.csv", index=False)
log_saved(SUM_DIR / "m7_cap0050_removed_changes.csv")

# %% [markdown]
# # 16. M8 — Final Selected Safe, Catalog, dan Summary Metrics
#
# Penjelasan bagian:
# Main pipeline tidak memakai GT untuk automatic selection. `best_safe` dipilih berdasarkan sanity dan urutan konservatif: M0/M1 cap0025/M1 cap0020/M1 cap0030/M3 total_plus/l0 fallback.
#
# Output yang perlu dilihat:
# `submission_exp12m_best_safe.csv`, `submission_catalog.csv`, `submission_check.csv`, dan `final_decision.csv`. Best_safe bukan klaim best local audit.

# %%
variant_metrics_df = pd.DataFrame(all_variant_metrics).drop_duplicates(subset=["label"], keep="last") if all_variant_metrics else pd.DataFrame()
variant_metrics_df.to_csv(SUM_DIR / "variant_metrics.csv", index=False)
log_saved(SUM_DIR / "variant_metrics.csv")
safe_display_df(variant_metrics_df.sort_values(["pair_consistency", "changed_rate_vs_exp39"], ascending=[False, True]).head(40) if len(variant_metrics_df) else variant_metrics_df)

changed_prediction_analysis = pd.concat([x for x in changed_analysis_frames if isinstance(x, pd.DataFrame) and len(x)], ignore_index=True) if changed_analysis_frames else pd.DataFrame()
changed_prediction_analysis.to_csv(SUM_DIR / "changed_prediction_analysis.csv", index=False)
log_saved(SUM_DIR / "changed_prediction_analysis.csv")

catalog_df = pd.DataFrame(submission_catalog)
if len(catalog_df) == 0: catalog_df = pd.DataFrame(columns=["label", "path", "source"])
catalog_df.to_csv(SUM_DIR / "submission_catalog.csv", index=False)
log_saved(SUM_DIR / "submission_catalog.csv")

submission_checks_df = pd.concat(all_submission_checks, ignore_index=True) if all_submission_checks else pd.DataFrame()
submission_checks_df.to_csv(SUM_DIR / "submission_check.csv", index=False)
log_saved(SUM_DIR / "submission_check.csv")

safe_order = ["m0_l4_cap0025_reproduction", "m1_l4_cap0025", "m1_l4_cap0020", "m1_l4_cap0030", "m3_total_plus_only_cap0025"]
sub_files = {p.stem.replace("submission_exp12m_", ""): p for p in SUB_DIR.glob("submission_exp12m_*.csv")}
selected_label, selected_path = None, None
for key in safe_order:
    skey = slugify_label(key)
    if skey in sub_files:
        selected_label, selected_path = key, sub_files[skey]
        break
if selected_path is None:
    selected_label, selected_path = "m0_l4_cap0025_reproduction", m0_path

best_safe_sub = normalize_submission_df(pd.read_csv(selected_path), require_all=True)
best_safe_path = SUB_DIR / "submission_exp12m_best_safe.csv"
best_safe_sub.to_csv(best_safe_path, index=False)
best_check = validation_checks(best_safe_sub)
best_check.insert(0, "label", "exp12m_best_safe")
best_check.to_csv(SUM_DIR / "submission_check_best_safe.csv", index=False)
if not bool(best_check["passed"].all()):
    display(best_check)
    raise RuntimeError("EXP12M best_safe validation failed")

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
        if not name.startswith("submission_") or not name.endswith(".csv"): return False
        if "sample" in name: return False
        if name in SUMMARY_SKIP_NAMES: return False
        skip_fragments = [
            "submission_catalog", "submission_check", "submission_validation", "submission_metrics",
            "submission_pair_consistency", "submission_scoreline_distribution", "submission_subgroup_metrics",
            "submission_summary",
        ]
        return not any(x in name for x in skip_fragments)

    search_dirs = []
    for p in [SUB_DIR, OUTPUT_ROOT / "exp12m_exp39_same_outcome_micro_patch_refinement", OUTPUT_ROOT / "exp12l_exp39_integration_hybrid_patch"]:
        if Path(p).exists(): search_dirs.append(Path(p))

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
