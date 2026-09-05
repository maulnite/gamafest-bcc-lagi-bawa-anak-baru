
# %% [markdown]
# # 00. EXP12Q — Same-GD Sweet-Spot Refinement
#
# Penjelasan bagian:
# EXP12Q adalah post-processing sweet-spot refinement di atas EXP12P. Eksperimen ini tidak melatih model baru, tetapi memperhalus family same-outcome + same-GD + total-lift di area cap0060.
#
# Output yang perlu dilihat:
# Pastikan Q0 mereproduce EXP12P champion, `same_gd_total_lift_candidate_table.csv` tersimpan, semua candidate pair-consistent, dan Section 99 menjadi satu-satunya bagian yang membaca ground truth.

# %% [markdown]
# # 01. Setup, Project Root, Logging, dan Guardrail
#
# Penjelasan bagian:
# Bagian ini menyiapkan import, seed, logging, output folder, dan project root resolver. Output dipaksa ke `PROJECT_ROOT/outputs/exp12q_same_gd_sweet_spot_refinement/` supaya tidak masuk ke folder notebook.
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
        try:
            cand = cand.resolve()
        except Exception:
            continue
        if (cand / "data").exists() or (cand / "dataset").exists():
            return cand
    if Path("/mnt/data").exists() and ((Path("/mnt/data") / "test.csv").exists() or (Path("/mnt/data") / "sample submission.csv").exists()):
        return Path("/mnt/data")
    return cwd

PROJECT_ROOT = infer_project_root()
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
EXP12Q_VARIANT = os.environ.get("EXP12Q_VARIANT", "q1_same_gd_sweet_spot")
OUT_DIR = OUTPUT_ROOT / "exp12q_same_gd_sweet_spot_refinement" / EXP12Q_VARIANT
PRED_DIR = OUT_DIR / "predictions"
SUB_DIR = OUT_DIR / "submissions"
SUM_DIR = OUT_DIR / "summaries"
FIG_DIR = OUT_DIR / "figures"

EXP12Q_CLEAN_OUTDIR = os.environ.get("EXP12Q_CLEAN_OUTDIR", "0").strip().lower() in {"1", "true", "yes", "y"}
if EXP12Q_CLEAN_OUTDIR and OUT_DIR.exists():
    log_info(f"Cleaning previous EXP12Q output folder: {OUT_DIR}")
    shutil.rmtree(OUT_DIR)

for d in [PRED_DIR, SUB_DIR, SUM_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RUN_LOCAL_GT_AUDIT_DEFAULT = os.environ.get("EXP12Q_RUN_LOCAL_GT_AUDIT", "1").strip().lower() in {"1", "true", "yes", "y"}
MAX_GOAL_SANITY = 40

log_section("Setup")
log_info(f"PROJECT_ROOT = {PROJECT_ROOT}")
log_info(f"OUTPUT_ROOT = {OUTPUT_ROOT}")
log_info(f"OUT_DIR = {OUT_DIR}")
log_info(f"SUB_DIR = {SUB_DIR}")
log_info(f"SUM_DIR = {SUM_DIR}")
log_info(f"EXP12Q_VARIANT = {EXP12Q_VARIANT}")
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
        try:
            d = d.resolve()
        except Exception:
            continue
        if d.exists() and d not in out:
            out.append(d)
    return out


def find_data_file(names, recursive=True) -> Path:
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
    if a > b:
        return 1
    if a < b:
        return -1
    return 0


def outcome_array(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return np.where(a > b, 1, np.where(a < b, -1, 0))


def slugify_label(label: str, max_len=140) -> str:
    s = str(label).strip().lower().replace("/", "_").replace("\\", "_")
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    if len(s) > max_len:
        s = s[:max_len].rstrip("_")
    return s or "candidate"


def normalize_submission_df(df: pd.DataFrame, require_all=True) -> pd.DataFrame:
    sub = df.copy()
    id_col = "Id" if "Id" in sub.columns else ("id" if "id" in sub.columns else sub.columns[0])
    sub = sub.rename(columns={id_col: "Id"}).copy()
    required = ["Id", "team_goals", "opp_goals"]
    missing = [c for c in required if c not in sub.columns]
    if missing:
        raise KeyError(f"submission missing columns: {missing}")
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
            if len(g) < 2:
                continue
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
        if id_a not in sub_idx.index or id_b not in sub_idx.index:
            continue
        ra, rb = sub_idx.loc[id_a], sub_idx.loc[id_b]
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
    if "detail" not in out.columns:
        out["detail"] = ""
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
        ch["old_outcome"] = [outcome_scalar(a, b) for a, b in zip(ch["old_a"], ch["old_b"])]
        ch["new_outcome"] = [outcome_scalar(a, b) for a, b in zip(ch["new_a"], ch["new_b"])]
        ch["same_outcome"] = ch["old_outcome"] == ch["new_outcome"]
        ch["same_gd"] = ch["old_gd"] == ch["new_gd"]
        ch["label"] = label
    return ch

log_section("Match Meta Audit")
safe_display_df(pd.DataFrame([{"key": "n_test_rows", "value": len(test_raw)}, {"key": "n_match_meta", "value": len(match_meta)}]))



# %% [markdown]
# # 04. Candidate Source Finder dan Valid Submission Loader
#
# Penjelasan bagian:
# Bagian ini mencari EXP39 final, EXP12N anchor, EXP12P champion, dan candidate submission lain. Finder membaca header CSV supaya tidak salah mengambil summary/check/catalog.
#
# Output yang perlu dilihat:
# Cek `exp39_final_path`, `exp12n_anchor_path`, dan `exp12p_champion_path`. File source wajib punya kolom Id/id, team_goals, dan opp_goals.

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
        return ("id" in low) and {"team_goals", "opp_goals"}.issubset(low)
    except Exception:
        return False


def is_submission_candidate_path(p: Path) -> bool:
    name = p.name.lower()
    parts = {x.lower() for x in p.parts}
    if p.suffix.lower() != ".csv":
        return False
    if "summaries" in parts or "figures" in parts:
        return False
    if name in SUMMARY_SKIP_NAMES:
        return False
    if any(frag in name for frag in SUMMARY_SKIP_FRAGMENTS):
        return False
    if "sample" in name:
        return False
    if not name.startswith("submission_"):
        return False
    return file_has_submission_columns(p)


def all_submission_candidates():
    roots = [PROJECT_ROOT, OUTPUT_ROOT, Path("/mnt/data")]
    out, seen = [], set()
    for root in roots:
        if not Path(root).exists():
            continue
        try:
            for p in Path(root).rglob("submission_*.csv"):
                if is_submission_candidate_path(p):
                    key = str(p.resolve())
                    if key not in seen:
                        seen.add(key)
                        out.append(p.resolve())
        except Exception:
            pass
    return sorted(out, key=lambda x: str(x))


def label_from_path(p: Path) -> str:
    name = p.stem
    parts = [part for part in p.parts if part.startswith("exp12") or part.startswith("exp39") or part.startswith("exp38")]
    prefix = parts[-1] if parts else ""
    lab = f"{prefix}_{name}" if prefix and prefix not in name else name
    return slugify_label(lab)


def priority_score(path: Path, priority_keywords):
    s = (path.name + " " + str(path)).lower().replace("_", " ").replace("-", " ")
    for i, kws in enumerate(priority_keywords):
        if all(kw.lower().replace("_", " ") in s for kw in kws):
            return i
    return 10_000


def find_by_priority(priority_keywords, description, required=True):
    candidates = all_submission_candidates()
    scored = [(priority_score(p, priority_keywords), len(str(p)), str(p), p) for p in candidates]
    scored = [x for x in scored if x[0] < 10_000]
    if not scored:
        if required:
            raise FileNotFoundError(f"Tidak menemukan {description}. Pastikan CSV submission tersedia.")
        return None
    scored.sort(key=lambda x: (x[0], x[1], x[2]))
    return scored[0][3]

EXP39_PRIORITY = [
    ["exp39", "safe", "final", "scoreline", "distribution", "validation", "only"],
    ["exp39", "safe", "final"],
    ["exp39", "scoreline", "distribution"],
    ["exp39", "router"],
]
EXP12N_PRIORITY = [
    ["exp12n", "n5", "best", "filtered", "cap0050"],
    ["exp12n", "best", "safe"],
    ["exp12m", "m3", "total", "plus"],
    ["exp12m", "best", "safe"],
]
EXP12P_PRIORITY = [
    ["exp12p", "p1", "same", "gd", "cap0060"],
    ["exp12p", "p6", "same", "gd", "cap0060", "guard", "0100"],
    ["exp12p", "p7", "cap0060", "minus", "high", "weight"],
    ["exp12p", "p7", "cap0060", "minus", "last", "10"],
    ["exp12p", "best", "safe"],
]
EXP12O_PRIORITY = [
    ["exp12o", "o1", "anchor", "plus", "same", "gd", "cap0030"],
    ["exp12o", "o7", "same", "gd", "cap0030", "minus"],
    ["exp12o", "best", "safe"],
]
L4_PRIORITY = [
    ["exp12l", "l4", "gd", "patch", "cap0025"],
    ["exp12l", "l4", "same", "outcome", "exact", "cap0025"],
    ["exp12l", "best", "safe"],
]

exp39_final_path = find_by_priority(EXP39_PRIORITY, "EXP39 final")
exp12n_anchor_path = find_by_priority(EXP12N_PRIORITY, "EXP12N anchor")
exp12p_champion_path = find_by_priority(EXP12P_PRIORITY, "EXP12P champion")
exp12o_reference_path = find_by_priority(EXP12O_PRIORITY, "EXP12O reference", required=False)
l4_reference_path = find_by_priority(L4_PRIORITY, "EXP12L L4 reference", required=False)

source_df = pd.DataFrame([
    {"source": "exp39_final_path", "path": str(exp39_final_path)},
    {"source": "exp12n_anchor_path", "path": str(exp12n_anchor_path)},
    {"source": "exp12p_champion_path", "path": str(exp12p_champion_path)},
    {"source": "exp12o_reference_path", "path": str(exp12o_reference_path) if exp12o_reference_path else "NOT_FOUND"},
    {"source": "l4_reference_path", "path": str(l4_reference_path) if l4_reference_path else "NOT_FOUND"},
])
source_df.to_csv(SUM_DIR / "source_paths.csv", index=False)
log_section("Source Paths")
safe_display_df(source_df)

# %% [markdown]
# # 05. Load Anchor Submissions dan Candidate Pool
#
# Penjelasan bagian:
# Bagian ini memuat EXP39 final, EXP12N anchor untuk konstruksi same-GD, dan EXP12P champion sebagai base terbaru. Candidate pool mengecualikan self-reference dari output patch EXP12L/M/N/O/P/Q.
#
# Output yang perlu dilihat:
# Candidate pool harus berisi beberapa EXP39/export/older candidate, tetapi tidak boleh berisi output patch EXP12P/Q sebagai support.

# %%

def read_submission_path(p: Path) -> pd.DataFrame:
    return normalize_submission_df(pd.read_csv(p), require_all=True)

exp39_final_sub = read_submission_path(exp39_final_path)
exp12n_anchor_sub = read_submission_path(exp12n_anchor_path)
exp12p_champion_sub = read_submission_path(exp12p_champion_path)
exp12o_reference_sub = read_submission_path(exp12o_reference_path) if exp12o_reference_path else None
l4_reference_sub = read_submission_path(l4_reference_path) if l4_reference_path else None

exp39_final_match = submission_to_match(exp39_final_sub, label="exp39_final")
exp12n_anchor_match = submission_to_match(exp12n_anchor_sub, label="exp12n_anchor")
exp12p_champion_match = submission_to_match(exp12p_champion_sub, label="exp12p_champion")
exp12o_reference_match = submission_to_match(exp12o_reference_sub, label="exp12o_reference") if exp12o_reference_sub is not None else None
l4_reference_match = submission_to_match(l4_reference_sub, label="l4_reference") if l4_reference_sub is not None else None

submission_catalog = []
submission_checks_all = []
variant_metrics_records = []
changed_records = []


def save_candidate(sub_df: pd.DataFrame, label: str, strict=False):
    aligned = normalize_submission_df(sub_df, require_all=True)
    path = SUB_DIR / f"submission_exp12q_{slugify_label(label)}.csv"
    aligned.to_csv(path, index=False)
    check_df = validation_checks(aligned)
    check_df.insert(0, "label", label)
    submission_checks_all.append(check_df)
    submission_catalog.append({"label": label, "path": str(path)})
    metrics = basic_submission_metrics(aligned, label)
    metrics["path"] = str(path)
    variant_metrics_records.append(metrics)
    log_saved(path)
    if strict and not bool(check_df["passed"].all()):
        display(check_df)
        raise RuntimeError(f"Submission validation failed for {label}")
    return aligned, path, check_df

# Q0 champion reproduction.
q0_sub, q0_path, _ = save_candidate(exp12p_champion_sub, "q0_exp12p_champion_reproduction", strict=False)

candidate_paths = all_submission_candidates()
raw_inventory = [{"label": label_from_path(p), "path": str(p)} for p in candidate_paths]
raw_inventory_df = pd.DataFrame(raw_inventory)
raw_inventory_df.to_csv(SUM_DIR / "raw_submission_inventory.csv", index=False)

SELF_REFERENCE_FRAGMENTS = [
    "exp12l_l4", "l4_cap0025",
    "exp12m_m0", "exp12m_m1", "exp12m_m2", "exp12m_m3", "exp12m_best",
    "exp12n_n0", "exp12n_n1", "exp12n_n2", "exp12n_n3", "exp12n_n4", "exp12n_n5", "exp12n_n6", "exp12n_n7", "exp12n_n8", "exp12n_best",
    "exp12o_o0", "exp12o_o1", "exp12o_o2", "exp12o_o3", "exp12o_o4", "exp12o_o5", "exp12o_o6", "exp12o_o7", "exp12o_best",
    "exp12p_p0", "exp12p_p1", "exp12p_p2", "exp12p_p3", "exp12p_p4", "exp12p_p5", "exp12p_p6", "exp12p_p7", "exp12p_p8", "exp12p_best",
    "exp12q",
]


def is_self_reference_label(label: str) -> bool:
    s = str(label).lower()
    return any(frag in s for frag in SELF_REFERENCE_FRAGMENTS)


def candidate_reliability_weight(label: str) -> float:
    s = str(label).lower()
    if "exp39" in s and ("safe" in s or "scoreline" in s):
        return 3.0
    if "exp39" in s or "exp38" in s:
        return 2.5
    if "exp12l_l6" in s or "export" in s or "router" in s:
        return 2.0
    if "exp12k" in s or "exp12j" in s or "exp12i" in s or "exp12h" in s:
        return 1.2
    if "exp12" in s:
        return 1.0
    return 0.8

candidate_match_pool = {}
load_errors = []
for p in candidate_paths:
    lab = label_from_path(p)
    if is_self_reference_label(lab):
        continue
    try:
        sub = read_submission_path(p)
        m = submission_to_match(sub, label=lab)
        if len(m) != len(match_meta):
            continue
        candidate_match_pool[lab] = m
    except Exception as e:
        load_errors.append({"label": lab, "path": str(p), "error": repr(e)})

candidate_match_pool["exp39_final_base"] = exp39_final_match.copy()
CANDIDATE_MATCH_POOL_INDEX = {lab: df.set_index("match_id", drop=False) for lab, df in candidate_match_pool.items() if "match_id" in df.columns}

candidate_pool_loaded = pd.DataFrame([
    {"label": lab, "n_matches": len(df), "weight": candidate_reliability_weight(lab), "self_reference": is_self_reference_label(lab)}
    for lab, df in candidate_match_pool.items()
]).sort_values(["weight", "label"], ascending=[False, True]).reset_index(drop=True)
candidate_pool_loaded.to_csv(SUM_DIR / "candidate_pool_loaded.csv", index=False)
if load_errors:
    pd.DataFrame(load_errors).to_csv(SUM_DIR / "candidate_load_errors.csv", index=False)
log_section("Candidate Pool")
safe_display_df(candidate_pool_loaded)
log_check("Q0 champion pair consistency", pair_consistency_report(exp12p_champion_sub)["pair_consistency"])

# %% [markdown]
# # 06. Same-GD Total-Lift Candidate Table
#
# Penjelasan bagian:
# Bagian ini membangun `same_gd_total_lift_candidate_table.csv`. Base konstruksi mengikuti EXP12P, yaitu EXP12N anchor, lalu family filter same-outcome + same-GD + total-lift diterapkan sebelum drop duplicate match.
#
# Output yang perlu dilihat:
# Tabel harus tidak kosong, bebas self-reference, dan unique by match_id setelah family filter.

# %%

def enrich_candidate_record(anchor_row, cand_a, cand_b, support_labels, support_weight_sum):
    old_a, old_b = int(anchor_row["pred_a"]), int(anchor_row["pred_b"])
    new_a, new_b = int(cand_a), int(cand_b)
    source_labels = sorted(support_labels)
    source_weights = [(lab, candidate_reliability_weight(lab)) for lab in source_labels]
    source_label_top, source_weight_top = (max(source_weights, key=lambda x: x[1]) if source_weights else ("", 0.0))
    old_total, new_total = old_a + old_b, new_a + new_b
    old_gd, new_gd = old_a - old_b, new_a - new_b
    tournament_weight = float(anchor_row.get("tournament_weight", 1.2))
    tournament = str(anchor_row.get("tournament", "unknown"))
    t_low = tournament.lower()
    return {
        "match_id": anchor_row["match_id"],
        "old_a": old_a, "old_b": old_b, "new_a": new_a, "new_b": new_b,
        "old_score": f"{old_a}-{old_b}", "new_score": f"{new_a}-{new_b}",
        "transition": f"{old_a}-{old_b} -> {new_a}-{new_b}",
        "support": int(len(source_labels)),
        "weighted_support": float(support_weight_sum),
        "source_labels": "|".join(source_labels),
        "source_label_top": source_label_top,
        "source_weight_top": float(source_weight_top),
        "gender": str(anchor_row.get("gender", "unknown")),
        "tournament": tournament,
        "tournament_weight": tournament_weight,
        "old_total": old_total,
        "new_total": new_total,
        "total_delta": new_total - old_total,
        "old_gd": old_gd,
        "new_gd": new_gd,
        "gd_delta": new_gd - old_gd,
        "abs_total_delta": abs(new_total - old_total),
        "abs_gd_delta": abs(new_gd - old_gd),
        "same_outcome": outcome_scalar(old_a, old_b) == outcome_scalar(new_a, new_b),
        "same_gd": old_gd == new_gd,
        "is_high_weight": tournament_weight >= 1.8,
        "is_low_weight": tournament_weight <= 0.96,
        "is_default_weight": abs(tournament_weight - 1.2) < 1e-9,
        "is_world_asian": ("world" in t_low) or ("asian" in t_low) or ("afc" in t_low),
        "is_friendly": "friendly" in t_low,
    }


def build_vote_records(anchor_match: pd.DataFrame):
    anchor_idx = anchor_match.set_index("match_id", drop=False)
    vote_records = []
    for mid, anchor_row in anchor_idx.iterrows():
        score_support = defaultdict(list)
        score_weight = defaultdict(float)
        for lab, cand_idx in CANDIDATE_MATCH_POOL_INDEX.items():
            if is_self_reference_label(lab):
                continue
            if mid not in cand_idx.index:
                continue
            row = cand_idx.loc[mid]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            a, b = int(row["pred_a"]), int(row["pred_b"])
            if a == int(anchor_row["pred_a"]) and b == int(anchor_row["pred_b"]):
                continue
            key = (a, b)
            score_support[key].append(lab)
            score_weight[key] += candidate_reliability_weight(lab)
        for (a, b), labs in score_support.items():
            vote_records.append(enrich_candidate_record(anchor_row, a, b, labs, score_weight[(a, b)]))
    return pd.DataFrame(vote_records)


def finalize_table(df: pd.DataFrame, sort_cols, asc):
    if df is None or len(df) == 0:
        return pd.DataFrame()
    out = df.copy()
    for c in sort_cols:
        if c not in out.columns:
            out[c] = 0
    out = out.sort_values(sort_cols, ascending=asc).drop_duplicates("match_id", keep="first").reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out

all_vote_candidate_table = build_vote_records(exp12n_anchor_match)
if len(all_vote_candidate_table):
    all_vote_candidate_table = all_vote_candidate_table.sort_values(
        ["support", "weighted_support", "source_weight_top", "is_default_weight", "is_high_weight", "abs_total_delta", "match_id"],
        ascending=[False, False, False, False, True, True, True],
    ).reset_index(drop=True)
    all_vote_candidate_table.insert(0, "vote_rank", np.arange(1, len(all_vote_candidate_table) + 1))
all_vote_candidate_table.to_csv(SUM_DIR / "all_vote_candidate_table.csv", index=False)

same_gd_filter = (
    (all_vote_candidate_table.get("same_outcome", pd.Series(False, index=all_vote_candidate_table.index)).astype(bool))
    & (all_vote_candidate_table.get("same_gd", pd.Series(False, index=all_vote_candidate_table.index)).astype(bool))
    & (all_vote_candidate_table.get("total_delta", pd.Series(0, index=all_vote_candidate_table.index)) > 0)
    & (all_vote_candidate_table.get("support", pd.Series(0, index=all_vote_candidate_table.index)) >= 2)
) if len(all_vote_candidate_table) else pd.Series([], dtype=bool)

same_gd_total_lift_candidate_table = finalize_table(
    all_vote_candidate_table[same_gd_filter].copy() if len(all_vote_candidate_table) else pd.DataFrame(),
    ["support", "weighted_support", "source_weight_top", "is_high_weight", "is_default_weight", "abs_total_delta", "match_id"],
    [False, False, False, True, False, True, True],
)
same_gd_total_lift_candidate_table.to_csv(SUM_DIR / "same_gd_total_lift_candidate_table.csv", index=False)


def table_summary(df: pd.DataFrame, name: str):
    if df is None or len(df) == 0:
        return pd.DataFrame([{"table": name, "n_rows": 0, "n_unique_matches": 0}])
    rows = [{
        "table": name,
        "n_rows": len(df),
        "n_unique_matches": int(df["match_id"].nunique()),
        "mean_support": float(df["support"].mean()),
        "mean_weighted_support": float(df["weighted_support"].mean()),
        "mean_total_delta": float(df["total_delta"].mean()),
        "n_high_weight": int(df["is_high_weight"].sum()),
        "n_default_weight": int(df["is_default_weight"].sum()),
        "n_low_weight": int(df["is_low_weight"].sum()),
    }]
    for cap in [0.0050, 0.0055, 0.0060, 0.0075]:
        rows[0][f"n_cap_{int(cap*10000):04d}"] = int(min(len(df), max(0, math.floor(len(match_meta) * cap))))
    return pd.DataFrame(rows)


def transition_summary(df: pd.DataFrame):
    if df is None or len(df) == 0:
        return pd.DataFrame()
    return df.groupby("transition", as_index=False).agg(
        n_candidates=("match_id", "count"),
        mean_support=("support", "mean"),
        mean_weighted_support=("weighted_support", "mean"),
        share_default_weight=("is_default_weight", "mean"),
        share_non_high_weight=("is_high_weight", lambda s: float((~s.astype(bool)).mean())),
        share_high_weight=("is_high_weight", "mean"),
        mean_total_delta=("total_delta", "mean"),
    ).sort_values(["n_candidates", "mean_weighted_support"], ascending=[False, False]).reset_index(drop=True)


def segment_summary(df: pd.DataFrame):
    if df is None or len(df) == 0:
        return pd.DataFrame()
    rows = []
    for col in ["gender", "tournament_weight", "is_high_weight", "is_default_weight", "is_low_weight", "is_world_asian", "is_friendly"]:
        if col in df.columns:
            g = df.groupby(col).agg(n=("match_id", "count"), mean_support=("support", "mean"), mean_total_delta=("total_delta", "mean")).reset_index()
            g.insert(0, "segment_col", col)
            g = g.rename(columns={col: "segment_value"})
            rows.append(g)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

same_gd_total_lift_summary = table_summary(same_gd_total_lift_candidate_table, "same_gd_total_lift")
same_gd_total_lift_transition_summary = transition_summary(same_gd_total_lift_candidate_table)
same_gd_total_lift_segment_summary = segment_summary(same_gd_total_lift_candidate_table)
same_gd_total_lift_summary.to_csv(SUM_DIR / "same_gd_total_lift_summary.csv", index=False)
same_gd_total_lift_transition_summary.to_csv(SUM_DIR / "same_gd_total_lift_transition_summary.csv", index=False)
same_gd_total_lift_segment_summary.to_csv(SUM_DIR / "same_gd_total_lift_segment_summary.csv", index=False)

log_section("Same-GD Table Summary")
safe_display_df(same_gd_total_lift_summary)
log_check("same_gd table unique match_id", len(same_gd_total_lift_candidate_table) == same_gd_total_lift_candidate_table["match_id"].nunique() if len(same_gd_total_lift_candidate_table) else True)

# %% [markdown]
# # 07. Patch Application Helper
#
# Penjelasan bagian:
# Helper ini memilih top candidate berdasar cap/prefix/target mean, menerapkan perubahan ke match-level base, lalu menyimpan submission pair-consistent.
#
# Output yang perlu dilihat:
# Semua candidate yang tersimpan harus pair_consistency=True. Changed rate harus kecil dan terkontrol.

# %%

def filter_table(df: pd.DataFrame, mask):
    if df is None or len(df) == 0:
        return pd.DataFrame()
    if callable(mask):
        mask = mask(df)
    if mask is None:
        mask = pd.Series(True, index=df.index)
    mask = pd.Series(mask, index=df.index).fillna(False).astype(bool)
    return df[mask].copy()


def reorder_table(df: pd.DataFrame, sort_cols, ascending):
    if df is None or len(df) == 0:
        return pd.DataFrame()
    out = df.copy()
    for c in sort_cols:
        if c not in out.columns:
            out[c] = 0
    out = out.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def select_top_by_cap(table: pd.DataFrame, cap_rate: float):
    if table is None or len(table) == 0 or cap_rate <= 0:
        return pd.DataFrame()
    n_select = int(math.floor(len(match_meta) * float(cap_rate)))
    n_select = min(max(n_select, 0), len(table))
    if n_select == 0 and cap_rate > 0 and len(table) > 0:
        n_select = 1
    return table.head(n_select).copy()


def select_top_n(table: pd.DataFrame, n: int):
    if table is None or len(table) == 0 or n <= 0:
        return pd.DataFrame()
    return table.head(min(int(n), len(table))).copy()


def apply_changes_to_base(base_match: pd.DataFrame, changes: pd.DataFrame, label: str):
    out = base_match.copy()
    if changes is None or len(changes) == 0:
        out["label"] = label
        return out
    out_idx = out.set_index("match_id", drop=False)
    for _, r in changes.iterrows():
        mid = r["match_id"]
        if mid in out_idx.index:
            out_idx.loc[mid, "pred_a"] = int(r["new_a"])
            out_idx.loc[mid, "pred_b"] = int(r["new_b"])
    out = out_idx.reset_index(drop=True)
    out["label"] = label
    return out


def save_match_candidate(match_df: pd.DataFrame, label: str, strict=False, changes=None, base_for_changed=None):
    sub = match_to_submission(match_df)
    sub, path, check = save_candidate(sub, label, strict=strict)
    metrics = basic_submission_metrics(sub, label)
    base_for_changed = base_for_changed if base_for_changed is not None else exp12n_anchor_match
    ch = transition_changes_df(base_for_changed, submission_to_match(sub, label=label), label)
    if len(ch):
        ch.to_csv(PRED_DIR / f"changed_{slugify_label(label)}.csv", index=False)
    changed_records.append({
        "label": label,
        "n_changed": int(len(ch)),
        "changed_rate": float(len(ch) / max(len(match_meta), 1)),
        "mean_total_delta_changed": float(ch["total_delta"].mean()) if len(ch) else 0.0,
        "n_high_weight_changed": int((ch["tournament_weight"] >= 1.8).sum()) if len(ch) else 0,
        "n_low_weight_changed": int((ch["tournament_weight"] <= 0.96).sum()) if len(ch) else 0,
        "path": str(path),
    })
    return sub, path, check, metrics, ch


def register_grid_record(records, label, changes, metrics, extra=None):
    extra = extra or {}
    records.append({
        "label": label,
        "n_selected": int(len(changes)) if changes is not None else 0,
        "changed_rate_expected": float((len(changes) if changes is not None else 0) / max(len(match_meta), 1)),
        "mean_pred_total": metrics.get("mean_pred_total", np.nan),
        "max_pred_goal": metrics.get("max_pred_goal", np.nan),
        "pair_consistency": metrics.get("pair_consistency", False),
        **extra,
    })


def make_from_anchor(changes, label, base_match=None):
    base_match = exp12n_anchor_match if base_match is None else base_match
    match_df = apply_changes_to_base(base_match, changes, label)
    return save_match_candidate(match_df, label, strict=False, changes=changes, base_for_changed=base_match)


def mean_total_after_changes(base_match, changes):
    match_df = apply_changes_to_base(base_match, changes, "tmp")
    return float((match_df["pred_a"] + match_df["pred_b"]).mean())

# %% [markdown]
# # 08. Q1 — Ultra-Fine Cap Sweep Around 0.0060
#
# Penjelasan bagian:
# Q1 mencari sweet spot lebih presisi di sekitar cap0060, karena cap0075 mulai memburuk.
#
# Output yang perlu dilihat:
# `q1_ultrafine_cap_sweep_grid.csv` menunjukkan apakah cap0062/0064/0066 lebih baik dari cap0060 secara sanity dan local audit.

# %%
q1_records = []
q1_caps = [0.0052, 0.0054, 0.0056, 0.0058, 0.0060, 0.0062, 0.0064, 0.0066, 0.0068, 0.0070, 0.0072, 0.0075]
for cap in q1_caps:
    changes = select_top_by_cap(same_gd_total_lift_candidate_table, cap)
    suffix = f"cap{int(round(cap * 10000)):04d}" + ("_diagnostic" if cap >= 0.0075 else "")
    label = f"q1_same_gd_{suffix}"
    _, _, _, metrics, _ = make_from_anchor(changes, label)
    mt = metrics.get("mean_pred_total", np.nan)
    register_grid_record(q1_records, label, changes, metrics, {"cap_rate": cap, "family": "ultrafine_same_gd", "mean_total_delta_vs_exp12p": mt - basic_submission_metrics(exp12p_champion_sub, "exp12p_champion")["mean_pred_total"]})
q1_grid = pd.DataFrame(q1_records)
q1_grid.to_csv(SUM_DIR / "q1_ultrafine_cap_sweep_grid.csv", index=False)
q1_grid.to_csv(SUM_DIR / "q1_ultrafine_cap_sweep_changes.csv", index=False)
log_saved(SUM_DIR / "q1_ultrafine_cap_sweep_grid.csv")
safe_display_df(q1_grid)

# %% [markdown]
# # 09. Q2 — Prefix Count Sweep
#
# Penjelasan bagian:
# Q2 memilih jumlah match langsung agar tidak bergantung pada pembulatan cap rate.
#
# Output yang perlu dilihat:
# `q2_prefix_count_sweep_grid.csv` menunjukkan prefix 110–160. Jika prefix tertentu menang, jumlah match itu lebih tepat daripada cap rate.

# %%
q2_records = []
for n in [110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160]:
    changes = select_top_n(same_gd_total_lift_candidate_table, n)
    label = f"q2_prefix_{n}"
    _, _, _, metrics, _ = make_from_anchor(changes, label)
    register_grid_record(q2_records, label, changes, metrics, {"n_prefix": n, "equivalent_cap_rate": float(n / max(len(match_meta), 1))})
q2_grid = pd.DataFrame(q2_records)
q2_grid.to_csv(SUM_DIR / "q2_prefix_count_sweep_grid.csv", index=False)
q2_grid.to_csv(SUM_DIR / "q2_prefix_count_sweep_changes.csv", index=False)
log_saved(SUM_DIR / "q2_prefix_count_sweep_grid.csv")
safe_display_df(q2_grid)

# %% [markdown]
# # 10. Q3 — Band Surgery Between Cap0050, Cap0055, Cap0060, Cap0075
#
# Penjelasan bagian:
# Q3 membedah band cap0050→0055→0060→0075. Tujuannya mengambil tail yang masih berguna dan membuang tail yang toxic.
#
# Output yang perlu dilihat:
# `q3_band_summary.csv`, `q3_band_surgery_grid.csv`, dan changed files menunjukkan transition/support pada band penting.

# %%
def band_slice(table, cap_start, cap_end):
    a = len(select_top_by_cap(table, cap_start))
    b = len(select_top_by_cap(table, cap_end))
    return table.iloc[a:b].copy().reset_index(drop=True)

cap0050 = select_top_by_cap(same_gd_total_lift_candidate_table, 0.0050)
cap0055 = select_top_by_cap(same_gd_total_lift_candidate_table, 0.0055)
cap0060 = select_top_by_cap(same_gd_total_lift_candidate_table, 0.0060)
cap0075 = select_top_by_cap(same_gd_total_lift_candidate_table, 0.0075)
bands = {
    "band_0050_0055": band_slice(same_gd_total_lift_candidate_table, 0.0050, 0.0055),
    "band_0055_0060": band_slice(same_gd_total_lift_candidate_table, 0.0055, 0.0060),
    "band_0060_0075": band_slice(same_gd_total_lift_candidate_table, 0.0060, 0.0075),
}
band_summary_rows = []
for bname, bdf in bands.items():
    band_summary_rows.append({
        "band": bname,
        "n_rows": len(bdf),
        "mean_support": float(bdf["support"].mean()) if len(bdf) else np.nan,
        "mean_weighted_support": float(bdf["weighted_support"].mean()) if len(bdf) else np.nan,
        "n_high_weight": int(bdf["is_high_weight"].sum()) if len(bdf) else 0,
        "top_transition": bdf["transition"].value_counts().index[0] if len(bdf) else "",
    })
q3_band_summary = pd.DataFrame(band_summary_rows)
q3_band_summary.to_csv(SUM_DIR / "q3_band_summary.csv", index=False)
q3_band_tail_candidates = bands["band_0060_0075"].copy()
q3_band_tail_candidates.to_csv(SUM_DIR / "q3_band_tail_candidates.csv", index=False)

q3_records = []
tail = q3_band_tail_candidates
for k in [5, 10, 15, 20]:
    add = tail.head(k).copy() if len(tail) else pd.DataFrame()
    changes = pd.concat([cap0060, add], ignore_index=True).drop_duplicates("match_id", keep="first") if len(add) else cap0060.copy()
    label = f"q3_cap0060_plus_tail_top{k:02d}_from_0075"
    _, _, _, metrics, _ = make_from_anchor(changes, label)
    register_grid_record(q3_records, label, changes, metrics, {"tail_added": len(add), "base_cap": 0.0060})


def q3_remove(base_changes, remove_mask_func, suffix):
    if len(base_changes) == 0:
        kept = pd.DataFrame(); removed = pd.DataFrame()
    else:
        mask = pd.Series(remove_mask_func(base_changes), index=base_changes.index).fillna(False).astype(bool)
        removed = base_changes[mask].copy()
        kept = base_changes[~mask].copy()
    label = f"q3_{suffix}"
    _, _, _, metrics, _ = make_from_anchor(kept, label)
    register_grid_record(q3_records, label, kept, metrics, {"removed_count": len(removed), "base_count": len(base_changes)})
    if len(removed):
        removed.to_csv(PRED_DIR / f"removed_{slugify_label(label)}.csv", index=False)

if len(cap0075):
    q3_remove(cap0075, lambda d: d["rank"] > max(d["rank"].max() - 10, 0), "cap0075_minus_last10")
    q3_remove(cap0075, lambda d: d["rank"] > max(d["rank"].max() - 20, 0), "cap0075_minus_last20")
    q3_remove(cap0075, lambda d: d["rank"] > max(d["rank"].max() - 30, 0), "cap0075_minus_last30")
    q3_remove(cap0075, lambda d: d["support"] <= 2, "cap0075_minus_low_support")
    q3_remove(cap0075, lambda d: d["is_high_weight"].astype(bool) & (d["support"] <= 3), "cap0075_minus_high_weight_low_support")
    noisy = cap0075.tail(max(1, len(cap0075)//3))["transition"].value_counts().index[0]
    q3_remove(cap0075, lambda d, tr=noisy: d["transition"].eq(tr), "cap0075_minus_noisy_transition")

q3_grid = pd.DataFrame(q3_records)
q3_grid.to_csv(SUM_DIR / "q3_band_surgery_grid.csv", index=False)
q3_grid.to_csv(SUM_DIR / "q3_band_removed_changes.csv", index=False)
log_saved(SUM_DIR / "q3_band_surgery_grid.csv")
safe_display_df(q3_band_summary)
safe_display_df(q3_grid)

# %% [markdown]
# # 11. Q4 — Transition-Aware Tail Pruning
#
# Penjelasan bagian:
# Q4 mencari transition pada tail cap0060→0075 yang masih useful atau toxic.
#
# Output yang perlu dilihat:
# `q4_tail_transition_summary.csv` memperlihatkan transition dominan di tail, lalu grid candidate menguji only/minus transition.

# %%
q4_tail = bands["band_0060_0075"].copy()
q4_tail_transition_summary = transition_summary(q4_tail)
q4_tail_transition_summary.to_csv(SUM_DIR / "q4_tail_transition_summary.csv", index=False)
q4_records = []
trs_tail = q4_tail_transition_summary["transition"].tolist() if len(q4_tail_transition_summary) else []
for k in [1, 2, 3]:
    trs = trs_tail[:k]
    add = q4_tail[q4_tail["transition"].isin(trs)].copy() if len(q4_tail) else pd.DataFrame()
    changes = pd.concat([cap0060, add], ignore_index=True).drop_duplicates("match_id", keep="first") if len(add) else cap0060.copy()
    label = f"q4_cap0060_plus_tail_only_top{k}_transition"
    _, _, _, metrics, _ = make_from_anchor(changes, label)
    register_grid_record(q4_records, label, changes, metrics, {"top_k_transition": k, "tail_added": len(add)})
if trs_tail:
    noisy_top = trs_tail[0]
    kept = cap0075[~cap0075["transition"].eq(noisy_top)].copy()
    label = "q4_cap0075_minus_top_noisy_transition"
    _, _, _, metrics, _ = make_from_anchor(kept, label)
    register_grid_record(q4_records, label, kept, metrics, {"removed_transition": noisy_top})
    low_support_tr = q4_tail.groupby("transition")["support"].mean().sort_values().index[0] if len(q4_tail) else noisy_top
    kept = cap0075[~cap0075["transition"].eq(low_support_tr)].copy()
    label = "q4_cap0075_minus_bottom_support_transition"
    _, _, _, metrics, _ = make_from_anchor(kept, label)
    register_grid_record(q4_records, label, kept, metrics, {"removed_transition": low_support_tr})
    high_weight_counts = q4_tail[q4_tail["is_high_weight"].astype(bool)]["transition"].value_counts()
    if len(high_weight_counts):
        tr_hw = high_weight_counts.index[0]
        kept = cap0075[~cap0075["transition"].eq(tr_hw)].copy()
        label = "q4_cap0075_minus_high_weight_transition"
        _, _, _, metrics, _ = make_from_anchor(kept, label)
        register_grid_record(q4_records, label, kept, metrics, {"removed_transition": tr_hw})
for tr, key in [
    ("1-0 -> 2-1", "tail_only_1_0_to_2_1"),
    ("0-1 -> 1-2", "tail_only_0_1_to_1_2"),
    ("1-1 -> 2-2", "tail_only_1_1_to_2_2"),
    ("2-1 -> 3-2", "tail_only_2_1_to_3_2"),
    ("1-2 -> 2-3", "tail_only_1_2_to_2_3"),
]:
    if len(q4_tail) and tr in set(q4_tail["transition"]):
        add = q4_tail[q4_tail["transition"].eq(tr)].copy()
        changes = pd.concat([cap0060, add], ignore_index=True).drop_duplicates("match_id", keep="first")
        label = f"q4_{key}"
        _, _, _, metrics, _ = make_from_anchor(changes, label)
        register_grid_record(q4_records, label, changes, metrics, {"transition": tr, "tail_added": len(add)})
q4_grid = pd.DataFrame(q4_records)
q4_grid.to_csv(SUM_DIR / "q4_transition_tail_pruning_grid.csv", index=False)
q4_grid.to_csv(SUM_DIR / "q4_transition_tail_changes.csv", index=False)
log_saved(SUM_DIR / "q4_transition_tail_pruning_grid.csv")
safe_display_df(q4_tail_transition_summary)
safe_display_df(q4_grid)

# %% [markdown]
# # 12. Q5 — Mean-Total Target Router
#
# Penjelasan bagian:
# Q5 memilih top same-GD changes sampai target mean_pred_total tertentu, bukan berdasarkan cap langsung.
#
# Output yang perlu dilihat:
# `q5_mean_total_target_grid.csv` menunjukkan target dan actual mean total. Ini membantu membaca batas underprediction correction.

# %%
def select_until_mean_total(table, target_mean_total, base_match):
    if table is None or len(table) == 0:
        return pd.DataFrame()
    selected = []
    current = base_match[["match_id", "pred_a", "pred_b"]].copy()
    current_idx = current.set_index("match_id", drop=False)
    for _, r in table.iterrows():
        mid = r["match_id"]
        old_a = int(current_idx.loc[mid, "pred_a"])
        old_b = int(current_idx.loc[mid, "pred_b"])
        next_total_sum = int(current_idx["pred_a"].sum() + current_idx["pred_b"].sum()) - old_a - old_b + int(r["new_a"]) + int(r["new_b"])
        next_mean = next_total_sum / len(current_idx)
        if next_mean <= target_mean_total + 1e-12:
            selected.append(r)
            current_idx.loc[mid, "pred_a"] = int(r["new_a"])
            current_idx.loc[mid, "pred_b"] = int(r["new_b"])
        else:
            break
    return pd.DataFrame(selected).reset_index(drop=True) if selected else pd.DataFrame()

q5_records = []
for target in [2.5385, 2.5390, 2.5395, 2.5400, 2.5405, 2.5410, 2.5415]:
    changes = select_until_mean_total(same_gd_total_lift_candidate_table, target, exp12n_anchor_match)
    label = f"q5_mean_total_target_{int(round(target*10000)):05d}"
    _, _, _, metrics, _ = make_from_anchor(changes, label)
    register_grid_record(q5_records, label, changes, metrics, {"target_mean_total": target, "actual_mean_total": metrics.get("mean_pred_total", np.nan), "equivalent_cap_rate": len(changes)/max(len(match_meta), 1)})
q5_grid = pd.DataFrame(q5_records)
q5_grid.to_csv(SUM_DIR / "q5_mean_total_target_grid.csv", index=False)
q5_grid.to_csv(SUM_DIR / "q5_mean_total_target_changes.csv", index=False)
log_saved(SUM_DIR / "q5_mean_total_target_grid.csv")
safe_display_df(q5_grid)

# %% [markdown]
# # 13. Q6 — Segment-Specific Cap Expansion
#
# Penjelasan bagian:
# Q6 menguji cap lebih besar hanya di segment yang dinilai lebih aman seperti non-high/default/M.
#
# Output yang perlu dilihat:
# `q6_segment_specific_expansion_grid.csv` menunjukkan apakah segment cap lebih besar lebih aman dari global cap.

# %%
q6_specs = [
    ("non_high_cap0065", 0.0065, lambda d: ~d["is_high_weight"].astype(bool)),
    ("non_high_cap0070", 0.0070, lambda d: ~d["is_high_weight"].astype(bool)),
    ("default_cap0065", 0.0065, lambda d: d["is_default_weight"].astype(bool)),
    ("default_cap0070", 0.0070, lambda d: d["is_default_weight"].astype(bool)),
    ("M_only_cap0065", 0.0065, lambda d: d["gender"].astype(str).str.upper().eq("M")),
    ("W_only_cap0030", 0.0030, lambda d: d["gender"].astype(str).str.upper().eq("W")),
    ("friendly_cap0050", 0.0050, lambda d: d["is_friendly"].astype(bool)),
    ("high_weight_support4_cap0020", 0.0020, lambda d: d["is_high_weight"].astype(bool) & (d["support"] >= 4)),
    ("world_asian_support4_cap0015", 0.0015, lambda d: d["is_world_asian"].astype(bool) & (d["support"] >= 4)),
]
q6_records = []
for name, cap, fn in q6_specs:
    tab = filter_table(same_gd_total_lift_candidate_table, fn)
    changes = select_top_by_cap(tab, cap)
    label = f"q6_{name}"
    _, _, _, metrics, _ = make_from_anchor(changes, label)
    register_grid_record(q6_records, label, changes, metrics, {"cap_rate": cap, "segment": name, "n_pool": len(tab)})
q6_grid = pd.DataFrame(q6_records)
q6_grid.to_csv(SUM_DIR / "q6_segment_specific_expansion_grid.csv", index=False)
q6_grid.to_csv(SUM_DIR / "q6_segment_specific_changes.csv", index=False)
log_saved(SUM_DIR / "q6_segment_specific_expansion_grid.csv")
safe_display_df(q6_grid)

# %% [markdown]
# # 14. Q7 — Sharper Anti-Overlift Guard
#
# Penjelasan bagian:
# Q7 mencoba guard yang lebih tajam pada cap0065/0070/0075 untuk membuang candidate high-risk.
#
# Output yang perlu dilihat:
# Jika guarded cap0070/0075 menang di local audit, cap besar bisa digunakan asal high-risk candidate dibuang.

# %%
q7_specs = [
    ("cap0065_no_old_total_ge4", 0.0065, lambda d: d["old_total"] < 4),
    ("cap0065_high_weight_support4", 0.0065, lambda d: (~d["is_high_weight"].astype(bool)) | (d["support"] >= 4)),
    ("cap0065_no_high_weight_low_support", 0.0065, lambda d: ~(d["is_high_weight"].astype(bool) & (d["support"] < 4))),
    ("cap0070_no_old_total_ge4", 0.0070, lambda d: d["old_total"] < 4),
    ("cap0070_high_weight_support4", 0.0070, lambda d: (~d["is_high_weight"].astype(bool)) | (d["support"] >= 4)),
    ("cap0070_no_high_weight_low_support", 0.0070, lambda d: ~(d["is_high_weight"].astype(bool) & (d["support"] < 4))),
    ("cap0075_strict_guard", 0.0075, lambda d: (d["old_total"] < 4) & (d["total_delta"] <= 2) & ((~d["is_high_weight"].astype(bool)) | (d["support"] >= 4)) & (d["source_weight_top"] >= 1.0)),
]
q7_records = []
for name, cap, fn in q7_specs:
    tab = filter_table(same_gd_total_lift_candidate_table, fn)
    changes = select_top_by_cap(tab, cap)
    label = f"q7_{name}"
    _, _, _, metrics, _ = make_from_anchor(changes, label)
    register_grid_record(q7_records, label, changes, metrics, {"cap_rate": cap, "guard": name, "n_pool": len(tab)})
q7_grid = pd.DataFrame(q7_records)
q7_grid.to_csv(SUM_DIR / "q7_sharp_guard_grid.csv", index=False)
q7_grid.to_csv(SUM_DIR / "q7_sharp_guard_changes.csv", index=False)
log_saved(SUM_DIR / "q7_sharp_guard_grid.csv")
safe_display_df(q7_grid)

# %% [markdown]
# # 15. Q8 — Optional Conservative Exact Complement
#
# Penjelasan bagian:
# Q8 mencoba exact-consensus sangat kecil setelah same-GD sweet spot. Ini optional dan tidak boleh agresif.
#
# Output yang perlu dilihat:
# `q8_exact_complement_grid.csv` menunjukkan apakah exact consensus berguna sebagai second-stage complement.

# %%
exact_filter = (
    (all_vote_candidate_table.get("same_outcome", pd.Series(False, index=all_vote_candidate_table.index)).astype(bool))
    & (all_vote_candidate_table.get("support", pd.Series(0, index=all_vote_candidate_table.index)) >= 3)
    & (all_vote_candidate_table.get("abs_total_delta", pd.Series(999, index=all_vote_candidate_table.index)) <= 2)
    & (all_vote_candidate_table.get("abs_gd_delta", pd.Series(999, index=all_vote_candidate_table.index)) <= 1)
) if len(all_vote_candidate_table) else pd.Series([], dtype=bool)
exact_consensus_candidate_table = finalize_table(
    all_vote_candidate_table[exact_filter].copy() if len(all_vote_candidate_table) else pd.DataFrame(),
    ["support", "weighted_support", "source_weight_top", "same_gd", "abs_total_delta", "abs_gd_delta", "is_high_weight", "match_id"],
    [False, False, False, False, True, True, True, True],
)
exact_consensus_candidate_table.to_csv(SUM_DIR / "q8_exact_consensus_candidate_table.csv", index=False)

q8_records = []
for base_cap, exact_cap, support4 in [(0.0060, 0.0005, False), (0.0060, 0.0010, False), (0.0060, 0.0005, True)]:
    base_changes = select_top_by_cap(same_gd_total_lift_candidate_table, base_cap)
    used = set(base_changes["match_id"].tolist()) if len(base_changes) else set()
    tab = exact_consensus_candidate_table[~exact_consensus_candidate_table["match_id"].isin(used)].copy() if len(exact_consensus_candidate_table) else pd.DataFrame()
    if support4 and len(tab):
        tab = tab[tab["support"] >= 4].copy()
    ex = select_top_by_cap(tab, exact_cap)
    combo = pd.concat([base_changes, ex], ignore_index=True) if len(base_changes) or len(ex) else pd.DataFrame()
    label = f"q8_best_same_gd_plus_exact_cap{int(round(exact_cap*10000)):04d}" + ("_support4" if support4 else "")
    _, _, _, metrics, _ = make_from_anchor(combo, label)
    register_grid_record(q8_records, label, combo, metrics, {"same_gd_cap": base_cap, "exact_cap": exact_cap, "n_exact": len(ex), "support4": support4})
q8_grid = pd.DataFrame(q8_records)
q8_grid.to_csv(SUM_DIR / "q8_exact_complement_grid.csv", index=False)
q8_grid.to_csv(SUM_DIR / "q8_exact_complement_changes.csv", index=False)
log_saved(SUM_DIR / "q8_exact_complement_grid.csv")
safe_display_df(q8_grid)

# %% [markdown]
# # 16. Final Selected Safe dan Artifact Summary
#
# Penjelasan bagian:
# Bagian ini memilih `best_safe` secara GT-free berdasarkan urutan konservatif, lalu menyimpan catalog, checks, metrics, changed analysis, dan final decision.
#
# Output yang perlu dilihat:
# `final_decision.csv` harus jelas bahwa `best_safe` bukan klaim best local audit. Champion tetap dibaca dari Section 99.

# %%
# Save key references for audit visibility.
save_candidate(exp39_final_sub, "reference_exp39_final", strict=False)
save_candidate(exp12n_anchor_sub, "reference_exp12n_anchor", strict=False)
if exp12o_reference_sub is not None:
    save_candidate(exp12o_reference_sub, "reference_exp12o_best", strict=False)
if l4_reference_sub is not None:
    save_candidate(l4_reference_sub, "reference_exp12l_l4_cap0025", strict=False)

safe_order = [
    "q0_exp12p_champion_reproduction",
    "q1_same_gd_cap0060",
    "q1_same_gd_cap0062",
    "q2_prefix_135",
    "q5_mean_total_target_25395",
    "reference_exp39_final",
]
cat_df_tmp = pd.DataFrame(submission_catalog)
selected_label, selected_path = None, None
for lab in safe_order:
    hit = cat_df_tmp[cat_df_tmp["label"].eq(lab)] if len(cat_df_tmp) else pd.DataFrame()
    if len(hit):
        selected_label = lab
        selected_path = Path(hit.iloc[0]["path"])
        break
if selected_label is None:
    selected_label = "q0_exp12p_champion_reproduction"
    selected_path = q0_path

best_safe_sub = normalize_submission_df(pd.read_csv(selected_path), require_all=True)
best_safe_path = SUB_DIR / "submission_exp12q_best_safe.csv"
best_safe_sub.to_csv(best_safe_path, index=False)
best_safe_check = validation_checks(best_safe_sub)
best_safe_check.insert(0, "label", "best_safe")
submission_checks_all.append(best_safe_check)
log_saved(best_safe_path)
if not bool(best_safe_check["passed"].all()):
    display(best_safe_check)
    raise RuntimeError("EXP12Q best_safe validation failed")

submission_catalog.append({"label": "best_safe", "path": str(best_safe_path)})
submission_catalog_df = pd.DataFrame(submission_catalog).drop_duplicates(["label", "path"]).reset_index(drop=True)
submission_catalog_df.to_csv(SUM_DIR / "submission_catalog.csv", index=False)

submission_check_df = pd.concat(submission_checks_all, ignore_index=True) if submission_checks_all else pd.DataFrame()
submission_check_df.to_csv(SUM_DIR / "submission_check.csv", index=False)

variant_metrics_df = pd.DataFrame(variant_metrics_records).drop_duplicates(["label", "path"]).reset_index(drop=True)
variant_metrics_df.to_csv(SUM_DIR / "variant_metrics.csv", index=False)

changed_prediction_analysis_df = pd.DataFrame(changed_records).drop_duplicates("label").reset_index(drop=True) if changed_records else pd.DataFrame()
changed_prediction_analysis_df.to_csv(SUM_DIR / "changed_prediction_analysis.csv", index=False)

final_decision = {
    "selected_by_pipeline": selected_label,
    "main_submission_path": str(best_safe_path),
    "selected_source_path": str(selected_path),
    "selection_is_gt_free": True,
    "recommended_for_local_audit": "Audit all EXP12Q candidates in Section 99; best_safe is not a GT-best claim.",
    "notes": "EXP12Q refines EXP12P same-GD total-lift sweet spot with ultra-fine cap, prefix, band surgery, transition pruning, mean-total target, segment expansion, and sharper guard.",
}
final_decision_df = pd.DataFrame([{"key": k, "value": str(v)} for k, v in final_decision.items()])
final_decision_df.to_csv(SUM_DIR / "final_decision.csv", index=False)
with open(SUM_DIR / "final_decision.json", "w") as f:
    json.dump(final_decision, f, indent=2)

log_section("Final Decision")
safe_display_df(final_decision_df)
log_check("best_safe strict validation", bool(best_safe_check["passed"].all()))

# %% [markdown]
# # 99. OPTIONAL LOCAL GT AUDIT — DELETE BEFORE CLEAN SUBMISSION NOTEBOOK
#
# Penjelasan bagian:
# Cell ini membaca `ground_truth_bersih.csv` dan semua submission CSV yang sudah dibuat, lalu menghitung AW-MAE lokal. Cell ini hanya untuk audit lokal setelah semua submission selesai dibuat.
#
# Output yang perlu dilihat:
# Lihat ranking AW-MAE, base_mae, exact_rate, outcome_rate, gd_rate, bias, pair_consistency, dan n_bad_pairs. Jangan gunakan cell ini untuk training, feature engineering, atau tuning otomatis.

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
        audit_log("ground_truth_bersih.csv not found; skipping local GT audit.")
    else:
        audit_log(f"GT_PATH = {GT_PATH}")
        gt = pd.read_csv(GT_PATH)
        id_col = "Id" if "Id" in gt.columns else ("id" if "id" in gt.columns else gt.columns[0])
        gt = gt.rename(columns={id_col: "Id"}).copy()
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

        audit_dirs = [
            SUB_DIR, OUT_DIR / "submissions",
            OUTPUT_ROOT / "exp12q_same_gd_sweet_spot_refinement",
            OUTPUT_ROOT / "exp12p_same_gd_total_lift_expansion",
            OUTPUT_ROOT / "exp12o_same_outcome_consensus_router",
            OUTPUT_ROOT / "exp12n_total_plus_ranked_micro_patch_router",
            OUTPUT_ROOT / "exp12m_exp39_same_outcome_micro_patch_refinement",
            OUTPUT_ROOT / "exp12l_exp39_integration_hybrid_patch",
        ]
        audit_dirs = [Path(d) for d in audit_dirs if Path(d).exists()]
        paths = []
        for d in audit_dirs:
            for p in d.rglob("submission_*.csv"):
                if is_submission_candidate_path(p):
                    paths.append(p.resolve())
        paths = sorted(set(paths), key=lambda x: str(x))
        audit_log(f"Found {len(paths)} submission files")

        rows, bad_rows = [], []
        for path in paths:
            try:
                sub = normalize_submission_df(pd.read_csv(path), require_all=True)
                merged = gt.merge(
                    sub.rename(columns={"team_goals": "pred_team_goals", "opp_goals": "pred_opp_goals"}),
                    on="Id", how="left", validate="one_to_one",
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
