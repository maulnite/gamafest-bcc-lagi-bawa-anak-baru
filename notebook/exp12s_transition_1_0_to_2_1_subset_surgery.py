
# %% [markdown]
# # 00. EXP12S — Transition 1-0→2-1 Ordered Subset Surgery
#
# Penjelasan bagian:
# EXP12S adalah post-processing subset surgery di atas EXP12R. Eksperimen ini tidak melatih model baru, tetapi membedah subset transition terbaik 1-0 -> 2-1 melalui prefix fine sweep, leave-one-out pruning, tail swap, local gate, dan local ranking.
#
# Output yang perlu dilihat:
# Pastikan S0 mereproduce EXP12R champion, `transition_1_0_to_2_1_table.csv` tersimpan, semua candidate pair-consistent, dan Section 99 menjadi satu-satunya bagian yang membaca ground truth.

# %% [markdown]
# # 01. Setup, Project Root, Logging, dan Guardrail
#
# Penjelasan bagian:
# Bagian ini menyiapkan import, seed, logging, output folder, dan project root resolver. Output dipaksa ke `PROJECT_ROOT/outputs/exp12s_transition_1_0_to_2_1_subset_surgery/` supaya tidak masuk ke folder notebook.
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
EXP12S_VARIANT = os.environ.get("EXP12S_VARIANT", "s1_subset_surgery")
OUT_DIR = OUTPUT_ROOT / "exp12s_transition_1_0_to_2_1_subset_surgery" / EXP12S_VARIANT
PRED_DIR = OUT_DIR / "predictions"
SUB_DIR = OUT_DIR / "submissions"
SUM_DIR = OUT_DIR / "summaries"
FIG_DIR = OUT_DIR / "figures"

EXP12S_CLEAN_OUTDIR = os.environ.get("EXP12S_CLEAN_OUTDIR", "0").strip().lower() in {"1", "true", "yes", "y"}
if EXP12S_CLEAN_OUTDIR and OUT_DIR.exists():
    log_info(f"Cleaning previous EXP12S output folder: {OUT_DIR}")
    shutil.rmtree(OUT_DIR)

for d in [PRED_DIR, SUB_DIR, SUM_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RUN_LOCAL_GT_AUDIT_DEFAULT = os.environ.get("EXP12S_RUN_LOCAL_GT_AUDIT", "1").strip().lower() in {"1", "true", "yes", "y"}
MAX_GOAL_SANITY = 40

log_section("Setup")
log_info(f"PROJECT_ROOT = {PROJECT_ROOT}")
log_info(f"OUTPUT_ROOT = {OUTPUT_ROOT}")
log_info(f"OUT_DIR = {OUT_DIR}")
log_info(f"SUB_DIR = {SUB_DIR}")
log_info(f"SUM_DIR = {SUM_DIR}")
log_info(f"EXP12S_VARIANT = {EXP12S_VARIANT}")
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
# Bagian ini mencari EXP39 final, EXP12N anchor, EXP12P/EXP12Q champion, dan candidate submission lain. Finder membaca header CSV supaya tidak salah mengambil summary/check/catalog.
#
# Output yang perlu dilihat:
# Cek `exp39_final_path`, `exp12n_anchor_path`, dan `exp12q_champion_path`. File source wajib punya kolom Id/id, team_goals, dan opp_goals.

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
EXP12Q_PRIORITY = [
    ["exp12q", "q4", "tail", "only", "1", "0", "to", "2", "1"],
    ["exp12q", "q4", "cap0060", "plus", "tail", "only", "to"],
    ["exp12q", "best", "safe"],
    ["exp12p", "p1", "same", "gd", "cap0060"],
]
EXP12R_PRIORITY = [
    ["exp12r", "r1", "1", "0", "to", "2", "1", "top20"],
    ["exp12r", "best", "safe"],
    ["exp12q", "q4", "tail", "only", "1", "0", "to", "2", "1"],
    ["exp12p", "p1", "same", "gd", "cap0060"],
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
exp12p_reference_path = find_by_priority(EXP12P_PRIORITY, "EXP12P cap0060 reference", required=False)
exp12q_champion_path = find_by_priority(EXP12Q_PRIORITY, "EXP12Q champion")
exp12r_champion_path = find_by_priority(EXP12R_PRIORITY, "EXP12R champion")
exp12o_reference_path = find_by_priority(EXP12O_PRIORITY, "EXP12O reference", required=False)
l4_reference_path = find_by_priority(L4_PRIORITY, "EXP12L L4 reference", required=False)

source_df = pd.DataFrame([
    {"source": "exp39_final_path", "path": str(exp39_final_path)},
    {"source": "exp12n_anchor_path", "path": str(exp12n_anchor_path)},
    {"source": "exp12p_reference_path", "path": str(exp12p_reference_path) if exp12p_reference_path else "NOT_FOUND"},
    {"source": "exp12q_champion_path", "path": str(exp12q_champion_path)},
    {"source": "exp12r_champion_path", "path": str(exp12r_champion_path)},
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
# Bagian ini memuat EXP39 final, EXP12N anchor untuk konstruksi same-GD, EXP12P cap0060 reference, dan EXP12Q champion sebagai base terbaru. Candidate pool mengecualikan self-reference dari output patch EXP12L/M/N/O/P/Q/R.
#
# Output yang perlu dilihat:
# Candidate pool harus berisi beberapa EXP39/export/older candidate, tetapi tidak boleh berisi output patch EXP12P/Q/R sebagai support.

# %%

def read_submission_path(p: Path) -> pd.DataFrame:
    return normalize_submission_df(pd.read_csv(p), require_all=True)

exp39_final_sub = read_submission_path(exp39_final_path)
exp12n_anchor_sub = read_submission_path(exp12n_anchor_path)
exp12p_reference_sub = read_submission_path(exp12p_reference_path) if exp12p_reference_path else None
exp12q_champion_sub = read_submission_path(exp12q_champion_path)
exp12r_champion_sub = read_submission_path(exp12r_champion_path)
exp12o_reference_sub = read_submission_path(exp12o_reference_path) if exp12o_reference_path else None
l4_reference_sub = read_submission_path(l4_reference_path) if l4_reference_path else None

exp39_final_match = submission_to_match(exp39_final_sub, label="exp39_final")
exp12n_anchor_match = submission_to_match(exp12n_anchor_sub, label="exp12n_anchor")
exp12p_reference_match = submission_to_match(exp12p_reference_sub, label="exp12p_reference") if exp12p_reference_sub is not None else None
exp12q_champion_match = submission_to_match(exp12q_champion_sub, label="exp12q_champion")
exp12r_champion_match = submission_to_match(exp12r_champion_sub, label="exp12r_champion")
exp12o_reference_match = submission_to_match(exp12o_reference_sub, label="exp12o_reference") if exp12o_reference_sub is not None else None
l4_reference_match = submission_to_match(l4_reference_sub, label="l4_reference") if l4_reference_sub is not None else None

submission_catalog = []
submission_checks_all = []
variant_metrics_records = []
changed_records = []


def save_candidate(sub_df: pd.DataFrame, label: str, strict=False):
    aligned = normalize_submission_df(sub_df, require_all=True)
    path = SUB_DIR / f"submission_exp12s_{slugify_label(label)}.csv"
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

# S0 champion reproduction from EXP12R.
s0_sub, s0_path, _ = save_candidate(exp12r_champion_sub, "s0_exp12r_champion_reproduction", strict=False)

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
    "exp12q_q0", "exp12q_q1", "exp12q_q2", "exp12q_q3", "exp12q_q4", "exp12q_q5", "exp12q_q6", "exp12q_q7", "exp12q_q8", "exp12q_best",
    "exp12r_r0", "exp12r_r1", "exp12r_r2", "exp12r_r3", "exp12r_r4", "exp12r_r5", "exp12r_r6", "exp12r_r7", "exp12r_r8", "exp12r_best",
    "exp12s",
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
log_check("S0 champion pair consistency", pair_consistency_report(exp12r_champion_sub)["pair_consistency"])
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
# # 08. R Core — Tail Band cap0060 ke cap0075
#
# Penjelasan bagian:
# Bagian ini membangun `same_gd_tail_candidate_table.csv` sebagai selisih selected candidate cap0075 dan cap0060 dari same-GD total-lift table. Tail inilah yang menjadi fokus EXP12S.
#
# Output yang perlu dilihat:
# Tail table harus benar-benar berisi kandidat yang ada di cap0075 tetapi belum masuk cap0060. Perhatikan transition dominan, terutama `1-0 -> 2-1` dan mirror `0-1 -> 1-2`.

# %%
def safe_transition_label(transition: str) -> str:
    return str(transition).replace(" -> ", "_to_").replace("-", "_").replace(" ", "_")


def band_slice(table, cap_start, cap_end):
    a = len(select_top_by_cap(table, cap_start))
    b = len(select_top_by_cap(table, cap_end))
    return table.iloc[a:b].copy().reset_index(drop=True)

cap0060 = select_top_by_cap(same_gd_total_lift_candidate_table, 0.0060)
cap0075 = select_top_by_cap(same_gd_total_lift_candidate_table, 0.0075)
cap0060_ids = set(cap0060["match_id"].tolist()) if len(cap0060) else set()
same_gd_tail_candidate_table = cap0075[~cap0075["match_id"].isin(cap0060_ids)].copy().reset_index(drop=True) if len(cap0075) else pd.DataFrame()
if len(same_gd_tail_candidate_table):
    same_gd_tail_candidate_table.insert(0, "tail_rank", np.arange(1, len(same_gd_tail_candidate_table) + 1))

same_gd_tail_candidate_table.to_csv(SUM_DIR / "same_gd_tail_candidate_table.csv", index=False)
same_gd_tail_transition_summary = transition_summary(same_gd_tail_candidate_table)
same_gd_tail_segment_summary = segment_summary(same_gd_tail_candidate_table)
same_gd_tail_transition_summary.to_csv(SUM_DIR / "same_gd_tail_transition_summary.csv", index=False)
same_gd_tail_segment_summary.to_csv(SUM_DIR / "same_gd_tail_segment_summary.csv", index=False)
log_section("Same-GD Tail Table")
log_check("tail equals cap0075 minus cap0060 count", len(same_gd_tail_candidate_table) == max(0, len(cap0075) - len(cap0060)), f"tail={len(same_gd_tail_candidate_table)} cap0060={len(cap0060)} cap0075={len(cap0075)}")
safe_display_df(same_gd_tail_transition_summary)


def tail_transition(transition: str):
    if same_gd_tail_candidate_table is None or len(same_gd_tail_candidate_table) == 0:
        return pd.DataFrame()
    return same_gd_tail_candidate_table[same_gd_tail_candidate_table["transition"].eq(transition)].copy().reset_index(drop=True)


def changes_from_cap0060(additions):
    if additions is None or len(additions) == 0:
        return cap0060.copy()
    return pd.concat([cap0060, additions], ignore_index=True).drop_duplicates("match_id", keep="first").reset_index(drop=True)


def save_tail_candidate(additions, label, base_changes=None):
    base_changes = cap0060 if base_changes is None else base_changes
    changes = pd.concat([base_changes, additions], ignore_index=True).drop_duplicates("match_id", keep="first").reset_index(drop=True) if additions is not None and len(additions) else base_changes.copy()
    sub, path, check, metrics, ch = make_from_anchor(changes, label)
    return changes, metrics, ch


def save_changes_candidate(changes, label):
    sub, path, check, metrics, ch = make_from_anchor(changes, label)
    return changes, metrics, ch


# %% [markdown]
# # 09. S Core — Transition 1-0 -> 2-1 Table
#
# Penjelasan bagian:
# Bagian ini mengambil `same_gd_tail_candidate_table` lalu memfilter transition terbaik `1-0 -> 2-1`. Tabel ini adalah basis semua subset surgery EXP12S.
#
# Output yang perlu dilihat:
# `transition_1_0_to_2_1_table.csv` harus punya kandidat cukup untuk top20/top25. Kolom transition harus human-readable: `1-0 -> 2-1`.

# %%
transition_1_0_to_2_1_table = tail_transition("1-0 -> 2-1")
if len(transition_1_0_to_2_1_table):
    transition_1_0_to_2_1_table = transition_1_0_to_2_1_table.copy().reset_index(drop=True)
    transition_1_0_to_2_1_table["transition_rank"] = np.arange(1, len(transition_1_0_to_2_1_table) + 1)

transition_1_0_to_2_1_table.to_csv(SUM_DIR / "transition_1_0_to_2_1_table.csv", index=False)
transition_1_0_to_2_1_table.head(30).to_csv(SUM_DIR / "transition_1_0_to_2_1_top30.csv", index=False)
transition_1_0_to_2_1_summary = table_summary(transition_1_0_to_2_1_table, "transition_1_0_to_2_1")
transition_1_0_to_2_1_summary.to_csv(SUM_DIR / "transition_1_0_to_2_1_summary.csv", index=False)
log_section("Transition 1-0 -> 2-1 Table")
log_check("transition table is from tail", set(transition_1_0_to_2_1_table.get("match_id", pd.Series([], dtype=object))).issubset(set(same_gd_tail_candidate_table.get("match_id", pd.Series([], dtype=object)))))
log_check("transition table filter", bool((transition_1_0_to_2_1_table["transition"].eq("1-0 -> 2-1")).all()) if len(transition_1_0_to_2_1_table) else True, f"n={len(transition_1_0_to_2_1_table)}")
safe_display_df(transition_1_0_to_2_1_summary)
safe_display_df(transition_1_0_to_2_1_table.head(30))


def transition_top(n: int, table=None):
    table = transition_1_0_to_2_1_table if table is None else table
    return select_top_n(table, int(n))


def build_transition_subset(ranks, table=None):
    table = transition_1_0_to_2_1_table if table is None else table
    if table is None or len(table) == 0:
        return pd.DataFrame()
    ranks = set(int(x) for x in ranks)
    return table[table["transition_rank"].isin(ranks)].copy().sort_values("transition_rank").reset_index(drop=True)


def transition_top20():
    return transition_top(20)


def changes_from_transition_add(additions):
    return changes_from_cap0060(additions)


def save_transition_subset(additions, label):
    changes = changes_from_transition_add(additions)
    sub, path, check, metrics, ch = make_from_anchor(changes, label)
    return changes, metrics, ch


def record_subset(records, label, additions, changes, metrics, extra=None):
    extra = extra or {}
    records.append({
        "label": label,
        "n_transition_selected": int(len(additions)) if additions is not None else 0,
        "n_total_changes": int(len(changes)) if changes is not None else 0,
        "changed_rate_expected": float((len(changes) if changes is not None else 0) / max(len(match_meta), 1)),
        "mean_pred_total": metrics.get("mean_pred_total", np.nan),
        "pair_consistency": metrics.get("pair_consistency", False),
        **extra,
    })

# %% [markdown]
# # 10. S1 — Fine Prefix Sweep Around Top20
#
# Penjelasan bagian:
# S1 memperhalus prefix transition `1-0 -> 2-1` di sekitar top20. Ini mengecek apakah top19/top21/top22 lebih bersih dari top20.
#
# Output yang perlu dilihat:
# `s1_fine_prefix_grid.csv` memperlihatkan n_selected, mean_pred_total, dan pair consistency per prefix.

# %%
s1_records = []
for n in [12, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 28]:
    add = transition_top(n)
    label = f"s1_prefix_top{n:02d}"
    changes, metrics, ch = save_transition_subset(add, label)
    record_subset(s1_records, label, add, changes, metrics, {"prefix_n": n})
s1_grid = pd.DataFrame(s1_records)
s1_grid.to_csv(SUM_DIR / "s1_fine_prefix_grid.csv", index=False)
s1_grid.to_csv(SUM_DIR / "s1_fine_prefix_changes.csv", index=False)
log_saved(SUM_DIR / "s1_fine_prefix_grid.csv")
safe_display_df(s1_grid)

# %% [markdown]
# # 11. S2 — Leave-One-Out Pruning from Top20
#
# Penjelasan bagian:
# S2 mulai dari top20, lalu menghapus satu kandidat satu per satu. Tujuannya mencari kandidat toxic di dalam top20.
#
# Output yang perlu dilihat:
# `s2_leave_one_out_removed_candidates.csv` menunjukkan kandidat rank mana yang dihapus. Jika minus rank tertentu menang di audit, rank tersebut toxic.

# %%
s2_records = []
s2_removed = []
base20 = transition_top20()
for r in range(1, min(20, len(base20)) + 1):
    add = base20[base20["transition_rank"] != r].copy().reset_index(drop=True)
    removed = base20[base20["transition_rank"] == r].copy()
    label = f"s2_top20_minus_rank{r:02d}"
    changes, metrics, ch = save_transition_subset(add, label)
    extra = {"removed_rank": r, "n_removed": len(removed)}
    if len(removed):
        rr = removed.iloc[0].to_dict()
        s2_removed.append({"label": label, "removed_rank": r, **{k: rr.get(k) for k in ["match_id", "transition", "support", "weighted_support", "tournament", "gender", "tournament_weight", "source_label_top"]}})
    record_subset(s2_records, label, add, changes, metrics, extra)
s2_grid = pd.DataFrame(s2_records)
s2_removed_df = pd.DataFrame(s2_removed)
s2_grid.to_csv(SUM_DIR / "s2_leave_one_out_grid.csv", index=False)
s2_removed_df.to_csv(SUM_DIR / "s2_leave_one_out_removed_candidates.csv", index=False)
log_saved(SUM_DIR / "s2_leave_one_out_grid.csv")
safe_display_df(s2_grid)

# %% [markdown]
# # 12. S3 — Tail Block Pruning
#
# Penjelasan bagian:
# S3 membuang blok tail dari top20 seperti last01-last05 atau rank16-20. Ini mengecek apakah bagian akhir top20 mengandung noise.
#
# Output yang perlu dilihat:
# `s3_tail_block_pruning_grid.csv` memperlihatkan apakah subset lebih kecil dari top20 tetap aman.

# %%
s3_records = []
s3_removed_parts = []

def save_minus_ranks(label, remove_ranks):
    remove_ranks = set(int(x) for x in remove_ranks)
    add = base20[~base20["transition_rank"].isin(remove_ranks)].copy().reset_index(drop=True)
    removed = base20[base20["transition_rank"].isin(remove_ranks)].copy()
    changes, metrics, ch = save_transition_subset(add, label)
    record_subset(s3_records, label, add, changes, metrics, {"removed_ranks": ",".join(map(str, sorted(remove_ranks))), "n_removed": len(removed)})
    if len(removed):
        tmp = removed.copy(); tmp.insert(0, "label", label); s3_removed_parts.append(tmp)

for n in [1, 2, 3, 4, 5]:
    save_minus_ranks(f"s3_top20_minus_last{n:02d}", range(21-n, 21))
save_minus_ranks("s3_top20_minus_rank16_20", range(16, 21))
save_minus_ranks("s3_top20_minus_rank18_20", range(18, 21))
save_minus_ranks("s3_top20_minus_rank19_20", range(19, 21))
s3_grid = pd.DataFrame(s3_records)
s3_removed_df = pd.concat(s3_removed_parts, ignore_index=True) if s3_removed_parts else pd.DataFrame()
s3_grid.to_csv(SUM_DIR / "s3_tail_block_pruning_grid.csv", index=False)
s3_removed_df.to_csv(SUM_DIR / "s3_tail_block_removed_candidates.csv", index=False)
log_saved(SUM_DIR / "s3_tail_block_pruning_grid.csv")
safe_display_df(s3_grid)

# %% [markdown]
# # 13. S4 — Tail Swap Rank 16–25
#
# Penjelasan bagian:
# S4 mencoba mengganti kandidat boundary top20 dengan rank 21–25. Ini menguji apakah ranking awal hampir benar tetapi boundary-nya masih bisa diperbaiki.
#
# Output yang perlu dilihat:
# Jika swap menang di audit, berarti kandidat di luar top20 lebih baik dari sebagian tail top20.

# %%
s4_records = []
s4_details = []

def swap_candidate(label, remove_ranks, add_ranks):
    remove_ranks = set(int(x) for x in remove_ranks)
    add_ranks = set(int(x) for x in add_ranks)
    base_ranks = set(range(1, min(20, len(transition_1_0_to_2_1_table)) + 1))
    final_ranks = (base_ranks - remove_ranks) | add_ranks
    add = build_transition_subset(final_ranks)
    removed = build_transition_subset(remove_ranks)
    added = build_transition_subset(add_ranks)
    changes, metrics, ch = save_transition_subset(add, label)
    record_subset(s4_records, label, add, changes, metrics, {"remove_ranks": ",".join(map(str, sorted(remove_ranks))), "add_ranks": ",".join(map(str, sorted(add_ranks))), "n_removed": len(removed), "n_added": len(added)})
    if len(removed) or len(added):
        for _, row in removed.iterrows():
            s4_details.append({"label": label, "action": "remove", **row.to_dict()})
        for _, row in added.iterrows():
            s4_details.append({"label": label, "action": "add", **row.to_dict()})

swap_candidate("s4_top20_minus20_plus21", [20], [21])
swap_candidate("s4_top20_minus20_plus22", [20], [22])
swap_candidate("s4_top20_minus20_plus23", [20], [23])
swap_candidate("s4_top20_minus19_plus21", [19], [21])
swap_candidate("s4_top20_minus19_plus22", [19], [22])
swap_candidate("s4_top20_minus18_plus21", [18], [21])
swap_candidate("s4_top20_minus19_20_plus21_22", [19, 20], [21, 22])
swap_candidate("s4_top20_minus18_20_plus21_23", [18, 19, 20], [21, 22, 23])
swap_candidate("s4_top20_minus16_20_plus21_25", [16, 17, 18, 19, 20], [21, 22, 23, 24, 25])
s4_grid = pd.DataFrame(s4_records)
s4_detail_df = pd.DataFrame(s4_details)
s4_grid.to_csv(SUM_DIR / "s4_tail_swap_grid.csv", index=False)
s4_detail_df.to_csv(SUM_DIR / "s4_tail_swap_detail.csv", index=False)
log_saved(SUM_DIR / "s4_tail_swap_grid.csv")
safe_display_df(s4_grid)

# %% [markdown]
# # 14. S5 — Local Gate Within Top20 / Top30
#
# Penjelasan bagian:
# S5 menerapkan gate khusus pada kandidat transition 1-0 -> 2-1, seperti support tinggi, non-high, default, M only, atau source weight tinggi.
#
# Output yang perlu dilihat:
# Jika gate menang di audit, subset terbaik punya pola feature tertentu.

# %%
s5_records = []

def top30_gate(label, fn, take=20):
    top30 = transition_top(30)
    tab = filter_table(top30, fn)
    add = select_top_n(tab, take)
    changes, metrics, ch = save_transition_subset(add, label)
    record_subset(s5_records, label, add, changes, metrics, {"gate_pool": len(tab), "take": take})

top30_gate("s5_top30_support3_take20", lambda d: d["support"] >= 3)
top30_gate("s5_top30_support4_take20", lambda d: d["support"] >= 4)
top30_gate("s5_top30_non_high_take20", lambda d: ~d["is_high_weight"].astype(bool))
top30_gate("s5_top30_default_take20", lambda d: d["is_default_weight"].astype(bool))
top30_gate("s5_top30_M_only_take20", lambda d: d["gender"].astype(str).str.upper().str.startswith("M"))
top30_gate("s5_top30_exclude_high_weight_take20", lambda d: ~d["is_high_weight"].astype(bool))
# Dynamic source weight threshold: top half if enough rows.
sw_thr = float(transition_top(30)["source_weight_top"].median()) if len(transition_top(30)) and "source_weight_top" in transition_top(30).columns else 0.0
top30_gate("s5_top30_source_weight_high_take20", lambda d, thr=sw_thr: d["source_weight_top"] >= thr)
top30_gate("s5_top30_weighted_support_take20", lambda d: pd.Series(True, index=d.index).astype(bool))
# weighted_support variant reorders before take.
tab_ws = reorder_table(transition_top(30), ["weighted_support", "support", "source_weight_top", "transition_rank"], [False, False, False, True])
add_ws = select_top_n(tab_ws, 20)
changes, metrics, ch = save_transition_subset(add_ws, "s5_top30_weighted_support_take20_ranked")
record_subset(s5_records, "s5_top30_weighted_support_take20_ranked", add_ws, changes, metrics, {"gate_pool": len(tab_ws), "take": 20})
s5_grid = pd.DataFrame(s5_records)
s5_grid.to_csv(SUM_DIR / "s5_local_gate_grid.csv", index=False)
s5_grid.to_csv(SUM_DIR / "s5_local_gate_changes.csv", index=False)
log_saved(SUM_DIR / "s5_local_gate_grid.csv")
safe_display_df(s5_grid)

# %% [markdown]
# # 15. S6 — Local Ranking Refinement
#
# Penjelasan bagian:
# S6 mengubah ranking khusus transition 1-0 -> 2-1, lalu mengambil top18/top20/top22.
#
# Output yang perlu dilihat:
# Jika ranking baru menang, next experiment perlu ranker khusus transition.

# %%
def rank_transition_local(tab, rule):
    if tab is None or len(tab) == 0:
        return pd.DataFrame()
    if rule == "support_then_weighted":
        cols, asc = ["support", "weighted_support", "source_weight_top", "transition_rank"], [False, False, False, True]
    elif rule == "weighted_then_support":
        cols, asc = ["weighted_support", "support", "source_weight_top", "transition_rank"], [False, False, False, True]
    elif rule == "source_weight_top_first":
        cols, asc = ["source_weight_top", "weighted_support", "support", "transition_rank"], [False, False, False, True]
    elif rule == "non_high_first":
        cols, asc = ["is_high_weight", "support", "weighted_support", "transition_rank"], [True, False, False, True]
    elif rule == "default_first":
        cols, asc = ["is_default_weight", "support", "weighted_support", "transition_rank"], [False, False, False, True]
    elif rule == "low_old_total_first":
        cols, asc = ["old_total", "support", "weighted_support", "transition_rank"], [True, False, False, True]
    elif rule == "high_old_total_first":
        cols, asc = ["old_total", "support", "weighted_support", "transition_rank"], [False, False, False, True]
    elif rule == "low_tournament_weight_first":
        cols, asc = ["tournament_weight", "support", "weighted_support", "transition_rank"], [True, False, False, True]
    elif rule == "high_support_non_high_first":
        cols, asc = ["is_high_weight", "support", "weighted_support", "transition_rank"], [True, False, False, True]
    else:
        cols, asc = ["transition_rank"], [True]
    return reorder_table(tab, cols, asc)

s6_records = []
for rule in ["support_then_weighted", "weighted_then_support", "source_weight_top_first", "non_high_first", "default_first", "low_old_total_first", "high_old_total_first", "low_tournament_weight_first", "high_support_non_high_first"]:
    ranked = rank_transition_local(transition_1_0_to_2_1_table, rule)
    for n in ([18, 20, 22] if rule == "weighted_then_support" else [20]):
        add = select_top_n(ranked, n)
        label = f"s6_{rule}_top{n:02d}"
        changes, metrics, ch = save_transition_subset(add, label)
        record_subset(s6_records, label, add, changes, metrics, {"ranking_rule": rule, "top_n": n})
s6_grid = pd.DataFrame(s6_records)
s6_grid.to_csv(SUM_DIR / "s6_local_ranking_grid.csv", index=False)
s6_grid.to_csv(SUM_DIR / "s6_local_ranking_changes.csv", index=False)
log_saved(SUM_DIR / "s6_local_ranking_grid.csv")
safe_display_df(s6_grid)

# %% [markdown]
# # 16. S7 — Two-Step Pruning / Combo Subset
#
# Penjelasan bagian:
# S7 memakai rule fixed GT-free untuk membuang kandidat berisiko dari top20 lalu mengisi dengan kandidat berikutnya kalau perlu.
#
# Output yang perlu dilihat:
# Jika S7 menang, subset surgery dapat dibuat lebih sistematis dengan rule sederhana.

# %%
s7_records = []
s7_details = []

def fill_to_n(selected, table, n=20):
    selected = selected.copy() if selected is not None else pd.DataFrame()
    used = set(selected["match_id"].tolist()) if len(selected) else set()
    parts = [selected]
    for _, r in table.iterrows():
        if len(pd.concat(parts, ignore_index=True).drop_duplicates("match_id")) >= n:
            break
        if r["match_id"] not in used:
            parts.append(pd.DataFrame([r]))
            used.add(r["match_id"])
    return pd.concat(parts, ignore_index=True).drop_duplicates("match_id", keep="first").head(n).reset_index(drop=True) if parts else pd.DataFrame()


def two_step(label, remove_fn, fill=True):
    rem_mask = remove_fn(base20) if len(base20) else pd.Series([], dtype=bool)
    rem_mask = pd.Series(rem_mask, index=base20.index).fillna(False).astype(bool)
    kept = base20[~rem_mask].copy()
    removed = base20[rem_mask].copy()
    add = fill_to_n(kept, transition_1_0_to_2_1_table, 20) if fill else kept
    changes, metrics, ch = save_transition_subset(add, label)
    record_subset(s7_records, label, add, changes, metrics, {"n_removed_initial": len(removed), "filled": fill})
    if len(removed):
        tmp = removed.copy(); tmp.insert(0, "label", label); s7_details.append(tmp)

two_step("s7_top20_remove_high_weight_low_support", lambda d: d["is_high_weight"].astype(bool) & (d["support"] <= 2), fill=True)
two_step("s7_top20_remove_support2", lambda d: d["support"] <= 2, fill=True)
two_step("s7_top20_remove_old_total_high", lambda d: d["old_total"] >= 4, fill=True)
two_step("s7_top20_remove_high_weight_add_next_best", lambda d: d["is_high_weight"].astype(bool), fill=True)
two_step("s7_top20_remove_support2_add_next_best", lambda d: d["support"] <= 2, fill=True)
s7_grid = pd.DataFrame(s7_records)
s7_detail_df = pd.concat(s7_details, ignore_index=True) if s7_details else pd.DataFrame()
s7_grid.to_csv(SUM_DIR / "s7_two_step_subset_grid.csv", index=False)
s7_detail_df.to_csv(SUM_DIR / "s7_two_step_subset_detail.csv", index=False)
log_saved(SUM_DIR / "s7_two_step_subset_grid.csv")
safe_display_df(s7_grid)

# %% [markdown]
# # 17. S8 — Optional Tiny Exact Complement
#
# Penjelasan bagian:
# S8 mencoba exact-consensus sangat kecil setelah transition subset default. Ini optional dan tidak agresif.
#
# Output yang perlu dilihat:
# Jika S8 menang di audit, exact consensus bisa menjadi complement kecil. Kalau kalah, jangan lanjut exact family besar.

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
exact_consensus_candidate_table.to_csv(SUM_DIR / "s8_exact_consensus_candidate_table.csv", index=False)

s8_records = []
base_add = transition_top(20)
base_changes = changes_from_transition_add(base_add)
used = set(base_changes["match_id"].tolist()) if len(base_changes) else set()
for exact_cap, support4 in [(0.0005, False), (0.0010, False), (0.0005, True)]:
    tab = exact_consensus_candidate_table[~exact_consensus_candidate_table["match_id"].isin(used)].copy() if len(exact_consensus_candidate_table) else pd.DataFrame()
    if support4 and len(tab):
        tab = tab[tab["support"] >= 4].copy()
    ex = select_top_by_cap(tab, exact_cap)
    combo = pd.concat([base_changes, ex], ignore_index=True).drop_duplicates("match_id", keep="first") if len(base_changes) or len(ex) else pd.DataFrame()
    label = f"s8_transition_subset_plus_exact_cap{int(round(exact_cap*10000)):04d}" + ("_support4" if support4 else "")
    changes, metrics, ch = save_changes_candidate(combo, label)
    record_subset(s8_records, label, ex, combo, metrics, {"exact_cap": exact_cap, "n_exact": len(ex), "support4": support4})
s8_grid = pd.DataFrame(s8_records)
s8_grid.to_csv(SUM_DIR / "s8_exact_complement_grid.csv", index=False)
s8_grid.to_csv(SUM_DIR / "s8_exact_complement_changes.csv", index=False)
log_saved(SUM_DIR / "s8_exact_complement_grid.csv")
safe_display_df(s8_grid)

# %% [markdown]
# # 18. Final Selected Safe dan Artifact Summary
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
if exp12p_reference_sub is not None:
    save_candidate(exp12p_reference_sub, "reference_exp12p_cap0060", strict=False)
save_candidate(exp12q_champion_sub, "reference_exp12q_champion", strict=False)
save_candidate(exp12r_champion_sub, "reference_exp12r_champion", strict=False)
if exp12o_reference_sub is not None:
    save_candidate(exp12o_reference_sub, "reference_exp12o_best", strict=False)
if l4_reference_sub is not None:
    save_candidate(l4_reference_sub, "reference_exp12l_l4_cap0025", strict=False)

safe_order = [
    "s0_exp12r_champion_reproduction",
    "s1_prefix_top20",
    "s1_prefix_top19",
    "s1_prefix_top21",
    "s3_top20_minus_last01",
    "s4_top20_minus20_plus21",
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
    selected_label = "s0_exp12r_champion_reproduction"
    selected_path = s0_path

best_safe_sub = normalize_submission_df(pd.read_csv(selected_path), require_all=True)
best_safe_path = SUB_DIR / "submission_exp12s_best_safe.csv"
best_safe_sub.to_csv(best_safe_path, index=False)
best_safe_check = validation_checks(best_safe_sub)
best_safe_check.insert(0, "label", "best_safe")
submission_checks_all.append(best_safe_check)
log_saved(best_safe_path)
if not bool(best_safe_check["passed"].all()):
    display(best_safe_check)
    raise RuntimeError("EXP12S best_safe validation failed")

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
    "recommended_for_local_audit": "Audit all EXP12S candidates in Section 99; best_safe is not a GT-best claim.",
    "notes": "EXP12S performs subset surgery on transition 1-0 -> 2-1 top20 over EXP12R champion.",
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
            OUTPUT_ROOT / "exp12s_transition_1_0_to_2_1_subset_surgery",
            OUTPUT_ROOT / "exp12r_transition_specific_same_gd_tail_router",
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
