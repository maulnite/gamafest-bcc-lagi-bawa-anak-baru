
# %% [markdown]
# # 00. EXP12O — Same-Outcome Consensus Router
#
# Penjelasan bagian:
# EXP12O adalah post-processing consensus router di atas EXP12N. Eksperimen ini tidak melatih model baru, tetapi memuat candidate submission yang sudah ada, membangun tabel consensus, lalu membuat patch kecil di atas anchor EXP12N.
#
# Output yang perlu dilihat:
# Pastikan O0 mereproduce EXP12N champion, `same_outcome_same_gd_candidate_table.csv` dan `exact_consensus_candidate_table.csv` tersimpan, semua candidate pair-consistent, dan Section 99 menjadi satu-satunya bagian yang membaca ground truth.

# %% [markdown]
# # 01. Setup, Project Root, Logging, dan Guardrail
#
# Penjelasan bagian:
# Bagian ini menyiapkan import, seed, logging, output folder, dan project root resolver. Output dipaksa ke `PROJECT_ROOT/outputs/exp12o_same_outcome_consensus_router/` supaya tidak masuk ke folder notebook.
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
EXP12O_VARIANT = os.environ.get("EXP12O_VARIANT", "o1_same_gd_consensus_router")
OUT_DIR = OUTPUT_ROOT / "exp12o_same_outcome_consensus_router" / EXP12O_VARIANT
PRED_DIR = OUT_DIR / "predictions"
SUB_DIR = OUT_DIR / "submissions"
SUM_DIR = OUT_DIR / "summaries"
FIG_DIR = OUT_DIR / "figures"

EXP12O_CLEAN_OUTDIR = os.environ.get("EXP12O_CLEAN_OUTDIR", "0").strip().lower() in {"1", "true", "yes", "y"}
if EXP12O_CLEAN_OUTDIR and OUT_DIR.exists():
    log_info(f"Cleaning previous EXP12O output folder: {OUT_DIR}")
    shutil.rmtree(OUT_DIR)

for d in [PRED_DIR, SUB_DIR, SUM_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RUN_LOCAL_GT_AUDIT_DEFAULT = os.environ.get("EXP12O_RUN_LOCAL_GT_AUDIT", "1").strip().lower() in {"1", "true", "yes", "y"}
MAX_GOAL_SANITY = 40

log_section("Setup")
log_info(f"PROJECT_ROOT = {PROJECT_ROOT}")
log_info(f"OUTPUT_ROOT = {OUTPUT_ROOT}")
log_info(f"OUT_DIR = {OUT_DIR}")
log_info(f"SUB_DIR = {SUB_DIR}")
log_info(f"SUM_DIR = {SUM_DIR}")
log_info(f"EXP12O_VARIANT = {EXP12O_VARIANT}")
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
# Bagian ini mencari EXP39 final, EXP12N champion, serta candidate submission lain. Finder membaca header CSV supaya tidak salah mengambil summary/check/catalog.
#
# Output yang perlu dilihat:
# Cek `exp39_final_path`, `exp12n_champion_path`, dan candidate inventory. File source wajib punya kolom Id/id, team_goals, dan opp_goals.

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
    out = []
    seen = set()
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
L4_PRIORITY = [
    ["exp12l", "l4", "gd", "patch", "cap0025"],
    ["exp12l", "l4", "same", "outcome", "exact", "cap0025"],
    ["exp12l", "best", "safe"],
]

exp39_final_path = find_by_priority(EXP39_PRIORITY, "EXP39 final")
exp12n_champion_path = find_by_priority(EXP12N_PRIORITY, "EXP12N champion")
l4_reference_path = find_by_priority(L4_PRIORITY, "EXP12L L4 reference", required=False)

source_df = pd.DataFrame([
    {"source": "exp39_final_path", "path": str(exp39_final_path)},
    {"source": "exp12n_champion_path", "path": str(exp12n_champion_path)},
    {"source": "l4_reference_path", "path": str(l4_reference_path) if l4_reference_path else "NOT_FOUND"},
])
source_df.to_csv(SUM_DIR / "source_paths.csv", index=False)
log_section("Source Paths")
safe_display_df(source_df)

# %% [markdown]
# # 05. Load Anchor Submissions dan Candidate Pool
#
# Penjelasan bagian:
# Bagian ini memuat EXP39 final sebagai reference, EXP12N champion sebagai anchor, dan candidate pool lain sebagai sumber consensus. Candidate table akan mengecualikan self-reference patch family EXP12L/EXP12M/EXP12N/EXP12O.
#
# Output yang perlu dilihat:
# Candidate pool harus berisi beberapa EXP39/export/older candidate. `pair_consistency` untuk anchor wajib True.

# %%

def read_submission_path(p: Path) -> pd.DataFrame:
    return normalize_submission_df(pd.read_csv(p), require_all=True)

exp39_final_sub = read_submission_path(exp39_final_path)
exp12n_anchor_sub = read_submission_path(exp12n_champion_path)
l4_reference_sub = read_submission_path(l4_reference_path) if l4_reference_path else None

exp39_final_match = submission_to_match(exp39_final_sub, label="exp39_final")
exp12n_anchor_match = submission_to_match(exp12n_anchor_sub, label="exp12n_anchor")
l4_reference_match = submission_to_match(l4_reference_sub, label="l4_reference") if l4_reference_sub is not None else None

submission_catalog = []
submission_checks_all = []
variant_metrics_records = []
changed_records = []


def save_candidate(sub_df: pd.DataFrame, label: str, strict=False):
    aligned = normalize_submission_df(sub_df, require_all=True)
    path = SUB_DIR / f"submission_exp12o_{slugify_label(label)}.csv"
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

# O0 anchor reproduction.
o0_sub, o0_path, _ = save_candidate(exp12n_anchor_sub, "o0_exp12n_champion_reproduction", strict=False)

candidate_paths = all_submission_candidates()
raw_inventory = []
for p in candidate_paths:
    label = label_from_path(p)
    raw_inventory.append({"label": label, "path": str(p)})
raw_inventory_df = pd.DataFrame(raw_inventory)
raw_inventory_df.to_csv(SUM_DIR / "raw_submission_inventory.csv", index=False)

SELF_REFERENCE_FRAGMENTS = [
    "exp12l_l4",
    "l4_cap0025",
    "l4_reference",
    "l4_reference_for_delta_only",
    "exp12m_m0",
    "exp12m_m1",
    "exp12m_m2",
    "exp12m_m3",
    "exp12m_best",
    "exp12n_n0",
    "exp12n_n1",
    "exp12n_n2",
    "exp12n_n3",
    "exp12n_n4",
    "exp12n_n5",
    "exp12n_n6",
    "exp12n_n7",
    "exp12n_n8",
    "exp12n_best",
    "exp12o",
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

# Explicit useful references can be added as comparison artifacts, but anchor-like self outputs stay excluded from support.
candidate_match_pool["exp39_final_base"] = exp39_final_match.copy()

# PATCH EXP12O-01:
# Do NOT add the L4 reference into the support/voting pool.
# L4 is only a reference/audit artifact and is saved later via save_candidate(...),
# so it must not strengthen T2/T3/T4 consensus candidates.
# Keeping it out of candidate_match_pool prevents hidden self-reference leakage.

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
log_check("O0 anchor pair consistency", pair_consistency_report(exp12n_anchor_sub)["pair_consistency"])

# %% [markdown]
# # 06. Candidate Table Builders — T2/T3/T4
#
# Penjelasan bagian:
# Bagian ini membangun tiga tabel consensus: T2 same-outcome same-GD total lift, T3 exact-consensus tiny patch, dan T4 same-outcome low-distance. Semua tabel bebas self-reference dan unique by `match_id`.
#
# Output yang perlu dilihat:
# Lihat jumlah kandidat, support distribution, transition summary, dan segment summary. Jika tabel kosong, variant terkait otomatis tidak mengubah submission.

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
            if lab not in candidate_match_pool:
                continue
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

all_vote_candidate_table = build_vote_records(exp12n_anchor_match)
if len(all_vote_candidate_table):
    # PATCH EXP12O-02:
    # Do NOT drop duplicates here. One match can have multiple alternative scorelines.
    # T2/T3/T4 must filter their own candidate families first, then deduplicate in finalize_table().
    # If we deduplicate before family filtering, a same-GD or exact-consensus candidate can be
    # accidentally removed just because another alternative for the same match ranks higher globally.
    all_vote_candidate_table = all_vote_candidate_table.sort_values(
        [
            "support",
            "weighted_support",
            "source_weight_top",
            "is_default_weight",
            "is_high_weight",
            "abs_total_delta",
            "abs_gd_delta",
            "match_id",
        ],
        ascending=[False, False, False, False, True, True, True, True],
    ).reset_index(drop=True)
    all_vote_candidate_table.insert(0, "vote_rank", np.arange(1, len(all_vote_candidate_table) + 1))
all_vote_candidate_table.to_csv(SUM_DIR / "all_vote_candidate_table.csv", index=False)


def finalize_table(df: pd.DataFrame, sort_cols, asc):
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=[
            "rank", "match_id", "old_a", "old_b", "new_a", "new_b", "old_score", "new_score", "transition",
            "support", "weighted_support", "source_labels", "source_label_top", "source_weight_top", "gender", "tournament",
            "tournament_weight", "old_total", "new_total", "total_delta", "old_gd", "new_gd", "gd_delta", "same_outcome", "same_gd",
            "is_high_weight", "is_low_weight", "is_default_weight", "is_world_asian", "is_friendly"
        ])
    out = df.copy().sort_values(sort_cols, ascending=asc).drop_duplicates("match_id", keep="first").reset_index(drop=True)
    if "rank" in out.columns:
        out = out.drop(columns=["rank"])
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out

same_gd_df = all_vote_candidate_table.copy()
if len(same_gd_df):
    same_gd_df = same_gd_df[
        (same_gd_df["same_outcome"].astype(bool))
        & (same_gd_df["same_gd"].astype(bool))
        & (same_gd_df["total_delta"] > 0)
        & (same_gd_df["support"] >= 2)
    ].copy()
same_outcome_same_gd_candidate_table = finalize_table(
    same_gd_df,
    ["support", "weighted_support", "source_weight_top", "is_high_weight", "is_default_weight", "abs_total_delta", "match_id"],
    [False, False, False, True, False, True, True],
)
same_outcome_same_gd_candidate_table.to_csv(SUM_DIR / "same_outcome_same_gd_candidate_table.csv", index=False)

exact_df = all_vote_candidate_table.copy()
if len(exact_df):
    exact_df = exact_df[
        (exact_df["same_outcome"].astype(bool))
        & (exact_df["abs_total_delta"] <= 2)
        & (exact_df["abs_gd_delta"] <= 2)
        & (exact_df["support"] >= 3)
    ].copy()
exact_consensus_candidate_table = finalize_table(
    exact_df,
    ["support", "weighted_support", "source_weight_top", "same_gd", "abs_total_delta", "abs_gd_delta", "is_high_weight", "match_id"],
    [False, False, False, False, True, True, True, True],
)
exact_consensus_candidate_table.to_csv(SUM_DIR / "exact_consensus_candidate_table.csv", index=False)

low_df = all_vote_candidate_table.copy()
if len(low_df):
    low_df = low_df[
        (low_df["same_outcome"].astype(bool))
        & (low_df["abs_total_delta"] <= 2)
        & (low_df["abs_gd_delta"] <= 1)
        & (low_df["support"] >= 3)
    ].copy()
same_outcome_low_distance_candidate_table = finalize_table(
    low_df,
    ["support", "weighted_support", "source_weight_top", "is_high_weight", "is_default_weight", "abs_total_delta", "abs_gd_delta", "match_id"],
    [False, False, False, True, False, True, True, True],
)
same_outcome_low_distance_candidate_table.to_csv(SUM_DIR / "same_outcome_low_distance_candidate_table.csv", index=False)


def table_summary(table, name):
    if table is None or len(table) == 0:
        return pd.DataFrame([{"table": name, "metric": "n_candidates", "value": 0}])
    rows = [
        {"table": name, "metric": "n_candidates", "value": len(table)},
        {"table": name, "metric": "n_match_unique", "value": table["match_id"].nunique()},
        {"table": name, "metric": "support_mean", "value": float(table["support"].mean())},
        {"table": name, "metric": "weighted_support_mean", "value": float(table["weighted_support"].mean())},
        {"table": name, "metric": "default_weight_share", "value": float(table["is_default_weight"].mean())},
        {"table": name, "metric": "non_high_weight_share", "value": float((~table["is_high_weight"].astype(bool)).mean())},
        {"table": name, "metric": "mean_total_delta", "value": float(table["total_delta"].mean())},
    ]
    return pd.DataFrame(rows)

same_gd_candidate_summary = table_summary(same_outcome_same_gd_candidate_table, "same_gd")
exact_consensus_summary = table_summary(exact_consensus_candidate_table, "exact_consensus")
low_distance_summary = table_summary(same_outcome_low_distance_candidate_table, "low_distance")
same_gd_candidate_summary.to_csv(SUM_DIR / "same_gd_candidate_summary.csv", index=False)
exact_consensus_summary.to_csv(SUM_DIR / "exact_consensus_summary.csv", index=False)
low_distance_summary.to_csv(SUM_DIR / "low_distance_summary.csv", index=False)


def transition_summary(table):
    if table is None or len(table) == 0:
        return pd.DataFrame()
    return table.groupby("transition").agg(
        n_candidates=("match_id", "count"),
        mean_support=("support", "mean"),
        mean_weighted_support=("weighted_support", "mean"),
        share_default_weight=("is_default_weight", "mean"),
        share_non_high_weight=("is_high_weight", lambda s: float((~s.astype(bool)).mean())),
    ).reset_index().sort_values(["n_candidates", "mean_weighted_support"], ascending=[False, False])


def segment_summary(table):
    if table is None or len(table) == 0:
        return pd.DataFrame()
    return table.groupby(["gender", "is_default_weight", "is_high_weight", "is_low_weight"]).agg(
        n_candidates=("match_id", "count"),
        mean_support=("support", "mean"),
        mean_weighted_support=("weighted_support", "mean"),
        mean_total_delta=("total_delta", "mean"),
    ).reset_index().sort_values("n_candidates", ascending=False)

same_gd_transition_summary = transition_summary(same_outcome_same_gd_candidate_table)
same_gd_segment_summary = segment_summary(same_outcome_same_gd_candidate_table)
exact_consensus_transition_summary = transition_summary(exact_consensus_candidate_table)
exact_consensus_segment_summary = segment_summary(exact_consensus_candidate_table)
same_gd_transition_summary.to_csv(SUM_DIR / "same_gd_transition_summary.csv", index=False)
same_gd_segment_summary.to_csv(SUM_DIR / "same_gd_segment_summary.csv", index=False)
exact_consensus_transition_summary.to_csv(SUM_DIR / "exact_consensus_transition_summary.csv", index=False)
exact_consensus_segment_summary.to_csv(SUM_DIR / "exact_consensus_segment_summary.csv", index=False)

log_section("Candidate Table Summary")
safe_display_df(pd.concat([same_gd_candidate_summary, exact_consensus_summary, low_distance_summary], ignore_index=True))
safe_display_df(same_gd_transition_summary.head(20) if len(same_gd_transition_summary) else same_gd_transition_summary)

# %% [markdown]
# # 07. Patch Application Helper dan Ranking Router
#
# Penjelasan bagian:
# Bagian ini menyiapkan helper untuk memilih top-N berdasarkan cap, menerapkan perubahan ke anchor EXP12N, menyimpan submission, dan mencatat metrics/changed analysis.
#
# Output yang perlu dilihat:
# Helper ini dipakai semua variant O1-O7. Semua final candidate harus pair-consistent.

# %%

def select_top_by_cap(table: pd.DataFrame, cap_rate: float, base_n_matches=None):
    if table is None or len(table) == 0:
        return pd.DataFrame(columns=[])
    if base_n_matches is None:
        base_n_matches = len(match_meta)
    n = max(int(math.floor(float(cap_rate) * base_n_matches)), 0)
    return table.head(min(n, len(table))).copy()


def apply_changes_to_anchor(changes: pd.DataFrame, label: str, base_match=None):
    if base_match is None:
        base_match = exp12n_anchor_match
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
    cand_match = submission_to_match(sub, label)
    if changes is None:
        changes = transition_changes_df(exp12n_anchor_match, cand_match, label)
    changes_path = PRED_DIR / f"changes_{slugify_label(label)}.csv"
    if changes is not None:
        changes.to_csv(changes_path, index=False)
    metrics["n_changed_vs_anchor"] = int(len(changes)) if changes is not None else 0
    metrics["changed_rate_vs_anchor"] = float(len(changes) / max(len(match_meta), 1)) if changes is not None else 0.0
    if changes is not None and len(changes):
        metrics["mean_total_delta_vs_anchor"] = float(changes["total_delta"].mean()) if "total_delta" in changes.columns else np.nan
        metrics["changed_high_weight"] = int((changes["tournament_weight"] >= 1.8).sum()) if "tournament_weight" in changes.columns else 0
        metrics["changed_low_weight"] = int((changes["tournament_weight"] <= 0.96).sum()) if "tournament_weight" in changes.columns else 0
        metrics["changed_M"] = int(changes["gender"].astype(str).str.upper().eq("M").sum()) if "gender" in changes.columns else 0
        metrics["changed_W"] = int(changes["gender"].astype(str).str.upper().eq("W").sum()) if "gender" in changes.columns else 0
    else:
        metrics["mean_total_delta_vs_anchor"] = 0.0
        metrics["changed_high_weight"] = metrics["changed_low_weight"] = metrics["changed_M"] = metrics["changed_W"] = 0
    variant_metrics_records.append(metrics)
    changed_records.append({
        "label": label,
        "n_changed": metrics["n_changed_vs_anchor"],
        "changed_rate": metrics["changed_rate_vs_anchor"],
        "changes_path": str(changes_path),
    })
    return sub, path, check, metrics, changes


def register_grid_record(records, label, changes, metrics, extra=None):
    rec = {"label": label}
    if extra:
        rec.update(extra)
    rec.update({
        "n_selected": int(len(changes)) if changes is not None else 0,
        "changed_rate": float(len(changes) / max(len(match_meta), 1)) if changes is not None else 0.0,
        "mean_pred_total": metrics.get("mean_pred_total", np.nan),
        "mean_total_delta_vs_anchor": metrics.get("mean_total_delta_vs_anchor", 0.0),
        "changed_high_weight": metrics.get("changed_high_weight", 0),
        "changed_low_weight": metrics.get("changed_low_weight", 0),
        "changed_M": metrics.get("changed_M", 0),
        "changed_W": metrics.get("changed_W", 0),
        "pair_consistency": metrics.get("pair_consistency", False),
        "path": metrics.get("path", ""),
    })
    records.append(rec)

# %% [markdown]
# # 08. O1 — Anchor + Same-GD Total-Lift Consensus Patch
#
# Penjelasan bagian:
# O1 memakai EXP12N champion sebagai anchor, lalu menambahkan T2 same-outcome same-GD total lift dengan cap sangat kecil.
#
# Output yang perlu dilihat:
# Lihat `o1_same_gd_patch_grid.csv`: cap, jumlah perubahan, mean total, dan pair consistency.

# %%
o1_records = []
for cap in [0.0005, 0.0010, 0.0015, 0.0020, 0.0025, 0.0030]:
    changes = select_top_by_cap(same_outcome_same_gd_candidate_table, cap)
    label = f"o1_anchor_plus_same_gd_cap{int(round(cap * 10000)):04d}"
    match_df = apply_changes_to_anchor(changes, label)
    _, _, _, metrics, changes_actual = save_match_candidate(match_df, label, strict=False, changes=changes)
    register_grid_record(o1_records, label, changes, metrics, {"cap_rate": cap, "family": "same_gd"})
o1_grid = pd.DataFrame(o1_records)
o1_grid.to_csv(SUM_DIR / "o1_same_gd_patch_grid.csv", index=False)
log_saved(SUM_DIR / "o1_same_gd_patch_grid.csv")
safe_display_df(o1_grid)

# %% [markdown]
# # 09. O2 — Anchor + Exact-Consensus Tiny Patch
#
# Penjelasan bagian:
# O2 menambahkan T3 exact-consensus patch di atas anchor EXP12N. Patch ini memakai source yang sepakat pada scoreline alternatif.
#
# Output yang perlu dilihat:
# Lihat `o2_exact_consensus_grid.csv`. Patch bagus harus changed_rate kecil dan pair_consistency True.

# %%
o2_records = []
for cap in [0.0005, 0.0010, 0.0015, 0.0020]:
    changes = select_top_by_cap(exact_consensus_candidate_table, cap)
    label = f"o2_anchor_plus_exact_consensus_cap{int(round(cap * 10000)):04d}"
    match_df = apply_changes_to_anchor(changes, label)
    _, _, _, metrics, _ = save_match_candidate(match_df, label, strict=False, changes=changes)
    register_grid_record(o2_records, label, changes, metrics, {"cap_rate": cap, "family": "exact_consensus"})

for suffix, table in [
    ("exact_consensus_support3_cap0010", exact_consensus_candidate_table[exact_consensus_candidate_table["support"] >= 3].copy() if len(exact_consensus_candidate_table) else exact_consensus_candidate_table),
    ("exact_consensus_support4_cap0010", exact_consensus_candidate_table[exact_consensus_candidate_table["support"] >= 4].copy() if len(exact_consensus_candidate_table) else exact_consensus_candidate_table),
    ("exact_consensus_weighted_support_cap0010", exact_consensus_candidate_table[exact_consensus_candidate_table["weighted_support"] >= exact_consensus_candidate_table["weighted_support"].median()].copy() if len(exact_consensus_candidate_table) else exact_consensus_candidate_table),
]:
    changes = select_top_by_cap(table, 0.0010)
    label = f"o2_{suffix}"
    match_df = apply_changes_to_anchor(changes, label)
    _, _, _, metrics, _ = save_match_candidate(match_df, label, strict=False, changes=changes)
    register_grid_record(o2_records, label, changes, metrics, {"cap_rate": 0.0010, "family": suffix})

o2_grid = pd.DataFrame(o2_records)
o2_grid.to_csv(SUM_DIR / "o2_exact_consensus_grid.csv", index=False)
log_saved(SUM_DIR / "o2_exact_consensus_grid.csv")
safe_display_df(o2_grid)

# %% [markdown]
# # 10. O3 — Anchor + T2 + T3 Combo
#
# Penjelasan bagian:
# O3 mencoba gabungan kecil: T2 same-GD total lift lalu T3 exact consensus pada match yang belum berubah. Reverse order juga disimpan sebagai diagnostic.
#
# Output yang perlu dilihat:
# `o3_combo_grid.csv` harus menunjukkan n_changed_t2, n_changed_t3, overlap skipped, mean total, dan pair consistency.

# %%
def combo_changes(t2_cap, t3_cap, reverse=False):
    t2 = select_top_by_cap(same_outcome_same_gd_candidate_table, t2_cap)
    t3 = select_top_by_cap(exact_consensus_candidate_table, t3_cap)
    if reverse:
        first, second = t3.copy(), t2.copy()
    else:
        first, second = t2.copy(), t3.copy()
    first_ids = set(first["match_id"].tolist()) if len(first) else set()
    second_kept = second[~second["match_id"].isin(first_ids)].copy() if len(second) else second
    out = pd.concat([first, second_kept], ignore_index=True) if len(first) or len(second_kept) else pd.DataFrame()
    return out, len(first), len(second_kept), max(len(second) - len(second_kept), 0)

o3_specs = [
    ("same_gd0010_then_exact0005", 0.0010, 0.0005, False),
    ("same_gd0010_then_exact0010", 0.0010, 0.0010, False),
    ("same_gd0015_then_exact0005", 0.0015, 0.0005, False),
    ("same_gd0020_then_exact0005", 0.0020, 0.0005, False),
    ("exact0010_then_same_gd0010", 0.0010, 0.0010, True),
]
o3_records = []
for name, t2_cap, t3_cap, reverse in o3_specs:
    changes, n_first, n_second, n_overlap = combo_changes(t2_cap, t3_cap, reverse=reverse)
    label = f"o3_{name}"
    match_df = apply_changes_to_anchor(changes, label)
    _, _, _, metrics, _ = save_match_candidate(match_df, label, strict=False, changes=changes)
    register_grid_record(o3_records, label, changes, metrics, {"t2_cap": t2_cap, "t3_cap": t3_cap, "reverse": reverse, "n_changed_first": n_first, "n_changed_second": n_second, "n_overlap_skipped": n_overlap})
o3_grid = pd.DataFrame(o3_records)
o3_grid.to_csv(SUM_DIR / "o3_combo_grid.csv", index=False)
log_saved(SUM_DIR / "o3_combo_grid.csv")
safe_display_df(o3_grid)

# %% [markdown]
# # 11. O4 — Segment-Safe Same-GD Consensus Router
#
# Penjelasan bagian:
# O4 menguji apakah same-GD total lift lebih aman di segmen tertentu seperti default/non-high, M/W, friendly, atau high-weight dengan support kuat.
#
# Output yang perlu dilihat:
# `o4_segment_safe_same_gd_grid.csv` menunjukkan segment_name, n_changed, mean total, dan pair consistency.

# %%
def filter_table(table, mask):
    if table is None or len(table) == 0:
        return table
    return table[mask].copy()

o4_specs = [
    ("same_gd_default_only_cap0015", 0.0015, lambda d: d["is_default_weight"].astype(bool)),
    ("same_gd_non_high_only_cap0015", 0.0015, lambda d: ~d["is_high_weight"].astype(bool)),
    ("same_gd_friendly_only_cap0010", 0.0010, lambda d: d["is_friendly"].astype(bool)),
    ("same_gd_M_only_cap0015", 0.0015, lambda d: d["gender"].astype(str).str.upper().eq("M")),
    ("same_gd_W_only_cap0010", 0.0010, lambda d: d["gender"].astype(str).str.upper().eq("W")),
    ("same_gd_high_weight_support4_cap0010", 0.0010, lambda d: d["is_high_weight"].astype(bool) & (d["support"] >= 4)),
    ("same_gd_world_asian_support4_cap0005", 0.0005, lambda d: d["is_world_asian"].astype(bool) & (d["support"] >= 4)),
]
o4_records = []
for name, cap, fn in o4_specs:
    tab = filter_table(same_outcome_same_gd_candidate_table, fn(same_outcome_same_gd_candidate_table)) if len(same_outcome_same_gd_candidate_table) else same_outcome_same_gd_candidate_table
    changes = select_top_by_cap(tab, cap)
    label = f"o4_{name}"
    match_df = apply_changes_to_anchor(changes, label)
    _, _, _, metrics, _ = save_match_candidate(match_df, label, strict=False, changes=changes)
    register_grid_record(o4_records, label, changes, metrics, {"cap_rate": cap, "segment_name": name, "n_pool": len(tab) if tab is not None else 0})
o4_grid = pd.DataFrame(o4_records)
o4_grid.to_csv(SUM_DIR / "o4_segment_safe_same_gd_grid.csv", index=False)
log_saved(SUM_DIR / "o4_segment_safe_same_gd_grid.csv")
safe_display_df(o4_grid)

# %% [markdown]
# # 12. O5 — Transition-Confidence Consensus Router
#
# Penjelasan bagian:
# O5 membuat transition confidence dari T2/T3, lalu mencoba patch hanya pada top transition atau mengecualikan transition noisy.
#
# Output yang perlu dilihat:
# Lihat `o5_transition_confidence_table.csv` dan `o5_transition_confidence_grid.csv`. Jika top transition only menang, eksperimen berikutnya bisa fokus ke transition router.

# %%
t2_conf = same_gd_transition_summary.copy()
if len(t2_conf):
    t2_conf["family"] = "same_gd"
t3_conf = exact_consensus_transition_summary.copy()
if len(t3_conf):
    t3_conf["family"] = "exact_consensus"
o5_transition_confidence_table = pd.concat([t2_conf, t3_conf], ignore_index=True) if len(t2_conf) or len(t3_conf) else pd.DataFrame()
o5_transition_confidence_table.to_csv(SUM_DIR / "o5_transition_confidence_table.csv", index=False)


def top_transitions(summary_df, k):
    if summary_df is None or len(summary_df) == 0:
        return []
    return summary_df.head(k)["transition"].tolist()

o5_records = []

def o5_make(table, label_suffix, cap):
    changes = select_top_by_cap(table, cap)
    label = f"o5_{label_suffix}_cap{int(round(cap*10000)):04d}"
    match_df = apply_changes_to_anchor(changes, label)
    _, _, _, metrics, _ = save_match_candidate(match_df, label, strict=False, changes=changes)
    register_grid_record(o5_records, label, changes, metrics, {"cap_rate": cap, "router": label_suffix, "n_pool": len(table) if table is not None else 0})

if len(same_outcome_same_gd_candidate_table):
    for k in [1, 2, 3]:
        trs = top_transitions(same_gd_transition_summary, k)
        o5_make(same_outcome_same_gd_candidate_table[same_outcome_same_gd_candidate_table["transition"].isin(trs)].copy(), f"same_gd_only_top{k}_transition", 0.0015)
    if len(same_gd_transition_summary):
        top_noisy = same_gd_transition_summary.iloc[0]["transition"]
        o5_make(same_outcome_same_gd_candidate_table[~same_outcome_same_gd_candidate_table["transition"].eq(top_noisy)].copy(), "same_gd_exclude_top_noisy_transition", 0.0015)
    for tr, key in [("1-0 -> 2-1", "only_1_0_to_2_1"), ("0-1 -> 1-2", "only_0_1_to_1_2"), ("1-1 -> 2-2", "only_1_1_to_2_2"), ("2-1 -> 3-2", "only_2_1_to_3_2")]:
        if tr in set(same_outcome_same_gd_candidate_table["transition"]):
            o5_make(same_outcome_same_gd_candidate_table[same_outcome_same_gd_candidate_table["transition"].eq(tr)].copy(), key, 0.0010)
if len(exact_consensus_candidate_table):
    for k in [1, 2]:
        trs = top_transitions(exact_consensus_transition_summary, k)
        o5_make(exact_consensus_candidate_table[exact_consensus_candidate_table["transition"].isin(trs)].copy(), f"exact_only_top{k}_transition", 0.0010)

o5_grid = pd.DataFrame(o5_records)
o5_grid.to_csv(SUM_DIR / "o5_transition_confidence_grid.csv", index=False)
log_saved(SUM_DIR / "o5_transition_confidence_grid.csv")
safe_display_df(o5_grid)

# %% [markdown]
# # 13. O6 — Same-Outcome Low-Distance Router
#
# Penjelasan bagian:
# O6 membuat patch lebih umum tetapi tetap low-distance: same outcome, distance kecil, support kuat, dan non-high/default preferred.
#
# Output yang perlu dilihat:
# `o6_low_distance_grid.csv` membantu membaca apakah low-distance consensus lebih baik daripada same-GD strict.

# %%
o6_specs = [
    ("low_distance_cap0005", 0.0005, lambda d: d),
    ("low_distance_cap0010", 0.0010, lambda d: d),
    ("low_distance_cap0015", 0.0015, lambda d: d),
    ("low_distance_non_high_cap0010", 0.0010, lambda d: ~d["is_high_weight"].astype(bool)),
    ("low_distance_default_cap0010", 0.0010, lambda d: d["is_default_weight"].astype(bool)),
    ("low_distance_support4_cap0010", 0.0010, lambda d: d["support"] >= 4),
]
o6_records = []
for name, cap, fn in o6_specs:
    tab = filter_table(same_outcome_low_distance_candidate_table, fn(same_outcome_low_distance_candidate_table)) if len(same_outcome_low_distance_candidate_table) else same_outcome_low_distance_candidate_table
    changes = select_top_by_cap(tab, cap)
    label = f"o6_{name}"
    match_df = apply_changes_to_anchor(changes, label)
    _, _, _, metrics, _ = save_match_candidate(match_df, label, strict=False, changes=changes)
    register_grid_record(o6_records, label, changes, metrics, {"cap_rate": cap, "router": name, "n_pool": len(tab) if tab is not None else 0})
o6_grid = pd.DataFrame(o6_records)
o6_grid.to_csv(SUM_DIR / "o6_low_distance_grid.csv", index=False)
log_saved(SUM_DIR / "o6_low_distance_grid.csv")
safe_display_df(o6_grid)

# %% [markdown]
# # 14. O7 — Toxic-Tail Surgery on O1/O3
#
# Penjelasan bagian:
# O7 mencoba cap lebih besar dari O1/O3 lalu membuang tail/segmen berisiko seperti high-weight, low-support, atau transition dominan/noisy.
#
# Output yang perlu dilihat:
# `o7_toxic_tail_surgery_grid.csv` menunjukkan removed_count dan sanity. Jika menang di local audit, patch lebih besar mungkin berguna dengan pruning.

# %%
o7_records = []

def o7_make(base_changes, remove_mask_func, label_suffix):
    if base_changes is None or len(base_changes) == 0:
        kept = base_changes
        removed = pd.DataFrame()
    else:
        mask = remove_mask_func(base_changes)
        removed = base_changes[mask].copy()
        kept = base_changes[~mask].copy()
    label = f"o7_{label_suffix}"
    match_df = apply_changes_to_anchor(kept, label)
    _, _, _, metrics, _ = save_match_candidate(match_df, label, strict=False, changes=kept)
    register_grid_record(o7_records, label, kept, metrics, {"removed_count": len(removed), "base_count": len(base_changes) if base_changes is not None else 0})

same_gd_cap0030 = select_top_by_cap(same_outcome_same_gd_candidate_table, 0.0030)
if len(same_gd_cap0030):
    o7_make(same_gd_cap0030, lambda d: d["rank"] > max(d["rank"].max() - 5, 0), "same_gd_cap0030_minus_last_05")
    o7_make(same_gd_cap0030, lambda d: d["rank"] > max(d["rank"].max() - 10, 0), "same_gd_cap0030_minus_last_10")
    o7_make(same_gd_cap0030, lambda d: d["is_high_weight"].astype(bool), "same_gd_cap0030_minus_high_weight")
    o7_make(same_gd_cap0030, lambda d: d["support"] <= 2, "same_gd_cap0030_minus_low_support")
    noisy = same_gd_cap0030.tail(max(1, len(same_gd_cap0030)//2))["transition"].value_counts().index[0]
    o7_make(same_gd_cap0030, lambda d, tr=noisy: d["transition"].eq(tr), "same_gd_cap0030_minus_noisy_transition")

combo_base, _, _, _ = combo_changes(0.0020, 0.0005, reverse=False)
if len(combo_base):
    o7_make(combo_base, lambda d: d["rank"] > max(d["rank"].max() - 5, 0) if "rank" in d.columns else pd.Series(False, index=d.index), "combo_minus_last_05")
    o7_make(combo_base, lambda d: d["is_high_weight"].astype(bool), "combo_minus_high_weight")

o7_grid = pd.DataFrame(o7_records)
o7_grid.to_csv(SUM_DIR / "o7_toxic_tail_surgery_grid.csv", index=False)
log_saved(SUM_DIR / "o7_toxic_tail_surgery_grid.csv")
safe_display_df(o7_grid)

# %% [markdown]
# # 15. Final Selected Safe dan Artifact Summary
#
# Penjelasan bagian:
# Bagian ini memilih `best_safe` secara GT-free berdasarkan urutan konservatif, lalu menyimpan catalog, checks, metrics, changed analysis, dan final decision.
#
# Output yang perlu dilihat:
# `final_decision.csv` harus jelas bahwa `best_safe` bukan klaim best local audit. Champion tetap dibaca dari Section 99.

# %%
# Save key references for audit visibility.
save_candidate(exp39_final_sub, "reference_exp39_final", strict=False)
if l4_reference_sub is not None:
    save_candidate(l4_reference_sub, "reference_exp12l_l4_cap0025", strict=False)

safe_order = [
    "o0_exp12n_champion_reproduction",
    "o1_anchor_plus_same_gd_cap0010",
    "o2_anchor_plus_exact_consensus_cap0010",
    "o3_same_gd0010_then_exact0005",
    "reference_exp39_final",
]
cat_df_tmp = pd.DataFrame(submission_catalog)
selected_label = None
selected_path = None
for lab in safe_order:
    hit = cat_df_tmp[cat_df_tmp["label"].eq(lab)] if len(cat_df_tmp) else pd.DataFrame()
    if len(hit):
        selected_label = lab
        selected_path = Path(hit.iloc[0]["path"])
        break
if selected_label is None:
    selected_label = "o0_exp12n_champion_reproduction"
    selected_path = o0_path

best_safe_sub = normalize_submission_df(pd.read_csv(selected_path), require_all=True)
best_safe_path = SUB_DIR / "submission_exp12o_best_safe.csv"
best_safe_sub.to_csv(best_safe_path, index=False)
best_safe_check = validation_checks(best_safe_sub)
best_safe_check.insert(0, "label", "best_safe")
submission_checks_all.append(best_safe_check)
log_saved(best_safe_path)
if not bool(best_safe_check["passed"].all()):
    display(best_safe_check)
    raise RuntimeError("EXP12O best_safe validation failed")

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
    "recommended_for_local_audit": "Audit all EXP12O candidates in Section 99; best_safe is not a GT-best claim.",
    "notes": "EXP12O uses EXP12N anchor, same-GD total-lift, exact-consensus, low-distance, and small combo routers only.",
}
final_decision_df = pd.DataFrame([{"key": k, "value": str(v)} for k, v in final_decision.items()])
final_decision_df.to_csv(SUM_DIR / "final_decision.csv", index=False)
with open(SUM_DIR / "final_decision.json", "w") as f:
    json.dump(final_decision, f, indent=2)

log_section("Final Decision")
safe_display_df(final_decision_df)
safe_display_df(variant_metrics_df.sort_values(["pair_consistency", "changed_rate_vs_anchor"], ascending=[False, True]).head(30) if len(variant_metrics_df) else variant_metrics_df)
log_saved(SUM_DIR / "submission_catalog.csv")
log_saved(SUM_DIR / "submission_check.csv")
log_saved(SUM_DIR / "variant_metrics.csv")
log_saved(SUM_DIR / "final_decision.csv")

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

        audit_dirs = [SUB_DIR, OUT_DIR / "submissions", OUTPUT_ROOT / "exp12o_same_outcome_consensus_router", OUTPUT_ROOT / "exp12n_total_plus_ranked_micro_patch_router", OUTPUT_ROOT / "exp12m_exp39_same_outcome_micro_patch_refinement", OUTPUT_ROOT / "exp12l_exp39_integration_hybrid_patch"]
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
