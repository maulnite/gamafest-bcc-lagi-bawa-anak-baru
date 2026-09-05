# %% [markdown]
# # 00. EXP12F — Ringkasan Eksperimen
#
# Penjelasan bagian:
# EXP12F adalah eksperimen post-processing di atas EXP12E. Fokusnya bukan training model besar baru, tetapi membuat kandidat submission melalui weighted-aware pair router, candidate consensus router, micro outcome booster, micro exact booster, dan micro bias alignment.
#
# Output yang perlu dilihat:
# Cek folder output `PROJECT_ROOT/outputs/exp12f_weighted_pair_router_micro_booster/<variant>/`. Kandidat penting akan tersimpan di `submissions/`, metrik sanity di `summaries/`, dan local GT audit ada di cell paling akhir.

# %% [markdown]
# # 01. Setup, Project Root, Logging, dan Guardrail
#
# Penjelasan bagian:
# Bagian ini menyiapkan import, seed, logging, project root resolver, dan output folder. Project root resolver penting supaya output tidak masuk ke `notebook/outputs/`.
#
# Output yang perlu dilihat:
# Pastikan log `PROJECT_ROOT`, `OUTPUT_ROOT`, `OUT_DIR`, `SUB_DIR`, dan `SUM_DIR` mengarah ke folder project utama, bukan folder `notebook`.

# %%
import os
import json
import math
import random
import warnings
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

try:
    from IPython.display import display
except Exception:
    def display(x):
        print(x)

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 200)
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


def infer_project_root() -> Path:
    cwd = Path.cwd().resolve()
    if cwd.name.lower() in {"notebook", "notebooks"}:
        return cwd.parent
    for candidate in [cwd, cwd.parent]:
        if (candidate / "data").exists() or (candidate / "dataset").exists():
            return candidate
    # fallback untuk environment ChatGPT /mnt/data
    if Path("/mnt/data").exists() and (Path("/mnt/data") / "test.csv").exists():
        return Path("/mnt/data")
    return cwd


PROJECT_ROOT = infer_project_root()
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
EXP12F_VARIANT = os.environ.get("EXP12F_VARIANT", "f1_weighted_pair_router")
OUT_DIR = OUTPUT_ROOT / "exp12f_weighted_pair_router_micro_booster" / EXP12F_VARIANT
PRED_DIR = OUT_DIR / "predictions"
SUB_DIR = OUT_DIR / "submissions"
SUM_DIR = OUT_DIR / "summaries"
FIG_DIR = OUT_DIR / "figures"
for d in [PRED_DIR, SUB_DIR, SUM_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RUN_LOCAL_GT_AUDIT_DEFAULT = os.environ.get("EXP12F_RUN_LOCAL_GT_AUDIT", "1").strip().lower() in {"1", "true", "yes", "y"}
STRICT_FINAL = True
MAX_GOAL_SANITY = 40

log_section("Setup")
log_info(f"PROJECT_ROOT = {PROJECT_ROOT}")
log_info(f"OUTPUT_ROOT = {OUTPUT_ROOT}")
log_info(f"OUT_DIR = {OUT_DIR}")
log_info(f"SUB_DIR = {SUB_DIR}")
log_info(f"SUM_DIR = {SUM_DIR}")
log_info(f"EXP12F_VARIANT = {EXP12F_VARIANT}")
log_info("GT guardrail: ground_truth_bersih.csv hanya dibaca di Section 99 optional local audit")

# %% [markdown]
# # 02. Robust Data Finder dan Basic Audit
#
# Penjelasan bagian:
# Bagian ini mencari `train.csv`, `test.csv`, dan sample submission secara robust. Sample finder sengaja menghindari folder `outputs/` agar tidak salah mengambil submission eksperimen sebagai sample.
#
# Output yang perlu dilihat:
# Pastikan `TRAIN_PATH`, `TEST_PATH`, dan `SAMPLE_PATH` mengarah ke data asli. `id_col` harus sesuai sample: bisa `Id` atau `id`.

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


def candidate_data_dirs() -> list[Path]:
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
        if d not in out and d.exists():
            out.append(d)
    return out


def find_data_file(names: list[str] | str, recursive: bool = True) -> Path:
    if isinstance(names, str):
        names = [names]
    lower_names = {n.lower() for n in names}
    # direct / candidate dirs
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

ID_COL = "Id" if "Id" in sample_submission.columns else ("id" if "id" in sample_submission.columns else sample_submission.columns[0])
TEST_ID_COL = "Id" if "Id" in test_raw.columns else ("id" if "id" in test_raw.columns else ID_COL)
MATCH_COL = "match_id" if "match_id" in test_raw.columns else None

if MATCH_COL is None:
    # Fallback: assume every two consecutive rows in sample/test are mirror rows.
    test_raw = test_raw.copy()
    test_raw["match_id"] = np.arange(len(test_raw)) // 2
    MATCH_COL = "match_id"
    log_info("test.csv tidak punya match_id; fallback membuat match_id dari pasangan dua row berurutan")

sample_submission = sample_submission.rename(columns={ID_COL: "Id"}).copy()
test_raw = test_raw.rename(columns={TEST_ID_COL: "Id"}).copy()
ID_COL = "Id"

log_section("Input Audit")
log_info(f"TRAIN_PATH = {TRAIN_PATH}")
log_info(f"TEST_PATH = {TEST_PATH}")
log_info(f"SAMPLE_PATH = {SAMPLE_PATH}")
log_info(f"train shape = {train_raw.shape}")
log_info(f"test shape = {test_raw.shape}")
log_info(f"sample shape = {sample_submission.shape}")
log_check("sample has Id", "Id" in sample_submission.columns)
log_check("test has Id", "Id" in test_raw.columns)
log_check("test has match_id", MATCH_COL in test_raw.columns)
log_check("sample row count equals test row count", len(sample_submission) == len(test_raw))

# %% [markdown]
# # 03. Metric-Free Submission Helpers dan Pair Consistency
#
# Penjelasan bagian:
# Bagian ini menyiapkan helper alignment submission ke sample, validasi submission, konversi row-level ke match-level, dan pair consistency check. Semua kandidat final harus lolos helper ini.
#
# Output yang perlu dilihat:
# Setiap kandidat akan punya `submission_check.csv`. Untuk final/best_safe, semua check harus `True`, terutama `pair_consistency`.

# %%
def normalize_submission_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Id" not in out.columns:
        if "id" in out.columns:
            out = out.rename(columns={"id": "Id"})
        else:
            out = out.rename(columns={out.columns[0]: "Id"})
    required = ["Id", "team_goals", "opp_goals"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise KeyError(f"submission missing columns: {missing}")
    out = out[required].copy()
    out["Id"] = out["Id"].astype(str)
    out["team_goals"] = pd.to_numeric(out["team_goals"], errors="raise").round().astype(int).clip(lower=0)
    out["opp_goals"] = pd.to_numeric(out["opp_goals"], errors="raise").round().astype(int).clip(lower=0)
    return out


def align_to_sample(sub_df: pd.DataFrame) -> pd.DataFrame:
    sub = normalize_submission_columns(sub_df)
    sample = sample_submission[["Id"]].copy()
    sample["_id_key"] = sample["Id"].astype(str)
    sub["_id_key"] = sub["Id"].astype(str)
    aligned = sample.merge(sub[["_id_key", "team_goals", "opp_goals"]], on="_id_key", how="left", validate="one_to_one")
    aligned = aligned.drop(columns=["_id_key"])
    if aligned[["team_goals", "opp_goals"]].isna().any().any():
        bad = aligned.loc[aligned[["team_goals", "opp_goals"]].isna().any(axis=1), "Id"].head().tolist()
        raise RuntimeError(f"Submission missing predictions after align, example Id={bad}")
    aligned["team_goals"] = aligned["team_goals"].astype(int).clip(lower=0)
    aligned["opp_goals"] = aligned["opp_goals"].astype(int).clip(lower=0)
    return aligned[["Id", "team_goals", "opp_goals"]]


def outcome_scalar(a: int, b: int) -> int:
    if int(a) > int(b):
        return 1
    if int(a) < int(b):
        return -1
    return 0


def outcome_array(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return np.where(a > b, 1, np.where(a < b, -1, 0))


def get_tournament_weight(tournament: str) -> float:
    t = str(tournament).lower().strip()
    if "fifa world cup" in t or t == "world cup":
        return 2.00
    if "afc championship" in t or "afc asian cup" in t or "asian cup" in t:
        return 1.80
    if "friendly" in t:
        return 0.96
    return 1.20


def build_test_pair_frame() -> pd.DataFrame:
    tmp = test_raw[["Id", MATCH_COL] + (["tournament"] if "tournament" in test_raw.columns else []) + (["gender"] if "gender" in test_raw.columns else [])].copy()
    tmp["_sample_order"] = tmp["Id"].astype(str).map({v: i for i, v in enumerate(sample_submission["Id"].astype(str).tolist())})
    rows = []
    for match_id, g in tmp.groupby(MATCH_COL, sort=False):
        g = g.sort_values("_sample_order").reset_index(drop=True)
        if len(g) != 2:
            # Keep malformed groups but flag them; validation will fail if final cannot be checked.
            continue
        rows.append({
            "match_id": match_id,
            "row_id_a": str(g.loc[0, "Id"]),
            "row_id_b": str(g.loc[1, "Id"]),
            "gender": str(g.loc[0, "gender"]) if "gender" in g.columns else "__MISSING__",
            "tournament": str(g.loc[0, "tournament"]) if "tournament" in g.columns else "__MISSING__",
            "tournament_weight": get_tournament_weight(str(g.loc[0, "tournament"])) if "tournament" in g.columns else 1.20,
            "sample_order_a": int(g.loc[0, "_sample_order"]),
            "sample_order_b": int(g.loc[1, "_sample_order"]),
        })
    out = pd.DataFrame(rows)
    if len(out) * 2 != len(test_raw):
        log_info(f"Pair frame has {len(out):,} matches for {len(test_raw):,} test rows; some groups may not have 2 rows")
    return out


pair_frame = build_test_pair_frame()
log_check("pair_frame non-empty", len(pair_frame) > 0, f"n_matches={len(pair_frame):,}")


def pair_consistency_report(submission_df: pd.DataFrame) -> tuple[bool, int, pd.DataFrame]:
    sub = align_to_sample(submission_df)
    by_id = sub.set_index(sub["Id"].astype(str))
    bad_rows = []
    for _, r in pair_frame.iterrows():
        ida, idb = str(r["row_id_a"]), str(r["row_id_b"])
        if ida not in by_id.index or idb not in by_id.index:
            bad_rows.append({"match_id": r["match_id"], "reason": "missing_id"})
            continue
        a = by_id.loc[ida]
        b = by_id.loc[idb]
        ok = (int(a["team_goals"]) == int(b["opp_goals"])) and (int(a["opp_goals"]) == int(b["team_goals"]))
        if not ok:
            bad_rows.append({
                "match_id": r["match_id"],
                "row_id_a": ida,
                "row_id_b": idb,
                "a_team": int(a["team_goals"]),
                "a_opp": int(a["opp_goals"]),
                "b_team": int(b["team_goals"]),
                "b_opp": int(b["opp_goals"]),
                "reason": "mirror_mismatch",
            })
    bad_df = pd.DataFrame(bad_rows)
    return len(bad_df) == 0, int(len(bad_df)), bad_df


def submission_to_match_df(submission_df: pd.DataFrame, label: str = "candidate") -> pd.DataFrame:
    sub = align_to_sample(submission_df)
    by_id = sub.set_index(sub["Id"].astype(str))
    records = []
    for _, r in pair_frame.iterrows():
        ida, idb = str(r["row_id_a"]), str(r["row_id_b"])
        if ida not in by_id.index or idb not in by_id.index:
            continue
        ra = by_id.loc[ida]
        rb = by_id.loc[idb]
        a0 = int(ra["team_goals"])
        b0 = int(ra["opp_goals"])
        a1 = int(rb["opp_goals"])
        b1 = int(rb["team_goals"])
        records.append({
            "match_id": r["match_id"],
            "row_id_a": ida,
            "row_id_b": idb,
            "gender": r["gender"],
            "tournament": r["tournament"],
            "tournament_weight": float(r["tournament_weight"]),
            "a_from_row_a": a0,
            "b_from_row_a": b0,
            "a_from_row_b": a1,
            "b_from_row_b": b1,
            "pred_a": a0,
            "pred_b": b0,
            "pair_consistent": bool(a0 == a1 and b0 == b1),
            "label": label,
        })
    return pd.DataFrame(records)


def match_df_to_submission(match_df: pd.DataFrame) -> pd.DataFrame:
    required = ["row_id_a", "row_id_b", "pred_a", "pred_b"]
    missing = [c for c in required if c not in match_df.columns]
    if missing:
        raise KeyError(f"match_df missing columns: {missing}")
    rows_a = pd.DataFrame({
        "Id": match_df["row_id_a"].astype(str),
        "team_goals": match_df["pred_a"].round().astype(int).clip(lower=0),
        "opp_goals": match_df["pred_b"].round().astype(int).clip(lower=0),
    })
    rows_b = pd.DataFrame({
        "Id": match_df["row_id_b"].astype(str),
        "team_goals": match_df["pred_b"].round().astype(int).clip(lower=0),
        "opp_goals": match_df["pred_a"].round().astype(int).clip(lower=0),
    })
    return align_to_sample(pd.concat([rows_a, rows_b], ignore_index=True))


def scoreline(a, b) -> str:
    return f"{int(a)}-{int(b)}"


def validate_submission(submission_df: pd.DataFrame, label: str, strict: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    aligned = align_to_sample(submission_df)
    pair_ok, n_bad, _ = pair_consistency_report(aligned)
    checks = [
        {"label": label, "check": "shape_matches_sample", "passed": tuple(aligned.shape) == tuple(sample_submission[["Id", "team_goals", "opp_goals"]].shape) if set(["team_goals", "opp_goals"]).issubset(sample_submission.columns) else len(aligned) == len(sample_submission)},
        {"label": label, "check": "id_order_matches_sample", "passed": aligned["Id"].astype(str).tolist() == sample_submission["Id"].astype(str).tolist()},
        {"label": label, "check": "no_missing", "passed": not aligned[["team_goals", "opp_goals"]].isna().any().any()},
        {"label": label, "check": "no_duplicate_id", "passed": not aligned["Id"].duplicated().any()},
        {"label": label, "check": "non_negative", "passed": bool((aligned[["team_goals", "opp_goals"]] >= 0).all().all())},
        {"label": label, "check": "integer_prediction", "passed": bool(np.all(np.equal(aligned[["team_goals", "opp_goals"]].values, aligned[["team_goals", "opp_goals"]].values.astype(int))))},
        {"label": label, "check": "pair_consistency", "passed": pair_ok, "detail": f"n_bad_pairs={n_bad}"},
        {"label": label, "check": "max_goal_sanity_le_40", "passed": int(aligned[["team_goals", "opp_goals"]].max().max()) <= MAX_GOAL_SANITY},
    ]
    check_df = pd.DataFrame(checks)
    if strict and not bool(check_df["passed"].all()):
        display(check_df)
        raise RuntimeError(f"Submission validation failed for {label}")
    return aligned, check_df


def save_submission(submission_df: pd.DataFrame, label: str, strict: bool = False) -> tuple[Path, pd.DataFrame]:
    aligned, check_df = validate_submission(submission_df, label, strict=strict)
    path = SUB_DIR / f"submission_exp12f_{label}.csv"
    aligned.to_csv(path, index=False)
    log_saved(path)
    return path, check_df

# %% [markdown]
# # 04. Load Candidate Submission Pool
#
# Penjelasan bagian:
# Bagian ini mencari candidate submission dari EXP12E/EXP12D/EXP12C/EXP12B sebagai comparison artifact. Kandidat lama hanya dipakai untuk post-processing/consensus, bukan sebagai fitur training model.
#
# Output yang perlu dilihat:
# Lihat `candidate_pool_df`. F0 harus menemukan baseline EXP12E best, idealnya `submission_exp12e_e2_consensus_outcome_agree_only.csv`.

# %%
def find_submission_files() -> list[Path]:
    search_roots = [
        OUTPUT_ROOT / "exp12f_weighted_pair_router_micro_booster",
        OUTPUT_ROOT / "exp12e_pair_repair_micro_audit",
        OUTPUT_ROOT / "exp12d_exp22_alignment_hybrid",
        OUTPUT_ROOT / "exp12c_exp22_strength_rebase_hybrid",
        OUTPUT_ROOT / "exp12b_no_pseudo_pair_native_decoder_alignment",
        OUTPUT_ROOT / "exp12a_friend_pipeline_rebase_ablation",
        PROJECT_ROOT / "outputs",
        Path("/mnt/data"),
    ]
    paths = []
    for root in search_roots:
        if not root.exists():
            continue
        for p in root.rglob("*.csv"):
            name = p.name.lower()
            if not name.startswith("submission_"):
                continue
            if "sample" in name:
                continue
            if p.resolve() == SAMPLE_PATH.resolve():
                continue
            paths.append(p.resolve())
    return sorted(set(paths), key=lambda x: str(x))


def label_from_path(p: Path) -> str:
    stem = p.stem
    # Keep informative but short enough.
    for prefix in ["submission_", "submission"]:
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
    return stem


all_submission_paths = find_submission_files()
rows = []
loaded_candidates = {}
for p in all_submission_paths:
    try:
        sub = pd.read_csv(p)
        sub = align_to_sample(sub)
        pair_ok, n_bad, _ = pair_consistency_report(sub)
        label = label_from_path(p)
        # Avoid overwriting label collision by appending hash-like index.
        unique_label = label
        k = 2
        while unique_label in loaded_candidates:
            unique_label = f"{label}_{k}"
            k += 1
        loaded_candidates[unique_label] = {"path": p, "submission": sub, "pair_ok": pair_ok, "n_bad_pairs": n_bad}
        rows.append({
            "label": unique_label,
            "file_name": p.name,
            "path": str(p),
            "pair_consistency": pair_ok,
            "n_bad_pairs": n_bad,
            "mean_pred_total": float((sub["team_goals"] + sub["opp_goals"]).mean()),
            "max_pred_goal": int(sub[["team_goals", "opp_goals"]].max().max()),
        })
    except Exception as e:
        rows.append({"label": p.stem, "file_name": p.name, "path": str(p), "load_error": repr(e)})

candidate_pool_df = pd.DataFrame(rows)
candidate_pool_df.to_csv(SUM_DIR / "candidate_pool_loaded.csv", index=False)
log_saved(SUM_DIR / "candidate_pool_loaded.csv")
display(candidate_pool_df.head(50))

BASE_PRIORITY_PATTERNS = [
    "exp12e_e2_consensus_outcome_agree_only",
    "exp12e_e1_repair_min_total_same_outcome",
    "exp12e_e1_repair_preserve_gd_lower_total",
    "exp12e_best_safe",
    "exp12d_best_safe",
    "exp12d_d1_exp15_first",
]

RAW_PRIORITY_PATTERNS = [
    "raw_exp15",
    "d1_exp15_first",
    "exp12d_d1_exp15_first",
    "exp12e_e0_exp12d_reproduction",
    "exp15_first",
]


def pick_candidate_by_patterns(patterns: list[str], require_pair_ok: bool | None = None):
    for pat in patterns:
        pat = pat.lower()
        for label, obj in loaded_candidates.items():
            hay = f"{label} {obj['path'].name} {obj['path']}".lower()
            if pat in hay:
                if require_pair_ok is not None and bool(obj["pair_ok"]) != bool(require_pair_ok):
                    continue
                return label, obj
    return None, None


base_label, base_obj = pick_candidate_by_patterns(BASE_PRIORITY_PATTERNS, require_pair_ok=True)
if base_obj is None:
    base_label, base_obj = pick_candidate_by_patterns(BASE_PRIORITY_PATTERNS, require_pair_ok=None)
if base_obj is None and loaded_candidates:
    # fallback: first pair-consistent candidate, then first candidate.
    for label, obj in loaded_candidates.items():
        if obj["pair_ok"]:
            base_label, base_obj = label, obj
            break
if base_obj is None and loaded_candidates:
    base_label, base_obj = next(iter(loaded_candidates.items()))
if base_obj is None:
    raise FileNotFoundError("Tidak ada submission candidate yang bisa diload. Jalankan EXP12E/EXP12D dulu atau taruh submission_*.csv di outputs.")

raw_label, raw_obj = pick_candidate_by_patterns(RAW_PRIORITY_PATTERNS, require_pair_ok=False)
if raw_obj is None:
    raw_label, raw_obj = base_label, base_obj

base_submission = align_to_sample(base_obj["submission"])
raw_pair_source_submission = align_to_sample(raw_obj["submission"])

log_info(f"F0 base candidate = {base_label} | path={base_obj['path']}")
log_info(f"Raw pair source candidate = {raw_label} | path={raw_obj['path']}")

# %% [markdown]
# # 05. Pair Repair Policy Library
#
# Penjelasan bagian:
# Bagian ini membuat beberapa pair-repair policy dari dua row mirror. Policy ini dipakai oleh F1 weighted router dan juga sebagai candidate tambahan untuk F2 consensus.
#
# Output yang perlu dilihat:
# Cek `pair_repair_policy_grid.csv`: setiap policy harus `pair_consistency=True`, `bad_pairs_after=0`, dan changed rate tidak liar.

# %%
def choose_score_by_policy(row, policy: str) -> tuple[int, int]:
    a0 = int(row["a_from_row_a"])
    b0 = int(row["b_from_row_a"])
    a1 = int(row["a_from_row_b"])
    b1 = int(row["b_from_row_b"])
    cand = [(a0, b0, "row_a"), (a1, b1, "row_b")]

    if policy == "repair_trust_row_a":
        return max(0, a0), max(0, b0)
    if policy == "repair_trust_row_b":
        return max(0, a1), max(0, b1)

    avg_a = (a0 + a1) / 2.0
    avg_b = (b0 + b1) / 2.0
    if policy == "repair_mean_floor":
        return max(0, int(math.floor(avg_a))), max(0, int(math.floor(avg_b)))
    if policy == "repair_mean_ceil":
        return max(0, int(math.ceil(avg_a))), max(0, int(math.ceil(avg_b)))
    if policy == "repair_mean_round":
        return max(0, int(round(avg_a))), max(0, int(round(avg_b)))

    if policy in {"repair_lower_total", "repair_higher_total"}:
        key = (lambda x: (x[0] + x[1], abs(x[0] - x[1])))
        chosen = min(cand, key=key) if policy == "repair_lower_total" else max(cand, key=key)
        return max(0, chosen[0]), max(0, chosen[1])

    out0 = outcome_scalar(a0, b0)
    out1 = outcome_scalar(a1, b1)
    gd0 = a0 - b0
    gd1 = a1 - b1

    if policy == "repair_min_total_same_outcome":
        if out0 == out1:
            chosen = min(cand, key=lambda x: (x[0] + x[1], abs(x[0] - x[1])))
            return max(0, chosen[0]), max(0, chosen[1])
        return max(0, int(round(avg_a))), max(0, int(round(avg_b)))

    if policy == "repair_max_total_same_outcome":
        if out0 == out1:
            chosen = max(cand, key=lambda x: (x[0] + x[1], -abs(x[0] - x[1])))
            return max(0, chosen[0]), max(0, chosen[1])
        return max(0, int(round(avg_a))), max(0, int(round(avg_b)))

    if policy == "repair_preserve_gd_lower_total":
        # Prefer candidates that preserve an agreed GD; if GD disagrees, choose lower total only when outcome agrees, else mean-round.
        if gd0 == gd1:
            chosen = min(cand, key=lambda x: x[0] + x[1])
            return max(0, chosen[0]), max(0, chosen[1])
        if out0 == out1:
            chosen = min(cand, key=lambda x: (abs((x[0] - x[1]) - round((gd0 + gd1) / 2.0)), x[0] + x[1]))
            return max(0, chosen[0]), max(0, chosen[1])
        return max(0, int(round(avg_a))), max(0, int(round(avg_b)))

    if policy == "repair_outcome_agree_only":
        if out0 == out1:
            chosen = min(cand, key=lambda x: (x[0] + x[1], abs(x[0] - x[1])))
            return max(0, chosen[0]), max(0, chosen[1])
        return max(0, int(round(avg_a))), max(0, int(round(avg_b)))

    raise ValueError(f"Unknown repair policy: {policy}")


REPAIR_POLICIES = [
    "repair_mean_round",
    "repair_mean_floor",
    "repair_mean_ceil",
    "repair_lower_total",
    "repair_higher_total",
    "repair_preserve_gd_lower_total",
    "repair_trust_row_a",
    "repair_trust_row_b",
    "repair_min_total_same_outcome",
    "repair_max_total_same_outcome",
    "repair_outcome_agree_only",
]


def repair_submission_by_policy(raw_submission: pd.DataFrame, policy: str, label: str | None = None) -> pd.DataFrame:
    raw_match = submission_to_match_df(raw_submission, label=label or policy)
    out = raw_match.copy()
    preds = out.apply(lambda r: choose_score_by_policy(r, policy), axis=1)
    out["pred_a"] = [x[0] for x in preds]
    out["pred_b"] = [x[1] for x in preds]
    return match_df_to_submission(out)


def compare_to_base_match(candidate_sub: pd.DataFrame, base_sub: pd.DataFrame, label: str) -> dict:
    cand = submission_to_match_df(candidate_sub, label=label)[["match_id", "pred_a", "pred_b", "gender", "tournament", "tournament_weight"]].rename(columns={"pred_a": "cand_a", "pred_b": "cand_b"})
    base = submission_to_match_df(base_sub, label="base")[["match_id", "pred_a", "pred_b"]].rename(columns={"pred_a": "base_a", "pred_b": "base_b"})
    m = cand.merge(base, on="match_id", how="left", validate="one_to_one")
    changed = (m["cand_a"] != m["base_a"]) | (m["cand_b"] != m["base_b"])
    high = m["tournament_weight"] >= 1.8
    low = m["tournament_weight"] <= 0.96
    return {
        "label": label,
        "n_matches": len(m),
        "n_changed": int(changed.sum()),
        "changed_rate": float(changed.mean()) if len(m) else 0.0,
        "changed_high_weight": int((changed & high).sum()),
        "changed_low_weight": int((changed & low).sum()),
        "mean_pred_total": float((m["cand_a"] + m["cand_b"]).mean()),
        "max_pred_goal": int(max(m["cand_a"].max(), m["cand_b"].max())) if len(m) else 0,
    }


policy_records = []
policy_submissions = {}
raw_pair_ok, raw_bad_before, _ = pair_consistency_report(raw_pair_source_submission)
for pol in REPAIR_POLICIES:
    sub = repair_submission_by_policy(raw_pair_source_submission, pol, label=pol)
    pair_ok, bad_after, _ = pair_consistency_report(sub)
    rec = compare_to_base_match(sub, base_submission, pol)
    rec.update({
        "policy": pol,
        "bad_pairs_before": int(raw_bad_before),
        "bad_pairs_after": int(bad_after),
        "pair_consistency": bool(pair_ok),
    })
    policy_records.append(rec)
    policy_submissions[pol] = sub

pair_repair_policy_grid = pd.DataFrame(policy_records).sort_values(["pair_consistency", "changed_rate"], ascending=[False, True])
pair_repair_policy_grid.to_csv(SUM_DIR / "pair_repair_policy_grid.csv", index=False)
log_saved(SUM_DIR / "pair_repair_policy_grid.csv")
display(pair_repair_policy_grid)

# %% [markdown]
# # 06. F0 — Reproduce EXP12E Best
#
# Penjelasan bagian:
# F0 menyimpan ulang baseline EXP12E best sebagai anchor. Jika base candidate tidak pair-consistent, script akan memperbaikinya dengan policy `repair_min_total_same_outcome` atau fallback `repair_mean_round`.
#
# Output yang perlu dilihat:
# `submission_exp12f_f0_exp12e_best_reproduction.csv` harus pair-consistent dan id order sama dengan sample.

# %%
f0_pair_ok, f0_bad, _ = pair_consistency_report(base_submission)
if not f0_pair_ok:
    log_info(f"F0 base pair inconsistency found: bad_pairs={f0_bad}; repairing with repair_min_total_same_outcome")
    f0_submission = policy_submissions.get("repair_min_total_same_outcome")
    if f0_submission is None:
        f0_submission = repair_submission_by_policy(base_submission, "repair_mean_round", label="f0_repaired")
else:
    f0_submission = base_submission.copy()

f0_path, f0_check = save_submission(f0_submission, "f0_exp12e_best_reproduction", strict=False)
f0_summary = pd.DataFrame([compare_to_base_match(f0_submission, base_submission, "f0_exp12e_best_reproduction")])
f0_summary["source_label"] = base_label
f0_summary["source_path"] = str(base_obj["path"])
f0_summary.to_csv(SUM_DIR / "f0_reproduction_summary.csv", index=False)
log_saved(SUM_DIR / "f0_reproduction_summary.csv")
display(f0_check)

# F0 is the new reference for all micro changes.
base_submission = align_to_sample(f0_submission)
base_match = submission_to_match_df(base_submission, label="f0_base")

# %% [markdown]
# # 07. F1 — Weighted-Aware Pair Repair Router
#
# Penjelasan bagian:
# F1 membuat router berbasis tournament weight. High-weight tournament diarahkan ke policy yang menjaga GD/outcome, sedangkan low-weight/friendly boleh lebih fleksibel ke lower-total atau min-total policy.
#
# Output yang perlu dilihat:
# Lihat `weighted_pair_router_grid.csv` dan `tournament_changed_analysis.csv`. Candidate aman harus pair-consistent, changed rate kecil, dan perubahan high-weight tidak liar.

# %%
def match_from_submission_for_policy(sub: pd.DataFrame, label: str) -> pd.DataFrame:
    return submission_to_match_df(sub, label=label)[["match_id", "row_id_a", "row_id_b", "gender", "tournament", "tournament_weight", "pred_a", "pred_b"]]


policy_match_cache = {pol: match_from_submission_for_policy(sub, pol) for pol, sub in policy_submissions.items()}
policy_match_cache["exp12e_base"] = match_from_submission_for_policy(base_submission, "exp12e_base")


def route_policy_for_match(row: pd.Series, router_name: str) -> str:
    w = float(row.get("tournament_weight", 1.2))
    t = str(row.get("tournament", "")).lower()
    high = w >= 1.8
    low = w <= 0.96

    if router_name == "f1_high_weight_preserve_gd_else_exp12e":
        return "repair_preserve_gd_lower_total" if high else "exp12e_base"
    if router_name == "f1_high_weight_outcome_agree_else_min_total":
        return "repair_outcome_agree_only" if high else "repair_min_total_same_outcome"
    if router_name == "f1_friendly_lower_total_high_weight_preserve_gd":
        if high:
            return "repair_preserve_gd_lower_total"
        if low or "friendly" in t:
            return "repair_lower_total"
        return "exp12e_base"
    if router_name == "f1_asian_world_strict_preserve_else_consensus":
        if "world cup" in t or "asian" in t or high:
            return "repair_preserve_gd_lower_total"
        return "repair_outcome_agree_only"
    if router_name == "f1_low_weight_min_total_default_consensus":
        return "repair_min_total_same_outcome" if low else "repair_outcome_agree_only"
    raise ValueError(router_name)


F1_ROUTERS = [
    "f1_high_weight_preserve_gd_else_exp12e",
    "f1_high_weight_outcome_agree_else_min_total",
    "f1_friendly_lower_total_high_weight_preserve_gd",
    "f1_asian_world_strict_preserve_else_consensus",
    "f1_low_weight_min_total_default_consensus",
]


def build_weighted_router_submission(router_name: str) -> pd.DataFrame:
    base_m = policy_match_cache["exp12e_base"].copy()
    pred_rows = []
    # dict for quick lookup policy -> match_id tuple
    cache_index = {pol: df.set_index("match_id") for pol, df in policy_match_cache.items()}
    for _, r in base_m.iterrows():
        pol = route_policy_for_match(r, router_name)
        src = cache_index[pol].loc[r["match_id"]]
        pred_rows.append({
            "match_id": r["match_id"],
            "row_id_a": r["row_id_a"],
            "row_id_b": r["row_id_b"],
            "gender": r["gender"],
            "tournament": r["tournament"],
            "tournament_weight": r["tournament_weight"],
            "pred_a": int(src["pred_a"]),
            "pred_b": int(src["pred_b"]),
            "chosen_policy": pol,
        })
    return match_df_to_submission(pd.DataFrame(pred_rows))


f1_records = []
f1_submissions = {}
for router in F1_ROUTERS:
    sub = build_weighted_router_submission(router)
    path, check = save_submission(sub, router, strict=False)
    rec = compare_to_base_match(sub, base_submission, router)
    rec.update({"variant": router, "path": str(path), "pair_consistency": bool(check.loc[check["check"] == "pair_consistency", "passed"].iloc[0])})
    f1_records.append(rec)
    f1_submissions[router] = sub

weighted_pair_router_grid = pd.DataFrame(f1_records).sort_values("changed_rate")
weighted_pair_router_grid.to_csv(SUM_DIR / "weighted_pair_router_grid.csv", index=False)
log_saved(SUM_DIR / "weighted_pair_router_grid.csv")
display(weighted_pair_router_grid)

def tournament_changed_analysis(candidate_subs: dict[str, pd.DataFrame], base_sub: pd.DataFrame) -> pd.DataFrame:
    base_m = submission_to_match_df(base_sub, label="base")[["match_id", "pred_a", "pred_b"]].rename(columns={"pred_a": "base_a", "pred_b": "base_b"})
    rows = []
    for label, sub in candidate_subs.items():
        cand = submission_to_match_df(sub, label=label)[["match_id", "pred_a", "pred_b", "tournament", "tournament_weight"]].rename(columns={"pred_a": "cand_a", "pred_b": "cand_b"})
        m = cand.merge(base_m, on="match_id", how="left", validate="one_to_one")
        m["changed"] = (m["cand_a"] != m["base_a"]) | (m["cand_b"] != m["base_b"])
        for key, g in m.groupby(["tournament", "tournament_weight"], dropna=False):
            rows.append({
                "variant": label,
                "tournament": key[0],
                "tournament_weight": key[1],
                "n_matches": len(g),
                "n_changed": int(g["changed"].sum()),
                "changed_rate": float(g["changed"].mean()),
            })
    return pd.DataFrame(rows).sort_values(["variant", "n_changed"], ascending=[True, False])

tournament_changed_df = tournament_changed_analysis(f1_submissions, base_submission)
tournament_changed_df.to_csv(SUM_DIR / "tournament_changed_analysis.csv", index=False)
log_saved(SUM_DIR / "tournament_changed_analysis.csv")
display(tournament_changed_df.head(30))

# %% [markdown]
# # 08. F2 — Candidate Consensus Router
#
# Penjelasan bagian:
# F2 memakai banyak candidate submission sebagai comparison artifact. Default tetap EXP12E best, lalu scoreline diganti hanya saat ada mayoritas exact atau mayoritas outcome+GD yang kuat.
#
# Output yang perlu dilihat:
# Lihat `candidate_consensus_grid.csv`, `candidate_agreement_matrix.csv`, dan `consensus_changed_analysis.csv`. Consensus yang baik harus changed rate kecil dan pair-consistent.

# %%
def build_candidate_match_pool(max_candidates: int = 18) -> dict[str, pd.DataFrame]:
    # Include F0, policy repairs, F1 routers, and previously loaded pair-consistent submissions.
    pool = {"f0_exp12e_base": submission_to_match_df(base_submission, "f0_exp12e_base")}
    for pol, sub in policy_submissions.items():
        pool[pol] = submission_to_match_df(sub, pol)
    for label, sub in f1_submissions.items():
        pool[label] = submission_to_match_df(sub, label)

    # Add loaded candidates, prioritizing pair-consistent and informative prior experiments.
    priority_terms = ["exp12e", "exp12d", "exp12c", "exp12b", "exp22", "best_safe", "exp15", "b5"]
    sorted_loaded = sorted(
        loaded_candidates.items(),
        key=lambda kv: (
            0 if kv[1]["pair_ok"] else 1,
            min([kv[0].lower().find(t) if t in kv[0].lower() else 999 for t in priority_terms]),
            kv[0],
        ),
    )
    for label, obj in sorted_loaded:
        if len(pool) >= max_candidates:
            break
        try:
            sub = obj["submission"]
            if not obj["pair_ok"]:
                # use mean-round repaired version as comparison artifact
                sub = repair_submission_by_policy(sub, "repair_mean_round", label=f"{label}_repaired")
            safe_label = f"loaded_{label}"[:120]
            if safe_label not in pool:
                pool[safe_label] = submission_to_match_df(sub, safe_label)
        except Exception:
            continue
    return pool


candidate_match_pool = build_candidate_match_pool(max_candidates=22)
log_info(f"Candidate match pool size = {len(candidate_match_pool)}")

# Precompute candidate indices once.
# Ini penting supaya F3/F4 tidak melakukan df.set_index("match_id") ribuan kali.
CANDIDATE_MATCH_POOL_INDEX = {
    lab: df.set_index("match_id", drop=False)
    for lab, df in candidate_match_pool.items()
}

# Agreement matrix by exact score agreement rate.
labels = list(candidate_match_pool.keys())
agreement_rows = []
for i, li in enumerate(labels):
    ai = candidate_match_pool[li][["match_id", "pred_a", "pred_b"]].rename(columns={"pred_a": "a_i", "pred_b": "b_i"})
    for lj in labels[i + 1:]:
        bj = candidate_match_pool[lj][["match_id", "pred_a", "pred_b"]].rename(columns={"pred_a": "a_j", "pred_b": "b_j"})
        m = ai.merge(bj, on="match_id", how="inner", validate="one_to_one")
        exact_agree = (m["a_i"] == m["a_j"]) & (m["b_i"] == m["b_j"])
        out_agree = outcome_array(m["a_i"], m["b_i"]) == outcome_array(m["a_j"], m["b_j"])
        gd_agree = (m["a_i"] - m["b_i"]) == (m["a_j"] - m["b_j"])
        agreement_rows.append({
            "candidate_a": li,
            "candidate_b": lj,
            "n": len(m),
            "exact_agreement_rate": float(exact_agree.mean()),
            "outcome_agreement_rate": float(out_agree.mean()),
            "gd_agreement_rate": float(gd_agree.mean()),
        })

candidate_agreement_matrix = pd.DataFrame(agreement_rows).sort_values("exact_agreement_rate", ascending=False)
candidate_agreement_matrix.to_csv(SUM_DIR / "candidate_agreement_matrix.csv", index=False)
log_saved(SUM_DIR / "candidate_agreement_matrix.csv")
display(candidate_agreement_matrix.head(20))


def consensus_choice_for_match(match_id, base_score, variant: str, candidate_scores: list[tuple[int, int, str]], meta_row: pd.Series) -> tuple[int, int]:
    base_a, base_b = base_score
    counts = Counter((a, b) for a, b, _ in candidate_scores)
    exact_score, exact_count = counts.most_common(1)[0]

    # Outcome + GD grouping.
    og_counts = Counter((outcome_scalar(a, b), a - b) for a, b, _ in candidate_scores)
    og_key, og_count = og_counts.most_common(1)[0]
    og_candidates = [(a, b) for a, b, _ in candidate_scores if (outcome_scalar(a, b), a - b) == og_key]
    if og_candidates:
        # choose closest total to base among majority outcome+GD group
        target_total = base_a + base_b
        og_score = min(og_candidates, key=lambda x: (abs((x[0] + x[1]) - target_total), abs(x[0] - base_a) + abs(x[1] - base_b)))
    else:
        og_score = (base_a, base_b)

    high_weight = float(meta_row.get("tournament_weight", 1.2)) >= 1.8

    if variant == "f2_exact_majority_router":
        return exact_score if exact_count >= 3 else (base_a, base_b)
    if variant == "f2_outcome_gd_majority_router":
        return og_score if og_count >= 3 else (base_a, base_b)
    if variant == "f2_conservative_consensus_only":
        if exact_count >= 3 and abs((exact_score[0] + exact_score[1]) - (base_a + base_b)) <= 1:
            return exact_score
        return (base_a, base_b)
    if variant == "f2_high_weight_consensus_only":
        if high_weight and og_count >= 3:
            return og_score
        if (not high_weight) and exact_count >= 3:
            return exact_score
        return (base_a, base_b)
    if variant == "f2_exp12e_exp12d_agreement_lock":
        # if base and any exp12d/best candidate agrees, lock; otherwise exact majority.
        has_base_agree = any((a, b) == (base_a, base_b) and ("exp12d" in lab.lower() or "best" in lab.lower()) for a, b, lab in candidate_scores)
        if has_base_agree:
            return (base_a, base_b)
        return exact_score if exact_count >= 3 else (base_a, base_b)
    raise ValueError(variant)


F2_VARIANTS = [
    "f2_exact_majority_router",
    "f2_outcome_gd_majority_router",
    "f2_conservative_consensus_only",
    "f2_high_weight_consensus_only",
    "f2_exp12e_exp12d_agreement_lock",
]


def build_consensus_submission(variant: str) -> pd.DataFrame:
    base_m = base_match.copy()
    pool_index = {lab: df.set_index("match_id") for lab, df in candidate_match_pool.items()}
    rows = []
    for _, r in base_m.iterrows():
        match_id = r["match_id"]
        scores = []
        for lab, idxdf in pool_index.items():
            if match_id in idxdf.index:
                rr = idxdf.loc[match_id]
                scores.append((int(rr["pred_a"]), int(rr["pred_b"]), lab))
        a, b = consensus_choice_for_match(match_id, (int(r["pred_a"]), int(r["pred_b"])), variant, scores, r)
        row = r.to_dict()
        row["pred_a"] = a
        row["pred_b"] = b
        rows.append(row)
    return match_df_to_submission(pd.DataFrame(rows))


f2_records = []
f2_submissions = {}
for var in F2_VARIANTS:
    sub = build_consensus_submission(var)
    path, check = save_submission(sub, var, strict=False)
    rec = compare_to_base_match(sub, base_submission, var)
    rec.update({"variant": var, "path": str(path), "pair_consistency": bool(check.loc[check["check"] == "pair_consistency", "passed"].iloc[0])})
    f2_records.append(rec)
    f2_submissions[var] = sub

candidate_consensus_grid = pd.DataFrame(f2_records).sort_values("changed_rate")
candidate_consensus_grid.to_csv(SUM_DIR / "candidate_consensus_grid.csv", index=False)
log_saved(SUM_DIR / "candidate_consensus_grid.csv")
display(candidate_consensus_grid)

consensus_changed_analysis = tournament_changed_analysis(f2_submissions, base_submission)
consensus_changed_analysis.to_csv(SUM_DIR / "consensus_changed_analysis.csv", index=False)
log_saved(SUM_DIR / "consensus_changed_analysis.csv")

# %% [markdown]
# # 09. F3 — Micro Outcome Booster
#
# Penjelasan bagian:
# F3 mengubah draw menjadi win/loss pada subset sangat kecil ketika minimal dua kandidat lain mendukung outcome baru. Ini dibuat dengan changed cap 0.5%, 1%, 2%, dan 3%.
#
# Output yang perlu dilihat:
# Lihat `micro_outcome_grid.csv` dan `micro_outcome_transition_analysis.csv`. Changed rate tidak boleh melewati cap, dan pair consistency wajib True.

# %%
def best_supported_alternative(
    base_row: pd.Series,
    mode: str,
    candidate_pool: dict[str, pd.DataFrame] | None = None,
) -> tuple[int, int, float, str] | None:
    match_id = base_row["match_id"]
    base_a, base_b = int(base_row["pred_a"]), int(base_row["pred_b"])
    base_out = outcome_scalar(base_a, base_b)
    base_gd = base_a - base_b
    base_total = base_a + base_b

    # Pakai precomputed index agar runtime tidak meledak.
    pool_index = globals().get("CANDIDATE_MATCH_POOL_INDEX", None)
    if pool_index is None:
        if candidate_pool is None:
            candidate_pool = candidate_match_pool
        pool_index = {
            lab: df.set_index("match_id", drop=False)
            for lab, df in candidate_pool.items()
        }

    scores = []
    for lab, idx in pool_index.items():
        if match_id not in idx.index:
            continue

        rr = idx.loc[match_id]

        # Safety kalau ada duplicate match_id tidak sengaja.
        if isinstance(rr, pd.DataFrame):
            rr = rr.iloc[0]

        a, b = int(rr["pred_a"]), int(rr["pred_b"])
        if (a, b) == (base_a, base_b):
            continue

        scores.append((a, b, lab))

    if not scores:
        return None

    if mode == "outcome":
        # Only draw -> decisive, supported by at least 2 candidates with same new outcome.
        if base_out != 0:
            return None

        counts = Counter(
            outcome_scalar(a, b)
            for a, b, _ in scores
            if outcome_scalar(a, b) != 0
        )

        if not counts:
            return None

        new_out, support = counts.most_common(1)[0]
        if support < 2:
            return None

        cands = [
            (a, b, lab)
            for a, b, lab in scores
            if outcome_scalar(a, b) == new_out
        ]

        a, b, lab = min(
            cands,
            key=lambda x: (
                abs((x[0] + x[1]) - base_total),
                abs((x[0] - x[1]) - base_gd),
                x[0] + x[1],
            ),
        )

        return a, b, float(support), lab

    if mode == "exact_same_outcome":
        cands = [
            (a, b, lab)
            for a, b, lab in scores
            if outcome_scalar(a, b) == base_out
            and abs((a + b) - base_total) <= 1
            and abs((a - b) - base_gd) <= 1
        ]

        if not cands:
            return None

        counts = Counter((a, b) for a, b, _ in cands)
        (a, b), support = counts.most_common(1)[0]

        if support < 2:
            return None

        lab = [lab for aa, bb, lab in cands if (aa, bb) == (a, b)][0]
        return a, b, float(support), lab

    raise ValueError(mode)


def apply_micro_candidate(base_sub: pd.DataFrame, variant: str, mode: str, cap_rate: float, extra_filter=None) -> pd.DataFrame:
    base_m = submission_to_match_df(base_sub, label="micro_base")
    rows = []
    candidates = []
    for _, r in base_m.iterrows():
        if extra_filter is not None and not extra_filter(r):
            continue
        alt = best_supported_alternative(r, mode=mode, candidate_pool=candidate_match_pool)
        if alt is None:
            continue
        a, b, support, source_lab = alt
        # Risk filter: avoid W high total and extreme changes.
        if str(r.get("gender", "")).upper() == "W" and max(a + b, int(r["pred_a"]) + int(r["pred_b"])) >= 5:
            continue
        if abs((a + b) - (int(r["pred_a"]) + int(r["pred_b"]))) > 1:
            continue
        candidates.append({
            "match_id": r["match_id"],
            "new_a": int(a),
            "new_b": int(b),
            "support": float(support),
            "source_label": source_lab,
            "tournament_weight": float(r.get("tournament_weight", 1.2)),
        })
    cand_df = pd.DataFrame(candidates)
    n_cap = int(math.floor(len(base_m) * cap_rate))
    n_cap = max(1, n_cap) if len(cand_df) > 0 and cap_rate > 0 else 0
    if len(cand_df) > 0:
        cand_df = cand_df.sort_values(["support", "tournament_weight"], ascending=[False, True]).head(n_cap)
    repl = cand_df.set_index("match_id").to_dict("index") if len(cand_df) else {}
    for _, r in base_m.iterrows():
        row = r.to_dict()
        if r["match_id"] in repl:
            row["pred_a"] = repl[r["match_id"]]["new_a"]
            row["pred_b"] = repl[r["match_id"]]["new_b"]
            row["micro_variant"] = variant
        rows.append(row)
    return match_df_to_submission(pd.DataFrame(rows))


F3_CONFIGS = [
    ("f3_draw_to_win_cap005", 0.005, None),
    ("f3_draw_to_win_cap010", 0.010, None),
    ("f3_draw_to_win_cap020", 0.020, None),
    ("f3_draw_to_win_cap030", 0.030, None),
    ("f3_high_weight_draw_to_win_cap010", 0.010, lambda r: float(r.get("tournament_weight", 1.2)) >= 1.2),
]

f3_records = []
f3_submissions = {}
for var, cap, filt in F3_CONFIGS:
    sub = apply_micro_candidate(base_submission, var, mode="outcome", cap_rate=cap, extra_filter=filt)
    path, check = save_submission(sub, var, strict=False)
    rec = compare_to_base_match(sub, base_submission, var)
    rec.update({"variant": var, "cap_rate": cap, "path": str(path), "pair_consistency": bool(check.loc[check["check"] == "pair_consistency", "passed"].iloc[0])})
    f3_records.append(rec)
    f3_submissions[var] = sub

micro_outcome_grid = pd.DataFrame(f3_records).sort_values("changed_rate")
micro_outcome_grid.to_csv(SUM_DIR / "micro_outcome_grid.csv", index=False)
log_saved(SUM_DIR / "micro_outcome_grid.csv")
display(micro_outcome_grid)

micro_outcome_transition = []
for label, sub in f3_submissions.items():
    cand = submission_to_match_df(sub, label=label)[["match_id", "pred_a", "pred_b"]].rename(columns={"pred_a": "cand_a", "pred_b": "cand_b"})
    b = base_match[["match_id", "pred_a", "pred_b", "gender", "tournament", "tournament_weight"]].rename(columns={"pred_a": "base_a", "pred_b": "base_b"})
    m = cand.merge(b, on="match_id", how="left", validate="one_to_one")
    m["changed"] = (m["cand_a"] != m["base_a"]) | (m["cand_b"] != m["base_b"])
    ch = m[m["changed"]].copy()
    if len(ch):
        ch["transition"] = ch.apply(lambda r: f"{int(r['base_a'])}-{int(r['base_b'])}->{int(r['cand_a'])}-{int(r['cand_b'])}", axis=1)
        for tr, g in ch.groupby("transition"):
            micro_outcome_transition.append({"variant": label, "transition": tr, "count": len(g)})

micro_outcome_transition_df = pd.DataFrame(micro_outcome_transition)
micro_outcome_transition_df.to_csv(SUM_DIR / "micro_outcome_transition_analysis.csv", index=False)
log_saved(SUM_DIR / "micro_outcome_transition_analysis.csv")

# %% [markdown]
# # 10. F4 — Micro Exact Booster
#
# Penjelasan bagian:
# F4 mengganti scoreline ke kandidat yang didukung candidate pool, tetapi hanya jika outcome sama dan total/GD tidak jauh. Cap kecil menjaga agar tidak berubah agresif.
#
# Output yang perlu dilihat:
# Lihat `micro_exact_grid.csv`. Candidate aman harus changed rate sesuai cap, same outcome dominan, dan pair consistency True.

# %%
F4_CONFIGS = [
    ("f4_same_outcome_exact_cap005", 0.005, "exact_same_outcome"),
    ("f4_same_outcome_exact_cap010", 0.010, "exact_same_outcome"),
    ("f4_same_outcome_exact_cap020", 0.020, "exact_same_outcome"),
    ("f4_same_outcome_exact_cap030", 0.030, "exact_same_outcome"),
]

f4_records = []
f4_submissions = {}
for var, cap, mode in F4_CONFIGS:
    sub = apply_micro_candidate(base_submission, var, mode=mode, cap_rate=cap, extra_filter=None)
    path, check = save_submission(sub, var, strict=False)
    rec = compare_to_base_match(sub, base_submission, var)
    rec.update({"variant": var, "cap_rate": cap, "path": str(path), "pair_consistency": bool(check.loc[check["check"] == "pair_consistency", "passed"].iloc[0])})
    f4_records.append(rec)
    f4_submissions[var] = sub

micro_exact_grid = pd.DataFrame(f4_records).sort_values("changed_rate")
micro_exact_grid.to_csv(SUM_DIR / "micro_exact_grid.csv", index=False)
log_saved(SUM_DIR / "micro_exact_grid.csv")
display(micro_exact_grid)

# transition detail for F4
micro_exact_transition = []
for label, sub in f4_submissions.items():
    cand = submission_to_match_df(sub, label=label)[["match_id", "pred_a", "pred_b"]].rename(columns={"pred_a": "cand_a", "pred_b": "cand_b"})
    b = base_match[["match_id", "pred_a", "pred_b", "gender", "tournament", "tournament_weight"]].rename(columns={"pred_a": "base_a", "pred_b": "base_b"})
    m = cand.merge(b, on="match_id", how="left", validate="one_to_one")
    m["changed"] = (m["cand_a"] != m["base_a"]) | (m["cand_b"] != m["base_b"])
    ch = m[m["changed"]].copy()
    if len(ch):
        ch["transition"] = ch.apply(lambda r: f"{int(r['base_a'])}-{int(r['base_b'])}->{int(r['cand_a'])}-{int(r['cand_b'])}", axis=1)
        for tr, g in ch.groupby("transition"):
            micro_exact_transition.append({"variant": label, "transition": tr, "count": len(g)})

micro_exact_transition_df = pd.DataFrame(micro_exact_transition)
micro_exact_transition_df.to_csv(SUM_DIR / "micro_exact_transition_analysis.csv", index=False)
log_saved(SUM_DIR / "micro_exact_transition_analysis.csv")

# %% [markdown]
# # 11. F5 — Micro Bias Alignment
#
# Penjelasan bagian:
# F5 menurunkan total goal sedikit pada subset low-risk untuk meniru kecenderungan underprediction yang sebelumnya terlihat cukup kuat. Perubahan dibatasi maksimal 1–5% match.
#
# Output yang perlu dilihat:
# Lihat `micro_bias_grid.csv`. Mean total hanya boleh turun kecil, pair consistency harus True, dan changed rate harus sesuai cap.

# %%
BIAS_TRANSITIONS = {
    (3, 2): (2, 1),
    (2, 3): (1, 2),
    (3, 1): (2, 0),
    (1, 3): (0, 2),
    (2, 2): (1, 1),
    (2, 1): (1, 0),
    (1, 2): (0, 1),
}


def apply_micro_bias(base_sub: pd.DataFrame, variant: str, cap_rate: float, require_same_gd: bool = False, non_high_weight: bool = False) -> pd.DataFrame:
    base_m = submission_to_match_df(base_sub, label="bias_base")
    candidates = []
    for _, r in base_m.iterrows():
        a, b = int(r["pred_a"]), int(r["pred_b"])
        if (a, b) not in BIAS_TRANSITIONS:
            continue
        na, nb = BIAS_TRANSITIONS[(a, b)]
        if outcome_scalar(a, b) != outcome_scalar(na, nb):
            continue
        if require_same_gd and (a - b) != (na - nb):
            continue
        if non_high_weight and float(r.get("tournament_weight", 1.2)) >= 1.8:
            continue
        if str(r.get("gender", "")).upper() == "W" and (a + b) >= 5:
            continue
        candidates.append({
            "match_id": r["match_id"],
            "new_a": na,
            "new_b": nb,
            "tournament_weight": float(r.get("tournament_weight", 1.2)),
            "old_total": a + b,
            "new_total": na + nb,
        })
    cand_df = pd.DataFrame(candidates)
    n_cap = int(math.floor(len(base_m) * cap_rate))
    n_cap = max(1, n_cap) if len(cand_df) > 0 and cap_rate > 0 else 0
    if len(cand_df) > 0:
        cand_df = cand_df.sort_values(["tournament_weight", "old_total"], ascending=[True, False]).head(n_cap)
    repl = cand_df.set_index("match_id").to_dict("index") if len(cand_df) else {}
    rows = []
    for _, r in base_m.iterrows():
        row = r.to_dict()
        if r["match_id"] in repl:
            row["pred_a"] = repl[r["match_id"]]["new_a"]
            row["pred_b"] = repl[r["match_id"]]["new_b"]
        rows.append(row)
    return match_df_to_submission(pd.DataFrame(rows))


F5_CONFIGS = [
    ("f5_low_total_bias_cap010", 0.010, False, False),
    ("f5_low_total_bias_cap020", 0.020, False, False),
    ("f5_same_gd_bias_cap020", 0.020, True, False),
    ("f5_non_high_weight_bias_cap030", 0.030, False, True),
    ("f5_non_high_weight_bias_cap050", 0.050, False, True),
]

f5_records = []
f5_submissions = {}
for var, cap, same_gd, non_high in F5_CONFIGS:
    sub = apply_micro_bias(base_submission, var, cap_rate=cap, require_same_gd=same_gd, non_high_weight=non_high)
    path, check = save_submission(sub, var, strict=False)
    rec = compare_to_base_match(sub, base_submission, var)
    rec.update({"variant": var, "cap_rate": cap, "require_same_gd": same_gd, "non_high_weight": non_high, "path": str(path), "pair_consistency": bool(check.loc[check["check"] == "pair_consistency", "passed"].iloc[0])})
    f5_records.append(rec)
    f5_submissions[var] = sub

micro_bias_grid = pd.DataFrame(f5_records).sort_values("changed_rate")
micro_bias_grid.to_csv(SUM_DIR / "micro_bias_grid.csv", index=False)
log_saved(SUM_DIR / "micro_bias_grid.csv")
display(micro_bias_grid)

micro_bias_transition = []
for label, sub in f5_submissions.items():
    cand = submission_to_match_df(sub, label=label)[["match_id", "pred_a", "pred_b"]].rename(columns={"pred_a": "cand_a", "pred_b": "cand_b"})
    b = base_match[["match_id", "pred_a", "pred_b", "gender", "tournament", "tournament_weight"]].rename(columns={"pred_a": "base_a", "pred_b": "base_b"})
    m = cand.merge(b, on="match_id", how="left", validate="one_to_one")
    m["changed"] = (m["cand_a"] != m["base_a"]) | (m["cand_b"] != m["base_b"])
    ch = m[m["changed"]].copy()
    if len(ch):
        ch["transition"] = ch.apply(lambda r: f"{int(r['base_a'])}-{int(r['base_b'])}->{int(r['cand_a'])}-{int(r['cand_b'])}", axis=1)
        for tr, g in ch.groupby("transition"):
            micro_bias_transition.append({"variant": label, "transition": tr, "count": len(g)})

micro_bias_transition_df = pd.DataFrame(micro_bias_transition)
micro_bias_transition_df.to_csv(SUM_DIR / "micro_bias_transition_analysis.csv", index=False)
log_saved(SUM_DIR / "micro_bias_transition_analysis.csv")

# %% [markdown]
# # 12. F6 — Optional True Native PMF Pair Consensus Status
#
# Penjelasan bagian:
# F6 hanya aktif jika PMF ordinal EXP15 tersedia di environment. Script post-processing ini biasanya tidak menyimpan PMF, jadi F6 akan skip dengan status jelas.
#
# Output yang perlu dilihat:
# `native_pmf_consensus_status.csv` harus menampilkan `active` atau `skipped_pmf_not_available`. Kalau skipped, itu bukan error.

# %%
pmf_vars = ["exp15_test_pmf", "test_pmf_team", "test_pmf_opp", "test_pred_team_raw", "test_pred_opp_raw"]
pmf_available = any(v in globals() for v in pmf_vars)
f6_status = pd.DataFrame([{
    "variant": "f6_true_native_pmf_pair_consensus",
    "status": "active_not_implemented_in_postprocessing_script" if pmf_available else "skipped_pmf_not_available",
    "available_vars": ",".join([v for v in pmf_vars if v in globals()]),
    "note": "EXP12F standalone post-processing tidak menyimpan PMF. Jalankan F1-F5 dulu; F6 bisa dibuat di EXP13 jika PMF diekspor.",
}])
f6_status.to_csv(SUM_DIR / "native_pmf_consensus_status.csv", index=False)
log_saved(SUM_DIR / "native_pmf_consensus_status.csv")
display(f6_status)

# %% [markdown]
# # 13. Variant Metrics, Scoreline Distribution, dan Changed Prediction Analysis
#
# Penjelasan bagian:
# Bagian ini menggabungkan semua kandidat F0-F5, menghitung sanity metric GT-free, scoreline distribution, changed rate, high-weight changes, dan menyimpan katalog submission.
#
# Output yang perlu dilihat:
# Cek `variant_metrics.csv`, `scoreline_distribution.csv`, dan `changed_prediction_analysis.csv`. Candidate yang terlalu banyak berubah atau scoreline collapse harus dicurigai sebelum audit lokal.

# %%
all_candidate_submissions = {"f0_exp12e_best_reproduction": base_submission}
all_candidate_submissions.update(f1_submissions)
all_candidate_submissions.update(f2_submissions)
all_candidate_submissions.update(f3_submissions)
all_candidate_submissions.update(f4_submissions)
all_candidate_submissions.update(f5_submissions)

variant_metrics_rows = []
scoreline_rows = []
changed_rows = []
submission_check_frames = []
submission_catalog_rows = []

for label, sub in all_candidate_submissions.items():
    pair_ok, n_bad, _ = pair_consistency_report(sub)
    m = submission_to_match_df(sub, label=label)
    rec = compare_to_base_match(sub, base_submission, label)
    top_counts = (m.assign(scoreline=lambda d: d["pred_a"].astype(int).astype(str) + "-" + d["pred_b"].astype(int).astype(str))["scoreline"].value_counts())
    rec.update({
        "pair_consistency": bool(pair_ok),
        "n_bad_pairs": int(n_bad),
        "top1_scoreline_share": float(top_counts.iloc[0] / len(m)) if len(top_counts) else 0.0,
        "top3_scoreline_share": float(top_counts.head(3).sum() / len(m)) if len(top_counts) else 0.0,
    })
    variant_metrics_rows.append(rec)
    for score, cnt in top_counts.head(15).items():
        scoreline_rows.append({"variant": label, "scoreline": score, "count": int(cnt), "share": float(cnt / len(m))})

    cand = m[["match_id", "pred_a", "pred_b", "gender", "tournament", "tournament_weight"]].rename(columns={"pred_a": "cand_a", "pred_b": "cand_b"})
    base = base_match[["match_id", "pred_a", "pred_b"]].rename(columns={"pred_a": "base_a", "pred_b": "base_b"})
    cm = cand.merge(base, on="match_id", how="left", validate="one_to_one")
    cm["changed"] = (cm["cand_a"] != cm["base_a"]) | (cm["cand_b"] != cm["base_b"])
    ch = cm[cm["changed"]].copy()
    if len(ch):
        ch["transition"] = ch.apply(lambda r: f"{int(r['base_a'])}-{int(r['base_b'])}->{int(r['cand_a'])}-{int(r['cand_b'])}", axis=1)
        for transition, g in ch.groupby("transition"):
            changed_rows.append({
                "variant": label,
                "transition": transition,
                "count": int(len(g)),
                "share_of_changed": float(len(g) / max(len(ch), 1)),
                "high_weight_count": int((g["tournament_weight"] >= 1.8).sum()),
                "w_count": int(g["gender"].astype(str).str.upper().eq("W").sum()),
            })

variant_metrics_df = pd.DataFrame(variant_metrics_rows).sort_values(["pair_consistency", "changed_rate", "top3_scoreline_share"], ascending=[False, True, True])
variant_metrics_df.to_csv(SUM_DIR / "variant_metrics.csv", index=False)
log_saved(SUM_DIR / "variant_metrics.csv")
display(variant_metrics_df)

scoreline_distribution_df = pd.DataFrame(scoreline_rows)
scoreline_distribution_df.to_csv(SUM_DIR / "scoreline_distribution.csv", index=False)
log_saved(SUM_DIR / "scoreline_distribution.csv")

changed_prediction_analysis_df = pd.DataFrame(changed_rows)
changed_prediction_analysis_df.to_csv(SUM_DIR / "changed_prediction_analysis.csv", index=False)
log_saved(SUM_DIR / "changed_prediction_analysis.csv")

# %% [markdown]
# # 14. F7 — Final Selected Safe dan Submission Catalog
#
# Penjelasan bagian:
# Main pipeline tidak memakai GT untuk memilih rule. Karena itu `best_safe` dipilih konservatif dari candidate yang pair-consistent dan changed-rate kecil. Ranking sebenarnya dibaca dari local GT audit di Section 99.
#
# Output yang perlu dilihat:
# Lihat `final_decision.csv` dan `submission_catalog.csv`. `selected_by_pipeline` bukan klaim best GT; semua kandidat tetap disimpan untuk audit lokal.

# %%
# Conservative pipeline selection: default F0. If a F1/F2 candidate is pair-consistent with small changed rate,
# choose the first safe consensus/router candidate as recommended. This remains GT-free.
safe_metrics = variant_metrics_df[(variant_metrics_df["pair_consistency"] == True) & (variant_metrics_df["changed_rate"] <= 0.05)].copy()
priority_prefixes = ["f2_conservative_consensus_only", "f2_exact_majority_router", "f1_high_weight_preserve_gd_else_exp12e", "f0_exp12e_best_reproduction"]
selected_label = "f0_exp12e_best_reproduction"
for pref in priority_prefixes:
    hit = safe_metrics[safe_metrics["label"].astype(str).str.startswith(pref)] if "label" in safe_metrics.columns else pd.DataFrame()
    if len(hit):
        selected_label = str(hit.iloc[0]["label"])
        break

# `compare_to_base_match` uses key label, so ensure label column exists.
if "label" not in variant_metrics_df.columns:
    variant_metrics_df["label"] = variant_metrics_df["label"] if "label" in variant_metrics_df else variant_metrics_df.index.astype(str)

if selected_label not in all_candidate_submissions:
    selected_label = "f0_exp12e_best_reproduction"

best_safe_submission = all_candidate_submissions[selected_label]
best_path, best_check = save_submission(best_safe_submission, "best_safe", strict=STRICT_FINAL)

# Also save explicit important filenames required by the experiment document if present.
# Already saved by save_submission for each variant above; catalog collects actual generated paths.
submission_catalog = []
for p in sorted(SUB_DIR.glob("submission_exp12f_*.csv")):
    submission_catalog.append({"file_name": p.name, "path": str(p)})
submission_catalog_df = pd.DataFrame(submission_catalog)
submission_catalog_df.to_csv(SUM_DIR / "submission_catalog.csv", index=False)
log_saved(SUM_DIR / "submission_catalog.csv")

all_checks = []
for p in sorted(SUB_DIR.glob("submission_exp12f_*.csv")):
    try:
        sub = pd.read_csv(p)
        _, chk = validate_submission(sub, p.stem, strict=False)
        all_checks.append(chk)
    except Exception as e:
        all_checks.append(pd.DataFrame([{"label": p.stem, "check": "load_validate", "passed": False, "detail": repr(e)}]))
submission_check_df = pd.concat(all_checks, ignore_index=True) if all_checks else pd.DataFrame()
submission_check_df.to_csv(SUM_DIR / "submission_check.csv", index=False)
log_saved(SUM_DIR / "submission_check.csv")

decision = {
    "experiment": "EXP12F",
    "selected_by_pipeline": selected_label,
    "selected_by_sanity": selected_label,
    "recommended_for_local_audit": "all generated submission_exp12f_*.csv, especially F1/F2/F3/F4/F5 candidates",
    "main_submission_path": str(best_path),
    "local_gt_audit_cell": "Section 99 only; not used for main pipeline selection",
    "notes": "best_safe is a GT-free safe candidate, not a claim of best local-audit score.",
}
with open(SUM_DIR / "final_decision.json", "w", encoding="utf-8") as f:
    json.dump(decision, f, indent=2)
final_decision_df = pd.DataFrame([{"key": k, "value": str(v)} for k, v in decision.items()])
final_decision_df.to_csv(SUM_DIR / "final_decision.csv", index=False)
log_saved(SUM_DIR / "final_decision.csv")
display(final_decision_df)
display(best_check)

# %% [markdown]
# # 99. OPTIONAL LOCAL GT AUDIT — DELETE BEFORE CLEAN SUBMISSION NOTEBOOK
#
# Penjelasan bagian:
# Cell ini membaca `ground_truth_bersih.csv` dan semua `submission_*.csv` yang sudah dibuat, lalu menghitung AW-MAE lokal. Cell ini hanya untuk audit lokal setelah semua submission selesai dibuat.
#
# Output yang perlu dilihat:
# Lihat ranking AW-MAE, base MAE, exact rate, outcome rate, GD rate, bias, mean total, dan max goal. Jangan gunakan cell ini untuk training, feature engineering, atau automatic selection di main pipeline.

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
        raise FileNotFoundError("ground_truth_bersih.csv not found in data/ or dataset/. Set EXP12F_RUN_LOCAL_GT_AUDIT=0 to skip.")

    audit_log(f"GT_PATH = {GT_PATH}")
    gt = pd.read_csv(GT_PATH)
    if "Id" not in gt.columns and "id" in gt.columns:
        gt = gt.rename(columns={"id": "Id"})
    required_gt_cols = ["Id", "team_goals", "opp_goals"]
    missing_gt_cols = [c for c in required_gt_cols if c not in gt.columns]
    if missing_gt_cols:
        raise KeyError(f"ground_truth_bersih.csv missing columns: {missing_gt_cols}")
    gt = gt[required_gt_cols].copy()
    gt["Id"] = gt["Id"].astype(str)
    gt = gt.rename(columns={"team_goals": "true_team_goals", "opp_goals": "true_opp_goals"})

    # Tournament weights from test if available.
    test_weight_df = test_raw[["Id"]].copy()
    test_weight_df["Id"] = test_weight_df["Id"].astype(str)
    if "tournament" in test_raw.columns:
        test_weight_df["weight"] = test_raw["tournament"].apply(get_tournament_weight).astype(float)
    else:
        test_weight_df["weight"] = 1.20
    gt = gt.merge(test_weight_df, on="Id", how="left")
    gt["weight"] = gt["weight"].fillna(1.20).astype(float)

    EXACT_PENALTY = 0.30
    OUTCOME_PENALTY = 0.25
    GD_PENALTY = 0.15
    WRONG_OUTCOME_MULTIPLIER = 1.50
    NONLINEAR_POWER = 1.50

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

    search_dirs = []
    for p in [
        SUB_DIR,
        OUT_DIR / "submissions",
        OUTPUT_ROOT,
        OUTPUT_ROOT / "exp12f_weighted_pair_router_micro_booster",
        OUTPUT_ROOT / "exp12e_pair_repair_micro_audit",
        OUTPUT_ROOT / "exp12d_exp22_alignment_hybrid",
        OUTPUT_ROOT / "exp12c_exp22_strength_rebase_hybrid",
        OUTPUT_ROOT / "exp12b_no_pseudo_pair_native_decoder_alignment",
    ]:
        p = Path(p)
        if p.exists():
            search_dirs.append(p)

    submission_paths = []
    for d in search_dirs:
        for p in d.rglob("*.csv"):
            name = p.name.lower()
            if not name.startswith("submission_"):
                continue
            if "sample" in name:
                continue
            submission_paths.append(p.resolve())
    submission_paths = sorted(set(submission_paths), key=lambda x: str(x))
    if not submission_paths:
        raise FileNotFoundError("No submission_*.csv files found for local audit.")
    audit_log(f"Found {len(submission_paths)} submission files")

    rows = []
    bad_rows = []
    for path in submission_paths:
        try:
            sub = pd.read_csv(path)
            sub = normalize_submission_columns(sub)
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
            pair_ok, n_bad, _ = pair_consistency_report(sub)
            metric.update({
                "file_name": path.name,
                "path": str(path),
                "n_rows": len(merged),
                "pair_consistency": bool(pair_ok),
                "n_bad_pairs": int(n_bad),
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
    log_info("RUN_LOCAL_GT_AUDIT=False, skipping optional local GT audit cell.")
