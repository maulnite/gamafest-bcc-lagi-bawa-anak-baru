# %% [markdown]
# # 00. EXP12J — 1-1 Draw-to-Win Booster Refinement
#
# Penjelasan bagian:
# EXP12J adalah post-processing refinement khusus transisi `1-1 draw-to-win` di atas EXP12I. Eksperimen ini tidak melatih model baru. Fokusnya adalah membuat `ordered_1_1_candidate_table.csv`, lalu menguji cap sweep khusus 1-1, direction split, support/reliability tuning, tournament/gender gating, marginal surgery, dan tiny exact/GD repair.
#
# Output yang perlu dilihat:
# Cek folder `PROJECT_ROOT/outputs/exp12j_one_one_draw_to_win_refinement/<variant>/`. Semua candidate CSV tersimpan di `submissions/`, semua summary di `summaries/`, dan local GT audit hanya ada di Section 99 paling akhir.

# %% [markdown]
# # 01. Setup, Project Root, Logging, dan Guardrail
#
# Penjelasan bagian:
# Bagian ini menyiapkan import, seed, logging, project root resolver, dan output folder. Project root resolver penting supaya output tidak masuk ke `notebook/outputs/`.
#
# Output yang perlu dilihat:
# Pastikan `PROJECT_ROOT`, `OUTPUT_ROOT`, `OUT_DIR`, `SUB_DIR`, dan `SUM_DIR` mengarah ke folder project utama, bukan folder `notebook`. Ground truth tidak dibaca di bagian ini.

# %%
import os
import json
import math
import random
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

try:
    from IPython.display import display
except Exception:
    def display(x):
        print(x)

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 220)
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
    if Path("/mnt/data").exists() and (Path("/mnt/data") / "test.csv").exists():
        return Path("/mnt/data")
    return cwd


PROJECT_ROOT = infer_project_root()
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
EXP12J_VARIANT = os.environ.get("EXP12J_VARIANT", "j1_one_one_cap_sweep")
OUT_DIR = OUTPUT_ROOT / "exp12j_one_one_draw_to_win_refinement" / EXP12J_VARIANT
PRED_DIR = OUT_DIR / "predictions"
SUB_DIR = OUT_DIR / "submissions"
SUM_DIR = OUT_DIR / "summaries"
FIG_DIR = OUT_DIR / "figures"
for d in [PRED_DIR, SUB_DIR, SUM_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RUN_LOCAL_GT_AUDIT_DEFAULT = os.environ.get("EXP12J_RUN_LOCAL_GT_AUDIT", "1").strip().lower() in {"1", "true", "yes", "y"}
STRICT_FINAL = True
MAX_GOAL_SANITY = 40

log_section("Setup")
log_info(f"PROJECT_ROOT = {PROJECT_ROOT}")
log_info(f"OUTPUT_ROOT = {OUTPUT_ROOT}")
log_info(f"OUT_DIR = {OUT_DIR}")
log_info(f"SUB_DIR = {SUB_DIR}")
log_info(f"SUM_DIR = {SUM_DIR}")
log_info(f"EXP12J_VARIANT = {EXP12J_VARIANT}")
log_info("GT guardrail: ground_truth_bersih.csv hanya dibaca di Section 99 optional local audit")

# %% [markdown]
# # 02. Robust Data Finder dan Basic Audit
#
# Penjelasan bagian:
# Bagian ini mencari `train.csv`, `test.csv`, dan sample submission secara robust. Finder sengaja menghindari folder `outputs/` agar tidak salah mengambil submission eksperimen sebagai sample.
#
# Output yang perlu dilihat:
# Pastikan `TRAIN_PATH`, `TEST_PATH`, dan `SAMPLE_PATH` mengarah ke data asli. `Id` harus tersedia setelah normalisasi kolom.

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

ID_COL = "Id" if "Id" in sample_submission.columns else ("id" if "id" in sample_submission.columns else sample_submission.columns[0])
TEST_ID_COL = "Id" if "Id" in test_raw.columns else ("id" if "id" in test_raw.columns else ID_COL)
MATCH_COL = "match_id" if "match_id" in test_raw.columns else None

if MATCH_COL is None:
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
# # 03. Submission Helpers, Match-Level Conversion, dan Pair Consistency
#
# Penjelasan bagian:
# Bagian ini menyiapkan helper alignment ke sample, validasi submission, konversi row-level ke match-level, dan pair consistency. Semua final candidate harus pair-consistent.
#
# Output yang perlu dilihat:
# Setiap candidate akan masuk `submission_check.csv`. Untuk final `best_safe`, semua check harus `True`, terutama `pair_consistency` dan `id_order_matches_sample`.

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
    cols = ["Id", MATCH_COL]
    if "tournament" in test_raw.columns:
        cols.append("tournament")
    if "gender" in test_raw.columns:
        cols.append("gender")
    tmp = test_raw[cols].copy()
    tmp["_sample_order"] = tmp["Id"].astype(str).map({v: i for i, v in enumerate(sample_submission["Id"].astype(str).tolist())})
    rows = []
    for match_id, g in tmp.groupby(MATCH_COL, sort=False):
        g = g.sort_values("_sample_order").reset_index(drop=True)
        if len(g) != 2:
            continue
        tournament = str(g.loc[0, "tournament"]) if "tournament" in g.columns else "__MISSING__"
        rows.append({
            "match_id": match_id,
            "row_id_a": str(g.loc[0, "Id"]),
            "row_id_b": str(g.loc[1, "Id"]),
            "gender": str(g.loc[0, "gender"]) if "gender" in g.columns else "__MISSING__",
            "tournament": tournament,
            "tournament_weight": get_tournament_weight(tournament),
            "sample_order_a": int(g.loc[0, "_sample_order"]),
            "sample_order_b": int(g.loc[1, "_sample_order"]),
        })
    out = pd.DataFrame(rows)
    if len(out) * 2 != len(test_raw):
        log_info(f"Pair frame has {len(out):,} matches for {len(test_raw):,} test rows; some groups may not have 2 rows")
    return out


pair_frame = build_test_pair_frame()
log_check("pair_frame non-empty", len(pair_frame) > 0, f"n_matches={len(pair_frame):,}")


def pair_consistency_report(submission_df: pd.DataFrame):
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


def validate_submission(submission_df: pd.DataFrame, label: str, strict: bool = False):
    aligned = align_to_sample(submission_df)
    pair_ok, n_bad, _ = pair_consistency_report(aligned)
    expected_shape_ok = len(aligned) == len(sample_submission)
    checks = [
        {"label": label, "check": "shape_matches_sample", "passed": bool(expected_shape_ok)},
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


def save_submission(submission_df: pd.DataFrame, label: str, strict: bool = False):
    aligned, check_df = validate_submission(submission_df, label, strict=strict)
    path = SUB_DIR / f"submission_exp12j_{label}.csv"
    aligned.to_csv(path, index=False)
    log_saved(path)
    return path, check_df

# %% [markdown]
# # 04. Load EXP12I Champion, Pre-Booster Base, dan Candidate Pool
#
# Penjelasan bagian:
# Bagian ini mencari EXP12I champion `submission_exp12i_i7_cap0290_only_1_1.csv` sebagai J0 dan mencari pre-booster base, idealnya EXP12E best. Submission lama hanya dibaca sebagai comparison/post-processing artifact, bukan fitur training.
#
# Output yang perlu dilihat:
# `j0_reproduction_summary.csv` harus menunjukkan J0 berasal dari `exp12i_i7_cap0290_only_1_1` atau tie fallback yang jelas. `candidate_pool_loaded.csv` harus berisi kandidat yang berhasil dimuat dan status pair consistency-nya.

# %%
SUMMARY_FILE_KEYWORDS = [
    "submission_catalog",
    "submission_check",
    "submission_validation",
    "submission_metrics",
    "submission_pair_consistency",
    "submission_scoreline_distribution",
    "submission_subgroup_metrics",
    "submission_summary",
]


def is_submission_candidate_path(p: Path) -> bool:
    name = p.name.lower()
    if not name.startswith("submission_"):
        return False
    if "sample" in name:
        return False
    if any(k in name for k in SUMMARY_FILE_KEYWORDS):
        return False
    return True


def submission_label_from_path(p: Path) -> str:
    stem = p.stem
    if stem.startswith("submission_"):
        stem = stem[len("submission_"):]
    return stem


def find_submission_paths_by_keywords(keyword_groups, max_results: int = 20):
    """
    Find submission files while preserving the priority order of keyword_groups.

    This is intentionally not sorted only by path length. For EXP12J, source priority
    matters: champion/pre-base lookup must prefer the first matching keyword group
    even when another lower-priority file has a shorter path.
    """
    search_roots = [
        OUTPUT_ROOT / "exp12j_one_one_draw_to_win_refinement",
        OUTPUT_ROOT / "exp12i_cap0290_marginal_band_surgery",
        OUTPUT_ROOT / "exp12h_outcome_ranking_damage_control",
        OUTPUT_ROOT / "exp12g_outcome_booster_refinement",
        OUTPUT_ROOT / "exp12f_weighted_pair_router_micro_booster",
        OUTPUT_ROOT / "exp12e_pair_repair_micro_audit",
        OUTPUT_ROOT / "exp12d_exp22_alignment_hybrid",
        OUTPUT_ROOT / "exp12c_exp22_strength_rebase_hybrid",
        OUTPUT_ROOT / "exp12b_no_pseudo_pair_native_decoder_alignment",
        Path("/mnt/data"),
    ]

    matched = []

    for root_idx, root in enumerate(search_roots):
        if not Path(root).exists():
            continue

        for p in Path(root).rglob("*.csv"):
            if not is_submission_candidate_path(p):
                continue

            name = p.name.lower()
            for group_idx, kws in enumerate(keyword_groups):
                if all(k.lower() in name for k in kws):
                    matched.append({
                        "path": p.resolve(),
                        "group_idx": group_idx,
                        "root_idx": root_idx,
                        "is_mnt": "/mnt/data" in str(p),
                        "path_len": len(str(p)),
                    })
                    break

    if not matched:
        return []

    df = pd.DataFrame(matched).drop_duplicates("path")
    df = df.sort_values(
        ["group_idx", "is_mnt", "root_idx", "path_len", "path"],
        ascending=[True, True, True, True, True],
    ).reset_index(drop=True)

    return [Path(x) for x in df["path"].head(max_results).tolist()]


def load_submission_path(path: Path, label: str = None):
    df = align_to_sample(pd.read_csv(path))
    lab = label or submission_label_from_path(path)
    return df, lab


# Champion J0 lookup.
champ_keyword_groups = [
    ["exp12i", "i7", "cap0290", "only_1_1"],
    ["exp12i", "i7", "cap0290", "only_0_0_1_1"],
    ["exp12i", "i7", "cap0290", "no_2_2"],
    ["exp12i", "i7", "cap0290", "no_0_0_and_2_2"],
]
champ_paths = find_submission_paths_by_keywords(champ_keyword_groups, max_results=10)
if not champ_paths:
    raise FileNotFoundError("Tidak menemukan EXP12I champion/tie candidate. Pastikan outputs EXP12I tersedia.")
champ_path = champ_paths[0]
champ_submission, champ_label = load_submission_path(champ_path)

# Pre-booster base lookup. Prefer EXP12E best / pre outcome booster base.
pre_base_keyword_groups = [
    ["exp12e", "e2", "consensus_outcome_agree_only"],
    ["exp12e", "repair_min_total_same_outcome"],
    ["exp12e", "repair_preserve_gd_lower_total"],
    ["exp12f", "f0", "exp12e_best_reproduction"],
    ["exp12f", "f3", "draw_to_win_cap020"],
    ["exp12i", "i0", "cap0290_reproduction"],
]
pre_paths = find_submission_paths_by_keywords(pre_base_keyword_groups, max_results=10)
if pre_paths:
    pre_path = pre_paths[0]
    base_submission, pre_label = load_submission_path(pre_path)
else:
    pre_path = champ_path
    base_submission = champ_submission.copy()
    pre_label = champ_label + "__fallback_as_base"
    log_info("WARNING: pre-booster base tidak ditemukan; fallback memakai champion sebagai base. Ordered 1-1 candidates mungkin berkurang.")

log_section("Source Candidates")
log_info(f"champ_path = {champ_path}")
log_info(f"champ_label = {champ_label}")
log_info(f"pre_base_path = {pre_path}")
log_info(f"pre_base_label = {pre_label}")

base_match = submission_to_match_df(base_submission, label="pre_booster_base")
champ_match = submission_to_match_df(champ_submission, label="champion")

j0_summary = pd.DataFrame([
    {"key": "champ_path", "value": str(champ_path)},
    {"key": "champ_label", "value": str(champ_label)},
    {"key": "pre_base_path", "value": str(pre_path)},
    {"key": "pre_base_label", "value": str(pre_label)},
    {"key": "n_matches", "value": str(len(base_match))},
    {"key": "champ_pair_consistency", "value": str(pair_consistency_report(champ_submission)[0])},
])
j0_summary.to_csv(SUM_DIR / "j0_reproduction_summary.csv", index=False)
log_saved(SUM_DIR / "j0_reproduction_summary.csv")
display(j0_summary)

# Candidate pool: all project submission artifacts. Deduplicate by path.
pool_roots = [
    OUTPUT_ROOT / "exp12i_cap0290_marginal_band_surgery",
    OUTPUT_ROOT / "exp12h_outcome_ranking_damage_control",
    OUTPUT_ROOT / "exp12g_outcome_booster_refinement",
    OUTPUT_ROOT / "exp12f_weighted_pair_router_micro_booster",
    OUTPUT_ROOT / "exp12e_pair_repair_micro_audit",
    OUTPUT_ROOT / "exp12d_exp22_alignment_hybrid",
    OUTPUT_ROOT / "exp12c_exp22_strength_rebase_hybrid",
    OUTPUT_ROOT / "exp12b_no_pseudo_pair_native_decoder_alignment",
]
all_paths = []
for root in pool_roots:
    if not Path(root).exists():
        continue
    for p in Path(root).rglob("*.csv"):
        if is_submission_candidate_path(p):
            all_paths.append(p.resolve())
all_paths = sorted(set(all_paths), key=lambda p: str(p))

candidate_match_pool = {}
candidate_load_rows = []
for p in all_paths:
    label = submission_label_from_path(p)
    try:
        sub = align_to_sample(pd.read_csv(p))
        pair_ok, n_bad, _ = pair_consistency_report(sub)
        m = submission_to_match_df(sub, label=label)
        candidate_match_pool[label] = m
        candidate_load_rows.append({
            "label": label,
            "path": str(p),
            "n_matches": len(m),
            "pair_consistency": bool(pair_ok),
            "n_bad_pairs": int(n_bad),
        })
    except Exception as e:
        candidate_load_rows.append({
            "label": label,
            "path": str(p),
            "n_matches": 0,
            "pair_consistency": False,
            "n_bad_pairs": -1,
            "error": repr(e),
        })

# Ensure source champion/base are in pool.
candidate_match_pool["__j0_champion__"] = champ_match
candidate_match_pool["__pre_booster_base__"] = base_match

candidate_pool_loaded = pd.DataFrame(candidate_load_rows)
candidate_pool_loaded.to_csv(SUM_DIR / "candidate_pool_loaded.csv", index=False)
log_saved(SUM_DIR / "candidate_pool_loaded.csv")
display(candidate_pool_loaded.head(50))

CANDIDATE_MATCH_POOL_INDEX = {
    lab: df.set_index("match_id", drop=False)
    for lab, df in candidate_match_pool.items()
    if isinstance(df, pd.DataFrame) and "match_id" in df.columns
}
log_info(f"candidate_match_pool size = {len(CANDIDATE_MATCH_POOL_INDEX):,}")

# %% [markdown]
# # 05. Candidate Reliability, Support, dan Booster Helpers
#
# Penjelasan bagian:
# Bagian ini membuat bobot reliabilitas kandidat, fungsi support, dan helper untuk menerapkan perubahan match-level. Cache `CANDIDATE_MATCH_POOL_INDEX` dipakai supaya tidak ada `set_index()` berulang di loop.
#
# Output yang perlu dilihat:
# Cek `candidate_reliability_map.csv` untuk memastikan kandidat kuat seperti EXP12I/EXP12H/EXP12G/EXP12F diberi bobot lebih tinggi daripada artifact lama atau raw EXP17.

# %%
def candidate_reliability_weight(label: str) -> float:
    s = str(label).lower()
    if "__j0_champion__" in s:
        return 3.2
    if "exp12i" in s and "only_1_1" in s:
        return 3.1
    if "exp12i" in s:
        return 2.8
    if "exp12h" in s and "cap0290" in s:
        return 2.7
    if "exp12g" in s and ("cap0275" in s or "cap029" in s):
        return 2.4
    if "exp12f" in s and "draw_to_win" in s:
        return 2.2
    if "exp12e" in s and ("consensus" in s or "repair" in s):
        return 2.0
    if "exp12d" in s and "best" in s:
        return 1.6
    if "exp17" in s:
        return 0.7
    if "advanced_base" in s:
        return 0.5
    if "exp12b" in s or "exp12c" in s:
        return 1.0
    return 1.0


candidate_reliability_map = pd.DataFrame([
    {"label": lab, "source_weight": candidate_reliability_weight(lab)}
    for lab in sorted(CANDIDATE_MATCH_POOL_INDEX.keys())
])
candidate_reliability_map.to_csv(SUM_DIR / "candidate_reliability_map.csv", index=False)
log_saved(SUM_DIR / "candidate_reliability_map.csv")
display(candidate_reliability_map.sort_values("source_weight", ascending=False).head(40))


def candidate_scores_for_match(match_id, base_a: int, base_b: int):
    scores = []
    for lab, idx in CANDIDATE_MATCH_POOL_INDEX.items():
        if lab == "__pre_booster_base__":
            continue
        if match_id not in idx.index:
            continue
        row = idx.loc[match_id]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        a = int(row["pred_a"])
        b = int(row["pred_b"])
        if a == int(base_a) and b == int(base_b):
            continue
        scores.append({
            "label": lab,
            "weight": candidate_reliability_weight(lab),
            "a": a,
            "b": b,
            "score": scoreline(a, b),
            "outcome": outcome_scalar(a, b),
            "total": a + b,
            "gd": a - b,
        })
    return scores


def is_world_asian_tournament(tournament: str) -> bool:
    t = str(tournament).lower()
    return ("world" in t) or ("asian" in t) or ("afc" in t)


def default_one_one_risk_filter(row, new_a: int, new_b: int) -> bool:
    old_a, old_b = int(row["pred_a"]), int(row["pred_b"])
    if scoreline(old_a, old_b) != "1-1":
        return False
    if outcome_scalar(new_a, new_b) == 0:
        return False
    if (new_a + new_b) < 1 or (new_a + new_b) > 3:
        return False
    if abs((new_a + new_b) - 2) > 1:
        return False
    if abs((new_a - new_b) - 0) > 2:
        return False
    return True


def direction_label(old_a: int, old_b: int, new_a: int, new_b: int) -> str:
    if int(old_a) == 1 and int(old_b) == 1 and int(new_a) == 1 and int(new_b) == 0:
        return "home_win_shift"
    if int(old_a) == 1 and int(old_b) == 1 and int(new_a) == 0 and int(new_b) == 1:
        return "away_win_shift"
    return "other"


def cap_to_n(cap_rate: float, n_matches: int, n_candidates: int) -> int:
    n = int(math.floor(float(n_matches) * float(cap_rate)))
    return max(0, min(n, int(n_candidates)))


def apply_selected_1_1_candidates(base_sub: pd.DataFrame, selected: pd.DataFrame, variant: str):
    base_m = submission_to_match_df(base_sub, label="apply_1_1_base")
    selected = selected.copy()
    if len(selected) == 0:
        selected["variant"] = variant
        return match_df_to_submission(base_m), selected
    repl = selected.set_index("match_id").to_dict("index")
    rows = []
    for _, r in base_m.iterrows():
        row = r.to_dict()
        if r["match_id"] in repl:
            row["pred_a"] = int(repl[r["match_id"]]["new_a"])
            row["pred_b"] = int(repl[r["match_id"]]["new_b"])
            row["variant"] = variant
        rows.append(row)
    selected["variant"] = variant
    return match_df_to_submission(pd.DataFrame(rows)), selected


ALL_CANDIDATE_SUBMISSIONS = {}
ALL_CHANGE_DETAILS = {}
ALL_VARIANT_METRICS = []
ALL_SUBMISSION_CHECKS = []
SUBMISSION_CATALOG = []


def compare_to_base(candidate_sub: pd.DataFrame, base_sub: pd.DataFrame, label: str) -> dict:
    cm = submission_to_match_df(candidate_sub, label=label)[["match_id", "pred_a", "pred_b"]].rename(columns={"pred_a": "cand_a", "pred_b": "cand_b"})
    bm = submission_to_match_df(base_sub, label="base")[["match_id", "pred_a", "pred_b", "gender", "tournament", "tournament_weight"]].rename(columns={"pred_a": "base_a", "pred_b": "base_b"})
    m = cm.merge(bm, on="match_id", how="left", validate="one_to_one")
    changed = (m["cand_a"] != m["base_a"]) | (m["cand_b"] != m["base_b"])
    cand_out = outcome_array(m["cand_a"], m["cand_b"])
    base_out = outcome_array(m["base_a"], m["base_b"])
    cand_gd = m["cand_a"] - m["cand_b"]
    base_gd = m["base_a"] - m["base_b"]
    top_scores = cm.assign(scoreline=cm["cand_a"].astype(int).astype(str) + "-" + cm["cand_b"].astype(int).astype(str))["scoreline"].value_counts(normalize=True)
    return {
        "variant": label,
        "n_matches": int(len(m)),
        "n_changed": int(changed.sum()),
        "changed_rate": float(changed.mean()) if len(m) else 0.0,
        "same_outcome_changed": int(((cand_out == base_out) & changed).sum()),
        "same_gd_changed": int(((cand_gd == base_gd) & changed).sum()),
        "changed_high_weight": int((changed & (m["tournament_weight"] >= 1.8)).sum()),
        "changed_low_weight": int((changed & (m["tournament_weight"] <= 0.96)).sum()),
        "changed_M": int((changed & m["gender"].astype(str).str.upper().eq("M")).sum()),
        "changed_W": int((changed & m["gender"].astype(str).str.upper().eq("W")).sum()),
        "mean_pred_total": float((cm["cand_a"] + cm["cand_b"]).mean()) if len(cm) else np.nan,
        "max_pred_goal": int(max(cm["cand_a"].max(), cm["cand_b"].max())) if len(cm) else -1,
        "top1_scoreline_share": float(top_scores.iloc[0]) if len(top_scores) else 0.0,
        "top3_scoreline_share": float(top_scores.head(3).sum()) if len(top_scores) else 0.0,
    }


def save_candidate(sub_df: pd.DataFrame, label: str, strict: bool = False):
    path, check = save_submission(sub_df, label, strict=strict)
    ALL_SUBMISSION_CHECKS.append(check)
    SUBMISSION_CATALOG.append({"label": label, "path": str(path)})
    return path, check


def register_candidate(label: str, sub_df: pd.DataFrame, changes_df: pd.DataFrame, strict: bool = False, base_for_compare: pd.DataFrame = None):
    if base_for_compare is None:
        base_for_compare = base_submission
    path, check = save_candidate(sub_df, label, strict=strict)
    rec = compare_to_base(sub_df, base_submission, label)
    rec.update({
        "n_selected": int(len(changes_df)) if isinstance(changes_df, pd.DataFrame) else 0,
        "path": str(path),
        "pair_consistency": bool(check.loc[check["check"] == "pair_consistency", "passed"].iloc[0]),
    })
    ALL_VARIANT_METRICS.append(rec)
    ALL_CANDIDATE_SUBMISSIONS[label] = sub_df
    ALL_CHANGE_DETAILS[label] = changes_df if isinstance(changes_df, pd.DataFrame) else pd.DataFrame()
    return rec


def transition_table(change_dict: dict) -> pd.DataFrame:
    rows = []
    for label, ch in change_dict.items():
        if ch is None or len(ch) == 0:
            continue
        tmp = ch.copy()
        if "transition" not in tmp.columns:
            tmp["transition"] = tmp["old_a"].astype(int).astype(str) + "-" + tmp["old_b"].astype(int).astype(str) + "->" + tmp["new_a"].astype(int).astype(str) + "-" + tmp["new_b"].astype(int).astype(str)
        for tr, g in tmp.groupby("transition"):
            rows.append({"variant": label, "transition": tr, "count": len(g)})
    return pd.DataFrame(rows)

# %% [markdown]
# # 06. J0 — Reproduce EXP12I Only-1-1 Champion
#
# Penjelasan bagian:
# J0 menyimpan ulang EXP12I champion/tie candidate sebagai anchor. Ini penting agar semua output EXP12J bisa dibandingkan dari titik awal yang jelas.
#
# Output yang perlu dilihat:
# Cek `submission_exp12j_j0_only_1_1_reproduction.csv` dan `j0_reproduction_summary.csv`. Pair consistency harus True dan base label harus mengandung EXP12I only-1-1 atau tie fallback.

# %%
j0_path, j0_check = save_candidate(champ_submission, "j0_only_1_1_reproduction", strict=False)
ALL_CANDIDATE_SUBMISSIONS["j0_only_1_1_reproduction"] = champ_submission
ALL_CHANGE_DETAILS["j0_only_1_1_reproduction"] = pd.DataFrame()
rec_j0 = compare_to_base(champ_submission, base_submission, "j0_only_1_1_reproduction")
rec_j0.update({"n_selected": int(rec_j0["n_changed"]), "path": str(j0_path), "pair_consistency": bool(j0_check.loc[j0_check["check"] == "pair_consistency", "passed"].iloc[0])})
ALL_VARIANT_METRICS.append(rec_j0)

j0_reproduction_summary = pd.DataFrame([{
    "label": "j0_only_1_1_reproduction",
    "source_label": champ_label,
    "source_path": str(champ_path),
    "pair_consistency": bool(rec_j0["pair_consistency"]),
    "n_changed_vs_pre_base": int(rec_j0["n_changed"]),
    "changed_rate_vs_pre_base": float(rec_j0["changed_rate"]),
    "mean_pred_total": float(rec_j0["mean_pred_total"]),
}])
j0_reproduction_summary.to_csv(SUM_DIR / "j0_reproduction_summary.csv", index=False)
log_saved(SUM_DIR / "j0_reproduction_summary.csv")
display(j0_reproduction_summary)

# %% [markdown]
# # 07. Ordered 1-1 Candidate Table
#
# Penjelasan bagian:
# Bagian ini membuat `ordered_1_1_candidate_table.csv`, yaitu urutan kandidat perubahan khusus `old_score = 1-1`. Tabel ini adalah inti EXP12J karena J1-J5 melakukan cap sweep, direction split, support/reliability tuning, dan marginal surgery dari ranking yang sama.
#
# Output yang perlu dilihat:
# Cek jumlah kandidat 1-1, direction split (`home_win_shift`, `away_win_shift`, `other`), support distribution, gender/tournament distribution, dan source label distribution.

# %%
def build_ordered_1_1_candidate_table(
    base_sub: pd.DataFrame,
    support_threshold=2,
    weighted_support_threshold=None,
    trusted_only: bool = False,
    exclude_exp17: bool = False,
    candidate_filter=None,
    rank_rule: str = "default",
    allowed_new_scores=None,
) -> pd.DataFrame:
    if allowed_new_scores is None:
        allowed_new_scores = {"1-0", "0-1"}
    base_m = submission_to_match_df(base_sub, label="ordered_1_1_base")
    all_rows = []
    for _, r in base_m.iterrows():
        base_a, base_b = int(r["pred_a"]), int(r["pred_b"])
        if scoreline(base_a, base_b) != "1-1":
            continue
        scores = candidate_scores_for_match(r["match_id"], base_a, base_b)
        if trusted_only:
            scores = [s for s in scores if s["weight"] >= 2.0]
        if exclude_exp17:
            scores = [s for s in scores if "exp17" not in s["label"].lower()]
        decisive = [s for s in scores if s["outcome"] != 0]
        if not decisive:
            continue
        by_score = defaultdict(list)
        for s in decisive:
            if allowed_new_scores is not None and s["score"] not in set(allowed_new_scores):
                continue
            by_score[s["score"]].append(s)
        if not by_score:
            continue
        for new_score, ss in by_score.items():
            sup = len(ss)
            wsup = float(sum(x["weight"] for x in ss))
            ok = wsup >= weighted_support_threshold if weighted_support_threshold is not None else sup >= support_threshold
            if not ok:
                continue
            # Choose representative source for this exact target score.
            chosen = sorted(ss, key=lambda x: (x["weight"], x["label"]), reverse=True)[0]
            new_a, new_b = int(chosen["a"]), int(chosen["b"])
            if not default_one_one_risk_filter(r, new_a, new_b):
                continue
            if candidate_filter is not None and not bool(candidate_filter(r, new_a, new_b)):
                continue
            tournament = str(r.get("tournament", "unknown"))
            tw = float(r.get("tournament_weight", 1.2))
            old_total = base_a + base_b
            new_total = new_a + new_b
            old_gd = base_a - base_b
            new_gd = new_a - new_b
            all_rows.append({
                "match_id": r["match_id"],
                "old_a": base_a,
                "old_b": base_b,
                "new_a": new_a,
                "new_b": new_b,
                "old_score": scoreline(base_a, base_b),
                "new_score": scoreline(new_a, new_b),
                "transition": f"{scoreline(base_a, base_b)}->{scoreline(new_a, new_b)}",
                "direction": direction_label(base_a, base_b, new_a, new_b),
                "support": float(sup),
                "weighted_support": float(wsup),
                "source_label": str(chosen["label"]),
                "source_weight": float(chosen["weight"]),
                "gender": str(r.get("gender", "unknown")).upper(),
                "tournament": tournament,
                "tournament_weight": tw,
                "old_total": old_total,
                "new_total": new_total,
                "total_delta": abs(new_total - old_total),
                "old_gd": old_gd,
                "new_gd": new_gd,
                "gd_delta": abs(new_gd - old_gd),
                "is_high_weight": bool(tw >= 1.8),
                "is_low_weight": bool(tw <= 0.96),
                "is_world_asian": bool(is_world_asian_tournament(tournament)),
            })
    df = pd.DataFrame(all_rows)
    if len(df) == 0:
        return df
    rr = str(rank_rule)
    if rr == "weighted_then_support":
        sort_cols, asc = ["weighted_support", "support", "source_weight", "tournament_weight", "total_delta", "gd_delta"], [False, False, False, True, True, True]
    elif rr == "support_then_weighted":
        sort_cols, asc = ["support", "weighted_support", "source_weight", "tournament_weight", "total_delta", "gd_delta"], [False, False, False, True, True, True]
    elif rr == "source_weight_then_support":
        sort_cols, asc = ["source_weight", "weighted_support", "support", "total_delta", "gd_delta"], [False, False, False, True, True]
    elif rr == "low_weight_first":
        sort_cols, asc = ["tournament_weight", "weighted_support", "support", "source_weight"], [True, False, False, False]
    else:
        sort_cols, asc = ["support", "weighted_support", "source_weight", "tournament_weight", "total_delta", "gd_delta"], [False, False, False, True, True, True]
    df = df.sort_values(sort_cols, ascending=asc).reset_index(drop=True)
    # One selected target per match for the ordered table, using the ranking rule.
    df = df.drop_duplicates("match_id", keep="first").reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df


ordered_1_1_table = build_ordered_1_1_candidate_table(base_submission, support_threshold=2)
if len(ordered_1_1_table) == 0:
    raise RuntimeError("ordered_1_1_candidate_table kosong. Candidate pool tidak mendukung booster 1-1.")
ordered_1_1_table.to_csv(SUM_DIR / "ordered_1_1_candidate_table.csv", index=False)
log_saved(SUM_DIR / "ordered_1_1_candidate_table.csv")
display(ordered_1_1_table.head(30))

one_one_candidate_summary_rows = []
one_one_candidate_summary_rows.append({"metric": "n_candidates", "value": len(ordered_1_1_table)})
for col in ["direction", "gender", "new_score", "source_label"]:
    vc = ordered_1_1_table[col].value_counts().head(15)
    for k, v in vc.items():
        one_one_candidate_summary_rows.append({"metric": f"{col}:{k}", "value": int(v)})
one_one_candidate_summary = pd.DataFrame(one_one_candidate_summary_rows)
one_one_candidate_summary.to_csv(SUM_DIR / "one_one_candidate_summary.csv", index=False)
log_saved(SUM_DIR / "one_one_candidate_summary.csv")
display(one_one_candidate_summary.head(50))

N_MATCHES = len(base_match)

def select_from_table(table: pd.DataFrame, cap_rate: float = None, n_select: int = None) -> pd.DataFrame:
    if n_select is None:
        n_select = cap_to_n(float(cap_rate), N_MATCHES, len(table))
    return table.head(int(n_select)).copy()


def save_ordered_variant(label: str, selected: pd.DataFrame, strict: bool = False):
    sub, changes = apply_selected_1_1_candidates(base_submission, selected, label)
    register_candidate(label, sub, changes, strict=strict)
    return sub, changes

# %% [markdown]
# # 08. J1 — 1-1 Cap Sweep
#
# Penjelasan bagian:
# J1 melakukan cap sweep khusus kandidat `old_score = 1-1`. Cap dihitung terhadap total match-level, sama seperti eksperimen post-processing sebelumnya.
#
# Output yang perlu dilihat:
# `one_one_cap_sweep_grid.csv`. Cek `cap_rate`, `n_selected`, direction split, changed_high_weight, changed_M/W, mean_pred_total, dan pair_consistency. Section 99 akan menentukan ranking AW-MAE lokal.

# %%
cap_sweep = [0.0100, 0.0125, 0.0150, 0.0175, 0.0200, 0.0225, 0.0250, 0.0275, 0.0290, 0.0300, 0.0325, 0.0350, 0.0400]
one_one_cap_records = []
for cap in cap_sweep:
    label = f"j1_only_1_1_cap{int(round(cap * 10000)):04d}"
    selected = select_from_table(ordered_1_1_table, cap_rate=cap)
    sub, changes = save_ordered_variant(label, selected)
    rec = compare_to_base(sub, base_submission, label)
    rec.update({
        "cap_rate": cap,
        "n_selected": int(len(selected)),
        "home_win_shift": int((selected["direction"] == "home_win_shift").sum()) if len(selected) else 0,
        "away_win_shift": int((selected["direction"] == "away_win_shift").sum()) if len(selected) else 0,
    })
    one_one_cap_records.append(rec)

one_one_cap_sweep_grid = pd.DataFrame(one_one_cap_records)
one_one_cap_sweep_grid.to_csv(SUM_DIR / "one_one_cap_sweep_grid.csv", index=False)
log_saved(SUM_DIR / "one_one_cap_sweep_grid.csv")
display(one_one_cap_sweep_grid)

# %% [markdown]
# # 09. J2 — Direction Split
#
# Penjelasan bagian:
# J2 memisahkan arah perubahan dari `1-1`: `1-1 -> 1-0` dan `1-1 -> 0-1`. Tujuannya melihat apakah salah satu arah lebih bersih daripada arah lain.
#
# Output yang perlu dilihat:
# `one_one_direction_grid.csv` dan `one_one_direction_summary.csv`. Cek jumlah selected per direction, changed_high_weight, dan pair consistency.

# %%
def build_direction_table(mode: str, cap_rate: float = 0.0290) -> pd.DataFrame:
    if mode == "only_1_0":
        table = ordered_1_1_table[ordered_1_1_table["new_score"].eq("1-0")].copy()
    elif mode == "only_0_1":
        table = ordered_1_1_table[ordered_1_1_table["new_score"].eq("0-1")].copy()
    elif mode == "direction_by_weighted_support":
        table = build_ordered_1_1_candidate_table(base_submission, support_threshold=2, rank_rule="weighted_then_support")
    elif mode == "direction_by_support":
        table = build_ordered_1_1_candidate_table(base_submission, support_threshold=2, rank_rule="support_then_weighted")
    elif mode == "balanced_direction":
        raw = ordered_1_1_table.copy()
        n_cap = cap_to_n(cap_rate, N_MATCHES, len(raw))
        max_per_dir = int(math.ceil(n_cap * 0.65))
        selected_parts = []
        counts = defaultdict(int)
        for _, row in raw.iterrows():
            d = row["direction"]
            if counts[d] >= max_per_dir:
                continue
            selected_parts.append(row)
            counts[d] += 1
            if len(selected_parts) >= n_cap:
                break
        return pd.DataFrame(selected_parts)
    else:
        table = ordered_1_1_table.copy()
    return select_from_table(table, cap_rate=cap_rate)


direction_configs = [
    ("j2_only_1_1_to_1_0_cap0290", "only_1_0"),
    ("j2_only_1_1_to_0_1_cap0290", "only_0_1"),
    ("j2_direction_by_support_cap0290", "direction_by_support"),
    ("j2_direction_by_weighted_support_cap0290", "direction_by_weighted_support"),
    ("j2_balanced_direction_cap0290", "balanced_direction"),
]
one_one_direction_records = []
for label, mode in direction_configs:
    selected = build_direction_table(mode, cap_rate=0.0290)
    sub, changes = save_ordered_variant(label, selected)
    rec = compare_to_base(sub, base_submission, label)
    rec.update({
        "direction_mode": mode,
        "n_selected": int(len(selected)),
        "home_win_shift": int((selected["direction"] == "home_win_shift").sum()) if len(selected) else 0,
        "away_win_shift": int((selected["direction"] == "away_win_shift").sum()) if len(selected) else 0,
    })
    one_one_direction_records.append(rec)

one_one_direction_grid = pd.DataFrame(one_one_direction_records)
one_one_direction_grid.to_csv(SUM_DIR / "one_one_direction_grid.csv", index=False)
log_saved(SUM_DIR / "one_one_direction_grid.csv")
display(one_one_direction_grid)

one_one_direction_summary = ordered_1_1_table.groupby(["direction", "new_score"]).agg(
    n=("match_id", "count"),
    mean_support=("support", "mean"),
    mean_weighted_support=("weighted_support", "mean"),
    high_weight=("is_high_weight", "sum"),
).reset_index()
one_one_direction_summary.to_csv(SUM_DIR / "one_one_direction_summary.csv", index=False)
log_saved(SUM_DIR / "one_one_direction_summary.csv")
display(one_one_direction_summary)

# %% [markdown]
# # 10. J3 — Support dan Reliability Tuning Khusus 1-1
#
# Penjelasan bagian:
# J3 menguji apakah kandidat 1-1 perlu support lebih kuat atau hanya source terpercaya. Ini mengurangi risiko kandidat noisy masuk ke cap0290.
#
# Output yang perlu dilihat:
# `one_one_support_reliability_grid.csv`. Cek support threshold, weighted support threshold, n_selected, source distribution, dan pair consistency.

# %%
support_configs = [
    {"label": "j3_support2_only_1_1_cap0290", "support": 2, "weighted": None, "trusted": False, "exclude_exp17": False},
    {"label": "j3_support3_only_1_1_cap0290", "support": 3, "weighted": None, "trusted": False, "exclude_exp17": False},
    {"label": "j3_support4_only_1_1_cap0290", "support": 4, "weighted": None, "trusted": False, "exclude_exp17": False},
    {"label": "j3_weighted_support_3p0_only_1_1_cap0290", "support": 2, "weighted": 3.0, "trusted": False, "exclude_exp17": False},
    {"label": "j3_weighted_support_4p0_only_1_1_cap0290", "support": 2, "weighted": 4.0, "trusted": False, "exclude_exp17": False},
    {"label": "j3_trusted_source_only_1_1_cap0290", "support": 2, "weighted": None, "trusted": True, "exclude_exp17": False},
    {"label": "j3_exclude_exp17_only_1_1_cap0290", "support": 2, "weighted": None, "trusted": False, "exclude_exp17": True},
]
one_one_support_records = []
for cfg in support_configs:
    table = build_ordered_1_1_candidate_table(
        base_submission,
        support_threshold=cfg["support"],
        weighted_support_threshold=cfg["weighted"],
        trusted_only=cfg["trusted"],
        exclude_exp17=cfg["exclude_exp17"],
    )
    selected = select_from_table(table, cap_rate=0.0290) if len(table) else table
    sub, changes = save_ordered_variant(cfg["label"], selected)
    rec = compare_to_base(sub, base_submission, cfg["label"])
    rec.update({
        "support_threshold": cfg["support"],
        "weighted_support_threshold": cfg["weighted"],
        "trusted_only": cfg["trusted"],
        "exclude_exp17": cfg["exclude_exp17"],
        "n_selected": int(len(selected)),
    })
    one_one_support_records.append(rec)

one_one_support_reliability_grid = pd.DataFrame(one_one_support_records)
one_one_support_reliability_grid.to_csv(SUM_DIR / "one_one_support_reliability_grid.csv", index=False)
log_saved(SUM_DIR / "one_one_support_reliability_grid.csv")
display(one_one_support_reliability_grid)

one_one_support_reliability_summary = ordered_1_1_table.groupby(["support", "source_label"]).agg(n=("match_id", "count"), weighted_support_mean=("weighted_support", "mean")).reset_index()
one_one_support_reliability_summary.to_csv(SUM_DIR / "one_one_support_reliability_summary.csv", index=False)
log_saved(SUM_DIR / "one_one_support_reliability_summary.csv")

# %% [markdown]
# # 11. J4 — Tournament dan Gender Gating Khusus 1-1
#
# Penjelasan bagian:
# J4 menguji apakah booster 1-1 aman di semua domain atau hanya di subset tertentu seperti M-only, no-high-weight, atau world/asian strict.
#
# Output yang perlu dilihat:
# `one_one_tournament_gender_grid.csv` dan `one_one_tournament_gender_breakdown.csv`. Cek changed_M/W, changed_high_weight, changed_world_asian, changed_friendly, dan pair consistency.

# %%
def table_with_filter(filter_func, support_threshold=2, cap_rate=0.0290):
    table = build_ordered_1_1_candidate_table(base_submission, support_threshold=support_threshold)
    if len(table) == 0:
        return table
    table = table[table.apply(filter_func, axis=1)].copy()
    return select_from_table(table, cap_rate=cap_rate)


def high_weight_support3_filter(row):
    if bool(row["is_high_weight"]):
        return float(row["support"]) >= 3
    return True


def world_asian_strict_filter(row):
    if bool(row["is_world_asian"]):
        return float(row["support"]) >= 3 and float(row["weighted_support"]) >= 4.0
    return True


gating_configs = [
    ("j4_M_only_1_1_cap0290", lambda r: str(r["gender"]).upper() == "M", 0.0290),
    ("j4_W_only_1_1_cap0100", lambda r: str(r["gender"]).upper() == "W", 0.0100),
    ("j4_M_cap0290_W_cap0000", lambda r: str(r["gender"]).upper() == "M", 0.0290),
    ("j4_M_cap0290_W_cap0050", lambda r: (str(r["gender"]).upper() == "M") or (str(r["gender"]).upper() == "W" and int(r["rank"]) <= cap_to_n(0.0050, N_MATCHES, len(ordered_1_1_table))), 0.0290),
    ("j4_high_weight_support3_1_1_cap0290", high_weight_support3_filter, 0.0290),
    ("j4_no_high_weight_1_1_cap0290", lambda r: not bool(r["is_high_weight"]), 0.0290),
    ("j4_friendly_only_1_1_cap0290", lambda r: float(r["tournament_weight"]) <= 0.96, 0.0290),
    ("j4_world_asian_strict_1_1_cap0290", world_asian_strict_filter, 0.0290),
]

gating_records = []
for label, filt, cap in gating_configs:
    selected = table_with_filter(filt, cap_rate=cap)
    sub, changes = save_ordered_variant(label, selected)
    rec = compare_to_base(sub, base_submission, label)
    rec.update({"cap_rate": cap, "n_selected": int(len(selected))})
    gating_records.append(rec)

one_one_tournament_gender_grid = pd.DataFrame(gating_records)
one_one_tournament_gender_grid.to_csv(SUM_DIR / "one_one_tournament_gender_grid.csv", index=False)
log_saved(SUM_DIR / "one_one_tournament_gender_grid.csv")
display(one_one_tournament_gender_grid)

one_one_tournament_gender_breakdown = ordered_1_1_table.groupby(["gender", "is_high_weight", "is_low_weight", "is_world_asian"]).agg(n=("match_id", "count"), mean_support=("support", "mean")).reset_index()
one_one_tournament_gender_breakdown.to_csv(SUM_DIR / "one_one_tournament_gender_breakdown.csv", index=False)
log_saved(SUM_DIR / "one_one_tournament_gender_breakdown.csv")

# %% [markdown]
# # 12. J5 — Marginal Surgery Khusus 1-1
#
# Penjelasan bagian:
# J5 melakukan band surgery khusus ordered 1-1 candidate. Variant `minus_last` menghapus tail dari cap0290, sedangkan `plus_next` menambah kandidat setelah cap0275.
#
# Output yang perlu dilihat:
# `one_one_marginal_surgery_grid.csv` dan `one_one_marginal_changes.csv`. Cek n_added/n_removed, rank range, direction marginal candidates, gender/tournament marginal candidates, dan pair consistency.

# %%
N_0275 = cap_to_n(0.0275, N_MATCHES, len(ordered_1_1_table))
N_0290 = cap_to_n(0.0290, N_MATCHES, len(ordered_1_1_table))
N_0300 = cap_to_n(0.0300, N_MATCHES, len(ordered_1_1_table))

marginal_records = []
marginal_change_frames = []

# cap0290 minus last K
for k in [2, 4, 6, 8]:
    label = f"j5_1_1_cap0290_minus_last_{k:02d}"
    n = max(0, N_0290 - k)
    selected = ordered_1_1_table.head(n).copy()
    removed = ordered_1_1_table.iloc[n:N_0290].copy()
    sub, changes = save_ordered_variant(label, selected)
    rec = compare_to_base(sub, base_submission, label)
    rec.update({"mode": "cap0290_minus_last", "n_removed": k, "n_selected": int(len(selected)), "removed_rank_start": int(n + 1), "removed_rank_end": int(N_0290)})
    marginal_records.append(rec)
    if len(removed):
        tmp = removed.copy(); tmp["variant"] = label; tmp["operation"] = "removed"; marginal_change_frames.append(tmp)

# cap0275 plus next K
for k in [2, 4, 6]:
    label = f"j5_1_1_cap0275_plus_next_{k:02d}"
    n = min(len(ordered_1_1_table), N_0275 + k)
    selected = ordered_1_1_table.head(n).copy()
    added = ordered_1_1_table.iloc[N_0275:n].copy()
    sub, changes = save_ordered_variant(label, selected)
    rec = compare_to_base(sub, base_submission, label)
    rec.update({"mode": "cap0275_plus_next", "n_added": k, "n_selected": int(len(selected)), "added_rank_start": int(N_0275 + 1), "added_rank_end": int(n)})
    marginal_records.append(rec)
    if len(added):
        tmp = added.copy(); tmp["variant"] = label; tmp["operation"] = "added"; marginal_change_frames.append(tmp)

# cap0300 minus last K
for k in [2, 4]:
    label = f"j5_1_1_cap0300_minus_last_{k:02d}"
    n = max(0, N_0300 - k)
    selected = ordered_1_1_table.head(n).copy()
    removed = ordered_1_1_table.iloc[n:N_0300].copy()
    sub, changes = save_ordered_variant(label, selected)
    rec = compare_to_base(sub, base_submission, label)
    rec.update({"mode": "cap0300_minus_last", "n_removed": k, "n_selected": int(len(selected)), "removed_rank_start": int(n + 1), "removed_rank_end": int(N_0300)})
    marginal_records.append(rec)
    if len(removed):
        tmp = removed.copy(); tmp["variant"] = label; tmp["operation"] = "removed"; marginal_change_frames.append(tmp)

one_one_marginal_surgery_grid = pd.DataFrame(marginal_records)
one_one_marginal_surgery_grid.to_csv(SUM_DIR / "one_one_marginal_surgery_grid.csv", index=False)
log_saved(SUM_DIR / "one_one_marginal_surgery_grid.csv")
display(one_one_marginal_surgery_grid)

one_one_marginal_changes = pd.concat(marginal_change_frames, ignore_index=True) if marginal_change_frames else pd.DataFrame()
one_one_marginal_changes.to_csv(SUM_DIR / "one_one_marginal_changes.csv", index=False)
log_saved(SUM_DIR / "one_one_marginal_changes.csv")

# %% [markdown]
# # 13. J6 — Tiny Exact/GD Repair After 1-1 Booster
#
# Penjelasan bagian:
# J6 optional menambahkan exact/GD repair sangat kecil setelah booster 1-1. Perubahan tidak boleh menyentuh match yang sudah diubah oleh booster 1-1.
#
# Output yang perlu dilihat:
# `one_one_tiny_repair_grid.csv` dan `one_one_tiny_repair_transition.csv`. Jika repair memperburuk audit lokal, jangan lanjut repair besar.

# %%
def get_supported_alternative(row, mode="exact_same_outcome", support_threshold=2, weighted_support_threshold=None):
    base_a, base_b = int(row["pred_a"]), int(row["pred_b"])
    base_out = outcome_scalar(base_a, base_b)
    base_total = base_a + base_b
    base_gd = base_a - base_b
    scores = candidate_scores_for_match(row["match_id"], base_a, base_b)
    if not scores:
        return None
    grouped = defaultdict(list)
    for s in scores:
        if mode == "exact_same_outcome":
            if s["outcome"] != base_out:
                continue
            if abs(s["total"] - base_total) > 1 or abs(s["gd"] - base_gd) > 1:
                continue
        elif mode == "gd_repair":
            if s["outcome"] != base_out:
                continue
            if abs(s["total"] - base_total) > 1 or abs(s["gd"] - base_gd) > 1:
                continue
        else:
            continue
        grouped[(s["a"], s["b"])].append(s)
    best = None
    best_key = None
    for (a, b), ss in grouped.items():
        sup = len(ss)
        wsup = float(sum(x["weight"] for x in ss))
        ok = wsup >= weighted_support_threshold if weighted_support_threshold is not None else sup >= support_threshold
        if not ok:
            continue
        key = (wsup, sup, -abs((a + b) - base_total), -abs((a - b) - base_gd))
        if best_key is None or key > best_key:
            best_key = key
            best = (int(a), int(b), float(wsup), sorted(ss, key=lambda x: x["weight"], reverse=True)[0]["label"])
    return best


def apply_tiny_repair(base_after_booster_sub: pd.DataFrame, original_base_sub: pd.DataFrame, label: str, cap_rate: float, mode: str = "exact"):
    base_m = submission_to_match_df(base_after_booster_sub, label="repair_base")
    orig_m = submission_to_match_df(original_base_sub, label="orig_base")
    merged0 = base_m[["match_id", "pred_a", "pred_b"]].merge(
        orig_m[["match_id", "pred_a", "pred_b"]].rename(columns={"pred_a": "orig_a", "pred_b": "orig_b"}),
        on="match_id", how="left", validate="one_to_one"
    )
    already_changed = set(merged0.loc[(merged0["pred_a"] != merged0["orig_a"]) | (merged0["pred_b"] != merged0["orig_b"]), "match_id"])
    rows = []
    for _, r in base_m.iterrows():
        if r["match_id"] in already_changed:
            continue
        alt = get_supported_alternative(r, mode="gd_repair" if mode == "gd" else "exact_same_outcome", support_threshold=2)
        if alt is None:
            continue
        a, b, support, source_label = alt
        if int(a) == int(r["pred_a"]) and int(b) == int(r["pred_b"]):
            continue
        rows.append({
            "match_id": r["match_id"],
            "old_a": int(r["pred_a"]),
            "old_b": int(r["pred_b"]),
            "new_a": int(a),
            "new_b": int(b),
            "old_score": scoreline(r["pred_a"], r["pred_b"]),
            "new_score": scoreline(a, b),
            "transition": f"{scoreline(r['pred_a'], r['pred_b'])}->{scoreline(a, b)}",
            "support": support,
            "source_label": source_label,
            "gender": str(r.get("gender", "unknown")).upper(),
            "tournament_weight": float(r.get("tournament_weight", 1.2)),
        })
    cand = pd.DataFrame(rows)
    if len(cand) == 0:
        return base_after_booster_sub, cand
    cand = cand.sort_values(["support", "tournament_weight"], ascending=[False, True])
    selected = cand.head(cap_to_n(cap_rate, len(base_m), len(cand))).copy()
    base_rows = []
    repl = selected.set_index("match_id").to_dict("index") if len(selected) else {}
    for _, r in base_m.iterrows():
        row = r.to_dict()
        if r["match_id"] in repl:
            row["pred_a"] = int(repl[r["match_id"]]["new_a"])
            row["pred_b"] = int(repl[r["match_id"]]["new_b"])
            row["variant"] = label
        base_rows.append(row)
    return match_df_to_submission(pd.DataFrame(base_rows)), selected

# Use J1 cap0290 as default repair base when available.
repair_base_label = "j1_only_1_1_cap0290"
repair_base_sub = ALL_CANDIDATE_SUBMISSIONS.get(repair_base_label, champ_submission)
repair_records = []
for mode, caps in [("exact", [0.0010, 0.0025, 0.0050]), ("gd", [0.0010, 0.0025])]:
    for cap in caps:
        label = f"j6_1_1_plus_{'gdrepair' if mode == 'gd' else 'exact'}_cap{int(round(cap * 10000)):04d}"
        sub, changes = apply_tiny_repair(repair_base_sub, base_submission, label, cap_rate=cap, mode=mode)
        register_candidate(label, sub, changes)
        rec = compare_to_base(sub, base_submission, label)
        rec.update({"repair_mode": mode, "cap_rate": cap, "n_selected": int(len(changes))})
        repair_records.append(rec)

one_one_tiny_repair_grid = pd.DataFrame(repair_records)
one_one_tiny_repair_grid.to_csv(SUM_DIR / "one_one_tiny_repair_grid.csv", index=False)
log_saved(SUM_DIR / "one_one_tiny_repair_grid.csv")
display(one_one_tiny_repair_grid)

one_one_tiny_repair_transition = transition_table({k: v for k, v in ALL_CHANGE_DETAILS.items() if k.startswith("j6_")})
one_one_tiny_repair_transition.to_csv(SUM_DIR / "one_one_tiny_repair_transition.csv", index=False)
log_saved(SUM_DIR / "one_one_tiny_repair_transition.csv")

# %% [markdown]
# # 14. Summary Metrics, Scoreline Distribution, dan Changed Prediction Analysis
#
# Penjelasan bagian:
# Bagian ini menggabungkan semua metrik GT-free, scoreline distribution, dan transition summary. Ini membantu membaca sanity sebelum Section 99 local audit.
#
# Output yang perlu dilihat:
# `variant_metrics.csv`, `changed_prediction_analysis.csv`, dan `scoreline_distribution.csv`. Perhatikan changed_rate, top scoreline share, changed_high_weight, dan pair_consistency.

# %%
variant_metrics_df = pd.DataFrame(ALL_VARIANT_METRICS).drop_duplicates("variant", keep="last")
variant_metrics_df = variant_metrics_df.sort_values(["pair_consistency", "changed_rate"], ascending=[False, True]).reset_index(drop=True)
variant_metrics_df.to_csv(SUM_DIR / "variant_metrics.csv", index=False)
variant_metrics_df.to_csv(SUM_DIR / "changed_prediction_analysis.csv", index=False)
log_saved(SUM_DIR / "variant_metrics.csv")
log_saved(SUM_DIR / "changed_prediction_analysis.csv")
display(variant_metrics_df.head(100))

score_rows = []
for lab, sub in ALL_CANDIDATE_SUBMISSIONS.items():
    m = submission_to_match_df(sub, lab)
    vc = m.assign(scoreline=m["pred_a"].astype(int).astype(str) + "-" + m["pred_b"].astype(int).astype(str))["scoreline"].value_counts().head(20)
    for sc, cnt in vc.items():
        score_rows.append({"variant": lab, "scoreline": sc, "count": int(cnt), "share": float(cnt / len(m)) if len(m) else 0.0})
scoreline_distribution = pd.DataFrame(score_rows)
scoreline_distribution.to_csv(SUM_DIR / "scoreline_distribution.csv", index=False)
log_saved(SUM_DIR / "scoreline_distribution.csv")

transition_all = transition_table(ALL_CHANGE_DETAILS)
transition_all.to_csv(SUM_DIR / "transition_all_variants.csv", index=False)
log_saved(SUM_DIR / "transition_all_variants.csv")

# %% [markdown]
# # 15. J7 — Final Selected Safe
#
# Penjelasan bagian:
# Bagian ini memilih `best_safe` secara GT-free. Default safe order memprioritaskan J1 cap0290, direction/reliability variants, marginal minus tail, lalu J0 reproduction.
#
# Output yang perlu dilihat:
# `final_decision.csv`, `submission_catalog.csv`, dan `submission_check.csv`. `best_safe` bukan klaim best local audit; champion dibaca dari Section 99.

# %%
preferred_order = [
    "j1_only_1_1_cap0290",
    "j2_direction_by_weighted_support_cap0290",
    "j3_support3_only_1_1_cap0290",
    "j5_1_1_cap0290_minus_last_02",
    "j0_only_1_1_reproduction",
]
selected_label = None
for lab in preferred_order:
    if lab in ALL_CANDIDATE_SUBMISSIONS:
        pair_ok, n_bad, _ = pair_consistency_report(ALL_CANDIDATE_SUBMISSIONS[lab])
        cr = compare_to_base(ALL_CANDIDATE_SUBMISSIONS[lab], base_submission, lab)["changed_rate"]
        if pair_ok and cr <= 0.07:
            selected_label = lab
            break
if selected_label is None:
    selected_label = "j0_only_1_1_reproduction"

best_path, best_check = save_candidate(ALL_CANDIDATE_SUBMISSIONS[selected_label], "best_safe", strict=True)

submission_check_df = pd.concat(ALL_SUBMISSION_CHECKS, ignore_index=True) if ALL_SUBMISSION_CHECKS else pd.DataFrame()
submission_check_df.to_csv(SUM_DIR / "submission_check.csv", index=False)
log_saved(SUM_DIR / "submission_check.csv")

submission_catalog_df = pd.DataFrame(SUBMISSION_CATALOG).drop_duplicates("path") if SUBMISSION_CATALOG else pd.DataFrame(columns=["label", "path"])
submission_catalog_df.to_csv(SUM_DIR / "submission_catalog.csv", index=False)
log_saved(SUM_DIR / "submission_catalog.csv")

final_decision = {
    "experiment": "EXP12J",
    "selected_by_pipeline": selected_label,
    "selected_by_sanity": selected_label,
    "recommended_for_local_audit": "all generated submission_exp12j_*.csv; champion is decided manually from Section 99 local audit",
    "main_submission_path": str(best_path),
    "j0_base_label": str(champ_label),
    "pre_booster_base_label": str(pre_label),
    "notes": "GT-free best_safe. Local champion is read from local_gt_audit_awmae_rank.csv.",
}
with open(SUM_DIR / "final_decision.json", "w", encoding="utf-8") as f:
    json.dump(final_decision, f, indent=2)
final_decision_df = pd.DataFrame([{"key": k, "value": str(v)} for k, v in final_decision.items()])
final_decision_df.to_csv(SUM_DIR / "final_decision.csv", index=False)
log_saved(SUM_DIR / "final_decision.csv")
display(final_decision_df)

# %% [markdown]
# # 99. OPTIONAL LOCAL GT AUDIT — DELETE BEFORE CLEAN SUBMISSION NOTEBOOK
#
# Penjelasan bagian:
# Cell paling akhir ini membaca `ground_truth_bersih.csv` dan semua `submission_*.csv` kandidat, lalu menghitung AW-MAE weighted local audit. Cell ini hanya untuk audit setelah semua submission dibuat.
#
# Output yang perlu dilihat:
# `local_gt_audit_awmae_rank.csv` menampilkan ranking AW-MAE, base MAE, exact rate, outcome rate, GD rate, bias, pair consistency, dan path file. Jangan gunakan cell ini untuk training atau rule selection otomatis.

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
        log_info("ground_truth_bersih.csv tidak ditemukan. Skip local GT audit.")
    else:
        audit_log(f"GT_PATH = {GT_PATH}")
        gt = pd.read_csv(GT_PATH)
        gt_id_col = "Id" if "Id" in gt.columns else ("id" if "id" in gt.columns else gt.columns[0])
        missing_gt = [c for c in [gt_id_col, "team_goals", "opp_goals"] if c not in gt.columns]
        if missing_gt:
            raise KeyError(f"ground_truth_bersih.csv missing columns: {missing_gt}")

        gt = gt[[gt_id_col, "team_goals", "opp_goals"]].rename(columns={
            gt_id_col: "Id",
            "team_goals": "true_team_goals",
            "opp_goals": "true_opp_goals",
        }).copy()
        gt["Id"] = gt["Id"].astype(str)

        weight_df = test_raw[["Id"]].copy()
        weight_df["Id"] = weight_df["Id"].astype(str)
        weight_df["weight"] = test_raw["tournament"].apply(get_tournament_weight).astype(float) if "tournament" in test_raw.columns else 1.20
        gt = gt.merge(weight_df, on="Id", how="left")
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
            unweighted_loss = (raw * mult) ** NONLINEAR_POWER
            weighted_loss = unweighted_loss * weights

            return {
                "awmae": float(weighted_loss.sum() / weights.sum()),
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

        search_dirs = [
            SUB_DIR,
            OUT_DIR / "submissions",
            OUTPUT_ROOT / "exp12j_one_one_draw_to_win_refinement",
            OUTPUT_ROOT / "exp12i_cap0290_marginal_band_surgery",
            OUTPUT_ROOT / "exp12h_outcome_ranking_damage_control",
            OUTPUT_ROOT / "exp12g_outcome_booster_refinement",
            OUTPUT_ROOT / "exp12f_weighted_pair_router_micro_booster",
            OUTPUT_ROOT / "exp12e_pair_repair_micro_audit",
            OUTPUT_ROOT / "exp12d_exp22_alignment_hybrid",
            OUTPUT_ROOT / "exp12c_exp22_strength_rebase_hybrid",
            OUTPUT_ROOT / "exp12b_no_pseudo_pair_native_decoder_alignment",
        ]
        submission_paths = []
        for d in search_dirs:
            if not Path(d).exists():
                continue
            for p in Path(d).rglob("*.csv"):
                if is_submission_candidate_path(p):
                    submission_paths.append(p.resolve())
        submission_paths = sorted(set(submission_paths), key=lambda x: str(x))
        audit_log(f"Found {len(submission_paths)} submission files")

        rows, bad_rows = [], []
        for path in submission_paths:
            try:
                sub = align_to_sample(pd.read_csv(path))
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
                    "pair_consistency": bool(pair_ok),
                    "n_bad_pairs": int(n_bad),
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
