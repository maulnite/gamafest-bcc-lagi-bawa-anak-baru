# %% [markdown]
# # 00. EXP12K — Champion Delta Surgery
#
# Penjelasan bagian:
# EXP12K adalah post-processing surgery di atas champion EXP12K/EXP12I. Eksperimen ini tidak melatih model baru. Fokusnya adalah membuat `champion_delta_table.csv`, lalu menguji transition undo, segment/domain undo, tail pruning, safe add-back, direction-specific undo, dan tiny repair.
#
# Output yang perlu dilihat:
# Cek folder `PROJECT_ROOT/outputs/exp12k_champion_delta_surgery/<variant>/`. Semua candidate CSV tersimpan di `submissions/`, semua summary di `summaries/`, dan local GT audit hanya ada di Section 99 paling akhir.

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
EXP12K_VARIANT = os.environ.get("EXP12K_VARIANT", "k1_delta_surgery")
OUT_DIR = OUTPUT_ROOT / "exp12k_champion_delta_surgery" / EXP12K_VARIANT
PRED_DIR = OUT_DIR / "predictions"
SUB_DIR = OUT_DIR / "submissions"
SUM_DIR = OUT_DIR / "summaries"
FIG_DIR = OUT_DIR / "figures"
for d in [PRED_DIR, SUB_DIR, SUM_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RUN_LOCAL_GT_AUDIT_DEFAULT = os.environ.get("EXP12K_RUN_LOCAL_GT_AUDIT", "1").strip().lower() in {"1", "true", "yes", "y"}
STRICT_FINAL = True
MAX_GOAL_SANITY = 40

log_section("Setup")
log_info(f"PROJECT_ROOT = {PROJECT_ROOT}")
log_info(f"OUTPUT_ROOT = {OUTPUT_ROOT}")
log_info(f"OUT_DIR = {OUT_DIR}")
log_info(f"SUB_DIR = {SUB_DIR}")
log_info(f"SUM_DIR = {SUM_DIR}")
log_info(f"EXP12K_VARIANT = {EXP12K_VARIANT}")
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
    path = SUB_DIR / f"submission_exp12k_{label}.csv"
    aligned.to_csv(path, index=False)
    log_saved(path)
    return path, check_df

# %% [markdown]

# %% [markdown]
# # 04. Load Champion, Pre-Booster Base, dan Candidate Pool
#
# Penjelasan bagian:
# Bagian ini mencari champion terbaru `submission_exp12j_j0_only_1_1_reproduction.csv` atau fallback EXP12I, lalu mencari pre-booster base idealnya EXP12E best. Submission lama hanya dibaca sebagai comparison/post-processing artifact, bukan fitur training.
#
# Output yang perlu dilihat:
# `k0_reproduction_summary.csv` harus menunjukkan champion source yang benar dan pre-base source yang benar. Kalau `pre_base_label` bukan EXP12E/f0, `champion_delta_table` bisa salah.

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
    search_roots = [
        OUTPUT_ROOT / "exp12k_champion_delta_surgery",
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
                    matched.append({"path": p.resolve(), "group_idx": group_idx, "root_idx": root_idx, "is_mnt": "/mnt/data" in str(p), "path_len": len(str(p))})
                    break
    if not matched:
        return []
    df = pd.DataFrame(matched).drop_duplicates("path")
    df = df.sort_values(["group_idx", "is_mnt", "root_idx", "path_len", "path"], ascending=[True, True, True, True, True]).reset_index(drop=True)
    return [Path(x) for x in df["path"].head(max_results).tolist()]


def load_submission_path(path: Path, label: str = None):
    df = align_to_sample(pd.read_csv(path))
    lab = label or submission_label_from_path(path)
    return df, lab


champ_keyword_groups = [
    ["exp12j", "j0", "only_1_1", "reproduction"],
    ["exp12i", "i7", "cap0290", "only_1_1"],
    ["exp12i", "i7", "cap0290", "no_2_2"],
    ["exp12i", "i7", "cap0290", "only_0_0_1_1"],
    ["exp12i", "i7", "cap0290", "no_0_0_and_2_2"],
]
champ_paths = find_submission_paths_by_keywords(champ_keyword_groups, max_results=10)
if not champ_paths:
    raise FileNotFoundError("Tidak menemukan champion EXP12J/EXP12I. Pastikan outputs EXP12J atau EXP12I tersedia.")
champ_path = champ_paths[0]
champ_submission, champ_label = load_submission_path(champ_path)

pre_base_keyword_groups = [
    ["exp12e", "e2", "consensus_outcome_agree_only"],
    ["exp12e", "repair_min_total_same_outcome"],
    ["exp12e", "repair_preserve_gd_lower_total"],
    ["exp12f", "f0", "exp12e_best_reproduction"],
]
pre_paths = find_submission_paths_by_keywords(pre_base_keyword_groups, max_results=10)
if pre_paths:
    pre_path = pre_paths[0]
    base_submission, pre_label = load_submission_path(pre_path)
else:
    pre_path = champ_path
    base_submission = champ_submission.copy()
    pre_label = champ_label + "__fallback_as_base"
    log_info("WARNING: pre-booster base tidak ditemukan; fallback memakai champion sebagai base. Delta table akan kosong/salah.")

log_section("Source Candidates")
log_info(f"champ_path = {champ_path}")
log_info(f"champ_label = {champ_label}")
log_info(f"pre_base_path = {pre_path}")
log_info(f"pre_base_label = {pre_label}")

base_match = submission_to_match_df(base_submission, label="pre_booster_base")
champ_match = submission_to_match_df(champ_submission, label="champion")

k0_summary = pd.DataFrame([
    {"key": "champ_path", "value": str(champ_path)},
    {"key": "champ_label", "value": str(champ_label)},
    {"key": "pre_base_path", "value": str(pre_path)},
    {"key": "pre_base_label", "value": str(pre_label)},
    {"key": "n_matches", "value": str(len(base_match))},
    {"key": "champ_pair_consistency", "value": str(pair_consistency_report(champ_submission)[0])},
])
k0_summary.to_csv(SUM_DIR / "k0_reproduction_summary.csv", index=False)
log_saved(SUM_DIR / "k0_reproduction_summary.csv")
display(k0_summary)

pool_roots = [
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
        candidate_load_rows.append({"label": label, "path": str(p), "n_matches": len(m), "pair_consistency": bool(pair_ok), "n_bad_pairs": int(n_bad)})
    except Exception as e:
        candidate_load_rows.append({"label": label, "path": str(p), "n_matches": 0, "pair_consistency": False, "n_bad_pairs": -1, "error": repr(e)})

candidate_match_pool["__champion__"] = champ_match
candidate_match_pool["__pre_booster_base__"] = base_match

candidate_pool_loaded = pd.DataFrame(candidate_load_rows)
candidate_pool_loaded.to_csv(SUM_DIR / "candidate_pool_loaded.csv", index=False)
log_saved(SUM_DIR / "candidate_pool_loaded.csv")
display(candidate_pool_loaded.head(50))

CANDIDATE_MATCH_POOL_INDEX = {lab: df.set_index("match_id", drop=False) for lab, df in candidate_match_pool.items() if isinstance(df, pd.DataFrame) and "match_id" in df.columns}
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
    if "__champion__" in s:
        return 3.4
    if "exp12j" in s and "j0" in s:
        return 3.3
    if "exp12i" in s and "i7" in s:
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


def candidate_scores_for_match(match_id, base_a: int, base_b: int, exclude_labels=None):
    """Return non-base candidate scorelines for one match.

    exclude_labels is important for EXP12K delta support diagnostics: when measuring
    support for the champion delta itself, __champion__ must be excluded so the
    support count is not artificially inflated by the target being measured.
    """
    exclude_labels = set(exclude_labels or [])
    scores = []

    for lab, idx in CANDIDATE_MATCH_POOL_INDEX.items():
        if lab == "__pre_booster_base__" or lab in exclude_labels:
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

    # Use the caller-provided comparison base. EXP12K surgery variants should often
    # be summarized relative to the champion, not always relative to the pre-base.
    rec = compare_to_base(sub_df, base_for_compare, label)
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
# # 06. K0 — Reproduce Champion J0
#
# Penjelasan bagian:
# K0 menyimpan ulang champion J0/EXP12I sebagai anchor. Ini penting agar semua output EXP12K bisa dibandingkan dari titik awal yang jelas.
#
# Output yang perlu dilihat:
# Cek `submission_exp12k_k0_j0_reproduction.csv` dan `k0_reproduction_summary.csv`. Pair consistency harus True dan champion source harus sesuai prioritas.

# %%
k0_path, k0_check = save_candidate(champ_submission, "k0_j0_reproduction", strict=False)
ALL_CANDIDATE_SUBMISSIONS["k0_j0_reproduction"] = champ_submission
ALL_CHANGE_DETAILS["k0_j0_reproduction"] = pd.DataFrame()
rec_k0 = compare_to_base(champ_submission, base_submission, "k0_j0_reproduction")
rec_k0.update({"n_selected": int(rec_k0["n_changed"]), "path": str(k0_path), "pair_consistency": bool(k0_check.loc[k0_check["check"] == "pair_consistency", "passed"].iloc[0])})
ALL_VARIANT_METRICS.append(rec_k0)

k0_reproduction_summary = pd.DataFrame([
    {"metric": "champ_label", "value": champ_label},
    {"metric": "champ_path", "value": str(champ_path)},
    {"metric": "pre_base_label", "value": pre_label},
    {"metric": "pre_base_path", "value": str(pre_path)},
    {"metric": "n_changed_vs_pre_base", "value": rec_k0["n_changed"]},
    {"metric": "changed_rate_vs_pre_base", "value": rec_k0["changed_rate"]},
    {"metric": "mean_pred_total", "value": rec_k0["mean_pred_total"]},
])
k0_reproduction_summary.to_csv(SUM_DIR / "k0_reproduction_summary.csv", index=False)
log_saved(SUM_DIR / "k0_reproduction_summary.csv")
display(k0_reproduction_summary)

# %% [markdown]
# # 07. K1 — Champion Delta Table
#
# Penjelasan bagian:
# K1 adalah inti EXP12K. Tabel ini membandingkan pre-base dengan champion dan menyimpan semua perubahan penuh dari pre-base ke champion.
#
# Output yang perlu dilihat:
# `champion_delta_table.csv` harus memiliki `n_delta` sekitar 614 match sesuai insight user. Kalau jauh berbeda, source pre-base/champion kemungkinan salah.

# %%
def load_previous_ordered_ranks():
    roots = [
        OUTPUT_ROOT / "exp12j_one_one_draw_to_win_refinement",
        OUTPUT_ROOT / "exp12i_cap0290_marginal_band_surgery",
        OUTPUT_ROOT / "exp12h_outcome_ranking_damage_control",
        OUTPUT_ROOT / "exp12g_outcome_booster_refinement",
    ]
    rows = []
    for root in roots:
        if not Path(root).exists():
            continue
        for name in ["ordered_outcome_candidate_table.csv", "ordered_1_1_candidate_table.csv"]:
            for p in Path(root).rglob(name):
                try:
                    df = pd.read_csv(p)
                    keep = [c for c in ["match_id", "new_a", "new_b", "rank", "support", "weighted_support", "source_label", "source_weight"] if c in df.columns]
                    if not {"match_id", "new_a", "new_b"}.issubset(set(keep)):
                        continue
                    small = df[keep].copy()
                    small["rank_source_path"] = str(p)
                    rows.append(small)
                except Exception:
                    pass
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    return out.sort_values(["match_id", "new_a", "new_b", "rank"], na_position="last").drop_duplicates(["match_id", "new_a", "new_b"], keep="first")

previous_rank_table = load_previous_ordered_ranks()

def support_for_target(match_id, old_a, old_b, new_a, new_b):
    scores = candidate_scores_for_match(
        match_id,
        int(old_a),
        int(old_b),
        exclude_labels={"__champion__"},
    )
    support = 0
    weighted = 0.0
    labels = []
    top_label = None
    top_weight = 0.0
    for s in scores:
        if int(s["a"]) == int(new_a) and int(s["b"]) == int(new_b):
            support += 1
            weighted += float(s["weight"])
            labels.append(s["label"])
            if float(s["weight"]) > top_weight:
                top_weight = float(s["weight"])
                top_label = s["label"]
    return support, weighted, top_label, top_weight, ";".join(labels[:10])

def build_champion_delta_table():
    b = base_match[["match_id", "pred_a", "pred_b", "gender", "tournament", "tournament_weight", "row_id_a", "row_id_b"]].rename(columns={"pred_a": "old_a", "pred_b": "old_b"})
    c = champ_match[["match_id", "pred_a", "pred_b"]].rename(columns={"pred_a": "new_a", "pred_b": "new_b"})
    d = b.merge(c, on="match_id", how="inner", validate="one_to_one")
    d = d[(d["old_a"] != d["new_a"]) | (d["old_b"] != d["new_b"])].copy().reset_index(drop=True)
    if len(d) == 0:
        return d
    d["old_score"] = d.apply(lambda r: scoreline(r["old_a"], r["old_b"]), axis=1)
    d["new_score"] = d.apply(lambda r: scoreline(r["new_a"], r["new_b"]), axis=1)
    d["transition"] = d["old_score"] + "->" + d["new_score"]
    d["old_total"] = d["old_a"] + d["old_b"]
    d["new_total"] = d["new_a"] + d["new_b"]
    d["total_delta"] = (d["new_total"] - d["old_total"]).abs()
    d["old_gd"] = d["old_a"] - d["old_b"]
    d["new_gd"] = d["new_a"] - d["new_b"]
    d["gd_delta"] = (d["new_gd"] - d["old_gd"]).abs()
    d["old_outcome"] = outcome_array(d["old_a"], d["old_b"])
    d["new_outcome"] = outcome_array(d["new_a"], d["new_b"])
    d["is_draw_to_win"] = (d["old_outcome"] == 0) & (d["new_outcome"] != 0)
    d["is_high_weight"] = d["tournament_weight"].astype(float) >= 1.8
    d["is_low_weight"] = d["tournament_weight"].astype(float) <= 0.96
    d["is_world_asian"] = d["tournament"].map(is_world_asian_tournament)
    d["direction"] = np.select(
        [(d["old_outcome"] == 0) & (d["new_outcome"] == 1), (d["old_outcome"] == 0) & (d["new_outcome"] == -1), d["new_outcome"] > d["old_outcome"], d["new_outcome"] < d["old_outcome"]],
        ["draw_to_a_win", "draw_to_b_win", "outcome_up", "outcome_down"],
        default="other",
    )
    support_rows = []
    for _, r in d.iterrows():
        sup, wsup, top_lab, top_w, labs = support_for_target(r["match_id"], r["old_a"], r["old_b"], r["new_a"], r["new_b"])
        support_rows.append({"match_id": r["match_id"], "support": sup, "weighted_support": wsup, "source_label_top": top_lab, "source_weight_top": top_w, "support_labels": labs})
    d = d.merge(pd.DataFrame(support_rows), on="match_id", how="left", validate="one_to_one")
    if len(previous_rank_table):
        rt = previous_rank_table.rename(columns={"rank": "rank_if_available"}).copy()
        keep = [c for c in ["match_id", "new_a", "new_b", "rank_if_available", "rank_source_path"] if c in rt.columns]
        d = d.merge(rt[keep], on=["match_id", "new_a", "new_b"], how="left")
    else:
        d["rank_if_available"] = np.nan
        d["rank_source_path"] = ""
    d = d.sort_values(["rank_if_available", "weighted_support", "support", "source_weight_top", "tournament_weight", "total_delta", "gd_delta"], ascending=[True, False, False, False, True, True, True], na_position="last").reset_index(drop=True)
    d["proxy_rank"] = np.arange(1, len(d) + 1)
    return d

champion_delta_table = build_champion_delta_table()
if len(champion_delta_table) == 0:
    raise RuntimeError("champion_delta_table kosong. Source pre-base atau champion kemungkinan salah.")
champion_delta_table.to_csv(SUM_DIR / "champion_delta_table.csv", index=False)
log_saved(SUM_DIR / "champion_delta_table.csv")
display(champion_delta_table.head(40))

champion_delta_summary = pd.DataFrame([
    {"metric": "n_delta", "value": int(len(champion_delta_table))},
    {"metric": "n_draw_to_win", "value": int(champion_delta_table["is_draw_to_win"].sum())},
    {"metric": "n_high_weight", "value": int(champion_delta_table["is_high_weight"].sum())},
    {"metric": "n_low_weight", "value": int(champion_delta_table["is_low_weight"].sum())},
    {"metric": "n_M", "value": int(champion_delta_table["gender"].astype(str).str.upper().eq("M").sum())},
    {"metric": "n_W", "value": int(champion_delta_table["gender"].astype(str).str.upper().eq("W").sum())},
])
champion_delta_summary.to_csv(SUM_DIR / "champion_delta_summary.csv", index=False)
log_saved(SUM_DIR / "champion_delta_summary.csv")
display(champion_delta_summary)


n_delta = int(len(champion_delta_table))
if not (500 <= n_delta <= 750):
    log_info(
        f"WARNING: n_delta={n_delta:,} jauh dari ekspektasi sekitar 614. "
        "Cek champ_path dan pre_base_path sebelum percaya hasil EXP12K."
    )

for group_col, fname in [("transition", "champion_delta_transition_summary.csv"), ("gender", "champion_delta_gender_summary.csv"), ("tournament", "champion_delta_tournament_summary.csv")]:
    tab = champion_delta_table.groupby(group_col, dropna=False).agg(n=("match_id", "count"), mean_weight=("tournament_weight", "mean"), mean_support=("support", "mean"), mean_weighted_support=("weighted_support", "mean")).reset_index().sort_values("n", ascending=False)
    tab.to_csv(SUM_DIR / fname, index=False)
    log_saved(SUM_DIR / fname)
    display(tab.head(30))

def apply_undo_delta(champion_m: pd.DataFrame, delta_rows: pd.DataFrame, variant: str):
    out = champion_m.copy()
    if len(delta_rows):
        repl = delta_rows.set_index("match_id")[["old_a", "old_b"]].to_dict("index")
        mask = out["match_id"].isin(repl.keys())
        out.loc[mask, "pred_a"] = out.loc[mask, "match_id"].map(lambda x: int(repl[x]["old_a"]))
        out.loc[mask, "pred_b"] = out.loc[mask, "match_id"].map(lambda x: int(repl[x]["old_b"]))
    ch = delta_rows.copy()
    ch["variant"] = variant
    ch["operation"] = "undo_to_pre_base"
    return match_df_to_submission(out), ch

def apply_addback(champion_m: pd.DataFrame, add_rows: pd.DataFrame, variant: str):
    out = champion_m.copy()
    if len(add_rows):
        repl = add_rows.set_index("match_id")[["new_a", "new_b"]].to_dict("index")
        mask = out["match_id"].isin(repl.keys())
        out.loc[mask, "pred_a"] = out.loc[mask, "match_id"].map(lambda x: int(repl[x]["new_a"]))
        out.loc[mask, "pred_b"] = out.loc[mask, "match_id"].map(lambda x: int(repl[x]["new_b"]))
    ch = add_rows.copy()
    ch["variant"] = variant
    ch["operation"] = "addback_from_candidate_pool"
    return match_df_to_submission(out), ch

def save_undo_variant(label: str, delta_rows: pd.DataFrame):
    sub, changes = apply_undo_delta(champ_match, delta_rows, label)
    register_candidate(label, sub, changes, strict=False, base_for_compare=champ_submission)
    return sub, changes

def save_addback_variant(label: str, add_rows: pd.DataFrame):
    sub, changes = apply_addback(champ_match, add_rows, label)
    register_candidate(label, sub, changes, strict=False, base_for_compare=champ_submission)
    return sub, changes

# %% [markdown]
# # 08. K2 — Leave-One-Transition-Out
#
# Penjelasan bagian:
# K2 mulai dari champion, lalu undo satu grup transition ke pre-base. Ini menguji apakah transition tertentu toxic.
#
# Output yang perlu dilihat:
# `transition_undo_grid.csv` dan candidate CSV seperti `submission_exp12k_k2_remove_all_2_2.csv`. Jika undo sebuah transition menang di audit lokal, transition itu kemungkinan toxic.

# %%
def filter_transition(transitions=None, old_score=None):
    d = champion_delta_table.copy()
    if transitions is not None:
        return d[d["transition"].isin(transitions)].copy()
    if old_score is not None:
        return d[d["old_score"].eq(old_score)].copy()
    return d.copy()

transition_configs = [
    ("k2_remove_1_1_to_1_0", filter_transition(transitions=["1-1->1-0"])),
    ("k2_remove_1_1_to_0_1", filter_transition(transitions=["1-1->0-1"])),
    ("k2_remove_all_1_1", filter_transition(old_score="1-1")),
    ("k2_remove_0_0_to_1_0", filter_transition(transitions=["0-0->1-0"])),
    ("k2_remove_0_0_to_0_1", filter_transition(transitions=["0-0->0-1"])),
    ("k2_remove_all_0_0", filter_transition(old_score="0-0")),
    ("k2_remove_2_2_to_2_1", filter_transition(transitions=["2-2->2-1"])),
    ("k2_remove_2_2_to_1_2", filter_transition(transitions=["2-2->1-2"])),
    ("k2_remove_all_2_2", filter_transition(old_score="2-2")),
    ("k2_remove_all_non_1_1", champion_delta_table[~champion_delta_table["old_score"].eq("1-1")].copy()),
]
transition_undo_records = []
for label, rows in transition_configs:
    sub, changes = save_undo_variant(label, rows)
    rec = compare_to_base(sub, champ_submission, label)
    rec.update({"transition_removed": label.replace("k2_remove_", ""), "n_removed": int(len(rows)), "removed_high_weight": int(rows["is_high_weight"].sum()) if len(rows) else 0, "removed_M": int(rows["gender"].astype(str).str.upper().eq("M").sum()) if len(rows) else 0, "removed_W": int(rows["gender"].astype(str).str.upper().eq("W").sum()) if len(rows) else 0})
    transition_undo_records.append(rec)
transition_undo_grid = pd.DataFrame(transition_undo_records)
transition_undo_grid.to_csv(SUM_DIR / "transition_undo_grid.csv", index=False)
log_saved(SUM_DIR / "transition_undo_grid.csv")
display(transition_undo_grid)
removed_summary = transition_table({label: rows for label, rows in transition_configs})
removed_summary.to_csv(SUM_DIR / "transition_undo_removed_summary.csv", index=False)
log_saved(SUM_DIR / "transition_undo_removed_summary.csv")
display(removed_summary.head(50))

# %% [markdown]
# # 09. K3 — Segment / Domain Undo
#
# Penjelasan bagian:
# K3 menguji apakah ada segmen domain yang toxic dalam champion delta, seperti W, high-weight, friendly, world/asian, atau low-support changes.
#
# Output yang perlu dilihat:
# `segment_undo_grid.csv`. Jika undo satu segmen improve di audit lokal, segmen itu perlu gate khusus di eksperimen berikutnya.

# %%
support_filled = champion_delta_table["support"].fillna(0)
segment_configs = [
    ("k3_remove_W_changes", champion_delta_table[champion_delta_table["gender"].astype(str).str.upper().eq("W")].copy()),
    ("k3_remove_M_changes", champion_delta_table[champion_delta_table["gender"].astype(str).str.upper().eq("M")].copy()),
    ("k3_remove_high_weight_changes", champion_delta_table[champion_delta_table["is_high_weight"]].copy()),
    ("k3_remove_low_weight_changes", champion_delta_table[champion_delta_table["is_low_weight"]].copy()),
    ("k3_remove_friendly_changes", champion_delta_table[champion_delta_table["tournament"].astype(str).str.lower().str.contains("friendly", na=False)].copy()),
    ("k3_remove_world_asian_changes", champion_delta_table[champion_delta_table["is_world_asian"]].copy()),
    ("k3_remove_high_weight_low_support", champion_delta_table[(champion_delta_table["is_high_weight"]) & (support_filled <= 2)].copy()),
    ("k3_remove_low_support_all", champion_delta_table[support_filled <= 2].copy()),
]
segment_undo_records = []
for label, rows in segment_configs:
    sub, changes = save_undo_variant(label, rows)
    rec = compare_to_base(sub, champ_submission, label)
    rec.update({"segment_removed": label.replace("k3_remove_", ""), "n_removed": int(len(rows)), "removed_high_weight": int(rows["is_high_weight"].sum()) if len(rows) else 0, "removed_low_weight": int(rows["is_low_weight"].sum()) if len(rows) else 0})
    segment_undo_records.append(rec)
segment_undo_grid = pd.DataFrame(segment_undo_records)
segment_undo_grid.to_csv(SUM_DIR / "segment_undo_grid.csv", index=False)
log_saved(SUM_DIR / "segment_undo_grid.csv")
display(segment_undo_grid)
segment_removed_summary = transition_table({label: rows for label, rows in segment_configs})
segment_removed_summary.to_csv(SUM_DIR / "segment_undo_removed_summary.csv", index=False)
log_saved(SUM_DIR / "segment_undo_removed_summary.csv")
display(segment_removed_summary.head(50))

# %% [markdown]
# # 10. K4 — Champion Delta Tail Removal
#
# Penjelasan bagian:
# K4 menghapus tail dari champion delta berdasarkan rank sebelumnya atau ranking proxy. Ini menguji apakah perubahan terakhir/rendah-confidence dalam champion toxic.
#
# Output yang perlu dilihat:
# `champion_tail_removal_grid.csv` dan `champion_tail_removed_changes.csv`. Jika minus_last_N menang, tail champion perlu dipangkas.

# %%
def ranked_delta_for_tail():
    d = champion_delta_table.copy()
    d["rank_key"] = pd.to_numeric(d.get("rank_if_available", np.nan), errors="coerce")
    if d["rank_key"].notna().any():
        d = d.sort_values(["rank_key", "proxy_rank"], ascending=[True, True], na_position="last").reset_index(drop=True)
    else:
        d = d.sort_values(["weighted_support", "support", "source_weight_top", "tournament_weight", "total_delta", "gd_delta"], ascending=[False, False, False, True, True, True]).reset_index(drop=True)
    d["tail_order"] = np.arange(1, len(d) + 1)
    return d

ranked_delta = ranked_delta_for_tail()
tail_ns = [5, 10, 15, 20, 30, 40]
champion_tail_records = []
tail_change_dict = {}
for n in tail_ns:
    rows = ranked_delta.tail(n).copy()
    label = f"k4_j0_minus_last_{n:02d}"
    sub, changes = save_undo_variant(label, rows)
    rec = compare_to_base(sub, champ_submission, label)
    rec.update({"n_removed": int(len(rows)), "removed_rank_min": float(rows["rank_key"].min()) if rows["rank_key"].notna().any() else np.nan, "removed_rank_max": float(rows["rank_key"].max()) if rows["rank_key"].notna().any() else np.nan, "removed_high_weight": int(rows["is_high_weight"].sum()) if len(rows) else 0, "removed_W": int(rows["gender"].astype(str).str.upper().eq("W").sum()) if len(rows) else 0})
    champion_tail_records.append(rec)
    tail_change_dict[label] = rows
champion_tail_removal_grid = pd.DataFrame(champion_tail_records)
champion_tail_removal_grid.to_csv(SUM_DIR / "champion_tail_removal_grid.csv", index=False)
log_saved(SUM_DIR / "champion_tail_removal_grid.csv")
display(champion_tail_removal_grid)
champion_tail_removed_changes = pd.concat([v.assign(variant=k) for k, v in tail_change_dict.items()], ignore_index=True) if tail_change_dict else pd.DataFrame()
champion_tail_removed_changes.to_csv(SUM_DIR / "champion_tail_removed_changes.csv", index=False)
log_saved(SUM_DIR / "champion_tail_removed_changes.csv")
display(champion_tail_removed_changes.head(50))

# %% [markdown]
# # 11. K5 — Safe Add-Back High-Support Candidates
#
# Penjelasan bagian:
# K5 mencoba menambah sedikit kandidat perubahan yang belum masuk champion, tetapi punya support tinggi dan risiko rendah.
#
# Output yang perlu dilihat:
# `safe_addback_grid.csv` dan `safe_addback_added_changes.csv`. Jika add-back improve, berarti champion masih kurang agresif pada kandidat tertentu.

# %%
def build_safe_addback_table():
    changed_ids = set(champion_delta_table["match_id"].tolist())
    rows = []
    for _, r in base_match.iterrows():
        mid = r["match_id"]
        if mid in changed_ids:
            continue
        old_a, old_b = int(r["pred_a"]), int(r["pred_b"])
        scores = candidate_scores_for_match(mid, old_a, old_b)
        groups = defaultdict(list)
        for s in scores:
            groups[(int(s["a"]), int(s["b"]))].append(s)
        for (new_a, new_b), ss in groups.items():
            old_out, new_out = outcome_scalar(old_a, old_b), outcome_scalar(new_a, new_b)
            if old_out != 0 or new_out == 0:
                continue
            support = len(ss)
            weighted = sum(float(x["weight"]) for x in ss)
            old_total, new_total = old_a + old_b, new_a + new_b
            old_gd, new_gd = old_a - old_b, new_a - new_b
            tw = float(r.get("tournament_weight", 1.2))
            if support < 3 or abs(new_total - old_total) > 1 or abs(new_gd - old_gd) > 2:
                continue
            if tw >= 1.8 and support < 4:
                continue
            top = sorted(ss, key=lambda x: (x["weight"], x["label"]), reverse=True)[0]
            rows.append({"match_id": mid, "old_a": old_a, "old_b": old_b, "new_a": new_a, "new_b": new_b, "old_score": scoreline(old_a, old_b), "new_score": scoreline(new_a, new_b), "transition": f"{scoreline(old_a, old_b)}->{scoreline(new_a, new_b)}", "gender": str(r.get("gender", "unknown")).upper(), "tournament": str(r.get("tournament", "unknown")), "tournament_weight": tw, "support": support, "weighted_support": weighted, "source_label_top": top["label"], "source_weight_top": top["weight"], "old_total": old_total, "new_total": new_total, "total_delta": abs(new_total - old_total), "old_gd": old_gd, "new_gd": new_gd, "gd_delta": abs(new_gd - old_gd), "is_high_weight": bool(tw >= 1.8), "is_low_weight": bool(tw <= 0.96)})
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df
    df = df.sort_values(["weighted_support", "support", "source_weight_top", "tournament_weight", "total_delta", "gd_delta"], ascending=[False, False, False, True, True, True]).drop_duplicates("match_id", keep="first").reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df

safe_addback_table = build_safe_addback_table()
safe_addback_table.to_csv(SUM_DIR / "safe_addback_candidates.csv", index=False)
log_saved(SUM_DIR / "safe_addback_candidates.csv")
display(safe_addback_table.head(40))

def select_cap(table, n):
    return table.head(int(min(n, len(table)))).copy() if len(table) else pd.DataFrame()

addback_configs = [
    ("k5_j0_plus_high_support_05", select_cap(safe_addback_table, 5)),
    ("k5_j0_plus_high_support_10", select_cap(safe_addback_table, 10)),
    ("k5_j0_plus_weighted_support_05", safe_addback_table.sort_values("weighted_support", ascending=False).head(5).copy() if len(safe_addback_table) else pd.DataFrame()),
    ("k5_j0_plus_low_weight_safe_10", safe_addback_table[safe_addback_table["is_low_weight"]].head(10).copy() if len(safe_addback_table) else pd.DataFrame()),
    ("k5_j0_plus_medium_low_weight_10", safe_addback_table[safe_addback_table["tournament_weight"] < 1.8].head(10).copy() if len(safe_addback_table) else pd.DataFrame()),
]
safe_addback_records = []
add_change_dict = {}
for label, rows in addback_configs:
    sub, changes = apply_addback(champ_match, rows, label)
    register_candidate(label, sub, changes, strict=False, base_for_compare=champ_submission)
    rec = compare_to_base(sub, champ_submission, label)
    rec.update({"n_added": int(len(rows)), "added_high_weight": int(rows["is_high_weight"].sum()) if len(rows) else 0, "added_W": int(rows["gender"].astype(str).str.upper().eq("W").sum()) if len(rows) else 0})
    safe_addback_records.append(rec)
    add_change_dict[label] = rows
safe_addback_grid = pd.DataFrame(safe_addback_records)
safe_addback_grid.to_csv(SUM_DIR / "safe_addback_grid.csv", index=False)
log_saved(SUM_DIR / "safe_addback_grid.csv")
display(safe_addback_grid)
safe_addback_added_changes = pd.concat([v.assign(variant=k) for k, v in add_change_dict.items()], ignore_index=True) if add_change_dict else pd.DataFrame()
safe_addback_added_changes.to_csv(SUM_DIR / "safe_addback_added_changes.csv", index=False)
log_saved(SUM_DIR / "safe_addback_added_changes.csv")
display(safe_addback_added_changes.head(50))

# %% [markdown]
# # 12. K6 — Direction-Specific Undo from Champion
#
# Penjelasan bagian:
# K6 mulai dari champion penuh, lalu undo hanya arah tertentu dalam champion delta.
#
# Output yang perlu dilihat:
# `direction_undo_grid.csv`. Jika undo salah satu direction improve di audit lokal, arah itu toxic.

# %%
home_shift = champion_delta_table[(champion_delta_table["old_outcome"] == 0) & (champion_delta_table["new_outcome"] == 1)].copy()
away_shift = champion_delta_table[(champion_delta_table["old_outcome"] == 0) & (champion_delta_table["new_outcome"] == -1)].copy()
one_one = champion_delta_table[champion_delta_table["old_score"].eq("1-1")].copy()
one_one_home = one_one[one_one["new_score"].eq("1-0")].copy()
one_one_away = one_one[one_one["new_score"].eq("0-1")].copy()
direction_configs = [
    ("k6_undo_1_1_to_1_0", one_one_home),
    ("k6_undo_1_1_to_0_1", one_one_away),
    ("k6_keep_only_1_1_to_1_0_among_1_1", one_one_away),
    ("k6_keep_only_1_1_to_0_1_among_1_1", one_one_home),
    ("k6_undo_home_win_shift_all", home_shift),
    ("k6_undo_away_win_shift_all", away_shift),
]
direction_undo_records = []
for label, rows in direction_configs:
    sub, changes = save_undo_variant(label, rows)
    rec = compare_to_base(sub, champ_submission, label)
    rec.update({"direction_removed": label.replace("k6_", ""), "n_removed": int(len(rows)), "home_win_removed": int(((rows["old_outcome"] == 0) & (rows["new_outcome"] == 1)).sum()) if len(rows) else 0, "away_win_removed": int(((rows["old_outcome"] == 0) & (rows["new_outcome"] == -1)).sum()) if len(rows) else 0})
    direction_undo_records.append(rec)
direction_undo_grid = pd.DataFrame(direction_undo_records)
direction_undo_grid.to_csv(SUM_DIR / "direction_undo_grid.csv", index=False)
log_saved(SUM_DIR / "direction_undo_grid.csv")
display(direction_undo_grid)
direction_undo_removed_summary = transition_table({label: rows for label, rows in direction_configs})
direction_undo_removed_summary.to_csv(SUM_DIR / "direction_undo_removed_summary.csv", index=False)
log_saved(SUM_DIR / "direction_undo_removed_summary.csv")
display(direction_undo_removed_summary.head(50))

# %% [markdown]
# # 13. K7 — Tiny Exact/GD Repair After Champion
#
# Penjelasan bagian:
# K7 optional. Setelah champion, coba repair exact/GD sangat kecil dari candidate pool pada match yang belum berubah di champion.
#
# Output yang perlu dilihat:
# `tiny_repair_after_champion_grid.csv`. Jika repair menang kecil di audit lokal, repair kecil layak dilanjutkan.

# %%
def build_tiny_repair_table(mode: str = "exact"):
    changed_ids = set(champion_delta_table["match_id"].tolist())
    rows = []
    for _, r in champ_match.iterrows():
        mid = r["match_id"]
        if mid in changed_ids:
            continue
        base_a, base_b = int(r["pred_a"]), int(r["pred_b"])
        scores = candidate_scores_for_match(mid, base_a, base_b)
        groups = defaultdict(list)
        for s in scores:
            groups[(int(s["a"]), int(s["b"]))].append(s)
        for (new_a, new_b), ss in groups.items():
            old_out, new_out = outcome_scalar(base_a, base_b), outcome_scalar(new_a, new_b)
            if new_out != old_out:
                continue
            old_total, new_total = base_a + base_b, new_a + new_b
            old_gd, new_gd = base_a - base_b, new_a - new_b
            if abs(new_total - old_total) > 1 or abs(new_gd - old_gd) > 1:
                continue
            if mode == "gd" and new_gd != old_gd:
                continue
            if len(ss) < 2:
                continue
            weighted = sum(float(x["weight"]) for x in ss)
            top = sorted(ss, key=lambda x: (x["weight"], x["label"]), reverse=True)[0]
            rows.append({"match_id": mid, "old_a": base_a, "old_b": base_b, "new_a": new_a, "new_b": new_b, "old_score": scoreline(base_a, base_b), "new_score": scoreline(new_a, new_b), "transition": f"{scoreline(base_a, base_b)}->{scoreline(new_a, new_b)}", "support": len(ss), "weighted_support": weighted, "source_label_top": top["label"], "source_weight_top": top["weight"], "gender": str(r.get("gender", "unknown")).upper(), "tournament": str(r.get("tournament", "unknown")), "tournament_weight": float(r.get("tournament_weight", 1.2)), "old_total": old_total, "new_total": new_total, "total_delta": abs(new_total-old_total), "old_gd": old_gd, "new_gd": new_gd, "gd_delta": abs(new_gd-old_gd)})
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df
    df = df.sort_values(["weighted_support", "support", "source_weight_top", "total_delta", "gd_delta"], ascending=[False, False, False, True, True]).drop_duplicates("match_id", keep="first").reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df

exact_repair_table = build_tiny_repair_table("exact")
gd_repair_table = build_tiny_repair_table("gd")
exact_repair_table.to_csv(SUM_DIR / "tiny_exact_repair_candidates.csv", index=False)
gd_repair_table.to_csv(SUM_DIR / "tiny_gd_repair_candidates.csv", index=False)

def cap_rows(table, cap_rate):
    n = cap_to_n(float(cap_rate), len(champ_match), len(table)) if len(table) else 0
    return table.head(n).copy() if len(table) else pd.DataFrame()

tiny_configs = [
    ("k7_j0_plus_exact_cap0010", exact_repair_table, 0.0010),
    ("k7_j0_plus_exact_cap0025", exact_repair_table, 0.0025),
    ("k7_j0_plus_gdrepair_cap0010", gd_repair_table, 0.0010),
    ("k7_j0_plus_gdrepair_cap0025", gd_repair_table, 0.0025),
    ("k7_j0_plus_same_outcome_cap0010", exact_repair_table, 0.0010),
]
tiny_records = []
tiny_change_dict = {}
for label, table, cap in tiny_configs:
    rows = cap_rows(table, cap)
    sub, changes = apply_addback(champ_match, rows, label)
    register_candidate(label, sub, changes, strict=False, base_for_compare=champ_submission)
    rec = compare_to_base(sub, champ_submission, label)
    rec.update({"cap_rate": cap, "n_added": int(len(rows))})
    tiny_records.append(rec)
    tiny_change_dict[label] = rows
tiny_repair_after_champion_grid = pd.DataFrame(tiny_records)
tiny_repair_after_champion_grid.to_csv(SUM_DIR / "tiny_repair_after_champion_grid.csv", index=False)
log_saved(SUM_DIR / "tiny_repair_after_champion_grid.csv")
display(tiny_repair_after_champion_grid)
tiny_repair_after_champion_transition = transition_table(tiny_change_dict)
tiny_repair_after_champion_transition.to_csv(SUM_DIR / "tiny_repair_after_champion_transition.csv", index=False)
log_saved(SUM_DIR / "tiny_repair_after_champion_transition.csv")
display(tiny_repair_after_champion_transition.head(50))

# %% [markdown]
# # 14. Aggregate Metrics, Changed Prediction Analysis, dan Scoreline Distribution
#
# Penjelasan bagian:
# Bagian ini menggabungkan semua metrik GT-free, transition summary, dan scoreline distribution. Ini membantu membaca sanity sebelum Section 99 local audit.
#
# Output yang perlu dilihat:
# `variant_metrics.csv`, `changed_prediction_analysis.csv`, dan `scoreline_distribution.csv`. Cek changed rate, mean_pred_total, dan pair consistency.

# %%
variant_metrics_df = pd.DataFrame(ALL_VARIANT_METRICS)
if len(variant_metrics_df):
    variant_metrics_df = variant_metrics_df.drop_duplicates("variant", keep="last")
variant_metrics_df.to_csv(SUM_DIR / "variant_metrics.csv", index=False)
log_saved(SUM_DIR / "variant_metrics.csv")
display(variant_metrics_df.sort_values(["pair_consistency", "changed_rate"], ascending=[False, True]).head(80))

changed_prediction_analysis = transition_table(ALL_CHANGE_DETAILS)
changed_prediction_analysis.to_csv(SUM_DIR / "changed_prediction_analysis.csv", index=False)
log_saved(SUM_DIR / "changed_prediction_analysis.csv")
display(changed_prediction_analysis.head(100))

score_rows = []
for label, sub in ALL_CANDIDATE_SUBMISSIONS.items():
    m = submission_to_match_df(sub, label=label)
    vc = m.assign(scoreline=m["pred_a"].astype(int).astype(str) + "-" + m["pred_b"].astype(int).astype(str))["scoreline"].value_counts(normalize=False).head(20)
    for sc, cnt in vc.items():
        score_rows.append({"variant": label, "scoreline": sc, "count": int(cnt), "share": float(cnt / len(m)) if len(m) else 0.0})
scoreline_distribution = pd.DataFrame(score_rows)
scoreline_distribution.to_csv(SUM_DIR / "scoreline_distribution.csv", index=False)
log_saved(SUM_DIR / "scoreline_distribution.csv")
display(scoreline_distribution.head(100))

submission_check_df = pd.concat(ALL_SUBMISSION_CHECKS, ignore_index=True) if ALL_SUBMISSION_CHECKS else pd.DataFrame()
submission_check_df.to_csv(SUM_DIR / "submission_check.csv", index=False)
log_saved(SUM_DIR / "submission_check.csv")
display(submission_check_df.head(100))

submission_catalog_df = pd.DataFrame(SUBMISSION_CATALOG).drop_duplicates("label", keep="last")
submission_catalog_df.to_csv(SUM_DIR / "submission_catalog.csv", index=False)
log_saved(SUM_DIR / "submission_catalog.csv")
display(submission_catalog_df)

# %% [markdown]
# # 15. K8 — Final Selected Safe
#
# Penjelasan bagian:
# Main pipeline tetap tidak memakai GT untuk automatic selection. `best_safe` dipilih berdasarkan urutan sanity GT-free, bukan local audit.
#
# Output yang perlu dilihat:
# `submission_exp12k_best_safe.csv` harus pair-consistent dan valid. `final_decision.csv` membedakan selected_by_pipeline dan recommended_for_local_audit.

# %%
safe_order = ["k0_j0_reproduction", "k4_j0_minus_last_05", "k2_remove_all_2_2", "k6_undo_1_1_to_1_0", "k7_j0_plus_exact_cap0010"]
selected_label = next((lab for lab in safe_order if lab in ALL_CANDIDATE_SUBMISSIONS), "k0_j0_reproduction")
selected_sub = ALL_CANDIDATE_SUBMISSIONS[selected_label]
best_path, best_check = save_submission(selected_sub, "best_safe", strict=True)

final_decision = {
    "selected_by_pipeline": selected_label,
    "selected_by_sanity": selected_label,
    "recommended_for_local_audit": "all generated submission_exp12k_*.csv; champion is decided manually from Section 99 local audit",
    "main_submission_path": str(best_path),
    "champion_label": str(champ_label),
    "pre_booster_base_label": str(pre_label),
    "n_delta": int(len(champion_delta_table)),
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
            weighted_loss = ((raw * np.where(outcome_miss, WRONG_OUTCOME_MULTIPLIER, 1.0)) ** NONLINEAR_POWER) * weights

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
            OUTPUT_ROOT / "exp12k_champion_delta_surgery",
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
