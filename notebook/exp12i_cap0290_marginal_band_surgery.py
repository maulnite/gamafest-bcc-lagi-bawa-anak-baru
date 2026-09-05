# %% [markdown]
# # 00. EXP12I — Cap0290 Marginal Band Surgery
#
# Penjelasan bagian:
# EXP12I adalah post-processing refinement di atas EXP12H. Fokusnya membedah marginal band sekitar cap0290: ultra-fine cap sweep, cap0285 plus marginal additions, cap0290 minus marginal removals, cap0295 toxic tail removal, segment/transition filter, dan tiny exact/GD repair.
#
# Output yang perlu dilihat:
# Cek folder output `PROJECT_ROOT/outputs/exp12i_cap0290_marginal_band_surgery/<variant>/`. Kandidat penting akan tersimpan di `submissions/`, metrik sanity di `summaries/`, dan local GT audit ada di cell paling akhir.

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
EXP12I_VARIANT = os.environ.get("EXP12I_VARIANT", "i1_ultra_fine_cap_sweep")
OUT_DIR = OUTPUT_ROOT / "exp12i_cap0290_marginal_band_surgery" / EXP12I_VARIANT
PRED_DIR = OUT_DIR / "predictions"
SUB_DIR = OUT_DIR / "submissions"
SUM_DIR = OUT_DIR / "summaries"
FIG_DIR = OUT_DIR / "figures"
for d in [PRED_DIR, SUB_DIR, SUM_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RUN_LOCAL_GT_AUDIT_DEFAULT = os.environ.get("EXP12I_RUN_LOCAL_GT_AUDIT", "1").strip().lower() in {"1", "true", "yes", "y"}
STRICT_FINAL = True
MAX_GOAL_SANITY = 40

log_section("Setup")
log_info(f"PROJECT_ROOT = {PROJECT_ROOT}")
log_info(f"OUTPUT_ROOT = {OUTPUT_ROOT}")
log_info(f"OUT_DIR = {OUT_DIR}")
log_info(f"SUB_DIR = {SUB_DIR}")
log_info(f"SUM_DIR = {SUM_DIR}")
log_info(f"EXP12I_VARIANT = {EXP12I_VARIANT}")
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
    path = SUB_DIR / f"submission_exp12i_{label}.csv"
    aligned.to_csv(path, index=False)
    log_saved(path)
    return path, check_df



# %% [markdown]
# # 04. Load EXP12H Cap0290 Champion, Pre-Booster Base, dan Candidate Pool
#
# Penjelasan bagian:
# Bagian ini mencari EXP12H champion `submission_exp12h_h1_cap0290.csv` sebagai I0 dan mencari EXP12E pre-booster base sebagai dasar untuk membangun ulang ordered outcome candidate table. Candidate lain dimuat sebagai comparison artifact, bukan fitur training.
#
# Output yang perlu dilihat:
# `i0_reproduction_summary.csv` harus menunjukkan I0 berasal dari `exp12h_h1_cap0290` atau fallback yang jelas. Jika fallback terjadi, hasil I1-I8 harus dibaca hati-hati karena base candidate berbeda.

# %%
SUMMARY_NAME_SKIP = {
    "submission_catalog.csv",
    "submission_check.csv",
    "submission_validation.csv",
    "submission_metrics.csv",
    "submission_pair_consistency.csv",
    "submission_scoreline_distribution.csv",
    "submission_subgroup_metrics.csv",
}


def is_submission_candidate_path(p: Path) -> bool:
    name = p.name.lower()
    if not name.startswith("submission_"):
        return False
    if name in SUMMARY_NAME_SKIP:
        return False
    if "sample" in name:
        return False
    if not name.endswith(".csv"):
        return False
    return True


def find_submission_files() -> list[Path]:
    search_dirs = [
        OUTPUT_ROOT,
        OUTPUT_ROOT / "exp12i_cap0290_marginal_band_surgery",
        OUTPUT_ROOT / "exp12h_outcome_ranking_damage_control",
        OUTPUT_ROOT / "exp12g_outcome_booster_refinement",
        OUTPUT_ROOT / "exp12f_weighted_pair_router_micro_booster",
        OUTPUT_ROOT / "exp12e_pair_repair_micro_audit",
        OUTPUT_ROOT / "exp12d_exp22_alignment_hybrid",
        OUTPUT_ROOT / "exp12c_exp22_strength_rebase_hybrid",
        OUTPUT_ROOT / "exp12b_no_pseudo_pair_native_decoder_alignment",
        PROJECT_ROOT,
        Path("/mnt/data"),
    ]
    out = []
    for d in search_dirs:
        if not Path(d).exists():
            continue
        try:
            for p in Path(d).rglob("*.csv"):
                if is_submission_candidate_path(p):
                    out.append(p.resolve())
        except Exception:
            pass
    return sorted(set(out), key=lambda x: str(x))


def try_read_submission(path: Path) -> pd.DataFrame | None:
    try:
        return align_to_sample(pd.read_csv(path))
    except Exception:
        return None


def label_from_path(path: Path) -> str:
    label = path.stem
    if label.startswith("submission_"):
        label = label[len("submission_"):]
    return label


loaded_candidates = {}
loaded_rows = []
for path in find_submission_files():
    sub = try_read_submission(path)
    if sub is None:
        continue
    label = label_from_path(path)
    unique_label = label
    idx = 2
    while unique_label in loaded_candidates:
        unique_label = f"{label}_{idx}"
        idx += 1
    pair_ok, bad_pairs, _ = pair_consistency_report(sub)
    loaded_candidates[unique_label] = {
        "path": path,
        "submission": sub,
        "pair_ok": pair_ok,
        "bad_pairs": bad_pairs,
    }
    loaded_rows.append({
        "label": unique_label,
        "path": str(path),
        "pair_ok": pair_ok,
        "bad_pairs": bad_pairs,
        "mean_pred_total": float((sub["team_goals"] + sub["opp_goals"]).mean()),
        "max_pred_goal": int(sub[["team_goals", "opp_goals"]].max().max()),
    })

candidate_pool_loaded_df = pd.DataFrame(loaded_rows).sort_values(["pair_ok", "label"], ascending=[False, True]) if loaded_rows else pd.DataFrame(columns=["label", "path", "pair_ok", "bad_pairs"])
candidate_pool_loaded_df.to_csv(SUM_DIR / "candidate_pool_loaded.csv", index=False)
log_saved(SUM_DIR / "candidate_pool_loaded.csv")
display(candidate_pool_loaded_df.head(80))


def pick_candidate(patterns: list[str], require_pair_ok: bool = True):
    pats = [p.lower() for p in patterns]
    for pat in pats:
        for label, obj in sorted(loaded_candidates.items(), key=lambda kv: kv[0]):
            hay = f"{label} {obj['path'].name} {obj['path']}".lower()
            if pat in hay:
                if require_pair_ok and not obj["pair_ok"]:
                    continue
                return label, obj["submission"], obj["path"]
    return None, None, None


champ_label, champion_submission, champ_path = pick_candidate([
    "exp12h_h1_cap0290",
    "h1_cap0290",
    "cap0290",
    "exp12h_h2_support3_cap0290",
    "exp12h_h2_support2_cap0290",
    "exp12h_h1_cap0295",
    "exp12g_g1_draw_to_win_cap0275",
], require_pair_ok=True)

pre_label, pre_booster_base_submission, pre_path = pick_candidate([
    "exp12e_e2_consensus_outcome_agree_only",
    "e2_consensus_outcome_agree_only",
    "exp12e_e1_repair_min_total_same_outcome",
    "repair_min_total_same_outcome",
    "exp12e_e1_repair_preserve_gd_lower_total",
    "preserve_gd_lower_total",
], require_pair_ok=True)

if champ_label is None:
    log_info("EXP12H champion cap0290 tidak ditemukan. Fallback ke EXP12G/EXP12E best jika tersedia.")
    champ_label, champion_submission, champ_path = pre_label, pre_booster_base_submission, pre_path
if pre_label is None:
    log_info("EXP12E pre-booster base tidak ditemukan. Fallback ke champion.")
    pre_label, pre_booster_base_submission, pre_path = champ_label, champion_submission, champ_path
if champion_submission is None:
    raise FileNotFoundError("Tidak menemukan EXP12H/EXP12G/EXP12E base submission. Jalankan eksperimen sebelumnya atau taruh candidate CSV di outputs/.")

i0_summary = pd.DataFrame([
    {"role": "i0_exp12h_cap0290_champion", "label": champ_label, "path": str(champ_path)},
    {"role": "pre_booster_base_for_ordered_table", "label": pre_label, "path": str(pre_path)},
])
i0_summary.to_csv(SUM_DIR / "i0_reproduction_summary.csv", index=False)
log_saved(SUM_DIR / "i0_reproduction_summary.csv")
display(i0_summary)

base_submission = pre_booster_base_submission
champion_base_submission = champion_submission
base_match = submission_to_match_df(base_submission, label="pre_booster_base")
champion_match = submission_to_match_df(champion_base_submission, label="i0_champion")

ALL_SUBMISSION_CHECKS = []
SUBMISSION_CATALOG = []
ALL_CANDIDATE_SUBMISSIONS = {}
ALL_CHANGE_DETAILS = {}


def save_candidate(sub: pd.DataFrame, label: str, strict: bool = False):
    path, check = save_submission(sub, label, strict=strict)
    ALL_SUBMISSION_CHECKS.append(check.assign(variant=label))
    SUBMISSION_CATALOG.append({"label": label, "path": str(path)})
    ALL_CANDIDATE_SUBMISSIONS[label] = align_to_sample(sub)
    return path, check


save_candidate(champion_base_submission, "i0_cap0290_reproduction", strict=False)
# %% [markdown]
# # 05. Candidate Match Pool dan Index Cache
#
# Penjelasan bagian:
# Bagian ini membangun pool kandidat match-level dan membuat index cache `CANDIDATE_MATCH_POOL_INDEX`. Cache wajib agar booster tidak melakukan `set_index()` berulang di loop.
#
# Output yang perlu dilihat:
# `candidate_match_pool_summary.csv` memperlihatkan kandidat yang menjadi sumber support. Pool idealnya berisi EXP12F/EXP12E/EXP12D/EXP12C/EXP12B candidate.

# %%
def repair_submission_mean_round(sub_df: pd.DataFrame) -> pd.DataFrame:
    return match_df_to_submission(submission_to_match_df(sub_df, label="repair_mean_round"))


def build_candidate_match_pool(max_candidates: int = 32) -> dict[str, pd.DataFrame]:
    pool = {
        "i0_exp12h_cap0290_champion": submission_to_match_df(champion_base_submission, "i0_exp12h_cap0290_champion"),
        "pre_booster_base": submission_to_match_df(base_submission, "pre_booster_base"),
    }
    priority_terms = [
        "exp12h", "cap0290", "exp12f", "f3", "cap020", "exp12e", "consensus", "min_total", "preserve_gd",
        "exp12d", "best_safe", "exp12c", "exp12b", "exp22", "exp15", "b5", "f4", "f1",
    ]
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
                sub = repair_submission_mean_round(sub)
            safe_label = f"loaded_{label}"[:140]
            if safe_label not in pool:
                pool[safe_label] = submission_to_match_df(sub, safe_label)
        except Exception:
            continue
    return pool


candidate_match_pool = build_candidate_match_pool(max_candidates=32)
CANDIDATE_MATCH_POOL_INDEX = {
    lab: df.set_index("match_id", drop=False)
    for lab, df in candidate_match_pool.items()
}

candidate_match_pool_summary = pd.DataFrame([
    {
        "label": lab,
        "n_matches": len(df),
        "mean_total": float((df["pred_a"] + df["pred_b"]).mean()) if len(df) else np.nan,
    }
    for lab, df in candidate_match_pool.items()
])
candidate_match_pool_summary.to_csv(SUM_DIR / "candidate_match_pool_summary.csv", index=False)
log_saved(SUM_DIR / "candidate_match_pool_summary.csv")
display(candidate_match_pool_summary.head(50))


# %% [markdown]
# # 06. Support, Reliability, dan Booster Core Functions
#
# Penjelasan bagian:
# Bagian ini membuat fungsi untuk mencari alternative scoreline dari candidate pool. EXP12I menambahkan support threshold, reliability weight, transition filter, tournament filter, gender filter, dan tiny exact combo.
#
# Output yang perlu dilihat:
# Bagian ini tidak punya output besar. Grid I1-I8 akan menunjukkan hasilnya.

# %%
def candidate_reliability_weight(label: str) -> float:
    l = str(label).lower()
    if "i0_exp12h_cap0290_champion" in l or "exp12h" in l or "cap0290" in l:
        return 3.0
    if "exp12e" in l or "consensus" in l or "min_total" in l or "preserve_gd" in l:
        return 2.5
    if "f1" in l or "high_weight" in l or "weighted" in l:
        return 2.0
    if "f4" in l or "exact" in l:
        return 1.5
    if "exp12d" in l or "best_safe" in l:
        return 1.5
    if "exp17" in l or "raw" in l:
        return 0.7
    return 1.0


candidate_reliability_map = pd.DataFrame([
    {"label": lab, "weight": candidate_reliability_weight(lab)}
    for lab in candidate_match_pool.keys()
]).sort_values("weight", ascending=False)
candidate_reliability_map.to_csv(SUM_DIR / "candidate_reliability_map.csv", index=False)
log_saved(SUM_DIR / "candidate_reliability_map.csv")


def candidate_scores_for_match(match_id, base_a: int, base_b: int) -> list[dict]:
    scores = []
    for lab, idx in CANDIDATE_MATCH_POOL_INDEX.items():
        if match_id not in idx.index:
            continue
        rr = idx.loc[match_id]
        if isinstance(rr, pd.DataFrame):
            rr = rr.iloc[0]
        a, b = int(rr["pred_a"]), int(rr["pred_b"])
        if (a, b) == (int(base_a), int(base_b)):
            continue
        scores.append({
            "a": a,
            "b": b,
            "label": lab,
            "outcome": outcome_scalar(a, b),
            "gd": a - b,
            "total": a + b,
            "weight": candidate_reliability_weight(lab),
        })
    return scores


def get_supported_alternative(
    base_row: pd.Series,
    mode: str = "outcome",
    support_threshold=2,
    weighted_support_threshold=None,
    trusted_only: bool = False,
    exclude_exp17: bool = False,
) -> tuple[int, int, float, str] | None:
    match_id = base_row["match_id"]
    base_a, base_b = int(base_row["pred_a"]), int(base_row["pred_b"])
    base_out = outcome_scalar(base_a, base_b)
    base_gd = base_a - base_b
    base_total = base_a + base_b

    scores = candidate_scores_for_match(match_id, base_a, base_b)
    if trusted_only:
        scores = [s for s in scores if s["weight"] >= 2.0]
    if exclude_exp17:
        scores = [s for s in scores if "exp17" not in s["label"].lower()]
    if not scores:
        return None

    if mode == "outcome":
        if base_out != 0:
            return None
        decisive = [s for s in scores if s["outcome"] != 0]
        if not decisive:
            return None
        by_out = defaultdict(list)
        for s in decisive:
            by_out[s["outcome"]].append(s)
        best_out, best_support, best_wsupport = None, -1, -1.0
        for out, ss in by_out.items():
            sup = len(ss)
            wsup = float(sum(x["weight"] for x in ss))
            ok = wsup >= weighted_support_threshold if weighted_support_threshold is not None else sup >= support_threshold
            if ok and (wsup, sup) > (best_wsupport, best_support):
                best_out, best_support, best_wsupport = out, sup, wsup
        if best_out is None:
            return None
        cands = by_out[best_out]
        chosen = min(cands, key=lambda s: (abs(s["total"] - base_total), abs(s["gd"] - base_gd), s["total"], -s["weight"]))
        return int(chosen["a"]), int(chosen["b"]), float(best_wsupport if weighted_support_threshold is not None else best_support), str(chosen["label"])

    if mode == "exact_same_outcome":
        cands = [
            s for s in scores
            if s["outcome"] == base_out and abs(s["total"] - base_total) <= 1 and abs(s["gd"] - base_gd) <= 1
        ]
        if not cands:
            return None
        by_score = defaultdict(list)
        for s in cands:
            by_score[(s["a"], s["b"])].append(s)
        best = None
        for (a, b), ss in by_score.items():
            sup = len(ss)
            wsup = float(sum(x["weight"] for x in ss))
            ok = wsup >= weighted_support_threshold if weighted_support_threshold is not None else sup >= support_threshold
            if not ok:
                continue
            score = (wsup, sup, -abs((a + b) - base_total), -abs((a - b) - base_gd))
            if best is None or score > best[0]:
                best = (score, a, b, wsup if weighted_support_threshold is not None else float(sup), ss[0]["label"])
        if best is None:
            return None
        return int(best[1]), int(best[2]), float(best[3]), str(best[4])

    raise ValueError(mode)


def default_risk_filter(row: pd.Series, new_a: int, new_b: int) -> bool:
    old_total = int(row["pred_a"]) + int(row["pred_b"])
    new_total = int(new_a) + int(new_b)
    if str(row.get("gender", "")).upper() == "W" and max(old_total, new_total) >= 5:
        return False
    if abs(new_total - old_total) > 1:
        return False
    if max(new_a, new_b) > MAX_GOAL_SANITY:
        return False
    return True


def apply_outcome_booster(
    base_sub: pd.DataFrame,
    variant: str,
    cap_rate: float = 0.02,
    support_threshold=2,
    weighted_support_threshold=None,
    extra_filter=None,
    candidate_filter=None,
    trusted_only: bool = False,
    exclude_exp17: bool = False,
    cap_rate_by_gender=None,
    rank_rule: str = "support_then_weight_then_delta",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_m = submission_to_match_df(base_sub, label="booster_base")
    candidates = []
    for _, r in base_m.iterrows():
        if extra_filter is not None and not bool(extra_filter(r)):
            continue
        st = int(support_threshold(r)) if callable(support_threshold) else int(support_threshold)
        wst = weighted_support_threshold(r) if callable(weighted_support_threshold) else weighted_support_threshold
        alt = get_supported_alternative(
            r,
            mode="outcome",
            support_threshold=st,
            weighted_support_threshold=wst,
            trusted_only=trusted_only,
            exclude_exp17=exclude_exp17,
        )
        if alt is None:
            continue
        a, b, support, source_lab = alt
        if not default_risk_filter(r, a, b):
            continue
        if candidate_filter is not None and not bool(candidate_filter(r, a, b)):
            continue
        candidates.append({
            "match_id": r["match_id"],
            "old_a": int(r["pred_a"]),
            "old_b": int(r["pred_b"]),
            "new_a": int(a),
            "new_b": int(b),
            "support": float(support),
            "source_label": source_lab,
            "gender": str(r.get("gender", "unknown")).upper(),
            "tournament": r.get("tournament", "unknown"),
            "tournament_weight": float(r.get("tournament_weight", 1.2)),
            "old_total": int(r["pred_a"]) + int(r["pred_b"]),
            "new_total": int(a) + int(b),
            "old_gd": int(r["pred_a"]) - int(r["pred_b"]),
            "new_gd": int(a) - int(b),
        })

    cand_df = pd.DataFrame(candidates)
    if len(cand_df) == 0:
        return match_df_to_submission(base_m), cand_df

    cand_df["total_delta"] = (cand_df["new_total"] - cand_df["old_total"]).abs()
    cand_df["gd_delta"] = (cand_df["new_gd"] - cand_df["old_gd"]).abs()
    cand_df["source_weight"] = cand_df["source_label"].map(candidate_reliability_weight).astype(float)

    rr = str(rank_rule)
    if rr == "support_then_total_delta":
        cand_df = cand_df.sort_values(["support", "total_delta", "gd_delta", "tournament_weight"], ascending=[False, True, True, True])
    elif rr == "support_then_gd_delta":
        cand_df = cand_df.sort_values(["support", "gd_delta", "total_delta", "tournament_weight"], ascending=[False, True, True, True])
    elif rr == "support_then_reliability":
        cand_df = cand_df.sort_values(["support", "source_weight", "total_delta", "gd_delta"], ascending=[False, False, True, True])
    elif rr == "reliability_then_support":
        cand_df = cand_df.sort_values(["source_weight", "support", "total_delta", "gd_delta"], ascending=[False, False, True, True])
    elif rr == "high_weight_if_support_strong":
        cand_df["high_weight_bonus"] = ((cand_df["tournament_weight"] >= 1.8) & (cand_df["support"] >= 3)).astype(int)
        cand_df = cand_df.sort_values(["high_weight_bonus", "support", "total_delta", "gd_delta"], ascending=[False, False, True, True])
    elif rr == "low_total_delta_first":
        cand_df = cand_df.sort_values(["total_delta", "support", "source_weight", "gd_delta"], ascending=[True, False, False, True])
    elif rr == "gd_delta_first_then_support":
        cand_df = cand_df.sort_values(["gd_delta", "support", "source_weight", "total_delta"], ascending=[True, False, False, True])
    elif rr == "total_delta_first_then_support":
        cand_df = cand_df.sort_values(["total_delta", "support", "source_weight", "gd_delta"], ascending=[True, False, False, True])
    else:
        cand_df = cand_df.sort_values(["support", "tournament_weight", "total_delta", "gd_delta"], ascending=[False, True, True, True])

    if cap_rate_by_gender is not None:
        selected_parts = []
        base_counts = base_m.assign(gender_key=base_m["gender"].astype(str).str.upper()).groupby("gender_key").size().to_dict()
        for gkey, cap in cap_rate_by_gender.items():
            gkey = str(gkey).upper()
            n_cap = int(math.floor(base_counts.get(gkey, 0) * float(cap)))
            if n_cap > 0:
                selected_parts.append(cand_df[cand_df["gender"].eq(gkey)].head(n_cap))
        selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else cand_df.iloc[0:0].copy()
    else:
        n_cap = int(math.floor(len(base_m) * float(cap_rate)))
        n_cap = max(1, n_cap) if len(cand_df) > 0 and cap_rate > 0 else 0
        selected = cand_df.head(n_cap)

    repl = selected.set_index("match_id").to_dict("index") if len(selected) else {}
    rows = []
    for _, r in base_m.iterrows():
        row = r.to_dict()
        if r["match_id"] in repl:
            row["pred_a"] = repl[r["match_id"]]["new_a"]
            row["pred_b"] = repl[r["match_id"]]["new_b"]
            row["variant"] = variant
        rows.append(row)
    selected = selected.copy()
    selected["variant"] = variant
    return match_df_to_submission(pd.DataFrame(rows)), selected


def apply_exact_booster(
    base_after_outcome_sub: pd.DataFrame,
    original_base_sub: pd.DataFrame,
    variant: str,
    cap_rate: float = 0.005,
    support_threshold=2,
    weighted_support_threshold=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_m = submission_to_match_df(base_after_outcome_sub, label="exact_base")
    original_m = submission_to_match_df(original_base_sub, label="original_base")
    merged0 = base_m[["match_id", "pred_a", "pred_b"]].merge(
        original_m[["match_id", "pred_a", "pred_b"]].rename(columns={"pred_a": "orig_a", "pred_b": "orig_b"}),
        on="match_id", how="left", validate="one_to_one"
    )
    already_changed = set(merged0.loc[(merged0["pred_a"] != merged0["orig_a"]) | (merged0["pred_b"] != merged0["orig_b"]), "match_id"])

    candidates = []
    for _, r in base_m.iterrows():
        if r["match_id"] in already_changed:
            continue
        alt = get_supported_alternative(r, mode="exact_same_outcome", support_threshold=support_threshold, weighted_support_threshold=weighted_support_threshold)
        if alt is None:
            continue
        a, b, support, source_lab = alt
        old_out = outcome_scalar(int(r["pred_a"]), int(r["pred_b"]))
        new_out = outcome_scalar(a, b)
        if new_out != old_out:
            continue
        if abs((a + b) - (int(r["pred_a"]) + int(r["pred_b"]))) > 1:
            continue
        if abs((a - b) - (int(r["pred_a"]) - int(r["pred_b"]))) > 1:
            continue
        candidates.append({
            "match_id": r["match_id"],
            "old_a": int(r["pred_a"]),
            "old_b": int(r["pred_b"]),
            "new_a": int(a),
            "new_b": int(b),
            "support": float(support),
            "source_label": source_lab,
            "gender": str(r.get("gender", "unknown")).upper(),
            "tournament_weight": float(r.get("tournament_weight", 1.2)),
        })
    cand_df = pd.DataFrame(candidates)
    if len(cand_df) == 0:
        return match_df_to_submission(base_m), cand_df
    cand_df = cand_df.sort_values(["support", "tournament_weight"], ascending=[False, True])
    n_cap = int(math.floor(len(base_m) * float(cap_rate)))
    n_cap = max(1, n_cap) if len(cand_df) > 0 and cap_rate > 0 else 0
    selected = cand_df.head(n_cap).copy()
    repl = selected.set_index("match_id").to_dict("index") if len(selected) else {}
    rows = []
    for _, r in base_m.iterrows():
        row = r.to_dict()
        if r["match_id"] in repl:
            row["pred_a"] = repl[r["match_id"]]["new_a"]
            row["pred_b"] = repl[r["match_id"]]["new_b"]
            row["variant"] = variant
        rows.append(row)
    selected["variant"] = variant
    return match_df_to_submission(pd.DataFrame(rows)), selected


def compare_to_base(candidate_sub: pd.DataFrame, base_sub: pd.DataFrame, label: str) -> dict:
    cm = submission_to_match_df(candidate_sub, label=label)[["match_id", "pred_a", "pred_b"]].rename(columns={"pred_a": "cand_a", "pred_b": "cand_b"})
    bm = submission_to_match_df(base_sub, label="base")[["match_id", "pred_a", "pred_b", "gender", "tournament", "tournament_weight"]].rename(columns={"pred_a": "base_a", "pred_b": "base_b"})
    m = cm.merge(bm, on="match_id", how="left", validate="one_to_one")
    changed = (m["cand_a"] != m["base_a"]) | (m["cand_b"] != m["base_b"])
    cand_out = outcome_array(m["cand_a"], m["cand_b"])
    base_out = outcome_array(m["base_a"], m["base_b"])
    cand_gd = m["cand_a"] - m["cand_b"]
    base_gd = m["base_a"] - m["base_b"]
    top_scores = (cm.assign(scoreline=cm["cand_a"].astype(int).astype(str) + "-" + cm["cand_b"].astype(int).astype(str))["scoreline"].value_counts(normalize=True))
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


def transition_table(change_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for label, ch in change_dict.items():
        if ch is None or len(ch) == 0:
            continue
        tmp = ch.copy()
        tmp["transition"] = tmp["old_a"].astype(int).astype(str) + "-" + tmp["old_b"].astype(int).astype(str) + "->" + tmp["new_a"].astype(int).astype(str) + "-" + tmp["new_b"].astype(int).astype(str)
        for tr, g in tmp.groupby("transition"):
            rows.append({"variant": label, "transition": tr, "count": len(g)})
    return pd.DataFrame(rows)




# %% [markdown]
# # 07. Ordered Outcome Candidate Table
#
# Penjelasan bagian:
# Bagian ini membangun `ordered_outcome_candidate_table.csv`, yaitu urutan kandidat draw-to-win sebelum cap diterapkan. Tabel ini adalah inti EXP12I karena I2/I3/I4 melakukan operasi marginal band berdasarkan rank kandidat yang sama dengan booster cap0290.
#
# Output yang perlu dilihat:
# Cek `ordered_outcome_candidate_table.csv` dan `marginal_band_summary.csv`. Rank sekitar cap0285–cap0295 harus terlihat, termasuk transition, gender, tournament_weight, support, weighted_support, source_label, total_delta, dan gd_delta.

# %%
def build_ordered_outcome_candidate_table(
    base_sub: pd.DataFrame,
    support_threshold=2,
    weighted_support_threshold=None,
    trusted_only: bool = False,
    exclude_exp17: bool = False,
    candidate_filter=None,
    rank_rule: str = "default",
) -> pd.DataFrame:
    base_m = submission_to_match_df(base_sub, label="ordered_base")
    rows = []
    for _, r in base_m.iterrows():
        base_a, base_b = int(r["pred_a"]), int(r["pred_b"])
        base_out = outcome_scalar(base_a, base_b)
        if base_out != 0:
            continue
        scores = candidate_scores_for_match(r["match_id"], base_a, base_b)
        if trusted_only:
            scores = [s for s in scores if s["weight"] >= 2.0]
        if exclude_exp17:
            scores = [s for s in scores if "exp17" not in s["label"].lower()]
        decisive = [s for s in scores if s["outcome"] != 0]
        if not decisive:
            continue
        by_out = defaultdict(list)
        for s in decisive:
            by_out[s["outcome"]].append(s)
        chosen_out = None
        best_key = None
        best_support = 0
        best_wsupport = 0.0
        for out, ss in by_out.items():
            sup = len(ss)
            wsup = float(sum(x["weight"] for x in ss))
            ok = wsup >= weighted_support_threshold if weighted_support_threshold is not None else sup >= support_threshold
            if not ok:
                continue
            key = (wsup, sup)
            if best_key is None or key > best_key:
                chosen_out = out
                best_key = key
                best_support = sup
                best_wsupport = wsup
        if chosen_out is None:
            continue
        group = by_out[chosen_out]
        base_total = base_a + base_b
        base_gd = base_a - base_b
        chosen = min(group, key=lambda s: (abs(s["total"] - base_total), abs(s["gd"] - base_gd), s["total"], -s["weight"]))
        new_a, new_b = int(chosen["a"]), int(chosen["b"])
        if not default_risk_filter(r, new_a, new_b):
            continue
        if candidate_filter is not None and not bool(candidate_filter(r, new_a, new_b)):
            continue
        old_total = base_total
        new_total = new_a + new_b
        old_gd = base_gd
        new_gd = new_a - new_b
        tournament = str(r.get("tournament", "unknown"))
        tw = float(r.get("tournament_weight", 1.2))
        rows.append({
            "match_id": r["match_id"],
            "old_a": base_a,
            "old_b": base_b,
            "new_a": new_a,
            "new_b": new_b,
            "old_score": scoreline(base_a, base_b),
            "new_score": scoreline(new_a, new_b),
            "transition": f"{scoreline(base_a, base_b)}->{scoreline(new_a, new_b)}",
            "support": float(best_support),
            "weighted_support": float(best_wsupport),
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
            "is_world_asian": bool(("world" in tournament.lower()) or ("asian" in tournament.lower()) or ("afc" in tournament.lower())),
        })
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df
    rr = str(rank_rule)
    if rr == "support_then_total_delta":
        sort_cols, asc = ["support", "total_delta", "gd_delta", "tournament_weight"], [False, True, True, True]
    elif rr == "support_then_gd_delta":
        sort_cols, asc = ["support", "gd_delta", "total_delta", "tournament_weight"], [False, True, True, True]
    elif rr == "weighted_then_support":
        sort_cols, asc = ["weighted_support", "support", "total_delta", "gd_delta"], [False, False, True, True]
    elif rr == "source_weight_then_support":
        sort_cols, asc = ["source_weight", "support", "total_delta", "gd_delta"], [False, False, True, True]
    else:
        sort_cols, asc = ["support", "tournament_weight", "total_delta", "gd_delta"], [False, True, True, True]
    df = df.sort_values(sort_cols, ascending=asc).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df


def cap_to_n(cap_rate: float, n_matches: int, n_candidates: int) -> int:
    n = int(math.floor(float(n_matches) * float(cap_rate)))
    n = max(0, min(n, int(n_candidates)))
    return n


def apply_selected_ordered_candidates(base_sub: pd.DataFrame, selected: pd.DataFrame, variant: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_m = submission_to_match_df(base_sub, label="apply_ordered_base")
    selected = selected.copy()
    if len(selected) == 0:
        out = match_df_to_submission(base_m)
        selected["variant"] = variant
        return out, selected
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


def save_ordered_variant(label: str, selected: pd.DataFrame, base_sub: pd.DataFrame = None, strict: bool = False):
    if base_sub is None:
        base_sub = base_submission
    sub, changes = apply_selected_ordered_candidates(base_sub, selected, label)
    path, check = save_candidate(sub, label, strict=strict)
    rec = compare_to_base(sub, base_submission, label)
    rec.update({
        "n_selected": int(len(selected)),
        "path": str(path),
        "pair_consistency": bool(check.loc[check["check"] == "pair_consistency", "passed"].iloc[0]),
    })
    ALL_CHANGE_DETAILS[label] = changes
    ALL_CANDIDATE_SUBMISSIONS[label] = sub
    return sub, changes, rec


ordered_table = build_ordered_outcome_candidate_table(base_submission, support_threshold=2)
if len(ordered_table) == 0:
    raise RuntimeError("ordered_outcome_candidate_table kosong. Candidate pool tidak mendukung outcome booster.")
ordered_table.to_csv(SUM_DIR / "ordered_outcome_candidate_table.csv", index=False)
log_saved(SUM_DIR / "ordered_outcome_candidate_table.csv")
display(ordered_table.head(30))

N_MATCHES = len(base_match)
N_0285 = cap_to_n(0.0285, N_MATCHES, len(ordered_table))
N_0290 = cap_to_n(0.0290, N_MATCHES, len(ordered_table))
N_0295 = cap_to_n(0.0295, N_MATCHES, len(ordered_table))

bands = []
for name, start_n, end_n in [
    ("band_cap0285_to_cap0290", N_0285, N_0290),
    ("band_cap0290_to_cap0295", N_0290, N_0295),
]:
    tmp = ordered_table.iloc[start_n:end_n].copy()
    bands.append({
        "band_name": name,
        "rank_start": int(start_n + 1) if len(tmp) else None,
        "rank_end": int(end_n) if len(tmp) else None,
        "n_rows": int(len(tmp)),
        "top_transitions": tmp["transition"].value_counts().head(5).to_dict() if len(tmp) else {},
        "gender_counts": tmp["gender"].value_counts().to_dict() if len(tmp) else {},
        "high_weight_count": int(tmp["is_high_weight"].sum()) if len(tmp) else 0,
        "low_weight_count": int(tmp["is_low_weight"].sum()) if len(tmp) else 0,
    })
marginal_band_summary = pd.DataFrame(bands)
marginal_band_summary.to_csv(SUM_DIR / "marginal_band_summary.csv", index=False)
log_saved(SUM_DIR / "marginal_band_summary.csv")
display(marginal_band_summary)

# %% [markdown]
# # 08. I1 — Ultra-Ultra Fine Cap Sweep Around 2.90%
#
# Penjelasan bagian:
# I1 mencoba cap sangat halus di sekitar cap0290. Semua variant dibuat dari ordered candidate table yang sama, sehingga perbedaan hanya jumlah kandidat ranking teratas yang diambil.
#
# Output yang perlu dilihat:
# `ultra_fine_cap_sweep_grid.csv`. Cek changed_rate, n_changed, changed_high_weight, changed_M/W, mean_pred_total, dan pair_consistency. Section 99 akan menentukan cap terbaik secara local audit.

# %%
I1_CAPS = [0.02860, 0.02870, 0.02880, 0.02890, 0.02900, 0.02910, 0.02920, 0.02930, 0.02940, 0.02950,
           0.02875, 0.02885, 0.02895, 0.02905, 0.02915, 0.02925]
i1_records = []
for cap in I1_CAPS:
    n = cap_to_n(cap, N_MATCHES, len(ordered_table))
    label = f"i1_cap{int(round(cap * 100000)):05d}"
    selected = ordered_table.head(n).copy()
    sub, changes, rec = save_ordered_variant(label, selected)
    rec.update({"cap_rate": cap, "n_cap": n, "changed_high_weight": int(changes["is_high_weight"].sum()) if len(changes) and "is_high_weight" in changes else 0, "changed_low_weight": int(changes["is_low_weight"].sum()) if len(changes) and "is_low_weight" in changes else 0})
    i1_records.append(rec)
ultra_fine_cap_sweep_grid = pd.DataFrame(i1_records).sort_values("cap_rate")
ultra_fine_cap_sweep_grid.to_csv(SUM_DIR / "ultra_fine_cap_sweep_grid.csv", index=False)
log_saved(SUM_DIR / "ultra_fine_cap_sweep_grid.csv")
display(ultra_fine_cap_sweep_grid)

# %% [markdown]
# # 09. I2 — Cap0285 Plus Marginal Additions
#
# Penjelasan bagian:
# I2 mulai dari cap0285 lalu menambahkan 2 sampai 20 kandidat berikutnya. Ini membedah apakah semua band cap0285→cap0290 dibutuhkan atau hanya subset awalnya yang bernilai.
#
# Output yang perlu dilihat:
# `cap0285_plus_marginal_grid.csv` dan `cap0285_plus_marginal_changes.csv`. Cek n_added, rank_start/rank_end, transisi tambahan, gender, tournament, dan source tambahan.

# %%
i2_records = []
i2_change_frames = []
for k in range(2, 22, 2):
    n = min(N_0285 + k, len(ordered_table))
    label = f"i2_cap0285_plus_next_{k:02d}"
    selected = ordered_table.head(n).copy()
    sub, changes, rec = save_ordered_variant(label, selected)
    marginal = ordered_table.iloc[N_0285:n].copy()
    marginal["variant"] = label
    marginal["n_added"] = k
    i2_change_frames.append(marginal)
    rec.update({"n_base": N_0285, "n_added": k, "rank_start": int(N_0285 + 1), "rank_end": int(n)})
    i2_records.append(rec)
cap0285_plus_marginal_grid = pd.DataFrame(i2_records)
cap0285_plus_marginal_grid.to_csv(SUM_DIR / "cap0285_plus_marginal_grid.csv", index=False)
log_saved(SUM_DIR / "cap0285_plus_marginal_grid.csv")
cap0285_plus_marginal_changes = pd.concat(i2_change_frames, ignore_index=True) if i2_change_frames else pd.DataFrame()
cap0285_plus_marginal_changes.to_csv(SUM_DIR / "cap0285_plus_marginal_changes.csv", index=False)
log_saved(SUM_DIR / "cap0285_plus_marginal_changes.csv")
display(cap0285_plus_marginal_grid)

# %% [markdown]
# # 10. I3 — Cap0290 Minus Marginal Removals
#
# Penjelasan bagian:
# I3 mulai dari cap0290 lalu menghapus kandidat paling akhir dalam ranking. Ini menguji apakah tail kecil dari cap0290 mengandung noise/toxic candidate.
#
# Output yang perlu dilihat:
# `cap0290_minus_tail_grid.csv` dan `cap0290_minus_tail_removed_changes.csv`. Jika minus_last kecil menang di audit, berarti tail cap0290 toxic.

# %%
i3_records = []
i3_removed_frames = []
for k in range(2, 16, 2):
    n = max(N_0290 - k, 0)
    label = f"i3_cap0290_minus_last_{k:02d}"
    selected = ordered_table.head(n).copy()
    sub, changes, rec = save_ordered_variant(label, selected)
    removed = ordered_table.iloc[n:N_0290].copy()
    removed["variant"] = label
    removed["n_removed"] = k
    i3_removed_frames.append(removed)
    rec.update({"n_base": N_0290, "n_removed": k, "removed_rank_start": int(n + 1), "removed_rank_end": int(N_0290)})
    i3_records.append(rec)
cap0290_minus_tail_grid = pd.DataFrame(i3_records)
cap0290_minus_tail_grid.to_csv(SUM_DIR / "cap0290_minus_tail_grid.csv", index=False)
log_saved(SUM_DIR / "cap0290_minus_tail_grid.csv")
cap0290_minus_tail_removed_changes = pd.concat(i3_removed_frames, ignore_index=True) if i3_removed_frames else pd.DataFrame()
cap0290_minus_tail_removed_changes.to_csv(SUM_DIR / "cap0290_minus_tail_removed_changes.csv", index=False)
log_saved(SUM_DIR / "cap0290_minus_tail_removed_changes.csv")
display(cap0290_minus_tail_grid)

# %% [markdown]
# # 11. I4 — Cap0295 Toxic Tail Removal
#
# Penjelasan bagian:
# Cap0295 lebih buruk dari cap0290 menurut audit sebelumnya. I4 mulai dari cap0295 lalu menghapus tail/segment berisiko untuk melihat apakah cap0295 memiliki useful additions yang tertutup toxic tail.
#
# Output yang perlu dilihat:
# `cap0295_toxic_tail_grid.csv` dan `cap0295_toxic_tail_removed_changes.csv`. Jika salah satu variant mengalahkan cap0290 di Section 99, toxic tail removal valid.

# %%
def remove_from_top_cap(cap_rate, remove_filter=None, remove_last_k: int = 0):
    n = cap_to_n(cap_rate, N_MATCHES, len(ordered_table))
    top = ordered_table.head(n).copy()
    if remove_last_k and remove_last_k > 0:
        keep = top.iloc[:-remove_last_k].copy() if len(top) > remove_last_k else top.iloc[0:0].copy()
        removed = top.iloc[len(keep):].copy()
        return keep, removed
    if remove_filter is not None:
        mask = top.apply(remove_filter, axis=1)
        removed = top[mask].copy()
        keep = top[~mask].copy()
        return keep, removed
    return top, top.iloc[0:0].copy()

I4_CONFIGS = []
for k in range(2, 14, 2):
    I4_CONFIGS.append((f"i4_cap0295_minus_last_{k:02d}", None, k))
I4_CONFIGS += [
    ("i4_cap0295_minus_high_weight_low_support", lambda r: bool(r["is_high_weight"] and r["support"] < 3), 0),
    ("i4_cap0295_minus_exp17_source", lambda r: "exp17" in str(r["source_label"]).lower(), 0),
    ("i4_cap0295_minus_w_changes", lambda r: str(r["gender"]).upper() == "W", 0),
    ("i4_cap0295_minus_0_0_transition", lambda r: str(r["old_score"]) == "0-0", 0),
    ("i4_cap0295_minus_2_2_transition", lambda r: str(r["old_score"]) == "2-2", 0),
]
i4_records = []
i4_removed_frames = []
for label, filt, last_k in I4_CONFIGS:
    keep, removed = remove_from_top_cap(0.0295, remove_filter=filt, remove_last_k=last_k)
    sub, changes, rec = save_ordered_variant(label, keep)
    removed = removed.copy()
    removed["variant"] = label
    i4_removed_frames.append(removed)
    rec.update({"removed_count": int(len(removed)), "n_kept": int(len(keep))})
    i4_records.append(rec)
cap0295_toxic_tail_grid = pd.DataFrame(i4_records)
cap0295_toxic_tail_grid.to_csv(SUM_DIR / "cap0295_toxic_tail_grid.csv", index=False)
log_saved(SUM_DIR / "cap0295_toxic_tail_grid.csv")
cap0295_toxic_tail_removed_changes = pd.concat(i4_removed_frames, ignore_index=True) if i4_removed_frames else pd.DataFrame()
cap0295_toxic_tail_removed_changes.to_csv(SUM_DIR / "cap0295_toxic_tail_removed_changes.csv", index=False)
log_saved(SUM_DIR / "cap0295_toxic_tail_removed_changes.csv")
display(cap0295_toxic_tail_grid)

# %% [markdown]
# # 12. I5 — Marginal Band Analysis
#
# Penjelasan bagian:
# I5 membuat diagnostic GT-free untuk dua band penting: cap0285→cap0290 dan cap0290→cap0295. Bagian ini tidak memilih rule otomatis, tetapi membantu membaca segment yang dominan di golden/toxic band.
#
# Output yang perlu dilihat:
# `marginal_band_analysis.csv`, `marginal_band_transition_summary.csv`, `marginal_band_tournament_summary.csv`, `marginal_band_gender_summary.csv`, dan `marginal_band_source_summary.csv`.

# %%
band_frames = []
for band_name, start_n, end_n in [
    ("cap0285_to_cap0290", N_0285, N_0290),
    ("cap0290_to_cap0295", N_0290, N_0295),
]:
    tmp = ordered_table.iloc[start_n:end_n].copy()
    tmp["band_name"] = band_name
    tmp["rank_start"] = int(start_n + 1)
    tmp["rank_end"] = int(end_n)
    band_frames.append(tmp)
marginal_band_analysis = pd.concat(band_frames, ignore_index=True) if band_frames else pd.DataFrame()
marginal_band_analysis.to_csv(SUM_DIR / "marginal_band_analysis.csv", index=False)
log_saved(SUM_DIR / "marginal_band_analysis.csv")

def summarize_band(col):
    if len(marginal_band_analysis) == 0:
        return pd.DataFrame()
    out = marginal_band_analysis.groupby(["band_name", col], dropna=False).size().reset_index(name="count")
    return out.sort_values(["band_name", "count"], ascending=[True, False])

for col, fname in [
    ("transition", "marginal_band_transition_summary.csv"),
    ("tournament", "marginal_band_tournament_summary.csv"),
    ("gender", "marginal_band_gender_summary.csv"),
    ("source_label", "marginal_band_source_summary.csv"),
]:
    s = summarize_band(col)
    s.to_csv(SUM_DIR / fname, index=False)
    log_saved(SUM_DIR / fname)

display(marginal_band_analysis.head(40))

# %% [markdown]
# # 13. I6 — Segment-Specific Marginal Filters
#
# Penjelasan bagian:
# I6 memakai cap0290 lalu menghapus perubahan berdasarkan segment gender/tournament/support. Ini menguji apakah subset tertentu dari cap0290 toxic.
#
# Output yang perlu dilihat:
# `segment_filter_grid.csv` dan `segment_filter_removed_changes.csv`. Cek n_removed, changed_M/W, changed_high_weight/low_weight, dan pair consistency.

# %%
I6_FILTERS = [
    ("i6_cap0290_no_W_changes", lambda r: str(r["gender"]).upper() == "W"),
    ("i6_cap0290_M_only", lambda r: str(r["gender"]).upper() != "M"),
    ("i6_cap0290_no_high_weight_tail", lambda r: bool(r["is_high_weight"] and r["rank"] > max(1, N_0285))),
    ("i6_cap0290_high_weight_support3", lambda r: bool(r["is_high_weight"] and r["support"] < 3)),
    ("i6_cap0290_high_weight_support4", lambda r: bool(r["is_high_weight"] and r["support"] < 4)),
    ("i6_cap0290_low_weight_only_tail", lambda r: bool((not r["is_low_weight"]) and r["rank"] > max(1, N_0285))),
    ("i6_cap0290_medium_low_weight_only", lambda r: bool(r["is_high_weight"])),
]
i6_records = []
i6_removed_frames = []
for label, filt in I6_FILTERS:
    keep, removed = remove_from_top_cap(0.0290, remove_filter=filt, remove_last_k=0)
    sub, changes, rec = save_ordered_variant(label, keep)
    removed = removed.copy()
    removed["variant"] = label
    i6_removed_frames.append(removed)
    rec.update({"n_removed": int(len(removed)), "n_kept": int(len(keep))})
    i6_records.append(rec)
segment_filter_grid = pd.DataFrame(i6_records)
segment_filter_grid.to_csv(SUM_DIR / "segment_filter_grid.csv", index=False)
log_saved(SUM_DIR / "segment_filter_grid.csv")
segment_filter_removed_changes = pd.concat(i6_removed_frames, ignore_index=True) if i6_removed_frames else pd.DataFrame()
segment_filter_removed_changes.to_csv(SUM_DIR / "segment_filter_removed_changes.csv", index=False)
log_saved(SUM_DIR / "segment_filter_removed_changes.csv")
display(segment_filter_grid)

# %% [markdown]
# # 14. I7 — Transition-Specific Marginal Filters
#
# Penjelasan bagian:
# I7 menguji apakah transisi tertentu seperti 0-0, 1-1, atau 2-2 menjadi sumber gain/toxic pada cap0290/cap0295.
#
# Output yang perlu dilihat:
# `transition_filter_grid.csv` dan `transition_filter_removed_changes.csv`. Kalau no_2_2 atau only_1_1 menang di audit, transisi itu menjadi petunjuk EXP berikutnya.

# %%
def transition_keep_filter(kind: str):
    if kind == "no_0_0":
        return lambda r: str(r["old_score"]) == "0-0"
    if kind == "no_2_2":
        return lambda r: str(r["old_score"]) == "2-2"
    if kind == "only_1_1":
        return lambda r: str(r["old_score"]) != "1-1"
    if kind == "only_0_0_1_1":
        return lambda r: str(r["old_score"]) not in {"0-0", "1-1"}
    if kind == "no_0_0_and_2_2":
        return lambda r: str(r["old_score"]) in {"0-0", "2-2"}
    return lambda r: False

I7_CONFIGS = [
    ("i7_cap0290_no_0_0", 0.0290, transition_keep_filter("no_0_0")),
    ("i7_cap0290_no_2_2", 0.0290, transition_keep_filter("no_2_2")),
    ("i7_cap0290_only_1_1", 0.0290, transition_keep_filter("only_1_1")),
    ("i7_cap0290_only_0_0_1_1", 0.0290, transition_keep_filter("only_0_0_1_1")),
    ("i7_cap0290_no_0_0_and_2_2", 0.0290, transition_keep_filter("no_0_0_and_2_2")),
    ("i7_cap0295_minus_0_0_tail", 0.0295, lambda r: str(r["old_score"]) == "0-0" and r["rank"] > N_0290),
    ("i7_cap0295_minus_2_2_tail", 0.0295, lambda r: str(r["old_score"]) == "2-2" and r["rank"] > N_0290),
]
i7_records = []
i7_removed_frames = []
for label, cap, filt in I7_CONFIGS:
    keep, removed = remove_from_top_cap(cap, remove_filter=filt)
    sub, changes, rec = save_ordered_variant(label, keep)
    removed = removed.copy()
    removed["variant"] = label
    i7_removed_frames.append(removed)
    rec.update({"cap_rate": cap, "n_removed": int(len(removed)), "n_kept": int(len(keep))})
    i7_records.append(rec)
transition_filter_grid = pd.DataFrame(i7_records)
transition_filter_grid.to_csv(SUM_DIR / "transition_filter_grid.csv", index=False)
log_saved(SUM_DIR / "transition_filter_grid.csv")
transition_filter_removed_changes = pd.concat(i7_removed_frames, ignore_index=True) if i7_removed_frames else pd.DataFrame()
transition_filter_removed_changes.to_csv(SUM_DIR / "transition_filter_removed_changes.csv", index=False)
log_saved(SUM_DIR / "transition_filter_removed_changes.csv")
display(transition_filter_grid)

# %% [markdown]
# # 15. I8 — Tiny Exact/GD Repair After Cap0290
#
# Penjelasan bagian:
# I8 optional menambahkan tiny exact/GD repair setelah cap0290. Perubahan harus sangat kecil dan tidak boleh mengubah match yang sudah diubah outcome booster.
#
# Output yang perlu dilihat:
# `tiny_repair_after_cap0290_grid.csv` dan `tiny_repair_after_cap0290_transition.csv`. Jika repair memperburuk audit, jangan lanjut repair besar.

# %%
i8_records = []
base0290_selected = ordered_table.head(N_0290).copy()
base0290_sub, base0290_changes = apply_selected_ordered_candidates(base_submission, base0290_selected, "i8_base_cap0290")
for cap in [0.0010, 0.0025, 0.0050]:
    label = f"i8_cap0290_plus_exact_cap{int(round(cap * 10000)):04d}"
    sub, changes = apply_exact_booster(base0290_sub, base_submission, label, cap_rate=cap, support_threshold=2)
    path, check = save_candidate(sub, label, strict=False)
    rec = compare_to_base(sub, base_submission, label)
    rec.update({"repair_type": "exact", "cap_rate": cap, "path": str(path), "pair_consistency": bool(check.loc[check["check"] == "pair_consistency", "passed"].iloc[0])})
    i8_records.append(rec)
    ALL_CHANGE_DETAILS[label] = changes
for cap in [0.0010, 0.0025]:
    label = f"i8_cap0290_plus_gdrepair_cap{int(round(cap * 10000)):04d}"
    sub, changes = apply_exact_booster(base0290_sub, base_submission, label, cap_rate=cap, support_threshold=3)
    path, check = save_candidate(sub, label, strict=False)
    rec = compare_to_base(sub, base_submission, label)
    rec.update({"repair_type": "gd_proxy", "cap_rate": cap, "path": str(path), "pair_consistency": bool(check.loc[check["check"] == "pair_consistency", "passed"].iloc[0])})
    i8_records.append(rec)
    ALL_CHANGE_DETAILS[label] = changes
tiny_repair_after_cap0290_grid = pd.DataFrame(i8_records)
tiny_repair_after_cap0290_grid.to_csv(SUM_DIR / "tiny_repair_after_cap0290_grid.csv", index=False)
log_saved(SUM_DIR / "tiny_repair_after_cap0290_grid.csv")
transition_i8 = transition_table({k: v for k, v in ALL_CHANGE_DETAILS.items() if k.startswith("i8_")})
transition_i8.to_csv(SUM_DIR / "tiny_repair_after_cap0290_transition.csv", index=False)
log_saved(SUM_DIR / "tiny_repair_after_cap0290_transition.csv")
display(tiny_repair_after_cap0290_grid)

# %% [markdown]
# # 16. Global Summaries dan Changed Prediction Analysis
#
# Penjelasan bagian:
# Bagian ini menggabungkan metrik sanity semua candidate, scoreline distribution, dan transition summary. Ini membantu membaca apakah candidate terlalu agresif atau collapse ke scoreline tertentu.
#
# Output yang perlu dilihat:
# `variant_metrics.csv`, `changed_prediction_analysis.csv`, `scoreline_distribution.csv`, dan `transition_all_variants.csv`.

# %%
variant_rows = []
for lab, sub in ALL_CANDIDATE_SUBMISSIONS.items():
    try:
        variant_rows.append(compare_to_base(sub, base_submission, lab))
    except Exception:
        pass
variant_metrics_df = pd.DataFrame(variant_rows).sort_values("variant") if variant_rows else pd.DataFrame()
variant_metrics_df.to_csv(SUM_DIR / "variant_metrics.csv", index=False)
variant_metrics_df.to_csv(SUM_DIR / "changed_prediction_analysis.csv", index=False)
log_saved(SUM_DIR / "variant_metrics.csv")
log_saved(SUM_DIR / "changed_prediction_analysis.csv")
display(variant_metrics_df.head(80))

score_rows = []
for lab, sub in ALL_CANDIDATE_SUBMISSIONS.items():
    m = submission_to_match_df(sub, lab)
    vc = (m.assign(scoreline=m["pred_a"].astype(int).astype(str) + "-" + m["pred_b"].astype(int).astype(str))["scoreline"].value_counts(normalize=False).head(20))
    for sc, cnt in vc.items():
        score_rows.append({"variant": lab, "scoreline": sc, "count": int(cnt), "share": float(cnt / len(m)) if len(m) else 0.0})
scoreline_distribution = pd.DataFrame(score_rows)
scoreline_distribution.to_csv(SUM_DIR / "scoreline_distribution.csv", index=False)
log_saved(SUM_DIR / "scoreline_distribution.csv")

transition_all = transition_table(ALL_CHANGE_DETAILS)
transition_all.to_csv(SUM_DIR / "transition_all_variants.csv", index=False)
log_saved(SUM_DIR / "transition_all_variants.csv")

# %% [markdown]
# # 17. I9 — Final Selected Safe
#
# Penjelasan bagian:
# Bagian ini memilih `best_safe` secara GT-free. Default safe order memprioritaskan cap0290-style candidate dan marginal variants yang pair-consistent serta tidak terlalu agresif.
#
# Output yang perlu dilihat:
# `final_decision.csv`, `submission_catalog.csv`, dan `submission_check.csv`. `best_safe` bukan klaim best local audit; champion dibaca dari Section 99.

# %%
preferred_order = [
    "i1_cap02900",
    "i1_cap02905",
    "i3_cap0290_minus_last_02",
    "i2_cap0285_plus_next_10",
    "i0_cap0290_reproduction",
]
selected_label = None
for lab in preferred_order:
    if lab in ALL_CANDIDATE_SUBMISSIONS:
        pair_ok, n_bad, _ = pair_consistency_report(ALL_CANDIDATE_SUBMISSIONS[lab])
        cr = compare_to_base(ALL_CANDIDATE_SUBMISSIONS[lab], base_submission, lab)["changed_rate"]
        if pair_ok and cr <= 0.06:
            selected_label = lab
            break
if selected_label is None:
    selected_label = "i0_cap0290_reproduction"

best_path, best_check = save_candidate(ALL_CANDIDATE_SUBMISSIONS[selected_label], "best_safe", strict=True)

submission_check_df = pd.concat(ALL_SUBMISSION_CHECKS, ignore_index=True) if ALL_SUBMISSION_CHECKS else pd.DataFrame()
submission_check_df.to_csv(SUM_DIR / "submission_check.csv", index=False)
log_saved(SUM_DIR / "submission_check.csv")

submission_catalog_df = pd.DataFrame(SUBMISSION_CATALOG).drop_duplicates("path")
submission_catalog_df.to_csv(SUM_DIR / "submission_catalog.csv", index=False)
log_saved(SUM_DIR / "submission_catalog.csv")

final_decision = {
    "experiment": "EXP12I",
    "selected_by_pipeline": selected_label,
    "selected_by_sanity": selected_label,
    "recommended_for_local_audit": "all generated submission_exp12i_*.csv; champion is decided manually from Section 99 local audit",
    "main_submission_path": str(best_path),
    "i0_base_label": str(champ_label),
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
        log_saved(SUM_DIR / "local_gt_audit_awmae_rank.csv")

        if bad_rows:
            bad_df = pd.DataFrame(bad_rows)
            display(bad_df.head(30))
            bad_df.to_csv(SUM_DIR / "local_gt_audit_bad_files.csv", index=False)
            log_saved(SUM_DIR / "local_gt_audit_bad_files.csv")
