# %% [markdown]
# # 00. EXP12H — Outcome Ranking Damage Control
#
# Penjelasan bagian:
# EXP12H adalah post-processing refinement di atas EXP12G. Fokusnya mempertahankan outcome gain cap0275 sambil mengurangi damage ke base MAE, exact rate, dan GD rate melalui ultra-fine cap sweep, support tuning, ranking rule tuning, damage-control filters, transition filtering, tournament/gender gating, dan tiny exact/GD repair.
#
# Output yang perlu dilihat:
# Cek folder output `PROJECT_ROOT/outputs/exp12h_outcome_ranking_damage_control/<variant>/`. Kandidat penting akan tersimpan di `submissions/`, metrik sanity di `summaries/`, dan local GT audit ada di cell paling akhir.

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
EXP12H_VARIANT = os.environ.get("EXP12H_VARIANT", "h1_ultrafine_cap_sweep")
OUT_DIR = OUTPUT_ROOT / "exp12h_outcome_ranking_damage_control" / EXP12H_VARIANT
PRED_DIR = OUT_DIR / "predictions"
SUB_DIR = OUT_DIR / "submissions"
SUM_DIR = OUT_DIR / "summaries"
FIG_DIR = OUT_DIR / "figures"
for d in [PRED_DIR, SUB_DIR, SUM_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RUN_LOCAL_GT_AUDIT_DEFAULT = os.environ.get("EXP12H_RUN_LOCAL_GT_AUDIT", "1").strip().lower() in {"1", "true", "yes", "y"}
STRICT_FINAL = True
MAX_GOAL_SANITY = 40

log_section("Setup")
log_info(f"PROJECT_ROOT = {PROJECT_ROOT}")
log_info(f"OUTPUT_ROOT = {OUTPUT_ROOT}")
log_info(f"OUT_DIR = {OUT_DIR}")
log_info(f"SUB_DIR = {SUB_DIR}")
log_info(f"SUM_DIR = {SUM_DIR}")
log_info(f"EXP12H_VARIANT = {EXP12H_VARIANT}")
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
    path = SUB_DIR / f"submission_exp12h_{label}.csv"
    aligned.to_csv(path, index=False)
    log_saved(path)
    return path, check_df


# %% [markdown]
# # 04. Load EXP12G Cap0275 Champion dan Candidate Pool
#
# Penjelasan bagian:
# Bagian ini mencari EXP12G champion `g1_draw_to_win_cap0275` sebagai H0 dan mencari EXP12E best sebagai pre-booster base untuk cap sweep. Candidate lain dimuat sebagai comparison artifact, bukan fitur training.
#
# Output yang perlu dilihat:
# `h0_reproduction_summary.csv` harus menunjukkan H0 berasal dari `g1_draw_to_win_cap0275`. Jika H0 tidak ketemu dan fallback terjadi, hasil H1-H7 harus dibaca hati-hati.

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
        OUTPUT_ROOT / "exp12h_outcome_ranking_damage_control",
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
display(candidate_pool_loaded_df.head(50))


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
    "exp12g_g1_draw_to_win_cap0275",
    "g1_draw_to_win_cap0275",
    "cap0275",
    "exp12g_g1_draw_to_win_cap0250",
    "exp12f_f3_draw_to_win_cap020",
    "exp12e_e2_consensus_outcome_agree_only",
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
    log_info("EXP12G champion cap0275 tidak ditemukan. Fallback ke cap0250 / EXP12F cap020 / EXP12E best.")
    champ_label, champion_submission, champ_path = pre_label, pre_booster_base_submission, pre_path
if pre_label is None:
    log_info("EXP12E pre-booster base tidak ditemukan. Fallback ke champion.")
    pre_label, pre_booster_base_submission, pre_path = champ_label, champion_submission, champ_path
if champion_submission is None:
    raise FileNotFoundError("Tidak menemukan EXP12F/EXP12E base submission. Jalankan EXP12F/EXP12E dulu atau taruh candidate CSV di outputs/.")

g0_summary = pd.DataFrame([
    {"role": "h0_exp12g_cap0275_champion", "label": champ_label, "path": str(champ_path)},
    {"role": "pre_booster_base_for_sweep", "label": pre_label, "path": str(pre_path)},
])
g0_summary.to_csv(SUM_DIR / "h0_reproduction_summary.csv", index=False)
g0_summary.to_csv(SUM_DIR / "g0_reproduction_summary.csv", index=False)
log_saved(SUM_DIR / "h0_reproduction_summary.csv")
log_saved(SUM_DIR / "g0_reproduction_summary.csv")
display(g0_summary)

base_submission = pre_booster_base_submission
champion_base_submission = champion_submission
base_match = submission_to_match_df(base_submission, label="pre_booster_base")
champion_match = submission_to_match_df(champion_base_submission, label="h0_champion")

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


save_candidate(champion_base_submission, "h0_cap0275_reproduction", strict=False)


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
        "h0_exp12g_cap0275_champion": submission_to_match_df(champion_base_submission, "h0_exp12g_cap0275_champion"),
        "pre_booster_base": submission_to_match_df(base_submission, "pre_booster_base"),
    }
    priority_terms = [
        "exp12g", "cap0275", "exp12f", "f3", "cap020", "exp12e", "consensus", "min_total", "preserve_gd",
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
# Bagian ini membuat fungsi untuk mencari alternative scoreline dari candidate pool. EXP12H menambahkan support threshold, reliability weight, transition filter, tournament filter, gender filter, dan tiny exact combo.
#
# Output yang perlu dilihat:
# Bagian ini tidak punya output besar. Grid H1-H7 akan menunjukkan hasilnya.

# %%
def candidate_reliability_weight(label: str) -> float:
    l = str(label).lower()
    if "h0_exp12g_cap0275_champion" in l or "exp12g" in l or "cap0275" in l:
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
    support_threshold: int = 2,
    weighted_support_threshold: float | None = None,
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
    support_threshold: int = 2,
    weighted_support_threshold: float | None = None,
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
# # 07. H1 — Ultra-Fine Cap Sweep Around 2.75%
#
# Penjelasan bagian:
# H1 mengulang logic draw-to-win dari EXP12G, tetapi cap disweep sangat halus di sekitar 2.75%. Tujuannya mencari apakah 2.70%, 2.80%, atau titik lain lebih bersih daripada 2.75%.
#
# Output yang perlu dilihat:
# `ultrafine_cap_sweep_grid.csv` memperlihatkan changed rate, high/low weight changes, gender changes, mean total, dan pair consistency. Candidate cap0275 harus mereproduksi gaya EXP12G best.

# %%
H1_CAPS = [0.0260, 0.0265, 0.0270, 0.0275, 0.0280, 0.0285, 0.0290, 0.0295]
h1_records, h1_submissions = [], {}
for cap in H1_CAPS:
    label = f"h1_cap{int(round(cap * 10000)):04d}"
    sub, changes = apply_outcome_booster(base_submission, label, cap_rate=cap, support_threshold=2)
    path, check = save_candidate(sub, label, strict=False)
    rec = compare_to_base(sub, base_submission, label)
    rec.update({"cap_rate": cap, "path": str(path), "pair_consistency": bool(check.loc[check["check"] == "pair_consistency", "passed"].iloc[0])})
    h1_records.append(rec)
    h1_submissions[label] = sub
    ALL_CANDIDATE_SUBMISSIONS[label] = sub
    ALL_CHANGE_DETAILS[label] = changes

ultrafine_cap_sweep_grid = pd.DataFrame(h1_records).sort_values("cap_rate")
ultrafine_cap_sweep_grid.to_csv(SUM_DIR / "ultrafine_cap_sweep_grid.csv", index=False)
log_saved(SUM_DIR / "ultrafine_cap_sweep_grid.csv")
display(ultrafine_cap_sweep_grid)

# %% [markdown]
# # 08. H2 — Support Threshold Around Cap0275
#
# Penjelasan bagian:
# H2 menguji support 2/3/4 di sekitar cap terbaik 2.65%–2.90%. Support lebih tinggi bisa mengurangi noise tambahan yang muncul setelah cap 2.75%.
#
# Output yang perlu dilihat:
# `support_around_cap0275_grid.csv`. Support3/4 yang tetap changed cukup banyak biasanya lebih bersih daripada support2 agresif.

# %%
H2_CONFIGS = [
    (2, 0.0265), (3, 0.0265),
    (2, 0.0275), (3, 0.0275), (4, 0.0275),
    (2, 0.0280), (3, 0.0280),
    (2, 0.0290), (3, 0.0290),
]
h2_records = []
for support, cap in H2_CONFIGS:
    label = f"h2_support{support}_cap{int(round(cap * 10000)):04d}"
    sub, changes = apply_outcome_booster(base_submission, label, cap_rate=cap, support_threshold=support)
    path, check = save_candidate(sub, label, strict=False)
    rec = compare_to_base(sub, base_submission, label)
    rec.update({"support_threshold": support, "cap_rate": cap, "path": str(path), "pair_consistency": bool(check.loc[check["check"] == "pair_consistency", "passed"].iloc[0])})
    h2_records.append(rec)
    ALL_CANDIDATE_SUBMISSIONS[label] = sub
    ALL_CHANGE_DETAILS[label] = changes

support_around_cap0275_grid = pd.DataFrame(h2_records)
support_around_cap0275_grid.to_csv(SUM_DIR / "support_around_cap0275_grid.csv", index=False)
log_saved(SUM_DIR / "support_around_cap0275_grid.csv")
display(support_around_cap0275_grid)

# %% [markdown]
# # 09. H3 — Ranking Rule Tuning
#
# Penjelasan bagian:
# H3 tetap memakai cap sekitar 2.75%, tetapi mengubah urutan kandidat yang dipilih. Ini penting karena cap bagus belum tentu cukup kalau ranking kandidat masih memasukkan match noisy lebih dulu.
#
# Output yang perlu dilihat:
# `ranking_rule_grid.csv` dan `ranking_rule_change_analysis.csv`. Cek changed rate, same GD, high-weight changes, dan mean total.

# %%
H3_CONFIGS = [
    ("h3_support_then_total_delta_cap0275", "support_then_total_delta", 0.0275),
    ("h3_support_then_gd_delta_cap0275", "support_then_gd_delta", 0.0275),
    ("h3_support_then_reliability_cap0275", "support_then_reliability", 0.0275),
    ("h3_reliability_then_support_cap0275", "reliability_then_support", 0.0275),
    ("h3_high_weight_if_support_strong_cap0275", "high_weight_if_support_strong", 0.0275),
    ("h3_low_total_delta_first_cap0275", "low_total_delta_first", 0.0275),
    ("h3_gd_delta_first_then_support_cap0275", "gd_delta_first_then_support", 0.0275),
    ("h3_total_delta_first_then_support_cap0275", "total_delta_first_then_support", 0.0275),
    ("h3_reliability_then_support_cap0280", "reliability_then_support", 0.0280),
]
h3_records = []
for label, rank_rule, cap in H3_CONFIGS:
    sub, changes = apply_outcome_booster(base_submission, label, cap_rate=cap, support_threshold=2, rank_rule=rank_rule)
    path, check = save_candidate(sub, label, strict=False)
    rec = compare_to_base(sub, base_submission, label)
    rec.update({"rank_rule": rank_rule, "cap_rate": cap, "path": str(path), "pair_consistency": bool(check.loc[check["check"] == "pair_consistency", "passed"].iloc[0])})
    h3_records.append(rec)
    ALL_CANDIDATE_SUBMISSIONS[label] = sub
    ALL_CHANGE_DETAILS[label] = changes

ranking_rule_grid = pd.DataFrame(h3_records)
ranking_rule_grid.to_csv(SUM_DIR / "ranking_rule_grid.csv", index=False)
log_saved(SUM_DIR / "ranking_rule_grid.csv")
display(ranking_rule_grid)
ranking_rule_change_analysis = transition_table({r["variant"]: ALL_CHANGE_DETAILS[r["variant"]] for r in h3_records})
ranking_rule_change_analysis.to_csv(SUM_DIR / "ranking_rule_change_analysis.csv", index=False)
log_saved(SUM_DIR / "ranking_rule_change_analysis.csv")

# %% [markdown]
# # 10. H4 — Damage-Control Filters
#
# Penjelasan bagian:
# H4 mempertahankan cap0275, tetapi menghapus perubahan yang berisiko: GD delta besar, total naik, support rendah di high-weight, sumber tidak trusted, W high-total, dan perubahan dari sumber EXP17.
#
# Output yang perlu dilihat:
# `damage_control_grid.csv` menunjukkan berapa kandidat tersisa setelah filter. Filter bagus kalau changed rate turun sedikit tetapi kandidat tetap cukup banyak.

# %%
def h_filter_no_gd_delta_gt_1(r, a, b):
    return abs((int(a) - int(b)) - (int(r["pred_a"]) - int(r["pred_b"]))) <= 1

def h_filter_same_or_lower_total_only(r, a, b):
    return int(a) + int(b) <= int(r["pred_a"]) + int(r["pred_b"])

def h_filter_no_total_increase(r, a, b):
    return int(a) + int(b) <= int(r["pred_a"]) + int(r["pred_b"])

def h_filter_no_w_high_total(r, a, b):
    if str(r.get("gender", "")).upper() == "W" and max(int(a) + int(b), int(r["pred_a"]) + int(r["pred_b"])) >= 4:
        return False
    return True

H4_CONFIGS = [
    ("h4_no_gd_delta_gt_1_cap0275", h_filter_no_gd_delta_gt_1, False, False, 2),
    ("h4_same_or_lower_total_only_cap0275", h_filter_same_or_lower_total_only, False, False, 2),
    ("h4_no_total_increase_cap0275", h_filter_no_total_increase, False, False, 2),
    ("h4_exclude_low_support_high_weight_cap0275", None, False, False, lambda r: 3 if float(r.get("tournament_weight", 1.2)) >= 1.8 else 2),
    ("h4_exclude_exp17_source_cap0275", None, False, True, 2),
    ("h4_trusted_source_only_cap0275", None, True, False, 2),
    ("h4_no_w_high_total_cap0275", h_filter_no_w_high_total, False, False, 2),
    ("h4_no_high_weight_unless_support3_cap0275", None, False, False, lambda r: 3 if float(r.get("tournament_weight", 1.2)) >= 1.8 else 2),
]
h4_records = []
for label, cand_filter, trusted_only, exclude_exp17, support_thr in H4_CONFIGS:
    sub, changes = apply_outcome_booster(
        base_submission,
        label,
        cap_rate=0.0275,
        support_threshold=support_thr,
        candidate_filter=cand_filter,
        trusted_only=trusted_only,
        exclude_exp17=exclude_exp17,
    )
    path, check = save_candidate(sub, label, strict=False)
    rec = compare_to_base(sub, base_submission, label)
    rec.update({"cap_rate": 0.0275, "trusted_only": trusted_only, "exclude_exp17": exclude_exp17, "path": str(path), "pair_consistency": bool(check.loc[check["check"] == "pair_consistency", "passed"].iloc[0])})
    if changes is not None and len(changes):
        rec["n_candidates_after_filter"] = int(len(changes))
    h4_records.append(rec)
    ALL_CANDIDATE_SUBMISSIONS[label] = sub
    ALL_CHANGE_DETAILS[label] = changes

damage_control_grid = pd.DataFrame(h4_records)
damage_control_grid.to_csv(SUM_DIR / "damage_control_grid.csv", index=False)
log_saved(SUM_DIR / "damage_control_grid.csv")
display(damage_control_grid)
damage_control_change_analysis = transition_table({r["variant"]: ALL_CHANGE_DETAILS[r["variant"]] for r in h4_records})
damage_control_change_analysis.to_csv(SUM_DIR / "damage_control_change_analysis.csv", index=False)
log_saved(SUM_DIR / "damage_control_change_analysis.csv")

# %% [markdown]
# # 11. H5 — Transition-Specific Cap0275
#
# Penjelasan bagian:
# H5 mencari apakah transisi tertentu di cap0275 adalah sumber noise. Fokusnya 0-0, 1-1, 2-2, low-total, dan kombinasi exclude.
#
# Output yang perlu dilihat:
# `transition_cap0275_grid.csv` dan `transition_cap0275_analysis.csv`. Jika exclude 2-2 lebih baik di audit, 2-2 transition harus dihindari berikutnya.

# %%
def score_filter(a=None, b=None, not_score=None, max_total=None, allowed_scores=None, excluded_scores=None):
    def _f(r):
        pa, pb = int(r["pred_a"]), int(r["pred_b"])
        if a is not None and b is not None and not (pa == a and pb == b):
            return False
        if not_score is not None and (pa, pb) == tuple(not_score):
            return False
        if allowed_scores is not None and (pa, pb) not in set(tuple(x) for x in allowed_scores):
            return False
        if excluded_scores is not None and (pa, pb) in set(tuple(x) for x in excluded_scores):
            return False
        if max_total is not None and (pa + pb) > int(max_total):
            return False
        return True
    return _f

H5_CONFIGS = [
    ("h5_no_0_0_at_cap0275", score_filter(excluded_scores=[(0, 0)])),
    ("h5_no_2_2_at_cap0275", score_filter(excluded_scores=[(2, 2)])),
    ("h5_only_1_1_at_cap0275", score_filter(1, 1)),
    ("h5_low_total_only_cap0275", score_filter(max_total=2)),
    ("h5_0_0_plus_1_1_only_cap0275", score_filter(allowed_scores=[(0, 0), (1, 1)])),
    ("h5_1_1_plus_2_2_only_cap0275", score_filter(allowed_scores=[(1, 1), (2, 2)])),
    ("h5_exclude_0_0_and_2_2_cap0275", score_filter(excluded_scores=[(0, 0), (2, 2)])),
]
h5_records = []
for label, filt in H5_CONFIGS:
    sub, changes = apply_outcome_booster(base_submission, label, cap_rate=0.0275, support_threshold=2, extra_filter=filt)
    path, check = save_candidate(sub, label, strict=False)
    rec = compare_to_base(sub, base_submission, label)
    rec.update({"cap_rate": 0.0275, "path": str(path), "pair_consistency": bool(check.loc[check["check"] == "pair_consistency", "passed"].iloc[0])})
    h5_records.append(rec)
    ALL_CANDIDATE_SUBMISSIONS[label] = sub
    ALL_CHANGE_DETAILS[label] = changes

transition_cap0275_grid = pd.DataFrame(h5_records)
transition_cap0275_grid.to_csv(SUM_DIR / "transition_cap0275_grid.csv", index=False)
log_saved(SUM_DIR / "transition_cap0275_grid.csv")
display(transition_cap0275_grid)
transition_cap0275_analysis = transition_table({r["variant"]: ALL_CHANGE_DETAILS[r["variant"]] for r in h5_records})
transition_cap0275_analysis.to_csv(SUM_DIR / "transition_cap0275_analysis.csv", index=False)
log_saved(SUM_DIR / "transition_cap0275_analysis.csv")

# %% [markdown]
# # 12. H6 — Tournament/Gender Cap0275
#
# Penjelasan bagian:
# H6 menguji apakah cap0275 harus berbeda untuk high-weight tournament atau gender M/W. Ini menjaga agar perubahan outcome tidak terlalu merusak match mahal atau domain W yang lebih noisy.
#
# Output yang perlu dilihat:
# `tournament_gender_cap0275_grid.csv` dan `tournament_gender_change_breakdown.csv`. Cek changed_high_weight, changed_M, dan changed_W.

# %%
def is_world_asian_tournament(t: str) -> bool:
    s = str(t).lower()
    return "world cup" in s or "asian" in s or "afc" in s

H6_CONFIGS = [
    ("h6_no_high_weight_cap0275", 0.0275, lambda r: float(r.get("tournament_weight", 1.2)) < 1.8, 2, None),
    ("h6_high_weight_support3_cap0275", 0.0275, None, lambda r: 3 if float(r.get("tournament_weight", 1.2)) >= 1.8 else 2, None),
    ("h6_high_weight_support4_cap0275", 0.0275, None, lambda r: 4 if float(r.get("tournament_weight", 1.2)) >= 1.8 else 2, None),
    ("h6_low_weight_cap030_high_weight_cap020", 0.0275, None, lambda r: 3 if float(r.get("tournament_weight", 1.2)) >= 1.8 else 2, None),
    ("h6_world_asian_strict_support3_cap0275", 0.0275, None, lambda r: 3 if is_world_asian_tournament(r.get("tournament", "")) else 2, None),
    ("h6_friendly_extra_only_cap0275", 0.0100, lambda r: float(r.get("tournament_weight", 1.2)) <= 0.96, 2, None),
    ("h6_m_only_cap0275", 0.0275, lambda r: str(r.get("gender", "")).upper() == "M", 2, None),
    ("h6_m_cap0275_w_cap000", 0.0275, None, 2, {"M": 0.0275, "W": 0.0000}),
    ("h6_m_cap0275_w_cap005", 0.0275, None, 2, {"M": 0.0275, "W": 0.0050}),
    ("h6_m_cap0275_w_cap010", 0.0275, None, 2, {"M": 0.0275, "W": 0.0100}),
    ("h6_m_support2_w_support3_cap0275", 0.0275, None, lambda r: 3 if str(r.get("gender", "")).upper() == "W" else 2, None),
]
h6_records = []
for label, cap, filt, support_thr, cap_by_gender in H6_CONFIGS:
    sub, changes = apply_outcome_booster(base_submission, label, cap_rate=cap, support_threshold=support_thr, extra_filter=filt, cap_rate_by_gender=cap_by_gender)
    path, check = save_candidate(sub, label, strict=False)
    rec = compare_to_base(sub, base_submission, label)
    rec.update({"cap_rate": cap, "path": str(path), "pair_consistency": bool(check.loc[check["check"] == "pair_consistency", "passed"].iloc[0])})
    h6_records.append(rec)
    ALL_CANDIDATE_SUBMISSIONS[label] = sub
    ALL_CHANGE_DETAILS[label] = changes

tournament_gender_cap0275_grid = pd.DataFrame(h6_records)
tournament_gender_cap0275_grid.to_csv(SUM_DIR / "tournament_gender_cap0275_grid.csv", index=False)
log_saved(SUM_DIR / "tournament_gender_cap0275_grid.csv")
display(tournament_gender_cap0275_grid)

tg_rows = []
for row in h6_records:
    label = row["variant"]
    ch = ALL_CHANGE_DETAILS.get(label, pd.DataFrame())
    if ch is None or len(ch) == 0:
        tg_rows.append({"variant": label, "bucket": "none", "n_changed": 0})
        continue
    tmp = ch.copy()
    tmp["weight_bucket"] = np.where(tmp["tournament_weight"] >= 1.8, "high_weight", np.where(tmp["tournament_weight"] <= 0.96, "low_weight", "medium_weight"))
    for b, g in tmp.groupby("weight_bucket"):
        tg_rows.append({"variant": label, "bucket": b, "n_changed": len(g)})
    for b, g in tmp.groupby("gender"):
        tg_rows.append({"variant": label, "bucket": f"gender_{b}", "n_changed": len(g)})
tournament_gender_change_breakdown = pd.DataFrame(tg_rows)
tournament_gender_change_breakdown.to_csv(SUM_DIR / "tournament_gender_change_breakdown.csv", index=False)
log_saved(SUM_DIR / "tournament_gender_change_breakdown.csv")

# %% [markdown]
# # 13. H7 — Tiny Exact/GD Repair After Cap0275
#
# Penjelasan bagian:
# H7 menambahkan repair sangat kecil setelah cap0275. Exact repair mencoba scoreline candidate yang sama outcome dan dekat secara total/GD. GD repair mencoba scoreline candidate same outcome dengan GD dekat, tanpa menyentuh match yang sudah diubah outcome booster.
#
# Output yang perlu dilihat:
# `tiny_repair_after_cap0275_grid.csv`. Additional changed rate harus kecil dan pair consistency wajib True.

# %%
def apply_gd_repair_booster(base_after_outcome_sub, original_base_sub, variant, cap_rate=0.0025, support_threshold=2):
    base_m = submission_to_match_df(base_after_outcome_sub, label="gd_base")
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
        scores = candidate_scores_for_match(r["match_id"], int(r["pred_a"]), int(r["pred_b"]))
        base_out = outcome_scalar(int(r["pred_a"]), int(r["pred_b"]))
        base_total = int(r["pred_a"]) + int(r["pred_b"])
        base_gd = int(r["pred_a"]) - int(r["pred_b"])
        scores = [s for s in scores if s["outcome"] == base_out and abs(s["total"] - base_total) <= 1 and abs(s["gd"] - base_gd) <= 1]
        if not scores:
            continue
        by_score = defaultdict(list)
        for s in scores:
            by_score[(s["a"], s["b"])].append(s)
        best = None
        for (a, b), ss in by_score.items():
            sup = len(ss)
            if sup < support_threshold:
                continue
            wsup = float(sum(x["weight"] for x in ss))
            gd_delta = abs((a - b) - base_gd)
            total_delta = abs((a + b) - base_total)
            score = (sup, wsup, -gd_delta, -total_delta)
            if best is None or score > best[0]:
                best = (score, a, b, sup, ss[0]["label"])
        if best is None:
            continue
        _, a, b, sup, src = best
        candidates.append({"match_id": r["match_id"], "old_a": int(r["pred_a"]), "old_b": int(r["pred_b"]), "new_a": int(a), "new_b": int(b), "support": float(sup), "source_label": src, "gender": str(r.get("gender", "unknown")).upper(), "tournament_weight": float(r.get("tournament_weight", 1.2))})
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

h7_records = []
h7_base_candidates = [lab for lab in ["h1_cap0275", "h1_cap0280", "h2_support3_cap0275", "h3_reliability_then_support_cap0275"] if lab in ALL_CANDIDATE_SUBMISSIONS]
if not h7_base_candidates:
    h7_base_candidates = ["h0_cap0275_reproduction"]

for base_label in h7_base_candidates:
    for exact_cap in [0.0025, 0.0050, 0.0075, 0.0100]:
        label = f"h7_{base_label}_plus_exact_cap{int(round(exact_cap * 10000)):04d}"
        sub, changes = apply_exact_booster(ALL_CANDIDATE_SUBMISSIONS[base_label], base_submission, label, cap_rate=exact_cap, support_threshold=2)
        path, check = save_candidate(sub, label, strict=False)
        rec = compare_to_base(sub, base_submission, label)
        rec.update({"base_outcome_variant": base_label, "repair_type": "exact", "repair_cap": exact_cap, "path": str(path), "pair_consistency": bool(check.loc[check["check"] == "pair_consistency", "passed"].iloc[0])})
        h7_records.append(rec)
        ALL_CANDIDATE_SUBMISSIONS[label] = sub
        ALL_CHANGE_DETAILS[label] = changes
    for gd_cap in [0.0025, 0.0050]:
        label = f"h7_{base_label}_plus_gdrepair_cap{int(round(gd_cap * 10000)):04d}"
        sub, changes = apply_gd_repair_booster(ALL_CANDIDATE_SUBMISSIONS[base_label], base_submission, label, cap_rate=gd_cap, support_threshold=2)
        path, check = save_candidate(sub, label, strict=False)
        rec = compare_to_base(sub, base_submission, label)
        rec.update({"base_outcome_variant": base_label, "repair_type": "gd", "repair_cap": gd_cap, "path": str(path), "pair_consistency": bool(check.loc[check["check"] == "pair_consistency", "passed"].iloc[0])})
        h7_records.append(rec)
        ALL_CANDIDATE_SUBMISSIONS[label] = sub
        ALL_CHANGE_DETAILS[label] = changes

tiny_repair_after_cap0275_grid = pd.DataFrame(h7_records)
tiny_repair_after_cap0275_grid.to_csv(SUM_DIR / "tiny_repair_after_cap0275_grid.csv", index=False)
log_saved(SUM_DIR / "tiny_repair_after_cap0275_grid.csv")
if len(tiny_repair_after_cap0275_grid):
    display(tiny_repair_after_cap0275_grid)
tiny_repair_transition_analysis = transition_table({r["variant"]: ALL_CHANGE_DETAILS[r["variant"]] for r in h7_records})
tiny_repair_transition_analysis.to_csv(SUM_DIR / "tiny_repair_transition_analysis.csv", index=False)
log_saved(SUM_DIR / "tiny_repair_transition_analysis.csv")
# %% [markdown]
# # 14. Combined Sanity Metrics dan Scoreline Distribution
#
# Penjelasan bagian:
# Bagian ini menggabungkan semua candidate H0-H7 ke tabel sanity GT-free. Ini membantu mendeteksi candidate yang terlalu agresif sebelum audit lokal.
#
# Output yang perlu dilihat:
# `changed_prediction_analysis.csv`, `variant_metrics.csv`, dan `scoreline_distribution.csv`. Changed rate besar harus dicurigai.

# %%
combined_rows = []
for label, sub in ALL_CANDIDATE_SUBMISSIONS.items():
    rec = compare_to_base(sub, base_submission, label)
    pair_ok, n_bad, _ = pair_consistency_report(sub)
    rec.update({"pair_consistency": pair_ok, "n_bad_pairs": n_bad})
    combined_rows.append(rec)

changed_prediction_analysis = pd.DataFrame(combined_rows).sort_values("changed_rate")
changed_prediction_analysis.to_csv(SUM_DIR / "changed_prediction_analysis.csv", index=False)
changed_prediction_analysis.to_csv(SUM_DIR / "variant_metrics.csv", index=False)
log_saved(SUM_DIR / "changed_prediction_analysis.csv")
log_saved(SUM_DIR / "variant_metrics.csv")
display(changed_prediction_analysis)

score_rows = []
for label, sub in ALL_CANDIDATE_SUBMISSIONS.items():
    m = submission_to_match_df(sub, label=label)
    vc = (m["pred_a"].astype(int).astype(str) + "-" + m["pred_b"].astype(int).astype(str)).value_counts().head(15)
    for rank, (sc, count) in enumerate(vc.items(), 1):
        score_rows.append({"variant": label, "rank": rank, "scoreline": sc, "count": int(count), "share": float(count / len(m))})
scoreline_distribution = pd.DataFrame(score_rows)
scoreline_distribution.to_csv(SUM_DIR / "scoreline_distribution.csv", index=False)
log_saved(SUM_DIR / "scoreline_distribution.csv")

transition_all = transition_table(ALL_CHANGE_DETAILS)
transition_all.to_csv(SUM_DIR / "transition_all_variants.csv", index=False)
log_saved(SUM_DIR / "transition_all_variants.csv")

# %% [markdown]
# # 15. H8 — Final Selected Safe
#
# Penjelasan bagian:
# Bagian ini memilih `best_safe` secara GT-free. Default-nya konservatif, sehingga champion sebenarnya tetap dibaca dari Section 99 local audit.
#
# Output yang perlu dilihat:
# `final_decision.csv`, `submission_catalog.csv`, dan `submission_check.csv`. `best_safe` bukan klaim best local audit.

# %%
preferred_order = [
    "h1_cap0275",
    "h2_support3_cap0275",
    "h1_cap0270",
    "h3_reliability_then_support_cap0275",
    "h6_high_weight_support3_cap0275",
    "h6_m_cap0275_w_cap005",
    "h0_cap0275_reproduction",
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
    selected_label = "h0_cap0275_reproduction"

best_path, best_check = save_candidate(ALL_CANDIDATE_SUBMISSIONS[selected_label], "best_safe", strict=True)

submission_check_df = pd.concat(ALL_SUBMISSION_CHECKS, ignore_index=True) if ALL_SUBMISSION_CHECKS else pd.DataFrame()
submission_check_df.to_csv(SUM_DIR / "submission_check.csv", index=False)
log_saved(SUM_DIR / "submission_check.csv")

submission_catalog_df = pd.DataFrame(SUBMISSION_CATALOG).drop_duplicates("path")
submission_catalog_df.to_csv(SUM_DIR / "submission_catalog.csv", index=False)
log_saved(SUM_DIR / "submission_catalog.csv")

final_decision = {
    "experiment": "EXP12H",
    "selected_by_pipeline": selected_label,
    "selected_by_sanity": selected_label,
    "recommended_for_local_audit": "all generated submission_exp12h_*.csv; champion is decided manually from Section 99 local audit",
    "main_submission_path": str(best_path),
    "h0_base_label": str(champ_label),
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
