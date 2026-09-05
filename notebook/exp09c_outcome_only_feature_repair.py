# %% [markdown]
# # 00. EXP09C — Ringkasan Eksperimen
#
# Bagian ini menjelaskan arah eksperimen sebelum kode dijalankan. EXP09C bukan rewrite dari nol, tetapi turunan dari pipeline EXP05A-LITE-FIX-V2 dengan satu ide utama: fitur repair dari EXP09A hanya diuji pada outcome head.
#
# ## Inti eksperimen
#
# - **Goal head** tetap memakai fitur asli EXP05A supaya prediksi jumlah gol tidak ikut terdorong oleh fitur domain-shift yang sebelumnya membuat exact/GD memburuk.
# - **Outcome head** boleh memakai fitur EXP05A + fitur repair EXP09A karena dari eksperimen sebelumnya fitur repair terlihat lebih membantu arah pertandingan.
# - **Tail head, decoder, dan submission mapping** tetap mengikuti gaya EXP05A supaya perubahan eksperimen tetap kecil dan mudah diaudit.
#
# ## Variant yang dibandingkan
#
# - **V0_baseline**: semua head memakai fitur EXP05A asli.
# - **V1_allhead**: semua head memakai fitur EXP05A + EXP09A. Variant ini hanya diagnostic, bukan final utama.
# - **V2_outcome_only**: goal/tail memakai EXP05A, outcome memakai EXP05A + EXP09A. Ini variant utama EXP09C.
# - **V3_outcome_tail**: opsional, hanya dijalankan kalau runtime masih aman.
# %% [markdown]
# # 01. Setup, Seed, Path, dan Guardrail
#
# Bagian ini menyiapkan environment eksperimen: import library, seed, path data/output, mode runtime, dan guardrail. Guardrail dibuat eksplisit supaya eksperimen tidak diam-diam memakai GT, external data, TabPFN, recursive update, atau progress bar `tqdm`.
#
# Output penting dari tahap ini adalah folder `outputs/exp09c_outcome_only_feature_repair/` beserta subfolder `figures`, `predictions`, `submissions`, dan `summaries`. Selain itu, logging dibuat ringkas agar run notebook tetap bisa dipantau tanpa raw training log yang terlalu panjang.
# %%
import os, json, math, random, warnings, copy
from pathlib import Path
from collections import defaultdict, deque
from itertools import product
import time
from contextlib import contextmanager

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
try:
    from IPython.display import display
except Exception:
    def display(x): print(x)
from sklearn.metrics import accuracy_score, mean_absolute_error

warnings.filterwarnings('ignore')
SEED = 42

def seed_everything(seed: int = 42) -> None:
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

seed_everything(SEED)
pd.set_option('display.max_columns', 220)
pd.set_option('display.width', 180)

PROJECT_ROOT = Path.cwd()

if PROJECT_ROOT.name.lower() in {'notebook', 'notebooks'}:
    PROJECT_ROOT = PROJECT_ROOT.parent
elif not (PROJECT_ROOT / 'data' / 'train.csv').exists() and (PROJECT_ROOT.parent / 'data' / 'train.csv').exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

DATA_DIR = PROJECT_ROOT / 'data'
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'exp09c_outcome_only_feature_repair'
FIG_DIR = OUTPUT_DIR / 'figures'
PRED_DIR = OUTPUT_DIR / 'predictions'
SUB_DIR = OUTPUT_DIR / 'submissions'
SUM_DIR = OUTPUT_DIR / 'summaries'
for d in [OUTPUT_DIR, FIG_DIR, PRED_DIR, SUB_DIR, SUM_DIR]:
    d.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATA_DIR / 'train.csv'
TEST_PATH = DATA_DIR / 'test.csv'
SAMPLE_SUB_PATH = DATA_DIR / 'sample submission.csv'
META_PATH = DATA_DIR / 'metadata.txt'

SCREEN_N_TRIALS = 10
REFINE_N_TRIALS = 20
BOOSTERS_TO_RUN = ['lgb', 'cat']
XGB_POLICY = 'skip'
USE_TABPFN = False
USE_RECURSIVE = False
USE_TEACHER_STUDENT = False
USE_PRETRAINED_MODEL = False
USE_EXTERNAL_DATA = False
USE_PROGRESS_BAR = False
MODEL_VERBOSE = False
LGB_EARLY_STOP_VERBOSE = False
NOTEBOOK_START_TIME = time.time()
SECTION_TIMES = {}

def log_section(title: str):
    print('\n' + '=' * 100, flush=True)
    print(f'[SECTION] {title}', flush=True)
    print('=' * 100, flush=True)

def log_info(msg: str):
    print(f'[INFO] {msg}', flush=True)

def log_check(name: str, passed: bool, detail: str = ''):
    status = 'PASS' if bool(passed) else 'FAIL'
    suffix = f' | {detail}' if detail else ''
    print(f'[CHECK] {name}: {status}{suffix}', flush=True)

def log_warn(msg: str):
    print(f'[WARN] {msg}', flush=True)

def log_result(msg: str):
    print(f'[RESULT] {msg}', flush=True)

def log_saved(path):
    print(f'[SAVED] {path}', flush=True)

@contextmanager
def timed_section(name: str):
    start = time.time()
    log_section(name)
    try:
        yield
    finally:
        elapsed = time.time() - start
        SECTION_TIMES[name] = elapsed
        print(f'[DONE] {name} finished in {elapsed:.2f}s', flush=True)

class ProgressLogger:
    def __init__(self, total, name='process', every=None, min_interval=5.0):
        self.total = int(total)
        self.name = str(name)
        self.every = int(every or max(1, self.total // 10))
        self.min_interval = float(min_interval)
        self.start = time.time()
        self.last_print = self.start
        self.current = 0
        print(f'[PROGRESS START] {self.name} | total={self.total:,}', flush=True)
    def update(self, current=None, step=1):
        if current is None:
            self.current += step
        else:
            self.current = int(current)
        now = time.time()
        if (self.current % self.every == 0) or ((now - self.last_print) >= self.min_interval) or (self.current >= self.total):
            elapsed = now - self.start
            speed = self.current / elapsed if elapsed > 0 else 0
            remaining = self.total - self.current
            eta = remaining / speed if speed > 0 else 0
            pct = 100 * self.current / self.total if self.total else 100
            print(f'[PROGRESS] {self.name} | {self.current:,}/{self.total:,} | {pct:.1f}% | elapsed={elapsed:.1f}s | speed={speed:.1f} rows/s | eta={eta:.1f}s', flush=True)
            self.last_print = now
    def close(self):
        elapsed = time.time() - self.start
        print(f'[PROGRESS DONE] {self.name} | total={self.total:,} | elapsed={elapsed:.1f}s', flush=True)

print('[INFO] output:', OUTPUT_DIR.resolve())
log_info('EXP09C source of truth: EXP05A-LITE-FIX-V2. EXP09A is used only for feature repair functions.')
print('[INFO] XGBoost policy:', XGB_POLICY)
log_check('TabPFN disabled', not USE_TABPFN)
log_check('Recursive update disabled', not USE_RECURSIVE)
log_check('Teacher-student disabled', not USE_TEACHER_STUDENT)
log_check('External data disabled', not USE_EXTERNAL_DATA)
log_check('Progress bar package disabled', not USE_PROGRESS_BAR)

RUN_V3_OPTIONAL = False

def iter_progress(iterable, total=None, desc='process'):
    """Small manual progress wrapper."""
    if total is None:
        try:
            total = len(iterable)
        except Exception:
            total = 0
    progress = ProgressLogger(total or 0, name=desc)
    for i, item in enumerate(iterable, start=1):
        yield item
        progress.update(i)
    progress.close()


# %% [markdown]
# # 02. Load Data dan Basic Audit
#
# Tahap ini membaca file input resmi kompetisi: `train.csv`, `test.csv`, `sample submission.csv`, dan `metadata.txt`. Di sini juga dilakukan audit awal seperti shape data, range tanggal, alignment `Id` test dengan sample submission, dan pengecekan file terlarang seperti `test_with_groundtruth.csv`.
#
# Tujuan bagian ini adalah memastikan notebook berjalan di input legal saja sebelum masuk ke canonicalization dan feature engineering.
# %%
def validate_input_files(file_paths: dict) -> None:
    missing = [(name, str(path)) for name, path in file_paths.items() if not Path(path).exists()]
    if missing:
        msg = '\n'.join([f'- {n}: {p}' for n, p in missing])
        raise FileNotFoundError('Missing input files:\n' + msg)

validate_input_files({'train': TRAIN_PATH, 'test': TEST_PATH, 'sample_submission': SAMPLE_SUB_PATH, 'metadata': META_PATH})
train_raw = pd.read_csv(TRAIN_PATH, parse_dates=['date'])
test_raw = pd.read_csv(TEST_PATH, parse_dates=['date'])
sample_submission = pd.read_csv(SAMPLE_SUB_PATH)
with open(META_PATH, 'r', encoding='utf-8') as f:
    metadata_text = f.read()
print('[INFO] train:', train_raw.shape, '| test:', test_raw.shape, '| sample:', sample_submission.shape)
display(train_raw.head(2))


# %% [markdown]
# # 03. Core Helper dari EXP05A
#
# Bagian ini berisi helper inti yang dipakai sepanjang pipeline, seperti normalisasi kolom, fungsi metric AW-MAE, tournament weight, outcome label, dan summary evaluasi. Helper ini dipertahankan agar perilaku evaluasi tetap sejalan dengan EXP05A.
#
# Fungsi-fungsi di bagian ini tidak memakai target test dan tidak membaca output eksperimen lama. Semua perhitungan validasi dilakukan dari train/validation fold yang legal.
# %%
def select_unique_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    seen, ordered = set(), []
    for c in cols:
        if c not in seen:
            ordered.append(c); seen.add(c)
    missing = [c for c in ordered if c not in df.columns]
    if missing:
        raise KeyError(f'select_unique_columns missing: {missing[:20]}')
    out = df.loc[:, ordered].copy()
    assert len(out.columns) == len(pd.Index(out.columns).unique()), 'Duplicate columns after select_unique_columns'
    return out

def assert_no_duplicate_columns(df: pd.DataFrame, context: str) -> None:
    if len(df.columns) != len(pd.Index(df.columns).unique()):
        dupes = pd.Index(df.columns)[pd.Index(df.columns).duplicated()].tolist()
        raise AssertionError(f'{context} duplicate columns: {dupes}')

def safe_to_string(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].astype('string').fillna('__MISSING__')
    return out

def safe_numeric(x):
    return pd.to_numeric(x, errors='coerce')

def safe_divide(a, b):
    out = pd.to_numeric(a, errors='coerce') / pd.to_numeric(b, errors='coerce')
    return out.replace([np.inf, -np.inf], np.nan)

def _outcome(a: int, b: int) -> int:
    a, b = int(a), int(b)
    return 0 if a > b else (1 if a == b else 2)

def get_tournament_weight(tournament: str) -> float:
    t = str(tournament).strip().lower()
    if ('fifa world cup' in t) or (t == 'world cup'):
        return 2.00
    if ('afc championship' in t) or ('afc asian cup' in t) or ('asian cup' in t):
        return 1.80
    if 'friendly' in t:
        return 0.96
    return 1.20

EXACT_PENALTY = 0.30
OUTCOME_PENALTY = 0.25
GD_PENALTY = 0.15
WRONG_OUTCOME_MULTIPLIER = 1.50
NONLINEAR_POWER = 1.50

def official_match_loss(y_team_true, y_opp_true, y_team_pred, y_opp_pred) -> float:
    y_team_true, y_opp_true, y_team_pred, y_opp_pred = map(int, [y_team_true, y_opp_true, y_team_pred, y_opp_pred])
    mae = (abs(y_team_true-y_team_pred) + abs(y_opp_true-y_opp_pred)) / 2
    exact = int(y_team_true == y_team_pred and y_opp_true == y_opp_pred)
    outcome_ok = int(_outcome(y_team_true, y_opp_true) == _outcome(y_team_pred, y_opp_pred))
    gd_ok = int((y_team_true-y_opp_true) == (y_team_pred-y_opp_pred))
    penalty = EXACT_PENALTY*(1-exact) + OUTCOME_PENALTY*(1-outcome_ok) + GD_PENALTY*(1-gd_ok)
    multiplier = 1.0 if outcome_ok else WRONG_OUTCOME_MULTIPLIER
    return float(((mae + penalty) * multiplier) ** NONLINEAR_POWER)

def awmae_score(y_team_true, y_opp_true, y_team_pred, y_opp_pred, tournaments) -> float:
    losses = np.array([official_match_loss(a, b, pa, pb) for a,b,pa,pb in zip(y_team_true, y_opp_true, y_team_pred, y_opp_pred)], dtype=float)
    weights = np.array([get_tournament_weight(t) for t in tournaments], dtype=float)
    return float(np.sum(losses * weights) / np.sum(weights))

def clean_row_level(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if 'altitude_venue' in out.columns:
        out.loc[out['altitude_venue'] == -9999, 'altitude_venue'] = np.nan
    if 'date' in out.columns:
        out['date'] = pd.to_datetime(out['date'], errors='coerce')
    return safe_to_string(out, ['team','opponent','gender','tournament','venue_country','confederation_team','confederation_opp'])

def build_match_level(df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
    df = df.copy()
    required = ['match_id','team','opponent','date','gender','tournament']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f'build_match_level missing: {missing}')
    bad_counts = df['match_id'].value_counts().loc[lambda s: s != 2]
    if len(bad_counts):
        raise AssertionError(f'Match id not two rows. Example: {bad_counts.head().to_dict()}')
    rows = []
    shared = ['date','gender','tournament','venue_country','neutral','altitude_venue','temperature_venue']
    side_map = {
        'is_home': 'is_home',
        'confederation': 'confederation_team',
        'population': 'population_team',
        'gdp_per_capita': 'gdp_per_capita_team',
        'distance_travel': 'distance_travel_team',
    }
    for mid, grp in iter_progress(df.groupby('match_id', sort=False), total=df['match_id'].nunique(), desc='build_match_level'):
        pair = grp.sort_values(['team','opponent']).reset_index(drop=True)
        a, b = pair.iloc[0], pair.iloc[1]
        row = {'match_id': mid, 'team_a': a['team'], 'team_b': b['team']}
        for c in shared:
            if c in pair.columns:
                vals = pair[c].dropna().unique().tolist()
                row[c] = vals[0] if vals else np.nan
        if 'Id' in pair.columns:
            row['row_id_a'], row['row_id_b'] = a['Id'], b['Id']
        for name, col in side_map.items():
            row[f'team_a_{name}'] = a[col] if col in pair.columns else np.nan
            row[f'team_b_{name}'] = b[col] if col in pair.columns else np.nan
        if is_train:
            row['team_a_goals'] = a['team_goals']
            row['team_b_goals'] = b['team_goals']
        rows.append(row)
    out = pd.DataFrame(rows).sort_values(['date','match_id']).reset_index(drop=True)
    assert_no_duplicate_columns(out, 'match_level')
    return out

def make_time_based_holdout(train_match: pd.DataFrame, valid_fraction: float = 0.2):
    df = train_match.sort_values(['date','match_id']).reset_index(drop=True)
    n_valid = max(1, int(len(df)*valid_fraction))
    tr, va = df.iloc[:-n_valid].copy(), df.iloc[-n_valid:].copy()
    assert set(tr['match_id']).isdisjoint(set(va['match_id']))
    print('[INFO] train fold:', tr.shape, tr['date'].min(), '->', tr['date'].max())
    print('[INFO] valid fold:', va.shape, va['date'].min(), '->', va['date'].max())
    return tr, va


# %% [markdown]
# # 04. Canonical Match-Level Builder
#
# Dataset kompetisi menyimpan satu pertandingan dalam dua baris, yaitu sudut pandang masing-masing tim. Bagian ini mengubah data row-level menjadi match-level dengan format `team_a` dan `team_b`.
#
# Output utama tahap ini adalah `train_match_base` dan `test_match_base`. Struktur ini penting karena model, decoder, evaluasi, dan reverse mapping submission bekerja lebih aman pada level pertandingan.
# %%
train_clean = clean_row_level(train_raw)
test_clean = clean_row_level(test_raw)
train_match_base = build_match_level(train_clean, is_train=True)
test_match_base = build_match_level(test_clean, is_train=False)
print('[INFO] train_match_base:', train_match_base.shape)
print('[INFO] test_match_base :', test_match_base.shape)
display(train_match_base.head(3))


# %% [markdown]
# # 05. Static + History Features dari EXP05A
#
# Bagian ini membangun fitur utama EXP05A: fitur statis, fitur tanggal, fitur kategori tim/turnamen, fitur confederation jika tersedia, rolling/history features, H2H, serta rating-like features yang dihitung secara legal dari histori sebelumnya.
#
# Ini adalah backbone EXP05A yang harus dipertahankan. EXP09C tidak mengganti cara history utama dibangun, karena target eksperimen hanya menguji pemakaian fitur repair pada outcome head.
# %%
def engineer_static_match_features(df_match: pd.DataFrame) -> pd.DataFrame:
    df = df_match.copy()
    df = safe_to_string(df, ['team_a','team_b','gender','tournament','venue_country','team_a_confederation','team_b_confederation'])
    df['pair_key'] = df['team_a'].astype(str) + '__VS__' + df['team_b'].astype(str)
    df['confed_pair_key'] = df['team_a_confederation'].astype(str) + '__VS__' + df['team_b_confederation'].astype(str)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['match_year'] = df['date'].dt.year
    df['match_month'] = df['date'].dt.month
    df['match_quarter'] = df['date'].dt.quarter
    df['match_dayofweek'] = df['date'].dt.dayofweek
    df['match_dayofyear'] = df['date'].dt.dayofyear
    df['match_is_weekend'] = (df['match_dayofweek'] >= 5).astype(int)
    df['match_decade'] = (df['match_year'] // 10) * 10
    neutral = safe_numeric(df.get('neutral', 0)).fillna(0)
    home_a = safe_numeric(df.get('team_a_is_home', 0)).fillna(0)
    home_b = safe_numeric(df.get('team_b_is_home', 0)).fillna(0)
    df['home_side'] = np.where(neutral == 1, 0, np.where(home_a == 1, 1, np.where(home_b == 1, -1, 0)))
    df['same_confederation'] = (df['team_a_confederation'].astype(str) == df['team_b_confederation'].astype(str)).astype(int)
    t = df['tournament'].astype(str).str.lower()
    df['is_friendly'] = t.str.contains('friendly').astype(int)
    df['is_world_cup'] = (t.str.contains('fifa world cup') | (t == 'world cup')).astype(int)
    df['is_qualification'] = t.str.contains('qual').astype(int)
    df['is_nations_league'] = t.str.contains('nations').astype(int)
    df['tournament_weight_proxy'] = df['tournament'].map(get_tournament_weight)
    for base, ca, cb in [('population','team_a_population','team_b_population'), ('gdp_per_capita','team_a_gdp_per_capita','team_b_gdp_per_capita'), ('distance_travel','team_a_distance_travel','team_b_distance_travel')]:
        df[ca], df[cb] = safe_numeric(df.get(ca)), safe_numeric(df.get(cb))
        df[f'{base}_diff'] = df[ca] - df[cb]
        df[f'{base}_abs_diff'] = (df[ca] - df[cb]).abs()
        df[f'log_{base}_a'] = np.log1p(df[ca].clip(lower=0))
        df[f'log_{base}_b'] = np.log1p(df[cb].clip(lower=0))
        df[f'log_{base}_diff'] = df[f'log_{base}_a'] - df[f'log_{base}_b']
        df[f'{base}_ratio_ab'] = safe_divide(df[ca], df[cb])
    for c in ['altitude_venue','temperature_venue','neutral']:
        if c in df.columns:
            df[c] = safe_numeric(df[c])
    assert_no_duplicate_columns(df, 'static_features')
    return df

def init_team_state():
    return {'matches_played':0, 'last_match_date':pd.NaT, 'elo_overall':1500.0, 'elo_goal_diff':0.0, 'ewm_points':1.0, 'ewm_gf':1.2, 'ewm_ga':1.2, 'ewm_gd':0.0,
            'points_last5':deque(maxlen=5), 'points_last10':deque(maxlen=10), 'gf_last5':deque(maxlen=5), 'ga_last5':deque(maxlen=5), 'gd_last5':deque(maxlen=5),
            'results_last10':deque(maxlen=10), 'clean_sheet_last5':deque(maxlen=5), 'failed_to_score_last5':deque(maxlen=5)}

def init_h2h_state():
    return {'matches_played':0, 'points_a_last3':deque(maxlen=3), 'gd_a_last3':deque(maxlen=3), 'total_goals_last3':deque(maxlen=3)}

def avg_deque(dq, default=np.nan):
    return float(np.mean(list(dq))) if len(dq) else default

def expected_elo_result(rating_a, rating_b, home_bonus=0.0):
    return 1 / (1 + 10 ** (-((rating_a + home_bonus - rating_b) / 400)))

def snapshot_team_features_full(st, match_date, side):
    days = np.nan if pd.isna(st['last_match_date']) or pd.isna(match_date) else (pd.Timestamp(match_date) - pd.Timestamp(st['last_match_date'])).days
    res = list(st['results_last10'])
    return {f'hist_matches_played_{side}':st['matches_played'], f'hist_elo_overall_{side}':st['elo_overall'], f'hist_elo_gd_{side}':st['elo_goal_diff'],
            f'hist_ewm_points_{side}':st['ewm_points'], f'hist_ewm_gf_{side}':st['ewm_gf'], f'hist_ewm_ga_{side}':st['ewm_ga'], f'hist_ewm_gd_{side}':st['ewm_gd'],
            f'hist_points_avg_last5_{side}':avg_deque(st['points_last5']), f'hist_points_avg_last10_{side}':avg_deque(st['points_last10']),
            f'hist_gf_avg_last5_{side}':avg_deque(st['gf_last5']), f'hist_ga_avg_last5_{side}':avg_deque(st['ga_last5']), f'hist_gd_avg_last5_{side}':avg_deque(st['gd_last5']),
            f'hist_win_rate_last10_{side}':float(np.mean([r==1 for r in res])) if res else np.nan,
            f'hist_draw_rate_last10_{side}':float(np.mean([r==0 for r in res])) if res else np.nan,
            f'hist_loss_rate_last10_{side}':float(np.mean([r==-1 for r in res])) if res else np.nan,
            f'hist_clean_sheet_rate_last5_{side}':avg_deque(st['clean_sheet_last5']), f'hist_failed_to_score_rate_last5_{side}':avg_deque(st['failed_to_score_last5']),
            f'hist_days_since_last_match_{side}':days, f'hist_has_history_{side}':int(st['matches_played']>0)}

def snapshot_h2h_features(st):
    return {'h2h_matches_played_pre':st['matches_played'], 'h2h_points_a_avg_last3':avg_deque(st['points_a_last3']), 'h2h_gd_a_avg_last3':avg_deque(st['gd_a_last3']), 'h2h_total_goals_avg_last3':avg_deque(st['total_goals_last3']), 'h2h_has_history':int(st['matches_played']>0)}

def build_history_feature_row(row, team_states, h2h_states):
    gender, ta, tb, date = str(row['gender']), str(row['team_a']), str(row['team_b']), pd.Timestamp(row['date'])
    sa, sb = team_states[(gender, ta)], team_states[(gender, tb)]
    h2h = h2h_states[(gender, ta, tb)]
    out = {'match_id': row['match_id']}
    out.update(snapshot_team_features_full(sa, date, 'a'))
    out.update(snapshot_team_features_full(sb, date, 'b'))
    out.update(snapshot_h2h_features(h2h))
    for base in ['hist_matches_played','hist_elo_overall','hist_elo_gd','hist_ewm_points','hist_ewm_gf','hist_ewm_ga','hist_ewm_gd','hist_points_avg_last5','hist_points_avg_last10','hist_gf_avg_last5','hist_ga_avg_last5','hist_gd_avg_last5','hist_win_rate_last10','hist_draw_rate_last10','hist_loss_rate_last10','hist_clean_sheet_rate_last5','hist_failed_to_score_rate_last5']:
        a, b = out.get(f'{base}_a'), out.get(f'{base}_b')
        out[f'{base}_diff'] = a - b if pd.notna(a) and pd.notna(b) else np.nan
        out[f'{base}_abs_diff'] = abs(a - b) if pd.notna(a) and pd.notna(b) else np.nan
    a, b = out.get('hist_days_since_last_match_a'), out.get('hist_days_since_last_match_b')
    out['hist_rest_days_diff'] = a - b if pd.notna(a) and pd.notna(b) else np.nan
    return out

def update_states_from_score(sa, sb, h2h, ctx, ga, gb):
    date = pd.Timestamp(ctx['date']); weight = get_tournament_weight(ctx.get('tournament','')); home_side = int(ctx.get('home_side',0) or 0)
    exp_a = expected_elo_result(sa['elo_overall'], sb['elo_overall'], 60*home_side)
    act_a = 1 if ga > gb else (0.5 if ga == gb else 0)
    delta = 24 * weight * (act_a - exp_a)
    sa['elo_overall'] += delta; sb['elo_overall'] -= delta
    gd = ga - gb
    resid = gd - ((sa['elo_goal_diff'] - sb['elo_goal_diff'] + 10*home_side)/100)
    sa['elo_goal_diff'] += 6*resid; sb['elo_goal_diff'] -= 6*resid
    pa = 3 if ga > gb else (1 if ga == gb else 0); pb = 3 if gb > ga else (1 if ga == gb else 0)
    ra = 1 if ga > gb else (0 if ga == gb else -1); rb = 1 if gb > ga else (0 if ga == gb else -1)
    for st, pts, gf, gc, res in [(sa,pa,ga,gb,ra),(sb,pb,gb,ga,rb)]:
        st['matches_played'] += 1; st['last_match_date'] = date
        alpha = 0.35
        st['ewm_points'] = alpha*pts + (1-alpha)*st['ewm_points']; st['ewm_gf'] = alpha*gf + (1-alpha)*st['ewm_gf']; st['ewm_ga'] = alpha*gc + (1-alpha)*st['ewm_ga']; st['ewm_gd'] = alpha*(gf-gc) + (1-alpha)*st['ewm_gd']
        st['points_last5'].append(pts); st['points_last10'].append(pts); st['gf_last5'].append(gf); st['ga_last5'].append(gc); st['gd_last5'].append(gf-gc); st['results_last10'].append(res); st['clean_sheet_last5'].append(int(gc==0)); st['failed_to_score_last5'].append(int(gf==0))
    h2h['matches_played'] += 1; h2h['points_a_last3'].append(pa); h2h['gd_a_last3'].append(gd); h2h['total_goals_last3'].append(ga+gb)

def build_train_history_features_full(match_df):
    df = match_df.sort_values(['date','match_id']).reset_index(drop=True)
    team_states, h2h_states, rows = defaultdict(init_team_state), defaultdict(init_h2h_state), []
    for _, row in iter_progress(df.iterrows(), total=len(df), desc='build_train_history_full'):
        rows.append(build_history_feature_row(row, team_states, h2h_states))
        ka, kb = (str(row['gender']),str(row['team_a'])), (str(row['gender']),str(row['team_b']))
        kh = (str(row['gender']),str(row['team_a']),str(row['team_b']))
        update_states_from_score(team_states[ka], team_states[kb], h2h_states[kh], {'date':row['date'], 'tournament':row.get('tournament',''), 'home_side':row.get('home_side',0)}, int(row['team_a_goals']), int(row['team_b_goals']))
    return pd.DataFrame(rows), copy.deepcopy(dict(team_states)), copy.deepcopy(dict(h2h_states))

def build_future_history_features_freeze(future_df, team_states_cutoff, h2h_states_cutoff):
    df = future_df.sort_values(['date','match_id']).reset_index(drop=True)
    team_states, h2h_states = defaultdict(init_team_state), defaultdict(init_h2h_state)
    team_states.update(copy.deepcopy(team_states_cutoff)); h2h_states.update(copy.deepcopy(h2h_states_cutoff))
    rows = []
    for _, row in iter_progress(df.iterrows(), total=len(df), desc='build_future_history_freeze'):
        rows.append(build_history_feature_row(row, team_states, h2h_states))
        for team in [str(row['team_a']), str(row['team_b'])]:
            team_states[(str(row['gender']), team)]['last_match_date'] = pd.Timestamp(row['date'])
    return pd.DataFrame(rows)

train_static_all = engineer_static_match_features(train_match_base)
test_static_features = engineer_static_match_features(test_match_base)
full_train_history_all, cutoff_team_states_full, cutoff_h2h_states_full = build_train_history_features_full(train_static_all)
train_feature_all = train_static_all.merge(full_train_history_all, on='match_id', how='left', validate='one_to_one')
full_test_history = build_future_history_features_freeze(test_static_features, cutoff_team_states_full, cutoff_h2h_states_full)
test_feature_full = test_static_features.merge(full_test_history, on='match_id', how='left', validate='one_to_one')

train_fold_base, valid_fold_base = make_time_based_holdout(train_match_base, valid_fraction=0.2)
train_fold_static = train_static_all[train_static_all['match_id'].isin(train_fold_base['match_id'])].copy()
valid_fold_static = train_static_all[train_static_all['match_id'].isin(valid_fold_base['match_id'])].copy()
train_fold_hist, cutoff_team_states_valid, cutoff_h2h_states_valid = build_train_history_features_full(train_fold_static)
valid_hist = build_future_history_features_freeze(valid_fold_static, cutoff_team_states_valid, cutoff_h2h_states_valid)
train_feature_full = train_fold_static.merge(train_fold_hist, on='match_id', how='left', validate='one_to_one')
valid_feature_full = valid_fold_static.merge(valid_hist, on='match_id', how='left', validate='one_to_one')

assert_no_duplicate_columns(train_feature_full, 'train_feature_full')
assert_no_duplicate_columns(valid_feature_full, 'valid_feature_full')
print('[INFO] train_feature_full:', train_feature_full.shape)
print('[INFO] valid_feature_full:', valid_feature_full.shape)


# %% [markdown]
# # 06. EXP09A Feature Repair Functions
#
# Bagian ini menambahkan fungsi repair feature yang idenya berasal dari EXP09A. Fitur repair ini menangani sinyal domain shift seperti GDP missing, struktur turnamen baru, cold-start team, women temporal trend, dan interaksi confederation.
#
# Penting: fungsi di bagian ini hanya menyiapkan fitur tambahan. Pemakaiannya per head baru diatur pada section feature set split, sehingga goal head tetap bisa memakai fitur EXP05A asli.
# %%
student_categorical_features = [
    'team_a', 'team_b', 'gender', 'tournament', 'venue_country',
    'team_a_confederation', 'team_b_confederation', 'pair_key', 'confed_pair_key'
]
student_numeric_features_static = [
    'neutral', 'altitude_venue', 'temperature_venue',
    'team_a_is_home', 'team_b_is_home',
    'team_a_population', 'team_b_population',
    'team_a_gdp_per_capita', 'team_b_gdp_per_capita',
    'team_a_distance_travel', 'team_b_distance_travel',
    'match_year', 'match_month', 'match_quarter', 'match_dayofweek',
    'match_dayofyear', 'match_is_weekend', 'match_decade',
    'home_side', 'same_confederation',
    'is_friendly', 'is_world_cup', 'is_qualification',
    'is_nations_league', 'tournament_weight_proxy',
    'population_diff', 'population_abs_diff',
    'log_population_a', 'log_population_b', 'log_population_diff',
    'population_ratio_ab',
    'gdp_per_capita_diff', 'gdp_per_capita_abs_diff',
    'log_gdp_per_capita_a', 'log_gdp_per_capita_b',
    'log_gdp_per_capita_diff', 'gdp_per_capita_ratio_ab',
    'distance_travel_diff', 'distance_travel_abs_diff',
    'log_distance_travel_a', 'log_distance_travel_b',
    'log_distance_travel_diff', 'distance_travel_ratio_ab',
]

def derive_exp05a_feature_columns(df):
    history_numeric_features = [c for c in df.columns if c.startswith('hist_') or c.startswith('h2h_')]
    feature_cols = [
        c for c in (student_categorical_features + student_numeric_features_static + history_numeric_features)
        if c in df.columns
    ]
    cat_features = [c for c in feature_cols if c in student_categorical_features]
    return feature_cols, cat_features

# EXP05A feature set is captured BEFORE the EXP09A repair patch.
exp05a_feature_cols, exp05a_cat_features = derive_exp05a_feature_columns(train_feature_full)
log_result(f'EXP05A original feature count: {len(exp05a_feature_cols)}')
log_result(f'EXP05A original categorical count: {len(exp05a_cat_features)}')



with timed_section('exp09c_feature_repair_patch'):
    EXP09A_ADDED_FEATURE_COLS = []
    EXP09A_ADDED_CAT_COLS = []
    EXP09A_PATCH_GROUPS = {}

    def _append_unique(base, items):
        out = list(base)
        for x in items:
            if x not in out:
                out.append(x)
        return out

    def _safe_num_col(df, col):
        if col in df.columns:
            return pd.to_numeric(df[col], errors='coerce')
        return pd.Series(np.nan, index=df.index)

    def _latest_team_gdp_lookup(fit_df):
        rows = []
        for side in ['a','b']:
            team_col = f'team_{side}'
            gdp_col = f'team_{side}_gdp_per_capita'
            conf_col = f'team_{side}_confederation'
            if team_col in fit_df.columns and gdp_col in fit_df.columns:
                tmp = fit_df[[team_col, 'gender', 'date', gdp_col] + ([conf_col] if conf_col in fit_df.columns else [])].copy()
                tmp = tmp.rename(columns={team_col:'team', gdp_col:'gdp', conf_col:'confederation'})
                rows.append(tmp)
        if not rows:
            return {}, pd.DataFrame()
        long = pd.concat(rows, ignore_index=True)
        long['date'] = pd.to_datetime(long['date'], errors='coerce')
        long['gdp'] = pd.to_numeric(long['gdp'], errors='coerce')
        long = long.dropna(subset=['gdp']).sort_values('date')
        latest = long.groupby(['gender','team'])['gdp'].last().to_dict()
        return latest, long

    def add_exp09a_gdp_imputation_features(train_df, test_df, feature_cols=None, cat_cols=None):
        fit_df = train_df.copy()
        latest_lookup, long_gdp = _latest_team_gdp_lookup(fit_df)
        global_median = float(long_gdp['gdp'].median()) if len(long_gdp) else 0.0
        if not np.isfinite(global_median):
            global_median = 0.0
        by_gender_conf = {}
        if len(long_gdp) and 'confederation' in long_gdp.columns:
            by_gender_conf = long_gdp.groupby(['gender','confederation'])['gdp'].median().to_dict()
        by_gender = long_gdp.groupby('gender')['gdp'].median().to_dict() if len(long_gdp) else {}

        def patch_one(df):
            out = df.copy()
            new_cols = []
            for side in ['a','b']:
                team_col = f'team_{side}'
                raw_col = f'team_{side}_gdp_per_capita'
                conf_col = f'team_{side}_confederation'
                raw = _safe_num_col(out, raw_col)
                out[f'gdp_team_{side}_raw'] = raw
                out[f'gdp_team_{side}_was_missing'] = raw.isna().astype(int)

                fill_vals = []
                for _, row in out.iterrows():
                    raw_val = row.get(raw_col, np.nan)
                    raw_val = pd.to_numeric(pd.Series([raw_val]), errors='coerce').iloc[0]
                    if pd.notna(raw_val):
                        fill_vals.append(float(raw_val)); continue
                    key = (str(row.get('gender','__MISSING__')), str(row.get(team_col,'__MISSING__')))
                    if key in latest_lookup:
                        fill_vals.append(float(latest_lookup[key])); continue
                    gc_key = (str(row.get('gender','__MISSING__')), str(row.get(conf_col,'__MISSING__')))
                    if gc_key in by_gender_conf and pd.notna(by_gender_conf[gc_key]):
                        fill_vals.append(float(by_gender_conf[gc_key])); continue
                    g_key = str(row.get('gender','__MISSING__'))
                    if g_key in by_gender and pd.notna(by_gender[g_key]):
                        fill_vals.append(float(by_gender[g_key])); continue
                    fill_vals.append(global_median)
                out[f'gdp_team_{side}_imputed'] = pd.Series(fill_vals, index=out.index).astype(float)
                out[f'gdp_team_{side}_was_imputed'] = out[f'gdp_team_{side}_was_missing'].astype(int)
                out[f'log_gdp_team_{side}_imputed'] = np.log1p(out[f'gdp_team_{side}_imputed'].clip(lower=0))
                new_cols += [f'gdp_team_{side}_raw', f'gdp_team_{side}_was_missing', f'gdp_team_{side}_imputed', f'gdp_team_{side}_was_imputed', f'log_gdp_team_{side}_imputed']
            out['both_gdp_missing'] = ((out['gdp_team_a_was_missing'] == 1) & (out['gdp_team_b_was_missing'] == 1)).astype(int)
            out['gdp_diff_imputed'] = out['gdp_team_a_imputed'] - out['gdp_team_b_imputed']
            out['gdp_abs_diff_imputed'] = out['gdp_diff_imputed'].abs()
            out['gdp_ratio_imputed'] = safe_divide(out['gdp_team_a_imputed'], out['gdp_team_b_imputed'])
            out['log_gdp_diff_imputed'] = out['log_gdp_team_a_imputed'] - out['log_gdp_team_b_imputed']
            new_cols += ['both_gdp_missing','gdp_diff_imputed','gdp_abs_diff_imputed','gdp_ratio_imputed','log_gdp_diff_imputed']
            return out, new_cols

        tr, cols = patch_one(train_df)
        te, _ = patch_one(test_df)
        return tr, te, _append_unique(feature_cols or [], cols), list(cat_cols or []), cols, []

    def infer_tournament_structure(tournament: str, gender: str = None) -> str:
        t = str(tournament).lower().strip()
        if t in ['', 'nan', 'none', '__missing__']:
            return 'unknown'
        knockout_words = ['knockout', 'final', 'semi', 'quarter', 'play-off', 'playoff', 'third place']
        group_words = ['group', 'league', 'round robin']
        is_knock = any(w in t for w in knockout_words)
        is_group = any(w in t for w in group_words)
        if 'friendly' in t:
            return 'friendly'
        if 'qualif' in t or 'qualification' in t or 'qualifier' in t:
            return 'qualifier'
        if 'nations league' in t:
            return 'nations_league_knockout' if is_knock else 'nations_league_group'
        if 'world cup' in t or 'fifa world cup' in t:
            return 'world_cup_knockout' if is_knock else ('world_cup_group' if is_group else 'world_cup_group')
        continental_tokens = ['asian cup','african cup','euro','european championship','gold cup','copa america','afc championship','concacaf','caf','uefa','ofc','saff','aff']
        if any(tok in t for tok in continental_tokens):
            return 'continental_knockout' if is_knock else 'continental_group'
        if 'games' in t or 'olympic' in t or 'pan american' in t:
            return 'regional_games'
        if 'cup' in t or 'championship' in t or 'tournament' in t:
            return 'other_competitive'
        return 'other_competitive'

    def _competition_family(struct):
        s = str(struct)
        if s == 'friendly': return 'friendly'
        if 'qualifier' in s: return 'qualifier'
        if 'nations_league' in s: return 'nations_league'
        if 'world_cup' in s: return 'world_cup'
        if 'continental' in s: return 'continental'
        if 'games' in s: return 'regional_games'
        if s == 'unknown': return 'unknown'
        return 'other_competitive'

    def _stage_type(struct):
        s = str(struct)
        if 'knockout' in s: return 'knockout'
        if 'group' in s or 'league' in s: return 'group_like'
        if 'qualifier' in s: return 'qualifier'
        if s == 'friendly': return 'friendly'
        return 'other'

    def add_exp09a_tournament_structure_features(train_df, test_df, feature_cols=None, cat_cols=None):
        def patch_one(df):
            out = df.copy()
            out['tournament_structure'] = [infer_tournament_structure(t,g) for t,g in zip(out['tournament'], out['gender'])]
            out['competition_family'] = out['tournament_structure'].map(_competition_family).astype(str)
            out['stage_type'] = out['tournament_structure'].map(_stage_type).astype(str)
            out['is_friendly_structure'] = (out['competition_family'] == 'friendly').astype(int)
            out['is_qualifier_structure'] = (out['competition_family'] == 'qualifier').astype(int)
            out['is_competitive_structure'] = (out['competition_family'] != 'friendly').astype(int)
            out['is_nations_league_structure'] = (out['competition_family'] == 'nations_league').astype(int)
            out['is_world_cup_structure'] = (out['competition_family'] == 'world_cup').astype(int)
            out['is_continental_structure'] = (out['competition_family'] == 'continental').astype(int)
            out['is_knockout_like'] = (out['stage_type'] == 'knockout').astype(int)
            out['is_group_like'] = (out['stage_type'] == 'group_like').astype(int)
            for c in ['tournament_structure','competition_family','stage_type']:
                out[c] = out[c].astype('string').fillna('__MISSING__')
            return out
        cat_new = ['tournament_structure','competition_family','stage_type']
        num_new = ['is_friendly_structure','is_qualifier_structure','is_competitive_structure','is_nations_league_structure','is_world_cup_structure','is_continental_structure','is_knockout_like','is_group_like']
        return patch_one(train_df), patch_one(test_df), _append_unique(feature_cols or [], cat_new + num_new), _append_unique(cat_cols or [], cat_new), cat_new + num_new, cat_new

    def add_exp09a_confederation_interaction_features(train_df, test_df, feature_cols=None, cat_cols=None):
        req = ['team_a_confederation','team_b_confederation']
        if not all(c in train_df.columns for c in req):
            log_warn('Confederation columns not found. Skip confederation interaction features.')
            return train_df, test_df, list(feature_cols or []), list(cat_cols or []), [], []
        def patch_one(df):
            out = df.copy()
            a = out['team_a_confederation'].astype('string').fillna('__MISSING__')
            b = out['team_b_confederation'].astype('string').fillna('__MISSING__')
            out['team_a_confederation_exp09a'] = a
            out['team_b_confederation_exp09a'] = b
            out['confed_pair_exp09a'] = a + '__VS__' + b
            out['same_confederation_exp09a'] = (a == b).astype(int)
            if 'tournament_structure' in out.columns:
                out['tournament_structure_x_confed_pair'] = out['tournament_structure'].astype(str) + '__' + out['confed_pair_exp09a'].astype(str)
            else:
                out['tournament_structure_x_confed_pair'] = '__MISSING__'
            if 'is_competitive_structure' in out.columns:
                out['is_competitive_x_same_confed'] = out['is_competitive_structure'].astype(int) * out['same_confederation_exp09a'].astype(int)
            else:
                out['is_competitive_x_same_confed'] = 0
            for c in ['team_a_confederation_exp09a','team_b_confederation_exp09a','confed_pair_exp09a','tournament_structure_x_confed_pair']:
                out[c] = out[c].astype('string').fillna('__MISSING__')
            return out
        cat_new = ['team_a_confederation_exp09a','team_b_confederation_exp09a','confed_pair_exp09a','tournament_structure_x_confed_pair']
        num_new = ['same_confederation_exp09a','is_competitive_x_same_confed']
        return patch_one(train_df), patch_one(test_df), _append_unique(feature_cols or [], cat_new + num_new), _append_unique(cat_cols or [], cat_new), cat_new + num_new, cat_new

    def add_exp09a_cold_start_features(train_df, test_df, feature_cols=None, cat_cols=None, low_threshold=10):
        def patch_one(df):
            out = df.copy()
            a_cnt = _safe_num_col(out, 'hist_matches_played_a').fillna(0)
            b_cnt = _safe_num_col(out, 'hist_matches_played_b').fillna(0)
            out['team_a_history_count_exp09a'] = a_cnt
            out['team_b_history_count_exp09a'] = b_cnt
            out['team_a_is_new_exp09a'] = (a_cnt <= 0).astype(int)
            out['team_b_is_new_exp09a'] = (b_cnt <= 0).astype(int)
            out['team_a_low_history_exp09a'] = (a_cnt < low_threshold).astype(int)
            out['team_b_low_history_exp09a'] = (b_cnt < low_threshold).astype(int)
            out['any_team_new_exp09a'] = ((out['team_a_is_new_exp09a'] == 1) | (out['team_b_is_new_exp09a'] == 1)).astype(int)
            out['both_team_new_exp09a'] = ((out['team_a_is_new_exp09a'] == 1) & (out['team_b_is_new_exp09a'] == 1)).astype(int)
            out['any_team_low_history_exp09a'] = ((out['team_a_low_history_exp09a'] == 1) | (out['team_b_low_history_exp09a'] == 1)).astype(int)
            out['history_count_min_exp09a'] = np.minimum(a_cnt, b_cnt)
            out['history_count_diff_exp09a'] = a_cnt - b_cnt
            is_w = out['gender'].astype(str).eq('W').astype(int)
            out['team_a_w_history_count_exp09a'] = a_cnt * is_w
            out['team_b_w_history_count_exp09a'] = b_cnt * is_w
            out['team_a_w_low_history_exp09a'] = out['team_a_low_history_exp09a'] * is_w
            out['team_b_w_low_history_exp09a'] = out['team_b_low_history_exp09a'] * is_w
            return out
        cols = ['team_a_history_count_exp09a','team_b_history_count_exp09a','team_a_is_new_exp09a','team_b_is_new_exp09a','team_a_low_history_exp09a','team_b_low_history_exp09a','any_team_new_exp09a','both_team_new_exp09a','any_team_low_history_exp09a','history_count_min_exp09a','history_count_diff_exp09a','team_a_w_history_count_exp09a','team_b_w_history_count_exp09a','team_a_w_low_history_exp09a','team_b_w_low_history_exp09a']
        return patch_one(train_df), patch_one(test_df), _append_unique(feature_cols or [], cols), list(cat_cols or []), cols, []

    def add_exp09a_women_temporal_features(train_df, test_df, feature_cols=None, cat_cols=None):
        def patch_one(df):
            out = df.copy()
            year = pd.to_datetime(out['date'], errors='coerce').dt.year.fillna(out.get('match_year', 0)).astype(float)
            out['year_exp09a'] = year
            out['is_women_exp09a'] = out['gender'].astype(str).eq('W').astype(int)
            out['year_centered_2018_exp09a'] = year - 2018
            out['w_year_centered_exp09a'] = out['is_women_exp09a'] * (year - 2018)
            out['is_post2018_exp09a'] = (year >= 2018).astype(int)
            out['is_post2018_women_exp09a'] = out['is_post2018_exp09a'] * out['is_women_exp09a']
            out['is_post2020_women_exp09a'] = (year >= 2020).astype(int) * out['is_women_exp09a']
            out['era_pre_1990_exp09a'] = (year < 1990).astype(int)
            out['era_1990_2000_exp09a'] = ((year >= 1990) & (year < 2000)).astype(int)
            out['era_2000_2010_exp09a'] = ((year >= 2000) & (year < 2010)).astype(int)
            out['era_2010_2018_exp09a'] = ((year >= 2010) & (year < 2018)).astype(int)
            out['era_post2018_exp09a'] = (year >= 2018).astype(int)
            return out
        cols = ['year_exp09a','is_women_exp09a','year_centered_2018_exp09a','w_year_centered_exp09a','is_post2018_exp09a','is_post2018_women_exp09a','is_post2020_women_exp09a','era_pre_1990_exp09a','era_1990_2000_exp09a','era_2000_2010_exp09a','era_2010_2018_exp09a','era_post2018_exp09a']
        return patch_one(train_df), patch_one(test_df), _append_unique(feature_cols or [], cols), list(cat_cols or []), cols, []

    def run_exp09a_feature_patch(train_df, test_df, feature_cols=None, cat_cols=None):
        feature_cols = list(feature_cols or [])
        cat_cols = list(cat_cols or [])
        added_cols_all, added_cat_all = [], []
        patch_steps = [
            ('gdp_imputation', add_exp09a_gdp_imputation_features),
            ('tournament_structure', add_exp09a_tournament_structure_features),
            ('confederation_interaction', add_exp09a_confederation_interaction_features),
            ('cold_start', add_exp09a_cold_start_features),
            ('women_temporal_trend', add_exp09a_women_temporal_features),
        ]
        tr, te = train_df.copy(), test_df.copy()
        patch_list = []
        for group_name, fn in patch_steps:
            tr, te, feature_cols, cat_cols, added, added_cat = fn(tr, te, feature_cols, cat_cols)
            added_cols_all = _append_unique(added_cols_all, added)
            added_cat_all = _append_unique(added_cat_all, added_cat)
            patch_list.append({'group': group_name, 'n_added_features': len(added), 'added_features': added, 'added_categorical_features': added_cat})
            log_info(f'Added {group_name} features: {len(added)} columns')
        return tr, te, feature_cols, cat_cols, added_cols_all, added_cat_all, patch_list

    # Patch fold train/valid using fold train as fitting source.
    train_feature_full, valid_feature_full, _, _, added_fold, added_cat_fold, patch_list_fold = run_exp09a_feature_patch(train_feature_full, valid_feature_full)

    # Patch full train/test using full train as fitting source.
    train_feature_all, test_feature_full, _, _, added_full, added_cat_full, patch_list_full = run_exp09a_feature_patch(train_feature_all, test_feature_full)

    EXP09A_ADDED_FEATURE_COLS = _append_unique(added_fold, added_full)
    EXP09A_ADDED_CAT_COLS = _append_unique(added_cat_fold, added_cat_full)
    EXP09A_ADDED_NUMERIC_COLS = [c for c in EXP09A_ADDED_FEATURE_COLS if c not in EXP09A_ADDED_CAT_COLS]
    EXP09A_PATCH_GROUPS = patch_list_full

    # Fill categorical missing safely.
    for df_name in ['train_feature_full','valid_feature_full','train_feature_all','test_feature_full']:
        df = globals()[df_name]
        for c in EXP09A_ADDED_CAT_COLS:
            if c in df.columns:
                df[c] = df[c].astype('string').fillna('__MISSING__')
        globals()[df_name] = df

    # GDP imputation audit after patch.
    gdp_imputation_audit_rows = []
    for dataset_name, df in [('train_fold', train_feature_full), ('valid_fold', valid_feature_full), ('full_train', train_feature_all), ('test', test_feature_full)]:
        row = {'dataset': dataset_name, 'n_rows': len(df)}
        for c in ['gdp_team_a_was_imputed','gdp_team_b_was_imputed','both_gdp_missing']:
            row[f'{c}_rate'] = float(pd.to_numeric(df.get(c, 0), errors='coerce').fillna(0).mean())
        gdp_imputation_audit_rows.append(row)
    gdp_imputation_audit = pd.DataFrame(gdp_imputation_audit_rows)
    gdp_imputation_audit.to_csv(SUM_DIR / 'gdp_imputation_audit.csv', index=False)

    patch_summary = []
    for item in EXP09A_PATCH_GROUPS:
        patch_summary.append({
            'feature_group': item['group'],
            'n_added_features': item['n_added_features'],
            'added_features': ', '.join(item['added_features']),
            'added_categorical_features': ', '.join(item['added_categorical_features']),
        })
    feature_patch_summary = pd.DataFrame(patch_summary)
    feature_patch_summary.to_csv(SUM_DIR / 'feature_patch_summary.csv', index=False)
    with open(SUM_DIR / 'exp09a_feature_patch_list.json', 'w', encoding='utf-8') as f:
        json.dump(EXP09A_PATCH_GROUPS, f, indent=2, ensure_ascii=False)

    log_result(f'EXP09A added feature count: {len(EXP09A_ADDED_FEATURE_COLS)}')
    log_result(f'EXP09A added categorical count: {len(EXP09A_ADDED_CAT_COLS)}')
    log_check('Train/valid patch alignment', set(EXP09A_ADDED_FEATURE_COLS).issubset(train_feature_full.columns) and set(EXP09A_ADDED_FEATURE_COLS).issubset(valid_feature_full.columns))
    log_check('Train/test patch alignment', set(EXP09A_ADDED_FEATURE_COLS).issubset(train_feature_all.columns) and set(EXP09A_ADDED_FEATURE_COLS).issubset(test_feature_full.columns))
    log_saved(SUM_DIR / 'exp09a_feature_patch_list.json')
    log_saved(SUM_DIR / 'feature_patch_summary.csv')
    log_saved(SUM_DIR / 'gdp_imputation_audit.csv')
    display(feature_patch_summary)


# %% [markdown]
# # 07. Feature Set Split: Goal vs Outcome vs Tail
#
# Bagian ini adalah inti EXP09C. Di sini fitur asli EXP05A dan fitur hasil repair EXP09A dipisahkan secara eksplisit.
#
# ## Konfigurasi utama V2
#
# - `goal_feature_cols` = fitur EXP05A asli.
# - `outcome_feature_cols` = fitur EXP05A + fitur repair EXP09A.
# - `tail_feature_cols` = fitur EXP05A asli.
#
# Dengan pemisahan ini, eksperimen bisa menguji apakah fitur domain-shift memang lebih cocok untuk membaca arah pertandingan tanpa merusak prediksi jumlah gol.
# %%
EXP09A_ADDED_FEATURE_COLS = [c for c in EXP09A_ADDED_FEATURE_COLS if c in train_feature_full.columns]
EXP09A_ADDED_CAT_COLS = [c for c in EXP09A_ADDED_CAT_COLS if c in EXP09A_ADDED_FEATURE_COLS]

exp09a_feature_cols = list(exp05a_feature_cols)
for c in EXP09A_ADDED_FEATURE_COLS:
    if c not in exp09a_feature_cols:
        exp09a_feature_cols.append(c)

exp09a_cat_features = list(exp05a_cat_features)
for c in EXP09A_ADDED_CAT_COLS:
    if c not in exp09a_cat_features:
        exp09a_cat_features.append(c)

# Keep the original EXP05A global names for screening/refine source-of-truth behavior.
student_feature_cols = list(exp05a_feature_cols)
student_cat_features = list(exp05a_cat_features)

feature_columns_before_after = {
    'experiment': 'EXP09C',
    'source_of_truth': 'EXP05A-LITE-FIX-V2',
    'old_feature_cols': exp05a_feature_cols,
    'old_cat_cols': exp05a_cat_features,
    'new_feature_cols': exp09a_feature_cols,
    'new_cat_cols': exp09a_cat_features,
    'added_feature_cols': EXP09A_ADDED_FEATURE_COLS,
    'added_cat_cols': EXP09A_ADDED_CAT_COLS,
}
with open(SUM_DIR / 'feature_columns_before_after.json', 'w', encoding='utf-8') as f:
    json.dump(feature_columns_before_after, f, indent=2, ensure_ascii=False)

with open(SUM_DIR / 'feature_columns.json', 'w', encoding='utf-8') as f:
    json.dump({
        'student_feature_cols_exp05a': exp05a_feature_cols,
        'student_cat_features_exp05a': exp05a_cat_features,
        'student_feature_cols_exp09a': exp09a_feature_cols,
        'student_cat_features_exp09a': exp09a_cat_features,
        'xgb_policy': XGB_POLICY,
    }, f, indent=2, ensure_ascii=False)

with open(SUM_DIR / 'objective_configurations.json', 'w', encoding='utf-8') as f:
    json.dump({'goal':'Poisson', 'outcome':'MultiClass', 'tail':'Binary Logloss', 'xgb':'skipped'}, f, indent=2)
with open(SUM_DIR / 'gender_split_config.json', 'w', encoding='utf-8') as f:
    json.dump({'gender_split': True, 'genders': sorted(train_match_base['gender'].astype(str).unique())}, f, indent=2)

feature_config_df = pd.DataFrame([
    {'head': 'goal_a', 'variant': 'V2_outcome_only', 'feature_set': 'EXP05A', 'n_features': len(exp05a_feature_cols), 'n_cat_features': len(exp05a_cat_features)},
    {'head': 'goal_b', 'variant': 'V2_outcome_only', 'feature_set': 'EXP05A', 'n_features': len(exp05a_feature_cols), 'n_cat_features': len(exp05a_cat_features)},
    {'head': 'outcome', 'variant': 'V2_outcome_only', 'feature_set': 'EXP05A_PLUS_EXP09A', 'n_features': len(exp09a_feature_cols), 'n_cat_features': len(exp09a_cat_features)},
    {'head': 'tail', 'variant': 'V2_outcome_only', 'feature_set': 'EXP05A', 'n_features': len(exp05a_feature_cols), 'n_cat_features': len(exp05a_cat_features)},
])
feature_config_df.to_csv(SUM_DIR / 'head_feature_config.csv', index=False)

exp09c_feature_sets = {
    'V0_baseline': {'goal': 'EXP05A', 'outcome': 'EXP05A', 'tail': 'EXP05A'},
    'V1_allhead': {'goal': 'EXP05A_PLUS_EXP09A', 'outcome': 'EXP05A_PLUS_EXP09A', 'tail': 'EXP05A_PLUS_EXP09A'},
    'V2_outcome_only': {'goal': 'EXP05A', 'outcome': 'EXP05A_PLUS_EXP09A', 'tail': 'EXP05A'},
    'V3_outcome_tail_optional': {'goal': 'EXP05A', 'outcome': 'EXP05A_PLUS_EXP09A', 'tail': 'EXP05A_PLUS_EXP09A'},
}
with open(SUM_DIR / 'exp09c_feature_sets.json', 'w', encoding='utf-8') as f:
    json.dump(exp09c_feature_sets, f, indent=2, ensure_ascii=False)
# Compatibility copy of patch list under EXP09C name.
if 'EXP09A_PATCH_GROUPS' in globals():
    with open(SUM_DIR / 'exp09c_feature_patch_list.json', 'w', encoding='utf-8') as f:
        json.dump(EXP09A_PATCH_GROUPS, f, indent=2, ensure_ascii=False)

log_result(f'EXP09A repaired feature count: {len(exp09a_feature_cols)}')
log_result(f'Added repair feature count: {len(EXP09A_ADDED_FEATURE_COLS)}')
log_check('EXP05A feature set preserved for goal heads', set(exp05a_feature_cols).issubset(set(exp09a_feature_cols)))
log_check('Added categorical features registered', set(EXP09A_ADDED_CAT_COLS).issubset(set(exp09a_cat_features)))
log_saved(SUM_DIR / 'feature_columns_before_after.json')
log_saved(SUM_DIR / 'head_feature_config.csv')
log_saved(SUM_DIR / 'exp09c_feature_sets.json')
display(feature_config_df)


# %% [markdown]
# # 08. EDA Tipis dan Feature Patch Summary
#
# Bagian ini membuat audit ringan untuk menjelaskan alasan feature repair. Audit yang dibuat meliputi missing GDP, struktur turnamen, cold-start team, women temporal coverage, dan ringkasan jumlah fitur sebelum/sesudah patch.
#
# Output tahap ini disimpan ke folder `summaries/`, misalnya `feature_columns_before_after.json`, `head_feature_config.csv`, dan beberapa audit CSV lain. Bagian ini tidak memengaruhi training secara langsung, tetapi membantu membaca kenapa fitur repair dicoba.
# %%
with timed_section('exp09c_domain_shift_eda'):
    def _year_col(df):
        return pd.to_datetime(df['date'], errors='coerce').dt.year

    def build_gdp_missing_audit(train_df, test_df):
        rows = []
        for name, df in [('train', train_df), ('test', test_df)]:
            tmp = df.copy()
            tmp['year'] = _year_col(tmp)
            g_team = 'gdp_per_capita_team' if 'gdp_per_capita_team' in tmp.columns else None
            g_opp = 'gdp_per_capita_opp' if 'gdp_per_capita_opp' in tmp.columns else None
            for (year, gender), g in tmp.groupby(['year','gender'], dropna=False):
                if g_team and g_opp:
                    mt = pd.to_numeric(g[g_team], errors='coerce').isna()
                    mo = pd.to_numeric(g[g_opp], errors='coerce').isna()
                    rows.append({
                        'dataset': name,
                        'year': year,
                        'gender': gender,
                        'n_rows': len(g),
                        'gdp_team_missing_rate': float(mt.mean()),
                        'gdp_opp_missing_rate': float(mo.mean()),
                        'both_gdp_missing_rate': float((mt & mo).mean()),
                    })
        return pd.DataFrame(rows).sort_values(['dataset','year','gender'])

    def infer_tournament_structure(tournament: str, gender: str = None) -> str:
        t = str(tournament).lower().strip()
        if t in ['', 'nan', 'none', '__missing__']:
            return 'unknown'
        knockout_words = ['knockout', 'final', 'semi', 'quarter', 'play-off', 'playoff', 'third place']
        group_words = ['group', 'league', 'round robin']
        is_knock = any(w in t for w in knockout_words)
        is_group = any(w in t for w in group_words)
        if 'friendly' in t:
            return 'friendly'
        if 'qualif' in t or 'qualification' in t or 'qualifier' in t:
            return 'qualifier'
        if 'nations league' in t:
            return 'nations_league_knockout' if is_knock else 'nations_league_group'
        if 'world cup' in t or 'fifa world cup' in t:
            return 'world_cup_knockout' if is_knock else ('world_cup_group' if is_group else 'world_cup_group')
        continental_tokens = ['asian cup','african cup','euro','european championship','gold cup','copa america','afc championship','concacaf','caf','uefa','ofc','saff','aff']
        if any(tok in t for tok in continental_tokens):
            return 'continental_knockout' if is_knock else 'continental_group'
        if 'games' in t or 'olympic' in t or 'pan american' in t:
            return 'regional_games'
        if 'cup' in t or 'championship' in t or 'tournament' in t:
            return 'other_competitive'
        return 'other_competitive'

    def build_tournament_structure_audit(train_df, test_df):
        rows=[]
        for name, df in [('train', train_df), ('test', test_df)]:
            tmp = df.copy()
            tmp['tournament_structure'] = [infer_tournament_structure(t,g) for t,g in zip(tmp['tournament'], tmp['gender'])]
            cnt = tmp.groupby(['gender','tournament_structure'], dropna=False).size().reset_index(name='n_rows')
            cnt['dataset'] = name
            cnt['share'] = cnt.groupby(['dataset','gender'])['n_rows'].transform(lambda s: s/s.sum())
            rows.append(cnt[['dataset','gender','tournament_structure','n_rows','share']])
        return pd.concat(rows, ignore_index=True)

    def build_cold_start_audit(train_df, test_df, low_threshold=10):
        rows=[]
        for gender in sorted(pd.concat([train_df['gender'], test_df['gender']]).astype(str).dropna().unique()):
            tr_g = train_df[train_df['gender'].astype(str)==gender]
            te_g = test_df[test_df['gender'].astype(str)==gender]
            train_teams = set(tr_g['team'].astype(str))
            test_teams = set(te_g['team'].astype(str))
            hist_count = tr_g.groupby('team').size().to_dict()
            is_new = ~te_g['team'].astype(str).isin(train_teams)
            low_hist = te_g['team'].astype(str).map(hist_count).fillna(0) < low_threshold
            rows.append({
                'gender': gender,
                'n_train_unique_teams': len(train_teams),
                'n_test_unique_teams': len(test_teams),
                'n_test_new_teams': len(test_teams - train_teams),
                'share_test_rows_with_new_team': float(is_new.mean()) if len(te_g) else np.nan,
                'share_test_rows_with_low_history_team': float(low_hist.mean()) if len(te_g) else np.nan,
            })
        return pd.DataFrame(rows)

    def build_women_shift_audit(train_df, test_df):
        rows=[]
        for name, df in [('train', train_df), ('test', test_df)]:
            tmp = df.copy()
            tmp['year'] = _year_col(tmp)
            audit = tmp.groupby(['year','gender'], dropna=False).agg(
                n_rows=('Id','size') if 'Id' in tmp.columns else ('team','size'),
                n_unique_teams=('team','nunique')
            ).reset_index()
            audit['dataset'] = name
            rows.append(audit[['dataset','year','gender','n_rows','n_unique_teams']])
        return pd.concat(rows, ignore_index=True).sort_values(['dataset','year','gender'])

    gdp_missing_audit = build_gdp_missing_audit(train_clean, test_clean)
    tournament_structure_audit = build_tournament_structure_audit(train_clean, test_clean)
    cold_start_audit = build_cold_start_audit(train_clean, test_clean)
    women_shift_audit = build_women_shift_audit(train_clean, test_clean)

    gdp_missing_audit.to_csv(SUM_DIR / 'gdp_missing_audit.csv', index=False)
    tournament_structure_audit.to_csv(SUM_DIR / 'tournament_structure_audit.csv', index=False)
    cold_start_audit.to_csv(SUM_DIR / 'cold_start_audit.csv', index=False)
    women_shift_audit.to_csv(SUM_DIR / 'women_shift_audit.csv', index=False)

    # Lightweight figures.
    try:
        if len(gdp_missing_audit):
            fig, ax = plt.subplots(figsize=(10, 4))
            plot_df = gdp_missing_audit[gdp_missing_audit['dataset'].eq('test')].copy()
            sns.lineplot(data=plot_df, x='year', y='both_gdp_missing_rate', hue='gender', marker='o', ax=ax)
            ax.set_title('Test GDP missing rate by year and gender')
            ax.set_ylabel('Both GDP missing rate')
            fig.tight_layout(); fig.savefig(FIG_DIR / 'gdp_missing_by_year_gender.png', dpi=140); plt.close(fig)
        if len(tournament_structure_audit):
            fig, ax = plt.subplots(figsize=(10, 5))
            plot_df = tournament_structure_audit.groupby(['dataset','tournament_structure'])['n_rows'].sum().reset_index()
            sns.barplot(data=plot_df, x='tournament_structure', y='n_rows', hue='dataset', ax=ax)
            ax.tick_params(axis='x', rotation=45)
            ax.set_title('Tournament structure distribution')
            fig.tight_layout(); fig.savefig(FIG_DIR / 'tournament_structure_distribution.png', dpi=140); plt.close(fig)
        if len(cold_start_audit):
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.barplot(data=cold_start_audit, x='gender', y='share_test_rows_with_new_team', ax=ax)
            ax.set_title('Share of test rows with new team by gender')
            fig.tight_layout(); fig.savefig(FIG_DIR / 'cold_start_team_count_by_gender.png', dpi=140); plt.close(fig)
        if len(women_shift_audit):
            fig, ax = plt.subplots(figsize=(10, 4))
            plot_df = women_shift_audit[women_shift_audit['gender'].astype(str).eq('W')]
            sns.lineplot(data=plot_df, x='year', y='n_rows', hue='dataset', marker='o', ax=ax)
            ax.set_title('Women match row count by year')
            fig.tight_layout(); fig.savefig(FIG_DIR / 'women_match_count_by_year.png', dpi=140); plt.close(fig)
    except Exception as e:
        log_warn(f'EDA plot creation skipped due to: {repr(e)}')

    log_saved(SUM_DIR / 'gdp_missing_audit.csv')
    log_saved(SUM_DIR / 'tournament_structure_audit.csv')
    log_saved(SUM_DIR / 'cold_start_audit.csv')
    log_saved(SUM_DIR / 'women_shift_audit.csv')
    display(gdp_missing_audit.tail(10))
    display(tournament_structure_audit.head(10))
    display(cold_start_audit)


# %% [markdown]
# # 09. Target Builder dan Validation Fold
#
# Bagian ini membangun target supervised untuk match-level training: `goal_a`, `goal_b`, `outcome_class`, dan label tail jika dipakai. Selain itu, dibuat validation fold temporal seperti gaya EXP05A agar evaluasi tetap leakage-safe.
#
# Target hanya dibuat dari train match-level. Test tidak pernah diberi target dan tidak dipakai untuk tuning berbasis ground truth.
# %%
def build_supervised_target_frame(match_df):
    out = pd.DataFrame({
        'match_id': match_df['match_id'].values,
        'y_goal_a': pd.to_numeric(match_df['team_a_goals'], errors='coerce').astype(int).values,
        'y_goal_b': pd.to_numeric(match_df['team_b_goals'], errors='coerce').astype(int).values,
    })
    out['y_outcome'] = [_outcome(a, b) for a, b in zip(out['y_goal_a'], out['y_goal_b'])]
    gd = out['y_goal_a'] - out['y_goal_b']
    total = out['y_goal_a'] + out['y_goal_b']
    out['is_team_a_blowout_5plus'] = (gd >= 5).astype(int)
    out['is_team_b_blowout_5plus'] = (gd <= -5).astype(int)
    out['is_team_a_blowout_7plus'] = (gd >= 7).astype(int)
    out['is_team_b_blowout_7plus'] = (gd <= -7).astype(int)
    out['is_high_total_6plus'] = (total >= 6).astype(int)
    return out

y_train_fold = build_supervised_target_frame(train_fold_base)
y_valid_fold = build_supervised_target_frame(valid_fold_base)
y_full_train = build_supervised_target_frame(train_match_base)

CUTOFF_REGIMES = {'full_history': None, 'modern_cutoff_1990': '1990-01-01'}
with open(SUM_DIR / 'cutoff_regime_config.json', 'w', encoding='utf-8') as f:
    json.dump(CUTOFF_REGIMES, f, indent=2)
with open(SUM_DIR / 'tail_target_definitions.json', 'w', encoding='utf-8') as f:
    json.dump({
        'directional_tail': [
            'is_team_a_blowout_5plus',
            'is_team_b_blowout_5plus',
            'is_team_a_blowout_7plus',
            'is_team_b_blowout_7plus',
            'is_high_total_6plus',
        ]
    }, f, indent=2)

# Keep the original EXP05A global names for screening/refine source-of-truth behavior.
student_feature_cols = list(exp05a_feature_cols)
student_cat_features = list(exp05a_cat_features)

log_info(f'Target frame train fold: {y_train_fold.shape}')
log_info(f'Target frame valid fold: {y_valid_fold.shape}')


# %% [markdown]
# # 10. Model Training Helpers
#
# Bagian ini berisi helper training model per head dan per gender. Perubahan utama dari EXP09C adalah wrapper training bisa menerima feature set berbeda untuk goal, outcome, dan tail.
#
# Goal model tetap bisa memakai fitur EXP05A, outcome model bisa memakai fitur EXP05A + EXP09A, dan tail model tetap bisa dikunci ke fitur EXP05A. Decoder tidak diubah di bagian ini.
# %%
try:
    import lightgbm as lgb
except Exception as e:
    lgb = None; print('[WARN] lightgbm import failed:', repr(e))
try:
    from catboost import CatBoostRegressor, CatBoostClassifier
except Exception as e:
    CatBoostRegressor = CatBoostClassifier = None; print('[WARN] catboost import failed:', repr(e))
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except Exception as e:
    optuna = None; print('[WARN] optuna import failed:', repr(e))

def make_default_params(booster, task, seed=42, n_estimators=700):
    if booster == 'cat':
        if task == 'goal': return dict(loss_function='Poisson', eval_metric='Poisson', iterations=n_estimators, learning_rate=0.03, depth=7, l2_leaf_reg=5.0, random_seed=seed, allow_writing_files=False, od_type='Iter', od_wait=100, verbose=MODEL_VERBOSE)
        if task == 'outcome': return dict(loss_function='MultiClass', eval_metric='MultiClass', iterations=n_estimators, learning_rate=0.03, depth=7, l2_leaf_reg=5.0, random_seed=seed, allow_writing_files=False, od_type='Iter', od_wait=100, verbose=MODEL_VERBOSE)
        if task == 'tail': return dict(loss_function='Logloss', eval_metric='Logloss', iterations=n_estimators, learning_rate=0.03, depth=7, l2_leaf_reg=5.0, random_seed=seed, allow_writing_files=False, od_type='Iter', od_wait=100, verbose=MODEL_VERBOSE)
    if booster == 'lgb':
        base = dict(n_estimators=n_estimators, learning_rate=0.03, num_leaves=63, min_child_samples=30, subsample=0.9, colsample_bytree=0.9, random_state=seed, verbosity=-1, n_jobs=-1)
        if task == 'goal': return {**base, 'objective':'poisson'}
        if task == 'outcome': return {**base, 'objective':'multiclass', 'num_class':3}
        if task == 'tail': return {**base, 'objective':'binary'}
    raise ValueError((booster, task))

def make_trial_params(booster, task, trial=None, seed=42, n_estimators=550):
    params = make_default_params(booster, task, seed, n_estimators)
    if trial is None: return params
    if booster == 'cat':
        params['depth'] = trial.suggest_int('depth', 5, 8); params['learning_rate'] = trial.suggest_float('learning_rate', 0.02, 0.06, log=True); params['l2_leaf_reg'] = trial.suggest_float('l2_leaf_reg', 3.0, 8.0)
    elif booster == 'lgb':
        params['num_leaves'] = trial.suggest_int('num_leaves', 31, 95); params['learning_rate'] = trial.suggest_float('learning_rate', 0.02, 0.06, log=True); params['min_child_samples'] = trial.suggest_int('min_child_samples', 20, 60); params['subsample'] = trial.suggest_float('subsample', 0.75, 1.0); params['colsample_bytree'] = trial.suggest_float('colsample_bytree', 0.75, 1.0)
    return params

def prepare_model_frame(df, feature_cols, cat_features):
    X = select_unique_columns(df, feature_cols)
    for c in cat_features:
        if c in X.columns: X[c] = X[c].astype('category')
    for c in X.columns:
        if c not in cat_features: X[c] = pd.to_numeric(X[c], errors='coerce')
    return X

def fit_goal_model(Xtr, ytr, Xva, yva, booster, params, cats):
    if booster == 'cat':
        m = CatBoostRegressor(**params); m.fit(Xtr, ytr, eval_set=(Xva, yva), cat_features=[c for c in cats if c in Xtr.columns], use_best_model=True, verbose=params.get('verbose', MODEL_VERBOSE)); return m
    if booster == 'lgb':
        m = lgb.LGBMRegressor(**params); m.fit(Xtr, ytr, eval_set=[(Xva, yva)], categorical_feature=[c for c in cats if c in Xtr.columns], callbacks=[lgb.early_stopping(100, verbose=LGB_EARLY_STOP_VERBOSE)]); return m
    raise ValueError(booster)

def fit_outcome_model(Xtr, ytr, Xva, yva, booster, params, cats):
    if booster == 'cat':
        m = CatBoostClassifier(**params); m.fit(Xtr, ytr, eval_set=(Xva, yva), cat_features=[c for c in cats if c in Xtr.columns], use_best_model=True, verbose=params.get('verbose', MODEL_VERBOSE)); return m
    if booster == 'lgb':
        m = lgb.LGBMClassifier(**params); m.fit(Xtr, ytr, eval_set=[(Xva, yva)], categorical_feature=[c for c in cats if c in Xtr.columns], callbacks=[lgb.early_stopping(100, verbose=LGB_EARLY_STOP_VERBOSE)]); return m
    raise ValueError(booster)

def fit_tail_model(Xtr, ytr, Xva, yva, booster, params, cats):
    if booster == 'cat':
        m = CatBoostClassifier(**params); m.fit(Xtr, ytr, eval_set=(Xva, yva), cat_features=[c for c in cats if c in Xtr.columns], use_best_model=True, verbose=params.get('verbose', MODEL_VERBOSE)); return m
    if booster == 'lgb':
        m = lgb.LGBMClassifier(**params); m.fit(Xtr, ytr, eval_set=[(Xva, yva)], categorical_feature=[c for c in cats if c in Xtr.columns], callbacks=[lgb.early_stopping(100, verbose=LGB_EARLY_STOP_VERBOSE)]); return m
    raise ValueError(booster)

def predict_reg(m, X): return np.clip(np.asarray(m.predict(X), dtype=float), 0, 30)
def predict_proba3(m, X):
    p = np.asarray(m.predict_proba(X), dtype=float)
    if p.shape[1] < 3:
        out = np.zeros((len(X), 3)) + 1e-6
        for i, cls in enumerate(getattr(m, 'classes_', range(p.shape[1]))): out[:, int(cls)] = p[:, i]
        p = out
    return p / p.sum(axis=1, keepdims=True)
def predict_pos(m, X):
    p = np.asarray(m.predict_proba(X), dtype=float)
    return np.clip(p[:, 1] if p.ndim == 2 and p.shape[1] > 1 else p.ravel(), 1e-6, 1-1e-6)

def filter_regime_train(df, regime):
    if regime == 'full_history': return df.copy()
    if regime == 'modern_cutoff_1990': return df[pd.to_datetime(df['date']) >= pd.Timestamp('1990-01-01')].copy()
    raise ValueError(regime)

def fit_gender_split_goal_models_only(train_df, valid_df, booster_name, regime_name, gender_value, params):
    tr = train_df[train_df['gender'].astype(str) == str(gender_value)].copy(); va = valid_df[valid_df['gender'].astype(str) == str(gender_value)].copy()
    feature_cols, cats = params['feature_cols'], params['cat_features']
    tr = select_unique_columns(tr, ['match_id','gender','tournament','date','y_goal_a','y_goal_b'] + feature_cols); va = select_unique_columns(va, ['match_id','gender','tournament','date','y_goal_a','y_goal_b'] + feature_cols)
    Xtr, Xva = prepare_model_frame(tr, feature_cols, cats), prepare_model_frame(va, feature_cols, cats)
    print(f'[INFO] fit goals-only | booster={booster_name} | regime={regime_name} | gender={gender_value} | n_train={len(tr)} | n_valid={len(va)}')
    return {'booster_name':booster_name, 'regime_name':regime_name, 'gender_value':gender_value, 'feature_cols':feature_cols, 'cat_features':cats,
            'goal_a':fit_goal_model(Xtr, tr['y_goal_a'], Xva, va['y_goal_a'], booster_name, params['goal_params'], cats),
            'goal_b':fit_goal_model(Xtr, tr['y_goal_b'], Xva, va['y_goal_b'], booster_name, params['goal_params'], cats)}

def fit_gender_split_full_heads(train_df, valid_df, booster_name, regime_name, gender_value, params):
    tr = train_df[train_df['gender'].astype(str) == str(gender_value)].copy(); va = valid_df[valid_df['gender'].astype(str) == str(gender_value)].copy()
    feature_cols, cats = params['feature_cols'], params['cat_features']
    targets = ['y_goal_a','y_goal_b','y_outcome','is_team_a_blowout_5plus','is_team_b_blowout_5plus','is_team_a_blowout_7plus','is_team_b_blowout_7plus','is_high_total_6plus']
    tr = select_unique_columns(tr, ['match_id','gender','tournament','date'] + targets + feature_cols); va = select_unique_columns(va, ['match_id','gender','tournament','date'] + targets + feature_cols)
    Xtr, Xva = prepare_model_frame(tr, feature_cols, cats), prepare_model_frame(va, feature_cols, cats)
    print(f'[INFO] fit full heads | booster={booster_name} | regime={regime_name} | gender={gender_value} | n_train={len(tr)} | n_valid={len(va)}')
    return {'booster_name':booster_name, 'regime_name':regime_name, 'gender_value':gender_value, 'feature_cols':feature_cols, 'cat_features':cats,
            'goal_a':fit_goal_model(Xtr, tr['y_goal_a'], Xva, va['y_goal_a'], booster_name, params['goal_params'], cats),
            'goal_b':fit_goal_model(Xtr, tr['y_goal_b'], Xva, va['y_goal_b'], booster_name, params['goal_params'], cats),
            'outcome':fit_outcome_model(Xtr, tr['y_outcome'], Xva, va['y_outcome'], booster_name, params['outcome_params'], cats),
            'tail_a5':fit_tail_model(Xtr, tr['is_team_a_blowout_5plus'], Xva, va['is_team_a_blowout_5plus'], booster_name, params['tail_params'], cats),
            'tail_b5':fit_tail_model(Xtr, tr['is_team_b_blowout_5plus'], Xva, va['is_team_b_blowout_5plus'], booster_name, params['tail_params'], cats),
            'tail_a7':fit_tail_model(Xtr, tr['is_team_a_blowout_7plus'], Xva, va['is_team_a_blowout_7plus'], booster_name, params['tail_params'], cats),
            'tail_b7':fit_tail_model(Xtr, tr['is_team_b_blowout_7plus'], Xva, va['is_team_b_blowout_7plus'], booster_name, params['tail_params'], cats),
            'tail_ht':fit_tail_model(Xtr, tr['is_high_total_6plus'], Xva, va['is_high_total_6plus'], booster_name, params['tail_params'], cats)}

def predict_gender_split_raw_outputs(df, models_by_gender):
    rows = []
    for g, bundle in models_by_gender.items():
        sub = df[df['gender'].astype(str) == str(g)].copy()
        if len(sub) == 0: continue
        X = prepare_model_frame(sub, bundle['feature_cols'], bundle['cat_features'])
        pa, pb = predict_reg(bundle['goal_a'], X), predict_reg(bundle['goal_b'], X)
        out = pd.DataFrame({'match_id':sub['match_id'].values, 'gender':sub['gender'].astype(str).values, 'tournament':sub['tournament'].astype(str).values, 'pred_goal_a_cont':pa, 'pred_goal_b_cont':pb})
        if 'actual_team_a_goals' in sub.columns: out['actual_team_a_goals'] = pd.to_numeric(sub['actual_team_a_goals'], errors='coerce').values
        if 'actual_team_b_goals' in sub.columns: out['actual_team_b_goals'] = pd.to_numeric(sub['actual_team_b_goals'], errors='coerce').values
        if 'outcome' in bundle:
            p = predict_proba3(bundle['outcome'], X)
        else:
            gd = pa - pb; p = np.vstack([1/(1+np.exp(-gd)), np.exp(-np.abs(gd)), 1/(1+np.exp(gd))]).T; p = p / p.sum(axis=1, keepdims=True)
        out['pred_outcome_proba_0'], out['pred_outcome_proba_1'], out['pred_outcome_proba_2'] = p[:,0], p[:,1], p[:,2]
        if 'tail_a5' in bundle:
            out['prob_a_blowout_5plus'] = predict_pos(bundle['tail_a5'], X); out['prob_b_blowout_5plus'] = predict_pos(bundle['tail_b5'], X); out['prob_a_blowout_7plus'] = predict_pos(bundle['tail_a7'], X); out['prob_b_blowout_7plus'] = predict_pos(bundle['tail_b7'], X); out['prob_high_total'] = predict_pos(bundle['tail_ht'], X)
        rows.append(out)
    return pd.concat(rows, ignore_index=True).sort_values('match_id').reset_index(drop=True)


def make_head_feature_config(goal_feature_cols, goal_cat_features, outcome_feature_cols, outcome_cat_features, tail_feature_cols, tail_cat_features):
    return {
        'goal': {'feature_cols': list(goal_feature_cols), 'cat_features': list(goal_cat_features)},
        'outcome': {'feature_cols': list(outcome_feature_cols), 'cat_features': list(outcome_cat_features)},
        'tail': {'feature_cols': list(tail_feature_cols), 'cat_features': list(tail_cat_features)},
    }

HEAD_CONFIGS = {
    'V0_baseline': make_head_feature_config(exp05a_feature_cols, exp05a_cat_features, exp05a_feature_cols, exp05a_cat_features, exp05a_feature_cols, exp05a_cat_features),
    'V1_allhead': make_head_feature_config(exp09a_feature_cols, exp09a_cat_features, exp09a_feature_cols, exp09a_cat_features, exp09a_feature_cols, exp09a_cat_features),
    'V2_outcome_only': make_head_feature_config(exp05a_feature_cols, exp05a_cat_features, exp09a_feature_cols, exp09a_cat_features, exp05a_feature_cols, exp05a_cat_features),
    'V3_outcome_tail': make_head_feature_config(exp05a_feature_cols, exp05a_cat_features, exp09a_feature_cols, exp09a_cat_features, exp09a_feature_cols, exp09a_cat_features),
}

def fit_gender_split_full_heads_with_config(train_df, valid_df, booster_name, regime_name, gender_value, params):
    tr = train_df[train_df['gender'].astype(str) == str(gender_value)].copy()
    va = valid_df[valid_df['gender'].astype(str) == str(gender_value)].copy()

    goal_cols = params['head_feature_config']['goal']['feature_cols']
    goal_cats = params['head_feature_config']['goal']['cat_features']
    outcome_cols = params['head_feature_config']['outcome']['feature_cols']
    outcome_cats = params['head_feature_config']['outcome']['cat_features']
    tail_cols_cfg = params['head_feature_config']['tail']['feature_cols']
    tail_cats = params['head_feature_config']['tail']['cat_features']

    Xtr_goal = prepare_model_frame(tr, goal_cols, goal_cats)
    Xva_goal = prepare_model_frame(va, goal_cols, goal_cats)
    Xtr_out = prepare_model_frame(tr, outcome_cols, outcome_cats)
    Xva_out = prepare_model_frame(va, outcome_cols, outcome_cats)
    Xtr_tail = prepare_model_frame(tr, tail_cols_cfg, tail_cats)
    Xva_tail = prepare_model_frame(va, tail_cols_cfg, tail_cats)

    print(
        f'[INFO] fit full heads by config | booster={booster_name} | regime={regime_name} | '
        f'gender={gender_value} | n_train={len(tr)} | n_valid={len(va)} | '
        f'goal_features={len(goal_cols)} | outcome_features={len(outcome_cols)} | tail_features={len(tail_cols_cfg)}',
        flush=True
    )

    return {
        'booster_name': booster_name,
        'regime_name': regime_name,
        'gender_value': gender_value,
        'head_feature_config': copy.deepcopy(params['head_feature_config']),
        'goal_a': fit_goal_model(Xtr_goal, tr['y_goal_a'], Xva_goal, va['y_goal_a'], booster_name, params['goal_params'], goal_cats),
        'goal_b': fit_goal_model(Xtr_goal, tr['y_goal_b'], Xva_goal, va['y_goal_b'], booster_name, params['goal_params'], goal_cats),
        'outcome': fit_outcome_model(Xtr_out, tr['y_outcome'], Xva_out, va['y_outcome'], booster_name, params['outcome_params'], outcome_cats),
        'tail_a5': fit_tail_model(Xtr_tail, tr['is_team_a_blowout_5plus'], Xva_tail, va['is_team_a_blowout_5plus'], booster_name, params['tail_params'], tail_cats),
        'tail_b5': fit_tail_model(Xtr_tail, tr['is_team_b_blowout_5plus'], Xva_tail, va['is_team_b_blowout_5plus'], booster_name, params['tail_params'], tail_cats),
        'tail_a7': fit_tail_model(Xtr_tail, tr['is_team_a_blowout_7plus'], Xva_tail, va['is_team_a_blowout_7plus'], booster_name, params['tail_params'], tail_cats),
        'tail_b7': fit_tail_model(Xtr_tail, tr['is_team_b_blowout_7plus'], Xva_tail, va['is_team_b_blowout_7plus'], booster_name, params['tail_params'], tail_cats),
        'tail_ht': fit_tail_model(Xtr_tail, tr['is_high_total_6plus'], Xva_tail, va['is_high_total_6plus'], booster_name, params['tail_params'], tail_cats),
    }

def predict_gender_split_raw_outputs_with_config(df, models_by_gender):
    rows = []
    for g, bundle in models_by_gender.items():
        sub = df[df['gender'].astype(str) == str(g)].copy()
        if len(sub) == 0:
            continue
        cfg = bundle['head_feature_config']
        X_goal = prepare_model_frame(sub, cfg['goal']['feature_cols'], cfg['goal']['cat_features'])
        X_out = prepare_model_frame(sub, cfg['outcome']['feature_cols'], cfg['outcome']['cat_features'])
        X_tail = prepare_model_frame(sub, cfg['tail']['feature_cols'], cfg['tail']['cat_features'])

        pa = predict_reg(bundle['goal_a'], X_goal)
        pb = predict_reg(bundle['goal_b'], X_goal)
        out = pd.DataFrame({
            'match_id': sub['match_id'].values,
            'gender': sub['gender'].astype(str).values,
            'tournament': sub['tournament'].astype(str).values,
            'pred_goal_a_cont': pa,
            'pred_goal_b_cont': pb,
        })
        if 'tournament_structure' in sub.columns:
            out['tournament_structure'] = sub['tournament_structure'].astype(str).values
        if 'actual_team_a_goals' in sub.columns:
            out['actual_team_a_goals'] = pd.to_numeric(sub['actual_team_a_goals'], errors='coerce').values
        if 'actual_team_b_goals' in sub.columns:
            out['actual_team_b_goals'] = pd.to_numeric(sub['actual_team_b_goals'], errors='coerce').values

        p = predict_proba3(bundle['outcome'], X_out)
        out['pred_outcome_proba_0'], out['pred_outcome_proba_1'], out['pred_outcome_proba_2'] = p[:, 0], p[:, 1], p[:, 2]
        out['prob_a_blowout_5plus'] = predict_pos(bundle['tail_a5'], X_tail)
        out['prob_b_blowout_5plus'] = predict_pos(bundle['tail_b5'], X_tail)
        out['prob_a_blowout_7plus'] = predict_pos(bundle['tail_a7'], X_tail)
        out['prob_b_blowout_7plus'] = predict_pos(bundle['tail_b7'], X_tail)
        out['prob_high_total'] = predict_pos(bundle['tail_ht'], X_tail)
        rows.append(out)

    return pd.concat(rows, ignore_index=True).sort_values('match_id').reset_index(drop=True)

def train_final_gender_models_by_head_config(train_df, valid_df, final_selected, head_feature_config, n_estimators=1200):
    models = {}
    for gender, cfg in final_selected.items():
        booster, regime = cfg['booster'], cfg['regime']
        tr = filter_regime_train(train_df, regime)
        models[gender] = fit_gender_split_full_heads_with_config(
            tr,
            valid_df,
            booster,
            regime,
            gender,
            {
                'head_feature_config': head_feature_config,
                'goal_params': make_default_params(booster, 'goal', SEED, n_estimators),
                'outcome_params': make_default_params(booster, 'outcome', SEED, n_estimators),
                'tail_params': make_default_params(booster, 'tail', SEED, n_estimators),
            },
        )
    return models


def build_scoreline_prior(goal_a, goal_b, max_goals, alpha=1.0):
    counts = np.zeros((max_goals+1, max_goals+1), dtype=float) + alpha
    for a,b in zip(goal_a, goal_b): counts[int(np.clip(a,0,max_goals)), int(np.clip(b,0,max_goals))] += 1
    counts /= counts.sum()
    return {(a,b):float(counts[a,b]) for a in range(max_goals+1) for b in range(max_goals+1)}
scoreline_prior_lookup = {mg:build_scoreline_prior(y_train_fold['y_goal_a'], y_train_fold['y_goal_b'], mg, 1.0) for mg in [7,9,11]}

def decode_anchor_batch_from_raw(raw_df, params, prior, eps=1e-9):
    max_goals = int(params['MAX_GOALS']); cand = [(a,b) for a in range(max_goals+1) for b in range(max_goals+1)]
    ca, cb = np.array([x[0] for x in cand]), np.array([x[1] for x in cand]); co = np.array([_outcome(a,b) for a,b in cand])
    prior_cost = np.array([-np.log(prior.get((a,b), eps)+eps) for a,b in cand])
    pred_a = raw_df['pred_goal_a_cont'].to_numpy(float)[:,None]; pred_b = raw_df['pred_goal_b_cont'].to_numpy(float)[:,None]
    p = raw_df[['pred_outcome_proba_0','pred_outcome_proba_1','pred_outcome_proba_2']].to_numpy(float)
    cost = params['w_direct']*(np.abs(ca-pred_a)+np.abs(cb-pred_b)) + params['w_outcome']*(-np.log(p[:,co]+eps)) + params['w_prior']*prior_cost
    idx = np.argmin(cost, axis=1)
    return pd.DataFrame({'match_id':raw_df['match_id'].values, 'pred_team_a_goals':ca[idx].astype(int), 'pred_team_b_goals':cb[idx].astype(int)})

def decode_directional_tail_batch(raw_df, tail_prob_df, params, prior, eps=1e-9):
    tail_cols = ['prob_a_blowout_5plus','prob_b_blowout_5plus','prob_a_blowout_7plus','prob_b_blowout_7plus','prob_high_total']
    raw_clean = raw_df.drop(columns=tail_cols, errors='ignore')
    merged = raw_clean.merge(tail_prob_df[['match_id']+tail_cols], on='match_id', how='left', validate='one_to_one')
    assert not any(c.endswith('_x') or c.endswith('_y') for c in merged.columns), 'suffix leak in tail merge'
    max_goals = int(params['MAX_GOALS']); cand = [(a,b) for a in range(max_goals+1) for b in range(max_goals+1)]
    ca, cb = np.array([x[0] for x in cand]), np.array([x[1] for x in cand]); co = np.array([_outcome(a,b) for a,b in cand])
    prior_cost = np.array([-np.log(prior.get((a,b), eps)+eps) for a,b in cand])
    pred_a = merged['pred_goal_a_cont'].to_numpy(float)[:,None]; pred_b = merged['pred_goal_b_cont'].to_numpy(float)[:,None]
    p = merged[['pred_outcome_proba_0','pred_outcome_proba_1','pred_outcome_proba_2']].to_numpy(float)
    cost = params['w_direct']*(np.abs(ca-pred_a)+np.abs(cb-pred_b)) + params['w_outcome']*(-np.log(p[:,co]+eps)) + params['w_prior']*prior_cost
    for c in tail_cols: merged[c] = pd.to_numeric(merged[c], errors='coerce').fillna(eps).clip(eps,1-eps)
    masks = {'w_a5':(ca-cb)>=5, 'w_b5':(cb-ca)>=5, 'w_a7':(ca-cb)>=7, 'w_b7':(cb-ca)>=7, 'w_ht':(ca+cb)>=6}
    probs = {'w_a5':'prob_a_blowout_5plus', 'w_b5':'prob_b_blowout_5plus', 'w_a7':'prob_a_blowout_7plus', 'w_b7':'prob_b_blowout_7plus', 'w_ht':'prob_high_total'}
    for w, mask in masks.items(): cost -= float(params.get(w,0))*mask*np.log(merged[probs[w]].to_numpy(float)[:,None]+eps)
    idx = np.argmin(cost, axis=1)
    return pd.DataFrame({'match_id':merged['match_id'].values, 'pred_team_a_goals':ca[idx].astype(int), 'pred_team_b_goals':cb[idx].astype(int)})

def tune_decoder_from_raw_outputs(raw_df, grid, priors):
    rows, cache = [], []
    for combo in iter_progress(list(product(*[grid[k] for k in grid])), desc='tune_decoder'):
        params = {k:v for k,v in zip(grid.keys(), combo)}; params['MAX_GOALS'] = int(params['MAX_GOALS'])
        pred = decode_anchor_batch_from_raw(raw_df, params, priors[params['MAX_GOALS']])
        score = awmae_score(raw_df['actual_team_a_goals'], raw_df['actual_team_b_goals'], pred['pred_team_a_goals'], pred['pred_team_b_goals'], raw_df['tournament'])
        rows.append({**params, 'valid_awmae':score}); cache.append((score, params, pred))
    best = sorted(cache, key=lambda x:x[0])[0]
    return pd.DataFrame(rows).sort_values('valid_awmae').reset_index(drop=True), best[1], best[2]

def tune_tail_aware_decoder_from_raw_outputs(raw_df, tail_prob_df, grid, priors, top_n=3):
    rows, cache = [], []
    raw_clean = raw_df.drop(columns=['prob_a_blowout_5plus','prob_b_blowout_5plus','prob_a_blowout_7plus','prob_b_blowout_7plus','prob_high_total'], errors='ignore')
    for combo in iter_progress(list(product(*[grid[k] for k in grid])), desc='tune_tail_decoder'):
        params = {k:v for k,v in zip(grid.keys(), combo)}; params['MAX_GOALS'] = int(params['MAX_GOALS'])
        pred = decode_directional_tail_batch(raw_clean, tail_prob_df, params, priors[params['MAX_GOALS']])
        score = awmae_score(raw_df['actual_team_a_goals'], raw_df['actual_team_b_goals'], pred['pred_team_a_goals'], pred['pred_team_b_goals'], raw_df['tournament'])
        rows.append({**params, 'valid_awmae':score}); cache.append((score, params, pred))
    best = sorted(cache, key=lambda x:x[0])[0]
    return pd.DataFrame(rows).sort_values('valid_awmae').reset_index(drop=True), best[1], best[2]


# %% [markdown]
# # 11. Anchor Validation Decode Fix
#
# Bagian ini memperbaiki sumber error lama `KeyError: 'gender'` langsung di logic asal. `valid_decode_df` dibuat dari `valid_feature_full.copy()` lalu hanya merge actual goals.
#
# Dengan pola ini, `gender` dan `tournament` tidak di-merge ulang, sehingga tidak muncul kolom suffix seperti `gender_x`, `gender_y`, `tournament_x`, atau `tournament_y`. Assert ditambahkan agar kerusakan dataframe langsung ketahuan sebelum decode.
# %%
train_sup_full = train_feature_full.merge(y_train_fold, on='match_id', how='left', validate='one_to_one')
valid_sup_full = valid_feature_full.merge(y_valid_fold, on='match_id', how='left', validate='one_to_one')

# Anchor validation decode fix from source.
# valid_feature_full already has gender and tournament, so merge only actual goals.
valid_decode_df = valid_feature_full.copy()
valid_decode_df = valid_decode_df.drop(columns=['actual_team_a_goals', 'actual_team_b_goals'], errors='ignore')
valid_target_df = valid_fold_base[['match_id', 'team_a_goals', 'team_b_goals']].rename(columns={
    'team_a_goals': 'actual_team_a_goals',
    'team_b_goals': 'actual_team_b_goals',
})
valid_decode_df = valid_decode_df.merge(valid_target_df, on='match_id', how='left', validate='one_to_one')
required_cols = ['match_id', 'gender', 'tournament', 'actual_team_a_goals', 'actual_team_b_goals']
missing_cols = [c for c in required_cols if c not in valid_decode_df.columns]
assert len(missing_cols) == 0, f'valid_decode_df missing columns: {missing_cols}'
suffix_cols = [c for c in valid_decode_df.columns if c.endswith('_x') or c.endswith('_y')]
assert len(suffix_cols) == 0, f'valid_decode_df has bad suffix columns: {suffix_cols}'
assert valid_decode_df['actual_team_a_goals'].notna().all()
assert valid_decode_df['actual_team_b_goals'].notna().all()
log_check('Anchor validation decode has gender', 'gender' in valid_decode_df.columns)
log_check('Anchor validation decode has no suffix columns', len(suffix_cols) == 0)


def run_screening_unit(train_df, valid_df, booster, regime, gender, n_trials=10):
    train_regime = filter_regime_train(train_df, regime)
    def objective(trial):
        params = make_trial_params(booster, 'goal', trial, SEED+trial.number, 550)
        bundle = fit_gender_split_goal_models_only(train_regime, valid_df, booster, regime, gender, {'feature_cols':student_feature_cols, 'cat_features':student_cat_features, 'goal_params':params})
        vg = valid_df[valid_df['gender'].astype(str)==str(gender)].rename(columns={'y_goal_a':'actual_team_a_goals','y_goal_b':'actual_team_b_goals'})
        va_decode = select_unique_columns(vg, ['match_id','gender','tournament','actual_team_a_goals','actual_team_b_goals'] + student_feature_cols)
        raw = predict_gender_split_raw_outputs(va_decode, {gender:bundle})
        pa, pb = np.clip(np.rint(raw['pred_goal_a_cont']),0,9).astype(int), np.clip(np.rint(raw['pred_goal_b_cont']),0,9).astype(int)
        return awmae_score(raw['actual_team_a_goals'], raw['actual_team_b_goals'], pa, pb, raw['tournament'])
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return {'regime':regime, 'gender':gender, 'booster':booster, 'screen_valid_awmae':float(study.best_value), 'best_params':study.best_params, 'n_trials':n_trials, 'status':'ok'}

screen_rows = []
for regime in CUTOFF_REGIMES:
    for gender in sorted(train_fold_base['gender'].astype(str).unique()):
        for booster in BOOSTERS_TO_RUN:
            try: screen_rows.append(run_screening_unit(train_sup_full, valid_sup_full, booster, regime, gender, SCREEN_N_TRIALS))
            except Exception as e: screen_rows.append({'regime':regime,'gender':gender,'booster':booster,'screen_valid_awmae':np.inf,'best_params':{},'n_trials':SCREEN_N_TRIALS,'status':f'failed: {repr(e)}'})
screening_results = pd.DataFrame(screen_rows).sort_values(['gender','screen_valid_awmae']).reset_index(drop=True)
screening_results.to_csv(SUM_DIR/'booster_screening_results.csv', index=False)
display(screening_results)
with open(SUM_DIR/'xgb_policy.json','w') as f: json.dump({'xgb_policy':XGB_POLICY, 'reason':'skipped for clean fix-v2 due prior unseen categorical error'}, f, indent=2)

def make_temporal_inner_splits(df, n_splits=2):
    d = df.sort_values(['date','match_id']).reset_index(drop=True); n=len(d); out=[]
    for frac in [0.60, 0.70][:n_splits]:
        cut=int(n*frac); vs=max(1,int(n*0.15)); tr=d.iloc[:cut].copy(); va=d.iloc[cut:min(n,cut+vs)].copy()
        if len(tr) and len(va): out.append((tr,va))
    return out

def run_refine_for_candidate(train_df, booster, regime, gender, n_trials=20):
    train_g = filter_regime_train(train_df, regime)
    train_g = train_g[train_g['gender'].astype(str)==str(gender)].copy()
    splits = make_temporal_inner_splits(train_g, 2)
    def objective(trial):
        scores=[]; params = make_trial_params(booster, 'goal', trial, SEED+1000+trial.number, 850)
        for tr, va in splits:
            bundle = fit_gender_split_goal_models_only(tr, va, booster, regime, gender, {'feature_cols':student_feature_cols, 'cat_features':student_cat_features, 'goal_params':params})
            va_decode = select_unique_columns(va, ['match_id','gender','tournament','y_goal_a','y_goal_b'] + student_feature_cols).rename(columns={'y_goal_a':'actual_team_a_goals','y_goal_b':'actual_team_b_goals'})
            assert_no_duplicate_columns(va_decode, 'refine va_decode')
            raw = predict_gender_split_raw_outputs(va_decode, {gender:bundle})
            pa, pb = np.clip(np.rint(raw['pred_goal_a_cont']),0,9).astype(int), np.clip(np.rint(raw['pred_goal_b_cont']),0,9).astype(int)
            scores.append(awmae_score(raw['actual_team_a_goals'], raw['actual_team_b_goals'], pa, pb, raw['tournament']))
        return float(np.mean(scores))
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=SEED+99))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return {'regime':regime, 'gender':gender, 'booster':booster, 'refine_valid_awmae':float(study.best_value), 'best_params':study.best_params, 'n_trials':n_trials, 'status':'ok', 'source':'refine'}

refine_candidates = screening_results[screening_results['status'].eq('ok')].sort_values('screen_valid_awmae').groupby('gender', as_index=False).head(1).copy()
refine_rows=[]
for _, r in refine_candidates.iterrows():
    try: refine_rows.append(run_refine_for_candidate(train_sup_full, r['booster'], r['regime'], r['gender'], REFINE_N_TRIALS))
    except Exception as e: refine_rows.append({'regime':r['regime'], 'gender':r['gender'], 'booster':r['booster'], 'refine_valid_awmae':np.inf, 'best_params':{}, 'n_trials':REFINE_N_TRIALS, 'status':f'failed: {repr(e)}', 'source':'refine_failed'})
refine_results = pd.DataFrame(refine_rows).sort_values(['gender','refine_valid_awmae']).reset_index(drop=True)
refine_results.to_csv(SUM_DIR/'booster_refine_results.csv', index=False)
display(refine_results)

final_selected = {}
for gender in sorted(train_fold_base['gender'].astype(str).unique()):
    rr = refine_results[(refine_results['gender'].astype(str)==str(gender)) & refine_results['status'].eq('ok')]
    if len(rr):
        best = rr.sort_values('refine_valid_awmae').iloc[0]
        final_selected[str(gender)] = {'booster':str(best['booster']), 'regime':str(best['regime']), 'source':'refine', 'valid_awmae':float(best['refine_valid_awmae'])}
    else:
        ss = screening_results[(screening_results['gender'].astype(str)==str(gender)) & screening_results['status'].eq('ok')].sort_values('screen_valid_awmae')
        best = ss.iloc[0]
        final_selected[str(gender)] = {'booster':str(best['booster']), 'regime':str(best['regime']), 'source':'screen_fallback', 'valid_awmae':float(best['screen_valid_awmae']), 'fallback_reason':'refine_failed'}
with open(SUM_DIR/'final_selected_boosters.json','w',encoding='utf-8') as f: json.dump(final_selected, f, indent=2, ensure_ascii=False)
final_selected_df = pd.DataFrame([{'gender': g, **cfg} for g, cfg in final_selected.items()])
display(final_selected_df)


def prediction_detail_table(pred_df, variant_name):
    df = pred_df.copy()
    df['variant_name'] = variant_name
    df['base_mae'] = (
        (df['actual_team_a_goals'] - df['pred_team_a_goals']).abs()
        + (df['actual_team_b_goals'] - df['pred_team_b_goals']).abs()
    ) / 2
    df['exact_hit'] = (
        (df['actual_team_a_goals'].astype(int) == df['pred_team_a_goals'].astype(int))
        & (df['actual_team_b_goals'].astype(int) == df['pred_team_b_goals'].astype(int))
    ).astype(int)
    df['true_outcome'] = [_outcome(a, b) for a, b in zip(df['actual_team_a_goals'], df['actual_team_b_goals'])]
    df['pred_outcome'] = [_outcome(a, b) for a, b in zip(df['pred_team_a_goals'], df['pred_team_b_goals'])]
    df['outcome_hit'] = (df['true_outcome'] == df['pred_outcome']).astype(int)
    df['true_gd'] = df['actual_team_a_goals'] - df['actual_team_b_goals']
    df['pred_gd'] = df['pred_team_a_goals'] - df['pred_team_b_goals']
    df['gd_hit'] = (df['true_gd'] == df['pred_gd']).astype(int)
    df['true_total'] = df['actual_team_a_goals'] + df['actual_team_b_goals']
    df['pred_total'] = df['pred_team_a_goals'] + df['pred_team_b_goals']
    df['abs_true_gd'] = df['true_gd'].abs()
    df['is_blowout_5plus'] = (df['abs_true_gd'] >= 5).astype(int)
    df['is_blowout_7plus'] = (df['abs_true_gd'] >= 7).astype(int)
    df['is_high_total_6plus'] = (df['true_total'] >= 6).astype(int)
    df['scoreline'] = df['pred_team_a_goals'].astype(int).astype(str) + '-' + df['pred_team_b_goals'].astype(int).astype(str)
    df['official_loss'] = [
        official_match_loss(a, b, pa, pb)
        for a, b, pa, pb in zip(df['actual_team_a_goals'], df['actual_team_b_goals'], df['pred_team_a_goals'], df['pred_team_b_goals'])
    ]
    df['tournament_weight'] = [get_tournament_weight(t) for t in df['tournament']]
    return df

def summarize_prediction_detail(detail_df, variant_name):
    score_counts = detail_df['scoreline'].value_counts(normalize=True)
    top3_share = float(score_counts.head(3).sum()) if len(score_counts) else 0.0
    return {
        'variant_name': variant_name,
        'awmae': awmae_score(
            detail_df['actual_team_a_goals'], detail_df['actual_team_b_goals'],
            detail_df['pred_team_a_goals'], detail_df['pred_team_b_goals'],
            detail_df['tournament'],
        ),
        'base_mae': float(detail_df['base_mae'].mean()),
        'exact_rate': float(detail_df['exact_hit'].mean()),
        'outcome_rate': float(detail_df['outcome_hit'].mean()),
        'gd_rate': float(detail_df['gd_hit'].mean()),
        'team_bias': float((detail_df['pred_team_a_goals'] - detail_df['actual_team_a_goals']).mean()),
        'opp_bias': float((detail_df['pred_team_b_goals'] - detail_df['actual_team_b_goals']).mean()),
        'total_goal_bias': float((detail_df['pred_total'] - detail_df['true_total']).mean()),
        'mean_pred_total': float(detail_df['pred_total'].mean()),
        'top3_scoreline_share': top3_share,
    }

def compute_domain_error(detail_df, group_col):
    if group_col not in detail_df.columns:
        return pd.DataFrame()
    rows = []
    for key, g in detail_df.groupby(group_col, dropna=False):
        rows.append({
            group_col: key,
            'n': len(g),
            'awmae': awmae_score(g['actual_team_a_goals'], g['actual_team_b_goals'], g['pred_team_a_goals'], g['pred_team_b_goals'], g['tournament']),
            'base_mae': float(g['base_mae'].mean()),
            'exact_rate': float(g['exact_hit'].mean()),
            'outcome_rate': float(g['outcome_hit'].mean()),
            'gd_rate': float(g['gd_hit'].mean()),
        })
    return pd.DataFrame(rows).sort_values('awmae').reset_index(drop=True)

def compute_domain_error_by_variant(detail_df, group_col):
    if group_col not in detail_df.columns:
        return pd.DataFrame()
    rows = []
    group_keys = ['variant_name', group_col] if 'variant_name' in detail_df.columns else [group_col]
    for key, g in detail_df.groupby(group_keys, dropna=False):
        if isinstance(key, tuple):
            variant_name, domain_value = key
        else:
            variant_name, domain_value = None, key
        row = {
            group_col: domain_value,
            'n': len(g),
            'awmae': awmae_score(g['actual_team_a_goals'], g['actual_team_b_goals'], g['pred_team_a_goals'], g['pred_team_b_goals'], g['tournament']),
            'base_mae': float(g['base_mae'].mean()),
            'exact_rate': float(g['exact_hit'].mean()),
            'outcome_rate': float(g['outcome_hit'].mean()),
            'gd_rate': float(g['gd_hit'].mean()),
        }
        if variant_name is not None:
            row = {'variant_name': variant_name, **row}
        rows.append(row)
    sort_cols = ['variant_name', 'awmae'] if 'variant_name' in detail_df.columns else ['awmae']
    return pd.DataFrame(rows).sort_values(sort_cols).reset_index(drop=True)

def scoreline_distribution(detail_df):
    rows = []
    for variant, g in detail_df.groupby('variant_name'):
        vc = g['scoreline'].value_counts().reset_index()
        vc.columns = ['scoreline', 'count']
        vc['share'] = vc['count'] / len(g)
        vc['variant_name'] = variant
        rows.append(vc[['variant_name', 'scoreline', 'count', 'share']])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

TAIL_COLS = ['prob_a_blowout_5plus','prob_b_blowout_5plus','prob_a_blowout_7plus','prob_b_blowout_7plus','prob_high_total']
TAIL_GRID = {
    'MAX_GOALS': [7, 9],
    'w_direct': [0.75, 1.0],
    'w_outcome': [1.5, 2.0],
    'w_prior': [0.2, 0.5],
    'w_a5': [0.0, 0.5],
    'w_b5': [0.0, 0.5],
    'w_a7': [0.0, 0.5],
    'w_b7': [0.0, 0.5],
    'w_ht': [0.0, 0.5],
}

def run_exp09c_variant(variant_name, head_feature_config, save_name):
    log_section(f'EXP09C variant: {variant_name}')
    models = train_final_gender_models_by_head_config(train_sup_full, valid_sup_full, final_selected, head_feature_config, 1200)
    raw_valid = predict_gender_split_raw_outputs_with_config(valid_decode_df, models)
    tail_prob_valid = raw_valid[['match_id'] + TAIL_COLS].copy()
    decoder_grid, best_decoder, pred = tune_tail_aware_decoder_from_raw_outputs(
        raw_valid,
        tail_prob_valid,
        TAIL_GRID,
        scoreline_prior_lookup,
        top_n=3,
    )
    pred_match = raw_valid.merge(pred, on='match_id', how='left', validate='one_to_one')
    detail = prediction_detail_table(pred_match, variant_name)
    pred_path = PRED_DIR / f'valid_pred_match_{save_name}.csv'
    pred_match.to_csv(pred_path, index=False)
    log_saved(pred_path)
    return {
        'variant_name': variant_name,
        'models': models,
        'raw_valid': raw_valid,
        'pred_match': pred_match,
        'detail': detail,
        'decoder_grid': decoder_grid,
        'best_decoder': best_decoder,
        'summary': summarize_prediction_detail(detail, variant_name),
    }

VARIANT_RESULTS = {}


# %% [markdown]
# # 12. Variant V0 — EXP05A Baseline Feature Set
#
# Tahap ini menjalankan baseline pembanding. Semua head memakai fitur asli EXP05A, sehingga hasil V0 menjadi anchor internal untuk menilai apakah V2 benar-benar memberi perbaikan.
#
# Kalau V2 tidak melewati safety rule, final submission akan kembali ke V0 demi menjaga prinsip preserve baseline.
# %%
VARIANT_RESULTS['V0_baseline'] = run_exp09c_variant('V0_baseline', HEAD_CONFIGS['V0_baseline'], 'v0_baseline')


# %% [markdown]
# # 13. Variant V1 — All-Head Feature Repair
#
# Tahap ini menjalankan variant diagnostic yang memakai fitur EXP05A + EXP09A pada semua head. Variant ini dipakai untuk melihat ulang pola EXP09A secara internal.
#
# Catatan penting: V1 tidak boleh otomatis dipilih sebagai final, karena eksperimen sebelumnya menunjukkan all-head repair dapat memburuk di GT walaupun validasi terlihat menarik.
# %%
VARIANT_RESULTS['V1_allhead'] = run_exp09c_variant('V1_allhead', HEAD_CONFIGS['V1_allhead'], 'v1_allhead')


# %% [markdown]
# # 14. Variant V2 — Outcome-Only Feature Repair
#
# Tahap ini menjalankan variant utama EXP09C. Goal dan tail tetap memakai fitur EXP05A, sedangkan outcome memakai fitur EXP05A + EXP09A.
#
# Variant ini hanya dipilih sebagai final jika AW-MAE membaik dan safety metrics seperti outcome rate, exact rate, GD rate, bias, serta scoreline concentration tetap sehat.
# %%
VARIANT_RESULTS['V2_outcome_only'] = run_exp09c_variant('V2_outcome_only', HEAD_CONFIGS['V2_outcome_only'], 'v2_outcome_only')


# %% [markdown]
# # 15. Optional V3 — Outcome + Tail Feature Repair
#
# Bagian ini opsional dan default-nya tidak dijalankan. V3 menguji apakah fitur repair juga berguna untuk tail head, sementara goal head tetap memakai fitur EXP05A.
#
# Karena runtime EXP09C sudah cukup berat, variant ini hanya disarankan untuk eksperimen lanjutan setelah V0 sampai V2 lolos smoke test.
# %%
if RUN_V3_OPTIONAL:
    VARIANT_RESULTS['V3_outcome_tail'] = run_exp09c_variant('V3_outcome_tail', HEAD_CONFIGS['V3_outcome_tail'], 'v3_outcome_tail')
else:
    log_info('V3_outcome_tail skipped by default. Set RUN_V3_OPTIONAL=True to run it.')


# %% [markdown]
# # 16. Variant Comparison dan Domain Error Analysis
#
# Bagian ini menggabungkan hasil validasi semua variant, menghitung leaderboard, scoreline distribution, outcome-head comparison, dan domain error per subgroup.
#
# Semua tabel penting ditampilkan sebagai DataFrame agar output notebook lebih rapi. Domain error dihitung per variant supaya interpretasinya tidak tercampur antara V0, V1, dan V2.
# %%
all_detail_df = pd.concat([v['detail'] for v in VARIANT_RESULTS.values()], ignore_index=True)
variant_metrics_df = pd.DataFrame([v['summary'] for v in VARIANT_RESULTS.values()]).sort_values('awmae').reset_index(drop=True)
variant_metrics_df.to_csv(SUM_DIR / 'variant_metrics.csv', index=False)
display(variant_metrics_df)

v0 = variant_metrics_df[variant_metrics_df['variant_name'].eq('V0_baseline')].iloc[0].to_dict()
v1 = variant_metrics_df[variant_metrics_df['variant_name'].eq('V1_allhead')].iloc[0].to_dict()
v2 = variant_metrics_df[variant_metrics_df['variant_name'].eq('V2_outcome_only')].iloc[0].to_dict()
outcome_head_comparison = pd.DataFrame([
    {'comparison': 'V1_allhead_minus_V0', 'awmae_delta': v1['awmae'] - v0['awmae'], 'outcome_rate_delta': v1['outcome_rate'] - v0['outcome_rate'], 'exact_rate_delta': v1['exact_rate'] - v0['exact_rate'], 'gd_rate_delta': v1['gd_rate'] - v0['gd_rate']},
    {'comparison': 'V2_outcome_only_minus_V0', 'awmae_delta': v2['awmae'] - v0['awmae'], 'outcome_rate_delta': v2['outcome_rate'] - v0['outcome_rate'], 'exact_rate_delta': v2['exact_rate'] - v0['exact_rate'], 'gd_rate_delta': v2['gd_rate'] - v0['gd_rate']},
])
outcome_head_comparison.to_csv(SUM_DIR / 'outcome_head_comparison.csv', index=False)
display(outcome_head_comparison)

scoreline_dist = scoreline_distribution(all_detail_df)
scoreline_dist.to_csv(SUM_DIR / 'scoreline_distribution.csv', index=False)

compute_domain_error_by_variant(all_detail_df, 'gender').to_csv(SUM_DIR / 'domain_error_gender.csv', index=False)
if 'tournament_structure' in all_detail_df.columns:
    compute_domain_error_by_variant(all_detail_df, 'tournament_structure').to_csv(SUM_DIR / 'domain_error_tournament_structure.csv', index=False)
else:
    pd.DataFrame().to_csv(SUM_DIR / 'domain_error_tournament_structure.csv', index=False)

blowout_rows = []
for flag in ['is_blowout_5plus', 'is_blowout_7plus', 'is_high_total_6plus']:
    if flag in all_detail_df.columns:
        tmp = compute_domain_error_by_variant(all_detail_df.assign(domain=flag + '=' + all_detail_df[flag].astype(str)), 'domain')
        blowout_rows.append(tmp)
domain_error_blowout = pd.concat(blowout_rows, ignore_index=True) if blowout_rows else pd.DataFrame()
domain_error_blowout.to_csv(SUM_DIR / 'domain_error_blowout.csv', index=False)

baseline_row = variant_metrics_df[variant_metrics_df['variant_name'].eq('V0_baseline')].iloc[0].to_dict()
outcome_only_row = variant_metrics_df[variant_metrics_df['variant_name'].eq('V2_outcome_only')].iloc[0].to_dict()
best_row = variant_metrics_df.iloc[0].to_dict()

v2_awmae_gain = baseline_row['awmae'] - outcome_only_row['awmae']
v2_outcome_delta = outcome_only_row['outcome_rate'] - baseline_row['outcome_rate']
v2_exact_delta = outcome_only_row['exact_rate'] - baseline_row['exact_rate']
v2_gd_delta = outcome_only_row['gd_rate'] - baseline_row['gd_rate']
v2_top3_delta = outcome_only_row['top3_scoreline_share'] - baseline_row['top3_scoreline_share']

v2_safety_pass = (
    v2_awmae_gain > 0
    and v2_outcome_delta >= -0.002
    and v2_exact_delta >= -0.015
    and v2_gd_delta >= -0.020
    and v2_top3_delta <= 0.05
)

if v2_safety_pass:
    selected_variant = 'V2_outcome_only'
    decision_code = 'A'
    decision_text = 'Outcome-only feature repair improves AW-MAE and passes safety checks.'
elif v2_outcome_delta > 0 and v2_awmae_gain <= 0:
    selected_variant = 'V0_baseline'
    decision_code = 'B'
    decision_text = 'Outcome-only improves outcome signal but not AW-MAE, so baseline remains selected.'
elif best_row['variant_name'] == 'V0_baseline':
    selected_variant = 'V0_baseline'
    decision_code = 'C'
    decision_text = 'Baseline remains best.'
else:
    # Safety-first:
    # V1_allhead is diagnostic only because all-head EXP09A previously did not transfer well to GT.
    # If V2 does not pass the safety rule, do not select V1/V3 as final even if validation looks better.
    selected_variant = 'V0_baseline'
    decision_code = 'C'
    decision_text = (
        'A non-baseline variant has best validation AW-MAE, but outcome-only safety rule did not pass. '
        'For safety, final selected variant is V0_baseline.'
    )

selected_head_config = HEAD_CONFIGS[selected_variant]
selected_decoder = VARIANT_RESULTS[selected_variant]['best_decoder']

final_decision = {
    'experiment': 'EXP09C',
    'base_notebook': 'EXP05A-LITE-FIX-V2',
    'main_change': 'EXP09A repair features are used only for outcome head in V2.',
    'selected_variant': selected_variant,
    'decision': decision_code,
    'decision_text': decision_text,
    'baseline_awmae': float(baseline_row['awmae']),
    'outcome_only_awmae': float(outcome_only_row['awmae']),
    'outcome_only_awmae_gain_vs_baseline': float(v2_awmae_gain),
    'outcome_only_outcome_rate_delta': float(v2_outcome_delta),
    'outcome_only_exact_rate_delta': float(v2_exact_delta),
    'outcome_only_gd_rate_delta': float(v2_gd_delta),
    'caveat': 'Notebook preserves EXP05A helper/history/decoder style. Feature repair is routed per head.',
}

log_saved(SUM_DIR / 'variant_metrics.csv')
log_saved(SUM_DIR / 'outcome_head_comparison.csv')
log_saved(SUM_DIR / 'scoreline_distribution.csv')


# %% [markdown]
# # 17. Final Model Training dan Test Inference
#
# Bagian ini melatih ulang model final menggunakan selected variant dari decision rule. Hanya satu konfigurasi final yang dipakai untuk test inference.
#
# Jika V2 terpilih, maka goal/tail tetap memakai fitur EXP05A dan outcome memakai fitur EXP05A + EXP09A. Jika V2 tidak lolos safety, notebook akan fallback ke V0 baseline.
# %%
full_train_feature_df = train_feature_all.merge(y_full_train, on='match_id', how='left', validate='one_to_one')
full_models = train_final_gender_models_by_head_config(full_train_feature_df, valid_sup_full, final_selected, selected_head_config, 1200)
test_raw = predict_gender_split_raw_outputs_with_config(test_feature_full, full_models)
test_pred = decode_directional_tail_batch(
    test_raw.drop(columns=TAIL_COLS, errors='ignore'),
    test_raw[['match_id'] + TAIL_COLS],
    selected_decoder,
    scoreline_prior_lookup[int(selected_decoder['MAX_GOALS'])],
)

test_pred_match_best = test_pred.merge(test_match_base[['match_id','team_a','team_b']], on='match_id', how='left', validate='one_to_one')
test_pred_match_best['selected_variant'] = selected_variant
test_pred_match_best.to_csv(PRED_DIR / 'test_pred_exp09c_best_safe.csv', index=False)
log_saved(PRED_DIR / 'test_pred_exp09c_best_safe.csv')

print('[TEST INFERENCE SUMMARY]', flush=True)
log_info(f'selected_variant: {selected_variant}')
log_info(f'n_test_matches: {len(test_pred_match_best):,}')
log_info(f"mean_pred_total: {float((test_pred_match_best['pred_team_a_goals'] + test_pred_match_best['pred_team_b_goals']).mean()):.4f}")
log_info(f"max_pred_goal: {int(test_pred_match_best[['pred_team_a_goals','pred_team_b_goals']].max().max())}")
top_scores = (
    test_pred_match_best.assign(scoreline=lambda d: d['pred_team_a_goals'].astype(int).astype(str) + '-' + d['pred_team_b_goals'].astype(int).astype(str))
    ['scoreline'].value_counts().head(10).reset_index()
)
top_scores.columns = ['scoreline', 'count']
display(top_scores)


# %% [markdown]
# # 18. Submission Mapping dan Pair Consistency
#
# Bagian ini mengubah prediksi match-level kembali ke format row-level submission resmi: `Id`, `team_goals`, dan `opp_goals`.
#
# Validasi wajib dilakukan di sini: shape sama dengan sample submission, urutan `Id` sama, prediksi integer non-negative, tidak ada missing, tidak ada duplicate `Id`, dan pair consistency pass.
# %%
def match_predictions_to_submission(test_row_df, pred_match_df):
    row = test_row_df[['Id', 'match_id', 'team']].copy()
    pred = pred_match_df[['match_id', 'team_a', 'team_b', 'pred_team_a_goals', 'pred_team_b_goals']].copy()
    m = row.merge(pred, on='match_id', how='left', validate='many_to_one')
    is_a = m['team'].astype(str) == m['team_a'].astype(str)
    is_b = m['team'].astype(str) == m['team_b'].astype(str)
    assert (is_a | is_b).all(), 'Some test rows cannot be mapped to team_a/team_b.'
    m['team_goals'] = np.where(is_a, m['pred_team_a_goals'], m['pred_team_b_goals']).astype(int)
    m['opp_goals'] = np.where(is_a, m['pred_team_b_goals'], m['pred_team_a_goals']).astype(int)
    sub = sample_submission[['Id']].merge(m[['Id', 'team_goals', 'opp_goals']], on='Id', how='left', validate='one_to_one')
    assert sub['team_goals'].notna().all() and sub['opp_goals'].notna().all()
    assert sub['Id'].tolist() == sample_submission['Id'].tolist()
    return sub[['Id', 'team_goals', 'opp_goals']]

def check_pair_consistency(submission_df, test_df):
    tmp = test_df[['Id', 'match_id', 'team', 'opponent']].merge(submission_df, on='Id', how='left', validate='one_to_one')
    for mid, g in tmp.groupby('match_id'):
        if len(g) != 2:
            return False
        a, b = g.iloc[0], g.iloc[1]
        if int(a['team_goals']) != int(b['opp_goals']):
            return False
        if int(a['opp_goals']) != int(b['team_goals']):
            return False
    return True

submission = match_predictions_to_submission(test_clean, test_pred_match_best)
submission_path = SUB_DIR / 'submission_exp09c_best_safe.csv'
submission.to_csv(submission_path, index=False)

log_check('Submission shape matches sample', submission.shape == sample_submission.shape, f'{submission.shape} vs {sample_submission.shape}')
log_check('Submission Id order matches sample', submission['Id'].tolist() == sample_submission['Id'].tolist())
log_check('Goals are non-negative integers', (submission[['team_goals','opp_goals']] >= 0).all().all())
log_check('Submission has no missing', not submission.isna().any().any())
log_check('Submission has no duplicate Id', not submission['Id'].duplicated().any())
log_check('Pair consistency', check_pair_consistency(submission, test_clean))
log_saved(submission_path)
display(submission.head())


# %% [markdown]
# # 19. Final Decision dan Runtime Summary
#
# Bagian terakhir menyimpan `final_decision.json`, tetapi menampilkan keputusan sebagai DataFrame agar output tidak berantakan oleh raw JSON panjang.
#
# Runtime setiap section juga disimpan supaya eksperimen berikutnya bisa memperkirakan bagian mana yang paling berat dan perlu dioptimasi.
# %%
final_decision_path = SUM_DIR / 'final_decision.json'
with open(final_decision_path, 'w', encoding='utf-8') as f:
    json.dump(final_decision, f, indent=2, ensure_ascii=False)

final_decision_df = pd.DataFrame([{'key': k, 'value': str(v)} for k, v in final_decision.items()])
display(final_decision_df)
log_saved(final_decision_path)

selected_config_rows = []
for head, cfg in selected_head_config.items():
    selected_config_rows.append({
        'selected_variant': selected_variant,
        'head': head,
        'n_features': len(cfg['feature_cols']),
        'n_cat_features': len(cfg['cat_features']),
        'feature_set': 'EXP09A_REPAIR' if len(cfg['feature_cols']) > len(exp05a_feature_cols) else 'EXP05A',
    })
selected_config_df = pd.DataFrame(selected_config_rows)
display(selected_config_df)

runtime_rows = [{'section': k, 'runtime_sec': float(v)} for k, v in SECTION_TIMES.items()]
runtime_rows.append({'section': 'total_elapsed_until_runtime_summary', 'runtime_sec': float(time.time() - NOTEBOOK_START_TIME)})
runtime_summary_df = pd.DataFrame(runtime_rows)
runtime_summary_df.to_csv(SUM_DIR / 'notebook_runtime_summary.csv', index=False)
display(runtime_summary_df)
log_saved(SUM_DIR / 'notebook_runtime_summary.csv')
