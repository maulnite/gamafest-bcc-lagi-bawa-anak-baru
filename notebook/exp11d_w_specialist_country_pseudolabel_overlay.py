# %% [markdown]
# # 00. EXP11D — Ringkasan Eksperimen
#
# Bagian ini menjelaskan tujuan EXP11C. Eksperimen ini melanjutkan EXP11A sebagai best terbaru
# dan hanya mengubah risk function MBR decoder. Ordinal PMF, EXP09D tournament gate,
# dan policy M-only dari EXP11A tetap dipreserve; yang diuji adalah temperature, smoothing, ordinal-Poisson blend, dan mean lift untuk PMF ordinal.

# %% [markdown]
# # 01. Setup, Seed, Path, dan Guardrail
#
# Bagian ini menyiapkan library, seed, path input/output, dan konfigurasi eksperimen. Guardrail
# dipakai supaya eksperimen tetap legal, tidak memakai GT/external data, tidak memakai
# XGBoost/TabPFN/tqdm, dan output disimpan ke folder EXP11D.

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
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'exp11d_w_specialist_country_pseudolabel_overlay'
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
USE_GT_AUDIT = False
USE_SUBMISSION_AS_FEATURE = False
USE_XGBOOST = False
USE_PROGRESS_BAR = False
MODEL_VERBOSE = False
LGB_EARLY_STOP_VERBOSE = False
RUN_DATA_DRIVEN_GATE = False  # optional EXP09D V4 remains disabled by default to avoid validation overfit
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
log_info('EXP11C source of truth: EXP05A-LITE-FIX-V2 / EXP09D / EXP11A. Only PMF calibration before ordinal MBR is changed.')
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
# Bagian ini membaca train, test, sample submission, dan metadata. Output yang diharapkan adalah shape
# data, range tanggal, serta pengecekan awal agar format input sesuai pipeline EXP05A.

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
# Bagian ini mempertahankan helper utama dari EXP05A, termasuk evaluator AW-MAE, cleaning row-level,
# fungsi score/outcome, dan utilitas validasi. Bagian ini tidak diubah besar-besaran agar behavior
# baseline tetap preserve.

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
# Dataset berisi dua row untuk satu match. Bagian ini mengubah data row-level menjadi match-level agar
# prediksi team_a dan team_b konsisten, lalu nanti bisa dikembalikan ke format submission.

# %%
train_clean = clean_row_level(train_raw)
test_clean = clean_row_level(test_raw)
train_match_base = build_match_level(train_clean, is_train=True)
test_match_base = build_match_level(test_clean, is_train=False)
print('[INFO] train_match_base:', train_match_base.shape)
print('[INFO] test_match_base :', test_match_base.shape)
display(train_match_base.head(3))

# %% [markdown]
# # 05. Static dan History Features dari EXP05A
#
# Bagian ini membangun fitur utama EXP05A, termasuk static features dan history-full legal features.
# Feature set ini menjadi baseline untuk goal head, tail head, dan baseline outcome head.

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
# # 06. EXP09A Feature Repair dan EXP09D Tournament Gate
#
# Bagian ini menambahkan fungsi feature repair dari EXP09A seperti GDP imputation, tournament
# structure, cold-start flags, women temporal trend, dan confederation interaction. Pada EXP09D,
# fitur ini hanya memengaruhi outcome repair model.

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
# # 07. Feature Set Split
#
# Bagian ini memisahkan EXP05A feature set dan EXP05A+EXP09A feature set. Pemisahan ini penting karena
# goal/tail tetap memakai fitur EXP05A, sedangkan repair outcome memakai fitur tambahan.

# %%
# Build EXP09A feature set AFTER patch. Only the outcome head uses this set in V2.
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
# # 08. EDA Tipis, Feature Patch Summary, dan Target Builder
#
# Bagian ini menyimpan audit fitur repair serta membangun target supervised seperti goal_a, goal_b,
# outcome, dan tail labels. Outputnya berupa summary feature sebelum/sesudah patch dan dataframe
# target untuk training/validation.

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
# # 09. Model Training Helpers dan Anchor Decode Fix
#
# Bagian ini berisi helper training CatBoost/LightGBM, fungsi prediksi raw, decoder EXP05A/EXP09C,
# serta fix langsung untuk error lama KeyError gender. valid_decode_df dibuat dari valid_feature_full
# dan hanya merge actual goals.

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

if optuna is None:
    raise ImportError('optuna belum terinstall. Install dulu: pip install optuna')

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
    goal_a_anchor_col = 'decode_goal_a_anchor' if 'decode_goal_a_anchor' in raw_df.columns else 'pred_goal_a_cont'
    goal_b_anchor_col = 'decode_goal_b_anchor' if 'decode_goal_b_anchor' in raw_df.columns else 'pred_goal_b_cont'
    pred_a = raw_df[goal_a_anchor_col].to_numpy(float)[:,None]; pred_b = raw_df[goal_b_anchor_col].to_numpy(float)[:,None]
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
    goal_a_anchor_col = 'decode_goal_a_anchor' if 'decode_goal_a_anchor' in merged.columns else 'pred_goal_a_cont'
    goal_b_anchor_col = 'decode_goal_b_anchor' if 'decode_goal_b_anchor' in merged.columns else 'pred_goal_b_cont'
    pred_a = merged[goal_a_anchor_col].to_numpy(float)[:,None]; pred_b = merged[goal_b_anchor_col].to_numpy(float)[:,None]
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
# # 10. Train EXP09D Base dan Repair Outcome Sources
#
# Bagian ini melatih sumber model seperti EXP09D. Goal dan tail tetap baseline EXP05A,
# sedangkan outcome punya dua sumber: baseline EXP05A dan repair EXP05A+EXP09A.

# %%
log_section('Train baseline and repair outcome sources')

# Baseline models: goal, outcome, and tail all use EXP05A feature set.
# Repair models: goal/tail still follow EXP05A in the V2-style config, while outcome uses EXP05A + EXP09A features.
# For gating, raw_base is the anchor because goal and tail must remain baseline.
BASE_HEAD_CONFIG = HEAD_CONFIGS['V0_baseline']
REPAIR_HEAD_CONFIG = HEAD_CONFIGS['V2_outcome_only']

base_models = train_final_gender_models_by_head_config(
    train_sup_full,
    valid_sup_full,
    final_selected,
    BASE_HEAD_CONFIG,
    1200,
)
repair_models = train_final_gender_models_by_head_config(
    train_sup_full,
    valid_sup_full,
    final_selected,
    REPAIR_HEAD_CONFIG,
    1200,
)

head_feature_config_df = pd.DataFrame([
    {'head': 'goal_a', 'source_for_gate': 'baseline', 'feature_set': 'EXP05A', 'n_features': len(BASE_HEAD_CONFIG['goal']['feature_cols']), 'n_cat_features': len(BASE_HEAD_CONFIG['goal']['cat_features'])},
    {'head': 'goal_b', 'source_for_gate': 'baseline', 'feature_set': 'EXP05A', 'n_features': len(BASE_HEAD_CONFIG['goal']['feature_cols']), 'n_cat_features': len(BASE_HEAD_CONFIG['goal']['cat_features'])},
    {'head': 'tail', 'source_for_gate': 'baseline', 'feature_set': 'EXP05A', 'n_features': len(BASE_HEAD_CONFIG['tail']['feature_cols']), 'n_cat_features': len(BASE_HEAD_CONFIG['tail']['cat_features'])},
    {'head': 'outcome_base', 'source_for_gate': 'baseline', 'feature_set': 'EXP05A', 'n_features': len(BASE_HEAD_CONFIG['outcome']['feature_cols']), 'n_cat_features': len(BASE_HEAD_CONFIG['outcome']['cat_features'])},
    {'head': 'outcome_repair', 'source_for_gate': 'repair', 'feature_set': 'EXP05A_PLUS_EXP09A', 'n_features': len(REPAIR_HEAD_CONFIG['outcome']['feature_cols']), 'n_cat_features': len(REPAIR_HEAD_CONFIG['outcome']['cat_features'])},
])
head_feature_config_df.to_csv(SUM_DIR / 'head_feature_config.csv', index=False)
display(head_feature_config_df)
log_saved(SUM_DIR / 'head_feature_config.csv')

# %% [markdown]
# # 11. Build Raw Prediction EXP09D Gate
#
# Bagian ini menghasilkan raw prediction validation. raw_base menjadi anchor untuk goal/tail,
# sedangkan raw_repair hanya dipakai mengambil outcome probability repair sebelum tournament gate.

# %%
log_section('Build raw prediction for tournament gate')

raw_base_valid = predict_gender_split_raw_outputs_with_config(valid_decode_df, base_models)
raw_repair_valid = predict_gender_split_raw_outputs_with_config(valid_decode_df, repair_models)

prob_cols = ['pred_outcome_proba_0', 'pred_outcome_proba_1', 'pred_outcome_proba_2']
assert raw_base_valid['match_id'].is_unique, 'raw_base_valid match_id is not unique'
assert raw_repair_valid['match_id'].is_unique, 'raw_repair_valid match_id is not unique'
assert set(raw_base_valid['match_id']) == set(raw_repair_valid['match_id']), 'base/repair raw predictions have different match ids'

raw_base_valid.to_csv(PRED_DIR / 'valid_raw_outcome_base.csv', index=False)
raw_repair_valid.to_csv(PRED_DIR / 'valid_raw_outcome_repair.csv', index=False)
log_saved(PRED_DIR / 'valid_raw_outcome_base.csv')
log_saved(PRED_DIR / 'valid_raw_outcome_repair.csv')

base_vs_repair_proba_delta = pd.DataFrame({
    'metric': ['mean_abs_delta_p0', 'mean_abs_delta_p1', 'mean_abs_delta_p2'],
    'value': [
        float(np.mean(np.abs(raw_base_valid['pred_outcome_proba_0'].values - raw_repair_valid['pred_outcome_proba_0'].values))),
        float(np.mean(np.abs(raw_base_valid['pred_outcome_proba_1'].values - raw_repair_valid['pred_outcome_proba_1'].values))),
        float(np.mean(np.abs(raw_base_valid['pred_outcome_proba_2'].values - raw_repair_valid['pred_outcome_proba_2'].values))),
    ],
})
display(base_vs_repair_proba_delta)


# %% [markdown]
# # 12. EXP09D Tournament-Gated Outcome Probability
#
# Bagian ini mempertahankan tournament gate EXP09D. Gate ini menentukan seberapa besar
# probability outcome repair dipakai berdasarkan tournament_structure. EXP11C memakai gate terpilih
# sebagai baseline sebelum menguji anchor skor.

# %%
def get_gate_alpha(tournament_structure: str, gate_config: dict, default_alpha: float = 0.0) -> float:
    ts = str(tournament_structure)
    return float(gate_config.get(ts, default_alpha))


def apply_tournament_gated_outcome(raw_base, raw_repair, gate_config, default_alpha=0.0):
    """
    Apply tournament-gated outcome probability only.
    Goal and tail signals are preserved from raw_base.
    """
    prob_cols = ['pred_outcome_proba_0', 'pred_outcome_proba_1', 'pred_outcome_proba_2']
    if 'tournament_structure' not in raw_base.columns:
        raise KeyError('raw_base must contain tournament_structure for EXP09D gate')

    base_cols = ['match_id', 'tournament_structure'] + prob_cols
    repair_cols = ['match_id'] + prob_cols
    base = raw_base[base_cols].copy()
    repair = raw_repair[repair_cols].copy()

    merged = base.merge(
        repair,
        on='match_id',
        how='inner',
        suffixes=('_base', '_repair'),
        validate='one_to_one',
    )
    assert len(merged) == len(raw_base), 'Gate merge changed row count'

    alpha = merged['tournament_structure'].map(
        lambda x: get_gate_alpha(x, gate_config, default_alpha)
    ).astype(float).to_numpy()
    alpha = np.clip(alpha, 0.0, 1.0)

    p_base = merged[[c + '_base' for c in prob_cols]].to_numpy(float)
    p_repair = merged[[c + '_repair' for c in prob_cols]].to_numpy(float)
    p = (1 - alpha[:, None]) * p_base + alpha[:, None] * p_repair
    p = np.clip(p, 1e-9, 1.0)
    p = p / p.sum(axis=1, keepdims=True)

    out = raw_base.copy().drop(columns=prob_cols, errors='ignore')
    out = out.merge(
        pd.DataFrame({
            'match_id': merged['match_id'].values,
            'gate_alpha': alpha,
            'pred_outcome_proba_0': p[:, 0],
            'pred_outcome_proba_1': p[:, 1],
            'pred_outcome_proba_2': p[:, 2],
        }),
        on='match_id',
        how='left',
        validate='one_to_one',
    )
    assert out[prob_cols].notna().all().all(), 'Gated outcome probability has NaN'
    assert out['gate_alpha'].notna().all(), 'Gate alpha has NaN'
    return out


GATE_CONFIGS = {
    'V0_baseline_outcome': {
        'default_alpha': 0.0,
        'gate_map': {},
        'description': 'All tournament structures use baseline EXP05A outcome probability.',
    },
    'V1_full_repair': {
        'default_alpha': 1.0,
        'gate_map': {},
        'description': 'All tournament structures use repair EXP09C outcome probability.',
    },
    'V2_hard_helpful_gate': {
        'default_alpha': 0.0,
        'gate_map': {
            'friendly': 1.0,
            'continental_group': 1.0,
            'regional_games': 1.0,
            'world_cup_group': 1.0,
        },
        'description': 'Repair is enabled only for tournament structures that looked helpful in EXP09C validation.',
    },
    'V3_conservative_soft_gate': {
        'default_alpha': 0.5,
        'gate_map': {
            'friendly': 1.0,
            'continental_group': 1.0,
            'regional_games': 1.0,
            'world_cup_group': 1.0,
            'qualifier': 0.25,
            'other_competitive': 0.25,
            'continental_knockout': 0.5,
            'world_cup_knockout': 0.5,
            'nations_league_group': 0.5,
            'nations_league_knockout': 0.5,
            'unknown': 0.5,
        },
        'description': 'A conservative soft gate: helpful structures get full repair, risky structures get weak repair.',
    },
}

with open(SUM_DIR / 'gate_variant_config.json', 'w', encoding='utf-8') as f:
    json.dump(GATE_CONFIGS, f, indent=2, ensure_ascii=False)

map_rows = []
known_structures = sorted(set(raw_base_valid.get('tournament_structure', pd.Series(dtype=str)).astype(str).unique().tolist()))
for variant_name, cfg in GATE_CONFIGS.items():
    covered = sorted(set(known_structures) | set(cfg['gate_map'].keys()))
    for ts in covered:
        alpha_value = get_gate_alpha(ts, cfg['gate_map'], cfg['default_alpha'])
        map_rows.append({
            'variant_name': variant_name,
            'tournament_structure': ts,
            'alpha': alpha_value,
            'source': 'repair' if alpha_value == 1.0 else ('baseline' if alpha_value == 0.0 else 'blend'),
            'default_alpha': cfg['default_alpha'],
            'description': cfg['description'],
        })

tournament_gate_map_df = pd.DataFrame(map_rows)
tournament_gate_map_df.to_csv(SUM_DIR / 'tournament_gate_map.csv', index=False)
display(tournament_gate_map_df.head(80))
log_saved(SUM_DIR / 'gate_variant_config.json')
log_saved(SUM_DIR / 'tournament_gate_map.csv')


# %% [markdown]
# # 13. EXP09D Gate Selection sebagai Baseline EXP11C
#
# Bagian ini mereplikasi selection gate EXP09D pada validation. Setiap gate variant mendapat decoder
# yang ditune sendiri supaya V0, V1, V2, dan V3 dibandingkan secara fair. Hasil gate terpilih menjadi
# raw_gate baseline yang akan dipakai oleh eksperimen PMF calibration.

# %%
def compute_domain_error_per_variant(detail_df, group_col):
    if group_col not in detail_df.columns:
        return pd.DataFrame()
    rows = []
    for (variant, key), g in detail_df.groupby(['variant_name', group_col], dropna=False):
        rows.append({
            'variant_name': variant,
            group_col: key,
            'n': len(g),
            'awmae': awmae_score(g['actual_team_a_goals'], g['actual_team_b_goals'], g['pred_team_a_goals'], g['pred_team_b_goals'], g['tournament']),
            'base_mae': float(g['base_mae'].mean()),
            'exact_rate': float(g['exact_hit'].mean()),
            'outcome_rate': float(g['outcome_hit'].mean()),
            'gd_rate': float(g['gd_hit'].mean()),
        })
    return pd.DataFrame(rows).sort_values(['variant_name', 'awmae']).reset_index(drop=True)


def evaluate_gate_prediction(variant_name, raw_base, raw_repair, gate_cfg, baseline_detail=None):
    raw_gate = apply_tournament_gated_outcome(
        raw_base,
        raw_repair,
        gate_cfg['gate_map'],
        default_alpha=gate_cfg['default_alpha'],
    )
    tail_prob = raw_base[['match_id'] + TAIL_COLS].copy()
    decoder_grid, best_decoder, pred = tune_tail_aware_decoder_from_raw_outputs(
        raw_gate,
        tail_prob,
        TAIL_GRID,
        scoreline_prior_lookup,
        top_n=3,
    )
    pred_match = raw_gate.merge(pred, on='match_id', how='left', validate='one_to_one')
    detail = prediction_detail_table(pred_match, variant_name)
    summary = summarize_prediction_detail(detail, variant_name)
    summary['mean_gate_alpha'] = float(raw_gate['gate_alpha'].mean())
    for value, label in [(0.0, 'n_alpha_0'), (0.25, 'n_alpha_025'), (0.5, 'n_alpha_050'), (0.75, 'n_alpha_075'), (1.0, 'n_alpha_100')]:
        summary[label] = int(np.isclose(raw_gate['gate_alpha'].to_numpy(float), value).sum())

    if baseline_detail is not None:
        compare = detail[['match_id', 'pred_team_a_goals', 'pred_team_b_goals', 'official_loss', 'tournament_weight']].merge(
            baseline_detail[['match_id', 'pred_team_a_goals', 'pred_team_b_goals', 'official_loss']],
            on='match_id',
            how='left',
            suffixes=('', '_v0'),
            validate='one_to_one',
        )
        changed = (
            (compare['pred_team_a_goals'] != compare['pred_team_a_goals_v0'])
            | (compare['pred_team_b_goals'] != compare['pred_team_b_goals_v0'])
        )
        delta = compare['official_loss'] - compare['official_loss_v0']
        summary['n_changed_vs_v0'] = int(changed.sum())
        summary['changed_rate_vs_v0'] = float(changed.mean())
        summary['n_improved_changed'] = int(((delta < 0) & changed).sum())
        summary['n_worsened_changed'] = int(((delta > 0) & changed).sum())
        summary['n_neutral_changed'] = int(((delta == 0) & changed).sum())
        summary['weighted_loss_delta_vs_v0'] = float((delta * compare['tournament_weight']).sum() / compare['tournament_weight'].sum())
    else:
        summary['n_changed_vs_v0'] = 0
        summary['changed_rate_vs_v0'] = 0.0
        summary['n_improved_changed'] = 0
        summary['n_worsened_changed'] = 0
        summary['n_neutral_changed'] = 0
        summary['weighted_loss_delta_vs_v0'] = 0.0

    return raw_gate, pred_match, detail, summary, decoder_grid, best_decoder


log_section('EXP09D gate selection baseline for EXP11C')
GATE_RESULTS = {}
gate_metric_rows = []
v0_detail = None

for variant_name, cfg in GATE_CONFIGS.items():
    raw_gate, pred_match, detail, summary, decoder_grid, best_decoder = evaluate_gate_prediction(
        variant_name,
        raw_base_valid,
        raw_repair_valid,
        cfg,
        baseline_detail=v0_detail,
    )
    if variant_name == 'V0_baseline_outcome':
        v0_detail = detail.copy()
        raw_gate, pred_match, detail, summary, decoder_grid, best_decoder = evaluate_gate_prediction(
            variant_name,
            raw_base_valid,
            raw_repair_valid,
            cfg,
            baseline_detail=v0_detail,
        )

    pred_path = PRED_DIR / f'exp09d_valid_pred_match_{variant_name}.csv'
    decoder_grid_path = SUM_DIR / f'exp09d_decoder_grid_{variant_name}.csv'
    pred_match.to_csv(pred_path, index=False)
    decoder_grid.to_csv(decoder_grid_path, index=False)
    log_saved(pred_path)
    log_saved(decoder_grid_path)

    GATE_RESULTS[variant_name] = {
        'raw': raw_gate,
        'pred_match': pred_match,
        'detail': detail,
        'summary': summary,
        'config': cfg,
        'decoder_grid': decoder_grid,
        'best_decoder': best_decoder,
    }
    gate_metric_rows.append(summary)

gate_variant_metrics_df = pd.DataFrame(gate_metric_rows)
v0_gate = gate_variant_metrics_df[gate_variant_metrics_df['variant_name'].eq('V0_baseline_outcome')].iloc[0]
v1_gate = gate_variant_metrics_df[gate_variant_metrics_df['variant_name'].eq('V1_full_repair')].iloc[0]
gate_variant_metrics_df['awmae_gain_vs_v0'] = float(v0_gate['awmae']) - gate_variant_metrics_df['awmae']
gate_variant_metrics_df = gate_variant_metrics_df.sort_values('awmae').reset_index(drop=True)
gate_variant_metrics_df.to_csv(SUM_DIR / 'exp09d_gate_variant_metrics.csv', index=False)
display(gate_variant_metrics_df)
log_saved(SUM_DIR / 'exp09d_gate_variant_metrics.csv')

# Safety-first EXP09D gate selection. This mirrors the tournament-gated repair spirit.
def select_exp09d_gate(metrics_df):
    metrics = metrics_df.copy()
    v0 = metrics[metrics['variant_name'].eq('V0_baseline_outcome')].iloc[0]
    v1 = metrics[metrics['variant_name'].eq('V1_full_repair')].iloc[0]
    gate_candidates = metrics[metrics['variant_name'].isin(['V2_hard_helpful_gate', 'V3_conservative_soft_gate'])].copy()
    safe_rows = []
    for _, row in gate_candidates.iterrows():
        safe = (
            (float(row['awmae']) < float(v0['awmae']))
            and (float(row['exact_rate']) >= float(v0['exact_rate']) - 0.010)
            and (float(row['gd_rate']) >= float(v0['gd_rate']) - 0.015)
            and (float(row['top3_scoreline_share']) <= float(v0['top3_scoreline_share']) + 0.030)
            and (1.0 <= float(row['mean_pred_total']) <= 4.5)
        )
        if safe:
            safe_rows.append(row)
    if safe_rows:
        best = pd.DataFrame(safe_rows).sort_values('awmae').iloc[0]
        return str(best['variant_name']), 'A', 'Tournament gate improves AW-MAE and passes EXP09D safety checks.'

    v1_safe = (
        (float(v1['awmae']) < float(v0['awmae']))
        and (float(v1['exact_rate']) >= float(v0['exact_rate']) - 0.010)
        and (float(v1['gd_rate']) >= float(v0['gd_rate']) - 0.015)
        and (float(v1['top3_scoreline_share']) <= float(v0['top3_scoreline_share']) + 0.030)
        and (1.0 <= float(v1['mean_pred_total']) <= 4.5)
    )
    if v1_safe:
        return 'V1_full_repair', 'C', 'Full repair remains best and passes safety; using EXP09D full repair as baseline.'
    return 'V0_baseline_outcome', 'D', 'No repair gate passed safety; fallback to baseline outcome.'

selected_gate_name, exp09d_gate_decision_code, exp09d_gate_decision_text = select_exp09d_gate(gate_variant_metrics_df)
selected_gate_cfg = GATE_CONFIGS[selected_gate_name]
selected_exp09d_decoder = GATE_RESULTS[selected_gate_name]['best_decoder']
raw_gate_valid = GATE_RESULTS[selected_gate_name]['raw'].copy()
exp09d_detail = GATE_RESULTS[selected_gate_name]['detail'].copy()

selected_gate_df = pd.DataFrame([{
    'selected_gate': selected_gate_name,
    'decision_code': exp09d_gate_decision_code,
    'decision_text': exp09d_gate_decision_text,
    'selected_gate_awmae': float(gate_variant_metrics_df[gate_variant_metrics_df['variant_name'].eq(selected_gate_name)]['awmae'].iloc[0]),
    'selected_decoder': str(selected_exp09d_decoder),
}])
display(selected_gate_df)
selected_gate_df.to_csv(SUM_DIR / 'exp09d_selected_gate_for_exp11c.csv', index=False)
log_saved(SUM_DIR / 'exp09d_selected_gate_for_exp11c.csv')

# %% [markdown]
# %% [markdown]
# # 14. Ordinal Goal Target Builder
#
# Bagian ini membuat target ordinal `goal >= k` untuk team_a dan team_b. Target ini dipakai untuk
# melatih binary classifiers yang mempelajari distribusi peluang gol, bukan hanya satu angka prediksi.
# Output yang diharapkan adalah ringkasan positive rate tiap threshold.

# %%
MAX_ORDINAL_GOALS = 10
MAX_DECODE_GOALS = 9
ORDINAL_N_ESTIMATORS = 650


def build_ordinal_goal_targets(y_goal_a, y_goal_b, max_goal=10):
    out = pd.DataFrame()
    y_a = pd.Series(y_goal_a).astype(int)
    y_b = pd.Series(y_goal_b).astype(int)
    for k in range(1, max_goal + 1):
        out[f'a_goal_ge_{k}'] = (y_a >= k).astype(int)
        out[f'b_goal_ge_{k}'] = (y_b >= k).astype(int)
    return out

ordinal_train_targets = build_ordinal_goal_targets(train_sup_full['y_goal_a'], train_sup_full['y_goal_b'], MAX_ORDINAL_GOALS)
ordinal_valid_targets = build_ordinal_goal_targets(valid_sup_full['y_goal_a'], valid_sup_full['y_goal_b'], MAX_ORDINAL_GOALS)
ordinal_target_summary_rows = []
for side in ['a', 'b']:
    for k in range(1, MAX_ORDINAL_GOALS + 1):
        col = f'{side}_goal_ge_{k}'
        ordinal_target_summary_rows.append({
            'side': side,
            'threshold': k,
            'train_positive_rate': float(ordinal_train_targets[col].mean()),
            'valid_positive_rate': float(ordinal_valid_targets[col].mean()),
            'train_positive_count': int(ordinal_train_targets[col].sum()),
            'valid_positive_count': int(ordinal_valid_targets[col].sum()),
        })
ordinal_target_summary_df = pd.DataFrame(ordinal_target_summary_rows)
ordinal_target_summary_df.to_csv(SUM_DIR / 'ordinal_target_summary.csv', index=False)
display(ordinal_target_summary_df)
log_saved(SUM_DIR / 'ordinal_target_summary.csv')

# %% [markdown]
# # 15. Train Ordinal Goal Distribution Models
#
# Bagian ini melatih ordinal classifier per gender, per side, dan per threshold. Feature set default
# memakai EXP05A agar tidak langsung mengulang noise EXP09A all-head. Booster mengikuti pilihan EXP09D
# per gender supaya pipeline tetap preserve.

# %%
class ConstantBinaryModel:
    def __init__(self, prob=0.0):
        self.prob = float(np.clip(prob, 1e-6, 1 - 1e-6))
        self.classes_ = np.array([0, 1])
    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([np.full(n, 1.0 - self.prob), np.full(n, self.prob)])


def fit_binary_model_safe(Xtr, ytr, Xva, yva, booster, params, cats):
    ytr = pd.Series(ytr).astype(int)
    if ytr.nunique() < 2:
        return ConstantBinaryModel(float(ytr.mean()))
    return fit_tail_model(Xtr, ytr, Xva, pd.Series(yva).astype(int), booster, params, cats)


def train_ordinal_models(train_df, valid_df, final_selected, feature_cols, cat_features, max_goal=10, n_estimators=650):
    models = {}
    config_rows = []
    threshold_metric_rows = []
    for gender, cfg in final_selected.items():
        booster, regime = cfg['booster'], cfg['regime']
        tr_all = filter_regime_train(train_df, regime)
        tr = tr_all[tr_all['gender'].astype(str).eq(str(gender))].copy()
        va = valid_df[valid_df['gender'].astype(str).eq(str(gender))].copy()
        if len(tr) == 0 or len(va) == 0:
            log_warn(f'Skip ordinal gender={gender} because train/valid is empty')
            continue
        cols = ['match_id', 'gender', 'date', 'y_goal_a', 'y_goal_b'] + list(feature_cols)
        tr = select_unique_columns(tr, cols)
        va = select_unique_columns(va, cols)
        Xtr = prepare_model_frame(tr, feature_cols, cat_features)
        Xva = prepare_model_frame(va, feature_cols, cat_features)
        params = make_default_params(booster, 'tail', SEED, n_estimators)
        models[str(gender)] = {
            'booster': booster,
            'regime': regime,
            'feature_cols': list(feature_cols),
            'cat_features': list(cat_features),
            'side_models': {'a': {}, 'b': {}},
        }
        log_info(f'Train ordinal models | gender={gender} | booster={booster} | regime={regime} | n_train={len(tr)} | n_valid={len(va)}')
        for side, target_col in [('a', 'y_goal_a'), ('b', 'y_goal_b')]:
            ytr_goal = pd.to_numeric(tr[target_col], errors='coerce').fillna(0).astype(int)
            yva_goal = pd.to_numeric(va[target_col], errors='coerce').fillna(0).astype(int)
            for k in range(1, max_goal + 1):
                ytr_bin = (ytr_goal >= k).astype(int)
                yva_bin = (yva_goal >= k).astype(int)
                model = fit_binary_model_safe(Xtr, ytr_bin, Xva, yva_bin, booster, params, cat_features)
                models[str(gender)]['side_models'][side][k] = model
                pred = predict_pos(model, Xva)
                threshold_metric_rows.append({
                    'gender': str(gender),
                    'side': side,
                    'threshold': k,
                    'booster': booster,
                    'regime': regime,
                    'valid_positive_rate': float(yva_bin.mean()),
                    'pred_positive_mean': float(np.mean(pred)),
                    'brier': float(np.mean((pred - yva_bin.to_numpy(float)) ** 2)),
                })
        config_rows.append({
            'gender': str(gender),
            'booster': booster,
            'regime': regime,
            'n_features': len(feature_cols),
            'n_cat_features': len(cat_features),
            'max_ordinal_goals': max_goal,
            'n_estimators': n_estimators,
        })
    return models, pd.DataFrame(config_rows), pd.DataFrame(threshold_metric_rows)

log_section('Train ordinal goal distribution models')
ordinal_models, ordinal_model_config_df, ordinal_threshold_metrics_df = train_ordinal_models(
    train_sup_full,
    valid_sup_full,
    final_selected,
    exp05a_feature_cols,
    exp05a_cat_features,
    max_goal=MAX_ORDINAL_GOALS,
    n_estimators=ORDINAL_N_ESTIMATORS,
)
ordinal_model_config_df.to_csv(SUM_DIR / 'ordinal_model_config.csv', index=False)
ordinal_threshold_metrics_df.to_csv(SUM_DIR / 'ordinal_threshold_metrics.csv', index=False)
display(ordinal_model_config_df)
display(ordinal_threshold_metrics_df.head(40))
log_saved(SUM_DIR / 'ordinal_model_config.csv')
log_saved(SUM_DIR / 'ordinal_threshold_metrics.csv')

# %% [markdown]
# # 16. Build Goal PMF from Ordinal Probabilities
#
# Bagian ini mengubah probability `P(goal >= k)` menjadi PMF `P(goal = k)`. Probability dibersihkan
# agar monotonic, non-negative, dan sum-to-one. Poisson PMF dari continuous goal EXP09D juga dibuat
# untuk variant blend.

# %%
def enforce_monotonic_ge_probs(p_ge):
    p = np.asarray(p_ge, dtype=float)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    for k in range(1, p.shape[1]):
        p[:, k] = np.minimum(p[:, k], p[:, k - 1])
    return p


def ge_probs_to_pmf(p_ge):
    p_ge = enforce_monotonic_ge_probs(p_ge)
    n, kmax = p_ge.shape
    pmf = np.zeros((n, kmax + 1), dtype=float)
    pmf[:, 0] = 1.0 - p_ge[:, 0]
    for k in range(1, kmax):
        pmf[:, k] = p_ge[:, k - 1] - p_ge[:, k]
    pmf[:, kmax] = p_ge[:, kmax - 1]
    pmf = np.clip(pmf, 1e-12, 1.0)
    pmf = pmf / pmf.sum(axis=1, keepdims=True)
    return pmf


def poisson_pmf_from_lambda(lam, max_goal=10):
    lam = np.asarray(lam, dtype=float)
    lam = np.clip(lam, 1e-6, 20)
    pmf = np.zeros((len(lam), max_goal + 1), dtype=float)
    pmf[:, 0] = np.exp(-lam)
    for k in range(1, max_goal + 1):
        pmf[:, k] = pmf[:, k - 1] * lam / k
    pmf = np.clip(pmf, 1e-12, 1.0)
    pmf = pmf / pmf.sum(axis=1, keepdims=True)
    return pmf


def blend_pmfs(pmf_ordinal, pmf_poisson, blend_ordinal=0.75):
    p = float(blend_ordinal) * pmf_ordinal + (1 - float(blend_ordinal)) * pmf_poisson
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    return p


def predict_ordinal_ge_probs(df, ordinal_models, side='a', max_goal=10):
    p = np.full((len(df), max_goal), 1e-6, dtype=float)
    df_work = df.reset_index(drop=False).rename(columns={'index': '_orig_index'})
    for gender, bundle in ordinal_models.items():
        mask = df_work['gender'].astype(str).eq(str(gender))
        if not mask.any():
            continue
        sub = df_work.loc[mask].copy()
        X = prepare_model_frame(sub, bundle['feature_cols'], bundle['cat_features'])
        for k in range(1, max_goal + 1):
            model = bundle['side_models'][side][k]
            p[sub['_orig_index'].to_numpy(int), k - 1] = predict_pos(model, X)
    return enforce_monotonic_ge_probs(p)

log_section('Build ordinal and Poisson PMFs for validation')
# PMF rows must follow raw_gate_valid order because MBR consumes raw_gate_valid row-by-row.
valid_feature_for_pmf = raw_gate_valid[['match_id']].merge(valid_decode_df, on='match_id', how='left', validate='one_to_one')
suffix_cols = [c for c in valid_feature_for_pmf.columns if c.endswith('_x') or c.endswith('_y')]
assert len(suffix_cols) == 0, f'valid_feature_for_pmf has bad suffix columns: {suffix_cols}'
valid_ge_a = predict_ordinal_ge_probs(valid_feature_for_pmf, ordinal_models, side='a', max_goal=MAX_ORDINAL_GOALS)
valid_ge_b = predict_ordinal_ge_probs(valid_feature_for_pmf, ordinal_models, side='b', max_goal=MAX_ORDINAL_GOALS)
valid_pmf_ord_a = ge_probs_to_pmf(valid_ge_a)
valid_pmf_ord_b = ge_probs_to_pmf(valid_ge_b)
valid_pmf_pois_a = poisson_pmf_from_lambda(raw_gate_valid['pred_goal_a_cont'], max_goal=MAX_ORDINAL_GOALS)
valid_pmf_pois_b = poisson_pmf_from_lambda(raw_gate_valid['pred_goal_b_cont'], max_goal=MAX_ORDINAL_GOALS)

pmf_quality_summary_df = pd.DataFrame([
    {'pmf_type': 'ordinal', 'side': 'a', 'mean_goal_from_pmf': float((valid_pmf_ord_a * np.arange(MAX_ORDINAL_GOALS + 1)).sum(axis=1).mean()), 'actual_mean_goal': float(valid_sup_full['y_goal_a'].mean()), 'mean_sum': float(valid_pmf_ord_a.sum(axis=1).mean())},
    {'pmf_type': 'ordinal', 'side': 'b', 'mean_goal_from_pmf': float((valid_pmf_ord_b * np.arange(MAX_ORDINAL_GOALS + 1)).sum(axis=1).mean()), 'actual_mean_goal': float(valid_sup_full['y_goal_b'].mean()), 'mean_sum': float(valid_pmf_ord_b.sum(axis=1).mean())},
    {'pmf_type': 'poisson_from_cont', 'side': 'a', 'mean_goal_from_pmf': float((valid_pmf_pois_a * np.arange(MAX_ORDINAL_GOALS + 1)).sum(axis=1).mean()), 'actual_mean_goal': float(valid_sup_full['y_goal_a'].mean()), 'mean_sum': float(valid_pmf_pois_a.sum(axis=1).mean())},
    {'pmf_type': 'poisson_from_cont', 'side': 'b', 'mean_goal_from_pmf': float((valid_pmf_pois_b * np.arange(MAX_ORDINAL_GOALS + 1)).sum(axis=1).mean()), 'actual_mean_goal': float(valid_sup_full['y_goal_b'].mean()), 'mean_sum': float(valid_pmf_pois_b.sum(axis=1).mean())},
])
pmf_quality_summary_df.to_csv(SUM_DIR / 'pmf_quality_summary.csv', index=False)
display(pmf_quality_summary_df)
log_saved(SUM_DIR / 'pmf_quality_summary.csv')



# %% [markdown]
# # 17. PMF Calibration Functions
#
# Bagian ini membuat fungsi kalibrasi PMF: temperature, smoothing, ordinal-Poisson blend,
# dan mean lift. Fungsi ini hanya mengubah distribusi probabilitas gol sebelum MBR decoder,
# bukan mengubah model training atau fitur utama.

# %%
def build_loss_matrix(max_pmf_goal=10, max_decode_goals=9):
    true_scores = [(a, b) for a in range(max_pmf_goal + 1) for b in range(max_pmf_goal + 1)]
    candidates = [(a, b) for a in range(max_decode_goals + 1) for b in range(max_decode_goals + 1)]
    loss = np.zeros((len(true_scores), len(candidates)), dtype=float)
    for i, (ta, tb) in enumerate(true_scores):
        for j, (pa, pb) in enumerate(candidates):
            loss[i, j] = official_match_loss(ta, tb, pa, pb)
    return true_scores, candidates, loss

TRUE_SCORES_MBR, CANDIDATES_MBR, LOSS_MATRIX_MBR = build_loss_matrix(MAX_ORDINAL_GOALS, MAX_DECODE_GOALS)


def mbr_decode_from_pmfs_basic(raw_df, pmf_a, pmf_b, max_decode_goals=9):
    """EXP11A-style MBR decoder: expected AW-MAE from PMF only, no penalty."""
    candidates = [(a, b) for a in range(max_decode_goals + 1) for b in range(max_decode_goals + 1)]
    if max_decode_goals == MAX_DECODE_GOALS and pmf_a.shape[1] == MAX_ORDINAL_GOALS + 1:
        loss_matrix = LOSS_MATRIX_MBR
    else:
        _, _, loss_matrix = build_loss_matrix(pmf_a.shape[1] - 1, max_decode_goals)

    joint_flat = (pmf_a[:, :, None] * pmf_b[:, None, :]).reshape(len(raw_df), -1)
    joint_flat = joint_flat / np.maximum(joint_flat.sum(axis=1, keepdims=True), 1e-12)
    risks = joint_flat @ loss_matrix

    cand_a = np.array([c[0] for c in candidates], dtype=float)
    cand_b = np.array([c[1] for c in candidates], dtype=float)
    idx = np.argmin(risks, axis=1)
    return pd.DataFrame({
        'match_id': raw_df['match_id'].values,
        'pred_team_a_goals': cand_a[idx].astype(int),
        'pred_team_b_goals': cand_b[idx].astype(int),
        'mbr_risk': risks[np.arange(len(raw_df)), idx].astype(float),
    })


def apply_pmf_temperature(pmf, temperature=1.0):
    p = np.asarray(pmf, dtype=float)
    p = np.clip(p, 1e-12, 1.0)
    temp = float(temperature)
    if abs(temp - 1.0) < 1e-12:
        return p / p.sum(axis=1, keepdims=True)
    logits = np.log(p) / temp
    logits = logits - logits.max(axis=1, keepdims=True)
    out = np.exp(logits)
    out = out / out.sum(axis=1, keepdims=True)
    return out


def apply_pmf_smoothing(pmf, epsilon=0.0):
    p = np.asarray(pmf, dtype=float)
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    eps = float(epsilon)
    if eps <= 0:
        return p
    eps = float(np.clip(eps, 0.0, 0.50))
    uniform = np.ones_like(p) / p.shape[1]
    out = (1.0 - eps) * p + eps * uniform
    out = np.clip(out, 1e-12, 1.0)
    out = out / out.sum(axis=1, keepdims=True)
    return out


def blend_pmfs(pmf_ordinal, pmf_poisson, blend_ordinal=1.0):
    alpha = float(np.clip(float(blend_ordinal), 0.0, 1.0))
    p = alpha * np.asarray(pmf_ordinal, dtype=float) + (1.0 - alpha) * np.asarray(pmf_poisson, dtype=float)
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    return p


def apply_mean_lift(pmf, mean_lift=0.0):
    """Shift a small probability mass from lower buckets to the next higher bucket."""
    p = np.asarray(pmf, dtype=float)
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    lift = float(mean_lift)
    if lift <= 0:
        return p
    lift = float(np.clip(lift, 0.0, 0.20))
    out = p.copy()
    max_k = p.shape[1] - 1
    movable = out[:, :max_k] * lift
    out[:, :max_k] -= movable
    out[:, 1:] += movable
    out = np.clip(out, 1e-12, 1.0)
    out = out / out.sum(axis=1, keepdims=True)
    return out


def calibrate_goal_pmf(pmf_ordinal, pmf_poisson, config):
    p = blend_pmfs(pmf_ordinal, pmf_poisson, blend_ordinal=float(config.get('blend_ordinal', 1.0)))
    p = apply_pmf_temperature(p, temperature=float(config.get('temperature', 1.0)))
    p = apply_pmf_smoothing(p, epsilon=float(config.get('smoothing_epsilon', 0.0)))
    p = apply_mean_lift(p, mean_lift=float(config.get('mean_lift', 0.0)))
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    return p


def pmf_entropy(pmf):
    p = np.clip(np.asarray(pmf, dtype=float), 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis=1)


def summarize_pmf_quality(variant_name, pmf_a, pmf_b):
    grid = np.arange(pmf_a.shape[1], dtype=float)
    return {
        'variant_name': variant_name,
        'pmf_a_mean': float((pmf_a * grid).sum(axis=1).mean()),
        'pmf_b_mean': float((pmf_b * grid).sum(axis=1).mean()),
        'pmf_total_mean': float(((pmf_a * grid).sum(axis=1) + (pmf_b * grid).sum(axis=1)).mean()),
        'pmf_a_entropy_mean': float(pmf_entropy(pmf_a).mean()),
        'pmf_b_entropy_mean': float(pmf_entropy(pmf_b).mean()),
        'pmf_a_p0_mean': float(pmf_a[:, 0].mean()),
        'pmf_b_p0_mean': float(pmf_b[:, 0].mean()),
        'pmf_a_p3plus_mean': float(pmf_a[:, 3:].sum(axis=1).mean()),
        'pmf_b_p3plus_mean': float(pmf_b[:, 3:].sum(axis=1).mean()),
    }


def decode_exp11c_policy(raw_gate_df, exp09d_pred_df, pmf_a_cal, pmf_b_cal, variant_name):
    """M rows use calibrated ordinal MBR. W rows keep EXP09D original prediction."""
    raw = raw_gate_df.reset_index(drop=True).copy()
    is_m = raw['gender'].astype(str).eq('M').to_numpy()
    rows = []
    if is_m.any():
        idx_m = np.where(is_m)[0]
        m_pred = mbr_decode_from_pmfs_basic(
            raw.iloc[idx_m].copy(),
            pmf_a_cal[idx_m],
            pmf_b_cal[idx_m],
            max_decode_goals=MAX_DECODE_GOALS,
        )
        rows.append(m_pred)
    if (~is_m).any():
        w_ids = raw.loc[~is_m, 'match_id']
        w_pred = exp09d_pred_df.loc[
            exp09d_pred_df['match_id'].isin(w_ids),
            ['match_id', 'pred_team_a_goals', 'pred_team_b_goals']
        ].copy()
        rows.append(w_pred)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=['match_id', 'pred_team_a_goals', 'pred_team_b_goals'])
    assert out['match_id'].nunique() == raw['match_id'].nunique(), 'EXP11C policy lost or duplicated match_id.'
    out['variant_name'] = variant_name
    return out[['match_id', 'pred_team_a_goals', 'pred_team_b_goals', 'variant_name']]

# %% [markdown]
# # 18. PMF Calibration Config Grid
#
# Bagian ini mendefinisikan grid kecil PMF calibration. C0 adalah EXP11A original.
# Variant lain menguji temperature, smoothing, ordinal-Poisson blend, mean lift,
# dan beberapa combo kecil tanpa membuka grid besar.

# %%
PMF_CALIBRATION_CONFIGS = [
    {'variant_name': 'C0_exp11a_original', 'temperature': 1.00, 'smoothing_epsilon': 0.00, 'blend_ordinal': 1.00, 'mean_lift': 0.00},
    {'variant_name': 'C1_temp_085', 'temperature': 0.85, 'smoothing_epsilon': 0.00, 'blend_ordinal': 1.00, 'mean_lift': 0.00},
    {'variant_name': 'C1_temp_095', 'temperature': 0.95, 'smoothing_epsilon': 0.00, 'blend_ordinal': 1.00, 'mean_lift': 0.00},
    {'variant_name': 'C1_temp_105', 'temperature': 1.05, 'smoothing_epsilon': 0.00, 'blend_ordinal': 1.00, 'mean_lift': 0.00},
    {'variant_name': 'C1_temp_115', 'temperature': 1.15, 'smoothing_epsilon': 0.00, 'blend_ordinal': 1.00, 'mean_lift': 0.00},
    {'variant_name': 'C2_smooth_0005', 'temperature': 1.00, 'smoothing_epsilon': 0.005, 'blend_ordinal': 1.00, 'mean_lift': 0.00},
    {'variant_name': 'C2_smooth_0010', 'temperature': 1.00, 'smoothing_epsilon': 0.010, 'blend_ordinal': 1.00, 'mean_lift': 0.00},
    {'variant_name': 'C2_smooth_0020', 'temperature': 1.00, 'smoothing_epsilon': 0.020, 'blend_ordinal': 1.00, 'mean_lift': 0.00},
    {'variant_name': 'C3_blend_ord_060', 'temperature': 1.00, 'smoothing_epsilon': 0.00, 'blend_ordinal': 0.60, 'mean_lift': 0.00},
    {'variant_name': 'C3_blend_ord_075', 'temperature': 1.00, 'smoothing_epsilon': 0.00, 'blend_ordinal': 0.75, 'mean_lift': 0.00},
    {'variant_name': 'C3_blend_ord_090', 'temperature': 1.00, 'smoothing_epsilon': 0.00, 'blend_ordinal': 0.90, 'mean_lift': 0.00},
    {'variant_name': 'C4_lift_003', 'temperature': 1.00, 'smoothing_epsilon': 0.00, 'blend_ordinal': 1.00, 'mean_lift': 0.03},
    {'variant_name': 'C4_lift_005', 'temperature': 1.00, 'smoothing_epsilon': 0.00, 'blend_ordinal': 1.00, 'mean_lift': 0.05},
    {'variant_name': 'C4_lift_008', 'temperature': 1.00, 'smoothing_epsilon': 0.00, 'blend_ordinal': 1.00, 'mean_lift': 0.08},
    {'variant_name': 'C5_temp105_blend090', 'temperature': 1.05, 'smoothing_epsilon': 0.00, 'blend_ordinal': 0.90, 'mean_lift': 0.00},
    {'variant_name': 'C5_temp105_lift003', 'temperature': 1.05, 'smoothing_epsilon': 0.00, 'blend_ordinal': 1.00, 'mean_lift': 0.03},
    {'variant_name': 'C5_blend090_lift003', 'temperature': 1.00, 'smoothing_epsilon': 0.00, 'blend_ordinal': 0.90, 'mean_lift': 0.03},
    {'variant_name': 'C5_temp105_smooth0005_lift003', 'temperature': 1.05, 'smoothing_epsilon': 0.005, 'blend_ordinal': 1.00, 'mean_lift': 0.03},
]
pmf_calibration_config_df = pd.DataFrame(PMF_CALIBRATION_CONFIGS)
pmf_calibration_config_df.to_csv(SUM_DIR / 'pmf_calibration_config.csv', index=False)
display(pmf_calibration_config_df)
log_saved(SUM_DIR / 'pmf_calibration_config.csv')

# %% [markdown]
# # 19. Evaluate PMF Calibration Variants
#
# Bagian ini mengevaluasi setiap calibration variant pada validation. M rows memakai calibrated
# ordinal PMF + MBR, sedangkan W rows tetap memakai prediksi EXP09D original seperti EXP11A.
# Output utama adalah metrics, PMF quality, changed prediction analysis, dan valid prediction CSV.

# %%
log_section('Evaluate EXP11C PMF calibration variants')
exp09d_pred_valid = GATE_RESULTS[selected_gate_name]['pred_match'][['match_id', 'pred_team_a_goals', 'pred_team_b_goals']].copy()

CALIBRATION_RESULTS = {}
metric_rows = []
quality_rows = []
change_rows = []
c0_detail = None

for cfg in PMF_CALIBRATION_CONFIGS:
    variant_name = cfg['variant_name']
    pmf_a_cal = calibrate_goal_pmf(valid_pmf_ord_a, valid_pmf_pois_a, cfg)
    pmf_b_cal = calibrate_goal_pmf(valid_pmf_ord_b, valid_pmf_pois_b, cfg)

    pred = decode_exp11c_policy(raw_gate_valid, exp09d_pred_valid, pmf_a_cal, pmf_b_cal, variant_name)
    pred_match = raw_gate_valid.merge(pred.drop(columns=['variant_name'], errors='ignore'), on='match_id', how='left', validate='one_to_one')
    detail = prediction_detail_table(pred_match, variant_name)
    summary = summarize_prediction_detail(detail, variant_name)
    summary.update({
        'temperature': float(cfg.get('temperature', 1.0)),
        'smoothing_epsilon': float(cfg.get('smoothing_epsilon', 0.0)),
        'blend_ordinal': float(cfg.get('blend_ordinal', 1.0)),
        'mean_lift': float(cfg.get('mean_lift', 0.0)),
    })
    quality = summarize_pmf_quality(variant_name, pmf_a_cal, pmf_b_cal)
    summary.update({
        'pmf_mean_a': quality['pmf_a_mean'],
        'pmf_mean_b': quality['pmf_b_mean'],
        'pmf_mean_total': quality['pmf_total_mean'],
    })

    if variant_name == 'C0_exp11a_original':
        c0_detail = detail.copy()
        change = {
            'variant_name': variant_name,
            'n_changed_vs_c0': 0,
            'changed_rate_vs_c0': 0.0,
            'n_improved_changed': 0,
            'n_worsened_changed': 0,
            'n_neutral_changed': 0,
            'weighted_loss_delta_vs_c0': 0.0,
        }
        pred_match.to_csv(PRED_DIR / 'valid_pred_match_c0_exp11a_original.csv', index=False)
        log_saved(PRED_DIR / 'valid_pred_match_c0_exp11a_original.csv')
    else:
        compare = detail[['match_id', 'pred_team_a_goals', 'pred_team_b_goals', 'official_loss', 'tournament_weight']].merge(
            c0_detail[['match_id', 'pred_team_a_goals', 'pred_team_b_goals', 'official_loss']],
            on='match_id', how='left', suffixes=('', '_c0'), validate='one_to_one'
        )
        changed = (
            (compare['pred_team_a_goals'] != compare['pred_team_a_goals_c0'])
            | (compare['pred_team_b_goals'] != compare['pred_team_b_goals_c0'])
        )
        delta = compare['official_loss'] - compare['official_loss_c0']
        change = {
            'variant_name': variant_name,
            'n_changed_vs_c0': int(changed.sum()),
            'changed_rate_vs_c0': float(changed.mean()),
            'n_improved_changed': int(((delta < 0) & changed).sum()),
            'n_worsened_changed': int(((delta > 0) & changed).sum()),
            'n_neutral_changed': int(((delta == 0) & changed).sum()),
            'weighted_loss_delta_vs_c0': float((delta * compare['tournament_weight']).sum() / compare['tournament_weight'].sum()),
        }

    summary.update(change)
    metric_rows.append(summary)
    quality_rows.append(quality)
    change_rows.append(change)
    CALIBRATION_RESULTS[variant_name] = {
        'config': copy.deepcopy(cfg),
        'pred_match': pred_match,
        'detail': detail,
        'summary': summary,
        'pmf_quality': quality,
    }

pmf_calibration_metrics_df = pd.DataFrame(metric_rows).sort_values('awmae').reset_index(drop=True)
pmf_quality_summary_df = pd.DataFrame(quality_rows)
pmf_changed_df = pd.DataFrame(change_rows)
scoreline_dist_df = scoreline_distribution(pd.concat([v['detail'] for v in CALIBRATION_RESULTS.values()], ignore_index=True))

pmf_calibration_metrics_df.to_csv(SUM_DIR / 'pmf_calibration_metrics.csv', index=False)
pmf_quality_summary_df.to_csv(SUM_DIR / 'pmf_quality_summary.csv', index=False)
pmf_changed_df.to_csv(SUM_DIR / 'pmf_calibration_changed_prediction_analysis.csv', index=False)
scoreline_dist_df.to_csv(SUM_DIR / 'scoreline_distribution.csv', index=False)

display(pmf_calibration_metrics_df)
display(pmf_quality_summary_df)
display(pmf_changed_df)
log_saved(SUM_DIR / 'pmf_calibration_metrics.csv')
log_saved(SUM_DIR / 'pmf_quality_summary.csv')
log_saved(SUM_DIR / 'pmf_calibration_changed_prediction_analysis.csv')
log_saved(SUM_DIR / 'scoreline_distribution.csv')

# %% [markdown]
# # 20. Domain Error Analysis
#
# Bagian ini mengecek performa calibration variant berdasarkan gender, tournament_structure,
# blowout, dan high-total flags. Tujuannya memastikan PMF calibration tidak hanya menang overall,
# tetapi juga tidak merusak subgroup penting.

# %%
log_section('Domain error analysis for PMF calibration variants')
all_detail_df = pd.concat([v['detail'] for v in CALIBRATION_RESULTS.values()], ignore_index=True)
domain_gender_df = compute_domain_error_per_variant(all_detail_df, 'gender')
domain_tournament_df = compute_domain_error_per_variant(all_detail_df, 'tournament_structure')
blow_rows = []
for group_col in ['is_blowout_5plus', 'is_blowout_7plus', 'is_high_total_6plus']:
    tmp = compute_domain_error_per_variant(all_detail_df, group_col)
    if len(tmp):
        tmp['domain'] = group_col
        blow_rows.append(tmp)
domain_blowout_df = pd.concat(blow_rows, ignore_index=True) if blow_rows else pd.DataFrame()

domain_gender_df.to_csv(SUM_DIR / 'domain_error_gender.csv', index=False)
domain_tournament_df.to_csv(SUM_DIR / 'domain_error_tournament_structure.csv', index=False)
domain_blowout_df.to_csv(SUM_DIR / 'domain_error_blowout.csv', index=False)
display(domain_gender_df.head(30))
display(domain_tournament_df.head(40))
display(domain_blowout_df.head(40))
log_saved(SUM_DIR / 'domain_error_gender.csv')
log_saved(SUM_DIR / 'domain_error_tournament_structure.csv')
log_saved(SUM_DIR / 'domain_error_blowout.csv')

# %% [markdown]
# # 21. Analisis Ide Notebook Teman dan Setup EXP11D
#
# Bagian ini merangkum adaptasi yang diambil dari notebook teman: W-specialist overlay, M frozen,
# dan ordinal PMF khusus W. Ide country prior dan pseudo-labelling ditambahkan sebagai overlay legal
# karena hanya memakai train/test/sample serta mapping country yang ditulis langsung di notebook.

# %%
log_section('EXP11D setup: W-specialist overlay over EXP11C')

EXP11D_BASELINE_VARIANT = 'C3_blend_ord_090'
if EXP11D_BASELINE_VARIANT not in CALIBRATION_RESULTS:
    log_warn(f'{EXP11D_BASELINE_VARIANT} not found in CALIBRATION_RESULTS. Fallback to C0_exp11a_original.')
    EXP11D_BASELINE_VARIANT = 'C0_exp11a_original'

D0_PRED_MATCH = CALIBRATION_RESULTS[EXP11D_BASELINE_VARIANT]['pred_match'].copy()
D0_DETAIL = prediction_detail_table(D0_PRED_MATCH, 'D0_EXP11C_baseline')
D0_SUMMARY = summarize_prediction_detail(D0_DETAIL, 'D0_EXP11C_baseline')

D0_PRED_MATCH.to_csv(PRED_DIR / 'valid_pred_d0_exp11c.csv', index=False)
log_saved(PRED_DIR / 'valid_pred_d0_exp11c.csv')

log_result(f'D0 baseline variant: {EXP11D_BASELINE_VARIANT}')
log_result(f"D0 AW-MAE: {D0_SUMMARY['awmae']:.6f}")

# Save baseline artifact early.
pd.DataFrame([D0_SUMMARY]).to_csv(SUM_DIR / 'd0_exp11c_baseline_metrics.csv', index=False)
log_saved(SUM_DIR / 'd0_exp11c_baseline_metrics.csv')

# %% [markdown]
# # 22. Country Prior Features untuk W Specialist
#
# Bagian ini menambahkan hand-crafted country prior langsung di notebook. Mapping ini bukan file eksternal,
# tidak memakai target test, dan hanya dipakai sebagai fitur tambahan untuk W overlay D2/D4. Unknown country
# diberi default aman agar tidak hard fail.

# %%
def normalize_country_name(x):
    if pd.isna(x):
        return 'unknown'
    s = str(x).strip()
    aliases = {
        'USA': 'United States',
        'US': 'United States',
        'United States of America': 'United States',
        'Korea Republic': 'South Korea',
        'Republic of Korea': 'South Korea',
        'Korea DPR': 'North Korea',
        'PR China': 'China PR',
        'China': 'China PR',
        'Czechia': 'Czech Republic',
        'Russia': 'Russia',
        'Republic of Ireland': 'Ireland',
    }
    return aliases.get(s, s)

COUNTRY_TIER_W = {
    'United States': 5, 'Germany': 5, 'England': 5, 'France': 5, 'Sweden': 5,
    'Netherlands': 5, 'Spain': 5, 'Japan': 5,
    'Brazil': 4, 'Norway': 4, 'Canada': 4, 'Australia': 4, 'China PR': 4,
    'Italy': 4, 'Denmark': 4,
    'Switzerland': 3, 'Belgium': 3, 'Portugal': 3, 'South Korea': 3,
    'New Zealand': 3, 'Mexico': 3, 'Argentina': 3, 'Colombia': 3,
    'Ireland': 3, 'Scotland': 3, 'Austria': 3, 'Iceland': 3,
    'Nigeria': 2, 'South Africa': 2, 'Morocco': 2, 'Zambia': 2,
    'Jamaica': 2, 'Costa Rica': 2, 'Chile': 2, 'Uruguay': 2,
    'Thailand': 2, 'Vietnam': 2, 'Philippines': 2,
}

COUNTRY_REGION_GROUP = {
    # UEFA
    **{c: 'uefa' for c in ['Germany','England','France','Sweden','Netherlands','Spain','Norway','Italy','Denmark','Switzerland','Belgium','Portugal','Ireland','Scotland','Austria','Iceland','Czech Republic','Finland','Poland']},
    # CONCACAF
    **{c: 'concacaf' for c in ['United States','Canada','Mexico','Jamaica','Costa Rica']},
    # CONMEBOL
    **{c: 'conmebol' for c in ['Brazil','Argentina','Colombia','Chile','Uruguay']},
    # AFC/OFC
    **{c: 'afc_ofc' for c in ['Japan','China PR','South Korea','Australia','New Zealand','Thailand','Vietnam','Philippines']},
    # CAF
    **{c: 'caf' for c in ['Nigeria','South Africa','Morocco','Zambia','Ghana','Cameroon']},
}

COUNTRY_PRIOR_FEATURES_NUM = [
    'team_country_tier', 'opp_country_tier', 'country_tier_diff', 'country_tier_abs_diff',
    'team_is_elite_country', 'opp_is_elite_country', 'elite_vs_nonelite_flag',
]
COUNTRY_PRIOR_FEATURES_CAT = ['team_region_group', 'opp_region_group', 'country_region_pair']


def add_country_prior_features(df, team_col='team_a', opp_col='team_b'):
    out = df.copy()
    if team_col not in out.columns or opp_col not in out.columns:
        log_warn(f'Country prior skipped because {team_col}/{opp_col} columns are missing.')
        for c in COUNTRY_PRIOR_FEATURES_NUM:
            out[c] = 0
        for c in COUNTRY_PRIOR_FEATURES_CAT:
            out[c] = 'unknown'
        return out

    out['team_country_norm'] = out[team_col].map(normalize_country_name)
    out['opp_country_norm'] = out[opp_col].map(normalize_country_name)
    out['team_country_tier'] = out['team_country_norm'].map(COUNTRY_TIER_W).fillna(0).astype(int)
    out['opp_country_tier'] = out['opp_country_norm'].map(COUNTRY_TIER_W).fillna(0).astype(int)
    out['country_tier_diff'] = out['team_country_tier'] - out['opp_country_tier']
    out['country_tier_abs_diff'] = out['country_tier_diff'].abs()
    out['team_is_elite_country'] = (out['team_country_tier'] >= 4).astype(int)
    out['opp_is_elite_country'] = (out['opp_country_tier'] >= 4).astype(int)
    out['elite_vs_nonelite_flag'] = (out['team_is_elite_country'] != out['opp_is_elite_country']).astype(int)
    out['team_region_group'] = out['team_country_norm'].map(COUNTRY_REGION_GROUP).fillna('unknown').astype(str)
    out['opp_region_group'] = out['opp_country_norm'].map(COUNTRY_REGION_GROUP).fillna('unknown').astype(str)
    out['country_region_pair'] = out['team_region_group'] + '_vs_' + out['opp_region_group']
    return out

# Build country-augmented frames. These preserve original rows and only add W specialist features.
train_sup_country = add_country_prior_features(train_sup_full)
valid_sup_country = add_country_prior_features(valid_sup_full)
test_feature_country = add_country_prior_features(test_feature_full)

country_prior_summary_df = pd.DataFrame({
    'feature_name': COUNTRY_PRIOR_FEATURES_NUM + COUNTRY_PRIOR_FEATURES_CAT,
    'feature_type': ['numeric'] * len(COUNTRY_PRIOR_FEATURES_NUM) + ['categorical'] * len(COUNTRY_PRIOR_FEATURES_CAT),
})
country_prior_summary_df.to_csv(SUM_DIR / 'country_prior_feature_summary.csv', index=False)
display(country_prior_summary_df)
log_saved(SUM_DIR / 'country_prior_feature_summary.csv')

# %% [markdown]
# # 23. W-Only Ordinal Helper dan Pseudo-Label Utility
#
# Bagian ini membuat helper W specialist. Model hanya dilatih untuk gender W, M tidak disentuh. Untuk
# pseudo-labelling, label berasal dari prediksi model pada test W high-confidence, bukan dari ground truth.

# %%
def make_w_feature_set(use_country_prior=False):
    feature_cols = list(exp05a_feature_cols)
    cat_cols = list(exp05a_cat_features)
    if use_country_prior:
        for c in COUNTRY_PRIOR_FEATURES_NUM + COUNTRY_PRIOR_FEATURES_CAT:
            if c not in feature_cols:
                feature_cols.append(c)
        for c in COUNTRY_PRIOR_FEATURES_CAT:
            if c not in cat_cols:
                cat_cols.append(c)
    return feature_cols, cat_cols


def _ensure_y_goal_columns(df):
    out = df.copy()
    if 'y_goal_a' not in out.columns and 'team_a_goals' in out.columns:
        out['y_goal_a'] = pd.to_numeric(out['team_a_goals'], errors='coerce')
    if 'y_goal_b' not in out.columns and 'team_b_goals' in out.columns:
        out['y_goal_b'] = pd.to_numeric(out['team_b_goals'], errors='coerce')
    return out


def fit_binary_model_safe_weighted(Xtr, ytr, Xva, yva, booster, params, cats, sample_weight=None):
    ytr = pd.Series(ytr).astype(int)
    if ytr.nunique() < 2:
        return ConstantBinaryModel(float(ytr.mean()))
    if booster == 'cat':
        m = CatBoostClassifier(**params)
        m.fit(
            Xtr, ytr,
            eval_set=(Xva, pd.Series(yva).astype(int)),
            cat_features=[c for c in cats if c in Xtr.columns],
            sample_weight=sample_weight,
            use_best_model=True,
            verbose=params.get('verbose', MODEL_VERBOSE),
        )
        return m
    if booster == 'lgb':
        m = lgb.LGBMClassifier(**params)
        m.fit(
            Xtr, ytr,
            sample_weight=sample_weight,
            eval_set=[(Xva, pd.Series(yva).astype(int))],
            categorical_feature=[c for c in cats if c in Xtr.columns],
            callbacks=[lgb.early_stopping(100, verbose=LGB_EARLY_STOP_VERBOSE)],
        )
        return m
    return fit_binary_model_safe(Xtr, ytr, Xva, yva, booster, params, cats)


def fit_w_only_ordinal_models(train_df, valid_df, feature_cols, cat_features, use_country_prior=False, pseudo_df=None, pseudo_weight=0.25, max_goal=10, n_estimators=650):
    booster = final_selected.get('W', final_selected.get('M', list(final_selected.values())[0]))['booster']
    regime = final_selected.get('W', final_selected.get('M', list(final_selected.values())[0]))['regime']
    tr = filter_regime_train(train_df, regime)
    tr = tr[tr['gender'].astype(str).str.upper().eq('W')].copy()
    va = valid_df[valid_df['gender'].astype(str).str.upper().eq('W')].copy()
    tr = _ensure_y_goal_columns(tr)
    va = _ensure_y_goal_columns(va)
    if pseudo_df is not None and len(pseudo_df) > 0:
        pseudo = pseudo_df.copy()
        pseudo['gender'] = 'W'
        pseudo['is_pseudo'] = 1
        tr['is_pseudo'] = 0
        # keep only columns that exist in either frame; features + target are required
        need_cols = list(dict.fromkeys(['match_id','gender','date','y_goal_a','y_goal_b','is_pseudo'] + feature_cols))
        for c in need_cols:
            if c not in tr.columns:
                tr[c] = np.nan
            if c not in pseudo.columns:
                pseudo[c] = np.nan
        tr = pd.concat([tr[need_cols], pseudo[need_cols]], ignore_index=True)
    else:
        tr['is_pseudo'] = 0
    if len(tr) < 250 or len(va) < 50:
        raise RuntimeError(f'W-only ordinal data too small: train={len(tr)} valid={len(va)}')

    cols = list(dict.fromkeys(['match_id','gender','date','y_goal_a','y_goal_b','is_pseudo'] + feature_cols))
    tr = select_unique_columns(tr, cols)
    va = select_unique_columns(va, [c for c in cols if c != 'is_pseudo'])
    Xtr = prepare_model_frame(tr, feature_cols, cat_features)
    Xva = prepare_model_frame(va, feature_cols, cat_features)
    params = make_default_params(booster, 'tail', SEED, n_estimators)
    sample_weight = np.where(tr.get('is_pseudo', 0).to_numpy(int) == 1, float(pseudo_weight), 1.0)
    models = {
        'booster': booster,
        'regime': regime,
        'feature_cols': list(feature_cols),
        'cat_features': list(cat_features),
        'use_country_prior': bool(use_country_prior),
        'pseudo_weight': float(pseudo_weight),
        'max_goal': int(max_goal),
        'side_models': {'a': {}, 'b': {}},
    }
    threshold_rows = []
    for side, target_col in [('a','y_goal_a'), ('b','y_goal_b')]:
        ytr_goal = pd.to_numeric(tr[target_col], errors='coerce').fillna(0).astype(int)
        yva_goal = pd.to_numeric(va[target_col], errors='coerce').fillna(0).astype(int)
        for k in range(1, max_goal + 1):
            ytr_bin = (ytr_goal >= k).astype(int)
            yva_bin = (yva_goal >= k).astype(int)
            model = fit_binary_model_safe_weighted(Xtr, ytr_bin, Xva, yva_bin, booster, params, cat_features, sample_weight=sample_weight)
            models['side_models'][side][k] = model
            pred = predict_pos(model, Xva)
            threshold_rows.append({
                'side': side,
                'threshold': k,
                'booster': booster,
                'regime': regime,
                'use_country_prior': bool(use_country_prior),
                'pseudo_rows': int((tr.get('is_pseudo', 0) == 1).sum()),
                'valid_positive_rate': float(yva_bin.mean()),
                'pred_positive_mean': float(np.mean(pred)),
                'brier': float(np.mean((pred - yva_bin.to_numpy(float)) ** 2)),
            })
    return models, pd.DataFrame(threshold_rows)


def predict_w_ordinal_pmfs(df, models, max_goal=10):
    sub = df.copy().reset_index(drop=True)
    feature_cols = models['feature_cols']
    cat_features = models['cat_features']
    X = prepare_model_frame(sub, feature_cols, cat_features)
    probs = {}
    for side in ['a', 'b']:
        arr = np.zeros((len(sub), max_goal), dtype=float)
        for k in range(1, max_goal + 1):
            arr[:, k-1] = predict_pos(models['side_models'][side][k], X)
        probs[side] = ge_probs_to_pmf(arr)
    return probs['a'], probs['b']


def scoreline_confidence_from_pmfs(pmf_a, pmf_b):
    joint = pmf_a[:, :, None] * pmf_b[:, None, :]
    flat = joint.reshape(joint.shape[0], -1)
    sorted_prob = np.sort(flat, axis=1)[:, ::-1]
    top1 = sorted_prob[:, 0]
    top2 = sorted_prob[:, 1] if sorted_prob.shape[1] > 1 else np.zeros_like(top1)
    margin = top1 - top2
    return top1, margin


def select_pseudo_labels(test_w_df, pred_w_df, pmf_a, pmf_b, coverage=0.10):
    df = test_w_df[['match_id']].copy().reset_index(drop=True)
    df = df.merge(pred_w_df[['match_id','pred_team_a_goals','pred_team_b_goals']], on='match_id', how='left', validate='one_to_one')
    top1, margin = scoreline_confidence_from_pmfs(pmf_a, pmf_b)
    outcome_conf = test_w_df[[c for c in ['pred_outcome_proba_0','pred_outcome_proba_1','pred_outcome_proba_2'] if c in test_w_df.columns]].max(axis=1).to_numpy(float)
    df['top1_scoreline_prob'] = top1
    df['top1_top2_margin'] = margin
    df['outcome_confidence'] = outcome_conf
    df['pseudo_confidence'] = 0.50 * top1 + 0.30 * margin + 0.20 * outcome_conf
    df['y_goal_a'] = pd.to_numeric(df['pred_team_a_goals'], errors='coerce').fillna(0).astype(int)
    df['y_goal_b'] = pd.to_numeric(df['pred_team_b_goals'], errors='coerce').fillna(0).astype(int)
    df = df.sort_values('pseudo_confidence', ascending=False).reset_index(drop=True)
    n_select = int(np.floor(len(df) * float(coverage)))
    n_select = max(0, min(len(df), n_select))
    selected = df.head(n_select).copy()
    selected['is_pseudo'] = 1
    return selected

# %% [markdown]
# # 24. W PMF Overlay Evaluation Variants D1-D4
#
# Bagian ini membangun W-only candidates. M selalu diambil dari D0 EXP11C. Variant yang diuji:
# D1 W PMF overlay, D2 + country prior, D3 + pseudo-label, dan D4 + country prior + pseudo-label.
# Semua output disimpan sebagai DataFrame agar mudah dibandingkan.

# %%
def align_array_by_match_id(source_df, arr, target_match_ids):
    tmp = pd.DataFrame({'match_id': source_df['match_id'].values, '_idx': np.arange(len(source_df))})
    idx = tmp.set_index('match_id').loc[list(target_match_ids), '_idx'].to_numpy(int)
    return np.asarray(arr)[idx]


def decode_w_overlay_from_pmfs(raw_gate_df, d0_pred_df, pmf_a, pmf_b, blend_ordinal=0.75, variant_name='D1'):
    raw = raw_gate_df.reset_index(drop=True).copy()
    is_w = raw['gender'].astype(str).str.upper().eq('W').to_numpy()
    base_out = d0_pred_df[['match_id','pred_team_a_goals','pred_team_b_goals']].copy()
    if is_w.any():
        idx_w = np.where(is_w)[0]
        pois_a = poisson_pmf_from_lambda(raw.loc[is_w, 'pred_goal_a_cont'], max_goal=pmf_a.shape[1]-1)
        pois_b = poisson_pmf_from_lambda(raw.loc[is_w, 'pred_goal_b_cont'], max_goal=pmf_b.shape[1]-1)
        cal_a = blend_pmfs(pmf_a, pois_a, blend_ordinal=blend_ordinal)
        cal_b = blend_pmfs(pmf_b, pois_b, blend_ordinal=blend_ordinal)
        w_pred = mbr_decode_from_pmfs_basic(raw.loc[is_w].copy(), cal_a, cal_b, max_decode_goals=MAX_DECODE_GOALS)
        base_out = base_out[~base_out['match_id'].isin(w_pred['match_id'])].copy()
        base_out = pd.concat([base_out, w_pred[['match_id','pred_team_a_goals','pred_team_b_goals']]], ignore_index=True)
    out = raw[['match_id','gender','tournament','tournament_structure']].merge(base_out, on='match_id', how='left', validate='one_to_one')
    out['variant_name'] = variant_name
    return out


def assert_m_predictions_frozen(candidate_df, baseline_df):
    merged = candidate_df.merge(
        baseline_df[['match_id','pred_team_a_goals','pred_team_b_goals','gender']],
        on='match_id', how='left', suffixes=('', '_baseline'), validate='one_to_one'
    )
    m_mask = merged['gender'].astype(str).str.upper().eq('M')
    changed_m = (
        (merged.loc[m_mask, 'pred_team_a_goals'] != merged.loc[m_mask, 'pred_team_a_goals_baseline']) |
        (merged.loc[m_mask, 'pred_team_b_goals'] != merged.loc[m_mask, 'pred_team_b_goals_baseline'])
    ).sum()
    assert int(changed_m) == 0, f'M predictions changed: {changed_m}'
    return int(changed_m)


def detail_and_summary_for_variant(pred_df, variant_name, d0_detail=None):
    pred_match = raw_gate_valid.merge(
        pred_df[['match_id','pred_team_a_goals','pred_team_b_goals']],
        on='match_id', how='left', validate='one_to_one'
    )
    detail = prediction_detail_table(pred_match, variant_name)
    summary = summarize_prediction_detail(detail, variant_name)
    if d0_detail is not None:
        compare = detail[['match_id','pred_team_a_goals','pred_team_b_goals','official_loss','tournament_weight']].merge(
            d0_detail[['match_id','pred_team_a_goals','pred_team_b_goals','official_loss']],
            on='match_id', how='left', suffixes=('', '_d0'), validate='one_to_one'
        )
        changed = (compare['pred_team_a_goals'] != compare['pred_team_a_goals_d0']) | (compare['pred_team_b_goals'] != compare['pred_team_b_goals_d0'])
        delta = compare['official_loss'] - compare['official_loss_d0']
        summary.update({
            'n_changed_vs_d0': int(changed.sum()),
            'changed_rate_vs_d0': float(changed.mean()),
            'n_improved_changed': int(((delta < 0) & changed).sum()),
            'n_worsened_changed': int(((delta > 0) & changed).sum()),
            'n_neutral_changed': int(((delta == 0) & changed).sum()),
            'weighted_loss_delta_vs_d0': float((delta * compare['tournament_weight']).sum() / compare['tournament_weight'].sum()),
        })
    else:
        summary.update({'n_changed_vs_d0':0,'changed_rate_vs_d0':0.0,'n_improved_changed':0,'n_worsened_changed':0,'n_neutral_changed':0,'weighted_loss_delta_vs_d0':0.0})
    return pred_match, detail, summary

# D0 baseline store.
D_VARIANT_RESULTS = {
    'D0_EXP11C_baseline': {
        'pred_match': D0_PRED_MATCH.copy(),
        'detail': D0_DETAIL.copy(),
        'summary': {**D0_SUMMARY, 'variant_name':'D0_EXP11C_baseline', 'overlay_group':'D0', 'blend_ordinal':np.nan, 'use_country_prior':False, 'use_pseudo_label':False, 'pseudo_coverage':0.0, 'pseudo_weight':0.0, 'm_changed_count':0},
        'config': {'variant_name':'D0_EXP11C_baseline'},
    }
}

# Prepare W masks and PMFs in raw_gate order.
w_valid_mask = raw_gate_valid['gender'].astype(str).str.upper().eq('W').to_numpy()
w_valid_ids = raw_gate_valid.loc[w_valid_mask, 'match_id'].values

valid_pmf_ord_a_w = valid_pmf_ord_a[w_valid_mask]
valid_pmf_ord_b_w = valid_pmf_ord_b[w_valid_mask]

# D1: W PMF overlay only with small blend grid.
for blend in [0.60, 0.75, 0.90]:
    name = f'D1_w_pmf_overlay_blend{int(blend*100):03d}'
    pred = decode_w_overlay_from_pmfs(raw_gate_valid, D0_PRED_MATCH, valid_pmf_ord_a_w, valid_pmf_ord_b_w, blend_ordinal=blend, variant_name=name)
    m_changed = assert_m_predictions_frozen(pred, D0_PRED_MATCH)
    pred_match, detail, summary = detail_and_summary_for_variant(pred, name, D0_DETAIL)
    summary.update({'overlay_group':'D1','blend_ordinal':blend,'use_country_prior':False,'use_pseudo_label':False,'pseudo_coverage':0.0,'pseudo_weight':0.0,'m_changed_count':m_changed})
    D_VARIANT_RESULTS[name] = {'pred_match':pred_match,'detail':detail,'summary':summary,'config':{'blend_ordinal':blend,'use_country_prior':False,'use_pseudo_label':False}}
    pred_match.to_csv(PRED_DIR / f'valid_pred_{name}.csv', index=False)

# D2: W PMF overlay + country prior features.
log_section('Train W-only country-prior ordinal model')
w_feature_country, w_cat_country = make_w_feature_set(use_country_prior=True)
w_country_models, w_country_threshold_metrics = fit_w_only_ordinal_models(
    train_sup_country, valid_sup_country, w_feature_country, w_cat_country,
    use_country_prior=True, pseudo_df=None, pseudo_weight=0.0,
    max_goal=MAX_ORDINAL_GOALS, n_estimators=ORDINAL_N_ESTIMATORS,
)
w_country_threshold_metrics.to_csv(SUM_DIR / 'w_country_ordinal_threshold_metrics.csv', index=False)
log_saved(SUM_DIR / 'w_country_ordinal_threshold_metrics.csv')

valid_country_for_w = raw_gate_valid.loc[w_valid_mask, ['match_id']].merge(valid_sup_country, on='match_id', how='left', validate='one_to_one')
w_country_pmf_a, w_country_pmf_b = predict_w_ordinal_pmfs(valid_country_for_w, w_country_models, max_goal=MAX_ORDINAL_GOALS)

for blend in [0.75, 0.90]:
    name = f'D2_w_country_prior_blend{int(blend*100):03d}'
    pred = decode_w_overlay_from_pmfs(raw_gate_valid, D0_PRED_MATCH, w_country_pmf_a, w_country_pmf_b, blend_ordinal=blend, variant_name=name)
    m_changed = assert_m_predictions_frozen(pred, D0_PRED_MATCH)
    pred_match, detail, summary = detail_and_summary_for_variant(pred, name, D0_DETAIL)
    summary.update({'overlay_group':'D2','blend_ordinal':blend,'use_country_prior':True,'use_pseudo_label':False,'pseudo_coverage':0.0,'pseudo_weight':0.0,'m_changed_count':m_changed})
    D_VARIANT_RESULTS[name] = {'pred_match':pred_match,'detail':detail,'summary':summary,'config':{'blend_ordinal':blend,'use_country_prior':True,'use_pseudo_label':False}}
    pred_match.to_csv(PRED_DIR / f'valid_pred_{name}.csv', index=False)

# Build pseudo-label candidates from test W using validation-stage models. This is transductive but legal: no GT is read.
log_section('Build W pseudo-label candidates from test predictions')
test_raw_base_pl = predict_gender_split_raw_outputs_with_config(test_feature_full, base_models)
test_raw_repair_pl = predict_gender_split_raw_outputs_with_config(test_feature_full, repair_models)
test_raw_gate_pl = apply_tournament_gated_outcome(test_raw_base_pl, test_raw_repair_pl, selected_gate_cfg['gate_map'], default_alpha=selected_gate_cfg['default_alpha'])
test_pred_exp09d_pl = decode_directional_tail_batch(
    test_raw_gate_pl.drop(columns=TAIL_COLS, errors='ignore'),
    test_raw_base_pl[['match_id'] + TAIL_COLS],
    selected_exp09d_decoder,
    scoreline_prior_lookup[int(selected_exp09d_decoder['MAX_GOALS'])],
)
test_for_pmf_pl = test_raw_gate_pl[['match_id']].merge(test_feature_full, on='match_id', how='left', validate='one_to_one')
test_ge_a_pl = predict_ordinal_ge_probs(test_for_pmf_pl, ordinal_models, side='a', max_goal=MAX_ORDINAL_GOALS)
test_ge_b_pl = predict_ordinal_ge_probs(test_for_pmf_pl, ordinal_models, side='b', max_goal=MAX_ORDINAL_GOALS)
test_pmf_ord_a_pl = ge_probs_to_pmf(test_ge_a_pl)
test_pmf_ord_b_pl = ge_probs_to_pmf(test_ge_b_pl)
test_pmf_pois_a_pl = poisson_pmf_from_lambda(test_raw_gate_pl['pred_goal_a_cont'], max_goal=MAX_ORDINAL_GOALS)
test_pmf_pois_b_pl = poisson_pmf_from_lambda(test_raw_gate_pl['pred_goal_b_cont'], max_goal=MAX_ORDINAL_GOALS)
pl_cfg = {'blend_ordinal': 0.90, 'temperature': 1.0, 'smoothing_epsilon': 0.0, 'mean_lift': 0.0}
test_pmf_cal_a_pl = calibrate_goal_pmf(test_pmf_ord_a_pl, test_pmf_pois_a_pl, pl_cfg)
test_pmf_cal_b_pl = calibrate_goal_pmf(test_pmf_ord_b_pl, test_pmf_pois_b_pl, pl_cfg)
test_pred_pl = decode_exp11c_policy(test_raw_gate_pl, test_pred_exp09d_pl, test_pmf_cal_a_pl, test_pmf_cal_b_pl, 'pseudo_source_exp11c')
test_w_mask_pl = test_raw_gate_pl['gender'].astype(str).str.upper().eq('W').to_numpy()
test_w_source_df = test_raw_gate_pl.loc[test_w_mask_pl].reset_index(drop=True)
test_w_pred_pl = test_pred_pl[test_pred_pl['match_id'].isin(test_w_source_df['match_id'])].copy()
test_w_pmf_a_pl = test_pmf_cal_a_pl[test_w_mask_pl]
test_w_pmf_b_pl = test_pmf_cal_b_pl[test_w_mask_pl]

pseudo_summary_rows = []
pseudo_label_cache = {}
for coverage in [0.10, 0.20]:
    selected = select_pseudo_labels(test_w_source_df, test_w_pred_pl, test_w_pmf_a_pl, test_w_pmf_b_pl, coverage=coverage)
    selected_feat = selected[['match_id','y_goal_a','y_goal_b','pseudo_confidence','top1_scoreline_prob','top1_top2_margin','outcome_confidence']].merge(
        test_feature_full, on='match_id', how='left', validate='one_to_one'
    )
    selected_feat_country = selected[['match_id','y_goal_a','y_goal_b','pseudo_confidence','top1_scoreline_prob','top1_top2_margin','outcome_confidence']].merge(
        test_feature_country, on='match_id', how='left', validate='one_to_one'
    )
    pseudo_label_cache[coverage] = {'plain': selected_feat, 'country': selected_feat_country, 'summary': selected}
    pseudo_summary_rows.append({
        'coverage': coverage,
        'n_pseudo_matches': len(selected),
        'mean_pseudo_confidence': float(selected['pseudo_confidence'].mean()) if len(selected) else np.nan,
        'mean_pseudo_pred_total': float((selected['y_goal_a'] + selected['y_goal_b']).mean()) if len(selected) else np.nan,
        'all_gender_w': True,
    })
pseudo_label_summary_df = pd.DataFrame(pseudo_summary_rows)
pseudo_label_summary_df.to_csv(SUM_DIR / 'pseudo_label_summary.csv', index=False)
display(pseudo_label_summary_df)
log_saved(SUM_DIR / 'pseudo_label_summary.csv')

# D3: pseudo-label only.
for coverage in [0.10, 0.20]:
    for pseudo_weight in [0.15, 0.25]:
        pseudo_df = pseudo_label_cache[coverage]['plain']
        w_feature_plain, w_cat_plain = make_w_feature_set(use_country_prior=False)
        w_pseudo_models, w_pseudo_threshold = fit_w_only_ordinal_models(
            train_sup_full, valid_sup_full, w_feature_plain, w_cat_plain,
            use_country_prior=False, pseudo_df=pseudo_df, pseudo_weight=pseudo_weight,
            max_goal=MAX_ORDINAL_GOALS, n_estimators=ORDINAL_N_ESTIMATORS,
        )
        valid_plain_for_w = raw_gate_valid.loc[w_valid_mask, ['match_id']].merge(valid_sup_full, on='match_id', how='left', validate='one_to_one')
        w_pseudo_pmf_a, w_pseudo_pmf_b = predict_w_ordinal_pmfs(valid_plain_for_w, w_pseudo_models, max_goal=MAX_ORDINAL_GOALS)
        name = f'D3_w_pseudo_cov{int(coverage*100):02d}_wt{int(pseudo_weight*100):02d}'
        pred = decode_w_overlay_from_pmfs(raw_gate_valid, D0_PRED_MATCH, w_pseudo_pmf_a, w_pseudo_pmf_b, blend_ordinal=0.75, variant_name=name)
        m_changed = assert_m_predictions_frozen(pred, D0_PRED_MATCH)
        pred_match, detail, summary = detail_and_summary_for_variant(pred, name, D0_DETAIL)
        summary.update({'overlay_group':'D3','blend_ordinal':0.75,'use_country_prior':False,'use_pseudo_label':True,'pseudo_coverage':coverage,'pseudo_weight':pseudo_weight,'m_changed_count':m_changed})
        D_VARIANT_RESULTS[name] = {'pred_match':pred_match,'detail':detail,'summary':summary,'config':{'blend_ordinal':0.75,'use_country_prior':False,'use_pseudo_label':True,'pseudo_coverage':coverage,'pseudo_weight':pseudo_weight}, 'pseudo_threshold_metrics':w_pseudo_threshold}
        pred_match.to_csv(PRED_DIR / f'valid_pred_{name}.csv', index=False)

# D4: country prior + pseudo-label.
for coverage in [0.10, 0.20]:
    pseudo_weight = 0.15
    pseudo_df = pseudo_label_cache[coverage]['country']
    w_country_pseudo_models, w_country_pseudo_threshold = fit_w_only_ordinal_models(
        train_sup_country, valid_sup_country, w_feature_country, w_cat_country,
        use_country_prior=True, pseudo_df=pseudo_df, pseudo_weight=pseudo_weight,
        max_goal=MAX_ORDINAL_GOALS, n_estimators=ORDINAL_N_ESTIMATORS,
    )
    valid_country_for_w = raw_gate_valid.loc[w_valid_mask, ['match_id']].merge(valid_sup_country, on='match_id', how='left', validate='one_to_one')
    w_cp_pmf_a, w_cp_pmf_b = predict_w_ordinal_pmfs(valid_country_for_w, w_country_pseudo_models, max_goal=MAX_ORDINAL_GOALS)
    name = f'D4_w_country_pseudo_cov{int(coverage*100):02d}_wt{int(pseudo_weight*100):02d}'
    pred = decode_w_overlay_from_pmfs(raw_gate_valid, D0_PRED_MATCH, w_cp_pmf_a, w_cp_pmf_b, blend_ordinal=0.75, variant_name=name)
    m_changed = assert_m_predictions_frozen(pred, D0_PRED_MATCH)
    pred_match, detail, summary = detail_and_summary_for_variant(pred, name, D0_DETAIL)
    summary.update({'overlay_group':'D4','blend_ordinal':0.75,'use_country_prior':True,'use_pseudo_label':True,'pseudo_coverage':coverage,'pseudo_weight':pseudo_weight,'m_changed_count':m_changed})
    D_VARIANT_RESULTS[name] = {'pred_match':pred_match,'detail':detail,'summary':summary,'config':{'blend_ordinal':0.75,'use_country_prior':True,'use_pseudo_label':True,'pseudo_coverage':coverage,'pseudo_weight':pseudo_weight}, 'pseudo_threshold_metrics':w_country_pseudo_threshold}
    pred_match.to_csv(PRED_DIR / f'valid_pred_{name}.csv', index=False)

# Save canonical required prediction names for the first available representative in each group.
REPRESENTATIVE_NAMES = {
    'D1': next((k for k in D_VARIANT_RESULTS if k.startswith('D1_')), None),
    'D2': next((k for k in D_VARIANT_RESULTS if k.startswith('D2_')), None),
    'D3': next((k for k in D_VARIANT_RESULTS if k.startswith('D3_')), None),
    'D4': next((k for k in D_VARIANT_RESULTS if k.startswith('D4_')), None),
}
for group, name in REPRESENTATIVE_NAMES.items():
    if name is not None:
        out_name = {
            'D1': 'valid_pred_d1_w_pmf_overlay.csv',
            'D2': 'valid_pred_d2_w_country_prior.csv',
            'D3': 'valid_pred_d3_w_pseudolabel.csv',
            'D4': 'valid_pred_d4_w_country_pseudolabel.csv',
        }[group]
        D_VARIANT_RESULTS[name]['pred_match'].to_csv(PRED_DIR / out_name, index=False)
        log_saved(PRED_DIR / out_name)

# %% [markdown]
# # 25. Variant Metrics, W-Specific Analysis, dan Changed Prediction
#
# Bagian ini mengevaluasi semua overlay. Tabel utama berisi metric overall, lalu disusul gender,
# tournament_structure, country tier gap, pseudo-label summary, dan changed prediction analysis.

# %%
variant_metrics_df = pd.DataFrame([v['summary'] for v in D_VARIANT_RESULTS.values()]).sort_values('awmae').reset_index(drop=True)
variant_metrics_df.to_csv(SUM_DIR / 'variant_metrics.csv', index=False)
display(variant_metrics_df)
log_saved(SUM_DIR / 'variant_metrics.csv')

all_d_detail_df = pd.concat([v['detail'] for v in D_VARIANT_RESULTS.values()], ignore_index=True)
gender_metrics_df = compute_domain_error_per_variant(all_d_detail_df, 'gender')
tournament_structure_metrics_df = compute_domain_error_per_variant(all_d_detail_df, 'tournament_structure')

# country tier gap domain only if feature can be joined.
def add_tier_gap_group_to_detail(detail_df):
    tmp = detail_df.merge(valid_sup_country[['match_id','country_tier_abs_diff']], on='match_id', how='left')
    def bucket(x):
        if pd.isna(x): return 'unknown'
        x = int(x)
        if x <= 0: return 'tier_gap_0'
        if x == 1: return 'tier_gap_1'
        if x == 2: return 'tier_gap_2'
        return 'tier_gap_3plus'
    tmp['country_tier_gap_group'] = tmp['country_tier_abs_diff'].map(bucket)
    return tmp

all_d_detail_with_tier = add_tier_gap_group_to_detail(all_d_detail_df)
country_tier_gap_metrics_df = compute_domain_error_per_variant(all_d_detail_with_tier, 'country_tier_gap_group')

# W-specific table.
w_metrics_rows = []
for variant, g in all_d_detail_df[all_d_detail_df['gender'].astype(str).str.upper().eq('W')].groupby('variant_name'):
    w_metrics_rows.append({
        'variant_name': variant,
        'w_n': len(g),
        'w_awmae': awmae_score(g['actual_team_a_goals'], g['actual_team_b_goals'], g['pred_team_a_goals'], g['pred_team_b_goals'], g['tournament']),
        'w_base_mae': float(g['base_mae'].mean()),
        'w_exact_rate': float(g['exact_hit'].mean()),
        'w_outcome_rate': float(g['outcome_hit'].mean()),
        'w_gd_rate': float(g['gd_hit'].mean()),
        'w_total_goal_bias': float(((g['pred_team_a_goals'] + g['pred_team_b_goals']) - (g['actual_team_a_goals'] + g['actual_team_b_goals'])).mean()),
        'w_mean_pred_total': float((g['pred_team_a_goals'] + g['pred_team_b_goals']).mean()),
    })
w_metrics_df = pd.DataFrame(w_metrics_rows).sort_values('w_awmae').reset_index(drop=True)

# Changed analysis is already in variant metrics; keep a dedicated CSV.
changed_prediction_df = variant_metrics_df[[c for c in variant_metrics_df.columns if c in ['variant_name','overlay_group','n_changed_vs_d0','changed_rate_vs_d0','n_improved_changed','n_worsened_changed','n_neutral_changed','weighted_loss_delta_vs_d0']]].copy()

gender_metrics_df.to_csv(SUM_DIR / 'gender_metrics.csv', index=False)
tournament_structure_metrics_df.to_csv(SUM_DIR / 'tournament_structure_metrics.csv', index=False)
country_tier_gap_metrics_df.to_csv(SUM_DIR / 'country_tier_gap_metrics.csv', index=False)
w_metrics_df.to_csv(SUM_DIR / 'w_metrics.csv', index=False)
changed_prediction_df.to_csv(SUM_DIR / 'changed_prediction_analysis.csv', index=False)

display(gender_metrics_df.head(40))
display(w_metrics_df)
display(tournament_structure_metrics_df.head(50))
display(country_tier_gap_metrics_df.head(40))
display(changed_prediction_df)
log_saved(SUM_DIR / 'gender_metrics.csv')
log_saved(SUM_DIR / 'tournament_structure_metrics.csv')
log_saved(SUM_DIR / 'country_tier_gap_metrics.csv')
log_saved(SUM_DIR / 'w_metrics.csv')
log_saved(SUM_DIR / 'changed_prediction_analysis.csv')

# %% [markdown]
# # 26. Final Decision EXP11D
#
# Bagian ini memilih variant hanya jika improvement jelas dan guardrail lolos. D0 EXP11C selalu menjadi
# fallback. Jika gain antarvariant kecil, variant yang lebih konservatif dipilih.

# %%
log_section('EXP11D final decision')
d0_row = variant_metrics_df[variant_metrics_df['variant_name'].eq('D0_EXP11C_baseline')].iloc[0]
d0_w_row = w_metrics_df[w_metrics_df['variant_name'].eq('D0_EXP11C_baseline')].iloc[0]

candidate_rows = []
for _, row in variant_metrics_df.iterrows():
    if row['variant_name'] == 'D0_EXP11C_baseline':
        continue
    w_row = w_metrics_df[w_metrics_df['variant_name'].eq(row['variant_name'])]
    if len(w_row) == 0:
        continue
    w_row = w_row.iloc[0]
    overall_gain = float(d0_row['awmae'] - row['awmae'])
    w_gain = float(d0_w_row['w_awmae'] - w_row['w_awmae'])
    w_outcome_drop = float(d0_w_row['w_outcome_rate'] - w_row['w_outcome_rate'])
    w_gd_drop = float(d0_w_row['w_gd_rate'] - w_row['w_gd_rate'])
    changed_rate = float(row.get('changed_rate_vs_d0', 1.0))
    top3_delta = float(row.get('top3_scoreline_share', 0.0) - d0_row.get('top3_scoreline_share', 0.0))
    guardrail_pass = (
        overall_gain >= 0.0015 and
        w_gain >= 0.0060 and
        int(row.get('m_changed_count', 999)) == 0 and
        w_outcome_drop <= 0.003 and
        w_gd_drop <= 0.003 and
        changed_rate <= 0.25 and
        1.0 <= float(w_row.get('w_mean_pred_total', 2.5)) <= 4.8 and
        top3_delta <= 0.04
    )
    candidate_rows.append({
        'variant_name': row['variant_name'],
        'overlay_group': row.get('overlay_group', ''),
        'awmae': float(row['awmae']),
        'overall_gain_vs_d0': overall_gain,
        'w_gain_vs_d0': w_gain,
        'w_outcome_drop_vs_d0': w_outcome_drop,
        'w_gd_drop_vs_d0': w_gd_drop,
        'changed_rate_vs_d0': changed_rate,
        'm_changed_count': int(row.get('m_changed_count', 999)),
        'guardrail_pass': bool(guardrail_pass),
    })

decision_candidates_df = pd.DataFrame(candidate_rows).sort_values(['guardrail_pass','awmae'], ascending=[False, True]).reset_index(drop=True)
decision_candidates_df.to_csv(SUM_DIR / 'decision_candidates.csv', index=False)
display(decision_candidates_df)

passed = decision_candidates_df[decision_candidates_df['guardrail_pass']].copy() if len(decision_candidates_df) else pd.DataFrame()
if len(passed) == 0:
    selected_variant = 'D0_EXP11C_baseline'
    decision_code = 'C'
    decision_text = 'No W specialist overlay passed the explicit guardrail. Fallback to EXP11C D0.'
else:
    passed = passed.sort_values('awmae').reset_index(drop=True)
    best = passed.iloc[0]
    # If gains are nearly tied, choose the more conservative group.
    conservative_rank = {'D1': 1, 'D2': 2, 'D3': 3, 'D4': 4, 'D5': 2}
    close = passed[passed['awmae'] <= float(best['awmae']) + 0.0005].copy()
    close['rank'] = close['overlay_group'].map(conservative_rank).fillna(9)
    chosen = close.sort_values(['rank','awmae']).iloc[0]
    selected_variant = str(chosen['variant_name'])
    decision_code = 'A'
    decision_text = 'Selected W specialist overlay because it improves validation and passes W/M guardrails.'

selected_config = D_VARIANT_RESULTS[selected_variant]['config']
final_decision = {
    'experiment': 'EXP11D',
    'base_experiment': 'EXP11C',
    'd0_variant': EXP11D_BASELINE_VARIANT,
    'selected_variant': selected_variant,
    'decision_code': decision_code,
    'decision_text': decision_text,
    'selected_config': selected_config,
    'd0_awmae': float(d0_row['awmae']),
    'selected_awmae': float(D_VARIANT_RESULTS[selected_variant]['summary']['awmae']),
    'm_frozen_required': True,
    'no_gt_used': True,
}
with open(SUM_DIR / 'final_decision.json', 'w', encoding='utf-8') as f:
    json.dump(final_decision, f, indent=2, ensure_ascii=False)
final_decision_df = pd.DataFrame([{'key': k, 'value': str(v)} for k, v in final_decision.items()])
final_decision_df.to_csv(SUM_DIR / 'final_decision.csv', index=False)
display(final_decision_df)
log_saved(SUM_DIR / 'final_decision.json')
log_saved(SUM_DIR / 'final_decision.csv')

# %% [markdown]
# # 27. Final Model Training dan Test Inference
#
# Bagian ini menjalankan inference test. M prediction selalu diambil dari D0 EXP11C. Jika selected variant
# bukan D0, overlay hanya diterapkan ke test W sesuai config selected. Jika fallback D0, submission tetap
# setara EXP11C baseline.

# %%
log_section('EXP11D final model training and test inference')
full_train_feature_df = train_feature_all.merge(y_full_train, on='match_id', how='left', validate='one_to_one')
full_train_country_df = add_country_prior_features(full_train_feature_df)
full_base_models = train_final_gender_models_by_head_config(full_train_feature_df, valid_sup_full, final_selected, BASE_HEAD_CONFIG, 1200)
full_repair_models = train_final_gender_models_by_head_config(full_train_feature_df, valid_sup_full, final_selected, REPAIR_HEAD_CONFIG, 1200)

test_raw_base = predict_gender_split_raw_outputs_with_config(test_feature_full, full_base_models)
test_raw_repair = predict_gender_split_raw_outputs_with_config(test_feature_full, full_repair_models)
test_raw_gate = apply_tournament_gated_outcome(test_raw_base, test_raw_repair, selected_gate_cfg['gate_map'], default_alpha=selected_gate_cfg['default_alpha'])
test_pred_exp09d = decode_directional_tail_batch(
    test_raw_gate.drop(columns=TAIL_COLS, errors='ignore'),
    test_raw_base[['match_id'] + TAIL_COLS],
    selected_exp09d_decoder,
    scoreline_prior_lookup[int(selected_exp09d_decoder['MAX_GOALS'])],
)

full_ordinal_models, _, _ = train_ordinal_models(
    full_train_feature_df, valid_sup_full, final_selected, exp05a_feature_cols, exp05a_cat_features,
    max_goal=MAX_ORDINAL_GOALS, n_estimators=ORDINAL_N_ESTIMATORS,
)
test_feature_for_pmf = test_raw_gate[['match_id']].merge(test_feature_full, on='match_id', how='left', validate='one_to_one')
test_ge_a = predict_ordinal_ge_probs(test_feature_for_pmf, full_ordinal_models, side='a', max_goal=MAX_ORDINAL_GOALS)
test_ge_b = predict_ordinal_ge_probs(test_feature_for_pmf, full_ordinal_models, side='b', max_goal=MAX_ORDINAL_GOALS)
test_pmf_ord_a = ge_probs_to_pmf(test_ge_a)
test_pmf_ord_b = ge_probs_to_pmf(test_ge_b)
test_pmf_pois_a = poisson_pmf_from_lambda(test_raw_gate['pred_goal_a_cont'], max_goal=MAX_ORDINAL_GOALS)
test_pmf_pois_b = poisson_pmf_from_lambda(test_raw_gate['pred_goal_b_cont'], max_goal=MAX_ORDINAL_GOALS)
exp11c_cfg = {'blend_ordinal': 0.90, 'temperature': 1.0, 'smoothing_epsilon': 0.0, 'mean_lift': 0.0}
test_pmf_cal_a = calibrate_goal_pmf(test_pmf_ord_a, test_pmf_pois_a, exp11c_cfg)
test_pmf_cal_b = calibrate_goal_pmf(test_pmf_ord_b, test_pmf_pois_b, exp11c_cfg)

test_pred_d0 = decode_exp11c_policy(test_raw_gate, test_pred_exp09d, test_pmf_cal_a, test_pmf_cal_b, 'D0_EXP11C_baseline')
test_pred_d0_match = test_raw_gate[['match_id','gender','tournament','tournament_structure']].merge(test_pred_d0[['match_id','pred_team_a_goals','pred_team_b_goals']], on='match_id', how='left', validate='one_to_one')
test_pred_d0_match.to_csv(PRED_DIR / 'test_pred_d0_exp11c.csv', index=False)
log_saved(PRED_DIR / 'test_pred_d0_exp11c.csv')

# Default selected prediction is D0 EXP11C baseline.
test_pred_selected = test_pred_d0_match.copy()

if selected_variant != 'D0_EXP11C_baseline':
    cfg = selected_config
    use_country = bool(cfg.get('use_country_prior', False))
    use_pseudo = bool(cfg.get('use_pseudo_label', False))
    blend = float(cfg.get('blend_ordinal', 0.75))

    # Important consistency patch:
    # D1 validation uses the existing EXP11C ordinal PMF source. Therefore final D1 inference
    # must also use full_ordinal_models/test_pmf_ord, not a newly trained W-only model.
    if str(selected_variant).startswith('D1_'):
        test_w_mask = test_raw_gate['gender'].astype(str).str.upper().eq('W').to_numpy()
        test_w_pmf_ord_a = test_pmf_ord_a[test_w_mask]
        test_w_pmf_ord_b = test_pmf_ord_b[test_w_mask]

        test_w_pred_overlay = decode_w_overlay_from_pmfs(
            test_raw_gate,
            test_pred_d0_match,
            test_w_pmf_ord_a,
            test_w_pmf_ord_b,
            blend_ordinal=blend,
            variant_name=selected_variant,
        )
        assert_m_predictions_frozen(test_w_pred_overlay, test_pred_d0_match)
        test_pred_selected = test_w_pred_overlay

    else:
        coverage = float(cfg.get('pseudo_coverage', 0.10))
        pseudo_weight = float(cfg.get('pseudo_weight', 0.15))

        if use_country:
            w_features, w_cats = make_w_feature_set(use_country_prior=True)
            train_frame = full_train_country_df
            valid_frame = valid_sup_country
            test_frame = test_feature_country
        else:
            w_features, w_cats = make_w_feature_set(use_country_prior=False)
            train_frame = full_train_feature_df
            valid_frame = valid_sup_full
            test_frame = test_feature_full

        pseudo_df_final = None
        if use_pseudo:
            test_w_source = test_raw_gate[test_raw_gate['gender'].astype(str).str.upper().eq('W')].reset_index(drop=True)
            test_w_pred_for_pl = test_pred_d0[test_pred_d0['match_id'].isin(test_w_source['match_id'])].copy()
            pmf_a_w = test_pmf_cal_a[test_raw_gate['gender'].astype(str).str.upper().eq('W').to_numpy()]
            pmf_b_w = test_pmf_cal_b[test_raw_gate['gender'].astype(str).str.upper().eq('W').to_numpy()]
            selected_pl = select_pseudo_labels(test_w_source, test_w_pred_for_pl, pmf_a_w, pmf_b_w, coverage=coverage)
            pseudo_base = selected_pl[['match_id','y_goal_a','y_goal_b','pseudo_confidence','top1_scoreline_prob','top1_top2_margin','outcome_confidence']]
            pseudo_df_final = pseudo_base.merge(test_frame, on='match_id', how='left', validate='one_to_one')

        w_final_models, _ = fit_w_only_ordinal_models(
            train_frame,
            valid_frame,
            w_features,
            w_cats,
            use_country_prior=use_country,
            pseudo_df=pseudo_df_final,
            pseudo_weight=pseudo_weight,
            max_goal=MAX_ORDINAL_GOALS,
            n_estimators=ORDINAL_N_ESTIMATORS,
        )

        test_w_mask = test_raw_gate['gender'].astype(str).str.upper().eq('W').to_numpy()
        test_w_feature = test_raw_gate.loc[test_w_mask, ['match_id']].merge(test_frame, on='match_id', how='left', validate='one_to_one')
        w_pmf_a, w_pmf_b = predict_w_ordinal_pmfs(test_w_feature, w_final_models, max_goal=MAX_ORDINAL_GOALS)

        test_w_pred_overlay = decode_w_overlay_from_pmfs(
            test_raw_gate,
            test_pred_d0_match,
            w_pmf_a,
            w_pmf_b,
            blend_ordinal=blend,
            variant_name=selected_variant,
        )
        assert_m_predictions_frozen(test_w_pred_overlay, test_pred_d0_match)
        test_pred_selected = test_w_pred_overlay
test_pred_selected['selected_variant'] = selected_variant
test_pred_selected.to_csv(PRED_DIR / 'test_pred_selected.csv', index=False)
test_pred_selected.to_csv(PRED_DIR / 'test_pred_exp11d_best_safe.csv', index=False)
log_saved(PRED_DIR / 'test_pred_selected.csv')
log_saved(PRED_DIR / 'test_pred_exp11d_best_safe.csv')

# %% [markdown]
# # 28. Submission Mapping dan Pair Consistency
#
# Bagian ini mengubah prediksi match-level ke row-level submission. Validasi wajib: id order sama dengan
# sample submission, tidak missing, tidak duplicate, goal integer non-negative, dan pair consistency pass.

# %%
sample_sub = sample_submission.copy()


def match_predictions_to_submission(test_rows: pd.DataFrame, pred_match: pd.DataFrame) -> pd.DataFrame:
    """
    Convert match-level prediction back to row-level submission.
    Row A gets pred_team_a_goals vs pred_team_b_goals.
    Row B gets mirrored pred_team_b_goals vs pred_team_a_goals.
    """
    required_pred = ['match_id', 'pred_team_a_goals', 'pred_team_b_goals']
    missing_pred = [c for c in required_pred if c not in pred_match.columns]
    if missing_pred:
        raise KeyError(f'pred_match missing columns: {missing_pred}')

    required_match = ['match_id', 'row_id_a', 'row_id_b']
    missing_match = [c for c in required_match if c not in test_match_base.columns]
    if missing_match:
        raise KeyError(f'test_match_base missing columns: {missing_match}')

    pred = pred_match[required_pred].copy()
    pred['pred_team_a_goals'] = pd.to_numeric(pred['pred_team_a_goals'], errors='coerce').round().astype(int)
    pred['pred_team_b_goals'] = pd.to_numeric(pred['pred_team_b_goals'], errors='coerce').round().astype(int)
    pred['pred_team_a_goals'] = pred['pred_team_a_goals'].clip(lower=0)
    pred['pred_team_b_goals'] = pred['pred_team_b_goals'].clip(lower=0)

    match_map = test_match_base[['match_id', 'row_id_a', 'row_id_b']].merge(
        pred,
        on='match_id',
        how='left',
        validate='one_to_one',
    )

    if match_map[['pred_team_a_goals', 'pred_team_b_goals']].isna().any().any():
        bad = match_map.loc[
            match_map[['pred_team_a_goals', 'pred_team_b_goals']].isna().any(axis=1),
            'match_id',
        ].head().tolist()
        raise RuntimeError(f'Missing match predictions for test match_id examples: {bad}')

    rows_a = pd.DataFrame({
        'id': match_map['row_id_a'],
        'team_goals': match_map['pred_team_a_goals'],
        'opp_goals': match_map['pred_team_b_goals'],
    })

    rows_b = pd.DataFrame({
        'id': match_map['row_id_b'],
        'team_goals': match_map['pred_team_b_goals'],
        'opp_goals': match_map['pred_team_a_goals'],
    })

    long_pred = pd.concat([rows_a, rows_b], ignore_index=True)
    long_pred['id_key'] = long_pred['id'].astype(str)

    sample_id_col = 'id' if 'id' in sample_sub.columns else 'Id'
    out = sample_sub[[sample_id_col]].copy()
    out = out.rename(columns={sample_id_col: 'id'})
    out['id_key'] = out['id'].astype(str)

    out = out.merge(
        long_pred[['id_key', 'team_goals', 'opp_goals']],
        on='id_key',
        how='left',
        validate='one_to_one',
    ).drop(columns=['id_key'])

    if out[['team_goals', 'opp_goals']].isna().any().any():
        missing_ids = out.loc[out[['team_goals', 'opp_goals']].isna().any(axis=1), 'id'].head().tolist()
        raise RuntimeError(f'Submission has missing predictions for id examples: {missing_ids}')

    out['team_goals'] = pd.to_numeric(out['team_goals'], errors='raise').astype(int).clip(lower=0)
    out['opp_goals'] = pd.to_numeric(out['opp_goals'], errors='raise').astype(int).clip(lower=0)

    return out[['id', 'team_goals', 'opp_goals']]


def check_pair_consistency(submission_df: pd.DataFrame, test_rows: pd.DataFrame) -> bool:
    """
    Check that for every match_id, the two mirrored rows are consistent:
    row A team_goals == row B opp_goals
    row A opp_goals == row B team_goals
    """
    test_id_col = 'Id' if 'Id' in test_rows.columns else 'id'

    id_map = test_rows[[test_id_col, 'match_id']].copy()
    id_map = id_map.rename(columns={test_id_col: 'id'})
    id_map['id_key'] = id_map['id'].astype(str)

    sub = submission_df.copy()
    sub['id_key'] = sub['id'].astype(str)

    merged = sub.merge(
        id_map[['id_key', 'match_id']],
        on='id_key',
        how='left',
        validate='one_to_one',
    )

    if merged['match_id'].isna().any():
        return False

    for _, g in merged.groupby('match_id'):
        if len(g) != 2:
            return False
        r0 = g.iloc[0]
        r1 = g.iloc[1]
        if int(r0['team_goals']) != int(r1['opp_goals']):
            return False
        if int(r0['opp_goals']) != int(r1['team_goals']):
            return False

    return True


submission_exp11d = match_predictions_to_submission(test_raw, test_pred_selected)
submission_exp11d.to_csv(SUB_DIR / 'submission_exp11d_best_safe.csv', index=False)

if selected_variant == 'D0_EXP11C_baseline':
    submission_exp11d.to_csv(SUB_DIR / 'submission_exp11d_fallback_exp11c.csv', index=False)

sample_id_col = 'id' if 'id' in sample_sub.columns else 'Id'
submission_checks = []
submission_checks.append({'check': 'shape_matches_sample', 'passed': tuple(submission_exp11d.shape) == tuple(sample_sub.shape)})
submission_checks.append({'check': 'id_order_matches_sample', 'passed': submission_exp11d['id'].astype(str).tolist() == sample_sub[sample_id_col].astype(str).tolist()})
submission_checks.append({'check': 'no_missing', 'passed': not submission_exp11d[['team_goals', 'opp_goals']].isna().any().any()})
submission_checks.append({'check': 'no_duplicate_id', 'passed': not submission_exp11d['id'].duplicated().any()})
submission_checks.append({'check': 'non_negative_goals', 'passed': bool((submission_exp11d[['team_goals', 'opp_goals']] >= 0).all().all())})
submission_checks.append({'check': 'pair_consistency', 'passed': check_pair_consistency(submission_exp11d, test_raw)})

submission_check_df = pd.DataFrame(submission_checks)
submission_check_df.to_csv(SUM_DIR / 'submission_check.csv', index=False)
display(submission_check_df)

for _, r in submission_check_df.iterrows():
    log_check(str(r['check']), bool(r['passed']))

if not bool(submission_check_df['passed'].all()):
    raise RuntimeError('Submission validation failed.')

log_saved(SUB_DIR / 'submission_exp11d_best_safe.csv')

print('[TEST INFERENCE SUMMARY]', flush=True)
log_info(f'selected_variant: {selected_variant}')
log_info(f'n_test_matches: {len(test_pred_selected):,}')
log_info(f"mean_pred_total: {float((test_pred_selected['pred_team_a_goals'] + test_pred_selected['pred_team_b_goals']).mean()):.4f}")
log_info(f"max_pred_goal: {int(test_pred_selected[['pred_team_a_goals','pred_team_b_goals']].max().max())}")

top_test_scores = (
    test_pred_selected
    .assign(scoreline=lambda d: d['pred_team_a_goals'].astype(int).astype(str) + '-' + d['pred_team_b_goals'].astype(int).astype(str))
    ['scoreline']
    .value_counts()
    .head(10)
    .reset_index()
)
top_test_scores.columns = ['scoreline', 'count']
display(top_test_scores)
# %% [markdown]
# # 29. Runtime Summary dan Catatan Akhir
#
# Bagian ini menyimpan runtime summary dan menampilkan ringkasan akhir eksperimen. Output akhir harus
# jelas: selected variant, decision code, path submission, dan status pair consistency.

# %%
SECTION_TIMES['total'] = time.time() - NOTEBOOK_START_TIME
runtime_summary_df = pd.DataFrame([{'section': k, 'runtime_sec': v} for k, v in SECTION_TIMES.items()])
runtime_summary_df.to_csv(SUM_DIR / 'notebook_runtime_summary.csv', index=False)
display(runtime_summary_df)

final_summary_df = pd.DataFrame([
    {'item': 'experiment', 'value': 'EXP11D'},
    {'item': 'selected_variant', 'value': selected_variant},
    {'item': 'decision_code', 'value': decision_code},
    {'item': 'submission_path', 'value': str(SUB_DIR / 'submission_exp11d_best_safe.csv')},
    {'item': 'pair_consistency', 'value': str(bool(submission_check_df.loc[submission_check_df['check'].eq('pair_consistency'), 'passed'].iloc[0]))},
])
display(final_summary_df)
log_saved(SUM_DIR / 'notebook_runtime_summary.csv')
log_result('EXP11D completed.')
