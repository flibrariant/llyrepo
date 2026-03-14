"""
日本上場S&P500 ETF統計比較分析 v3
方針: 株式分割遷移日のリターンを除外し、クリーンな日次リターンで比較

対象: 1655, 2558, 2633, 1547
データソース: yfinance
"""

import warnings
warnings.filterwarnings('ignore')

import yfinance as yf
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

OUTPUT_DIR = '/home/like_rapid/GT-SOAR/reports/sp500_etf_compare'
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 120

ETFS = {
    '1655.T': '1655 iShares',
    '2558.T': '2558 MAXIS',
    '2633.T': '2633 NEXT FUNDS',
    '1547.T': '1547 Listed Index',
}

# 株式分割で壊れた日付を除外リストに
BAD_DATES = {
    '1655.T': ['2017-09-27', '2017-09-28', '2017-09-29',
               '2022-02-07', '2022-02-08', '2022-02-09', '2022-02-10', '2022-02-14'],
    '2633.T': ['2021-03-30', '2021-03-31', '2021-04-01',
               '2023-12-05', '2023-12-06', '2023-12-07', '2023-12-08', '2023-12-11'],
}

# ============================================================
# 1. データ取得
# ============================================================
print("=" * 70)
print("1. データ取得（壊れた日付のリターンを除外）")
print("=" * 70)

def get_clean_returns(ticker_str):
    """クリーンな日次リターン系列を取得"""
    t = yf.Ticker(ticker_str)
    hist = t.history(period='max')
    hist.index = hist.index.tz_localize(None)
    prices = hist['Close'].copy()
    volume = hist['Volume'].copy()
    divs = t.dividends
    if len(divs) > 0:
        divs.index = divs.index.tz_localize(None)

    # 日次リターン
    ret = prices.pct_change()

    # 壊れた日を除外
    if ticker_str in BAD_DATES:
        bad = pd.to_datetime(BAD_DATES[ticker_str])
        mask = ~ret.index.isin(bad)
        ret = ret[mask]
        # ±50%超の日もスパイクとして除外
        spike = abs(ret) > 0.5
        if spike.any():
            print(f"  追加スパイク除外: {ret[spike].index.tolist()}")
            ret = ret[~spike]

    ret = ret.dropna()

    # リターンからトータルリターン指数を復元
    tr = (1 + ret).cumprod() * 100

    return prices, tr, ret, volume, divs

data = {}
for ticker, name in ETFS.items():
    print(f"\n{name} ({ticker}) データ取得中...")
    price, tr, ret, vol, divs = get_clean_returns(ticker)
    data[ticker] = {'price': price, 'tr': tr, 'ret': ret, 'vol': vol, 'divs': divs, 'name': name}
    print(f"  全期間: {ret.index[0].date()} ~ {ret.index[-1].date()}, {len(ret)}日")
    print(f"  日次リターン std: {ret.std()*100:.2f}%")

# 共通期間
all_starts = [data[t]['ret'].index[0] for t in ETFS]
common_start = max(all_starts)
common_end = min([data[t]['ret'].index[-1] for t in ETFS])

# 全ETFの共通日付
common_dates = None
for ticker in ETFS:
    idx = data[ticker]['ret'][common_start:common_end].index
    common_dates = idx if common_dates is None else common_dates.intersection(idx)

print(f"\n共通期間: {common_start.date()} ~ {common_end.date()}")
print(f"共通営業日数: {len(common_dates)}")

# 共通期間のリターンとTR指数
ret_common = {}
tr_common = {}
for ticker in ETFS:
    r = data[ticker]['ret'].loc[common_dates]
    ret_common[ticker] = r
    tr = (1 + r).cumprod() * 100
    tr_common[ticker] = tr

# サニティチェック
print("\n[サニティチェック: 日次リターン相関]")
corr = pd.DataFrame({ETFS[t]: ret_common[t] for t in ETFS}).corr()
print(corr.round(4).to_string())

# 相関が低い場合は警告
for t1 in ETFS:
    for t2 in ETFS:
        if t1 < t2:
            c = corr.loc[ETFS[t1], ETFS[t2]]
            if c < 0.9:
                print(f"  WARNING: {ETFS[t1]} vs {ETFS[t2]} 相関 {c:.4f} が低い")

# ============================================================
# 2. 基本統計量
# ============================================================
print("\n" + "=" * 70)
print("2. 基本統計量（共通期間、rf=0%）")
print("=" * 70)

def calc_stats(tr_series, ret_series):
    n_days = len(ret_series)
    n_years = n_days / 252
    total_return = tr_series.iloc[-1] / tr_series.iloc[0]
    cagr = total_return ** (1 / n_years) - 1
    vol = ret_series.std() * np.sqrt(252)
    cummax = tr_series.cummax()
    drawdown = (tr_series - cummax) / cummax
    mdd = drawdown.min()
    sharpe = ret_series.mean() / ret_series.std() * np.sqrt(252) if ret_series.std() > 0 else 0
    downside = ret_series[ret_series < 0]
    downside_std = np.sqrt((downside ** 2).mean()) * np.sqrt(252)
    sortino = ret_series.mean() * 252 / downside_std if downside_std > 0 else 0
    calmar = cagr / abs(mdd) if mdd != 0 else 0
    monthly = tr_series.resample('ME').last().pct_change().dropna()
    return {
        'cagr': cagr * 100,
        'vol': vol * 100,
        'mdd': mdd * 100,
        'sharpe': sharpe,
        'sortino': sortino,
        'calmar': calmar,
        'total_ret': (total_return - 1) * 100,
        'monthly_win': (monthly > 0).mean() * 100,
        'monthly_mean': monthly.mean() * 100,
        'monthly_skew': monthly.skew(),
        'monthly_kurt': monthly.kurtosis(),
    }

all_stats = {}
for ticker in ETFS:
    all_stats[ticker] = calc_stats(tr_common[ticker], ret_common[ticker])

metrics = [
    ('累積リターン(%)', 'total_ret', '{:.2f}'),
    ('CAGR(%)', 'cagr', '{:.2f}'),
    ('年率Vol(%)', 'vol', '{:.2f}'),
    ('最大DD(%)', 'mdd', '{:.2f}'),
    ('Sharpe', 'sharpe', '{:.3f}'),
    ('Sortino', 'sortino', '{:.3f}'),
    ('Calmar', 'calmar', '{:.3f}'),
    ('月次勝率(%)', 'monthly_win', '{:.1f}'),
    ('月次平均(%)', 'monthly_mean', '{:.3f}'),
    ('月次歪度', 'monthly_skew', '{:.3f}'),
    ('月次尖度', 'monthly_kurt', '{:.3f}'),
]

header = f"{'指標':<20}"
for ticker in ETFS:
    header += f" {ETFS[ticker]:>16}"
print(header)
print("-" * (20 + 17 * len(ETFS)))

for label, key, fmt in metrics:
    row = f"{label:<20}"
    for ticker in ETFS:
        row += f" {fmt.format(all_stats[ticker][key]):>16}"
    print(row)

# ============================================================
# 3. ペアワイズ統計検定
# ============================================================
print("\n" + "=" * 70)
print("3. ペアワイズ統計検定")
print("=" * 70)

tickers = list(ETFS.keys())

print("\n--- 日次リターン差 ---")
print(f"{'ペア':<40} {'年率差':>10} {'t値':>8} {'p値(t)':>8} {'p値(W)':>8}")
print("-" * 78)
for i, t1 in enumerate(tickers):
    for t2 in tickers[i+1:]:
        diff = (ret_common[t1] - ret_common[t2]).dropna()
        t_stat, t_pval = stats.ttest_1samp(diff, 0)
        ann = diff.mean() * 252 * 100
        diff_nz = diff[diff != 0]
        if len(diff_nz) > 10:
            _, w_pval = stats.wilcoxon(diff_nz)
        else:
            w_pval = np.nan
        sig = '*' if t_pval < 0.05 else ''
        print(f"{ETFS[t1]} vs {ETFS[t2]:<20} {ann:>+9.2f}% {t_stat:>+7.3f} {t_pval:>7.4f} {w_pval:>7.4f} {sig}")

# 週次・月次
print("\n--- 週次リターン差 t検定 ---")
for i, t1 in enumerate(tickers):
    for t2 in tickers[i+1:]:
        w1 = tr_common[t1].resample('W').last().pct_change().dropna()
        w2 = tr_common[t2].resample('W').last().pct_change().dropna()
        ci = w1.index.intersection(w2.index)
        diff = (w1.loc[ci] - w2.loc[ci]).dropna()
        t_stat, p_val = stats.ttest_1samp(diff, 0)
        ann = diff.mean() * 52 * 100
        sig = '*' if p_val < 0.05 else ''
        print(f"  {ETFS[t1]} vs {ETFS[t2]}: 年率={ann:+.2f}%, t={t_stat:+.3f}, p={p_val:.4f} {sig}")

# ============================================================
# 4. Sharpe差検定 & ブートストラップ
# ============================================================
print("\n" + "=" * 70)
print("4. Sharpe差検定 & ブートストラップ")
print("=" * 70)

# JK-Memmel
print("\n[Jobson-Korkie-Memmel検定]")
for i, t1 in enumerate(tickers):
    for t2 in tickers[i+1:]:
        r1, r2 = ret_common[t1].values, ret_common[t2].values
        mu1, mu2 = r1.mean(), r2.mean()
        s1, s2 = r1.std(), r2.std()
        rho = np.corrcoef(r1, r2)[0, 1]
        n_obs = len(r1)
        sr1, sr2 = mu1/s1, mu2/s2
        theta = (1/n_obs) * (2*(1-rho) + 0.5*(sr1**2 + sr2**2 - 2*sr1*sr2*rho))
        z = (sr1 - sr2) / np.sqrt(theta) if theta > 0 else 0
        p_val = 2 * (1 - stats.norm.cdf(abs(z)))
        sig = '*有意(5%)' if p_val < 0.05 else '非有意'
        print(f"  {ETFS[t1]} vs {ETFS[t2]}: z={z:+.4f}, p={p_val:.4f} {sig}")

# ブートストラップ
np.random.seed(42)
N_BOOT = 10000
BLOCK_SIZE = 21

def sharpe_ratio(r):
    return r.mean() / r.std() * np.sqrt(252) if len(r) > 1 and r.std() > 0 else 0

def cagr_from_ret(r):
    cum = (1 + r).prod()
    ny = len(r) / 252
    return cum ** (1/ny) - 1 if ny > 0 and cum > 0 else 0

def mdd_from_ret(r):
    cum = (1 + r).cumprod()
    return (cum / cum.cummax() - 1).min()

ret_df = pd.DataFrame({t: ret_common[t] for t in ETFS})
n = len(ret_df)

boot = {t: {'sharpe': [], 'cagr': [], 'mdd': []} for t in ETFS}
for _ in range(N_BOOT):
    n_blocks = n // BLOCK_SIZE + 1
    starts = np.random.randint(0, n - BLOCK_SIZE + 1, size=n_blocks)
    indices = np.concatenate([np.arange(s, s + BLOCK_SIZE) for s in starts])[:n]
    sample = ret_df.iloc[indices]
    for t in ETFS:
        boot[t]['sharpe'].append(sharpe_ratio(sample[t]))
        boot[t]['cagr'].append(cagr_from_ret(sample[t]))
        boot[t]['mdd'].append(mdd_from_ret(sample[t]))

for t in ETFS:
    for k in boot[t]:
        boot[t][k] = np.array(boot[t][k])

print(f"\n[ブロックブートストラップ n={N_BOOT}, block={BLOCK_SIZE}日]")
# 主要ペアのみ表示
key_pairs = [(0,1), (0,2), (0,3), (1,3)]
for i1, i2 in key_pairs:
    t1, t2 = tickers[i1], tickers[i2]
    ds = boot[t1]['sharpe'] - boot[t2]['sharpe']
    dc = boot[t1]['cagr'] - boot[t2]['cagr']
    dm = boot[t1]['mdd'] - boot[t2]['mdd']
    print(f"\n  {ETFS[t1]} vs {ETFS[t2]}:")
    print(f"    Sharpe差: mean={ds.mean():.4f}, 95%CI=[{np.percentile(ds,2.5):.4f}, {np.percentile(ds,97.5):.4f}], {ETFS[t1]}優位={((ds>0).mean()*100):.1f}%")
    print(f"    CAGR差:   mean={dc.mean()*100:.2f}%, 95%CI=[{np.percentile(dc,2.5)*100:.2f}%, {np.percentile(dc,97.5)*100:.2f}%]")

# ============================================================
# 5. ローリング分析
# ============================================================
print("\n" + "=" * 70)
print("5. ローリング分析")
print("=" * 70)

base = '1655.T'
for window in [126, 252]:
    print(f"\n--- ローリング{window}日 ---")
    rolling_sharpe = {}
    for t in ETFS:
        rolling_sharpe[t] = ret_common[t].rolling(window).apply(
            lambda x: x.mean()/x.std()*np.sqrt(252), raw=True).dropna()
    for t in ETFS:
        if t == base:
            continue
        common_r = rolling_sharpe[base].index.intersection(rolling_sharpe[t].index)
        diff = rolling_sharpe[base].loc[common_r] - rolling_sharpe[t].loc[common_r]
        pct_win = (diff > 0).mean() * 100
        print(f"  1655 vs {ETFS[t]}: 1655優位 {pct_win:.1f}%, mean {diff.mean():+.3f}")

# ============================================================
# 6. トラッキング差異
# ============================================================
print("\n" + "=" * 70)
print("6. トラッキング差異")
print("=" * 70)

print(f"\nトラッキングエラー（年率）:")
for i, t1 in enumerate(tickers):
    for t2 in tickers[i+1:]:
        te = (ret_common[t1] - ret_common[t2]).std() * np.sqrt(252)
        print(f"  {ETFS[t1]} vs {ETFS[t2]}: TE = {te*100:.3f}%")

# ============================================================
# 7. コスト影響の推定
# ============================================================
print("\n" + "=" * 70)
print("7. コスト影響の推定")
print("=" * 70)

print("""
信託報酬の年間差:
  2633(0.077%) vs 1655(0.0825%): 差 0.0055% → 100万円あたり年55円
  1655(0.0825%) vs 2558(0.0858%): 差 0.0033% → 100万円あたり年33円
  1547(0.165%) vs 他3本: 約0.08~0.09%の差 → 100万円あたり年800~900円

→ 信託報酬差は経済的にほぼ無視できる水準
→ 1547だけが2倍のコストで年間約0.09%不利
""")

# ============================================================
# 8. 図の作成
# ============================================================
print("=" * 70)
print("8. 図の作成")
print("=" * 70)

COLORS = {'1655.T': '#1565C0', '2558.T': '#2E7D32', '2633.T': '#E65100', '1547.T': '#6A1B9A'}

# 図1: TR推移
fig, ax = plt.subplots(figsize=(14, 7))
for t in ETFS:
    ax.plot(tr_common[t].index, tr_common[t].values, label=ETFS[t], linewidth=1.5, color=COLORS[t])
ax.set_title('S&P500 ETF Total Return (Common Period, 100 Start)', fontsize=14)
ax.set_ylabel('Index')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.xticks(rotation=45); plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/01_total_return.png'); plt.close()
print("  01_total_return.png")

# 図2: DD
fig, ax = plt.subplots(figsize=(14, 6))
for t in ETFS:
    dd = (tr_common[t] - tr_common[t].cummax()) / tr_common[t].cummax()
    ax.plot(dd.index, dd.values*100, label=ETFS[t], linewidth=1, color=COLORS[t], alpha=0.8)
ax.set_title('Drawdown Comparison (%)', fontsize=14)
ax.set_ylabel('Drawdown (%)')
ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45); plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/02_drawdown.png'); plt.close()
print("  02_drawdown.png")

# 図3: ローリングSharpe差
others = [t for t in tickers if t != base]
fig, axes = plt.subplots(3, 1, figsize=(14, 12))
for i, t in enumerate(others):
    ax = axes[i]
    for window, lw, al in [(252, 1.5, 0.9), (126, 0.8, 0.5)]:
        rs1 = ret_common[base].rolling(window).apply(lambda x: x.mean()/x.std()*np.sqrt(252), raw=True)
        rs2 = ret_common[t].rolling(window).apply(lambda x: x.mean()/x.std()*np.sqrt(252), raw=True)
        common_r = rs1.dropna().index.intersection(rs2.dropna().index)
        diff = rs1.loc[common_r] - rs2.loc[common_r]
        ax.plot(diff.index, diff.values, linewidth=lw, alpha=al, label=f'{window}d', color=COLORS[t] if window==252 else '#999')
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_title(f'Rolling Sharpe Diff: 1655 - {ETFS[t]}', fontsize=12)
    ax.set_ylabel('Sharpe Diff'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/03_rolling_sharpe_diff.png'); plt.close()
print("  03_rolling_sharpe_diff.png")

# 図4: ブートストラップ
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, t in enumerate(others):
    diff_boot = boot[base]['sharpe'] - boot[t]['sharpe']
    ax = axes[i]
    ax.hist(diff_boot, bins=80, color=COLORS[t], alpha=0.6, edgecolor='white')
    ax.axvline(0, color='red', linewidth=2, linestyle='--')
    ax.axvline(np.percentile(diff_boot,2.5), color='blue', linewidth=1, linestyle=':')
    ax.axvline(np.percentile(diff_boot,97.5), color='blue', linewidth=1, linestyle=':')
    ax.set_title(f'Bootstrap Sharpe: 1655 - {ETFS[t]}')
    ax.set_xlabel('Sharpe Diff')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/04_bootstrap_sharpe.png'); plt.close()
print("  04_bootstrap_sharpe.png")

# 図5: 累積差
fig, ax = plt.subplots(figsize=(14, 6))
for t in others:
    d = tr_common[base] - tr_common[t]
    ax.plot(d.index, d.values, label=f'1655 - {ETFS[t]}', linewidth=1.5, color=COLORS[t])
ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax.set_title('Cumulative Return Difference vs 1655 (Index Points)', fontsize=14)
ax.set_ylabel('Diff (pts)'); ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45); plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/05_cumulative_diff.png'); plt.close()
print("  05_cumulative_diff.png")

# 図6: 年次リターン
fig, ax = plt.subplots(figsize=(14, 7))
yearly = {t: tr_common[t].resample('YE').last().pct_change().dropna() for t in ETFS}
cy = yearly[tickers[0]].index
for t in tickers[1:]:
    cy = cy.intersection(yearly[t].index)
x = np.arange(len(cy)); w = 0.2
for i, t in enumerate(ETFS):
    vals = yearly[t].loc[cy].values * 100
    ax.bar(x + i*w - 1.5*w, vals, w, label=ETFS[t], color=COLORS[t])
ax.set_xticks(x); ax.set_xticklabels([d.strftime('%Y') for d in cy])
ax.set_title('Annual Total Return (%)', fontsize=14)
ax.set_ylabel('Return (%)'); ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y'); ax.axhline(0, color='black', linewidth=0.8)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/06_annual_returns.png'); plt.close()
print("  06_annual_returns.png")

# 図7: リターン差ヒストグラム
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, t in enumerate(others):
    diff = (ret_common[base] - ret_common[t]).dropna() * 100
    ax = axes[i]
    ax.hist(diff, bins=100, color=COLORS[t], alpha=0.6, edgecolor='white')
    ax.axvline(0, color='red', linewidth=1.5, linestyle='--')
    ax.axvline(diff.mean(), color='blue', linewidth=1.5, label=f'mean={diff.mean():.4f}%')
    ax.set_title(f'Daily Return Diff: 1655 - {ETFS[t]} (%)')
    ax.set_xlabel('Return Diff (%)'); ax.legend()
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/07_return_diff_dist.png'); plt.close()
print("  07_return_diff_dist.png")

# 図8: 2558 vs 1547 長期比較（最もクリーンなペア）
long_start = max(data['2558.T']['ret'].index[0], data['1547.T']['ret'].index[0])
long_dates = data['2558.T']['ret'][long_start:].index.intersection(data['1547.T']['ret'][long_start:].index)
tr_2558_l = (1 + data['2558.T']['ret'].loc[long_dates]).cumprod() * 100
tr_1547_l = (1 + data['1547.T']['ret'].loc[long_dates]).cumprod() * 100

fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(tr_2558_l.index, tr_2558_l.values, label='2558 MAXIS', linewidth=1.5, color=COLORS['2558.T'])
ax.plot(tr_1547_l.index, tr_1547_l.values, label='1547 Listed Index', linewidth=1.5, color=COLORS['1547.T'])
ax.set_title(f'Long-term: 2558 vs 1547 ({long_dates[0].strftime("%Y/%m")}~)', fontsize=14)
ax.set_ylabel('Total Return Index (100 Start)')
ax.legend(fontsize=12); ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45); plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/08_long_term_2558_1547.png'); plt.close()
print("  08_long_term_2558_1547.png")

s_2558 = calc_stats(tr_2558_l, data['2558.T']['ret'].loc[long_dates])
s_1547 = calc_stats(tr_1547_l, data['1547.T']['ret'].loc[long_dates])
print(f"\n[長期比較: 2558 vs 1547 ({long_dates[0].date()} ~ {long_dates[-1].date()}, {len(long_dates)/252:.1f}年)]")
print(f"{'指標':<20} {'2558':>12} {'1547':>12} {'差':>12}")
print("-" * 58)
for label, key, fmt in metrics:
    v1, v2 = s_2558[key], s_1547[key]
    print(f"{label:<20} {fmt.format(v1):>12} {fmt.format(v2):>12} {fmt.format(v1-v2):>12}")
diff_l = (data['2558.T']['ret'].loc[long_dates] - data['1547.T']['ret'].loc[long_dates]).dropna()
t_l, p_l = stats.ttest_1samp(diff_l, 0)
print(f"t検定: t={t_l:.4f}, p={p_l:.4f}, 年率差={diff_l.mean()*252*100:+.2f}%")

print("\n" + "=" * 70)
print("全出力完了")
print(f"出力先: {OUTPUT_DIR}/")
print("=" * 70)
