"""
日本上場S&P500 ETF（為替ヘッジなし）統計比較分析
対象: 1655, 2558, 2633, 1547 — 全て同一指数(S&P500)、為替ヘッジなし

データソース: yfinance
欠損処理: 共通取引日のintersectionのみ使用
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

# ============================================================
# 1. データ取得・トータルリターン系列構築
# ============================================================
print("=" * 70)
print("1. データ取得")
print("=" * 70)

def get_total_return_series(ticker_str):
    """分配金再投資のトータルリターン指数を構築"""
    t = yf.Ticker(ticker_str)
    hist = t.history(period='max')
    hist.index = hist.index.tz_localize(None)
    divs = t.dividends
    if len(divs) > 0:
        divs.index = divs.index.tz_localize(None)

    prices = hist['Close'].copy()
    volume = hist['Volume'].copy()

    tr_index = pd.Series(index=prices.index, dtype=float)
    tr_index.iloc[0] = 100.0

    for i in range(1, len(prices)):
        date = prices.index[i]
        prev_date = prices.index[i-1]
        price_return = prices.iloc[i] / prices.iloc[i-1] - 1

        div_yield = 0.0
        if len(divs) > 0:
            matching = divs[(divs.index >= prev_date) & (divs.index <= date)]
            if len(matching) > 0:
                div_yield = matching.sum() / prices.iloc[i-1]

        tr_index.iloc[i] = tr_index.iloc[i-1] * (1 + price_return + div_yield)

    return prices, tr_index, volume, divs

data = {}
for ticker, name in ETFS.items():
    print(f"{name} ({ticker}) データ取得中...")
    price, tr, vol, divs = get_total_return_series(ticker)
    data[ticker] = {'price': price, 'tr': tr, 'vol': vol, 'divs': divs, 'name': name}
    print(f"  期間: {price.index[0].date()} ~ {price.index[-1].date()}, {len(price)}日, 分配金{len(divs)}回")

# 共通期間（全4本が揃う期間）
all_starts = [data[t]['tr'].index[0] for t in ETFS]
all_ends = [data[t]['tr'].index[-1] for t in ETFS]
common_start = max(all_starts)
common_end = min(all_ends)

# 共通日付
common_dates = data[list(ETFS.keys())[0]]['tr'][common_start:common_end].index
for ticker in list(ETFS.keys())[1:]:
    common_dates = common_dates.intersection(data[ticker]['tr'][common_start:common_end].index)

print(f"\n共通期間: {common_start.date()} ~ {common_end.date()}")
print(f"共通営業日数: {len(common_dates)}")

# 共通期間で切り出し & 100スタート正規化
tr_common = {}
ret_common = {}
for ticker in ETFS:
    s = data[ticker]['tr'].loc[common_dates]
    s = s / s.iloc[0] * 100
    tr_common[ticker] = s
    ret_common[ticker] = s.pct_change().dropna()

# ============================================================
# 2. 基本統計量
# ============================================================
print("\n" + "=" * 70)
print("2. 基本統計量（共通期間、トータルリターン、rf=0%）")
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
        'daily_mean': ret_series.mean() * 252 * 100,
        'daily_std': ret_series.std() * 100,
    }

all_stats = {}
for ticker in ETFS:
    all_stats[ticker] = calc_stats(tr_common[ticker], ret_common[ticker])

# 表示
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
    vals = [all_stats[t][key] for t in ETFS]
    best_idx = vals.index(max(vals)) if key not in ['vol', 'mdd', 'monthly_kurt'] else vals.index(min(vals)) if key == 'vol' else (vals.index(max(vals)) if key == 'mdd' else vals.index(min(vals)))
    for i, ticker in enumerate(ETFS):
        v = fmt.format(all_stats[ticker][key])
        if i == best_idx:
            v = f"*{v}*"
        row += f" {v:>16}"
    print(row)

# ============================================================
# 3. ペアワイズ統計検定（基準: 1655）
# ============================================================
print("\n" + "=" * 70)
print("3. ペアワイズ統計検定（基準: 1655 iShares）")
print("=" * 70)

base = '1655.T'
for ticker in ETFS:
    if ticker == base:
        continue
    diff = ret_common[base] - ret_common[ticker]
    diff = diff.dropna()

    t_stat, t_pval = stats.ttest_1samp(diff, 0)
    w_stat, w_pval = stats.wilcoxon(diff)

    ann_diff = diff.mean() * 252 * 100

    print(f"\n--- 1655 vs {ETFS[ticker]} ---")
    print(f"  日次平均差: {diff.mean()*100:.5f}% (年率: {ann_diff:+.2f}%)")
    print(f"  t検定:    t={t_stat:+.4f}, p={t_pval:.4f} {'*有意' if t_pval < 0.05 else ''}")
    print(f"  Wilcoxon: W={w_stat:.0f}, p={w_pval:.4f} {'*有意' if w_pval < 0.05 else ''}")

# 全ペアの検定
print("\n\n--- 全ペア リターン差 年率換算 & t検定 p値 ---")
tickers = list(ETFS.keys())
header = f"{'vs':>20}"
for t in tickers:
    header += f" {ETFS[t]:>16}"
print(header)
print("-" * (20 + 17 * len(tickers)))

for t1 in tickers:
    row = f"{ETFS[t1]:>20}"
    for t2 in tickers:
        if t1 == t2:
            row += f" {'---':>16}"
        else:
            diff = ret_common[t1] - ret_common[t2]
            diff = diff.dropna()
            ann = diff.mean() * 252 * 100
            _, pval = stats.ttest_1samp(diff, 0)
            sig = '*' if pval < 0.05 else ''
            row += f" {ann:+.2f}%(p={pval:.2f}){sig}".rjust(16)
    print(row)

# ============================================================
# 4. ブートストラップ（全ペア Sharpe差）
# ============================================================
print("\n" + "=" * 70)
print("4. ブロックブートストラップ（Sharpe差、CAGR差）")
print("=" * 70)

np.random.seed(42)
N_BOOT = 10000
BLOCK_SIZE = 21

def sharpe_ratio(r):
    return r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0

def cagr_from_ret(r):
    cum = (1 + r).prod()
    ny = len(r) / 252
    return cum ** (1/ny) - 1 if ny > 0 else 0

# 全ETFのリターンをDataFrame化
ret_df = pd.DataFrame({t: ret_common[t] for t in ETFS})
n = len(ret_df)

# ブートストラップ
boot_results = {t: {'sharpe': [], 'cagr': []} for t in ETFS}
for _ in range(N_BOOT):
    n_blocks = n // BLOCK_SIZE + 1
    starts = np.random.randint(0, n - BLOCK_SIZE + 1, size=n_blocks)
    indices = np.concatenate([np.arange(s, s + BLOCK_SIZE) for s in starts])[:n]
    sample = ret_df.iloc[indices]
    for t in ETFS:
        boot_results[t]['sharpe'].append(sharpe_ratio(sample[t]))
        boot_results[t]['cagr'].append(cagr_from_ret(sample[t]))

for t in ETFS:
    boot_results[t]['sharpe'] = np.array(boot_results[t]['sharpe'])
    boot_results[t]['cagr'] = np.array(boot_results[t]['cagr'])

# ペアワイズSharpe差
print(f"\n[ブロックブートストラップ n={N_BOOT}, block={BLOCK_SIZE}日]")
for t1 in tickers:
    for t2 in tickers:
        if t1 >= t2:
            continue
        diff_s = boot_results[t1]['sharpe'] - boot_results[t2]['sharpe']
        diff_c = boot_results[t1]['cagr'] - boot_results[t2]['cagr']
        print(f"\n  {ETFS[t1]} vs {ETFS[t2]}:")
        print(f"    Sharpe差: mean={diff_s.mean():.4f}, 95%CI=[{np.percentile(diff_s,2.5):.4f}, {np.percentile(diff_s,97.5):.4f}], {ETFS[t1]}優位確率={((diff_s>0).mean()*100):.1f}%")
        print(f"    CAGR差:   mean={diff_c.mean()*100:.2f}%, 95%CI=[{np.percentile(diff_c,2.5)*100:.2f}%, {np.percentile(diff_c,97.5)*100:.2f}%]")

# Jobson-Korkie-Memmel
print("\n[Jobson-Korkie-Memmel検定 (Sharpe差)]")
for t1 in tickers:
    for t2 in tickers:
        if t1 >= t2:
            continue
        r1, r2 = ret_common[t1].values, ret_common[t2].values
        mu1, mu2 = r1.mean(), r2.mean()
        s1, s2 = r1.std(), r2.std()
        rho = np.corrcoef(r1, r2)[0, 1]
        n_obs = len(r1)
        sr1, sr2 = mu1/s1, mu2/s2
        theta = (1/n_obs) * (2*(1-rho) + 0.5*(sr1**2 + sr2**2 - 2*sr1*sr2*rho))
        z = (sr1 - sr2) / np.sqrt(theta) if theta > 0 else 0
        p_val = 2 * (1 - stats.norm.cdf(abs(z)))
        print(f"  {ETFS[t1]} vs {ETFS[t2]}: z={z:+.4f}, p={p_val:.4f} {'*有意(5%)' if p_val < 0.05 else '非有意'}")

# ============================================================
# 5. ローリング分析
# ============================================================
print("\n" + "=" * 70)
print("5. ローリング分析")
print("=" * 70)

for window in [126, 252]:
    print(f"\n--- ローリング{window}日 ---")
    rolling_sharpe = {}
    for t in ETFS:
        rolling_sharpe[t] = ret_common[t].rolling(window).apply(
            lambda x: x.mean()/x.std()*np.sqrt(252), raw=True).dropna()

    # 各ペアで1655基準
    for t in ETFS:
        if t == base:
            continue
        diff = (rolling_sharpe[base] - rolling_sharpe[t]).dropna()
        pct_win = (diff > 0).mean() * 100
        print(f"  1655 vs {ETFS[t]}: 1655優位 {pct_win:.1f}%, 平均差 {diff.mean():+.3f}, range [{diff.min():.3f}, {diff.max():.3f}]")

# ============================================================
# 6. トラッキング差異分析
# ============================================================
print("\n" + "=" * 70)
print("6. トラッキング差異分析（ペアワイズ相関・TE）")
print("=" * 70)

print(f"\n日次リターン相関行列:")
corr_df = pd.DataFrame({ETFS[t]: ret_common[t] for t in ETFS}).corr()
print(corr_df.round(6).to_string())

# トラッキングエラー（1655基準）
print(f"\nトラッキングエラー（1655基準、年率）:")
for t in ETFS:
    if t == base:
        continue
    te = (ret_common[base] - ret_common[t]).std() * np.sqrt(252)
    print(f"  1655 vs {ETFS[t]}: TE = {te*100:.3f}%")

# 日次リターン差の統計
print(f"\n日次リターン差の分布（1655 - 各ETF）:")
for t in ETFS:
    if t == base:
        continue
    diff = ret_common[base] - ret_common[t]
    print(f"  vs {ETFS[t]}: mean={diff.mean()*100:.5f}%, std={diff.std()*100:.4f}%, max={diff.max()*100:.3f}%, min={diff.min()*100:.3f}%")

# ============================================================
# 7. 出来高・流動性分析
# ============================================================
print("\n" + "=" * 70)
print("7. 出来高・流動性分析")
print("=" * 70)

for ticker in ETFS:
    vol = data[ticker]['vol'].loc[common_dates]
    price = data[ticker]['price'].loc[common_dates]
    turnover = (vol * price).tail(60)
    print(f"\n{ETFS[ticker]} ({ticker}):")
    print(f"  60日平均出来高: {vol.tail(60).mean():,.0f} 口")
    print(f"  60日平均売買代金: {turnover.mean():,.0f} 円 ({turnover.mean()/1e6:.1f}百万円)")
    print(f"  出来高中央値: {vol.tail(60).median():,.0f} 口")
    print(f"  出来高ゼロ日数(直近60日): {(vol.tail(60) == 0).sum()}")

# ============================================================
# 8. コスト構造
# ============================================================
print("\n" + "=" * 70)
print("8. ETFコスト・構造比較")
print("=" * 70)

print("""
                     1655 iShares    2558 MAXIS     2633 NEXT FUNDS   1547 Listed Idx
信託報酬(税込)        0.0825%         0.0858%         0.077%           0.165%
純資産                1,529億円       1,059億円       227億円           603億円
60日平均出来高        207万株         2.7万口          17.9万口          2.9万口
1口あたり価格        約754円         約30,120円        約485円           約11,345円
上場日                2017/09         2020/01         2021/03           2010/10
分配頻度              年2回(2月/8月)   年2回(6月/12月)  年2回(3月/9月)    年1回(7月)

[コスト順位（低い方が良い）]
1位: 2633 NEXT FUNDS (0.077%)
2位: 1655 iShares (0.0825%)
3位: 2558 MAXIS (0.0858%)
4位: 1547 Listed Index (0.165%) ← 倍以上のコスト差

[流動性順位（売買代金ベース）]
1655が圧倒的（60日平均売買代金で他を桁違いに上回る）
""")

# ============================================================
# 9. 長期ペア分析（1655 vs 1547、2017年〜）
# ============================================================
print("=" * 70)
print("9. 長期ペア分析（1655 vs 1547）")
print("=" * 70)

long_start = max(data['1655.T']['tr'].index[0], data['1547.T']['tr'].index[0])
long_dates = data['1655.T']['tr'][long_start:].index.intersection(data['1547.T']['tr'][long_start:].index)
tr_1655_long = data['1655.T']['tr'].loc[long_dates] / data['1655.T']['tr'].loc[long_dates].iloc[0] * 100
tr_1547_long = data['1547.T']['tr'].loc[long_dates] / data['1547.T']['tr'].loc[long_dates].iloc[0] * 100
ret_1655_long = tr_1655_long.pct_change().dropna()
ret_1547_long = tr_1547_long.pct_change().dropna()

s_1655 = calc_stats(tr_1655_long, ret_1655_long)
s_1547 = calc_stats(tr_1547_long, ret_1547_long)

print(f"期間: {long_dates[0].date()} ~ {long_dates[-1].date()} ({len(long_dates)}日, {len(long_dates)/252:.1f}年)")
print(f"\n{'指標':<20} {'1655':>12} {'1547':>12} {'差':>12}")
print("-" * 58)
for label, key, fmt in metrics:
    v1 = s_1655[key]
    v2 = s_1547[key]
    print(f"{label:<20} {fmt.format(v1):>12} {fmt.format(v2):>12} {fmt.format(v1-v2):>12}")

diff_long = ret_1655_long - ret_1547_long
diff_long = diff_long.dropna()
t_l, p_l = stats.ttest_1samp(diff_long, 0)
print(f"\nリターン差 t検定: t={t_l:.4f}, p={p_l:.4f}")
print(f"年率差: {diff_long.mean()*252*100:+.2f}%")

# ============================================================
# 10. 図の作成
# ============================================================
print("\n" + "=" * 70)
print("10. 図の作成")
print("=" * 70)

COLORS = {'1655.T': '#1565C0', '2558.T': '#2E7D32', '2633.T': '#E65100', '1547.T': '#6A1B9A'}

# --- 図1: トータルリターン推移 ---
fig, ax = plt.subplots(figsize=(14, 7))
for t in ETFS:
    ax.plot(tr_common[t].index, tr_common[t].values, label=ETFS[t], linewidth=1.5, color=COLORS[t])
ax.set_title('S&P500 ETF Total Return Index (Common Period, 100 Start)', fontsize=14)
ax.set_ylabel('Index')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/01_total_return.png')
plt.close()
print("  01_total_return.png")

# --- 図2: ドローダウン ---
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
for i, t in enumerate(ETFS):
    ax = axes[i//2][i%2]
    dd = (tr_common[t] - tr_common[t].cummax()) / tr_common[t].cummax()
    ax.fill_between(dd.index, dd.values * 100, 0, alpha=0.5, color=COLORS[t])
    ax.set_title(f'{ETFS[t]} Drawdown', fontsize=12)
    ax.set_ylabel('DD (%)')
    ax.set_ylim(dd.min()*100 - 2, 2)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
plt.suptitle('Drawdown Comparison', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/02_drawdown.png')
plt.close()
print("  02_drawdown.png")

# --- 図3: ローリングSharpe差（1655基準） ---
fig, axes = plt.subplots(3, 1, figsize=(14, 12))
for i, t in enumerate([t for t in ETFS if t != base]):
    ax = axes[i]
    for window, alpha in [(252, 0.9), (126, 0.4)]:
        rs1 = ret_common[base].rolling(window).apply(lambda x: x.mean()/x.std()*np.sqrt(252), raw=True)
        rs2 = ret_common[t].rolling(window).apply(lambda x: x.mean()/x.std()*np.sqrt(252), raw=True)
        diff = (rs1 - rs2).dropna()
        lbl = f'{window}d'
        ax.plot(diff.index, diff.values, linewidth=1, alpha=alpha, label=lbl, color=COLORS[t])
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.fill_between(diff.index, diff.values, 0, where=diff.values > 0, alpha=0.2, color='#1565C0')
    ax.fill_between(diff.index, diff.values, 0, where=diff.values <= 0, alpha=0.2, color='#C62828')
    ax.set_title(f'Rolling Sharpe Diff: 1655 - {ETFS[t]}', fontsize=12)
    ax.set_ylabel('Sharpe Diff')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/03_rolling_sharpe_diff.png')
plt.close()
print("  03_rolling_sharpe_diff.png")

# --- 図4: ブートストラップSharpe差分布 ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
others = [t for t in tickers if t != base]
for i, t in enumerate(others):
    diff_boot = boot_results[base]['sharpe'] - boot_results[t]['sharpe']
    ax = axes[i]
    ax.hist(diff_boot, bins=80, color=COLORS[t], alpha=0.6, edgecolor='white')
    ax.axvline(0, color='red', linewidth=2, linestyle='--')
    ax.axvline(np.percentile(diff_boot, 2.5), color='blue', linewidth=1, linestyle=':')
    ax.axvline(np.percentile(diff_boot, 97.5), color='blue', linewidth=1, linestyle=':')
    ax.set_title(f'Bootstrap Sharpe Diff: 1655 - {ETFS[t]}')
    ax.set_xlabel('Sharpe Diff')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/04_bootstrap_sharpe.png')
plt.close()
print("  04_bootstrap_sharpe.png")

# --- 図5: 累積リターン差（1655基準） ---
fig, ax = plt.subplots(figsize=(14, 6))
for t in ETFS:
    if t == base:
        continue
    diff_cum = tr_common[base] - tr_common[t]
    ax.plot(diff_cum.index, diff_cum.values, label=f'1655 - {ETFS[t]}', linewidth=1.5, color=COLORS[t])
ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax.set_title('Cumulative Return Difference vs 1655 (Index Points)', fontsize=14)
ax.set_ylabel('Difference (pts)')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/05_cumulative_diff.png')
plt.close()
print("  05_cumulative_diff.png")

# --- 図6: 年次リターン比較 ---
fig, ax = plt.subplots(figsize=(14, 7))
yearly_data = {}
for t in ETFS:
    yearly_data[t] = tr_common[t].resample('YE').last().pct_change().dropna()

common_years = yearly_data[tickers[0]].index
for t in tickers[1:]:
    common_years = common_years.intersection(yearly_data[t].index)

x = np.arange(len(common_years))
w = 0.2
for i, t in enumerate(ETFS):
    vals = yearly_data[t].loc[common_years].values * 100
    ax.bar(x + i*w - 1.5*w, vals, w, label=ETFS[t], color=COLORS[t])
ax.set_xticks(x)
ax.set_xticklabels([d.strftime('%Y') for d in common_years])
ax.set_title('Annual Total Return by ETF (%)', fontsize=14)
ax.set_ylabel('Return (%)')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(0, color='black', linewidth=0.8)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/06_annual_returns.png')
plt.close()
print("  06_annual_returns.png")

# --- 図7: 長期比較（1655 vs 1547） ---
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(tr_1655_long.index, tr_1655_long.values, label='1655 iShares', linewidth=1.5, color=COLORS['1655.T'])
ax.plot(tr_1547_long.index, tr_1547_long.values, label='1547 Listed Index', linewidth=1.5, color=COLORS['1547.T'])
ax.set_title(f'Long-term Comparison: 1655 vs 1547 ({long_dates[0].strftime("%Y/%m")}~)', fontsize=14)
ax.set_ylabel('Total Return Index (100 Start)')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/07_long_term_1655_vs_1547.png')
plt.close()
print("  07_long_term_1655_vs_1547.png")

# --- 図8: 日次リターン差の分布（1655 vs 各） ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, t in enumerate(others):
    diff = (ret_common[base] - ret_common[t]).dropna() * 100
    ax = axes[i]
    ax.hist(diff, bins=100, color=COLORS[t], alpha=0.6, edgecolor='white')
    ax.axvline(0, color='red', linewidth=1.5, linestyle='--')
    ax.axvline(diff.mean(), color='blue', linewidth=1.5, linestyle='-', label=f'mean={diff.mean():.4f}%')
    ax.set_title(f'Daily Return Diff: 1655 - {ETFS[t]}')
    ax.set_xlabel('Return Diff (%)')
    ax.legend()
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/08_return_diff_dist.png')
plt.close()
print("  08_return_diff_dist.png")

print("\n" + "=" * 70)
print("全出力完了")
print(f"図表出力先: {OUTPUT_DIR}/")
print("=" * 70)
