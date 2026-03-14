"""
日本上場S&P500 ETF（為替ヘッジなし）統計比較分析 v2
株式分割データ修正版

対象: 1655, 2558, 2633, 1547 — 全て同一指数(S&P500)、為替ヘッジなし
データソース: yfinance（株式分割未調整を手動修正）
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

# 既知の株式分割（yfinanceが未調整のもの）
# 形式: {ticker: [(split_date, ratio), ...]}
# ratio: 例えば1:10分割なら10（旧価格をratioで割る）
KNOWN_SPLITS = {
    '1655.T': [
        ('2017-09-28', 10),  # 上場直後の1:10分割
        ('2022-02-09', 10),  # 1:10分割（2/8落ち、2/10にデータ復帰）
    ],
    '2633.T': [
        ('2021-03-31', 10),  # 上場直後の1:10分割
        ('2023-12-07', 10),  # 1:10分割（12/6落ち）
    ],
}

# ============================================================
# 1. データ取得・分割修正・トータルリターン系列構築
# ============================================================
print("=" * 70)
print("1. データ取得 & 株式分割修正")
print("=" * 70)

def fix_splits(prices, volume, ticker):
    """既知の株式分割を手動修正"""
    if ticker not in KNOWN_SPLITS:
        return prices, volume

    prices = prices.copy()
    volume = volume.copy()

    for split_date_str, ratio in KNOWN_SPLITS[ticker]:
        split_date = pd.Timestamp(split_date_str)
        # 分割日より前のデータを調整
        mask = prices.index < split_date
        prices.loc[mask] = prices.loc[mask] / ratio
        volume.loc[mask] = volume.loc[mask] * ratio

    return prices, volume

def get_total_return_series(ticker_str):
    """分配金再投資のトータルリターン指数を構築（分割修正済み）"""
    t = yf.Ticker(ticker_str)
    hist = t.history(period='max')
    hist.index = hist.index.tz_localize(None)
    divs = t.dividends
    if len(divs) > 0:
        divs.index = divs.index.tz_localize(None)
        # 分割前の分配金も調整
        if ticker_str in KNOWN_SPLITS:
            divs = divs.copy()
            for split_date_str, ratio in KNOWN_SPLITS[ticker_str]:
                split_date = pd.Timestamp(split_date_str)
                mask = divs.index < split_date
                divs.loc[mask] = divs.loc[mask] / ratio

    prices = hist['Close'].copy()
    volume = hist['Volume'].copy()

    # 分割修正
    prices, volume = fix_splits(prices, volume, ticker_str)

    # 修正後の異常値チェック（前日比50%超の変動を除去）
    ret = prices.pct_change()
    spike_mask = abs(ret) > 0.5
    if spike_mask.any():
        spike_dates = prices.index[spike_mask]
        print(f"  WARNING: {ticker_str} 修正後も異常値あり: {[d.strftime('%Y-%m-%d') for d in spike_dates]}")
        # スパイクの翌日から使用
        for sd in spike_dates:
            idx = prices.index.get_loc(sd)
            if idx > 0:
                # 前日比で異常→前日の値で埋める
                prices.iloc[idx] = prices.iloc[idx-1]

    # トータルリターン指数構築
    tr_index = pd.Series(index=prices.index, dtype=float)
    tr_index.iloc[0] = 100.0

    for i in range(1, len(prices)):
        date = prices.index[i]
        prev_date = prices.index[i-1]
        if prices.iloc[i-1] == 0:
            tr_index.iloc[i] = tr_index.iloc[i-1]
            continue
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

    # サニティチェック
    ret = tr.pct_change().dropna()
    print(f"  期間: {price.index[0].date()} ~ {price.index[-1].date()}, {len(price)}日")
    print(f"  価格レンジ: {price.min():.1f} ~ {price.max():.1f}")
    print(f"  日次リターン std: {ret.std()*100:.2f}%")
    print(f"  分配金{len(divs)}回")

# 共通期間
all_starts = [data[t]['tr'].index[0] for t in ETFS]
common_start = max(all_starts)
common_end = min([data[t]['tr'].index[-1] for t in ETFS])

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

# サニティ: 相関チェック
print("\n[サニティチェック: 日次リターン相関]")
corr = pd.DataFrame({ETFS[t]: ret_common[t] for t in ETFS}).corr()
print(corr.round(4).to_string())

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

# ベスト判定
print("\n[指標別ベスト]")
for label, key, fmt in metrics:
    vals = {ETFS[t]: all_stats[t][key] for t in ETFS}
    if key in ['vol']:  # 低いほうが良い
        best = min(vals, key=vals.get)
    elif key in ['mdd']:  # 0に近いほうが良い（値は負）
        best = max(vals, key=vals.get)
    elif key in ['monthly_kurt']:
        best = min(vals, key=vals.get)
    else:
        best = max(vals, key=vals.get)
    print(f"  {label}: {best} ({fmt.format(vals[best])})")

# ============================================================
# 3. ペアワイズ統計検定
# ============================================================
print("\n" + "=" * 70)
print("3. ペアワイズ統計検定")
print("=" * 70)

tickers = list(ETFS.keys())

# 日次
print("\n--- 日次リターン差 t検定 ---")
print(f"{'ペア':<35} {'年率差':>10} {'t値':>8} {'p値':>8} {'判定':>8}")
print("-" * 72)
for i, t1 in enumerate(tickers):
    for t2 in tickers[i+1:]:
        diff = (ret_common[t1] - ret_common[t2]).dropna()
        t_stat, p_val = stats.ttest_1samp(diff, 0)
        ann = diff.mean() * 252 * 100
        sig = '*' if p_val < 0.05 else ''
        print(f"{ETFS[t1]} vs {ETFS[t2]:<16} {ann:>+9.2f}% {t_stat:>+7.3f} {p_val:>7.4f} {sig:>8}")

# Wilcoxon
print("\n--- Wilcoxon符号付順位検定 ---")
print(f"{'ペア':<35} {'W統計量':>10} {'p値':>8} {'判定':>8}")
print("-" * 64)
for i, t1 in enumerate(tickers):
    for t2 in tickers[i+1:]:
        diff = (ret_common[t1] - ret_common[t2]).dropna()
        # ゼロ除去(Wilcoxonはゼロを扱えない)
        diff_nz = diff[diff != 0]
        if len(diff_nz) > 10:
            w_stat, w_pval = stats.wilcoxon(diff_nz)
            sig = '*' if w_pval < 0.05 else ''
            print(f"{ETFS[t1]} vs {ETFS[t2]:<16} {w_stat:>10.0f} {w_pval:>7.4f} {sig:>8}")

# 週次
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
        if abs(ann) > 0.5 or p_val < 0.1:  # 意味のある差のみ表示
            print(f"  {ETFS[t1]} vs {ETFS[t2]}: 年率差={ann:+.2f}%, p={p_val:.4f} {sig}")

# ============================================================
# 4. Sharpe差検定 & ブートストラップ
# ============================================================
print("\n" + "=" * 70)
print("4. Sharpe差検定 & ブートストラップ")
print("=" * 70)

# Jobson-Korkie-Memmel
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
    return r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0

def cagr_from_ret(r):
    cum = (1 + r).prod()
    ny = len(r) / 252
    return cum ** (1/ny) - 1 if ny > 0 else 0

def mdd_from_ret(r):
    cum = (1 + r).cumprod()
    return (cum / cum.cummax() - 1).min()

ret_df = pd.DataFrame({t: ret_common[t] for t in ETFS})
n = len(ret_df)

boot_results = {t: {'sharpe': [], 'cagr': [], 'mdd': []} for t in ETFS}
for _ in range(N_BOOT):
    n_blocks = n // BLOCK_SIZE + 1
    starts = np.random.randint(0, n - BLOCK_SIZE + 1, size=n_blocks)
    indices = np.concatenate([np.arange(s, s + BLOCK_SIZE) for s in starts])[:n]
    sample = ret_df.iloc[indices]
    for t in ETFS:
        boot_results[t]['sharpe'].append(sharpe_ratio(sample[t]))
        boot_results[t]['cagr'].append(cagr_from_ret(sample[t]))
        boot_results[t]['mdd'].append(mdd_from_ret(sample[t]))

for t in ETFS:
    for k in boot_results[t]:
        boot_results[t][k] = np.array(boot_results[t][k])

print(f"\n[ブロックブートストラップ n={N_BOOT}, block={BLOCK_SIZE}日]")
for i, t1 in enumerate(tickers):
    for t2 in tickers[i+1:]:
        ds = boot_results[t1]['sharpe'] - boot_results[t2]['sharpe']
        dc = boot_results[t1]['cagr'] - boot_results[t2]['cagr']
        dm = boot_results[t1]['mdd'] - boot_results[t2]['mdd']
        print(f"\n  {ETFS[t1]} vs {ETFS[t2]}:")
        print(f"    Sharpe差: mean={ds.mean():.4f}, 95%CI=[{np.percentile(ds,2.5):.4f}, {np.percentile(ds,97.5):.4f}], {ETFS[t1]}優位={((ds>0).mean()*100):.1f}%")
        print(f"    CAGR差:   mean={dc.mean()*100:.2f}%, 95%CI=[{np.percentile(dc,2.5)*100:.2f}%, {np.percentile(dc,97.5)*100:.2f}%]")
        print(f"    MDD差:    mean={dm.mean()*100:.2f}%pt, {ETFS[t1]}がマシな確率={((dm>0).mean()*100):.1f}%")

# ============================================================
# 5. ローリング分析
# ============================================================
print("\n" + "=" * 70)
print("5. ローリング分析")
print("=" * 70)

base = '1655.T'
for window in [126, 252]:
    print(f"\n--- ローリング{window}日 Sharpe（1655基準）---")
    rolling_sharpe = {}
    for t in ETFS:
        rolling_sharpe[t] = ret_common[t].rolling(window).apply(
            lambda x: x.mean()/x.std()*np.sqrt(252), raw=True).dropna()

    for t in ETFS:
        if t == base:
            continue
        diff = (rolling_sharpe[base] - rolling_sharpe[t]).dropna()
        pct_win = (diff > 0).mean() * 100
        print(f"  vs {ETFS[t]}: 1655優位 {pct_win:.1f}%, mean {diff.mean():+.3f}, [{diff.min():.3f}, {diff.max():.3f}]")

# ============================================================
# 6. トラッキング差異
# ============================================================
print("\n" + "=" * 70)
print("6. トラッキング差異分析")
print("=" * 70)

print(f"\n日次リターン相関行列:")
corr = pd.DataFrame({ETFS[t]: ret_common[t] for t in ETFS}).corr()
print(corr.round(4).to_string())

print(f"\nトラッキングエラー（年率、全ペア）:")
for i, t1 in enumerate(tickers):
    for t2 in tickers[i+1:]:
        te = (ret_common[t1] - ret_common[t2]).std() * np.sqrt(252)
        print(f"  {ETFS[t1]} vs {ETFS[t2]}: TE = {te*100:.3f}%")

# ============================================================
# 7. 出来高・流動性
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
    print(f"  60日平均売買代金: {turnover.mean()/1e6:.1f} 百万円")
    print(f"  1口あたり価格: {price.iloc[-1]:,.0f}円")

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
上場日                2017/09         2020/01         2021/03           2010/10
分配頻度              年2回(2月/8月)   年2回(6月/12月)  年2回(3月/9月)    年1回(7月)
""")

# ============================================================
# 9. 長期ペア分析（1655 vs 1547）
# ============================================================
print("=" * 70)
print("9. 長期ペア分析（1655 vs 1547: 2017/9〜）")
print("=" * 70)

long_start = max(data['1655.T']['tr'].index[0], data['1547.T']['tr'].index[0])
long_dates = data['1655.T']['tr'][long_start:].index.intersection(data['1547.T']['tr'][long_start:].index)
tr_1655_l = data['1655.T']['tr'].loc[long_dates] / data['1655.T']['tr'].loc[long_dates].iloc[0] * 100
tr_1547_l = data['1547.T']['tr'].loc[long_dates] / data['1547.T']['tr'].loc[long_dates].iloc[0] * 100
ret_1655_l = tr_1655_l.pct_change().dropna()
ret_1547_l = tr_1547_l.pct_change().dropna()

s_1655 = calc_stats(tr_1655_l, ret_1655_l)
s_1547 = calc_stats(tr_1547_l, ret_1547_l)

print(f"期間: {long_dates[0].date()} ~ {long_dates[-1].date()} ({len(long_dates)/252:.1f}年)")
print(f"\n{'指標':<20} {'1655':>12} {'1547':>12} {'差':>12}")
print("-" * 58)
for label, key, fmt in metrics:
    v1, v2 = s_1655[key], s_1547[key]
    print(f"{label:<20} {fmt.format(v1):>12} {fmt.format(v2):>12} {fmt.format(v1-v2):>12}")

diff_l = (ret_1655_l - ret_1547_l).dropna()
t_l, p_l = stats.ttest_1samp(diff_l, 0)
print(f"\nt検定: t={t_l:.4f}, p={p_l:.4f}, 年率差={diff_l.mean()*252*100:+.2f}%")

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

# --- 図2: ドローダウン比較（重ね合わせ）---
fig, ax = plt.subplots(figsize=(14, 6))
for t in ETFS:
    dd = (tr_common[t] - tr_common[t].cummax()) / tr_common[t].cummax()
    ax.plot(dd.index, dd.values * 100, label=ETFS[t], linewidth=1, color=COLORS[t], alpha=0.8)
ax.set_title('Drawdown Comparison (%)', fontsize=14)
ax.set_ylabel('Drawdown (%)')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/02_drawdown.png')
plt.close()
print("  02_drawdown.png")

# --- 図3: ローリングSharpe差（1655基準）---
fig, axes = plt.subplots(3, 1, figsize=(14, 12))
others = [t for t in tickers if t != base]
for i, t in enumerate(others):
    ax = axes[i]
    for window, lw in [(252, 1.5), (126, 0.8)]:
        rs1 = ret_common[base].rolling(window).apply(lambda x: x.mean()/x.std()*np.sqrt(252), raw=True)
        rs2 = ret_common[t].rolling(window).apply(lambda x: x.mean()/x.std()*np.sqrt(252), raw=True)
        diff = (rs1 - rs2).dropna()
        ax.plot(diff.index, diff.values, linewidth=lw, label=f'{window}d', color=COLORS[t] if window == 252 else '#999')
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    diff252 = (ret_common[base].rolling(252).apply(lambda x: x.mean()/x.std()*np.sqrt(252), raw=True) -
               ret_common[t].rolling(252).apply(lambda x: x.mean()/x.std()*np.sqrt(252), raw=True)).dropna()
    ax.fill_between(diff252.index, diff252.values, 0, where=diff252.values > 0, alpha=0.15, color='#1565C0')
    ax.fill_between(diff252.index, diff252.values, 0, where=diff252.values <= 0, alpha=0.15, color='#C62828')
    ax.set_title(f'Rolling Sharpe Diff: 1655 - {ETFS[t]}', fontsize=12)
    ax.set_ylabel('Sharpe Diff')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/03_rolling_sharpe_diff.png')
plt.close()
print("  03_rolling_sharpe_diff.png")

# --- 図4: ブートストラップSharpe分布 ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, t in enumerate(others):
    diff_boot = boot_results[base]['sharpe'] - boot_results[t]['sharpe']
    ax = axes[i]
    ax.hist(diff_boot, bins=80, color=COLORS[t], alpha=0.6, edgecolor='white')
    ax.axvline(0, color='red', linewidth=2, linestyle='--')
    ax.axvline(np.percentile(diff_boot, 2.5), color='blue', linewidth=1, linestyle=':')
    ax.axvline(np.percentile(diff_boot, 97.5), color='blue', linewidth=1, linestyle=':')
    ax.set_title(f'Bootstrap Sharpe: 1655 - {ETFS[t]}')
    ax.set_xlabel('Sharpe Diff')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/04_bootstrap_sharpe.png')
plt.close()
print("  04_bootstrap_sharpe.png")

# --- 図5: 累積リターン差 ---
fig, ax = plt.subplots(figsize=(14, 6))
for t in others:
    diff_cum = tr_common[base] - tr_common[t]
    ax.plot(diff_cum.index, diff_cum.values, label=f'1655 - {ETFS[t]}', linewidth=1.5, color=COLORS[t])
ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax.set_title('Cumulative Total Return Difference vs 1655 (Index Points)', fontsize=14)
ax.set_ylabel('Difference (pts)')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/05_cumulative_diff.png')
plt.close()
print("  05_cumulative_diff.png")

# --- 図6: 年次リターン ---
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
ax.set_title('Annual Total Return (%)', fontsize=14)
ax.set_ylabel('Return (%)')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(0, color='black', linewidth=0.8)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/06_annual_returns.png')
plt.close()
print("  06_annual_returns.png")

# --- 図7: 長期1655 vs 1547 ---
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(tr_1655_l.index, tr_1655_l.values, label='1655 iShares', linewidth=1.5, color=COLORS['1655.T'])
ax.plot(tr_1547_l.index, tr_1547_l.values, label='1547 Listed Index', linewidth=1.5, color=COLORS['1547.T'])
ax.set_title(f'Long-term: 1655 vs 1547 ({long_dates[0].strftime("%Y/%m")}~)', fontsize=14)
ax.set_ylabel('Total Return Index (100 Start)')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/07_long_term.png')
plt.close()
print("  07_long_term.png")

# --- 図8: リターン差分布 ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, t in enumerate(others):
    diff = (ret_common[base] - ret_common[t]).dropna() * 100
    ax = axes[i]
    ax.hist(diff, bins=100, color=COLORS[t], alpha=0.6, edgecolor='white')
    ax.axvline(0, color='red', linewidth=1.5, linestyle='--')
    ax.axvline(diff.mean(), color='blue', linewidth=1.5, label=f'mean={diff.mean():.4f}%')
    ax.set_title(f'Daily Return Diff: 1655 - {ETFS[t]} (%)')
    ax.set_xlabel('Return Diff (%)')
    ax.legend()
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/08_return_diff_dist.png')
plt.close()
print("  08_return_diff_dist.png")

print("\n" + "=" * 70)
print("全出力完了")
print(f"出力先: {OUTPUT_DIR}/")
print("=" * 70)
