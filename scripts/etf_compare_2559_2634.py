"""
2559 vs 2634 ETF統計比較分析
2559: MAXIS全世界株式(MSCI ACWI) ETF — 為替ヘッジなし
2634: NEXT FUNDS S&P500(円ヘッジ) ETF — 円ヘッジあり

データソース: yfinance (Yahoo Finance Japan)
欠損処理: 前方補完(ffill)後、残存NaNを除去
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
from datetime import datetime
import json
import os

OUTPUT_DIR = '/home/like_rapid/GT-SOAR/reports/etf_compare'
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.figsize'] = (14, 7)
plt.rcParams['figure.dpi'] = 120

# ============================================================
# 1. データ取得・トータルリターン系列構築
# ============================================================
print("=" * 60)
print("1. データ取得")
print("=" * 60)

def get_total_return_series(ticker_str):
    """分配金再投資のトータルリターン指数を構築"""
    t = yf.Ticker(ticker_str)
    hist = t.history(period='max')
    hist.index = hist.index.tz_localize(None)

    # 分配金取得
    divs = t.dividends
    if len(divs) > 0:
        divs.index = divs.index.tz_localize(None)

    # 終値ベースのトータルリターン指数
    # yfinanceのCloseは分配落ち調整済みの場合があるが、念のため自作
    prices = hist['Close'].copy()

    # 分配金再投資の指数を構築
    tr_index = pd.Series(index=prices.index, dtype=float)
    tr_index.iloc[0] = 100.0

    for i in range(1, len(prices)):
        date = prices.index[i]
        prev_date = prices.index[i-1]
        price_return = prices.iloc[i] / prices.iloc[i-1] - 1

        # この日に分配金があれば再投資として加算
        div_yield = 0.0
        if len(divs) > 0:
            # 分配金の権利落ち日に該当するか
            matching = divs[(divs.index >= prev_date) & (divs.index <= date)]
            if len(matching) > 0:
                div_yield = matching.sum() / prices.iloc[i-1]

        tr_index.iloc[i] = tr_index.iloc[i-1] * (1 + price_return + div_yield)

    return prices, tr_index, hist['Volume'], divs

print("2559.T データ取得中...")
price_2559, tr_2559, vol_2559, div_2559 = get_total_return_series('2559.T')
print(f"  期間: {price_2559.index[0].date()} ~ {price_2559.index[-1].date()}, {len(price_2559)}日")

print("2634.T データ取得中...")
price_2634, tr_2634, vol_2634, div_2634 = get_total_return_series('2634.T')
print(f"  期間: {price_2634.index[0].date()} ~ {price_2634.index[-1].date()}, {len(price_2634)}日")

# ベンチマーク: ACWI(USD)
print("ACWI(ベンチマーク) データ取得中...")
acwi_t = yf.Ticker('ACWI')
acwi_hist = acwi_t.history(period='max')
acwi_hist.index = acwi_hist.index.tz_localize(None)

# VIXも取得(レジーム分析用)
print("VIX データ取得中...")
vix_t = yf.Ticker('^VIX')
vix_hist = vix_t.history(period='max')
vix_hist.index = vix_hist.index.tz_localize(None)

# USD/JPY
print("USD/JPY データ取得中...")
fx_t = yf.Ticker('JPY=X')
fx_hist = fx_t.history(period='max')
fx_hist.index = fx_hist.index.tz_localize(None)

# 共通期間の特定
common_start = max(tr_2559.index[0], tr_2634.index[0])
common_end = min(tr_2559.index[-1], tr_2634.index[-1])
print(f"\n共通期間: {common_start.date()} ~ {common_end.date()}")

# 共通期間で切り出し
tr_2559_c = tr_2559[common_start:common_end]
tr_2634_c = tr_2634[common_start:common_end]

# 日付を揃える(両方に存在する日のみ)
common_dates = tr_2559_c.index.intersection(tr_2634_c.index)
tr_2559_c = tr_2559_c.loc[common_dates]
tr_2634_c = tr_2634_c.loc[common_dates]
print(f"共通営業日数: {len(common_dates)}")

# 100スタートに再正規化
tr_2559_c = tr_2559_c / tr_2559_c.iloc[0] * 100
tr_2634_c = tr_2634_c / tr_2634_c.iloc[0] * 100

# 日次リターン
ret_2559 = tr_2559_c.pct_change().dropna()
ret_2634 = tr_2634_c.pct_change().dropna()

# ============================================================
# 2. 基本統計量の計算
# ============================================================
print("\n" + "=" * 60)
print("2. 基本統計量")
print("=" * 60)

def calc_stats(tr_series, ret_series, name, rf_annual=0.0):
    """包括的な統計量を計算"""
    n_days = len(ret_series)
    n_years = n_days / 252

    # CAGR
    total_return = tr_series.iloc[-1] / tr_series.iloc[0]
    cagr = total_return ** (1 / n_years) - 1

    # 年率ボラティリティ
    vol = ret_series.std() * np.sqrt(252)

    # 最大ドローダウン
    cummax = tr_series.cummax()
    drawdown = (tr_series - cummax) / cummax
    mdd = drawdown.min()

    # Sharpe
    rf_daily = (1 + rf_annual) ** (1/252) - 1
    excess = ret_series - rf_daily
    sharpe = excess.mean() / excess.std() * np.sqrt(252) if excess.std() > 0 else 0

    # Sortino
    downside = ret_series[ret_series < rf_daily] - rf_daily
    downside_std = np.sqrt((downside ** 2).mean()) * np.sqrt(252)
    sortino = (ret_series.mean() - rf_daily) * 252 / downside_std if downside_std > 0 else 0

    # Calmar
    calmar = cagr / abs(mdd) if mdd != 0 else 0

    # 月次統計
    monthly = tr_series.resample('ME').last().pct_change().dropna()
    win_rate = (monthly > 0).mean()
    monthly_mean = monthly.mean()
    monthly_skew = monthly.skew()
    monthly_kurt = monthly.kurtosis()

    return {
        'name': name,
        'n_days': n_days,
        'n_years': round(n_years, 2),
        'total_return': (total_return - 1) * 100,
        'cagr': cagr * 100,
        'volatility': vol * 100,
        'mdd': mdd * 100,
        'sharpe': sharpe,
        'sortino': sortino,
        'calmar': calmar,
        'monthly_win_rate': win_rate * 100,
        'monthly_mean': monthly_mean * 100,
        'monthly_skew': monthly_skew,
        'monthly_kurt': monthly_kurt,
        'daily_mean': ret_series.mean() * 100,
        'daily_std': ret_series.std() * 100,
    }

# 無リスク金利 = 0% で計算
stats_2559 = calc_stats(tr_2559_c, ret_2559, '2559(ACWI,非ヘッジ)')
stats_2634 = calc_stats(tr_2634_c, ret_2634, '2634(SP500,円ヘッジ)')

# 日本短期金利(0.25%想定 — 2024-2025平均的な水準)で再計算
stats_2559_rf = calc_stats(tr_2559_c, ret_2559, '2559(rf=0.25%)', rf_annual=0.0025)
stats_2634_rf = calc_stats(tr_2634_c, ret_2634, '2634(rf=0.25%)', rf_annual=0.0025)

def print_comparison(s1, s2, label=""):
    """比較テーブルを表示"""
    print(f"\n{'指標':<25} {'2559':>12} {'2634':>12} {'差(2559-2634)':>14}")
    print("-" * 65)
    metrics = [
        ('営業日数', 'n_days', '{:.0f}'),
        ('年数', 'n_years', '{:.2f}'),
        ('累積リターン(%)', 'total_return', '{:.2f}'),
        ('CAGR(%)', 'cagr', '{:.2f}'),
        ('年率Vol(%)', 'volatility', '{:.2f}'),
        ('最大DD(%)', 'mdd', '{:.2f}'),
        ('Sharpe', 'sharpe', '{:.3f}'),
        ('Sortino', 'sortino', '{:.3f}'),
        ('Calmar', 'calmar', '{:.3f}'),
        ('月次勝率(%)', 'monthly_win_rate', '{:.1f}'),
        ('月次平均(%)', 'monthly_mean', '{:.3f}'),
        ('月次歪度', 'monthly_skew', '{:.3f}'),
        ('月次尖度', 'monthly_kurt', '{:.3f}'),
    ]
    for label_str, key, fmt in metrics:
        v1 = s1[key]
        v2 = s2[key]
        diff = v1 - v2
        print(f"{label_str:<25} {fmt.format(v1):>12} {fmt.format(v2):>12} {fmt.format(diff):>14}")

print("\n--- 共通期間 (rf=0%) ---")
print_comparison(stats_2559, stats_2634)

print("\n--- 共通期間 (rf=0.25%) ---")
print_comparison(stats_2559_rf, stats_2634_rf)

# ============================================================
# 3. 統計検定
# ============================================================
print("\n" + "=" * 60)
print("3. 統計検定")
print("=" * 60)

# 日次超過リターン差
diff_daily = ret_2559 - ret_2634
diff_daily = diff_daily.dropna()

# t検定
t_stat, t_pval = stats.ttest_1samp(diff_daily, 0)
print(f"\n[日次リターン差の t検定]")
print(f"  平均差: {diff_daily.mean()*100:.4f}% /日 ({diff_daily.mean()*252*100:.2f}% /年換算)")
print(f"  t統計量: {t_stat:.4f}")
print(f"  p値: {t_pval:.4f}")
print(f"  判定: {'有意(5%)' if t_pval < 0.05 else '非有意(5%)'}")

# Wilcoxon符号付順位検定
w_stat, w_pval = stats.wilcoxon(diff_daily)
print(f"\n[Wilcoxon符号付順位検定]")
print(f"  統計量: {w_stat:.0f}")
print(f"  p値: {w_pval:.4f}")
print(f"  判定: {'有意(5%)' if w_pval < 0.05 else '非有意(5%)'}")

# 週次でも検定
weekly_2559 = tr_2559_c.resample('W').last().pct_change().dropna()
weekly_2634 = tr_2634_c.resample('W').last().pct_change().dropna()
common_w = weekly_2559.index.intersection(weekly_2634.index)
diff_weekly = (weekly_2559.loc[common_w] - weekly_2634.loc[common_w]).dropna()

t_w, p_w = stats.ttest_1samp(diff_weekly, 0)
print(f"\n[週次リターン差の t検定]")
print(f"  平均差: {diff_weekly.mean()*100:.4f}% /週")
print(f"  t統計量: {t_w:.4f}")
print(f"  p値: {p_w:.4f}")

# 月次でも
monthly_2559 = tr_2559_c.resample('ME').last().pct_change().dropna()
monthly_2634 = tr_2634_c.resample('ME').last().pct_change().dropna()
common_m = monthly_2559.index.intersection(monthly_2634.index)
diff_monthly = (monthly_2559.loc[common_m] - monthly_2634.loc[common_m]).dropna()

t_m, p_m = stats.ttest_1samp(diff_monthly, 0)
print(f"\n[月次リターン差の t検定]")
print(f"  平均差: {diff_monthly.mean()*100:.3f}% /月")
print(f"  t統計量: {t_m:.4f}")
print(f"  p値: {p_m:.4f}")

# ============================================================
# 4. Sharpe差の検定（ブートストラップ）
# ============================================================
print("\n" + "=" * 60)
print("4. Sharpe差検定 & ブートストラップ")
print("=" * 60)

def sharpe_ratio(returns):
    return returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

def cagr_from_returns(returns):
    cum = (1 + returns).prod()
    n_years = len(returns) / 252
    return cum ** (1/n_years) - 1 if n_years > 0 else 0

def max_drawdown_from_returns(returns):
    cum = (1 + returns).cumprod()
    dd = cum / cum.cummax() - 1
    return dd.min()

# ブロックブートストラップ（ブロックサイズ=21営業日≒1ヶ月）
np.random.seed(42)
N_BOOT = 10000
BLOCK_SIZE = 21
n = len(ret_2559)

boot_sharpe_diff = []
boot_cagr_diff = []
boot_mdd_diff = []

aligned = pd.DataFrame({'r2559': ret_2559.values, 'r2634': ret_2634.values})

for _ in range(N_BOOT):
    # ブロック開始点をランダムに選ぶ
    n_blocks = n // BLOCK_SIZE + 1
    starts = np.random.randint(0, n - BLOCK_SIZE + 1, size=n_blocks)
    indices = np.concatenate([np.arange(s, s + BLOCK_SIZE) for s in starts])[:n]

    sample = aligned.iloc[indices]
    s1 = sharpe_ratio(sample['r2559'])
    s2 = sharpe_ratio(sample['r2634'])
    boot_sharpe_diff.append(s1 - s2)

    c1 = cagr_from_returns(sample['r2559'])
    c2 = cagr_from_returns(sample['r2634'])
    boot_cagr_diff.append(c1 - c2)

    m1 = max_drawdown_from_returns(sample['r2559'])
    m2 = max_drawdown_from_returns(sample['r2634'])
    boot_mdd_diff.append(m1 - m2)  # positive = 2559 has shallower DD

boot_sharpe_diff = np.array(boot_sharpe_diff)
boot_cagr_diff = np.array(boot_cagr_diff)
boot_mdd_diff = np.array(boot_mdd_diff)

print(f"\n[ブロックブートストラップ (n={N_BOOT}, block={BLOCK_SIZE}日)]")
print(f"\nSharpe差 (2559 - 2634):")
print(f"  平均: {boot_sharpe_diff.mean():.4f}")
print(f"  95%CI: [{np.percentile(boot_sharpe_diff, 2.5):.4f}, {np.percentile(boot_sharpe_diff, 97.5):.4f}]")
print(f"  2559優位確率: {(boot_sharpe_diff > 0).mean()*100:.1f}%")

print(f"\nCAGR差 (2559 - 2634):")
print(f"  平均: {boot_cagr_diff.mean()*100:.2f}%")
print(f"  95%CI: [{np.percentile(boot_cagr_diff, 2.5)*100:.2f}%, {np.percentile(boot_cagr_diff, 97.5)*100:.2f}%]")
print(f"  2559優位確率: {(boot_cagr_diff > 0).mean()*100:.1f}%")

print(f"\nMDD差 (2559 - 2634, 正=2559がマシ):")
print(f"  平均: {boot_mdd_diff.mean()*100:.2f}%")
print(f"  95%CI: [{np.percentile(boot_mdd_diff, 2.5)*100:.2f}%, {np.percentile(boot_mdd_diff, 97.5)*100:.2f}%]")
print(f"  2559のDDが浅い確率: {(boot_mdd_diff > 0).mean()*100:.1f}%")

# Jobson-Korkie-Memmel検定
def jk_memmel_test(r1, r2):
    """Sharpe比の差のJobson-Korkie (Memmel補正)検定"""
    mu1, mu2 = r1.mean(), r2.mean()
    s1, s2 = r1.std(), r2.std()
    rho = np.corrcoef(r1, r2)[0, 1]
    n = len(r1)

    sr1, sr2 = mu1/s1, mu2/s2

    theta = (1/n) * (2 * (1 - rho) + 0.5 * (sr1**2 + sr2**2 - 2*sr1*sr2*rho))

    z = (sr1 - sr2) / np.sqrt(theta) if theta > 0 else 0
    p_val = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p_val

z_jk, p_jk = jk_memmel_test(ret_2559.values, ret_2634.values)
print(f"\n[Jobson-Korkie-Memmel検定 (Sharpe差)]")
print(f"  z統計量: {z_jk:.4f}")
print(f"  p値: {p_jk:.4f}")
print(f"  判定: {'有意(5%)' if p_jk < 0.05 else '非有意(5%)'}")

# ============================================================
# 5. 期間頑健性（ローリング分析）
# ============================================================
print("\n" + "=" * 60)
print("5. ローリング分析")
print("=" * 60)

for window in [126, 252]:
    rolling_sharpe_2559 = ret_2559.rolling(window).apply(lambda x: x.mean()/x.std()*np.sqrt(252), raw=True)
    rolling_sharpe_2634 = ret_2634.rolling(window).apply(lambda x: x.mean()/x.std()*np.sqrt(252), raw=True)
    rolling_diff = (rolling_sharpe_2559 - rolling_sharpe_2634).dropna()

    pct_2559_wins = (rolling_diff > 0).mean() * 100
    print(f"\nローリング{window}日 Sharpe:")
    print(f"  2559優位の割合: {pct_2559_wins:.1f}%")
    print(f"  2634優位の割合: {100-pct_2559_wins:.1f}%")
    print(f"  Sharpe差 平均: {rolling_diff.mean():.3f}")
    print(f"  Sharpe差 最大(2559優位): {rolling_diff.max():.3f}")
    print(f"  Sharpe差 最小(2634優位): {rolling_diff.min():.3f}")

# ============================================================
# 6. レジーム別分析
# ============================================================
print("\n" + "=" * 60)
print("6. レジーム別分析")
print("=" * 60)

# VIXレジーム
vix_aligned = vix_hist['Close'].reindex(ret_2559.index, method='ffill')
vix_median = vix_aligned.median()
high_vix = vix_aligned > vix_median
low_vix = ~high_vix

print(f"\n[VIXレジーム (中央値: {vix_median:.1f})]")
for label, mask in [('VIX高(リスクオフ)', high_vix), ('VIX低(リスクオン)', low_vix)]:
    m = mask.reindex(ret_2559.index).fillna(False)
    r1 = ret_2559[m].dropna()
    r2 = ret_2634[m].dropna()
    if len(r1) > 20:
        s1 = r1.mean()/r1.std()*np.sqrt(252)
        s2 = r2.mean()/r2.std()*np.sqrt(252)
        print(f"  {label}: n={len(r1)}")
        print(f"    2559 年率リターン: {r1.mean()*252*100:.1f}%, Sharpe: {s1:.3f}")
        print(f"    2634 年率リターン: {r2.mean()*252*100:.1f}%, Sharpe: {s2:.3f}")
        print(f"    差: Sharpe {s1-s2:+.3f}")

# 円高/円安レジーム
fx_aligned = fx_hist['Close'].reindex(ret_2559.index, method='ffill')
fx_ret_aligned = fx_aligned.pct_change()
yen_weak = fx_ret_aligned > 0  # ドル高円安
yen_strong = fx_ret_aligned <= 0  # ドル安円高

print(f"\n[為替レジーム]")
for label, mask in [('円安局面(USD/JPY上昇)', yen_weak), ('円高局面(USD/JPY下落)', yen_strong)]:
    m = mask.reindex(ret_2559.index).fillna(False)
    r1 = ret_2559[m].dropna()
    r2 = ret_2634[m].dropna()
    if len(r1) > 20:
        s1 = r1.mean()/r1.std()*np.sqrt(252) if r1.std() > 0 else 0
        s2 = r2.mean()/r2.std()*np.sqrt(252) if r2.std() > 0 else 0
        print(f"  {label}: n={len(r1)}")
        print(f"    2559 年率リターン: {r1.mean()*252*100:.1f}%, Sharpe: {s1:.3f}")
        print(f"    2634 年率リターン: {r2.mean()*252*100:.1f}%, Sharpe: {s2:.3f}")
        print(f"    差: Sharpe {s1-s2:+.3f}")

# 上昇/下落局面（ACWI基準）
acwi_aligned = acwi_hist['Close'].reindex(ret_2559.index, method='ffill')
acwi_ret_aligned = acwi_aligned.pct_change()
bull = acwi_ret_aligned > 0
bear = acwi_ret_aligned <= 0

print(f"\n[相場局面レジーム(ACWI基準)]")
for label, mask in [('上昇局面(ACWI上昇日)', bull), ('下落局面(ACWI下落日)', bear)]:
    m = mask.reindex(ret_2559.index).fillna(False)
    r1 = ret_2559[m].dropna()
    r2 = ret_2634[m].dropna()
    if len(r1) > 20:
        print(f"  {label}: n={len(r1)}")
        print(f"    2559 平均日次: {r1.mean()*100:.3f}%")
        print(f"    2634 平均日次: {r2.mean()*100:.3f}%")
        print(f"    差: {(r1.mean()-r2.mean())*100:+.3f}%")

# ============================================================
# 7. コスト・乖離の補助診断
# ============================================================
print("\n" + "=" * 60)
print("7. コスト・構造比較")
print("=" * 60)

print("""
[ETF構造比較]
                        2559                    2634
対象指数              MSCI ACWI               S&P500
為替ヘッジ            なし                    あり（円ヘッジ）
信託報酬(税込)        0.0858%                 0.077%
運用会社              三菱UFJアセット          野村アセット
純資産                約965億円               約343億円
分配頻度              年2回(6月/12月)          年2回(3月/9月)
上場日                2020/01/09              2021/03/25
構成銘柄数            約2,800                 約500
地域分散              全世界(米国約60%)        米国100%

[構造的差異の影響]
1. 為替: 2559は円建てリターンが為替変動の影響を受ける
   → 円安局面では2559有利、円高局面では2634有利
2. 指数: ACWIはS&P500に比べ分散が広い(欧州・新興国含む)
   → 米国一強局面では2634有利、分散効果局面では2559有利
3. ヘッジコスト: 2634は日米金利差分のヘッジコスト負担あり
   → 金利差拡大局面では2634にコスト負担増
4. 信託報酬差: 0.009%とごくわずか
""")

# ============================================================
# 8. 図の作成
# ============================================================
print("=" * 60)
print("8. 図の作成")
print("=" * 60)

# --- 図1: トータルリターン推移 ---
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(tr_2559_c.index, tr_2559_c.values, label='2559 (ACWI, Unhedged)', linewidth=1.5, color='#1565C0')
ax.plot(tr_2634_c.index, tr_2634_c.values, label='2634 (S&P500, Yen-Hedged)', linewidth=1.5, color='#C62828')
ax.set_title('Total Return Index (Common Period, 100 Start)', fontsize=14)
ax.set_ylabel('Index')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/01_total_return.png')
plt.close()
print("  01_total_return.png saved")

# --- 図2: ドローダウン比較 ---
dd_2559 = (tr_2559_c - tr_2559_c.cummax()) / tr_2559_c.cummax()
dd_2634 = (tr_2634_c - tr_2634_c.cummax()) / tr_2634_c.cummax()

fig, ax = plt.subplots(figsize=(14, 5))
ax.fill_between(dd_2559.index, dd_2559.values * 100, 0, alpha=0.4, label='2559 DD', color='#1565C0')
ax.fill_between(dd_2634.index, dd_2634.values * 100, 0, alpha=0.4, label='2634 DD', color='#C62828')
ax.set_title('Drawdown Comparison (%)', fontsize=14)
ax.set_ylabel('Drawdown (%)')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/02_drawdown.png')
plt.close()
print("  02_drawdown.png saved")

# --- 図3: ローリングSharpe差 ---
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

for i, window in enumerate([126, 252]):
    rs1 = ret_2559.rolling(window).apply(lambda x: x.mean()/x.std()*np.sqrt(252), raw=True)
    rs2 = ret_2634.rolling(window).apply(lambda x: x.mean()/x.std()*np.sqrt(252), raw=True)
    diff = (rs1 - rs2).dropna()

    ax = axes[i]
    ax.plot(diff.index, diff.values, linewidth=1, color='#333333')
    ax.fill_between(diff.index, diff.values, 0, where=diff.values > 0, alpha=0.4, color='#1565C0', label='2559 > 2634')
    ax.fill_between(diff.index, diff.values, 0, where=diff.values <= 0, alpha=0.4, color='#C62828', label='2634 > 2559')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_title(f'Rolling {window}-day Sharpe Difference (2559 - 2634)', fontsize=13)
    ax.set_ylabel('Sharpe Diff')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/03_rolling_sharpe_diff.png')
plt.close()
print("  03_rolling_sharpe_diff.png saved")

# --- 図4: ブートストラップ分布 ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].hist(boot_sharpe_diff, bins=80, color='#666', alpha=0.7, edgecolor='white')
axes[0].axvline(0, color='red', linewidth=2, linestyle='--')
axes[0].axvline(np.percentile(boot_sharpe_diff, 2.5), color='blue', linewidth=1, linestyle=':')
axes[0].axvline(np.percentile(boot_sharpe_diff, 97.5), color='blue', linewidth=1, linestyle=':')
axes[0].set_title('Bootstrap: Sharpe Diff (2559-2634)')
axes[0].set_xlabel('Sharpe Difference')

axes[1].hist(boot_cagr_diff*100, bins=80, color='#666', alpha=0.7, edgecolor='white')
axes[1].axvline(0, color='red', linewidth=2, linestyle='--')
axes[1].axvline(np.percentile(boot_cagr_diff*100, 2.5), color='blue', linewidth=1, linestyle=':')
axes[1].axvline(np.percentile(boot_cagr_diff*100, 97.5), color='blue', linewidth=1, linestyle=':')
axes[1].set_title('Bootstrap: CAGR Diff (2559-2634)')
axes[1].set_xlabel('CAGR Diff (%)')

axes[2].hist(boot_mdd_diff*100, bins=80, color='#666', alpha=0.7, edgecolor='white')
axes[2].axvline(0, color='red', linewidth=2, linestyle='--')
axes[2].axvline(np.percentile(boot_mdd_diff*100, 2.5), color='blue', linewidth=1, linestyle=':')
axes[2].axvline(np.percentile(boot_mdd_diff*100, 97.5), color='blue', linewidth=1, linestyle=':')
axes[2].set_title('Bootstrap: MDD Diff (2559-2634, +ve=2559 better)')
axes[2].set_xlabel('MDD Diff (%)')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/04_bootstrap_distributions.png')
plt.close()
print("  04_bootstrap_distributions.png saved")

# --- 図5: レジーム別棒グラフ ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# VIXレジーム
regimes_vix = {}
for label, mask in [('VIX High', high_vix), ('VIX Low', low_vix)]:
    m = mask.reindex(ret_2559.index).fillna(False)
    r1, r2 = ret_2559[m].dropna(), ret_2634[m].dropna()
    if len(r1) > 20:
        regimes_vix[label] = {
            '2559': r1.mean()*252*100,
            '2634': r2.mean()*252*100
        }

if regimes_vix:
    x = np.arange(len(regimes_vix))
    w = 0.35
    ax = axes[0]
    ax.bar(x - w/2, [v['2559'] for v in regimes_vix.values()], w, label='2559', color='#1565C0')
    ax.bar(x + w/2, [v['2634'] for v in regimes_vix.values()], w, label='2634', color='#C62828')
    ax.set_xticks(x)
    ax.set_xticklabels(regimes_vix.keys())
    ax.set_title('Ann. Return by VIX Regime')
    ax.set_ylabel('Ann. Return (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)

# 為替レジーム
regimes_fx = {}
for label, mask in [('Yen Weak\n(USD/JPY up)', yen_weak), ('Yen Strong\n(USD/JPY down)', yen_strong)]:
    m = mask.reindex(ret_2559.index).fillna(False)
    r1, r2 = ret_2559[m].dropna(), ret_2634[m].dropna()
    if len(r1) > 20:
        regimes_fx[label] = {
            '2559': r1.mean()*252*100,
            '2634': r2.mean()*252*100
        }

if regimes_fx:
    x = np.arange(len(regimes_fx))
    ax = axes[1]
    ax.bar(x - w/2, [v['2559'] for v in regimes_fx.values()], w, label='2559', color='#1565C0')
    ax.bar(x + w/2, [v['2634'] for v in regimes_fx.values()], w, label='2634', color='#C62828')
    ax.set_xticks(x)
    ax.set_xticklabels(regimes_fx.keys())
    ax.set_title('Ann. Return by FX Regime')
    ax.set_ylabel('Ann. Return (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)

# 相場局面
regimes_mkt = {}
for label, mask in [('Bull\n(ACWI up)', bull), ('Bear\n(ACWI down)', bear)]:
    m = mask.reindex(ret_2559.index).fillna(False)
    r1, r2 = ret_2559[m].dropna(), ret_2634[m].dropna()
    if len(r1) > 20:
        regimes_mkt[label] = {
            '2559': r1.mean()*252*100,
            '2634': r2.mean()*252*100
        }

if regimes_mkt:
    x = np.arange(len(regimes_mkt))
    ax = axes[2]
    ax.bar(x - w/2, [v['2559'] for v in regimes_mkt.values()], w, label='2559', color='#1565C0')
    ax.bar(x + w/2, [v['2634'] for v in regimes_mkt.values()], w, label='2634', color='#C62828')
    ax.set_xticks(x)
    ax.set_xticklabels(regimes_mkt.keys())
    ax.set_title('Ann. Return by Market Regime')
    ax.set_ylabel('Ann. Return (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/05_regime_analysis.png')
plt.close()
print("  05_regime_analysis.png saved")

# --- 図6: 年次リターン比較 ---
yearly_2559 = tr_2559_c.resample('YE').last().pct_change().dropna()
yearly_2634 = tr_2634_c.resample('YE').last().pct_change().dropna()
common_y = yearly_2559.index.intersection(yearly_2634.index)

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(common_y))
w = 0.35
ax.bar(x - w/2, yearly_2559.loc[common_y].values * 100, w, label='2559', color='#1565C0')
ax.bar(x + w/2, yearly_2634.loc[common_y].values * 100, w, label='2634', color='#C62828')
ax.set_xticks(x)
ax.set_xticklabels([d.strftime('%Y') for d in common_y])
ax.set_title('Annual Total Return Comparison (%)')
ax.set_ylabel('Return (%)')
ax.legend()
ax.grid(True, alpha=0.3)
ax.axhline(0, color='black', linewidth=0.8)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/06_annual_returns.png')
plt.close()
print("  06_annual_returns.png saved")

# 年次リターン表示
print("\n年次リターン:")
print(f"{'年':<6} {'2559':>10} {'2634':>10} {'差':>10}")
for d in common_y:
    v1 = yearly_2559.loc[d] * 100
    v2 = yearly_2634.loc[d] * 100
    print(f"{d.strftime('%Y'):<6} {v1:>9.1f}% {v2:>9.1f}% {v1-v2:>+9.1f}%")

print("\n" + "=" * 60)
print("全出力完了")
print(f"図表出力先: {OUTPUT_DIR}/")
print("=" * 60)
