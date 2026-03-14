#!/usr/bin/env python3
"""
NEC (6701.T) 短期信用買い戦略バックテスト
戦略A: 確認型（5MA上抜け確認）
戦略B: 逆張り型（支持帯で反発ピック）
損切り: -3% / -5% / -6% の3案比較
利確: +10% 固定
最大保有: 40営業日
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. データ取得
# ============================================================
print("=" * 70)
print("1. データ取得")
print("=" * 70)

ticker = yf.Ticker("6701.T")

# 日足: 過去3年
end_date = datetime(2026, 2, 18)
start_date_3y = end_date - timedelta(days=3*365+30)
df_daily = ticker.history(start=start_date_3y.strftime('%Y-%m-%d'),
                          end=end_date.strftime('%Y-%m-%d'),
                          interval='1d')
print(f"日足データ: {len(df_daily)}行, 期間: {df_daily.index[0].strftime('%Y-%m-%d')} ~ {df_daily.index[-1].strftime('%Y-%m-%d')}")
print(f"直近終値: {df_daily['Close'].iloc[-1]:.0f}円")
print(f"直近5日の終値: {df_daily['Close'].tail(5).values}")

# 60分足: 直近取得可能な範囲
try:
    df_hourly = ticker.history(period='60d', interval='60m')
    print(f"60分足データ: {len(df_hourly)}行")
    hourly_available = True
except Exception as e:
    print(f"60分足取得失敗: {e}")
    hourly_available = False

# ============================================================
# 2. テクニカル指標計算
# ============================================================
df = df_daily.copy()
df['MA5'] = df['Close'].rolling(5).mean()
df['MA20'] = df['Close'].rolling(20).mean()
df['MA60'] = df['Close'].rolling(60).mean()

# ATR(14)
df['H-L'] = df['High'] - df['Low']
df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
df['ATR14'] = df['TR'].rolling(14).mean()

# RSI(14)
delta = df['Close'].diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
df['RSI14'] = 100 - (100 / (1 + gain / loss))

# ボリンジャーバンド(20, 2)
df['BB_mid'] = df['MA20']
df['BB_std'] = df['Close'].rolling(20).std()
df['BB_lower'] = df['BB_mid'] - 2 * df['BB_std']
df['BB_upper'] = df['BB_mid'] + 2 * df['BB_std']

# 日次リターン
df['Return'] = df['Close'].pct_change()

# 下ヒゲ比率
df['LowerShadow'] = df[['Open', 'Close']].min(axis=1) - df['Low']
df['BodySize'] = abs(df['Close'] - df['Open'])
df['CandleRange'] = df['High'] - df['Low']
df['LowerShadowRatio'] = df['LowerShadow'] / df['CandleRange'].replace(0, np.nan)

# 陽線フラグ
df['Bullish'] = df['Close'] > df['Open']

# 前日比
df['PrevClose'] = df['Close'].shift(1)
df['PrevMA5'] = df['MA5'].shift(1)

print(f"\nATR14(直近): {df['ATR14'].iloc[-1]:.0f}円")
print(f"RSI14(直近): {df['RSI14'].iloc[-1]:.1f}")
print(f"20日ボラティリティ(年率): {df['Return'].tail(20).std() * np.sqrt(252) * 100:.1f}%")

# ============================================================
# 3. 4000円帯の支持/抵抗分析
# ============================================================
print("\n" + "=" * 70)
print("2. 4,000円帯（3,980〜4,020）の支持/抵抗分析")
print("=" * 70)

zone_low, zone_high = 3980, 4020

# 4000帯を通過した回数
touches = df[(df['Low'] <= zone_high) & (df['High'] >= zone_low)]
print(f"4,000帯タッチ回数（全期間）: {len(touches)}回")

# 4000帯で反発した回数（安値が帯内 or 帯付近で、終値が帯より上）
bounces = df[(df['Low'] >= zone_low - 100) & (df['Low'] <= zone_high) & (df['Close'] > zone_high)]
print(f"4,000帯付近で反発（終値>4,020）: {len(bounces)}回")

# 4000帯を下抜けた回数
breakdowns = df[(df['Open'] >= zone_low) & (df['Close'] < zone_low)]
print(f"4,000帯を下抜け（終値<3,980）: {len(breakdowns)}回")

# 出来高の集中度
if len(touches) > 0:
    avg_vol_all = df['Volume'].mean()
    avg_vol_zone = touches['Volume'].mean()
    print(f"帯付近の平均出来高 vs 全体平均: {avg_vol_zone/avg_vol_all:.2f}倍")

# ============================================================
# 4. バックテスト関数
# ============================================================
def run_backtest(df, entry_signals, stop_loss_pct, take_profit_pct=0.10, max_hold=40, label=""):
    """
    entry_signals: エントリー日のインデックスリスト
    stop_loss_pct: 損切り幅（負の値、例: -0.03）
    take_profit_pct: 利確幅（正の値、例: 0.10）
    max_hold: 最大保有営業日数
    """
    results = []

    for entry_idx in entry_signals:
        pos = df.index.get_loc(entry_idx)
        entry_price = df['Close'].iloc[pos]  # 終値でエントリー

        tp_price = entry_price * (1 + take_profit_pct)
        sl_price = entry_price * (1 + stop_loss_pct)

        exit_reason = 'time_exit'
        exit_price = None
        hold_days = 0
        max_dd = 0

        for d in range(1, max_hold + 1):
            if pos + d >= len(df):
                exit_reason = 'data_end'
                exit_price = df['Close'].iloc[-1]
                hold_days = d
                break

            day = df.iloc[pos + d]
            low = day['Low']
            high = day['High']
            close = day['Close']

            # 日中の最大ドローダウン
            dd = (low - entry_price) / entry_price
            if dd < max_dd:
                max_dd = dd

            # 損切り判定（安値ベース）
            if low <= sl_price:
                exit_reason = 'stop_loss'
                exit_price = sl_price  # 逆指値で約定と仮定
                hold_days = d
                break

            # 利確判定（高値ベース）
            if high >= tp_price:
                exit_reason = 'take_profit'
                exit_price = tp_price
                hold_days = d
                break

            hold_days = d
            exit_price = close

        if exit_price is None:
            if pos + max_hold < len(df):
                exit_price = df['Close'].iloc[pos + max_hold]
                hold_days = max_hold
            else:
                exit_price = df['Close'].iloc[-1]
                hold_days = len(df) - pos - 1

        pnl_pct = (exit_price - entry_price) / entry_price

        results.append({
            'entry_date': entry_idx,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'pnl_pct': pnl_pct,
            'hold_days': hold_days,
            'max_dd': max_dd
        })

    if not results:
        return None

    res_df = pd.DataFrame(results)

    n = len(res_df)
    wins = len(res_df[res_df['exit_reason'] == 'take_profit'])
    stops = len(res_df[res_df['exit_reason'] == 'stop_loss'])
    time_exits = len(res_df[res_df['exit_reason'] == 'time_exit'])

    win_rate = wins / n * 100 if n > 0 else 0
    stop_rate = stops / n * 100 if n > 0 else 0
    avg_pnl = res_df['pnl_pct'].mean() * 100
    expected_value = avg_pnl  # 期待値 = 平均損益
    worst_dd = res_df['max_dd'].min() * 100
    avg_hold_win = res_df[res_df['exit_reason'] == 'take_profit']['hold_days'].mean() if wins > 0 else np.nan

    stats = {
        'label': label,
        'trades': n,
        'win_rate': win_rate,
        'stop_rate': stop_rate,
        'time_exit_rate': time_exits / n * 100 if n > 0 else 0,
        'avg_pnl': avg_pnl,
        'expected_value': expected_value,
        'worst_dd': worst_dd,
        'avg_hold_win': avg_hold_win,
    }

    return stats, res_df

# ============================================================
# 5. 戦略定義
# ============================================================

# --- 戦略A: 確認型 ---
# 条件: 終値が5MAを上抜け（前日は5MA以下、当日は5MA以上）
#        かつ 前日比プラス（陽線）
#        かつ RSI14 < 60（過熱でない）
def strategy_a_signals(df):
    signals = []
    for i in range(1, len(df)):
        if (df['PrevClose'].iloc[i] <= df['PrevMA5'].iloc[i] and  # 前日: 5MA以下
            df['Close'].iloc[i] > df['MA5'].iloc[i] and           # 当日: 5MA上抜け
            df['Bullish'].iloc[i] and                              # 陽線
            df['RSI14'].iloc[i] < 60 and                           # 過熱でない
            not pd.isna(df['MA5'].iloc[i])):
            signals.append(df.index[i])
    return signals

# --- 戦略B: 逆張り型 ---
# 条件: 安値が支持帯（エントリー価格の-2%〜+0.5%）に到達
#        かつ 下ヒゲが実体より長い（下ヒゲ比率 > 0.4）
#        かつ 翌日が陽線（翌日の終値 > 翌日の始値）→ 翌日の終値でエントリー
# ※ 実際は「エントリー候補価格」の概念を一般化し、各タッチの安値帯で判定
def strategy_b_signals(df, support_width_pct=0.02):
    """
    一般化版: 直近20日安値付近（-2%以内）で下ヒゲ＋翌日陽線
    """
    signals = []
    for i in range(21, len(df) - 1):
        recent_low = df['Low'].iloc[i-20:i].min()
        price = df['Close'].iloc[i]
        low = df['Low'].iloc[i]

        # 安値が直近20日安値の近辺（-2%〜+2%）
        near_support = (low <= recent_low * 1.02) and (low >= recent_low * 0.98)

        # 下ヒゲが長い
        long_lower_shadow = (df['LowerShadowRatio'].iloc[i] > 0.4) if not pd.isna(df['LowerShadowRatio'].iloc[i]) else False

        # 翌日が陽線
        next_bullish = df['Bullish'].iloc[i + 1] if i + 1 < len(df) else False

        if near_support and long_lower_shadow and next_bullish:
            signals.append(df.index[i + 1])  # 翌日終値でエントリー

    return signals


# ============================================================
# 6. バックテスト実行
# ============================================================
print("\n" + "=" * 70)
print("3. バックテスト結果")
print("=" * 70)

signals_a = strategy_a_signals(df)
signals_b = strategy_b_signals(df)

print(f"\n戦略Aシグナル数（全期間）: {len(signals_a)}")
print(f"戦略Bシグナル数（全期間）: {len(signals_b)}")

stop_losses = [-0.03, -0.05, -0.06]
all_stats = []

for sl in stop_losses:
    sl_label = f"{sl*100:.0f}%"

    # 戦略A
    result_a = run_backtest(df, signals_a, sl, label=f"A (SL {sl_label})")
    if result_a:
        all_stats.append(result_a[0])

    # 戦略B
    result_b = run_backtest(df, signals_b, sl, label=f"B (SL {sl_label})")
    if result_b:
        all_stats.append(result_b[0])

# 直近1年に限定
one_year_ago = end_date - timedelta(days=370)
signals_a_1y = [s for s in signals_a if s >= pd.Timestamp(one_year_ago, tz=signals_a[0].tz if signals_a else None)]
signals_b_1y = [s for s in signals_b if s >= pd.Timestamp(one_year_ago, tz=signals_b[0].tz if signals_b else None)]

print(f"\n戦略Aシグナル数（直近1年）: {len(signals_a_1y)}")
print(f"戦略Bシグナル数（直近1年）: {len(signals_b_1y)}")

stats_1y = []
for sl in stop_losses:
    sl_label = f"{sl*100:.0f}%"
    result_a_1y = run_backtest(df, signals_a_1y, sl, label=f"A-1Y (SL {sl_label})")
    if result_a_1y:
        stats_1y.append(result_a_1y[0])
    result_b_1y = run_backtest(df, signals_b_1y, sl, label=f"B-1Y (SL {sl_label})")
    if result_b_1y:
        stats_1y.append(result_b_1y[0])

# ============================================================
# 7. 結果表示
# ============================================================
print("\n" + "=" * 70)
print("【全期間バックテスト結果】")
print("=" * 70)
cols = ['label', 'trades', 'win_rate', 'stop_rate', 'time_exit_rate', 'avg_pnl', 'expected_value', 'worst_dd', 'avg_hold_win']
headers = ['戦略', 'トレード数', '勝率(%)', 'SLヒット率(%)', '時間撤退率(%)', '平均損益(%)', '期待値(%)', '最大DD(%)', '利確平均日数']

fmt = "{:<16} {:>6} {:>8.1f} {:>10.1f} {:>10.1f} {:>9.2f} {:>8.2f} {:>8.1f} {:>10.1f}"
hdr = "{:<16} {:>6} {:>8} {:>10} {:>10} {:>9} {:>8} {:>8} {:>10}"
print(hdr.format(*headers))
print("-" * 100)
for s in all_stats:
    avg_hold = s['avg_hold_win'] if not np.isnan(s['avg_hold_win']) else 0
    print(fmt.format(s['label'], s['trades'], s['win_rate'], s['stop_rate'],
                     s['time_exit_rate'], s['avg_pnl'], s['expected_value'],
                     s['worst_dd'], avg_hold))

print("\n" + "=" * 70)
print("【直近1年バックテスト結果】")
print("=" * 70)
print(hdr.format(*headers))
print("-" * 100)
for s in stats_1y:
    avg_hold = s['avg_hold_win'] if not np.isnan(s['avg_hold_win']) else 0
    print(fmt.format(s['label'], s['trades'], s['win_rate'], s['stop_rate'],
                     s['time_exit_rate'], s['avg_pnl'], s['expected_value'],
                     s['worst_dd'], avg_hold))

# ============================================================
# 8. ボラティリティとリスク分析
# ============================================================
print("\n" + "=" * 70)
print("4. 直近のボラティリティ分析")
print("=" * 70)

recent_20 = df.tail(20)
recent_60 = df.tail(60)

print(f"直近20日:")
print(f"  日次リターン標準偏差: {recent_20['Return'].std()*100:.2f}%")
print(f"  年率ボラティリティ:   {recent_20['Return'].std()*np.sqrt(252)*100:.1f}%")
print(f"  ATR14:                {df['ATR14'].iloc[-1]:.0f}円 ({df['ATR14'].iloc[-1]/df['Close'].iloc[-1]*100:.1f}%)")
print(f"  最大日次下落:         {recent_20['Return'].min()*100:.2f}%")
print(f"  最大日次上昇:         {recent_20['Return'].max()*100:.2f}%")

print(f"\n直近60日:")
print(f"  日次リターン標準偏差: {recent_60['Return'].std()*100:.2f}%")
print(f"  年率ボラティリティ:   {recent_60['Return'].std()*np.sqrt(252)*100:.1f}%")

# 4000円到達シナリオ分析
current_price = df['Close'].iloc[-1]
target_entry = 4000
print(f"\n現在値: {current_price:.0f}円 → エントリー候補 4,000円 (差: {(target_entry/current_price-1)*100:.1f}%)")

# モンテカルロ的な到達確率（簡易版：正規分布仮定）
daily_mu = recent_60['Return'].mean()
daily_sigma = recent_60['Return'].std()
print(f"\n直近60日の日次リターン: μ={daily_mu*100:.3f}%, σ={daily_sigma*100:.2f}%")

# +10%到達の理論確率（ランダムウォーク仮定、40日）
from scipy import stats as sp_stats

n_sims = 100000
np.random.seed(42)
sim_returns = np.random.normal(daily_mu, daily_sigma, (n_sims, 40))
sim_prices = 4000 * np.cumprod(1 + sim_returns, axis=1)

tp_price = 4400  # +10%
sl_prices = {'-3%': 3880, '-5%': 3800, '-6%': 3760}

print(f"\n--- モンテカルロシミュレーション（10万回, 40営業日, エントリー4,000円）---")
for sl_name, sl_p in sl_prices.items():
    tp_hit = 0
    sl_hit = 0
    time_exit = 0
    pnl_list = []

    for sim in range(n_sims):
        path = sim_prices[sim]
        hit_tp = False
        hit_sl = False

        for d in range(40):
            if path[d] >= tp_price:
                hit_tp = True
                pnl_list.append(0.10)
                break
            if path[d] <= sl_p:
                hit_sl = True
                pnl_list.append((sl_p - 4000) / 4000)
                break

        if hit_tp:
            tp_hit += 1
        elif hit_sl:
            sl_hit += 1
        else:
            time_exit += 1
            pnl_list.append((path[-1] - 4000) / 4000)

    avg_pnl = np.mean(pnl_list) * 100
    print(f"  損切り{sl_name}: 利確到達={tp_hit/n_sims*100:.1f}%, SLヒット={sl_hit/n_sims*100:.1f}%, "
          f"時間撤退={time_exit/n_sims*100:.1f}%, 期待値={avg_pnl:.2f}%")

# ============================================================
# 9. 価格帯別の分析
# ============================================================
print("\n" + "=" * 70)
print("5. 4,000円帯エントリー固有の分析")
print("=" * 70)

# 過去に4000円付近（3900-4100）でエントリーした場合の成績
zone_entries = df[(df['Close'] >= 3900) & (df['Close'] <= 4100)]
print(f"終値3,900〜4,100円の日数: {len(zone_entries)}")

if len(zone_entries) > 0:
    zone_signals = zone_entries.index.tolist()
    print("\n4,000円帯エントリーのバックテスト:")
    print(hdr.format(*headers))
    print("-" * 100)
    for sl in stop_losses:
        sl_label = f"{sl*100:.0f}%"
        result = run_backtest(df, zone_signals, sl, label=f"4K帯 (SL {sl_label})")
        if result:
            s = result[0]
            avg_hold = s['avg_hold_win'] if not np.isnan(s['avg_hold_win']) else 0
            print(fmt.format(s['label'], s['trades'], s['win_rate'], s['stop_rate'],
                             s['time_exit_rate'], s['avg_pnl'], s['expected_value'],
                             s['worst_dd'], avg_hold))

print("\n\n===== バックテスト完了 =====")
