#!/usr/bin/env python3
"""
NEC (6701) vs 同業8社 比較分析
- 株価モメンタム・ボラティリティ
- バリュエーション（PER/PBR）
- 決算の質（業績進捗率・アナリスト目標株価との乖離）
- 短期トレード適性（+10%到達バックテスト）
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 銘柄定義
# ============================================================
PEERS = {
    '6701.T': {'name': 'NEC',       'sector': 'IT総合・防衛'},
    '6702.T': {'name': '富士通',     'sector': 'IT総合'},
    '6501.T': {'name': '日立製作所', 'sector': '総合電機・IT'},
    '9613.T': {'name': 'NTTデータ',  'sector': 'SIer'},
    '3626.T': {'name': 'TIS',       'sector': 'SIer'},
    '9719.T': {'name': 'SCSK',      'sector': 'SIer'},
    '6503.T': {'name': '三菱電機',   'sector': '総合電機・防衛'},
    '4307.T': {'name': 'NRI',       'sector': 'コンサル・SIer'},
    '9432.T': {'name': 'NTT',       'sector': '通信・IT'},
}

end_date = datetime(2026, 2, 18)
start_date = end_date - timedelta(days=400)  # 約1年+α

# ============================================================
# データ取得
# ============================================================
print("=" * 80)
print("データ取得中...")
print("=" * 80)

all_data = {}
info_data = {}

for ticker_str, meta in PEERS.items():
    try:
        tkr = yf.Ticker(ticker_str)
        df = tkr.history(start=start_date.strftime('%Y-%m-%d'),
                         end=end_date.strftime('%Y-%m-%d'), interval='1d')
        info = tkr.info
        all_data[ticker_str] = df
        info_data[ticker_str] = info
        print(f"  {meta['name']:8s} ({ticker_str}): {len(df)}行取得, 直近終値={df['Close'].iloc[-1]:.0f}円")
    except Exception as e:
        print(f"  {meta['name']:8s} ({ticker_str}): 取得失敗 - {e}")

# ============================================================
# 指標計算
# ============================================================
print("\n" + "=" * 80)
print("指標計算中...")
print("=" * 80)

results = []

for ticker_str, meta in PEERS.items():
    if ticker_str not in all_data:
        continue

    df = all_data[ticker_str]
    info = info_data.get(ticker_str, {})

    if len(df) < 60:
        continue

    # --- 価格・リターン ---
    current_price = df['Close'].iloc[-1]
    df['Return'] = df['Close'].pct_change()

    # 各期間リターン
    ret_5d  = (df['Close'].iloc[-1] / df['Close'].iloc[-6] - 1) * 100 if len(df) >= 6 else np.nan
    ret_20d = (df['Close'].iloc[-1] / df['Close'].iloc[-21] - 1) * 100 if len(df) >= 21 else np.nan
    ret_60d = (df['Close'].iloc[-1] / df['Close'].iloc[-61] - 1) * 100 if len(df) >= 61 else np.nan

    # 52週高値からの下落率
    high_52w = df['High'].max()
    low_52w  = df['Low'].min()
    drawdown_from_high = (current_price / high_52w - 1) * 100

    # --- ボラティリティ ---
    vol_20d = df['Return'].tail(20).std() * np.sqrt(252) * 100
    vol_60d = df['Return'].tail(60).std() * np.sqrt(252) * 100

    # ATR(14)
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    atr14 = df['TR'].rolling(14).mean().iloc[-1]
    atr14_pct = atr14 / current_price * 100

    # --- RSI(14) ---
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rsi14 = (100 - (100 / (1 + gain / loss))).iloc[-1]

    # --- 5MA / 20MA トレンド ---
    ma5  = df['Close'].rolling(5).mean().iloc[-1]
    ma20 = df['Close'].rolling(20).mean().iloc[-1]
    above_ma5  = current_price > ma5
    above_ma20 = current_price > ma20

    # --- バリュエーション（yfinance info） ---
    pe_trailing = info.get('trailingPE', np.nan)
    pe_forward  = info.get('forwardPE', np.nan)
    pb          = info.get('priceToBook', np.nan)
    target_mean = info.get('targetMeanPrice', np.nan)
    target_upside = ((target_mean / current_price) - 1) * 100 if not np.isnan(target_mean) and target_mean > 0 else np.nan

    # --- +10% 到達バックテスト（5MA上抜けシグナル、SL -5%、最大40日） ---
    df['MA5'] = df['Close'].rolling(5).mean()
    df['PrevClose'] = df['Close'].shift(1)
    df['PrevMA5'] = df['MA5'].shift(1)
    df['Bullish'] = df['Close'] > df['Open']

    signals = []
    for i in range(1, len(df)):
        if (not pd.isna(df['MA5'].iloc[i]) and
            not pd.isna(df['PrevMA5'].iloc[i]) and
            df['PrevClose'].iloc[i] <= df['PrevMA5'].iloc[i] and
            df['Close'].iloc[i] > df['MA5'].iloc[i] and
            df['Bullish'].iloc[i] and
            rsi14 < 70):
            signals.append(i)

    # バックテスト
    wins = 0
    stops = 0
    total = 0
    pnl_list = []

    for sig_pos in signals:
        entry_price = df['Close'].iloc[sig_pos]
        tp = entry_price * 1.10
        sl = entry_price * 0.95
        exit_pnl = None

        for d in range(1, 41):
            if sig_pos + d >= len(df):
                exit_pnl = (df['Close'].iloc[-1] / entry_price - 1)
                break
            day = df.iloc[sig_pos + d]
            if day['Low'] <= sl:
                exit_pnl = -0.05
                stops += 1
                break
            if day['High'] >= tp:
                exit_pnl = 0.10
                wins += 1
                break
        if exit_pnl is None:
            if sig_pos + 40 < len(df):
                exit_pnl = (df['Close'].iloc[sig_pos + 40] / entry_price - 1)
            else:
                exit_pnl = (df['Close'].iloc[-1] / entry_price - 1)

        pnl_list.append(exit_pnl)
        total += 1

    win_rate = (wins / total * 100) if total > 0 else np.nan
    avg_pnl  = (np.mean(pnl_list) * 100) if pnl_list else np.nan

    results.append({
        'ticker': ticker_str,
        'name': meta['name'],
        'sector': meta['sector'],
        'price': current_price,
        'ret_5d': ret_5d,
        'ret_20d': ret_20d,
        'ret_60d': ret_60d,
        'dd_from_high': drawdown_from_high,
        'vol_20d': vol_20d,
        'vol_60d': vol_60d,
        'atr14_pct': atr14_pct,
        'rsi14': rsi14,
        'above_ma5': above_ma5,
        'above_ma20': above_ma20,
        'pe_trailing': pe_trailing,
        'pe_forward': pe_forward,
        'pb': pb,
        'target_upside': target_upside,
        'bt_trades': total,
        'bt_winrate': win_rate,
        'bt_avg_pnl': avg_pnl,
    })

res_df = pd.DataFrame(results)

# ============================================================
# 結果表示
# ============================================================
print("\n" + "=" * 80)
print("【1】 株価モメンタム比較")
print("=" * 80)
fmt1 = "{:<10s} {:>7.0f} {:>8.1f}% {:>8.1f}% {:>8.1f}% {:>10.1f}%"
hdr1 = "{:<10s} {:>7s} {:>9s} {:>9s} {:>9s} {:>10s}"
print(hdr1.format('銘柄', '株価', '5日', '20日', '60日', '高値比'))
print("-" * 60)
for _, r in res_df.iterrows():
    print(fmt1.format(r['name'], r['price'], r['ret_5d'], r['ret_20d'], r['ret_60d'], r['dd_from_high']))

print("\n" + "=" * 80)
print("【2】 ボラティリティ・テクニカル比較")
print("=" * 80)
fmt2 = "{:<10s} {:>8.1f}% {:>8.1f}% {:>8.1f}% {:>6.1f} {:>5s} {:>5s}"
hdr2 = "{:<10s} {:>9s} {:>9s} {:>9s} {:>6s} {:>5s} {:>5s}"
print(hdr2.format('銘柄', 'Vol20日', 'Vol60日', 'ATR14%', 'RSI14', '>MA5', '>MA20'))
print("-" * 60)
for _, r in res_df.iterrows():
    print(fmt2.format(r['name'], r['vol_20d'], r['vol_60d'], r['atr14_pct'],
                      r['rsi14'],
                      '○' if r['above_ma5'] else '×',
                      '○' if r['above_ma20'] else '×'))

print("\n" + "=" * 80)
print("【3】 バリュエーション比較")
print("=" * 80)
fmt3 = "{:<10s} {:>8s} {:>8s} {:>6s} {:>10s}"
hdr3 = "{:<10s} {:>8s} {:>8s} {:>6s} {:>10s}"
print(hdr3.format('銘柄', 'PER実績', 'PER予想', 'PBR', '目標乖離%'))
print("-" * 50)
for _, r in res_df.iterrows():
    pe_t  = f"{r['pe_trailing']:.1f}" if not np.isnan(r['pe_trailing']) else "N/A"
    pe_f  = f"{r['pe_forward']:.1f}" if not np.isnan(r['pe_forward']) else "N/A"
    pb_s  = f"{r['pb']:.2f}" if not np.isnan(r['pb']) else "N/A"
    tgt_s = f"{r['target_upside']:+.1f}%" if not np.isnan(r['target_upside']) else "N/A"
    print(fmt3.format(r['name'], pe_t, pe_f, pb_s, tgt_s))

print("\n" + "=" * 80)
print("【4】 +10%到達バックテスト比較（5MA上抜け、SL -5%、最大40日）")
print("=" * 80)
fmt4 = "{:<10s} {:>6d} {:>8s} {:>10s}"
hdr4 = "{:<10s} {:>6s} {:>8s} {:>10s}"
print(hdr4.format('銘柄', 'trade数', '勝率', '平均損益'))
print("-" * 40)
for _, r in res_df.iterrows():
    wr = f"{r['bt_winrate']:.1f}%" if not np.isnan(r['bt_winrate']) else "N/A"
    ap = f"{r['bt_avg_pnl']:+.2f}%" if not np.isnan(r['bt_avg_pnl']) else "N/A"
    print(fmt4.format(r['name'], r['bt_trades'], wr, ap))

# ============================================================
# 総合スコアリング
# ============================================================
print("\n" + "=" * 80)
print("【5】 総合スコアリング（短期+10%トレード適性）")
print("=" * 80)

def score_stock(r):
    """各観点にスコアを付けて合計"""
    s = 0
    reasons = []

    # 1. 目標株価との乖離（上昇余地）: 大きいほど良い
    if not np.isnan(r['target_upside']):
        if r['target_upside'] >= 30:
            s += 3; reasons.append(f"目標乖離{r['target_upside']:+.0f}% ★★★")
        elif r['target_upside'] >= 15:
            s += 2; reasons.append(f"目標乖離{r['target_upside']:+.0f}% ★★")
        elif r['target_upside'] >= 5:
            s += 1; reasons.append(f"目標乖離{r['target_upside']:+.0f}% ★")
        else:
            reasons.append(f"目標乖離{r['target_upside']:+.0f}% (低い)")

    # 2. RSI: 売られ過ぎ = リバウンド余地
    if r['rsi14'] < 30:
        s += 2; reasons.append(f"RSI{r['rsi14']:.0f} 売られ過ぎ ★★")
    elif r['rsi14'] < 40:
        s += 1; reasons.append(f"RSI{r['rsi14']:.0f} やや売られ過ぎ ★")
    elif r['rsi14'] > 70:
        s -= 1; reasons.append(f"RSI{r['rsi14']:.0f} 過熱 ▼")

    # 3. 高値からの下落率: 大きい下落 = リバウンド余地 (ただし下落し過ぎは危険)
    if -35 < r['dd_from_high'] <= -20:
        s += 2; reasons.append(f"高値比{r['dd_from_high']:.0f}% 反発余地 ★★")
    elif -20 < r['dd_from_high'] <= -10:
        s += 1; reasons.append(f"高値比{r['dd_from_high']:.0f}% やや割安 ★")
    elif r['dd_from_high'] <= -35:
        s -= 1; reasons.append(f"高値比{r['dd_from_high']:.0f}% 下落過大 ▼")

    # 4. バックテスト勝率
    if not np.isnan(r['bt_winrate']):
        if r['bt_winrate'] >= 50:
            s += 2; reasons.append(f"BT勝率{r['bt_winrate']:.0f}% ★★")
        elif r['bt_winrate'] >= 35:
            s += 1; reasons.append(f"BT勝率{r['bt_winrate']:.0f}% ★")
        else:
            reasons.append(f"BT勝率{r['bt_winrate']:.0f}%")

    # 5. ボラティリティ（適度が良い）
    if 25 <= r['vol_20d'] <= 50:
        s += 1; reasons.append(f"Vol{r['vol_20d']:.0f}% 適度 ★")
    elif r['vol_20d'] > 70:
        s -= 1; reasons.append(f"Vol{r['vol_20d']:.0f}% 過大 ▼")

    # 6. PER(forward) が割安
    if not np.isnan(r['pe_forward']):
        if r['pe_forward'] < 15:
            s += 1; reasons.append(f"PER予{r['pe_forward']:.0f}倍 割安 ★")
        elif r['pe_forward'] > 35:
            s -= 1; reasons.append(f"PER予{r['pe_forward']:.0f}倍 割高 ▼")

    return s, reasons

print()
scored = []
for _, r in res_df.iterrows():
    score, reasons = score_stock(r)
    scored.append({'name': r['name'], 'ticker': r['ticker'], 'score': score, 'reasons': reasons})

scored.sort(key=lambda x: x['score'], reverse=True)
for rank, item in enumerate(scored, 1):
    bar = "█" * max(item['score'], 0) + "░" * max(0, 10 - max(item['score'], 0))
    print(f"  {rank}. {item['name']:8s} スコア: {item['score']:+2d}  {bar}")
    for reason in item['reasons']:
        print(f"       └ {reason}")
    print()

# ============================================================
# JSON出力（HTML用）
# ============================================================
import json

json_out = []
for _, r in res_df.iterrows():
    score, reasons = score_stock(r)
    json_out.append({
        'ticker': r['ticker'],
        'name': r['name'],
        'sector': r['sector'],
        'price': round(r['price'], 0),
        'ret_5d': round(r['ret_5d'], 1),
        'ret_20d': round(r['ret_20d'], 1),
        'ret_60d': round(r['ret_60d'], 1),
        'dd_from_high': round(r['dd_from_high'], 1),
        'vol_20d': round(r['vol_20d'], 1),
        'atr14_pct': round(r['atr14_pct'], 1),
        'rsi14': round(r['rsi14'], 1),
        'above_ma5': bool(r['above_ma5']),
        'above_ma20': bool(r['above_ma20']),
        'pe_trailing': round(r['pe_trailing'], 1) if not np.isnan(r['pe_trailing']) else None,
        'pe_forward': round(r['pe_forward'], 1) if not np.isnan(r['pe_forward']) else None,
        'pb': round(r['pb'], 2) if not np.isnan(r['pb']) else None,
        'target_upside': round(r['target_upside'], 1) if not np.isnan(r['target_upside']) else None,
        'bt_trades': int(r['bt_trades']),
        'bt_winrate': round(r['bt_winrate'], 1) if not np.isnan(r['bt_winrate']) else None,
        'bt_avg_pnl': round(r['bt_avg_pnl'], 2) if not np.isnan(r['bt_avg_pnl']) else None,
        'score': score,
        'reasons': reasons,
    })

with open('/home/like_rapid/GT-SOAR/peers_data.json', 'w', encoding='utf-8') as f:
    json.dump(json_out, f, ensure_ascii=False, indent=2)

print("peers_data.json に出力完了")
print("\n===== 比較分析完了 =====")
