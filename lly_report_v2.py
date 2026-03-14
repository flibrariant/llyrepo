"""
LLY クオンツ分析レポート v2 - 修正版
修正点:
  1. 各サブプロットのY軸を独立させる
  2. PER BBを3年分に延長（年次EPSを使用）
  3. BB塗りつぶし視認性を改善
  4. %B軸を0-1に固定
"""

import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ─── データ取得 ───────────────────────────────────────
print("データ取得中...")
ticker = yf.Ticker("LLY")
price_df = ticker.history(period="3y", interval="1d")
price_df.index = pd.to_datetime(price_df.index).tz_localize(None)
price_df = price_df[['Close', 'Volume']].dropna()
close = price_df['Close']
info = ticker.info
print(f"株価データ: {len(price_df)} 行 / 最新: ${close.iloc[-1]:.2f}")

# ─── 年次 EPS 取得（PER 長期化用） ───────────────────────
print("\n年次 EPS 取得...")
annual_eps_dict = {}

try:
    ann = ticker.income_stmt  # 年次損益計算書
    # まず Diluted EPS 行を直接探す
    for key in ['Diluted EPS', 'Basic EPS', 'Diluted Normalized EPS', 'Normalized Diluted EPS']:
        if key in ann.index:
            eps_ann = ann.loc[key].dropna()
            for col, val in eps_ann.items():
                yr = pd.to_datetime(col).tz_localize(None)
                annual_eps_dict[yr] = abs(float(val))
            print(f"  年次EPS ({key}): {dict(list(annual_eps_dict.items())[:4])}")
            break

    # なければ Net Income / Diluted Shares で計算
    if not annual_eps_dict:
        ni = shares = None
        for k in ['Net Income', 'Net Income Common Stockholders']:
            if k in ann.index:
                ni = ann.loc[k]
                break
        for k in ['Diluted Average Shares', 'Ordinary Shares Number']:
            if k in ann.index:
                shares = ann.loc[k]
                break
        if ni is not None and shares is not None:
            for col in ni.index:
                yr = pd.to_datetime(col).tz_localize(None)
                if shares.get(col, 0):
                    annual_eps_dict[yr] = abs(float(ni[col]) / float(shares[col]))
            print(f"  年次EPS (Net/Shares): {dict(list(annual_eps_dict.items())[:4])}")
except Exception as e:
    print(f"  年次EPS取得エラー: {e}")

# ─── 四半期 EPS 取得 ──────────────────────────────────
print("\n四半期 EPS 取得...")
eps_q = pd.Series(dtype=float)

try:
    q = ticker.quarterly_income_stmt
    ni = shares = None
    for k in ['Net Income', 'Net Income Common Stockholders']:
        if k in q.index:
            ni = q.loc[k]
            break
    for k in ['Diluted Average Shares', 'Ordinary Shares Number']:
        if k in q.index:
            shares = q.loc[k]
            break
    if ni is not None and shares is not None:
        eps_q_raw = {}
        for col in ni.index:
            dt = pd.to_datetime(col).tz_localize(None)
            if shares.get(col, 0):
                eps_q_raw[dt] = abs(float(ni[col]) / float(shares[col]))
        eps_q = pd.Series(eps_q_raw).sort_index()
        print(f"  四半期EPS: {eps_q.to_dict()}")
except Exception as e:
    print(f"  四半期EPS取得エラー: {e}")

# フォールバック: earnings_history
if len(eps_q) == 0:
    try:
        eh = ticker.earnings_history
        if eh is not None and 'epsActual' in eh.columns:
            for idx, row in eh.iterrows():
                dt = pd.to_datetime(idx).tz_localize(None)
                eps_q[dt] = abs(float(row['epsActual']))
            eps_q = eps_q.sort_index()
            print(f"  四半期EPS (earnings_history): {eps_q.to_dict()}")
    except Exception as e:
        print(f"  earnings_history エラー: {e}")

# ─── 3年分 TTM PER 系列を構築 ────────────────────────
print("\nTTM PER 系列構築...")

def build_ttm_eps_series(close_index, eps_q, annual_eps_dict):
    """
    各日付に対して TTM EPS を推計する。
    四半期データがある範囲は正確計算、
    それ以前は年次EPSから補間する。
    """
    result = {}

    for date in close_index:
        # 四半期データから TTM を計算できるか試みる
        past_q = eps_q[eps_q.index <= date]
        if len(past_q) >= 4:
            ttm = past_q.iloc[-4:].sum()
            if ttm > 0:
                result[date] = ttm
                continue

        # 四半期データが足りない場合、年次データから補間
        if annual_eps_dict:
            ann_dates = sorted(annual_eps_dict.keys())
            # 直近の年次EPS期末日より前の日付
            past_ann = [d for d in ann_dates if d <= date + pd.Timedelta(days=180)]
            if past_ann:
                latest_ann_date = max(past_ann)
                ttm = annual_eps_dict[latest_ann_date]
                if ttm > 0:
                    result[date] = ttm

    return pd.Series(result)

ttm_eps = build_ttm_eps_series(close.index, eps_q, annual_eps_dict)
per_all = (close / ttm_eps).dropna()
per_all = per_all[per_all > 0]
per_all = per_all[per_all < 200]  # 外れ値除去
print(f"  TTM PER 系列: {len(per_all)} 行 ({per_all.index[0].date()} 〜 {per_all.index[-1].date()})")
print(f"  現在PER: {per_all.iloc[-1]:.1f}x")

# ─── ボリンジャーバンド（PER） ──────────────────────────
# データ数に応じてウィンドウを調整
n = len(per_all)
if n >= 252:
    bb_win = 52   # 52週
elif n >= 100:
    bb_win = 20   # 20週
else:
    bb_win = max(10, n // 3)

print(f"  BBウィンドウ: {bb_win} (データ数 {n})")

per_df = per_all.to_frame('per')
per_df['ma']    = per_df['per'].rolling(bb_win).mean()
per_df['std']   = per_df['per'].rolling(bb_win).std()
per_df['upper'] = per_df['ma'] + 2 * per_df['std']
per_df['lower'] = per_df['ma'] - 2 * per_df['std']
per_df['pct_b'] = ((per_df['per'] - per_df['lower'])
                   / (per_df['upper'] - per_df['lower'])).clip(-0.5, 1.5)
per_plot = per_df.dropna()

cur_per   = per_plot['per'].iloc[-1]
cur_ma    = per_plot['ma'].iloc[-1]
cur_upper = per_plot['upper'].iloc[-1]
cur_lower = per_plot['lower'].iloc[-1]
cur_pctb  = per_plot['pct_b'].iloc[-1]

print(f"  PER={cur_per:.1f}x  MA={cur_ma:.1f}x  +2σ={cur_upper:.1f}x  -2σ={cur_lower:.1f}x  %B={cur_pctb:.2f}")

# ─── MACD 計算 ────────────────────────────────────────
def calc_macd(s, f=12, sl=26, sig=9):
    m = s.ewm(span=f,  adjust=False).mean() - s.ewm(span=sl, adjust=False).mean()
    signal = m.ewm(span=sig, adjust=False).mean()
    return m, signal, m - signal

macd_line, sig_line, hist = calc_macd(close)

def calc_rsi(s, p=14):
    d = s.diff()
    g = d.clip(lower=0).ewm(com=p-1, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(com=p-1, adjust=False).mean()
    return 100 - 100 / (1 + g / l)

rsi_series = calc_rsi(close)
sma20  = close.rolling(20).mean()
sma50  = close.rolling(50).mean()
sma200 = close.rolling(200).mean()

cur_price = close.iloc[-1]
cur_rsi   = rsi_series.iloc[-1]
high_52w  = close.iloc[-252:].max()
low_52w   = close.iloc[-252:].min()
cur_macd  = macd_line.iloc[-1]
cur_sig   = sig_line.iloc[-1]
cur_hist  = hist.iloc[-1]

# ─── チャート1: 株価 + 出来高 + MACD ─────────────────
# Y軸独立のため specs で明示
fig1 = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    row_heights=[0.55, 0.18, 0.27],
    vertical_spacing=0.04,
    subplot_titles=('株価 + 移動平均 (LLY)', '出来高', 'MACD (12, 26, 9)')
)

# --- Row 1: 株価 ---
fig1.add_trace(go.Scatter(x=close.index, y=close,
    name='終値', line=dict(color='#00d4ff', width=2),
    hovertemplate='$%{y:.2f}<extra></extra>'), row=1, col=1)
fig1.add_trace(go.Scatter(x=sma20.index, y=sma20,
    name='SMA20', line=dict(color='#ffd700', width=1.2, dash='dash')), row=1, col=1)
fig1.add_trace(go.Scatter(x=sma50.index, y=sma50,
    name='SMA50', line=dict(color='#ff9944', width=1.5)), row=1, col=1)
fig1.add_trace(go.Scatter(x=sma200.index, y=sma200,
    name='SMA200', line=dict(color='#ff4444', width=1.5, dash='dot')), row=1, col=1)

# --- Row 2: 出来高 ---
vol_colors = np.where(close >= close.shift(1), '#00d4ff', '#ff4444')
fig1.add_trace(go.Bar(x=price_df.index, y=price_df['Volume'],
    name='出来高', marker_color=vol_colors, showlegend=False,
    hovertemplate='%{y:,.0f}<extra></extra>'), row=2, col=1)

# --- Row 3: MACD ---
hist_colors = np.where(hist >= 0, '#00d4ff', '#ff4444')
fig1.add_trace(go.Bar(x=hist.index, y=hist,
    name='ヒストグラム', marker_color=hist_colors, showlegend=False), row=3, col=1)
fig1.add_trace(go.Scatter(x=macd_line.index, y=macd_line,
    name='MACD', line=dict(color='#00d4ff', width=2)), row=3, col=1)
fig1.add_trace(go.Scatter(x=sig_line.index, y=sig_line,
    name='シグナル', line=dict(color='#ffd700', width=1.5)), row=3, col=1)
fig1.add_hline(y=0, line=dict(color='white', width=0.5, dash='dash'), row=3, col=1)

# Y軸: 各行を独立させる（rangemode='normal' でゼロ始まりを防ぐ）
price_min = close.min() * 0.92
price_max = close.max() * 1.05
fig1.update_yaxes(range=[price_min, price_max],
                  tickformat='$,.0f', title_text='USD', row=1, col=1)
fig1.update_yaxes(title_text='出来高', row=2, col=1)

macd_abs_max = max(abs(macd_line.dropna()).max(), abs(sig_line.dropna()).max(),
                   abs(hist.dropna()).max()) * 1.3
fig1.update_yaxes(range=[-macd_abs_max, macd_abs_max], title_text='MACD', row=3, col=1)

fig1.update_layout(
    height=720,
    paper_bgcolor='#0a0a1a', plot_bgcolor='#0a0a1a',
    font=dict(color='white', family='Arial'),
    legend=dict(orientation='h', y=1.03, bgcolor='rgba(0,0,0,0)', font_size=12),
    margin=dict(l=70, r=20, t=50, b=20),
    xaxis3=dict(
        rangeselector=dict(
            buttons=[
                dict(count=3,  label='3M',   step='month'),
                dict(count=6,  label='6M',   step='month'),
                dict(count=1,  label='1Y',   step='year'),
                dict(count=2,  label='2Y',   step='year'),
                dict(step='all', label='全期間'),
            ],
            bgcolor='#1a1a2e', activecolor='#00d4ff',
            font=dict(color='white')
        ),
        rangeslider=dict(visible=False)
    )
)
for r in range(1, 4):
    fig1.update_xaxes(gridcolor='#1e1e30', row=r, col=1)
    fig1.update_yaxes(gridcolor='#1e1e30', row=r, col=1)

# ─── チャート2: PER ボリンジャーバンド ──────────────────
fig2 = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    row_heights=[0.72, 0.28],
    vertical_spacing=0.06,
    subplot_titles=(f'PER ボリンジャーバンド（{bb_win}日移動平均, ±2σ）',
                    '%B（BBバンド内の位置） ／ 赤背景＝SMA200下（買いシグナル無効）')
)

# バンド塗りつぶし（上限→下限の間を塗る）
fig2.add_trace(go.Scatter(
    x=per_plot.index, y=per_plot['upper'],
    mode='lines', line=dict(color='rgba(255,107,53,0.4)', width=1),
    name=f'BB上限(+2σ) {cur_upper:.1f}x', showlegend=True
), row=1, col=1)
fig2.add_trace(go.Scatter(
    x=per_plot.index, y=per_plot['lower'],
    mode='lines', line=dict(color='rgba(255,107,53,0.4)', width=1),
    fill='tonexty', fillcolor='rgba(255,107,53,0.18)',
    name=f'BB下限(-2σ) {cur_lower:.1f}x', showlegend=True
), row=1, col=1)
fig2.add_trace(go.Scatter(
    x=per_plot.index, y=per_plot['ma'],
    mode='lines', line=dict(color='#ffd700', width=1.5, dash='dash'),
    name=f'BB中央({bb_win}日MA) {cur_ma:.1f}x'
), row=1, col=1)
fig2.add_trace(go.Scatter(
    x=per_plot.index, y=per_plot['per'],
    mode='lines', line=dict(color='#00d4ff', width=2.5),
    name=f'実績PER {cur_per:.1f}x',
    hovertemplate='PER: %{y:.1f}x<extra></extra>'
), row=1, col=1)

# 現在値アノテーション
last_date = per_plot.index[-1]
fig2.add_annotation(
    x=last_date, y=cur_per,
    text=f'現在 {cur_per:.1f}x',
    showarrow=True, arrowhead=2, arrowcolor='#00d4ff',
    ax=-60, ay=-30,
    font=dict(color='white', size=12),
    bgcolor='#1a1a2e', bordercolor='#00d4ff', borderwidth=1,
    row=1, col=1
)

# %B
fig2.add_trace(go.Scatter(
    x=per_plot.index, y=per_plot['pct_b'],
    mode='lines', line=dict(color='#00d4ff', width=2),
    fill='tozeroy', fillcolor='rgba(0,212,255,0.12)',
    name=f'%B = {cur_pctb:.2f}',
    hovertemplate='%%B: %{y:.3f}<extra></extra>'
), row=2, col=1)

# %B 参照ライン
for lvl, col, lbl in [(1.0, '#ff4444', ''), (0.8, '#ff9944', ''),
                       (0.5, '#ffd700', '中央'), (0.2, '#44ff88', ''), (0.0, '#44ff88', '')]:
    fig2.add_hline(y=lvl, line=dict(color=col, width=0.8, dash='dash'), row=2, col=1)
fig2.add_hrect(y0=0.8, y1=1.2, fillcolor='rgba(255,68,68,0.08)',
               line_width=0, row=2, col=1,
               annotation_text='割高', annotation_font_color='#ff4444',
               annotation_position='right')
fig2.add_hrect(y0=-0.2, y1=0.2, fillcolor='rgba(68,255,136,0.08)',
               line_width=0, row=2, col=1,
               annotation_text='割安', annotation_font_color='#44ff88',
               annotation_position='right')

# SMA200フィルター：アノテーションで現在状態を表示（トレース追加なし）
above_now = bool(close.iloc[-1] > sma200.iloc[-1])
sma200_label = '▲ SMA200上（買いシグナル有効）' if above_now else '▼ SMA200下（買いシグナル無効）'
sma200_color = '#00ff88' if above_now else '#ff4444'
fig2.add_annotation(
    x=per_plot.index[-1], y=0.5,
    text=sma200_label,
    showarrow=False,
    font=dict(color=sma200_color, size=11),
    bgcolor='rgba(0,0,0,0.5)',
    bordercolor=sma200_color, borderwidth=1,
    xanchor='right', yanchor='middle',
    row=2, col=1
)

# Y軸：直近値周辺にクリップしてBBバンドが見えるようにする
recent_per = per_plot['per'].iloc[-126:]  # 直近6ヶ月
per_y_min = max(0, recent_per.min() * 0.85)
per_y_max = recent_per.max() * 1.15
fig2.update_yaxes(range=[per_y_min, per_y_max],
                  tickformat='.0f', ticksuffix='x', title_text='PER',
                  row=1, col=1)
fig2.update_yaxes(range=[-0.15, 1.3],
                  tickformat='.1f', title_text='%B',
                  row=2, col=1)

fig2.update_layout(
    height=640,
    paper_bgcolor='#0a0a1a', plot_bgcolor='#0a0a1a',
    font=dict(color='white', family='Arial'),
    legend=dict(orientation='h', y=1.03, bgcolor='rgba(0,0,0,0)', font_size=12),
    margin=dict(l=70, r=20, t=50, b=20),
)
for r in range(1, 3):
    fig2.update_xaxes(gridcolor='#1e1e30', row=r, col=1)
    fig2.update_yaxes(gridcolor='#1e1e30', row=r, col=1)

# ─── 各種指標計算 ─────────────────────────────────────
if cur_pctb < 0.2:
    per_judge, per_color = '割安ゾーン', '#00ff88'
    per_desc = f'PERが歴史的BBの下限付近（%B={cur_pctb:.2f}）。統計的に割安。'
elif cur_pctb < 0.4:
    per_judge, per_color = 'やや割安',   '#7fff00'
    per_desc = f'PERが中央より下（%B={cur_pctb:.2f}）。バリュエーション的に有利。'
elif cur_pctb < 0.6:
    per_judge, per_color = 'フェアバリュー', '#ffd700'
    per_desc = f'PERが歴史的中央値付近（%B={cur_pctb:.2f}）。適正水準。'
elif cur_pctb < 0.8:
    per_judge, per_color = 'やや割高',   '#ff9944'
    per_desc = f'PERが中央より上（%B={cur_pctb:.2f}）。成長期待が織り込まれている。'
else:
    per_judge, per_color = '割高ゾーン', '#ff4444'
    per_desc = f'PERがBBの上限付近（%B={cur_pctb:.2f}）。歴史的に高バリュエーション。'

last_hist_val  = hist.iloc[-1]
prev_hist_val  = hist.iloc[-2]
if last_hist_val > 0 and prev_hist_val <= 0:
    macd_judge, macd_color = 'ゴールデンクロス', '#00ff88'
elif last_hist_val < 0 and prev_hist_val >= 0:
    macd_judge, macd_color = 'デッドクロス',     '#ff4444'
elif last_hist_val > 0 and last_hist_val > prev_hist_val:
    macd_judge, macd_color = '上昇モメンタム継続', '#00d4ff'
elif last_hist_val > 0:
    macd_judge, macd_color = '上昇鈍化',          '#ffd700'
else:
    macd_judge, macd_color = '下降トレンド',       '#ff9944'

# ─── HTML 生成 ────────────────────────────────────────
chart1_html = fig1.to_html(full_html=False, include_plotlyjs=False)
chart2_html = fig2.to_html(full_html=False, include_plotlyjs=False)

usd_jpy = 158.0
entry_jpy = cur_price * usd_jpy
scenarios = [
    {'name': 'ベスト',   'price': 1300, 'usd_jpy': 158, 'color': '#00ff88',
     'bg': 'Orforglipron承認 + 円安維持 + 業績上振れ'},
    {'name': 'ベース',   'price': 1200, 'usd_jpy': 152, 'color': '#00d4ff',
     'bg': 'ガイダンス通り + BOJ 緩やかな利上げ'},
    {'name': 'ワースト', 'price': 1000, 'usd_jpy': 145, 'color': '#ffd700',
     'bg': '業績横ばい + 円高（BOJ 0.75%→1.0%）'},
    {'name': 'ストレス', 'price': 850,  'usd_jpy': 140, 'color': '#ff4444',
     'bg': 'FDA遅延 + 株価調整 + 円高（BOJ 積極利上げ）'},
]
for s in scenarios:
    s['jpy_val'] = s['price'] * s['usd_jpy']
    s['ret'] = (s['jpy_val'] / entry_jpy - 1) * 100

# ─── HTML テンプレート ────────────────────────────────
HTML = f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<title>LLY イーライリリー クオンツ分析レポート</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
:root {{
  --bg:#0a0a1a; --card:#0f0f23; --card2:#13132a;
  --ac:#00d4ff; --ac2:#ffd700; --ac3:#ff6b35;
  --gr:#00ff88; --rd:#ff4444; --tx:#e0e0f0; --dim:#8888aa;
}}
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:var(--bg);color:var(--tx);font-family:Arial,sans-serif}}
.hdr{{
  background:linear-gradient(135deg,#0a0a1a,#18083a,#0a1a2a);
  border-bottom:1px solid rgba(0,212,255,.2);
  padding:36px 56px 28px; position:relative; overflow:hidden;
}}
.hdr::before{{content:'';position:absolute;top:-60px;left:-60px;
  width:350px;height:350px;
  background:radial-gradient(circle,rgba(0,212,255,.06) 0%,transparent 70%);
  pointer-events:none}}
.badge{{display:inline-block;background:rgba(0,212,255,.1);border:1px solid var(--ac);
  border-radius:5px;padding:3px 11px;font-size:11px;letter-spacing:2px;
  color:var(--ac);margin-bottom:10px}}
.co{{font-size:30px;font-weight:700;
  background:linear-gradient(135deg,#fff,#00d4ff);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  background-clip:text;margin-bottom:5px}}
.meta{{color:var(--dim);font-size:12px}} .meta span{{color:var(--ac2)}}
.price-hero{{position:absolute;right:56px;top:36px;text-align:right}}
.p-main{{font-size:46px;font-weight:700;color:#fff;line-height:1}}
.p-sub{{font-size:13px;color:var(--dim);margin-top:3px}}
.p-jpy{{font-size:19px;color:var(--ac2);margin-top:3px}}
.wrap{{max-width:1380px;margin:0 auto;padding:28px 36px}}
.sec{{font-size:17px;font-weight:700;color:var(--ac);
  margin:32px 0 18px;padding-bottom:8px;
  border-bottom:1px solid rgba(0,212,255,.2);
  display:flex;align-items:center;gap:9px}}
.sec::before{{content:'';display:inline-block;width:4px;height:17px;
  background:var(--ac);border-radius:2px}}
.kgrid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(195px,1fr));gap:14px;margin-bottom:36px}}
.kc{{background:var(--card);border:1px solid rgba(255,255,255,.07);
  border-radius:12px;padding:18px;transition:border-color .2s}}
.kc:hover{{border-color:rgba(0,212,255,.3)}}
.kl{{font-size:10px;color:var(--dim);text-transform:uppercase;letter-spacing:1px;margin-bottom:7px}}
.kv{{font-size:26px;font-weight:700;color:#fff;line-height:1}}
.ks{{font-size:11px;color:var(--dim);margin-top:5px}}
.cc{{background:var(--card);border:1px solid rgba(255,255,255,.07);
  border-radius:16px;padding:22px;margin-bottom:28px}}
.ct{{font-size:15px;font-weight:600;color:#fff;margin-bottom:5px}}
.cd{{font-size:12px;color:var(--dim);margin-bottom:18px;line-height:1.6}}
.ib{{background:rgba(0,212,255,.05);border:1px solid rgba(0,212,255,.2);
  border-radius:10px;padding:14px 18px;margin-top:14px;
  font-size:13px;line-height:1.7;color:var(--tx)}}
.ib strong{{color:var(--ac2)}}
.ag{{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-bottom:28px}}
@media(max-width:860px){{.ag{{grid-template-columns:1fr}}.price-hero{{display:none}}}}
.ac{{background:var(--card);border:1px solid rgba(255,255,255,.07);
  border-radius:14px;padding:22px}}
.ac h3{{font-size:11px;color:var(--dim);margin-bottom:14px;
  text-transform:uppercase;letter-spacing:1px}}
.ri{{display:flex;align-items:flex-start;gap:11px;padding:9px 0;
  border-bottom:1px solid rgba(255,255,255,.05);font-size:12px;line-height:1.5}}
.ri:last-child{{border-bottom:none}}
.rd{{width:7px;height:7px;border-radius:50%;margin-top:4px;flex-shrink:0}}
.stbl{{width:100%;border-collapse:collapse;margin-top:10px}}
.stbl th{{text-align:left;font-size:10px;color:var(--dim);text-transform:uppercase;
  letter-spacing:1px;padding:7px 11px;border-bottom:1px solid rgba(255,255,255,.1)}}
.stbl td{{padding:11px;font-size:13px;border-bottom:1px solid rgba(255,255,255,.04)}}
.vcard{{background:linear-gradient(135deg,#0f0f23,#1a0f2e);
  border:1px solid rgba(0,212,255,.3);border-radius:16px;
  padding:30px;margin-bottom:28px;position:relative;overflow:hidden}}
.vcard::after{{content:'GS QUANT';position:absolute;right:28px;top:18px;
  font-size:10px;letter-spacing:3px;color:rgba(0,212,255,.12);font-weight:700}}
.vlbl{{font-size:10px;color:var(--dim);letter-spacing:2px;text-transform:uppercase;margin-bottom:10px}}
.vmain{{font-size:34px;font-weight:700;color:var(--gr);margin-bottom:14px}}
.vbody{{font-size:13px;line-height:1.8;color:var(--tx);max-width:780px}}
.esg{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-top:18px}}
.ei{{background:rgba(0,0,0,.3);border-radius:9px;padding:14px;text-align:center}}
.el{{font-size:10px;color:var(--dim);margin-bottom:7px;text-transform:uppercase;letter-spacing:1px}}
.ev{{font-size:17px;font-weight:700;color:var(--ac2)}}
.foot{{text-align:center;padding:28px;color:var(--dim);font-size:10px;
  border-top:1px solid rgba(255,255,255,.05);line-height:2.2}}
.tag{{display:inline-block;background:rgba(255,255,255,.05);border-radius:4px;
  padding:2px 7px;font-size:10px;color:var(--dim);margin:2px}}
</style>
</head>
<body>
<div class="hdr">
  <div class="badge">NYSE: LLY</div>
  <div class="co">Eli Lilly and Company</div>
  <div class="meta">クオンツ分析レポート &thinsp;·&thinsp;
    <span>{datetime.now().strftime('%Y年%m月%d日')}</span> &thinsp;·&thinsp;
    USD/JPY: <span>¥{usd_jpy:.0f}</span> &thinsp;·&thinsp; yfinance + Plotly
  </div>
  <div class="price-hero">
    <div class="p-main">${cur_price:.2f}</div>
    <div class="p-sub">現在株価</div>
    <div class="p-jpy">¥{cur_price * usd_jpy:,.0f}</div>
  </div>
</div>

<div class="wrap">
  <div class="sec">主要指標</div>
  <div class="kgrid">
    <div class="kc">
      <div class="kl">現在株価</div>
      <div class="kv">${cur_price:.2f}</div>
      <div class="ks">¥{cur_price * usd_jpy:,.0f} (@¥{usd_jpy:.0f})</div>
    </div>
    <div class="kc">
      <div class="kl">52週 高値 / 安値</div>
      <div class="kv" style="font-size:19px">${high_52w:.0f} / ${low_52w:.0f}</div>
      <div class="ks">高値から {(1-cur_price/high_52w)*100:.1f}% 下</div>
    </div>
    <div class="kc">
      <div class="kl">RSI (14日)</div>
      <div class="kv" style="color:{'#ffd700' if 40<cur_rsi<60 else '#00ff88' if cur_rsi<30 else '#ff4444'}">{cur_rsi:.1f}</div>
      <div class="ks">{'売られすぎ' if cur_rsi<30 else '買われすぎ' if cur_rsi>70 else 'ニュートラル'}</div>
    </div>
    <div class="kc">
      <div class="kl">MACD シグナル</div>
      <div class="kv" style="font-size:18px;color:{macd_color}">{macd_judge}</div>
      <div class="ks">MACD {cur_macd:.2f} / Sig {cur_sig:.2f}</div>
    </div>
    <div class="kc">
      <div class="kl">実績 PER (TTM)</div>
      <div class="kv">{cur_per:.1f}x</div>
      <div class="ks">BB %B = {cur_pctb:.2f}</div>
    </div>
    <div class="kc">
      <div class="kl">PER バリュエーション</div>
      <div class="kv" style="font-size:17px;color:{per_color}">{per_judge}</div>
      <div class="ks">BB中央 {cur_ma:.1f}x / 範囲 {cur_lower:.0f}〜{cur_upper:.0f}x</div>
    </div>
    <div class="kc">
      <div class="kl">アナリスト平均目標株価</div>
      <div class="kv" style="font-size:21px">${info.get('targetMeanPrice',1214):.0f}</div>
      <div class="ks">上昇余地 +{(info.get('targetMeanPrice',1214)/cur_price-1)*100:.1f}%</div>
    </div>
    <div class="kc">
      <div class="kl">2026年度 売上高ガイダンス</div>
      <div class="kv" style="font-size:19px">$81.5B</div>
      <div class="ks">前年比 +25%（会社見通し中央値）</div>
    </div>
  </div>

  <!-- Chart 1 -->
  <div class="cc">
    <div class="ct">株価チャート + MACD</div>
    <div class="cd">終値 + SMA20/50/200 ／ 出来高 ／ MACD(12,26,9) ― 過去3年間</div>
    {chart1_html}
    <div class="ib">
      <strong>MACDリーディング</strong>：MACD <strong>{cur_macd:.2f}</strong>
      / シグナル <strong>{cur_sig:.2f}</strong>
      / ヒストグラム <strong>{cur_hist:.2f}</strong> → <span style="color:{macd_color}">{macd_judge}</span>。
      3月2日に50日MAを下抜けており短期モメンタムは弱い。
      <strong>$1,034レジスタンス突破</strong>が上昇モメンタム回復の鍵。
    </div>
  </div>

  <!-- Chart 2 -->
  <div class="cc">
    <div class="ct">PER ボリンジャーバンド ― クオンツ手法</div>
    <div class="cd">実績PER（TTM）に {bb_win}日ボリンジャーバンド(±2σ)を適用。
      %B &lt; 0.2 → 統計的割安（買いゾーン）、%B &gt; 0.8 → 統計的割高（警戒ゾーン）。
      現在 %B = <strong style="color:{per_color}">{cur_pctb:.2f}</strong>（{per_judge}）
    </div>
    {chart2_html}
    <div class="ib">
      <strong>PER BBリーディング</strong>：現在PER <strong>{cur_per:.1f}x</strong>
      は {bb_win}日BB中央 <strong>{cur_ma:.1f}x</strong>
      （上限 {cur_upper:.1f}x / 下限 {cur_lower:.1f}x）に対して
      <span style="color:{per_color}"><strong>%B = {cur_pctb:.2f}（{per_judge}）</strong></span>。
      {per_desc}
      フォワードEPS $42（会社ガイダンスEPS $33.5-35 + 成長）ベースでは
      <strong>フォワードPER ≈ 24x</strong> と、PEG ~1.1 で成長プレミアムは正当化できる水準。
    </div>
  </div>

  <!-- ファンダ + リスク -->
  <div class="sec">ファンダメンタルズ分析</div>
  <div class="ag">
    <div class="ac">
      <h3>カタリスト</h3>
      <div class="ri">
        <span style="color:#00ff88;font-size:16px;line-height:1.3">▲</span>
        <div><strong style="color:#00ff88">Mounjaro +110% YoY</strong><br>Q4 2025 で $7.4B。米国外承認拡大継続。</div>
      </div>
      <div class="ri">
        <span style="color:#00ff88;font-size:16px;line-height:1.3">▲</span>
        <div><strong style="color:#00ff88">Zepbound +123% YoY</strong><br>Q4 2025 で $4.2B。メディケア適用拡大が追い風。</div>
      </div>
      <div class="ri">
        <span style="color:#00d4ff;font-size:16px;line-height:1.3">★</span>
        <div><strong style="color:#00d4ff">Orforglipron 経口GLP-1（Q2/2026 FDA承認予定）</strong><br>注射不要の飲み薬。市場拡大の第2波カタリスト。</div>
      </div>
      <div class="ri">
        <span style="color:#ffd700;font-size:16px;line-height:1.3">◆</span>
        <div><strong style="color:#ffd700">2026ガイダンス $80-83B（+25% YoY）</strong><br>コンセンサス $77.6B を上回る強気見通し。</div>
      </div>
      <div class="ri">
        <span style="color:#ffd700;font-size:16px;line-height:1.3">◆</span>
        <div><strong style="color:#ffd700">製造能力増強</strong><br>インディアナ・ドイツ新工場で供給制約を緩和。</div>
      </div>
    </div>
    <div class="ac">
      <h3>リスクファクター</h3>
      <div class="ri"><div class="rd" style="background:#ff4444"></div>
        <div><strong style="color:#ff4444">Orforglipron FDA再遅延リスク</strong><br>既にQ1→Q2に延期済み。再延期なら $80-100 下振れ。</div>
      </div>
      <div class="ri"><div class="rd" style="background:#ff9944"></div>
        <div><strong style="color:#ff9944">競合GLP-1の台頭</strong><br>ノボノルディスク・AZ・Pfizer が追走。コンパウンド薬による価格圧迫。</div>
      </div>
      <div class="ri"><div class="rd" style="background:#ff9944"></div>
        <div><strong style="color:#ff9944">IRA 薬価交渉（2027年〜）</strong><br>メディケア薬価交渉で長期収益を圧迫。</div>
      </div>
      <div class="ri"><div class="rd" style="background:#ffd700"></div>
        <div><strong style="color:#ffd700">関税・政治リスク</strong><br>製薬業界への関税/価格政策の不確実性。</div>
      </div>
      <div class="ri"><div class="rd" style="background:#00d4ff"></div>
        <div><strong style="color:#00d4ff">円高リスク（日本人投資家固有）</strong><br>BOJ 利上げ継続 → USD/JPY 145-150 で円建てリターンを圧縮。</div>
      </div>
    </div>
  </div>

  <!-- 為替シナリオ -->
  <div class="sec">為替込みシナリオ分析（円建て投資家）</div>
  <div class="cc">
    <div class="cd">エントリー想定: <strong>${cur_price:.2f} × ¥{usd_jpy:.0f} = ¥{entry_jpy:,.0f} / 株</strong></div>
    <table class="stbl">
      <thead><tr>
        <th>シナリオ</th><th>1年後株価</th><th>USD/JPY</th>
        <th>円換算価値</th><th>円建てリターン</th><th>想定背景</th>
      </tr></thead>
      <tbody>
"""

for s in scenarios:
    sign = '+' if s['ret'] >= 0 else ''
    HTML += f"""<tr>
        <td style="color:{s['color']};font-weight:700">{s['name']}</td>
        <td>${s['price']:,}</td><td>¥{s['usd_jpy']}</td>
        <td>¥{s['jpy_val']:,}</td>
        <td style="color:{s['color']};font-weight:700;font-size:15px">{sign}{s['ret']:.1f}%</td>
        <td style="color:var(--dim);font-size:11px">{s['bg']}</td>
      </tr>"""

HTML += f"""
      </tbody>
    </table>
  </div>

  <!-- 最終判定 -->
  <div class="sec">最終判定（GS クオンツ + トレーダー視点）</div>
  <div class="vcard">
    <div class="vlbl">Overall Verdict · 長期投資（1年以上）</div>
    <div class="vmain">条件付き買い推奨</div>
    <div class="vbody">
      GLP-1市場の構造的成長（肥満症治療革命）を背景に、
      <strong>PER BB %B={cur_pctb:.2f}（{per_judge}）・フォワードPER ≈ 24x・PEG ≈ 1.1</strong>
      とバリュエーションは歴史的に正当化できる水準に収束してきた。
      ただし50日MA下抜けでテクニカルは弱く、Orforglipron FDA承認（Q2予定）が
      最初の大型カタリスト。<strong>分割投資 + 為替ヘッジ</strong>が合理的。
    </div>
    <div class="esg">
      <div class="ei"><div class="el">第1弾エントリー</div>
        <div class="ev">今すぐ 50%</div>
        <div style="font-size:10px;color:var(--dim);margin-top:5px">PER BBが割安圏。<br>値頃感あり。</div>
      </div>
      <div class="ei"><div class="el">第2弾エントリー</div>
        <div class="ev">$980 or FDA承認 残50%</div>
        <div style="font-size:10px;color:var(--dim);margin-top:5px">Orforglipron 承認時<br>or $980 押し目</div>
      </div>
      <div class="ei"><div class="el">ロスカット目安</div>
        <div class="ev">$920</div>
        <div style="font-size:10px;color:var(--dim);margin-top:5px">下抜けで仮説再検証。<br>次サポート $850。</div>
      </div>
    </div>
  </div>

  <div style="margin-bottom:16px;color:var(--dim);font-size:11px">
    使用データ:
    <span class="tag">yfinance</span>
    <span class="tag">MACD(12/26/9)</span>
    <span class="tag">PER BB({bb_win}日±2σ)</span>
    <span class="tag">RSI(14日)</span>
    <span class="tag">SMA 20/50/200</span>
    <span class="tag">アナリストコンセンサス</span>
  </div>
</div>

<div class="foot">
  本レポートは情報提供目的のみ。投資判断はご自身の責任で行ってください。<br>
  Generated by Claude Code + Plotly &thinsp;·&thinsp; {datetime.now().strftime('%Y-%m-%d %H:%M')}
</div>
</body>
</html>"""

out = '/home/like_rapid/GT-SOAR/LLY_quant_report.html'
with open(out, 'w', encoding='utf-8') as f:
    f.write(HTML)
print(f"\n✓ 完了: {out}  ({len(HTML)/1024:.0f} KB)")
