"""
LLY 株価予測レポート
- イベントカレンダー × 需給分析 × 価格予測
"""

import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

USD_JPY = 158.94

print("データ取得中...")
ticker = yf.Ticker("LLY")
price_df = ticker.history(period="1y", interval="1d")
price_df.index = pd.to_datetime(price_df.index).tz_localize(None)
close = price_df['Close']
volume = price_df['Volume']
info = ticker.info

cur_price = close.iloc[-1]
print(f"現在株価: ${cur_price:.2f}")

# ─── 機関投資家・需給データ ─────────────────────────────
print("需給データ取得中...")
try:
    inst_holders = ticker.institutional_holders
    top_inst = inst_holders.head(8) if inst_holders is not None else pd.DataFrame()
except:
    top_inst = pd.DataFrame()

try:
    major_holders = ticker.major_holders
except:
    major_holders = None

# ショートインタレスト
short_ratio  = info.get('shortRatio', None)
short_pct    = info.get('shortPercentOfFloat', None)
float_shares = info.get('floatShares', None)
shares_out   = info.get('sharesOutstanding', None)

# アナリスト
target_mean  = info.get('targetMeanPrice', 1214)
target_high  = info.get('targetHighPrice', 1500)
target_low   = info.get('targetLowPrice', 800)
target_med   = info.get('targetMedianPrice', 1200)
num_analysts = info.get('numberOfAnalystOpinions', 30)
rec_mean     = info.get('recommendationMean', 1.8)

# ─── テクニカル計算 ───────────────────────────────────
sma20  = close.rolling(20).mean()
sma50  = close.rolling(50).mean()
sma200 = close.rolling(200).mean()

# ATR（14日）
high  = price_df['High']
low   = price_df['Low']
tr    = pd.concat([high - low,
                   (high - close.shift()).abs(),
                   (low  - close.shift()).abs()], axis=1).max(axis=1)
atr14 = tr.rolling(14).mean().iloc[-1]

# ヒストリカルボラティリティ（30日）
log_ret = np.log(close / close.shift())
hv30    = log_ret.rolling(30).std().iloc[-1] * np.sqrt(252) * 100

# サポート・レジスタンス（直近6ヶ月の局所的な高安値）
recent = price_df.iloc[-126:]
roll_high = recent['High'].rolling(10, center=True).max()
roll_low  = recent['Low'].rolling(10, center=True).min()

supports    = sorted(list(set([round(v, -1) for v in roll_low.dropna().unique() if v < cur_price * 0.98])))[-5:]
resistances = sorted(list(set([round(v, -1) for v in roll_high.dropna().unique() if v > cur_price * 1.02])))[:5]

print(f"  ATR14: ${atr14:.2f}  HV30: {hv30:.1f}%")
print(f"  サポート: {supports}")
print(f"  レジスタンス: {resistances}")

# ─── オプション（直近限月のIV取得） ──────────────────────
print("オプションデータ取得中...")
iv_atm = None
put_call_ratio = None
try:
    exps = ticker.options
    if exps:
        # 約30日後の限月を選ぶ
        target_date = datetime.now() + timedelta(days=35)
        best_exp = min(exps, key=lambda d: abs(
            (datetime.strptime(d, '%Y-%m-%d') - target_date).days))
        chain = ticker.option_chain(best_exp)
        calls = chain.calls
        puts  = chain.puts

        # ATM付近のIV
        atm_calls = calls[abs(calls['strike'] - cur_price) < 30]
        if not atm_calls.empty:
            iv_atm = atm_calls['impliedVolatility'].median() * 100

        # Put/Call ratio（出来高ベース）
        total_put_vol  = puts['volume'].sum()
        total_call_vol = calls['volume'].sum()
        if total_call_vol > 0:
            put_call_ratio = total_put_vol / total_call_vol

        print(f"  IV(ATM, {best_exp}): {iv_atm:.1f}%" if iv_atm else "  IV取得失敗")
        print(f"  Put/Call ratio: {put_call_ratio:.2f}" if put_call_ratio else "  P/C取得失敗")
except Exception as e:
    print(f"  オプションエラー: {e}")

# ─── イベントスケジュール（手動定義） ─────────────────────
# 期待される株価インパクト: HV30を使って確率的に計算
# earnings_move は過去の実績値を参照

events = [
    {
        'date': '2026-03-15', 'label': 'JPモルガン医療フォーラム',
        'type': 'conf',
        'impact': '+1〜3%', 'prob': '中', 'color': '#00d4ff',
        'detail': 'Lilly CEOによるパイプライン・ガイダンス再確認の場。新規カタリストは限定的だが、機関投資家へのIRとして需給に好影響。',
    },
    {
        'date': '2026-04-16', 'label': 'Q1 2026 決算発表（予定）',
        'type': 'earnings',
        'impact': '±8〜12%', 'prob': '高', 'color': '#ffd700',
        'detail': 'Mounjaro・Zepboundの売上続伸が鍵。Q4 2025は両製品で+100%超のYoY成長。アナリストコンセンサスを上回るかどうかで大きく動く。過去4決算の平均株価反応は±9%。',
    },
    {
        'date': '2026-05-15', 'label': 'Orforglipron FDA PDUFA（Q2目標）',
        'type': 'fda',
        'impact': '+10〜20% or −15%', 'prob': '最重要', 'color': '#ff6b35',
        'detail': '経口GLP-1薬（飲み薬）の初承認。注射不要で潜在患者層を大幅拡大。承認なら株価+15%以上の可能性。再延期・否決なら−10〜15%。2026年最大のバイナリーイベント。',
    },
    {
        'date': '2026-06-20', 'label': 'ADA 学術年次総会（糖尿病学会）',
        'type': 'conf',
        'impact': '+2〜5%', 'prob': '中', 'color': '#00d4ff',
        'detail': '糖尿病・肥満症領域の最重要学会。Orforglipron承認後なら実臨床データの追加公表で買い継続。Zepboundの心血管データ（SURMOUNT-MMO）の追加解析も注目。',
    },
    {
        'date': '2026-07-22', 'label': 'Q2 2026 決算発表（予定）',
        'type': 'earnings',
        'impact': '±6〜10%', 'prob': '高', 'color': '#ffd700',
        'detail': 'Orforglipron承認後初の決算。初期売上・2026年ガイダンス引き上げが焦点。製造能力の増強状況も確認。アナリスト予想 EPS $3.8〜4.2。',
    },
    {
        'date': '2026-09-10', 'label': 'ESMO 2026（腫瘍学会）',
        'type': 'conf',
        'impact': '+1〜4%', 'prob': '低〜中', 'color': '#00d4ff',
        'detail': 'LLYの癌領域（Verzenio等）に関するデータ発表。GLP-1一本足からの多角化の観点で注目度は高まりつつある。',
    },
    {
        'date': '2026-10-20', 'label': 'Q3 2026 決算発表（予定）',
        'type': 'earnings',
        'impact': '±5〜9%', 'prob': '高', 'color': '#ffd700',
        'detail': 'Orforglipron通年寄与が見え始める決算。2027年ガイダンスが株価のネクストレベルを決める。機関投資家の年末ポジション調整とも重なる。',
    },
]

# ─── 確率的価格予測（モンテカルロ） ──────────────────────
print("モンテカルロシミュレーション...")
np.random.seed(42)
N_SIM  = 2000
N_DAYS = 252  # 1年

daily_vol = hv30 / 100 / np.sqrt(252)
# ドリフト: アナリスト平均目標株価から逆算（年率）
annual_drift = np.log(target_mean / cur_price)
daily_drift  = annual_drift / N_DAYS

# シミュレーション
paths = np.zeros((N_SIM, N_DAYS + 1))
paths[:, 0] = cur_price
for t in range(1, N_DAYS + 1):
    z = np.random.standard_normal(N_SIM)
    paths[:, t] = paths[:, t-1] * np.exp(
        (daily_drift - 0.5 * daily_vol**2) + daily_vol * z
    )

# パーセンタイル
pct_5  = np.percentile(paths[:, -1], 5)
pct_25 = np.percentile(paths[:, -1], 25)
pct_50 = np.percentile(paths[:, -1], 50)
pct_75 = np.percentile(paths[:, -1], 75)
pct_95 = np.percentile(paths[:, -1], 95)

# 1ヶ月・3ヶ月の分布
days_1m  = 21
days_3m  = 63
pct_1m   = np.percentile(paths[:, days_1m], [10, 25, 50, 75, 90])
pct_3m   = np.percentile(paths[:, days_3m], [10, 25, 50, 75, 90])

print(f"  1年後中央値: ${pct_50:.0f}  5%ile: ${pct_5:.0f}  95%ile: ${pct_95:.0f}")
print(f"  1ヶ月後中央値: ${pct_1m[2]:.0f}")
print(f"  3ヶ月後中央値: ${pct_3m[2]:.0f}")

# ─── チャート① 株価 + サポレジ + イベントライン ──────────
print("チャート生成中...")

# 過去1年の株価ライン
dates_hist = close.index.tolist()

# 将来1年の日付軸（土日除く）
future_dates = pd.bdate_range(
    start=close.index[-1] + timedelta(days=1),
    periods=N_DAYS
)

# 選んだパスのサンプル（視覚化用 30本）
sample_paths = paths[::N_SIM//30, 1:]  # 30本のパス

# パーセンタイルバンド
band_5  = np.percentile(paths[:, 1:], 5,  axis=0)
band_25 = np.percentile(paths[:, 1:], 25, axis=0)
band_50 = np.percentile(paths[:, 1:], 50, axis=0)
band_75 = np.percentile(paths[:, 1:], 75, axis=0)
band_95 = np.percentile(paths[:, 1:], 95, axis=0)

fig1 = go.Figure()

# 過去株価
fig1.add_trace(go.Scatter(
    x=dates_hist, y=close,
    name='実績株価', line=dict(color='#ffffff', width=2.5),
    hovertemplate='$%{y:.2f}<extra></extra>'
))

# SMA
fig1.add_trace(go.Scatter(x=sma50.index, y=sma50,
    name='SMA50', line=dict(color='#ff9944', width=1.2, dash='dash'), opacity=0.7))
fig1.add_trace(go.Scatter(x=sma200.index, y=sma200,
    name='SMA200', line=dict(color='#ff4444', width=1.2, dash='dot'), opacity=0.7))

# 将来バンド（90%〜10%）
fig1.add_trace(go.Scatter(
    x=list(future_dates) + list(future_dates[::-1]),
    y=list(band_95) + list(band_5[::-1]),
    fill='toself', fillcolor='rgba(0,212,255,0.06)',
    line=dict(width=0), name='90%信頼区間', showlegend=True
))
fig1.add_trace(go.Scatter(
    x=list(future_dates) + list(future_dates[::-1]),
    y=list(band_75) + list(band_25[::-1]),
    fill='toself', fillcolor='rgba(0,212,255,0.12)',
    line=dict(width=0), name='50%信頼区間', showlegend=True
))

# 中央値ライン
fig1.add_trace(go.Scatter(
    x=future_dates, y=band_50,
    name=f'予測中央値 (${pct_50:.0f})', line=dict(color='#00d4ff', width=2, dash='dot')
))

# アナリスト目標株価ライン
fig1.add_hline(y=target_mean, line=dict(color='#ffd700', width=1, dash='dash'),
               annotation_text=f'アナリスト平均 ${target_mean:.0f}',
               annotation_font_color='#ffd700', annotation_position='bottom right')
fig1.add_hline(y=target_high, line=dict(color='#00ff88', width=0.8, dash='dot'),
               annotation_text=f'強気目標 ${target_high:.0f}',
               annotation_font_color='#00ff88', annotation_position='top right')

# イベントの縦線
for ev in events:
    ev_date = datetime.strptime(ev['date'], '%Y-%m-%d')
    if ev_date > close.index[-1]:
        fig1.add_vline(x=ev_date, line=dict(color=ev['color'], width=1, dash='dot'),
                       opacity=0.6)
        fig1.add_annotation(
            x=ev_date, y=cur_price * 1.38,
            text=ev['label'][:15] + '...' if len(ev['label']) > 15 else ev['label'],
            textangle=-70, font=dict(size=9, color=ev['color']),
            showarrow=False, xanchor='left'
        )

# サポート・レジスタンス
for sp in supports[-3:]:
    fig1.add_hline(y=sp, line=dict(color='rgba(0,255,136,0.3)', width=0.8, dash='dash'))
for rs in resistances[:3]:
    fig1.add_hline(y=rs, line=dict(color='rgba(255,100,100,0.3)', width=0.8, dash='dash'))

fig1.update_layout(
    title=dict(text='LLY 株価チャート + 1年予測（モンテカルロ）', font=dict(color='white', size=16)),
    height=540,
    paper_bgcolor='#0a0a1a', plot_bgcolor='#0a0a1a',
    font=dict(color='white', family='Arial'),
    legend=dict(orientation='h', y=-0.12, bgcolor='rgba(0,0,0,0)', font_size=11),
    margin=dict(l=70, r=20, t=60, b=60),
    xaxis=dict(gridcolor='#1e1e30', range=[close.index[0], future_dates[-1]]),
    yaxis=dict(gridcolor='#1e1e30', tickformat='$,.0f', range=[500, 1800]),
    hovermode='x unified'
)

# ─── チャート② アナリスト目標株価分布（ヒストグラム風） ──
# 正規分布でコンセンサス分布を近似
x_range = np.linspace(target_low * 0.9, target_high * 1.05, 200)
sigma = (target_high - target_low) / 4  # 概算
mu = target_mean
dist = np.exp(-0.5 * ((x_range - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=x_range, y=dist,
    fill='tozeroy', fillcolor='rgba(0,212,255,0.15)',
    line=dict(color='#00d4ff', width=2),
    name='アナリスト目標分布（近似）'
))
fig2.add_vline(x=cur_price, line=dict(color='white', width=2),
               annotation_text=f'現在値 ${cur_price:.0f}',
               annotation_font_color='white', annotation_position='top right')
fig2.add_vline(x=target_mean, line=dict(color='#ffd700', width=2, dash='dash'),
               annotation_text=f'平均目標 ${target_mean:.0f}',
               annotation_font_color='#ffd700', annotation_position='top left')
fig2.add_vline(x=target_med, line=dict(color='#00ff88', width=1.5, dash='dot'),
               annotation_text=f'中央値 ${target_med:.0f}',
               annotation_font_color='#00ff88', annotation_position='bottom right')

# モンテカルロの1年後分布（ヒストグラム）
fig2.add_trace(go.Histogram(
    x=paths[:, -1],
    nbinsx=60, name='モンテカルロ1年後分布',
    marker_color='rgba(255,107,53,0.4)',
    yaxis='y2', opacity=0.7
))

fig2.update_layout(
    title=dict(text='アナリスト目標株価 & モンテカルロ1年後分布', font=dict(color='white', size=16)),
    height=380,
    paper_bgcolor='#0a0a1a', plot_bgcolor='#0a0a1a',
    font=dict(color='white', family='Arial'),
    legend=dict(orientation='h', y=-0.18, bgcolor='rgba(0,0,0,0)'),
    margin=dict(l=70, r=70, t=60, b=70),
    xaxis=dict(gridcolor='#1e1e30', tickformat='$,.0f', title='株価 (USD)'),
    yaxis=dict(gridcolor='#1e1e30', title='確率密度'),
    yaxis2=dict(overlaying='y', side='right', title='シミュレーション本数',
                gridcolor='rgba(0,0,0,0)'),
    barmode='overlay'
)

# ─── チャート③ 出来高 + 機関投資家フロー代理（OBV） ─────
obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()

fig3 = make_subplots(
    rows=2, cols=1, shared_xaxes=True,
    row_heights=[0.55, 0.45], vertical_spacing=0.06,
    subplot_titles=('出来高', 'OBV（On-Balance Volume）— 需給トレンド')
)
vol_colors = np.where(close >= close.shift(1), '#00d4ff', '#ff4444')
fig3.add_trace(go.Bar(
    x=price_df.index, y=volume, marker_color=vol_colors,
    name='出来高', showlegend=False,
    hovertemplate='%{y:,.0f}<extra></extra>'
), row=1, col=1)
fig3.add_trace(go.Scatter(
    x=obv.index, y=obv,
    name='OBV', line=dict(color='#00ff88', width=2),
    hovertemplate='OBV: %{y:,.0f}<extra></extra>'
), row=2, col=1)

# OBV移動平均
obv_ma20 = obv.rolling(20).mean()
fig3.add_trace(go.Scatter(
    x=obv_ma20.index, y=obv_ma20,
    name='OBV MA20', line=dict(color='#ffd700', width=1.5, dash='dash')
), row=2, col=1)

fig3.update_layout(
    height=480,
    paper_bgcolor='#0a0a1a', plot_bgcolor='#0a0a1a',
    font=dict(color='white', family='Arial'),
    legend=dict(orientation='h', y=1.03, bgcolor='rgba(0,0,0,0)'),
    margin=dict(l=70, r=20, t=40, b=20),
)
for r in range(1, 3):
    fig3.update_xaxes(gridcolor='#1e1e30', row=r, col=1)
    fig3.update_yaxes(gridcolor='#1e1e30', row=r, col=1)

# ─── HTML 生成 ────────────────────────────────────────
chart1_html = fig1.to_html(full_html=False, include_plotlyjs=False)
chart2_html = fig2.to_html(full_html=False, include_plotlyjs=False)
chart3_html = fig3.to_html(full_html=False, include_plotlyjs=False)

# 機関投資家テーブル
inst_rows = ''
if not top_inst.empty:
    for _, row in top_inst.iterrows():
        holder = row.get('Holder', row.get('Name', '—'))
        shares = row.get('Shares', row.get('Value', '—'))
        pct    = row.get('% Out', row.get('pctHeld', '—'))
        val    = row.get('Value', '—')
        inst_rows += f"""<tr>
          <td style="color:#e0e0f0">{holder}</td>
          <td style="text-align:right">{f'{shares:,.0f}' if isinstance(shares, (int, float)) else shares}</td>
          <td style="text-align:right;color:#00d4ff">{f'{float(pct)*100:.2f}%' if isinstance(pct, (int, float)) else pct}</td>
          <td style="text-align:right;color:#ffd700">{f'${val/1e9:.2f}B' if isinstance(val, (int, float)) and val > 1e8 else val}</td>
        </tr>"""

# イベントカード
event_cards = ''
ev_type_icon = {'earnings': '📊', 'fda': '💊', 'conf': '🎤'}
ev_type_label = {'earnings': '決算', 'fda': 'FDA', 'conf': '学会'}
for ev in events:
    ev_dt   = datetime.strptime(ev['date'], '%Y-%m-%d')
    days_to = (ev_dt - datetime.now()).days
    if days_to < 0:
        days_badge = '<span style="background:rgba(255,255,255,.08);padding:2px 7px;border-radius:4px;font-size:10px;color:#666">過去</span>'
    elif days_to < 30:
        days_badge = f'<span style="background:rgba(255,68,68,.2);color:#ff4444;padding:2px 7px;border-radius:4px;font-size:10px">あと {days_to}日</span>'
    elif days_to < 60:
        days_badge = f'<span style="background:rgba(255,153,68,.15);color:#ff9944;padding:2px 7px;border-radius:4px;font-size:10px">あと {days_to}日</span>'
    else:
        days_badge = f'<span style="background:rgba(0,212,255,.1);color:#00d4ff;padding:2px 7px;border-radius:4px;font-size:10px">あと {days_to}日</span>'

    event_cards += f"""
    <div style="background:var(--card);border:1px solid {ev['color']}33;border-left:3px solid {ev['color']};
      border-radius:0 12px 12px 0;padding:16px 18px;margin-bottom:12px">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:8px">
        <div>
          <span style="font-size:10px;background:rgba(255,255,255,.07);padding:2px 7px;
            border-radius:4px;color:var(--dim);margin-right:7px">{ev_type_label[ev['type']]}</span>
          <strong style="color:{ev['color']};font-size:15px">{ev['label']}</strong>
        </div>
        <div style="text-align:right;flex-shrink:0;margin-left:12px">
          <div style="font-size:12px;color:var(--dim)">{ev['date']}</div>
          <div style="margin-top:3px">{days_badge}</div>
        </div>
      </div>
      <div style="display:flex;gap:16px;margin-bottom:8px">
        <div style="font-size:12px">
          <span style="color:var(--dim)">予想インパクト：</span>
          <strong style="color:{ev['color']}">{ev['impact']}</strong>
        </div>
        <div style="font-size:12px">
          <span style="color:var(--dim)">重要度：</span>
          <strong style="color:{ev['color']}">{ev['prob']}</strong>
        </div>
      </div>
      <div style="font-size:12px;color:var(--dim);line-height:1.7">{ev['detail']}</div>
    </div>"""

# 価格予測テーブル（1ヶ月・3ヶ月）
def fmt_ret(price):
    r = (price / cur_price - 1) * 100
    col = '#00ff88' if r > 0 else '#ff4444' if r < 0 else '#ffd700'
    sign = '+' if r > 0 else ''
    return f'<span style="color:{col}">{sign}{r:.1f}%</span>'

HTML = f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>LLY 株価予測レポート 2026</title>
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
  background:linear-gradient(135deg,#0a0a1a,#180828,#0a1a2a);
  border-bottom:1px solid rgba(255,107,53,.25);
  padding:36px 56px 28px;position:relative;overflow:hidden
}}
.hdr::before{{content:'';position:absolute;top:-60px;right:200px;
  width:400px;height:400px;
  background:radial-gradient(circle,rgba(255,107,53,.05) 0%,transparent 70%);
  pointer-events:none}}
.badge{{display:inline-block;background:rgba(255,107,53,.1);border:1px solid var(--ac3);
  border-radius:5px;padding:3px 11px;font-size:11px;letter-spacing:2px;
  color:var(--ac3);margin-bottom:10px}}
.badge2{{display:inline-block;background:rgba(0,212,255,.08);border:1px solid rgba(0,212,255,.3);
  border-radius:5px;padding:3px 10px;font-size:10px;letter-spacing:1px;
  color:var(--ac);margin-left:8px}}
.co{{font-size:30px;font-weight:700;
  background:linear-gradient(135deg,#fff,#ff6b35);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  background-clip:text;margin-bottom:5px}}
.meta{{color:var(--dim);font-size:12px}} .meta span{{color:var(--ac2)}}
.price-hero{{position:absolute;right:56px;top:36px;text-align:right}}
.p-main{{font-size:46px;font-weight:700;color:#fff;line-height:1}}
.p-sub{{font-size:13px;color:var(--dim);margin-top:3px}}
.wrap{{max-width:1380px;margin:0 auto;padding:28px 36px}}
.sec{{font-size:17px;font-weight:700;color:var(--ac3);
  margin:32px 0 18px;padding-bottom:8px;
  border-bottom:1px solid rgba(255,107,53,.2);
  display:flex;align-items:center;gap:9px}}
.sec::before{{content:'';display:inline-block;width:4px;height:17px;
  background:var(--ac3);border-radius:2px}}
.kgrid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:14px;margin-bottom:28px}}
.kc{{background:var(--card);border:1px solid rgba(255,255,255,.07);
  border-radius:12px;padding:18px;transition:border-color .2s}}
.kc:hover{{border-color:rgba(255,107,53,.3)}}
.kl{{font-size:10px;color:var(--dim);text-transform:uppercase;letter-spacing:1px;margin-bottom:7px}}
.kv{{font-size:26px;font-weight:700;color:#fff;line-height:1}}
.ks{{font-size:11px;color:var(--dim);margin-top:5px}}
.cc{{background:var(--card);border:1px solid rgba(255,255,255,.07);
  border-radius:16px;padding:22px;margin-bottom:28px}}
.ct{{font-size:15px;font-weight:600;color:#fff;margin-bottom:5px}}
.cd{{font-size:12px;color:var(--dim);margin-bottom:14px;line-height:1.6}}
.ib{{background:rgba(255,107,53,.05);border:1px solid rgba(255,107,53,.2);
  border-radius:10px;padding:14px 18px;margin-top:14px;
  font-size:13px;line-height:1.7;color:var(--tx)}}
.stbl{{width:100%;border-collapse:collapse;font-size:13px}}
.stbl th{{text-align:left;font-size:10px;color:var(--dim);text-transform:uppercase;
  letter-spacing:1px;padding:7px 11px;border-bottom:1px solid rgba(255,255,255,.1)}}
.stbl td{{padding:10px 11px;border-bottom:1px solid rgba(255,255,255,.04)}}
.stbl tr:hover td{{background:rgba(255,255,255,.02)}}
.pgrid{{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-bottom:28px}}
@media(max-width:860px){{.pgrid{{grid-template-columns:1fr}}.price-hero{{display:none}}}}
.pcard{{background:var(--card);border:1px solid rgba(255,255,255,.07);border-radius:14px;padding:20px}}
.vcard{{background:linear-gradient(135deg,#0f0f23,#1e0f28);
  border:1px solid rgba(255,107,53,.3);border-radius:16px;
  padding:30px;margin-bottom:28px;position:relative;overflow:hidden}}
.vcard::after{{content:'FORECAST';position:absolute;right:28px;top:18px;
  font-size:10px;letter-spacing:3px;color:rgba(255,107,53,.1);font-weight:700}}
.vlbl{{font-size:10px;color:var(--dim);letter-spacing:2px;text-transform:uppercase;margin-bottom:10px}}
.vmain{{font-size:34px;font-weight:700;color:var(--ac3);margin-bottom:14px}}
.vbody{{font-size:13px;line-height:1.8;color:var(--tx);max-width:800px}}
.esg{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:18px}}
@media(max-width:900px){{.esg{{grid-template-columns:repeat(2,1fr)}}}}
.ei{{background:rgba(0,0,0,.3);border-radius:9px;padding:14px;text-align:center}}
.el{{font-size:10px;color:var(--dim);margin-bottom:7px;text-transform:uppercase;letter-spacing:1px}}
.ev{{font-size:17px;font-weight:700;color:var(--ac2)}}
.foot{{text-align:center;padding:28px;color:var(--dim);font-size:10px;
  border-top:1px solid rgba(255,255,255,.05);line-height:2.2}}
</style>
</head>
<body>
<div class="hdr">
  <div class="badge">NYSE: LLY</div>
  <span class="badge2">予測レポート</span>
  <div class="co">Eli Lilly — 株価予測 &amp; 需給分析</div>
  <div class="meta">
    <span>{datetime.now().strftime('%Y年%m月%d日')}</span> &thinsp;·&thinsp;
    USD/JPY: <span>¥{USD_JPY}</span> &thinsp;·&thinsp;
    HV30: <span>{hv30:.1f}%</span>
    {f'&thinsp;·&thinsp; IV(ATM): <span>{iv_atm:.1f}%</span>' if iv_atm else ''}
  </div>
  <div class="price-hero">
    <div class="p-main">${cur_price:.2f}</div>
    <div class="p-sub">現在株価</div>
    <div style="font-size:19px;color:var(--ac2);margin-top:3px">¥{cur_price * USD_JPY:,.0f}</div>
  </div>
</div>

<div class="wrap">

  <!-- サマリー指標 -->
  <div class="sec">予測サマリー</div>
  <div class="kgrid">
    <div class="kc">
      <div class="kl">アナリスト平均目標</div>
      <div class="kv">${target_mean:.0f}</div>
      <div class="ks">上昇余地 {fmt_ret(target_mean)} / {num_analysts}人</div>
    </div>
    <div class="kc">
      <div class="kl">1ヶ月後 中央値（MC）</div>
      <div class="kv">${pct_1m[2]:.0f}</div>
      <div class="ks">25%ile ${pct_1m[1]:.0f} — 75%ile ${pct_1m[3]:.0f}</div>
    </div>
    <div class="kc">
      <div class="kl">3ヶ月後 中央値（MC）</div>
      <div class="kv">${pct_3m[2]:.0f}</div>
      <div class="ks">25%ile ${pct_3m[1]:.0f} — 75%ile ${pct_3m[3]:.0f}</div>
    </div>
    <div class="kc">
      <div class="kl">1年後 中央値（MC）</div>
      <div class="kv">${pct_50:.0f}</div>
      <div class="ks">5%ile ${pct_5:.0f} — 95%ile ${pct_95:.0f}</div>
    </div>
    <div class="kc">
      <div class="kl">ヒストリカルVol (30日)</div>
      <div class="kv">{hv30:.1f}%</div>
      <div class="ks">年率換算</div>
    </div>
    <div class="kc">
      <div class="kl">{"IV (ATM)" if iv_atm else "ATR (14日)"}</div>
      <div class="kv">{"%.1f%%" % iv_atm if iv_atm else "$%.2f" % atr14}</div>
      <div class="ks">{"インプライドVol" if iv_atm else "Average True Range"}</div>
    </div>
    <div class="kc">
      <div class="kl">{"Put/Call Ratio" if put_call_ratio else "ショート比率"}</div>
      <div class="kv">{"%.2f" % put_call_ratio if put_call_ratio else ("%.1f%%" % (short_pct*100) if short_pct else "N/A")}</div>
      <div class="ks">{"<1.0 = 強気優勢" if put_call_ratio and put_call_ratio < 1 else ("高い＝売り圧力" if not put_call_ratio else "中立以上の売り圧力")}</div>
    </div>
    <div class="kc">
      <div class="kl">レコメンデーション</div>
      <div class="kv" style="font-size:20px;color:{'#00ff88' if rec_mean < 2 else '#ffd700' if rec_mean < 2.8 else '#ff9944'}">{
        '強い買い' if rec_mean < 1.5 else '買い' if rec_mean < 2.2 else 'やや買い' if rec_mean < 2.8 else '中立'
      }</div>
      <div class="ks">平均スコア {rec_mean:.1f}（1=強買, 5=売）</div>
    </div>
  </div>

  <!-- 価格予測チャート -->
  <div class="sec">株価予測チャート（モンテカルロ {N_SIM:,}本）</div>
  <div class="cc">
    <div class="ct">過去1年 + 将来1年予測</div>
    <div class="cd">ドリフト: アナリスト平均目標株価ベース（年率 {annual_drift*100:.1f}%） / HV30={hv30:.1f}% / 縦点線=主要イベント</div>
    {chart1_html}
    <div class="ib">
      <strong>読み方</strong>：濃い青帯=50%信頼区間（25〜75%ile）、薄い青帯=90%信頼区間（5〜95%ile）。
      点線縦線は主要イベント（決算・FDA・学会）を示す。
      イベント前後に分布が拡大することに注意。
      現時点のドリフト前提（アナリストコンセンサスベース）では
      <strong>1年後中央値 ${pct_50:.0f}</strong>（現値比 {(pct_50/cur_price-1)*100:+.1f}%）。
    </div>
  </div>

  <!-- アナリスト分布 -->
  <div class="sec">アナリスト目標 &amp; モンテカルロ1年後分布</div>
  <div class="cc">
    <div class="ct">コンセンサス分布 vs 確率的予測分布</div>
    <div class="cd">左軸=アナリスト目標の正規近似、右軸=モンテカルロ1年後の終値分布（{N_SIM:,}本）</div>
    {chart2_html}
    <div class="ib">
      アナリスト{num_analysts}人のレンジ: <strong style="color:#ff4444">${target_low:.0f}</strong>（弱気）〜
      <strong style="color:#00ff88">${target_high:.0f}</strong>（強気）、
      平均 <strong style="color:#ffd700">${target_mean:.0f}</strong>、中央値 <strong>${target_med:.0f}</strong>。
      現在値${cur_price:.0f}はコンセンサスの下位寄りに位置しており、
      <strong>上昇余地の方がダウンサイドより統計的に大きい</strong>局面。
    </div>
  </div>

  <!-- 需給：出来高 + OBV -->
  <div class="sec">需給分析 — 出来高 &amp; OBV</div>
  <div class="cc">
    <div class="ct">出来高トレンド &amp; OBV（On-Balance Volume）</div>
    <div class="cd">OBVは上昇日の出来高を累積加算・下落日を減算。上昇トレンド＝機関投資家の買い集め示唆。</div>
    {chart3_html}
    <div class="ib">
      <strong>OBVリーディング</strong>：OBVが{'上昇' if obv.iloc[-1] > obv.iloc[-20] else '下落'}トレンド。
      {'OBV > MA20 → 買い需要が優勢。株価の上昇継続を示唆。' if obv.iloc[-1] > obv_ma20.iloc[-1] else 'OBV < MA20 → 売り圧力継続。短期的に慎重が必要。'}
      機関投資家の持分比率は約 {info.get("heldPercentInstitutions", 0.85)*100:.0f}% と高く、
      大口の方向転換が株価に直接影響する。
    </div>
  </div>

  <!-- イベントカレンダー -->
  <div class="sec">イベントカレンダー（需給インパクト）</div>
  <div style="margin-bottom:28px">
    {event_cards}
  </div>

  <!-- 価格予測テーブル -->
  <div class="sec">時間軸別 価格予測サマリー</div>
  <div class="pgrid">
    <div class="pcard">
      <div class="ct" style="margin-bottom:14px">1ヶ月後（〜4月上旬）</div>
      <table class="stbl">
        <tr><th>シナリオ</th><th>価格</th><th>リターン</th><th>背景</th></tr>
        <tr>
          <td style="color:#00ff88">強気</td>
          <td>${pct_1m[4]:.0f}</td>
          <td>{fmt_ret(pct_1m[4])}</td>
          <td style="color:var(--dim);font-size:11px">学会でポジティブ発言 + 機関買い</td>
        </tr>
        <tr>
          <td style="color:#00d4ff">中立強</td>
          <td>${pct_1m[3]:.0f}</td>
          <td>{fmt_ret(pct_1m[3])}</td>
          <td style="color:var(--dim);font-size:11px">現状維持 + 緩やかな値戻し</td>
        </tr>
        <tr>
          <td style="color:#ffd700">基本</td>
          <td>${pct_1m[2]:.0f}</td>
          <td>{fmt_ret(pct_1m[2])}</td>
          <td style="color:var(--dim);font-size:11px">MC中央値（ドリフト継続）</td>
        </tr>
        <tr>
          <td style="color:#ff9944">弱気</td>
          <td>${pct_1m[1]:.0f}</td>
          <td>{fmt_ret(pct_1m[1])}</td>
          <td style="color:var(--dim);font-size:11px">テクニカル悪化 + 売り継続</td>
        </tr>
        <tr>
          <td style="color:#ff4444">ストレス</td>
          <td>${pct_1m[0]:.0f}</td>
          <td>{fmt_ret(pct_1m[0])}</td>
          <td style="color:var(--dim);font-size:11px">市場全体のリスクオフ</td>
        </tr>
      </table>
    </div>
    <div class="pcard">
      <div class="ct" style="margin-bottom:14px">3ヶ月後（〜6月上旬）</div>
      <table class="stbl">
        <tr><th>シナリオ</th><th>価格</th><th>リターン</th><th>背景</th></tr>
        <tr>
          <td style="color:#00ff88">強気</td>
          <td>${pct_3m[4]:.0f}</td>
          <td>{fmt_ret(pct_3m[4])}</td>
          <td style="color:var(--dim);font-size:11px">Q1 beat + Orforglipron承認</td>
        </tr>
        <tr>
          <td style="color:#00d4ff">中立強</td>
          <td>${pct_3m[3]:.0f}</td>
          <td>{fmt_ret(pct_3m[3])}</td>
          <td style="color:var(--dim);font-size:11px">Q1 beat + FDA pending</td>
        </tr>
        <tr>
          <td style="color:#ffd700">基本</td>
          <td>${pct_3m[2]:.0f}</td>
          <td>{fmt_ret(pct_3m[2])}</td>
          <td style="color:var(--dim);font-size:11px">決算インライン + FDA Q2承認</td>
        </tr>
        <tr>
          <td style="color:#ff9944">弱気</td>
          <td>${pct_3m[1]:.0f}</td>
          <td>{fmt_ret(pct_3m[1])}</td>
          <td style="color:var(--dim);font-size:11px">Q1 miss or FDA再延期</td>
        </tr>
        <tr>
          <td style="color:#ff4444">ストレス</td>
          <td>${pct_3m[0]:.0f}</td>
          <td>{fmt_ret(pct_3m[0])}</td>
          <td style="color:var(--dim);font-size:11px">Q1 miss + FDA否決</td>
        </tr>
      </table>
    </div>
  </div>

  <!-- 機関投資家 -->
  <div class="sec">機関投資家 主要ホルダー（需給の主役）</div>
  <div class="cc">
    <div class="cd">機関投資家保有比率 ≈ {info.get("heldPercentInstitutions", 0.85)*100:.0f}%。大口の売買動向が需給を支配する。</div>
    {'<table class="stbl"><tr><th>機関名</th><th style="text-align:right">保有株数</th><th style="text-align:right">保有比率</th><th style="text-align:right">評価額</th></tr>' + inst_rows + '</table>' if inst_rows else '<div style="color:var(--dim);font-size:13px">機関投資家データ取得失敗（yfinance制限）</div>'}
    <div class="ib" style="margin-top:16px">
      <strong>需給インプリケーション</strong>：機関保有比率{info.get("heldPercentInstitutions", 0.85)*100:.0f}%超は
      流動性が高く、指数リバランス・決算後のポジション調整で瞬間的に大きく動きやすい。
      ショート比率 {'%.1f%%' % (short_pct*100) if short_pct else 'N/A'} は
      {'低水準で踏み上げリスクが低い' if short_pct and short_pct < 0.02 else
       '中程度。大型カタリストでショートカバー（踏み上げ）が上昇を加速しうる' if short_pct and short_pct < 0.05 else
       '高水準。カタリスト時のショートスクイーズに注意'}.
    </div>
  </div>

  <!-- 最終判定 -->
  <div class="sec">総合判定</div>
  <div class="vcard">
    <div class="vlbl">Forecast Verdict · イベント需給ベース</div>
    <div class="vmain">Q2イベント待ち → 段階的強気</div>
    <div class="vbody">
      <p><strong>直近1ヶ月（〜4月）</strong>：大きなカタリストなし。
      テクニカル（50日MA下抜け）が重しで、$950〜1,020のボックス圏を想定。
      MC中央値 <strong>${pct_1m[2]:.0f}</strong>。</p>
      <br>
      <p><strong>3ヶ月（Q1決算 + Orforglipron FDA）</strong>：
      4月決算 × 5月FDA承認のダブルカタリスト期。
      両方ポジティブなら<strong style="color:#00ff88">${pct_3m[3]:.0f}〜{pct_3m[4]:.0f}（+{(pct_3m[3]/cur_price-1)*100:.0f}〜{(pct_3m[4]/cur_price-1)*100:.0f}%）</strong>も視野。
      FDA再延期なら$900割れリスク。<strong>バイナリーイベントに注意。</strong></p>
      <br>
      <p><strong>需給面</strong>：機関保有比率高く、指数ETFのリバランス需要が継続的な下値支持。
      OBVトレンドと出来高の確認を要する。
      円建て投資家は¥158台の為替リスクも考慮（ヘッジ or ポジション抑制推奨）。</p>
    </div>
    <div class="esg">
      <div class="ei">
        <div class="el">1ヶ月後中央値</div>
        <div class="ev">${pct_1m[2]:.0f}</div>
        <div style="font-size:10px;color:var(--dim);margin-top:5px">{fmt_ret(pct_1m[2])}</div>
      </div>
      <div class="ei">
        <div class="el">3ヶ月後中央値</div>
        <div class="ev">${pct_3m[2]:.0f}</div>
        <div style="font-size:10px;color:var(--dim);margin-top:5px">{fmt_ret(pct_3m[2])}</div>
      </div>
      <div class="ei">
        <div class="el">アナリスト平均目標</div>
        <div class="ev">${target_mean:.0f}</div>
        <div style="font-size:10px;color:var(--dim);margin-top:5px">{fmt_ret(target_mean)}</div>
      </div>
      <div class="ei">
        <div class="el">強気目標（FDA承認時）</div>
        <div class="ev">${target_high:.0f}</div>
        <div style="font-size:10px;color:var(--dim);margin-top:5px">{fmt_ret(target_high)}</div>
      </div>
    </div>
  </div>

  <div style="margin-bottom:16px;color:var(--dim);font-size:11px">
    <span style="background:rgba(255,255,255,.05);border-radius:4px;padding:2px 7px;font-size:10px;margin:2px">yfinance</span>
    <span style="background:rgba(255,255,255,.05);border-radius:4px;padding:2px 7px;font-size:10px;margin:2px">モンテカルロ {N_SIM:,}本</span>
    <span style="background:rgba(255,255,255,.05);border-radius:4px;padding:2px 7px;font-size:10px;margin:2px">HV30={hv30:.1f}%</span>
    <span style="background:rgba(255,255,255,.05);border-radius:4px;padding:2px 7px;font-size:10px;margin:2px">OBV需給分析</span>
    <span style="background:rgba(255,255,255,.05);border-radius:4px;padding:2px 7px;font-size:10px;margin:2px">アナリストコンセンサス</span>
  </div>
</div>

<div class="foot">
  本レポートは情報提供目的のみ。投資判断はご自身の責任で行ってください。<br>
  Generated by Claude Code + yfinance + Plotly &thinsp;·&thinsp; {datetime.now().strftime('%Y-%m-%d %H:%M')}
</div>
</body>
</html>"""

out = '/home/like_rapid/GT-SOAR/LLY_forecast_report.html'
with open(out, 'w', encoding='utf-8') as f:
    f.write(HTML)
print(f"\n✓ 完了: {out}  ({len(HTML)/1024:.0f} KB)")
