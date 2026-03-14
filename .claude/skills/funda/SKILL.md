---
name: funda
description: 日本株の銘柄コードリストを受け取り、詳細ファンダメンタルズ分析HTMLレポートを生成する
allowed-tools: Bash, Read, Write
---

# /funda

銘柄コードを受け取り、yfinanceで財務データを取得してファンダメンタルズ比較分析HTMLを生成します。

## 使い方

```
/funda $ARGUMENTS
```

`$ARGUMENTS` にはスペース区切りの銘柄コード（数字4桁、TSEコード）を指定してください。

例:
```
/funda 9908 8101 9441 3333 256A
```

## 処理内容

### データ取得（yfinance）
各銘柄について以下を取得:
- `t.info`: PER, PBR, PSR, 配当利回り, 時価総額, ROE, bookValue, dividendRate
- `t.financials`: 過去4期の損益計算書（売上・営業利益・純利益）
- `t.quarterly_financials`: 直近4四半期のP&L
- `t.balance_sheet`: 総資産, 自己資本, 有利子負債
- `t.cashflow`: 営業CF, 投資CF, 設備投資, FCF
- `t.dividends`: 配当履歴
- `t.history(period='2y', interval='1mo')`: 月次株価（52週高値・安値）

### 計算指標
- **DOE** = dividendRate / bookValue × 100
  - fallback: dividendYield × priceToBook × 100
- **自己資本比率** = 自己資本 / 総資産 × 100
- **D/E比率** = 有利子負債 / 自己資本
- **FCF** = 営業CF + 投資CF（または OCF - Capex）
- **営業利益率** = 営業利益 / 売上高 × 100

### HTMLレポート構成
1. **比較テーブル**: 全銘柄の主要指標一覧
2. **52週レンジバー**: 現在株価の位置を視覚化
3. **P&Lトレンドチャート**: 4期分の売上・営業利益・純利益の推移
4. **配当履歴チャート**: 年次DPS推移
5. **SWOTグリッド**: 各銘柄の強み・弱み・機会・リスク
6. **投資判定カード**: ★評価 + 推奨理由 + カタリスト + リスク

### 判定基準（参考）
- ★★★★★: 成長 + 低バリュエーション + DOE高 + 無借金
- ★★★★: 上記2-3条件が揃う
- ★★★: バリュエーション割安だが懸念あり
- ★★: 様子見
- ★: 見送り

### 出力
- ファイル名: `reports/screening/funda_{銘柄1}_{銘柄2}_..._{日付}.html`
- 中間データ: `/tmp/funda_data.json`

## 技術メモ

```python
import yfinance as yf

tickers = ['9908', '8101', '9441', '3333', '256A']
data = {}
for code in tickers:
    t = yf.Ticker(f'{code}.T')
    info = t.info
    data[code] = {
        'info': info,
        'financials': t.financials.to_dict() if t.financials is not None else {},
        'balance_sheet': t.balance_sheet.to_dict() if t.balance_sheet is not None else {},
        'cashflow': t.cashflow.to_dict() if t.cashflow is not None else {},
        'dividends': t.dividends.to_dict(),
        'price_history': t.history(period='2y', interval='1mo').to_dict()
    }

# DOE計算
div_rate = info.get('dividendRate')
book_val = info.get('bookValue')
doe = (div_rate / book_val * 100) if (book_val and div_rate) else None
if doe is None:
    div_yield = info.get('dividendYield', 0) or 0
    pbr = info.get('priceToBook', 0) or 0
    doe = div_yield * pbr * 100 if (div_yield and pbr) else None
```

## 分析視点（CLAUDE.mdより）

1. 業績進捗率と上方修正余地
2. 来期見通しの強弱（保守的か強気か）
3. 自社株買い・増配など株価カタリストの有無
4. 1ヶ月以内に10%上昇の材料になりうるか
5. リスク要因
6. 最終判定：買い・様子見・見送り
