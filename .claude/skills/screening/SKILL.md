---
name: screening
description: 銘柄コードリストのDOE/PSR/PBR/PERを計算・スコアリングし、1-3ヶ月で上昇期待の上位銘柄を選定してHTML出力する
allowed-tools: Bash, Read, Write
---

# /screening

銘柄コードのリストを受け取り、各社の財務指標を取得してスコアリングし、上位銘柄を選定したHTMLレポートを生成します。

## 使い方

```
/screening $ARGUMENTS
```

`$ARGUMENTS` には以下のいずれかを指定:
- スペース区切りの銘柄コード
- スクリーニング結果のスクリーンショット画像パス（PILで自動読み取り）

例（コード直接指定）:
```
/screening 9908 8101 9441 3333 256A 2060 7971 5013 7438 2475
```

例（スクリーンショット指定）:
```
/screening data/screenshots/screening_20260220.png
```

## 処理内容

### データ取得（yfinance）
各銘柄 `{code}.T` について:
- `t.info`: dividendRate, bookValue, dividendYield, priceToBook, priceToSalesTrailing12Months, trailingPE, marketCap, returnOnEquity, shortName

### 指標計算
| 指標 | 計算方法 |
|------|---------|
| DOE  | dividendRate / bookValue × 100（fallback: dividendYield × priceToBook × 100） |
| PSR  | priceToSalesTrailing12Months |
| PBR  | priceToBook |
| PER  | trailingPE |
| 配当利回り | dividendYield × 100 |

### スコアリング（100点満点）
| 指標 | 配点 | 基準 |
|------|------|------|
| DOE  | 30点 | ≥4%=30, ≥3%=22, ≥2%=15, ≥1%=8 |
| PSR  | 25点 | ≤0.3=25, ≤0.5=20, ≤0.8=15, ≤1.2=10, ≤2.0=5 |
| PBR  | 20点 | ≤0.5=20, ≤0.8=16, ≤1.0=12, ≤1.5=8, ≤2.0=4 |
| PER  | 15点 | ≤8=15, ≤12=12, ≤15=9, ≤20=6, ≤30=3 |
| 配当利回り | 10点 | ≥4%=10, ≥3%=8, ≥2%=6, ≥1%=3 |

### HTMLレポート構成
1. **PSR vs DOE バブルチャート**: バブルサイズ=時価総額、色=スコア
2. **上位10銘柄 スコアバー**: 指標別の内訳付き横棒グラフ
3. **全銘柄テーブル**: スコア順ソート、DOE/PSR/PBR/PER/配当利回り/時価総額
4. **上位10銘柄 ピックカード**: 各社の投資論点・カタリスト・リスク

### 出力
- ファイル名: `reports/screening/DOE_screening_{日付}.html`
- 中間データ: `/tmp/screening_data.json`

## 技術メモ

```python
import yfinance as yf
import json

records = []
for code in tickers:
    try:
        t = yf.Ticker(f'{code}.T')
        info = t.info

        div_rate = info.get('dividendRate')
        book_val = info.get('bookValue')
        doe = (div_rate / book_val * 100) if (book_val and div_rate) else None
        if doe is None:
            div_yield = info.get('dividendYield', 0) or 0
            pbr_val = info.get('priceToBook', 0) or 0
            doe = div_yield * pbr_val * 100 if (div_yield and pbr_val) else None

        psr = info.get('priceToSalesTrailing12Months')
        pbr = info.get('priceToBook')
        per = info.get('trailingPE')
        mcap = info.get('marketCap', 0) / 1e8  # 億円

        records.append({
            'code': code,
            'name': info.get('shortName', code),
            'doe': doe, 'psr': psr, 'pbr': pbr, 'per': per,
            'mcap': mcap, 'score': calculate_score(doe, psr, pbr, per, div_yield)
        })
    except Exception as e:
        print(f'{code}: {e}')

# スコア計算
def calculate_score(doe, psr, pbr, per, div_yield):
    score = 0
    if doe:
        if doe >= 4: score += 30
        elif doe >= 3: score += 22
        elif doe >= 2: score += 15
        elif doe >= 1: score += 8
    # PSR, PBR, PER, divYield も同様に...
    return score
```

## スクリーンショットからの銘柄抽出

画像パスが指定された場合は PIL で銘柄コード列を読み取る:

```python
from PIL import Image
img = Image.open(screenshot_path)
# 左端の銘柄コード列を3倍ズームしてターミナルに表示
# 4桁数字（または英数字混在の新形式コード）を手動確認 or OCR
```

注: TSEの新形式コード（例: 256A）も有効。`{code}.T` でyfinance取得可能。
