---
name: trade-analyze
description: SBI証券の信用取引注文CSVから3分足チャートに取引を重ねて表示し、P&L分析HTMLを生成する
allowed-tools: Bash, Read, Write
---

# /trade-analyze

SBI証券の信用取引注文CSVを受け取り、以下を実行してHTMLレポートを生成します：
1. CSVをShift-JIS読み込みで解析し、約定済み取引を抽出
2. yfinanceで当日の1分足を取得して3分足にリサンプル
3. Plotlyでローソク足チャートに取引エントリー/エグジットを重ねて表示
4. 取引ペアをFIFO方式でマッチングしてP&L計算
5. 累積P&Lチャートと取引明細テーブルを含むHTMLを出力

## 使い方

```
/trade-analyze $ARGUMENTS
```

`$ARGUMENTS` には注文CSVのファイルパスを指定してください。

例:
```
/trade-analyze data/stockorder(JP)_20260220.csv
```

## 処理内容

### CSV解析
- エンコード: Shift-JIS (`encoding='shift_jis', errors='replace'`)
- 状況が「約定」の行のみ抽出
- 売買区分: 買建(Long IN), 売建(Short IN), 買埋(Short OUT), 売埋(Long OUT)
- 対象銘柄は自動検出（最初の約定銘柄コードを使用）

### チャート生成
- yfinanceで1分足取得 → 3分足リサンプル（`origin='09:00:00'`）
- X軸: `type:'date'` 形式（カテゴリ型にしない）
- タイムゾーン: Asia/Tokyo
- マーカー: 買建(▼緑), 売埋(△緑), 売建(▲赤), 買埋(▽赤)
- エグジット点にP&L金額をアノテーション表示

### P&L計算
- FIFOペアリング
- 勝率・プロフィットファクター・最大利益/損失を集計

### 出力
- ファイル名: `reports/trade_log/{銘柄コード}_{日付}_analysis.html`

## 技術メモ

```python
# Shift-JIS CSV読み込み
with open(csv_path, encoding='shift_jis', errors='replace') as f:
    reader = csv.reader(f)

# 3分足リサンプル
df1m = ticker.history(period='1d', interval='1m', prepost=False)
df1m.index = df1m.index.tz_convert('Asia/Tokyo')  # ※pytzオブジェクトではなく文字列を使う
origin_ts = pd.Timestamp(f'{date} 09:00:00', tz='Asia/Tokyo')
df3m = df1m.resample('3min', origin=origin_ts).agg(
    {'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}
).dropna(subset=['Open'])

# Plotly X軸設定（カテゴリ型だとマーカーが軸末尾にずれるのでdate型必須）
xaxis=dict(type='date', tickformat='%H:%M', range=[...])
```
