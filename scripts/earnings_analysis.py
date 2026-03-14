#!/usr/bin/env python3
"""NEC (6701) / 富士通 (6702) 決算データ取得・分析"""
import yfinance as yf
import pandas as pd
import numpy as np
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

for code, name in [('6701.T', 'NEC'), ('6702.T', '富士通')]:
    print(f"\n{'='*70}")
    print(f"  {name} ({code})")
    print(f"{'='*70}")
    t = yf.Ticker(code)
    info = t.info

    # 基本情報
    keys = ['marketCap','enterpriseValue','trailingPE','forwardPE','priceToBook',
            'trailingEps','forwardEps','dividendYield','dividendRate',
            'profitMargins','operatingMargins','returnOnEquity','returnOnAssets',
            'revenueGrowth','earningsGrowth','earningsQuarterlyGrowth',
            'totalRevenue','totalCash','totalDebt','freeCashflow',
            'targetMeanPrice','targetHighPrice','targetLowPrice','numberOfAnalystOpinions',
            'recommendationMean','recommendationKey']
    print("\n--- info ---")
    for k in keys:
        v = info.get(k, 'N/A')
        if isinstance(v, (int, float)) and abs(v) > 1e9:
            print(f"  {k}: {v/1e9:.1f}B")
        elif isinstance(v, float):
            print(f"  {k}: {v}")
        else:
            print(f"  {k}: {v}")

    # 財務諸表
    print("\n--- 損益計算書(年次) ---")
    inc = t.income_stmt
    if inc is not None and not inc.empty:
        for col in inc.columns[:3]:
            print(f"\n  期: {col.strftime('%Y-%m-%d') if hasattr(col,'strftime') else col}")
            for row in ['Total Revenue','Operating Income','Net Income','Basic EPS','EBITDA']:
                if row in inc.index:
                    v = inc.loc[row, col]
                    if pd.notna(v):
                        if abs(v) > 1e9:
                            print(f"    {row}: {v/1e9:.1f}B ({v/1e6:.0f}M)")
                        else:
                            print(f"    {row}: {v}")

    print("\n--- 損益計算書(四半期) ---")
    qi = t.quarterly_income_stmt
    if qi is not None and not qi.empty:
        for col in qi.columns[:6]:
            print(f"\n  四半期: {col.strftime('%Y-%m-%d') if hasattr(col,'strftime') else col}")
            for row in ['Total Revenue','Operating Income','Net Income','Basic EPS','EBITDA']:
                if row in qi.index:
                    v = qi.loc[row, col]
                    if pd.notna(v):
                        if abs(v) > 1e9:
                            print(f"    {row}: {v/1e9:.1f}B ({v/1e6:.0f}M)")
                        else:
                            print(f"    {row}: {v}")

    print("\n--- バランスシート(四半期) ---")
    qb = t.quarterly_balance_sheet
    if qb is not None and not qb.empty:
        col = qb.columns[0]
        print(f"  直近: {col.strftime('%Y-%m-%d') if hasattr(col,'strftime') else col}")
        for row in ['Total Assets','Total Debt','Stockholders Equity','Cash And Cash Equivalents',
                     'Net Debt','Working Capital','Goodwill And Other Intangible Assets']:
            if row in qb.index:
                v = qb.loc[row, col]
                if pd.notna(v):
                    print(f"    {row}: {v/1e9:.1f}B")

    print("\n--- キャッシュフロー(四半期) ---")
    qcf = t.quarterly_cashflow
    if qcf is not None and not qcf.empty:
        for col in qcf.columns[:4]:
            print(f"\n  四半期: {col.strftime('%Y-%m-%d') if hasattr(col,'strftime') else col}")
            for row in ['Operating Cash Flow','Capital Expenditure','Free Cash Flow',
                         'Repurchase Of Capital Stock']:
                if row in qcf.index:
                    v = qcf.loc[row, col]
                    if pd.notna(v):
                        print(f"    {row}: {v/1e9:.1f}B")

print("\n\n===== データ取得完了 =====")
