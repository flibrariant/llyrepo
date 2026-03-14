# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 目的

1ヶ月で10%上昇が期待できる銘柄を探す株式分析システム。

## 分析思想

カタリストがある銘柄を先にピックアップし、テクニカルが崩れている銘柄を除外する。
「材料があるか」＋「今が買える状態か」の2軸で判断する。

## 個別銘柄分析の視点

1. 業績進捗率と上方修正余地
2. 来期見通しの強弱（保守的か強気か）
3. 自社株買い・増配など株価カタリストの有無
4. 1ヶ月以内に10%上昇の材料になりうるか
5. リスク要因
6. 最終判定：買い・様子見・見送り

## データソース

- yfinance：財務数値・株価・アナリスト目標株価
- Playwright MCP：TDnet決算短信・ウェブ情報の自動取得

## 作業ディレクトリ

/home/like_rapid/GT-SOAR（WSL側）
OneDrive側のGT-SOARは無視する。
