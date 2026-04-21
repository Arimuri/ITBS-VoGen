# ITBS-VoGen

AI音源分離で抽出したアーティファクトだらけのボーカルから、クリーンなボーカルを新規生成するツール。

## 目的

Demucs / UVR 等のAI音源分離で得られたボーカルには、分離アーティファクト・帯域欠損・ガビガビノイズが残ることが多い。ITBS-VoGen はこれを入力として、歌い方・節回し（メロディ、フレージング、ビブラート）を保持したままクリーンなボーカルを再生成する。

## アプローチ

[RVC (Retrieval-based Voice Conversion WebUI)](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) をコア推論エンジンとして使用。

- 入力ボーカルから HuBERT で内容特徴、RMVPE で F0（ピッチ）を抽出
- 学習済み話者モデルの VITS デコーダで**一から再合成**することで、入力のアーティファクトは構造上除去される
- 出力話者は選択した話者モデルの声になる（入力話者は保持されない仕様）

## 動作環境

- macOS (Apple Silicon 想定)
- Python 3.10
- PyTorch (CPU / MPS)

## セットアップ

```bash
# 依存取得
git clone --recurse-submodules https://github.com/Arimuri/ITBS-VoGen.git
cd ITBS-VoGen

# Python環境 (uv推奨)
uv sync

# 事前学習済みウェイト取得
bash scripts/download_models.sh
```

※ 事前学習済みウェイトはリポジトリに同梱していない。各ウェイトの入手元・ライセンスは
[THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md) 参照。

## ライセンス

本リポジトリのコードは [MIT License](./LICENSE)。

依存する第三者ソフトウェア・モデルのライセンスは
[THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md) を参照のこと。
