# ITBS-VoGen

AI音源分離で抽出したアーティファクトだらけのボーカルから、クリーンなボーカルを新規生成するツール。

## 目的

Demucs / UVR 等のAI音源分離で得られたボーカルには、分離アーティファクト・帯域欠損・ガビガビノイズが残ることが多い。ITBS-VoGen はこれを入力として、歌い方・節回し（メロディ、フレージング、ビブラート）を保持したままクリーンなボーカルを再生成する。

## アプローチ

[RVC (Retrieval-based Voice Conversion WebUI)](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) をコア推論エンジンとして使用（`third_party/rvc/` に git submodule として参照）。

- 入力ボーカルから HuBERT で内容特徴、RMVPE で F0（ピッチ）を抽出
- 学習済み話者モデルの VITS デコーダで**一から再合成**することで、入力のアーティファクトは構造上除去される
- 出力話者は選択した話者モデルの声になる（入力話者は保持されない仕様）

## 動作環境

- macOS (Apple Silicon 想定)
- Python 3.10（RVCの `numba==0.56.4` 等の制約で 3.11 以降は非推奨）
- PyTorch (CPU / MPS)

## セットアップ

```bash
# 1. リポジトリ取得（submodule込み）
git clone --recurse-submodules https://github.com/Arimuri/ITBS-VoGen.git
cd ITBS-VoGen

# 2. Python 3.10 の仮想環境
python3.10 -m venv .venv
source .venv/bin/activate

# 3. pip を <24.1 にダウングレード（RVC依存の omegaconf メタデータが古いため必須）
pip install 'pip<24.1'

# 4. RVC の依存をインストール（aria2 を除外：Mac arm64 で未対応）
grep -v '^aria2$' third_party/rvc/requirements.txt | pip install -r /dev/stdin

# 5. このプロジェクト自体をインストール（CLI コマンド itbs-vogen が入る）
pip install -e .

# 6. RVC ベースモデル（HuBERT + RMVPE + pretrained_v2）を取得
bash scripts/download_models.sh

# 7. 話者モデルを手動配置
# models/speakers/<speaker_name>/model.pth   （必須）
# models/speakers/<speaker_name>/model.index （任意、あれば音色再現が安定）
```

## 使い方

### 1. 話者モデルを学習

商用ライセンスがクリアな公開歌唱モデルは実質存在しないため、本プロジェクトでは**自前の学習**を前提とする。学習は GPU が強く推奨で、**Google Colab (T4) での学習運用**を[notebooks/train_on_colab.ipynb](./notebooks/train_on_colab.ipynb) で用意している。

Mac CPU でも動くが 29分 / 200epoch で 20-50 時間コースになる。検証用:

```bash
itbs-vogen train \
    -s mymodel \
    -d Train/set1 \
    --sr 48k \
    --epochs 20 \
    --batch-size 4
```

学習が完了すると `models/speakers/mymodel/model.pth` + `model.index` が生成される。

### 2. 劣化ボーカルをクリーン化

```bash
itbs-vogen infer inputs/test_vocal.wav \
    -o outputs/restored.wav \
    -s mymodel
```

主要オプション:

| オプション | 既定値 | 説明 |
|---|---|---|
| `--f0-method` | `rmvpe` | F0抽出器。劣化耐性が最も高いので既定は rmvpe |
| `--f0-up-key` | `0` | ピッチシフト(半音)。メロディ保持なら 0 のまま |
| `--index-rate` | `0.66` | 話者インデックス適用率。高いほど話者音色寄り |
| `--protect` | `0.33` | 子音保護。子音がモゴつく場合は上げる |
| `--device` | 自動 | 省略で mps/cuda/cpu を自動検出 |

## ライセンス

本リポジトリのコードは [MIT License](./LICENSE)。依存する第三者ソフトウェア・モデルのライセンスは [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md) を参照のこと。
