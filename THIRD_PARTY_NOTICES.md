# Third-Party Notices

ITBS-VoGen は以下の第三者ソフトウェア・モデルに依存している。各コンポーネントのライセンスは個別に適用される。

---

## ソフトウェア

### RVC (Retrieval-based Voice Conversion WebUI)

- **入手元**: https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI
- **ライセンス**: MIT License
- **著作権表記**:
  ```
  Copyright (c) 2023 liujing04
  Copyright (c) 2023 源文雨
  Copyright (c) 2023 Ftps
  ```
- **組み込み方法**: `third_party/rvc/` に git submodule として参照。コード改変なし。
- **ライセンス全文**: `third_party/rvc/LICENSE` を参照

---

### Apollo (JusperLee/Apollo) — 使用する場合のみ

- **入手元**: https://github.com/JusperLee/Apollo
- **ライセンス**: Creative Commons Attribution-ShareAlike 4.0 International (CC-BY-SA 4.0)
- **注意**: CC-BY-SA 4.0 はコピーレフト条項を含む。ソースコードを本リポジトリに取り込まず、**別プロセス・CLI・pip依存として呼び出す形式**でのみ利用する。これにより派生著作物ではなく集合著作物（Aggregation）として扱い、ShareAlike の感染範囲を ITBS-VoGen 本体のコードに及ぼさない方針。
- **組み込み方法**: 現時点では未統合。利用時は `pip install` もしくは独立したサブプロセスとして呼び出す。

---

## 事前学習済みモデル / ウェイト

事前学習済みモデルは**本リポジトリに同梱していない**。`scripts/download_models.sh` で各配布元から取得する。以下は代表的なモデルと注意事項：

### HuBERT (Facebook/Meta)

- 一般に Facebook/Meta が公開する HuBERT 重みを RVC が使用
- ライセンス: 配布元の利用規約に従う。商用利用制限が付くことがあるため個別確認

### RVC 話者モデル（各コミュニティ公開）

- Hugging Face、weights.gg 等で配布されている個別話者モデルは **各配布者が独自ライセンスを指定**することが多い
- 「個人利用のみ」「商用不可」「歌唱利用時は本人クレジット必須」などの条件が付く場合がある
- 使用前に必ず配布元の利用規約を確認すること

### Apollo 事前学習済みウェイト

- Hugging Face 等で配布。配布ページに記載のライセンスに従う

---

## 本プロジェクトのライセンス

ITBS-VoGen 本体のコード（`src/` 以下および設定ファイル等、第三者提供部分を除く）は
[MIT License](./LICENSE) で提供する。
