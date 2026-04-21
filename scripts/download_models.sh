#!/usr/bin/env bash
# Download pre-trained base models required by RVC inference.
#
# This script fetches only the base assets (HuBERT content encoder and RMVPE
# F0 extractor) that RVC needs regardless of which speaker model is used.
#
# Speaker-specific models (*.pth + *.index) are NOT downloaded here — they
# are user-selected and carry individual licenses. Place them manually under
# models/speakers/<name>/ after reviewing their distribution terms.
#
# Usage:
#   bash scripts/download_models.sh
#
# Requires: curl, or wget.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASSETS_DIR="${REPO_ROOT}/third_party/rvc/assets"

HUBERT_URL="https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt"
RMVPE_URL="https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt"

HUBERT_DEST="${ASSETS_DIR}/hubert/hubert_base.pt"
RMVPE_DEST="${ASSETS_DIR}/rmvpe/rmvpe.pt"

mkdir -p "$(dirname "${HUBERT_DEST}")" "$(dirname "${RMVPE_DEST}")" \
         "${REPO_ROOT}/inputs" "${REPO_ROOT}/outputs" "${REPO_ROOT}/models/speakers"

download() {
    local url="$1"
    local dest="$2"
    if [[ -f "${dest}" ]]; then
        echo "[skip] ${dest} already exists"
        return 0
    fi
    echo "[download] ${url}"
    if command -v curl >/dev/null 2>&1; then
        curl -L --fail --progress-bar -o "${dest}" "${url}"
    elif command -v wget >/dev/null 2>&1; then
        wget --show-progress -O "${dest}" "${url}"
    else
        echo "error: neither curl nor wget is available" >&2
        exit 1
    fi
}

download "${HUBERT_URL}" "${HUBERT_DEST}"
download "${RMVPE_URL}"  "${RMVPE_DEST}"

cat <<'EOF'

Base models downloaded.

Next steps:
  1. Obtain a speaker model (*.pth + *.index) from a source you trust and whose
     license permits your intended use.
  2. Place the files under:
       models/speakers/<speaker_name>/model.pth
       models/speakers/<speaker_name>/model.index
  3. See THIRD_PARTY_NOTICES.md for licensing guidance.
EOF
