#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARCHIVE_PATH="${REPO_ROOT}/third_party/LKH3/LKH-3.0.14.tar"
EXTRACT_ROOT="${REPO_ROOT}/third_party/LKH3"
SOURCE_DIR="${EXTRACT_ROOT}/LKH-3.0.14"
BINARY_PATH="${SOURCE_DIR}/LKH"

echo "[setup_lkh] repo root: ${REPO_ROOT}"
echo "[setup_lkh] pinned archive: ${ARCHIVE_PATH}"

if [[ ! -f "${ARCHIVE_PATH}" ]]; then
  echo "[setup_lkh] ERROR: pinned archive not found at ${ARCHIVE_PATH}" >&2
  exit 1
fi

echo "[setup_lkh] archive top entries:"
tar -tf "${ARCHIVE_PATH}" | sed -n '1,10p'

if [[ ! -d "${SOURCE_DIR}" ]]; then
  echo "[setup_lkh] extracting ${ARCHIVE_PATH} into ${EXTRACT_ROOT}"
  tar -xf "${ARCHIVE_PATH}" -C "${EXTRACT_ROOT}"
else
  echo "[setup_lkh] source directory already exists: ${SOURCE_DIR}"
fi

echo "[setup_lkh] building LKH in ${SOURCE_DIR}"
make -C "${SOURCE_DIR}"

if [[ ! -f "${BINARY_PATH}" ]]; then
  echo "[setup_lkh] ERROR: build completed but binary not found at ${BINARY_PATH}" >&2
  exit 1
fi
if [[ ! -x "${BINARY_PATH}" ]]; then
  echo "[setup_lkh] ERROR: binary exists but is not executable: ${BINARY_PATH}" >&2
  exit 1
fi

echo "[setup_lkh] SUCCESS"
echo "[setup_lkh] executable path: ${BINARY_PATH}"
