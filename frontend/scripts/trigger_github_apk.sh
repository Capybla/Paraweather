#!/usr/bin/env bash
set -euo pipefail

# Trigger the GitHub Actions workflow that builds the Android APK.
# Required env vars:
#   GITHUB_TOKEN  -> token with repo/actions permissions
#   GITHUB_REPO   -> owner/repo (example: capybla/paraweather)
# Optional:
#   GITHUB_REF    -> git ref to build (default: main)
#   WORKFLOW_FILE -> workflow filename (default: android-apk.yml)

if ! command -v curl >/dev/null 2>&1; then
  echo "Error: curl no está disponible en el sistema." >&2
  exit 1
fi

: "${GITHUB_TOKEN:?Falta GITHUB_TOKEN (token con permisos repo/actions).}"
: "${GITHUB_REPO:?Falta GITHUB_REPO con formato owner/repo.}"

GITHUB_REF="${GITHUB_REF:-main}"
WORKFLOW_FILE="${WORKFLOW_FILE:-android-apk.yml}"

API_URL="https://api.github.com/repos/${GITHUB_REPO}/actions/workflows/${WORKFLOW_FILE}/dispatches"

HTTP_CODE=$(curl -sS -o /tmp/trigger_apk_workflow_response.json -w "%{http_code}" \
  -X POST "$API_URL" \
  -H "Accept: application/vnd.github+json" \
  -H "Authorization: Bearer ${GITHUB_TOKEN}" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  -d "{\"ref\":\"${GITHUB_REF}\"}")

if [[ "$HTTP_CODE" != "204" ]]; then
  echo "❌ No se pudo disparar el workflow. HTTP: ${HTTP_CODE}" >&2
  cat /tmp/trigger_apk_workflow_response.json >&2 || true
  exit 1
fi

echo "✅ Workflow disparado correctamente en ${GITHUB_REPO} (${WORKFLOW_FILE}, ref=${GITHUB_REF})."
echo "➡️ Revisa: https://github.com/${GITHUB_REPO}/actions/workflows/${WORKFLOW_FILE}"
