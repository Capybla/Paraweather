#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MANIFEST="$FRONTEND_DIR/android/app/src/main/AndroidManifest.xml"
PERMS_FILE="$FRONTEND_DIR/resources/android-permissions.xml"

if [[ ! -f "$MANIFEST" ]]; then
  echo "⚠️ AndroidManifest no encontrado en $MANIFEST (ejecuta 'npx cap add android' primero)."
  exit 0
fi

if [[ ! -f "$PERMS_FILE" ]]; then
  echo "⚠️ Archivo de permisos no encontrado: $PERMS_FILE"
  exit 0
fi

python3 - "$MANIFEST" "$PERMS_FILE" <<'PY'
from pathlib import Path
import re
import sys

manifest_path = Path(sys.argv[1])
perms_path = Path(sys.argv[2])
manifest = manifest_path.read_text()
perms = perms_path.read_text()

entries = re.findall(r'<uses-(?:permission|feature)[^>]+>', perms)
updated = manifest
for entry in entries:
    if entry not in updated:
        updated = updated.replace('<application', f'    {entry}\n\n    <application', 1)

if updated != manifest:
    manifest_path.write_text(updated)
    print('✅ Permisos Android aplicados al AndroidManifest.xml')
else:
    print('ℹ️ Permisos Android ya estaban presentes.')
PY
