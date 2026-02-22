#!/usr/bin/env bash
set -euo pipefail

APP_NAME="Paraweather"
APP_ID="com.paraweather.app"
ZIP_NAME_DEFAULT="paraweather-android-studio.zip"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OUT_DIR="$FRONTEND_DIR/dist"
ZIP_PATH="${1:-$OUT_DIR/$ZIP_NAME_DEFAULT}"

cd "$FRONTEND_DIR"

if ! command -v zip >/dev/null 2>&1; then
  echo "Error: 'zip' no está instalado en el sistema."
  exit 1
fi

mkdir -p "$OUT_DIR"

if [ ! -d node_modules ]; then
  echo "[1/6] Instalando dependencias npm..."
  npm install
fi

echo "[2/6] Instalando Capacitor (si hace falta)..."
npm install @capacitor/core @capacitor/cli @capacitor/android --save-dev >/dev/null

echo "[3/6] Build de producción..."
npm run build

if [ ! -f capacitor.config.json ] && [ ! -f capacitor.config.ts ]; then
  echo "[4/6] Inicializando Capacitor..."
  npx cap init "$APP_NAME" "$APP_ID" --web-dir=build
fi

if [ ! -d android ]; then
  echo "[5/6] Creando proyecto Android..."
  npx cap add android
fi

echo "[6/6] Sincronizando proyecto Android..."
npx cap sync android

rm -f "$ZIP_PATH"

# Empaquetado portable para abrir directamente en Android Studio.
(
  cd "$FRONTEND_DIR"
  zip -r "$ZIP_PATH" android capacitor.config.* package.json package-lock.json >/dev/null
)

echo "✅ ZIP generado: $ZIP_PATH"
echo "➡️  Android Studio: File > Open > seleccionar carpeta descomprimida/android"
