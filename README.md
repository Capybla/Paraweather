# Paraweather

Aplicación con frontend en React y backend en FastAPI.

## Cambios aplicados

- Ajuste de dependencias del frontend para evitar conflictos de instalación (`react` 18 + `date-fns` 3).
- Fallback de `REACT_APP_BACKEND_URL` en frontend para que no falle cuando la variable no esté definida.
- Limpieza de `index.html` para uso de app móvil (sin badge/scripts externos).
- Se añadió `manifest.json` para dejar la web lista para empaquetar como APK (PWA/WebView).

## Ejecutar backend

```bash
cd backend
pip install -r requirements.txt
# opcional: export MONGO_URL="mongodb://localhost:27017" y DB_NAME="paraweather"
uvicorn server:app --host 0.0.0.0 --port 8001
```

## Ejecutar frontend

```bash
cd frontend
npm install
npm start
```

## Build de producción

```bash
cd frontend
npm run build
```

## Empaquetar APK (Capacitor recomendado)

```bash
cd frontend
npm install @capacitor/core @capacitor/cli @capacitor/android
npx cap init Paraweather com.paraweather.app --web-dir=build
npm run build
npx cap add android
npx cap sync android
npx cap open android
```

Con eso se abre Android Studio para generar el APK/AAB.
