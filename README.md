# Paraweather

Aplicación con frontend en React y backend en FastAPI.

## Cambios aplicados

- Ajuste de dependencias del frontend para evitar conflictos de instalación (`react` 18 + `date-fns` 3 + `react-leaflet` 4 compatible).
- Fallback de `REACT_APP_BACKEND_URL` en frontend para que no falle cuando la variable no esté definida.
- Limpieza de `index.html` para uso de app móvil (sin badge/scripts externos).
- Se añadió `manifest.json` para dejar la web lista para empaquetar como APK (PWA/WebView).

## Ejecutar backend

```bash
cd backend
pip install -r requirements.txt
# opcional: export MONGO_URL="mongodb://localhost:27017" (o MONGODB_URI) y DB_NAME="paraweather"
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



## Export rápido para Android Studio (ZIP)

Para generar un ZIP importable directamente en Android Studio:

```bash
cd frontend
npm run apk:zip
```

Salida esperada:

- `frontend/dist/paraweather-android-studio.zip`

Qué incluye el ZIP:

- Carpeta `android/` lista para abrir en Android Studio
- `capacitor.config.*`
- `package.json` y `package-lock.json`

Importación:

1. Descomprime el ZIP.
2. En Android Studio: **File > Open**.
3. Selecciona la carpeta descomprimida y abre `android/`.
4. Espera a que Gradle sincronice y compila APK/AAB desde Android Studio.

> Nota: el script crea/sincroniza Capacitor automáticamente y reconstruye el frontend antes de empaquetar.

## APK compilada en GitHub (automático)

Sí: ahora el repo incluye un workflow de GitHub Actions que compila la APK y la sube a GitHub.

- Archivo: `.github/workflows/android-apk.yml`
- Evento manual: **Actions > Build Android APK > Run workflow**
- Evento por release: al crear un tag `v*` (ejemplo: `v1.0.0`) también compila.

Dónde descargar la APK:

1. **Artifacts de Actions**: descarga `paraweather-debug-apk` desde la ejecución.
2. **Releases**: si disparas con tag `v*`, se adjunta `app-debug.apk` al release.

Comandos recomendados para publicar una APK en Releases:

```bash
git tag v1.0.0
git push origin v1.0.0
```


## Disparar la compilación APK en GitHub desde terminal

Si quieres ejecutarlo ya en GitHub (sin entrar al panel web), usa este script:

```bash
cd frontend
export GITHUB_TOKEN="<token_con_repo_y_actions>"
export GITHUB_REPO="tu-usuario/tu-repo"
export GITHUB_REF="main"   # opcional
npm run apk:trigger:github
```

En **Windows PowerShell** usa esto (cada línea por separado):

```powershell
cd frontend
$env:GITHUB_TOKEN="<token_con_repo_y_actions>"
$env:GITHUB_REPO="tu-usuario/tu-repo"
$env:GITHUB_REF="main"   # opcional
npm run apk:trigger:github
```

También puedes lanzarlo en una sola línea en PowerShell:

```powershell
cd frontend; $env:GITHUB_TOKEN="<token>"; $env:GITHUB_REPO="tu-usuario/tu-repo"; $env:GITHUB_REF="main"; npm run apk:trigger:github
```


### Problemas comunes en Windows (PowerShell)

Si te aparece:

- `cd : ... \Desktop\frontend no existe`
- `npm error Missing script: "apk:trigger:github"`

significa que **no estás dentro del repositorio correcto** (o tienes una copia vieja).

Pasos correctos desde cero:

```powershell
cd $HOME\Desktop
git clone https://github.com/Capybla/Paraweather.git
cd Paraweather
git pull
cd frontend
$env:GITHUB_TOKEN="<token_con_repo_y_actions>"
$env:GITHUB_REPO="Capybla/Paraweather"
$env:GITHUB_REF="main"
npm install
npm run apk:trigger:github
```

Si ya tienes el repo clonado en otra ruta, entra a esa carpeta primero:

```powershell
cd "C:\ruta\donde\tengas\Paraweather\frontend"
npm run
```

En la salida de `npm run` debe aparecer `apk:trigger:github`.

Qué hace:

- Llama a la API de GitHub (`workflow_dispatch`) del workflow `.github/workflows/android-apk.yml`.
- Inicia la build en GitHub Actions para generar la APK.


Alternativa sin npm (disparo directo de API desde PowerShell):

```powershell
$headers = @{
  Authorization = "Bearer $env:GITHUB_TOKEN"
  Accept = "application/vnd.github+json"
  "X-GitHub-Api-Version" = "2022-11-28"
}
$body = @{ ref = "main" } | ConvertTo-Json
Invoke-RestMethod -Method POST -Uri "https://api.github.com/repos/Capybla/Paraweather/actions/workflows/android-apk.yml/dispatches" -Headers $headers -Body $body
```

Si todo va bien, GitHub responde sin contenido (HTTP 204) y la ejecución aparece en Actions.

Luego descárgala desde:

- **Actions artifacts**: `paraweather-debug-apk`
- **Releases** (si disparas por tag `v*`)

## Modo offline

- La app guarda el último parte meteorológico y lo muestra cuando no hay red.
- Se muestra alerta de desconexión y se mantiene navegación con GPS local.
- En build de producción se registra un service worker para cachear recursos base.


## Estructura para mantenimiento (Open Source)

- `frontend/src/App.js`: UI principal y lógica de widgets.
- `frontend/src/config.js`: valores por defecto de páginas/widgets y claves de almacenamiento (punto recomendado para personalización).
- `frontend/src/serviceWorkerRegistration.js` + `frontend/public/sw.js`: estrategia offline/cache.
- `backend/server.py`: API y lógica de reglas de vuelo.

## Guía rápida para contribuir

- Haz cambios pequeños y atómicos por PR.
- Mantén configuraciones editables en archivos de configuración (ejemplo: `frontend/src/config.js`) en lugar de hardcodear en componentes.
- Verifica siempre con:
  - `cd frontend && node --check src/App.js`
  - `cd frontend && npm run build`

## Personalización Open Source

- Puedes definir `REACT_APP_REPO_URL` para que el panel Open Source de la UI apunte a tu repositorio real en GitHub/GitLab.
- El diseño está pensado para ser profesional y reusable: componentes visuales y configuración separada en `frontend/src/App.js` + `frontend/src/config.js`.
