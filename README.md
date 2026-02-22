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
- Evento por push: ramas `main`, `work` y `codex/**` (incluye `codex/paraweather`).
- Evento por release: al crear un tag `v*` (ejemplo: `v1.0.0`) también compila.

Dónde descargar la APK:

1. **Artifacts de Actions**: descarga `paraweather-debug-apk` desde la ejecución.
2. **Releases**: si disparas con tag `v*`, se adjunta `app-debug.apk` al release.

Comandos recomendados para publicar una APK en Releases:

```bash
git tag v1.0.0
git push origin v1.0.0
```


## Métodos para extraer la APK (detallado)

### Método A — GitHub Actions (recomendado, sin Android Studio local)

1. Ve a **GitHub > Actions > Build Android APK**.
2. Pulsa **Run workflow** sobre `main`.
3. Espera a que termine `build-apk`.
4. Descarga el artifact `paraweather-debug-apk`.

Ventajas:
- Reproducible para todo el equipo.
- No depende de tu SDK local.

### Método B — Generar release con APK adjunta

Si creas un tag `v*`, el workflow adjunta automáticamente `app-debug.apk` al release.

```bash
git tag v1.0.0
git push origin v1.0.0
```

Luego descarga la APK desde **GitHub > Releases**.

### Método C — Android Studio (local)

```bash
cd frontend
npm install
npm run build
npx cap add android   # solo la primera vez
npx cap sync android
npx cap open android
```

En Android Studio:
- **Build > Build Bundle(s) / APK(s) > Build APK(s)**.
- Ruta típica de salida: `frontend/android/app/build/outputs/apk/debug/app-debug.apk`.

### Método D — ZIP importable para Android Studio

```bash
cd frontend
npm run apk:zip
```

Genera `frontend/dist/paraweather-android-studio.zip` para compartir/importar proyecto Android rápidamente.

### Error común de build local: `Unsupported class file major version 69`

Ese error suele ocurrir cuando Gradle se ejecuta con Java muy nuevo/incompatible.

Solución recomendada (Linux/macOS):

```bash
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
export PATH="$JAVA_HOME/bin:$PATH"
cd frontend/android
./gradlew --no-daemon assembleDebug
```

En Windows PowerShell:

```powershell
$env:JAVA_HOME="C:\Program Files\Java\jdk-17"
$env:Path="$env:JAVA_HOME\bin;$env:Path"
cd frontend\android
.\gradlew.bat assembleDebug
```

### Método E — Disparo por terminal (Bash/PowerShell)

Usa `npm run apk:trigger:github` para lanzar la compilación remota (ver sección siguiente con ejemplos completos).

### ¿Se puede dejar la APK directamente dentro del repositorio?

Sí, técnicamente se puede, **pero no es recomendable** para un repo de código:

- Aumenta mucho el peso y el historial de Git.
- Cada build binaria ensucia los diffs.
- Es mejor publicar APK en **Releases** o en **Artifacts**.

Si igualmente quieres versionar binarios en Git, usa **Git LFS** y una carpeta dedicada (`artifacts/`), pero para este proyecto se recomienda Releases.

## Disparar la compilación APK en GitHub desde terminal

Si quieres ejecutarlo ya en GitHub (sin entrar al panel web), usa este script:

```bash
cd frontend
export GITHUB_TOKEN="<token_con_repo_y_actions>"
export GITHUB_REPO="tu-usuario/tu-repo"   # opcional si tienes remote origin
export GITHUB_REF="codex/paraweather"        # opcional; por defecto usa tu rama actual
npm run apk:trigger:github
```

En **Windows PowerShell** usa esto (cada línea por separado):

```powershell
cd frontend
$env:GITHUB_TOKEN="<token_con_repo_y_actions>"
$env:GITHUB_REPO="tu-usuario/tu-repo"   # opcional si tienes remote origin
$env:GITHUB_REF="codex/paraweather"        # opcional; por defecto usa tu rama actual
npm run apk:trigger:github
```

También puedes lanzarlo en una sola línea en PowerShell:

```powershell
cd frontend; $env:GITHUB_TOKEN="<token>"; $env:GITHUB_REPO="Capybla/Paraweather"; $env:GITHUB_REF="codex/paraweather"; npm run apk:trigger:github
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
$env:GITHUB_REF="codex/paraweather"
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
- Si no defines `GITHUB_REF`, usa automáticamente la rama actual (por ejemplo `codex/paraweather`).
- Si no defines `GITHUB_REPO`, intenta leerlo desde `git remote origin`.


Alternativa sin npm (disparo directo de API desde PowerShell):

```powershell
$headers = @{
  Authorization = "Bearer $env:GITHUB_TOKEN"
  Accept = "application/vnd.github+json"
  "X-GitHub-Api-Version" = "2022-11-28"
}
$body = @{ ref = "codex/paraweather" } | ConvertTo-Json
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
