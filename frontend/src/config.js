export const UI_DEFAULT_PAGES = ['Principal'];

export const UI_DEFAULT_WIDGETS = {
  map: { visible: true, page: 'Principal', x: 16, y: 120, width: 900, height: 580 },
  weather: { visible: true, page: 'Principal', x: 940, y: 120, width: 360, height: 320 },
  altimeter: { visible: true, page: 'Principal', x: 940, y: 460, width: 360, height: 240 },
};

export const STORAGE_KEYS = {
  lastWeatherSnapshot: 'lastWeatherSnapshot',
};

export const WEATHER_CACHE_TTL_MS = 1000 * 60 * 90;
