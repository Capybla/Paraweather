import React, { useEffect, useMemo, useRef, useState } from 'react';
import mapboxgl from 'mapbox-gl';
import 'mapbox-gl/dist/mapbox-gl.css';

const DEFAULT_CENTER = { lng: -3.7038, lat: 40.4168 };

const normalizeCoordinates = (coords = []) => {
  if (!Array.isArray(coords)) return [];
  const normalized = coords
    .map((coord) => {
      const lat = Number(coord?.lat);
      const lng = Number(coord?.lng);
      if (!Number.isFinite(lat) || !Number.isFinite(lng)) return null;
      if (Math.abs(lat) <= 90 && Math.abs(lng) <= 180) return [lng, lat];
      if (Math.abs(lng) <= 90 && Math.abs(lat) <= 180) return [lat, lng];
      return null;
    })
    .filter(Boolean);

  if (normalized.length > 2) {
    const [firstLng, firstLat] = normalized[0];
    const [lastLng, lastLat] = normalized[normalized.length - 1];
    if (firstLng !== lastLng || firstLat !== lastLat) {
      normalized.push([firstLng, firstLat]);
    }
  }

  return normalized;
};

const airspacesToGeoJSON = (airspaces = []) => ({
  type: 'FeatureCollection',
  features: airspaces
    .map((a, index) => {
      const ring = normalizeCoordinates(a.coordinates || []);
      if (ring.length < 4) return null;
      return {
        type: 'Feature',
        id: a.id || `${a.type || 'U'}-${a.name || 'airspace'}-${index}`,
        geometry: { type: 'Polygon', coordinates: [ring] },
        properties: {
          name: a.name || 'Airspace',
          type: a.type || 'Unknown',
          class: a.type || 'Unknown',
          floor: a.floor || 'N/A',
          ceiling: a.ceiling || 'N/A',
          country: a.country || 'N/A'
        }
      };
    })
    .filter(Boolean)
});

const emptyFC = { type: 'FeatureCollection', features: [] };

const styleOptions = {
  outdoors: 'mapbox://styles/mapbox/outdoors-v12',
  satellite: 'mapbox://styles/mapbox/satellite-streets-v12'
};

const MapboxFlightMap = ({
  mapboxToken,
  defaultCenter = DEFAULT_CENTER,
  currentPosition,
  airspaces = [],
  airspaceGeoJsonUrl,
  notamGeoJsonUrl,
  routeColor = '#1976d2',
  routeWidth = 4
}) => {
  const containerRef = useRef(null);
  const mapRef = useRef(null);
  const popupRef = useRef(null);
  const userMarkerRef = useRef(null);

  const [mapStyleKey, setMapStyleKey] = useState('outdoors');
  const [drawMode, setDrawMode] = useState(false);
  const [showAirspaces, setShowAirspaces] = useState(true);
  const [showNotams, setShowNotams] = useState(true);
  const [waypoints, setWaypoints] = useState([]);

  const airspacesGeoJson = useMemo(() => airspacesToGeoJSON(airspaces), [airspaces]);

  const addOrUpdateSource = (map, sourceId, data) => {
    const src = map.getSource(sourceId);
    if (src) {
      src.setData(data);
    } else {
      map.addSource(sourceId, { type: 'geojson', data });
    }
  };

  const ensureLayers = (map) => {
    addOrUpdateSource(map, 'airspaces-src', airspacesGeoJson);
    addOrUpdateSource(map, 'notams-src', emptyFC);
    addOrUpdateSource(map, 'route-src', {
      type: 'FeatureCollection',
      features: waypoints.length >= 2
        ? [{ type: 'Feature', geometry: { type: 'LineString', coordinates: waypoints }, properties: {} }]
        : []
    });
    addOrUpdateSource(map, 'waypoints-src', {
      type: 'FeatureCollection',
      features: waypoints.map((coords, i) => ({
        type: 'Feature',
        geometry: { type: 'Point', coordinates: coords },
        properties: { index: i + 1 }
      }))
    });

    if (!map.getLayer('airspaces-fill')) {
      map.addLayer({
        id: 'airspaces-fill',
        type: 'fill',
        source: 'airspaces-src',
        paint: {
          'fill-color': [
            'match', ['get', 'type'],
            'CTR', '#1e88e5',
            'TMA', '#5e35b1',
            'R', '#e53935',
            'P', '#b71c1c',
            'G', '#43a047',
            '#757575'
          ],
          'fill-opacity': 0.28
        }
      });

      map.addLayer({
        id: 'airspaces-outline',
        type: 'line',
        source: 'airspaces-src',
        paint: {
          'line-color': [
            'match', ['get', 'type'],
            'CTR', '#0d47a1',
            'TMA', '#311b92',
            'R', '#c62828',
            'P', '#7f0000',
            'G', '#1b5e20',
            '#424242'
          ],
          'line-width': 2
        }
      });
    }

    if (!map.getLayer('notams-fill')) {
      map.addLayer({
        id: 'notams-fill',
        type: 'fill',
        source: 'notams-src',
        filter: ['==', ['geometry-type'], 'Polygon'],
        paint: {
          'fill-color': '#ff9800',
          'fill-opacity': 0.26
        }
      });
      map.addLayer({
        id: 'notams-outline',
        type: 'line',
        source: 'notams-src',
        filter: ['==', ['geometry-type'], 'Polygon'],
        paint: {
          'line-color': '#ef6c00',
          'line-width': 2
        }
      });
      map.addLayer({
        id: 'notams-point',
        type: 'circle',
        source: 'notams-src',
        filter: ['==', ['geometry-type'], 'Point'],
        paint: {
          'circle-color': '#fb8c00',
          'circle-radius': 6,
          'circle-stroke-color': '#ffffff',
          'circle-stroke-width': 1.5
        }
      });
    }

    if (!map.getLayer('route-line')) {
      map.addLayer({
        id: 'route-line',
        type: 'line',
        source: 'route-src',
        paint: {
          'line-color': routeColor,
          'line-width': routeWidth
        }
      });
      map.addLayer({
        id: 'route-waypoints',
        type: 'circle',
        source: 'waypoints-src',
        paint: {
          'circle-color': '#1565c0',
          'circle-radius': 5,
          'circle-stroke-color': '#fff',
          'circle-stroke-width': 1.5
        }
      });
    }
  };

  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;
    mapboxgl.accessToken = mapboxToken || process.env.REACT_APP_MAPBOX_TOKEN || '';

    const start = currentPosition
      ? [currentPosition.lng, currentPosition.lat]
      : [defaultCenter.lng, defaultCenter.lat];

    const map = new mapboxgl.Map({
      container: containerRef.current,
      style: styleOptions[mapStyleKey],
      center: start,
      zoom: 8
    });

    map.addControl(new mapboxgl.NavigationControl(), 'top-right');
    map.addControl(new mapboxgl.GeolocateControl({
      positionOptions: { enableHighAccuracy: true },
      trackUserLocation: true,
      showUserHeading: true
    }), 'top-right');

    map.on('load', () => ensureLayers(map));
    map.on('style.load', () => ensureLayers(map));

    map.on('click', 'airspaces-fill', (e) => {
      const feature = e.features?.[0];
      if (!feature) return;
      const p = feature.properties || {};
      popupRef.current?.remove();
      popupRef.current = new mapboxgl.Popup({ closeButton: true })
        .setLngLat(e.lngLat)
        .setHTML(`<strong>${p.name || 'Airspace'}</strong><br/>Clase: ${p.class || p.type || '-'}<br/>Floor: ${p.floor || '-'}<br/>Ceiling: ${p.ceiling || '-'}`)
        .addTo(map);
    });

    const showNotamPopup = (e) => {
      const feature = e.features?.[0];
      if (!feature) return;
      const p = feature.properties || {};
      popupRef.current?.remove();
      popupRef.current = new mapboxgl.Popup({ closeButton: true })
        .setLngLat(e.lngLat)
        .setHTML(`<strong>NOTAM</strong><br/>${p.title || p.id || 'Activo'}<br/>${p.text || p.description || ''}<br/><small>Vigencia: ${p.valid_from || '-'} → ${p.valid_to || '-'}</small>`)
        .addTo(map);
    };

    map.on('click', 'notams-fill', showNotamPopup);
    map.on('click', 'notams-point', showNotamPopup);

    map.on('click', (e) => {
      if (!drawMode) return;
      setWaypoints((prev) => [...prev, [e.lngLat.lng, e.lngLat.lat]]);
    });

    mapRef.current = map;

    return () => {
      popupRef.current?.remove();
      map.remove();
      mapRef.current = null;
    };
  }, [mapboxToken, defaultCenter, currentPosition, drawMode, mapStyleKey]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    ensureLayers(map);
  }, [airspacesGeoJson, waypoints, routeColor, routeWidth]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    const visibility = showAirspaces ? 'visible' : 'none';
    ['airspaces-fill', 'airspaces-outline'].forEach((layerId) => {
      if (map.getLayer(layerId)) map.setLayoutProperty(layerId, 'visibility', visibility);
    });
  }, [showAirspaces]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    const visibility = showNotams ? 'visible' : 'none';
    ['notams-fill', 'notams-outline', 'notams-point'].forEach((layerId) => {
      if (map.getLayer(layerId)) map.setLayoutProperty(layerId, 'visibility', visibility);
    });
  }, [showNotams]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !airspaceGeoJsonUrl) return;
    fetch(airspaceGeoJsonUrl)
      .then((r) => r.json())
      .then((geojson) => {
        addOrUpdateSource(map, 'airspaces-src', geojson);
      })
      .catch(() => {});
  }, [airspaceGeoJsonUrl]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !notamGeoJsonUrl) return;
    fetch(notamGeoJsonUrl)
      .then((r) => r.json())
      .then((geojson) => {
        addOrUpdateSource(map, 'notams-src', geojson);
      })
      .catch(() => {
        addOrUpdateSource(map, 'notams-src', emptyFC);
      });
  }, [notamGeoJsonUrl]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !currentPosition) return;

    const lngLat = [currentPosition.lng, currentPosition.lat];
    if (!userMarkerRef.current) {
      userMarkerRef.current = new mapboxgl.Marker({ color: '#0066ff' }).setLngLat(lngLat).addTo(map);
    } else {
      userMarkerRef.current.setLngLat(lngLat);
    }
  }, [currentPosition]);

  return (
    <div className="mapbox-wrapper">
      <div className="mapbox-toolbar">
        <button onClick={() => setMapStyleKey((prev) => (prev === 'outdoors' ? 'satellite' : 'outdoors'))}>
          Estilo: {mapStyleKey === 'outdoors' ? 'Outdoors' : 'Satellite'}
        </button>
        <button onClick={() => setShowAirspaces((v) => !v)}>{showAirspaces ? 'Ocultar' : 'Mostrar'} espacios aéreos</button>
        <button onClick={() => setShowNotams((v) => !v)}>{showNotams ? 'Ocultar' : 'Mostrar'} NOTAMs</button>
        <button onClick={() => setDrawMode((v) => !v)}>{drawMode ? 'Finalizar ruta' : 'Dibujar ruta'}</button>
        <button onClick={() => setWaypoints([])}>Limpiar ruta</button>
      </div>
      <div className="mapbox-waypoints">Waypoints: {waypoints.length}</div>
      <div ref={containerRef} className="mapbox-map" />
    </div>
  );
};

export default MapboxFlightMap;
