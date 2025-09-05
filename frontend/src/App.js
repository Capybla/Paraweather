import React, { useState, useEffect, useRef } from 'react';
import { MapContainer, TileLayer, Polygon, Marker, Popup, Polyline, useMapEvents } from 'react-leaflet';
import L from 'leaflet';
import axios from 'axios';
import './App.css';

// Fix for default markers in react-leaflet
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

// Custom icons for different airspace types
const getAirspaceColor = (type) => {
  const colors = {
    'R': '#ff4444', // Restricted - Red
    'P': '#ff0000', // Prohibited - Dark Red
    'D': '#ff8800', // Danger - Orange
    'CTR': '#4444ff', // Control Zone - Blue
    'TMA': '#6666ff', // Terminal Control Area - Light Blue
    'A': '#8844ff', // Class A - Purple
    'B': '#aa44ff', // Class B - Light Purple
    'C': '#44ff44', // Class C - Green
    'E': '#88ff88'  // Class E - Light Green
  };
  return colors[type] || '#888888';
};

// Get route segment color based on segment type
const getSegmentColor = (segmentType) => {
  const colors = {
    'safe': '#00ff00',      // Green - Safe
    'caution': '#ffaa00',   // Orange - Caution
    'avoid': '#ff6600',     // Red-Orange - Avoid
    'restricted': '#ff0000' // Red - Restricted
  };
  return colors[segmentType] || '#888888';
};

// Country filter component
const CountryFilter = ({ countries, selectedCountries, onCountryToggle }) => {
  return (
    <div className="country-filters">
      <h4>Countries</h4>
      <div className="country-list">
        {countries.map(country => (
          <label key={country.code} className="country-checkbox">
            <input
              type="checkbox"
              checked={selectedCountries.includes(country.code)}
              onChange={() => onCountryToggle(country.code)}
            />
            <span className="country-flag">{country.flag}</span>
            {country.name}
          </label>
        ))}
      </div>
      <button 
        onClick={() => onCountryToggle('all')}
        className="select-all-countries-btn"
      >
        Show All Countries
      </button>
    </div>
  );
};

// Airspace preferences component
const AirspacePreferences = ({ preferences, onPreferencesChange, isOpen, onToggle }) => {
  const handleChange = (field, value) => {
    onPreferencesChange({
      ...preferences,
      [field]: value
    });
  };

  if (!isOpen) {
    return (
      <div className="preferences-collapsed">
        <button onClick={onToggle} className="preferences-toggle">
          ⚙️ Flight Settings
        </button>
      </div>
    );
  }

  return (
    <div className="preferences-panel">
      <div className="preferences-header">
        <h3>⚙️ Flight Preferences</h3>
        <button onClick={onToggle} className="preferences-close">×</button>
      </div>
      
      <div className="preferences-content">
        <div className="preference-section">
          <h4>Airspace Avoidance</h4>
          <label>
            <input
              type="checkbox"
              checked={preferences.avoid_ctr}
              onChange={(e) => handleChange('avoid_ctr', e.target.checked)}
            />
            Avoid CTR (Control Zones)
          </label>
          <label>
            <input
              type="checkbox"
              checked={preferences.avoid_tma}
              onChange={(e) => handleChange('avoid_tma', e.target.checked)}
            />
            Avoid TMA (Terminal Areas)
          </label>
          <label>
            <input
              type="checkbox"
              checked={preferences.avoid_danger}
              onChange={(e) => handleChange('avoid_danger', e.target.checked)}
            />
            Avoid Danger Areas
          </label>
        </div>

        <div className="preference-section">
          <h4>Altitude Settings (meters)</h4>
          <label>
            Minimum Altitude:
            <input
              type="number"
              value={preferences.minimum_altitude}
              onChange={(e) => handleChange('minimum_altitude', parseFloat(e.target.value))}
              min="50"
              max="500"
              step="10"
            />
          </label>
          <label>
            Preferred Altitude:
            <input
              type="number"
              value={preferences.preferred_altitude}
              onChange={(e) => handleChange('preferred_altitude', parseFloat(e.target.value))}
              min="100"
              max="1500"
              step="50"
            />
          </label>
          <label>
            Maximum Altitude:
            <input
              type="number"
              value={preferences.maximum_altitude}
              onChange={(e) => handleChange('maximum_altitude', parseFloat(e.target.value))}
              min="500"
              max="3000"
              step="50"
            />
          </label>
        </div>
      </div>
    </div>
  );
};

// Route segments visualization component
const RouteSegments = ({ route }) => {
  if (!route || !route.route_segments || route.route_segments.length === 0) {
    return null;
  }

  return (
    <>
      {route.route_segments.map((segment, index) => {
        const positions = [
          [segment.start.lat, segment.start.lng],
          [segment.end.lat, segment.end.lng]
        ];

        return (
          <Polyline
            key={index}
            positions={positions}
            pathOptions={{
              color: getSegmentColor(segment.segment_type),
              weight: 4,
              opacity: 0.8
            }}
          >
            <Popup>
              <div className="segment-popup">
                <h4>Route Segment {index + 1}</h4>
                <p><strong>Distance:</strong> {segment.distance.toFixed(1)} km</p>
                <p><strong>Altitude:</strong> {segment.altitude}m AGL</p>
                <p><strong>Status:</strong> {segment.segment_type.toUpperCase()}</p>
                {segment.airspace_warnings.length > 0 && (
                  <div className="segment-warnings">
                    <strong>Warnings:</strong>
                    {segment.airspace_warnings.slice(0, 5).map((warning, idx) => (
                      <p key={idx} className="warning-text">{warning}</p>
                    ))}
                    {segment.airspace_warnings.length > 5 && (
                      <p className="warning-text">... and {segment.airspace_warnings.length - 5} more warnings</p>
                    )}
                  </div>
                )}
              </div>
            </Popup>
          </Polyline>
        );
      })}
      
      {/* Altitude annotations */}
      {route.route_segments.map((segment, index) => {
        // Calculate midpoint for altitude annotation
        const midLat = (segment.start.lat + segment.end.lat) / 2;
        const midLng = (segment.start.lng + segment.end.lng) / 2;
        
        return (
          <Marker
            key={`alt-${index}`}
            position={[midLat, midLng]}
            icon={L.divIcon({
              className: 'altitude-marker',
              html: `<div class="altitude-annotation">${segment.altitude}m</div>`,
              iconSize: [60, 20],
              iconAnchor: [30, 10]
            })}
          />
        );
      })}
    </>
  );
};

// Route planning component with enhanced preferences
const RoutePlanner = ({ onRouteCreate, isPlanning, setIsPlanning, airspacePreferences, onPreferencesChange }) => {
  const [startPoint, setStartPoint] = useState(null);
  const [endPoint, setEndPoint] = useState(null);
  const [routeName, setRouteName] = useState('');
  const [routeDescription, setRouteDescription] = useState('');
  const [loading, setLoading] = useState(false);

  const MapClickHandler = () => {
    useMapEvents({
      click(e) {
        if (!isPlanning) return;
        
        const point = {
          lat: e.latlng.lat,
          lng: e.latlng.lng,
          altitude: airspacePreferences.preferred_altitude
        };

        if (!startPoint) {
          setStartPoint(point);
        } else if (!endPoint) {
          setEndPoint(point);
        }
      }
    });
    return null;
  };

  const createRoute = async () => {
    if (!startPoint || !endPoint || !routeName) return;

    setLoading(true);
    try {
      const routeData = {
        name: routeName,
        description: routeDescription,
        start_point: startPoint,
        end_point: endPoint,
        waypoints: [],
        airspace_preferences: airspacePreferences
      };

      const response = await axios.post(`${API}/routes`, routeData);
      onRouteCreate(response.data);
      
      // Reset form
      setStartPoint(null);
      setEndPoint(null);
      setRouteName('');
      setRouteDescription('');
      setIsPlanning(false);
    } catch (error) {
      console.error('Error creating route:', error);
      alert('Failed to create route. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const cancelPlanning = () => {
    setStartPoint(null);
    setEndPoint(null);
    setIsPlanning(false);
  };

  return (
    <>
      <MapClickHandler />
      {startPoint && (
        <Marker position={[startPoint.lat, startPoint.lng]}>
          <Popup>
            <div>
              <strong>Start Point</strong>
              <p>Altitude: {startPoint.altitude}m AGL</p>
            </div>
          </Popup>
        </Marker>
      )}
      {endPoint && (
        <Marker position={[endPoint.lat, endPoint.lng]}>
          <Popup>
            <div>
              <strong>End Point</strong>
              <p>Altitude: {endPoint.altitude}m AGL</p>
            </div>
          </Popup>
        </Marker>
      )}
      
      {isPlanning && (
        <div className="route-planner-panel">
          <h3>Plan New Route</h3>
          <div className="form-group">
            <label>Route Name:</label>
            <input
              type="text"
              value={routeName}
              onChange={(e) => setRouteName(e.target.value)}
              placeholder="Enter route name"
            />
          </div>
          <div className="form-group">
            <label>Description:</label>
            <textarea
              value={routeDescription}
              onChange={(e) => setRouteDescription(e.target.value)}
              placeholder="Optional description"
            />
          </div>
          
          <div className="planning-status">
            {!startPoint && <p>📍 Click on map to set start point</p>}
            {startPoint && !endPoint && <p>🎯 Click on map to set end point</p>}
            {startPoint && endPoint && (
              <div>
                <p>✅ Ready to create route!</p>
                <p className="altitude-info">Using altitude: {airspacePreferences.preferred_altitude}m</p>
              </div>
            )}
          </div>
          
          <div className="button-group">
            <button 
              onClick={createRoute} 
              disabled={!startPoint || !endPoint || !routeName || loading}
              className="btn-primary"
            >
              {loading ? 'Creating Route...' : 'Create Route'}
            </button>
            <button onClick={cancelPlanning} className="btn-secondary">
              Cancel
            </button>
          </div>
        </div>
      )}
    </>
  );
};

// Enhanced route display component
const RouteDisplay = ({ routes, selectedRoute, onRouteSelect }) => {
  const shareRoute = async (routeId) => {
    try {
      const response = await axios.post(`${API}/routes/${routeId}/share`);
      const shareUrl = `${window.location.origin}/shared/${response.data.share_token}`;
      navigator.clipboard.writeText(shareUrl);
      alert('Share link copied to clipboard!');
    } catch (error) {
      console.error('Error sharing route:', error);
      alert('Failed to create share link.');
    }
  };

  return (
    <div className="route-list">
      <h3>Flight Routes</h3>
      {routes.map(route => (
        <div 
          key={route.id} 
          className={`route-item ${selectedRoute?.id === route.id ? 'selected' : ''}`}
          onClick={() => onRouteSelect(route)}
        >
          <h4>{route.name}</h4>
          <div className="route-stats">
            <p>📏 Distance: {route.total_distance?.toFixed(1)} km</p>
            <p>⏱️ Est. Time: {route.estimated_time?.toFixed(1)} hours</p>
            <p>✈️ Segments: {route.route_segments?.length || 0}</p>
          </div>
          
          {/* Flight warnings */}
          {route.flight_warnings && route.flight_warnings.length > 0 && (
            <div className="flight-warnings">
              {route.flight_warnings.map((warning, idx) => (
                <p key={idx} className="flight-warning">{warning}</p>
              ))}
            </div>
          )}
          
          {/* Airspace warnings */}
          {route.airspace_warnings && route.airspace_warnings.length > 0 && (
            <div className="warnings">
              <strong>⚠️ Airspace Warnings:</strong>
              {route.airspace_warnings.slice(0, 3).map((warning, idx) => (
                <p key={idx} className="warning">{warning}</p>
              ))}
              {route.airspace_warnings.length > 3 && (
                <p className="warning">... and {route.airspace_warnings.length - 3} more</p>
              )}
            </div>
          )}
          
          {/* Route segment summary */}
          {route.route_segments && route.route_segments.length > 0 && (
            <div className="segment-summary">
              <strong>Route Status:</strong>
              <div className="segment-indicators">
                {route.route_segments.map((segment, idx) => (
                  <span 
                    key={idx}
                    className="segment-indicator"
                    style={{ backgroundColor: getSegmentColor(segment.segment_type) }}
                    title={`Segment ${idx + 1}: ${segment.segment_type} (${segment.altitude}m)`}
                  ></span>
                ))}
              </div>
            </div>
          )}
          
          <button 
            onClick={(e) => {
              e.stopPropagation();
              shareRoute(route.id);
            }}
            className="share-btn"
          >
            📤 Share Route
          </button>
        </div>
      ))}
    </div>
  );
};

// Main App Component
const App = () => {
  const [airspaces, setAirspaces] = useState([]);
  const [airspaceTypes, setAirspaceTypes] = useState([]);
  const [countries, setCountries] = useState([]);
  const [selectedAirspaceTypes, setSelectedAirspaceTypes] = useState([]);
  const [selectedCountries, setSelectedCountries] = useState(['ES', 'FR', 'DE', 'IT']); // Default to major European countries
  const [routes, setRoutes] = useState([]);
  const [selectedRoute, setSelectedRoute] = useState(null);
  const [isPlanning, setIsPlanning] = useState(false);
  const [loading, setLoading] = useState(true);
  const [mapCenter] = useState([46.2276, 2.2137]); // Center of Europe
  const [showSidebar, setShowSidebar] = useState(true);
  const [preferencesOpen, setPreferencesOpen] = useState(false);
  const [airspacePreferences, setAirspacePreferences] = useState({
    avoid_ctr: true,
    avoid_tma: true,
    avoid_restricted: true,
    avoid_prohibited: true,
    avoid_danger: true,
    minimum_altitude: 150,
    maximum_altitude: 1500,
    preferred_altitude: 300
  });

  useEffect(() => {
    loadInitialData();
  }, []);

  useEffect(() => {
    // Reload airspaces when country selection changes
    loadAirspaces();
  }, [selectedCountries]);

  const loadInitialData = async () => {
    try {
      setLoading(true);
      
      // Load countries
      const countriesResponse = await axios.get(`${API}/countries`);
      setCountries(countriesResponse.data.countries);
      
      // Load airspace types
      const typesResponse = await axios.get(`${API}/airspace-types`);
      setAirspaceTypes(typesResponse.data.types);
      
      // Load initial airspaces
      await loadAirspaces();
      
      // Load routes
      const routesResponse = await axios.get(`${API}/routes`);
      setRoutes(routesResponse.data);
      
    } catch (error) {
      console.error('Error loading data:', error);
    } finally {
      setLoading(false);
    }
  };

  const loadAirspaces = async () => {
    try {
      let airspacesData = [];
      
      // Load airspaces for each selected country
      for (const countryCode of selectedCountries) {
        const response = await axios.get(`${API}/airspaces?country=${countryCode}`);
        airspacesData = [...airspacesData, ...response.data];
      }
      
      setAirspaces(airspacesData);
    } catch (error) {
      console.error('Error loading airspaces:', error);
    }
  };

  const handleRouteCreate = (newRoute) => {
    setRoutes([...routes, newRoute]);
    setSelectedRoute(newRoute);
  };

  const toggleAirspaceType = (typeCode) => {
    setSelectedAirspaceTypes(prev => 
      prev.includes(typeCode) 
        ? prev.filter(t => t !== typeCode)
        : [...prev, typeCode]
    );
  };

  const handleCountryToggle = (countryCode) => {
    if (countryCode === 'all') {
      // Toggle all countries
      if (selectedCountries.length === countries.length) {
        setSelectedCountries([]);
      } else {
        setSelectedCountries(countries.map(c => c.code));
      }
    } else {
      setSelectedCountries(prev => 
        prev.includes(countryCode) 
          ? prev.filter(c => c !== countryCode)
          : [...prev, countryCode]
      );
    }
  };

  const filteredAirspaces = airspaces.filter(airspace => 
    selectedAirspaceTypes.length === 0 || selectedAirspaceTypes.includes(airspace.type)
  );

  if (loading) {
    return (
      <div className="loading-screen">
        <div className="spinner"></div>
        <p>Loading European Paramotorist Flight Planner...</p>
        <p className="loading-details">Preparing airspace data for Spain, France, Germany, Italy...</p>
      </div>
    );
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>🪂 European Paramotorist Flight Planner</h1>
        <div className="header-stats">
          <span className="stat-item">🌍 {countries.length} Countries</span>
          <span className="stat-item">✈️ {airspaces.length} Airspaces</span>
          <span className="stat-item">🛤️ {routes.length} Routes</span>
        </div>
        <div className="header-controls">
          <button 
            onClick={() => setShowSidebar(!showSidebar)}
            className="sidebar-toggle"
          >
            {showSidebar ? '◀' : '▶'}
          </button>
          <button 
            onClick={() => setIsPlanning(!isPlanning)}
            className={`plan-route-btn ${isPlanning ? 'active' : ''}`}
          >
            {isPlanning ? 'Cancel Planning' : 'Plan New Route'}
          </button>
        </div>
      </header>

      <div className="app-content">
        {showSidebar && (
          <div className="sidebar">
            <AirspacePreferences
              preferences={airspacePreferences}
              onPreferencesChange={setAirspacePreferences}
              isOpen={preferencesOpen}
              onToggle={() => setPreferencesOpen(!preferencesOpen)}
            />
            
            <CountryFilter
              countries={countries}
              selectedCountries={selectedCountries}
              onCountryToggle={handleCountryToggle}
            />
            
            <div className="airspace-controls">
              <h3>Airspace Filters</h3>
              <div className="airspace-types">
                {airspaceTypes.map(type => (
                  <label key={type.code} className="airspace-type-checkbox">
                    <input
                      type="checkbox"
                      checked={selectedAirspaceTypes.includes(type.code)}
                      onChange={() => toggleAirspaceType(type.code)}
                    />
                    <span 
                      className="airspace-color-indicator"
                      style={{ backgroundColor: getAirspaceColor(type.code) }}
                    ></span>
                    {type.code} - {type.name}
                    {!type.avoidable && <span className="unavoidable">⚠️</span>}
                  </label>
                ))}
              </div>
              <button 
                onClick={() => setSelectedAirspaceTypes([])}
                className="clear-filters-btn"
              >
                Clear All Filters
              </button>
            </div>

            <RouteDisplay 
              routes={routes}
              selectedRoute={selectedRoute}
              onRouteSelect={setSelectedRoute}
            />
          </div>
        )}

        <div className="map-container">
          <MapContainer 
            center={mapCenter} 
            zoom={5} 
            className="leaflet-map"
          >
            <TileLayer
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
              attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            />
            
            {/* Render airspaces */}
            {filteredAirspaces.map(airspace => (
              <Polygon
                key={airspace.id}
                positions={airspace.coordinates.map(coord => [coord.lat, coord.lng])}
                pathOptions={{
                  color: getAirspaceColor(airspace.type),
                  fillColor: getAirspaceColor(airspace.type),
                  fillOpacity: 0.2,
                  weight: 2
                }}
              >
                <Popup>
                  <div className="airspace-popup">
                    <h4>{airspace.name}</h4>
                    <p><strong>Country:</strong> {airspace.country} {countries.find(c => c.code === airspace.country)?.flag}</p>
                    <p><strong>Type:</strong> {airspace.type}</p>
                    {airspace.floor && <p><strong>Floor:</strong> {airspace.floor}</p>}
                    {airspace.ceiling && <p><strong>Ceiling:</strong> {airspace.ceiling}</p>}
                    {airspace.frequency && <p><strong>Frequency:</strong> {airspace.frequency}</p>}
                    {airspace.requirements && airspace.requirements.length > 0 && (
                      <div className="requirements">
                        <strong>Requirements:</strong>
                        {airspace.requirements.map((req, idx) => (
                          <p key={idx} className="requirement">{req}</p>
                        ))}
                      </div>
                    )}
                  </div>
                </Popup>
              </Polygon>
            ))}

            {/* Render selected route with segments */}
            {selectedRoute && <RouteSegments route={selectedRoute} />}

            {/* Render route waypoints */}
            {selectedRoute && selectedRoute.waypoints && selectedRoute.waypoints.map((waypoint, index) => (
              <Marker key={`waypoint-${index}`} position={[waypoint.lat, waypoint.lng]}>
                <Popup>
                  <div>
                    <strong>{waypoint.waypoint_name || `Waypoint ${index + 1}`}</strong>
                    {waypoint.altitude && <p>Altitude: {waypoint.altitude}m AGL</p>}
                  </div>
                </Popup>
              </Marker>
            ))}

            <RoutePlanner 
              onRouteCreate={handleRouteCreate}
              isPlanning={isPlanning}
              setIsPlanning={setIsPlanning}
              airspacePreferences={airspacePreferences}
              onPreferencesChange={setAirspacePreferences}
            />
          </MapContainer>
        </div>
      </div>
    </div>
  );
};

export default App;