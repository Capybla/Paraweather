import React, { useState, useEffect, useRef } from 'react';
import { MapContainer, TileLayer, Polygon, Marker, Popup, useMapEvents } from 'react-leaflet';
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

// Route planning component
const RoutePlanner = ({ onRouteCreate, isPlanning, setIsPlanning }) => {
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
          lng: e.latlng.lng
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
        wind_consideration: true
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
          <Popup>Start Point</Popup>
        </Marker>
      )}
      {endPoint && (
        <Marker position={[endPoint.lat, endPoint.lng]}>
          <Popup>End Point</Popup>
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
            {!startPoint && <p>Click on the map to set start point</p>}
            {startPoint && !endPoint && <p>Click on the map to set end point</p>}
            {startPoint && endPoint && <p>Ready to create route!</p>}
          </div>
          <div className="button-group">
            <button 
              onClick={createRoute} 
              disabled={!startPoint || !endPoint || !routeName || loading}
              className="btn-primary"
            >
              {loading ? 'Creating...' : 'Create Route'}
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

// Route display component
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
          <p>Distance: {route.total_distance?.toFixed(1)} km</p>
          <p>Est. Time: {route.estimated_time?.toFixed(1)} hours</p>
          {route.airspace_warnings.length > 0 && (
            <div className="warnings">
              <strong>⚠️ Airspace Warnings:</strong>
              {route.airspace_warnings.map((warning, idx) => (
                <p key={idx} className="warning">{warning}</p>
              ))}
            </div>
          )}
          <button 
            onClick={(e) => {
              e.stopPropagation();
              shareRoute(route.id);
            }}
            className="share-btn"
          >
            Share Route
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
  const [selectedAirspaceTypes, setSelectedAirspaceTypes] = useState([]);
  const [routes, setRoutes] = useState([]);
  const [selectedRoute, setSelectedRoute] = useState(null);
  const [isPlanning, setIsPlanning] = useState(false);
  const [loading, setLoading] = useState(true);
  const [mapCenter] = useState([52.5, -1.5]); // UK center
  const [showSidebar, setShowSidebar] = useState(true);

  useEffect(() => {
    loadInitialData();
  }, []);

  const loadInitialData = async () => {
    try {
      setLoading(true);
      
      // Load airspace types
      const typesResponse = await axios.get(`${API}/airspace-types`);
      setAirspaceTypes(typesResponse.data.types);
      
      // Load all airspaces
      const airspacesResponse = await axios.get(`${API}/airspaces`);
      setAirspaces(airspacesResponse.data);
      
      // Load routes
      const routesResponse = await axios.get(`${API}/routes`);
      setRoutes(routesResponse.data);
      
    } catch (error) {
      console.error('Error loading data:', error);
    } finally {
      setLoading(false);
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

  const filteredAirspaces = airspaces.filter(airspace => 
    selectedAirspaceTypes.length === 0 || selectedAirspaceTypes.includes(airspace.type)
  );

  if (loading) {
    return (
      <div className="loading-screen">
        <div className="spinner"></div>
        <p>Loading Paramotorist Flight Planner...</p>
      </div>
    );
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>🪂 Paramotorist Flight Planner</h1>
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
            zoom={7} 
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
                  <div>
                    <h4>{airspace.name}</h4>
                    <p><strong>Type:</strong> {airspace.type}</p>
                    {airspace.floor && <p><strong>Floor:</strong> {airspace.floor}</p>}
                    {airspace.ceiling && <p><strong>Ceiling:</strong> {airspace.ceiling}</p>}
                    {airspace.frequency && <p><strong>Frequency:</strong> {airspace.frequency}</p>}
                  </div>
                </Popup>
              </Polygon>
            ))}

            {/* Render selected route */}
            {selectedRoute && (
              <>
                <Marker position={[selectedRoute.start_point.lat, selectedRoute.start_point.lng]}>
                  <Popup>
                    <strong>Start:</strong> {selectedRoute.name}
                  </Popup>
                </Marker>
                <Marker position={[selectedRoute.end_point.lat, selectedRoute.end_point.lng]}>
                  <Popup>
                    <strong>End:</strong> {selectedRoute.name}
                  </Popup>
                </Marker>
              </>
            )}

            <RoutePlanner 
              onRouteCreate={handleRouteCreate}
              isPlanning={isPlanning}
              setIsPlanning={setIsPlanning}
            />
          </MapContainer>
        </div>
      </div>
    </div>
  );
};

export default App;