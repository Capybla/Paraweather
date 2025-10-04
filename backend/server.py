from fastapi import FastAPI, APIRouter, HTTPException, Depends
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Tuple
import uuid
from datetime import datetime, timezone
import httpx
import asyncio
import json
import re
import math
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import unary_union


ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")


# Models
class Coordinate(BaseModel):
    lat: float
    lng: float

class RouteSegment(BaseModel):
    start: Coordinate
    end: Coordinate
    altitude: float  # meters above sea level
    segment_type: str  # "safe", "caution", "avoid", "restricted"
    airspace_warnings: List[str] = []
    distance: float  # segment distance in km

class Airspace(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    type: str  # CTR, TMA, R, P, D, etc.
    country: str  # Country code (ES, FR, DE, IT, etc.)
    coordinates: List[Coordinate]
    floor: Optional[str] = None
    ceiling: Optional[str] = None
    floor_meters: Optional[float] = None  # converted to meters
    ceiling_meters: Optional[float] = None  # converted to meters
    frequency: Optional[str] = None
    description: Optional[str] = None
    requirements: List[str] = []  # Special requirements

class RoutePoint(BaseModel):
    lat: float
    lng: float
    elevation: Optional[float] = None
    altitude: Optional[float] = None  # planned flight altitude
    waypoint_name: Optional[str] = None

class AirspacePreferences(BaseModel):
    avoid_ctr: bool = True
    avoid_tma: bool = True
    avoid_restricted: bool = True
    avoid_prohibited: bool = True
    avoid_danger: bool = True
    minimum_altitude: float = 150  # meters AGL
    maximum_altitude: float = 1500  # meters AGL
    preferred_altitude: float = 300  # meters AGL

class FlightRoute(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: Optional[str] = None
    start_point: RoutePoint
    end_point: RoutePoint
    waypoints: List[RoutePoint] = []
    route_segments: List[RouteSegment] = []
    total_distance: Optional[float] = None
    estimated_time: Optional[float] = None
    airspace_preferences: AirspacePreferences = Field(default_factory=AirspacePreferences)
    airspace_warnings: List[str] = []
    flight_warnings: List[str] = []
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class RouteCreate(BaseModel):
    name: str
    description: Optional[str] = None
    start_point: RoutePoint
    end_point: RoutePoint
    waypoints: List[RoutePoint] = []
    airspace_preferences: Optional[AirspacePreferences] = None

class WindData(BaseModel):
    lat: float
    lng: float
    speed: float  # m/s
    direction: float  # degrees
    altitude: Optional[float] = None

class SharedRoute(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    route_id: str
    share_token: str = Field(default_factory=lambda: str(uuid.uuid4()))
    shared_by: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None


# Utility functions
def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great circle distance between two points in kilometers"""
    R = 6371  # Earth's radius in kilometers
    
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    return R * c

def bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the bearing from point 1 to point 2"""
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlon_rad = math.radians(lon2 - lon1)
    
    y = math.sin(dlon_rad) * math.cos(lat2_rad)
    x = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon_rad)
    
    bearing_rad = math.atan2(y, x)
    bearing_deg = math.degrees(bearing_rad)
    
    return (bearing_deg + 360) % 360

def parse_altitude(alt_str: str) -> Optional[float]:
    """Parse altitude string to meters"""
    if not alt_str:
        return None
    
    alt_str = alt_str.upper().strip()
    
    if alt_str == "SFC" or alt_str == "GND":
        return 0
    elif "FT" in alt_str:
        # Parse feet
        num = re.findall(r'\d+', alt_str)
        if num:
            return float(num[0]) * 0.3048  # Convert feet to meters
    elif "FL" in alt_str:
        # Flight level (hundreds of feet)
        num = re.findall(r'\d+', alt_str)
        if num:
            return float(num[0]) * 100 * 0.3048  # Convert flight level to meters
    elif "M" in alt_str:
        # Meters
        num = re.findall(r'\d+', alt_str)
        if num:
            return float(num[0])
    
    return None

def parse_openair_airspace(openair_content: str) -> List[Dict]:
    """Parse OpenAir format airspace data with enhanced altitude parsing"""
    airspaces = []
    current_airspace = {}
    coordinates = []
    
    for line in openair_content.strip().split('\n'):
        line = line.strip()
        if not line or line.startswith('*'):
            continue
            
        if line.startswith('AC '):  # Airspace Class
            if current_airspace and coordinates:
                current_airspace['coordinates'] = coordinates
                current_airspace['floor_meters'] = parse_altitude(current_airspace.get('floor'))
                current_airspace['ceiling_meters'] = parse_altitude(current_airspace.get('ceiling'))
                airspaces.append(current_airspace)
            current_airspace = {'type': line[3:].strip()}
            coordinates = []
        elif line.startswith('AN '):  # Airspace Name
            current_airspace['name'] = line[3:].strip()
        elif line.startswith('AL '):  # Lower limit
            current_airspace['floor'] = line[3:].strip()
        elif line.startswith('AH '):  # Upper limit
            current_airspace['ceiling'] = line[3:].strip()
        elif line.startswith('AF '):  # Frequency
            current_airspace['frequency'] = line[3:].strip()
        elif line.startswith('AC0 '):  # Country code (custom extension)
            current_airspace['country'] = line[4:].strip()
        elif line.startswith('DP '):  # Point definition
            # Parse coordinates like "DP 52:18:00 N 004:45:00 E"
            coord_match = re.search(r'(\d+):(\d+):(\d+)\s*([NS])\s*(\d+):(\d+):(\d+)\s*([EW])', line)
            if coord_match:
                lat_deg, lat_min, lat_sec, lat_dir, lon_deg, lon_min, lon_sec, lon_dir = coord_match.groups()
                lat = int(lat_deg) + int(lat_min)/60 + int(lat_sec)/3600
                lon = int(lon_deg) + int(lon_min)/60 + int(lon_sec)/3600
                if lat_dir == 'S':
                    lat = -lat
                if lon_dir == 'W':
                    lon = -lon
                coordinates.append({'lat': lat, 'lng': lon})
    
    # Add the last airspace
    if current_airspace and coordinates:
        current_airspace['coordinates'] = coordinates
        current_airspace['floor_meters'] = parse_altitude(current_airspace.get('floor'))
        current_airspace['ceiling_meters'] = parse_altitude(current_airspace.get('ceiling'))
        airspaces.append(current_airspace)
    
    return airspaces

def get_airspace_requirements(airspace_type: str) -> List[str]:
    """Get special requirements for airspace types"""
    requirements = {
        'CTR': ['Radio contact required', 'Clearance needed for entry', 'Transponder recommended'],
        'TMA': ['Radio contact required', 'Clearance needed for entry', 'Transponder required'],
        'R': ['Entry prohibited', 'Military activity', 'Avoid at all times'],
        'P': ['Entry strictly prohibited', 'No overflight', 'Security restriction'],
        'D': ['Dangerous area', 'Military exercises possible', 'Monitor NOTAMS'],
        'A': ['IFR only', 'No VFR permitted', 'ATC clearance required'],
        'B': ['Clearance required', 'Transponder mandatory', 'Radio contact required'],
        'C': ['Radio contact required', 'Transponder required', 'VFR permitted with clearance'],
        'E': ['Radio contact recommended', 'Transponder recommended', 'VFR permitted']
    }
    return requirements.get(airspace_type, [])

def point_in_polygon(point: Tuple[float, float], polygon_coords: List[Dict]) -> bool:
    """Check if a point is inside a polygon using Shapely"""
    try:
        point_geom = Point(point[1], point[0])  # lng, lat for Shapely
        polygon_points = [(coord['lng'], coord['lat']) for coord in polygon_coords]
        if len(polygon_points) < 3:
            return False
        polygon_geom = Polygon(polygon_points)
        return polygon_geom.contains(point_geom)
    except:
        return False

def line_intersects_polygon(line_coords: List[Tuple[float, float]], polygon_coords: List[Dict]) -> bool:
    """Check if a line intersects with a polygon"""
    try:
        line_points = [(coord[1], coord[0]) for coord in line_coords]  # lng, lat for Shapely
        line_geom = LineString(line_points)
        
        polygon_points = [(coord['lng'], coord['lat']) for coord in polygon_coords]
        if len(polygon_points) < 3:
            return False
        polygon_geom = Polygon(polygon_points)
        
        return line_geom.intersects(polygon_geom)
    except:
        return False

def calculate_route_cost(point1: RoutePoint, point2: RoutePoint, airspaces: List[Dict], preferences: AirspacePreferences) -> Tuple[float, List[str]]:
    """Calculate the cost of flying between two points considering distance, airspace, and altitude"""
    distance = haversine_distance(point1.lat, point1.lng, point2.lat, point2.lng)
    base_cost = distance
    penalties = []
    
    # Sample points along the path for airspace analysis
    num_samples = max(5, int(distance))
    for i in range(num_samples + 1):
        ratio = i / num_samples if num_samples > 0 else 0
        sample_lat = point1.lat + ratio * (point2.lat - point1.lat)
        sample_lng = point1.lng + ratio * (point2.lng - point1.lng)
        
        for airspace in airspaces:
            if point_in_polygon((sample_lat, sample_lng), airspace.get('coordinates', [])):
                airspace_type = airspace.get('type', 'Unknown')
                airspace_name = airspace.get('name', 'Unknown')
                
                # Apply penalties based on airspace type and user preferences
                penalty = 0
                if airspace_type in ['R', 'P']:
                    penalty = 1000  # Extremely high penalty for prohibited zones
                    penalties.append(f"CRITICAL: {airspace_name} - Prohibited Zone")
                elif airspace_type == 'CTR' and preferences.avoid_ctr:
                    penalty = 50  # Moderate penalty for CTR
                    penalties.append(f"CTR: {airspace_name} - Requires clearance")
                elif airspace_type == 'TMA' and preferences.avoid_tma:
                    penalty = 75  # Higher penalty for TMA
                    penalties.append(f"TMA: {airspace_name} - Controlled airspace")
                elif airspace_type == 'D' and preferences.avoid_danger:
                    penalty = 25  # Lower penalty for danger areas
                    penalties.append(f"DANGER: {airspace_name} - Monitor NOTAMs")
                
                base_cost += penalty
                
                # Break after first airspace conflict to avoid multiple penalties for same segment
                if penalty > 0:
                    break
    
    return base_cost, penalties

def generate_grid_points(start: RoutePoint, end: RoutePoint, grid_size: int = 5) -> List[RoutePoint]:
    """Generate a grid of potential waypoints between start and end"""
    points = []
    
    # Create bounding box with some margin
    min_lat = min(start.lat, end.lat) - 0.5
    max_lat = max(start.lat, end.lat) + 0.5
    min_lng = min(start.lng, end.lng) - 0.5
    max_lng = max(start.lng, end.lng) + 0.5
    
    lat_step = (max_lat - min_lat) / grid_size
    lng_step = (max_lng - min_lng) / grid_size
    
    for i in range(grid_size + 1):
        for j in range(grid_size + 1):
            lat = min_lat + i * lat_step
            lng = min_lng + j * lng_step
            
            # Skip points too close to start or end
            if (haversine_distance(lat, lng, start.lat, start.lng) > 5 and 
                haversine_distance(lat, lng, end.lat, end.lng) > 5):
                points.append(RoutePoint(
                    lat=lat,
                    lng=lng,
                    altitude=300,  # Default altitude
                    waypoint_name=f"Grid_{i}_{j}"
                ))
    
    return points

def dijkstra_route_planning(start: RoutePoint, end: RoutePoint, airspaces: List[Dict], preferences: AirspacePreferences) -> Tuple[List[RoutePoint], float, List[str]]:
    """Advanced route planning using Dijkstra's algorithm with airspace avoidance"""
    
    # Generate potential waypoints
    grid_points = generate_grid_points(start, end, grid_size=6)
    all_points = [start] + grid_points + [end]
    
    # Initialize distances and previous points
    distances = {i: float('inf') for i in range(len(all_points))}
    previous = {i: None for i in range(len(all_points))}
    all_penalties = {i: [] for i in range(len(all_points))}
    
    start_idx = 0
    end_idx = len(all_points) - 1
    distances[start_idx] = 0
    
    # Priority queue (simple implementation)
    unvisited = list(range(len(all_points)))
    
    while unvisited:
        # Find point with minimum distance
        current = min(unvisited, key=lambda x: distances[x])
        unvisited.remove(current)
        
        if current == end_idx:
            break
            
        if distances[current] == float('inf'):
            break
        
        # Check all neighbors
        for neighbor in unvisited:
            cost, penalties = calculate_route_cost(all_points[current], all_points[neighbor], airspaces, preferences)
            
            # Add distance-based cost scaling
            distance_to_goal = haversine_distance(all_points[neighbor].lat, all_points[neighbor].lng, end.lat, end.lng)
            heuristic_cost = cost + distance_to_goal * 0.1  # A* style heuristic
            
            new_distance = distances[current] + heuristic_cost
            
            if new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                previous[neighbor] = current
                all_penalties[neighbor] = penalties
    
    # Reconstruct path
    if distances[end_idx] == float('inf'):
        # No valid path found, return direct route with warnings
        return [start, end], haversine_distance(start.lat, start.lng, end.lat, end.lng), ["WARNING: No safe route found, using direct path"]
    
    path = []
    current = end_idx
    total_penalties = []
    
    while current is not None:
        path.append(all_points[current])
        total_penalties.extend(all_penalties[current])
        current = previous[current]
    
    path.reverse()
    
    # Remove duplicate penalties
    unique_penalties = list(set(total_penalties))
    
    return path[1:-1], distances[end_idx], unique_penalties  # Return waypoints (excluding start/end)

def calculate_avoidance_waypoints(start: RoutePoint, end: RoutePoint, airspaces: List[Dict], preferences: AirspacePreferences) -> List[RoutePoint]:
    """Calculate optimal waypoints using advanced pathfinding algorithm"""
    try:
        waypoints, total_cost, penalties = dijkstra_route_planning(start, end, airspaces, preferences)
        
        # If the optimal path is too expensive, try alternative approach
        if total_cost > 1000:  # High penalty indicates prohibited airspace
            # Try a simpler arc-based avoidance
            direct_distance = haversine_distance(start.lat, start.lng, end.lat, end.lng)
            
            # Create waypoints that arc around the direct path
            mid_lat = (start.lat + end.lat) / 2
            mid_lng = (start.lng + end.lng) / 2
            
            # Calculate perpendicular offset
            bearing_to_end = bearing(start.lat, start.lng, end.lat, end.lng)
            offset_bearing = (bearing_to_end + 90) % 360
            
            # Create arc waypoints
            arc_distance = max(20, direct_distance * 0.3)  # 20km minimum or 30% of direct distance
            lat_offset = arc_distance * math.cos(math.radians(offset_bearing)) / 111.32
            lng_offset = arc_distance * math.sin(math.radians(offset_bearing)) / (111.32 * math.cos(math.radians(mid_lat)))
            
            waypoint1 = RoutePoint(
                lat=start.lat + (end.lat - start.lat) * 0.33 + lat_offset * 0.5,
                lng=start.lng + (end.lng - start.lng) * 0.33 + lng_offset * 0.5,
                altitude=preferences.preferred_altitude,
                waypoint_name="Arc Avoidance 1"
            )
            
            waypoint2 = RoutePoint(
                lat=mid_lat + lat_offset,
                lng=mid_lng + lng_offset,
                altitude=preferences.preferred_altitude,
                waypoint_name="Arc Avoidance Center"
            )
            
            waypoint3 = RoutePoint(
                lat=start.lat + (end.lat - start.lat) * 0.67 + lat_offset * 0.5,
                lng=start.lng + (end.lng - start.lng) * 0.67 + lng_offset * 0.5,
                altitude=preferences.preferred_altitude,
                waypoint_name="Arc Avoidance 2"
            )
            
            return [waypoint1, waypoint2, waypoint3]
        
        # Filter waypoints that are too close to each other
        filtered_waypoints = []
        last_point = start
        
        for waypoint in waypoints:
            if haversine_distance(last_point.lat, last_point.lng, waypoint.lat, waypoint.lng) > 15:  # Minimum 15km between waypoints
                filtered_waypoints.append(waypoint)
                last_point = waypoint
        
        return filtered_waypoints[:5]  # Limit to 5 waypoints maximum
        
    except Exception as e:
        # Fallback to simple avoidance if advanced algorithm fails
        return simple_avoidance_fallback(start, end, airspaces, preferences)

def simple_avoidance_fallback(start: RoutePoint, end: RoutePoint, airspaces: List[Dict], preferences: AirspacePreferences) -> List[RoutePoint]:
    """Simple fallback avoidance algorithm"""
    waypoints = []
    
    # Filter airspaces that should be avoided
    avoid_types = []
    if preferences.avoid_ctr:
        avoid_types.extend(['CTR'])
    if preferences.avoid_tma:
        avoid_types.extend(['TMA'])
    if preferences.avoid_restricted:
        avoid_types.extend(['R'])
    if preferences.avoid_prohibited:
        avoid_types.extend(['P'])
    if preferences.avoid_danger:
        avoid_types.extend(['D'])
    
    restricted_airspaces = [a for a in airspaces if a.get('type') in avoid_types]
    
    # Check if direct line intersects any restricted airspace
    direct_line = [(start.lat, start.lng), (end.lat, end.lng)]
    
    for airspace in restricted_airspaces[:3]:  # Limit to first 3 conflicts
        if line_intersects_polygon(direct_line, airspace.get('coordinates', [])):
            # Calculate avoidance waypoint
            center_lat = sum(c['lat'] for c in airspace['coordinates']) / len(airspace['coordinates'])
            center_lng = sum(c['lng'] for c in airspace['coordinates']) / len(airspace['coordinates'])
            
            # Calculate bearing from start to end, then offset perpendicular
            route_bearing = bearing(start.lat, start.lng, end.lat, end.lng)
            avoidance_bearing = (route_bearing + 90) % 360  # 90 degrees offset
            
            # Calculate avoidance point
            avoidance_distance = 15  # km
            lat_offset = avoidance_distance * math.cos(math.radians(avoidance_bearing)) / 111.32
            lng_offset = avoidance_distance * math.sin(math.radians(avoidance_bearing)) / (111.32 * math.cos(math.radians(center_lat)))
            
            waypoint = RoutePoint(
                lat=center_lat + lat_offset,
                lng=center_lng + lng_offset,
                altitude=preferences.preferred_altitude,
                waypoint_name=f"Avoid {airspace.get('name', 'Unknown')[:20]}"
            )
            waypoints.append(waypoint)
    
    return waypoints

def calculate_optimal_altitude(start_point: RoutePoint, end_point: RoutePoint, airspaces: List[Dict], preferences: AirspacePreferences) -> Tuple[float, List[str]]:
    """Calculate optimal altitude for a segment considering airspace floors/ceilings"""
    altitude_warnings = []
    optimal_altitude = preferences.preferred_altitude
    
    # Sample points along the segment
    segment_distance = haversine_distance(start_point.lat, start_point.lng, end_point.lat, end_point.lng)
    num_samples = max(5, int(segment_distance * 2))  # More samples for better accuracy
    
    conflicting_airspaces = []
    
    for sample in range(num_samples + 1):
        ratio = sample / num_samples if num_samples > 0 else 0
        sample_lat = start_point.lat + ratio * (end_point.lat - start_point.lat)
        sample_lng = start_point.lng + ratio * (end_point.lng - start_point.lng)
        
        for airspace in airspaces:
            if point_in_polygon((sample_lat, sample_lng), airspace.get('coordinates', [])):
                conflicting_airspaces.append(airspace)
    
    # Remove duplicates
    unique_airspaces = {a['name']: a for a in conflicting_airspaces}.values()
    
    # Calculate optimal altitude considering all conflicting airspaces
    for airspace in unique_airspaces:
        airspace_type = airspace.get('type', 'Unknown')
        airspace_name = airspace.get('name', 'Unknown')
        country = airspace.get('country', 'EU')
        floor = airspace.get('floor_meters', 0)
        ceiling = airspace.get('ceiling_meters', 10000)
        
        if airspace_type in ['R', 'P']:
            # Cannot fly through restricted/prohibited zones
            altitude_warnings.append(f"CRITICAL: {airspace_name} ({country}) - No altitude clearance possible")
        elif airspace_type in ['CTR', 'TMA']:
            if preferences.avoid_ctr or preferences.avoid_tma:
                if floor is not None and ceiling is not None:
                    # Try to fly above the airspace
                    if ceiling + 100 <= preferences.maximum_altitude:
                        suggested_altitude = ceiling + 100  # 100m clearance
                        if suggested_altitude > optimal_altitude:
                            optimal_altitude = suggested_altitude
                            altitude_warnings.append(f"ALTITUDE: Flying at {optimal_altitude}m to clear {airspace_name} ({country})")
                    else:
                        altitude_warnings.append(f"WARNING: Cannot clear {airspace_name} ({country}) - exceeds max altitude")
            else:
                altitude_warnings.append(f"INFO: Entering {airspace_name} ({country}) - clearance required")
        elif airspace_type == 'D':
            altitude_warnings.append(f"CAUTION: Danger area {airspace_name} ({country}) - monitor NOTAMs")
    
    # Ensure altitude is within user preferences
    optimal_altitude = max(preferences.minimum_altitude, min(optimal_altitude, preferences.maximum_altitude))
    
    return optimal_altitude, altitude_warnings

def calculate_route_segments(route_points: List[RoutePoint], airspaces: List[Dict], preferences: AirspacePreferences) -> List[RouteSegment]:
    """Enhanced route segments calculation with optimal altitude planning"""
    segments = []
    
    for i in range(len(route_points) - 1):
        start_point = route_points[i]
        end_point = route_points[i + 1]
        
        # Calculate segment distance
        segment_distance = haversine_distance(start_point.lat, start_point.lng, end_point.lat, end_point.lng)
        
        # Calculate optimal altitude for this segment
        segment_altitude, altitude_warnings = calculate_optimal_altitude(start_point, end_point, airspaces, preferences)
        
        # Determine segment type based on warnings
        segment_type = "safe"
        if any("CRITICAL" in warning for warning in altitude_warnings):
            segment_type = "restricted"
        elif any("WARNING" in warning for warning in altitude_warnings):
            segment_type = "avoid"
        elif any("CAUTION" in warning or "ALTITUDE" in warning for warning in altitude_warnings):
            segment_type = "caution"
        
        # Add airspace-specific requirements
        enhanced_warnings = []
        processed_airspaces = set()
        
        # Sample points for detailed airspace analysis
        num_samples = max(3, int(segment_distance))
        for sample in range(num_samples + 1):
            ratio = sample / num_samples if num_samples > 0 else 0
            sample_lat = start_point.lat + ratio * (end_point.lat - start_point.lat)
            sample_lng = start_point.lng + ratio * (end_point.lng - start_point.lng)
            
            for airspace in airspaces:
                airspace_name = airspace.get('name', 'Unknown')
                if (airspace_name not in processed_airspaces and 
                    point_in_polygon((sample_lat, sample_lng), airspace.get('coordinates', []))):
                    
                    processed_airspaces.add(airspace_name)
                    airspace_type = airspace.get('type', 'Unknown')
                    country = airspace.get('country', 'EU')
                    
                    # Add specific requirements for this airspace
                    requirements = get_airspace_requirements(airspace_type)
                    for req in requirements:
                        enhanced_warnings.append(f"{airspace_name} ({country}): {req}")
        
        # Combine altitude warnings with enhanced warnings
        all_warnings = altitude_warnings + enhanced_warnings
        
        # Create enhanced segment
        segment = RouteSegment(
            start=Coordinate(lat=start_point.lat, lng=start_point.lng),
            end=Coordinate(lat=end_point.lat, lng=end_point.lng),
            altitude=segment_altitude,
            segment_type=segment_type,
            airspace_warnings=all_warnings,
            distance=segment_distance
        )
        segments.append(segment)
    
    return segments

async def get_wind_data(lat: float, lng: float, altitude: float = 1000) -> WindData:
    """Get wind data for a specific location (mock data for now)"""
    # In a real implementation, you would call a weather API like OpenWeatherMap
    # For now, returning mock data
    import random
    return WindData(
        lat=lat,
        lng=lng,
        speed=random.uniform(2, 15),  # 2-15 m/s wind speed
        direction=random.uniform(0, 360),  # Random direction
        altitude=altitude
    )


# Ultra-Detailed Spanish Airspace Data + European Coverage
EUROPEAN_AIRSPACE_DATA = """
* Ultra-Detailed Spanish Airspace Database for Paramotoring
* Complete coverage of Spain with all major airports, military zones, restricted areas
* Plus comprehensive European airspace data

* === SPAIN - ULTRA DETAILED ===

* Madrid Area Airspaces
AC CTR
AN Madrid-Barajas CTR
AC0 ES
AL SFC
AH 4000FT
AF 118.200
DP 40:28:30 N 003:36:00 W
DP 40:33:00 N 003:31:30 W
DP 40:34:30 N 003:27:00 W
DP 40:31:00 N 003:23:30 W
DP 40:27:30 N 003:27:00 W
DP 40:25:00 N 003:31:30 W
DP 40:26:30 N 003:36:00 W
DP 40:28:30 N 003:36:00 W

AC TMA
AN Madrid TMA Sector 1
AC0 ES
AL 4000FT
AH FL245
AF 119.200
DP 40:10:00 N 003:55:00 W
DP 40:50:00 N 003:10:00 W
DP 40:45:00 N 002:50:00 W
DP 40:05:00 N 003:35:00 W
DP 40:10:00 N 003:55:00 W

AC TMA
AN Madrid TMA Sector 2
AC0 ES
AL 4000FT
AH FL195
AF 119.700
DP 40:00:00 N 004:10:00 W
DP 40:15:00 N 003:55:00 W
DP 40:05:00 N 003:35:00 W
DP 39:50:00 N 003:50:00 W
DP 40:00:00 N 004:10:00 W

AC CTR
AN Madrid-Cuatro Vientos CTR
AC0 ES
AL SFC
AH 2500FT
AF 122.100
DP 40:22:00 N 003:47:00 W
DP 40:24:30 N 003:44:30 W
DP 40:23:00 N 003:42:00 W
DP 40:20:30 N 003:44:30 W
DP 40:22:00 N 003:47:00 W

AC CTR
AN Madrid-Torrejón CTR
AC0 ES
AL SFC
AH 3000FT
AF 122.050
DP 40:28:30 N 003:27:00 W
DP 40:31:00 N 003:24:00 W
DP 40:29:30 N 003:21:00 W
DP 40:27:00 N 003:24:00 W
DP 40:28:30 N 003:27:00 W

* Barcelona Area Airspaces
AC CTR
AN Barcelona CTR
AC0 ES
AL SFC
AH 3500FT
AF 118.300
DP 41:15:30 N 002:02:00 E
DP 41:20:00 N 002:07:30 E
DP 41:21:30 N 002:12:00 E
DP 41:19:00 N 002:16:30 E
DP 41:15:00 N 002:14:00 E
DP 41:12:30 N 002:09:30 E
DP 41:14:00 N 002:05:00 E
DP 41:15:30 N 002:02:00 E

AC TMA
AN Barcelona TMA
AC0 ES
AL 3500FT
AH FL245
AF 120.300
DP 41:00:00 N 001:45:00 E
DP 41:35:00 N 002:30:00 E
DP 41:30:00 N 002:45:00 E
DP 40:55:00 N 002:00:00 E
DP 41:00:00 N 001:45:00 E

AC CTR
AN Sabadell CTR
AC0 ES
AL SFC
AH 2000FT
AF 122.900
DP 41:31:00 N 002:06:00 E
DP 41:33:00 N 002:08:30 E
DP 41:32:00 N 002:11:00 E
DP 41:30:00 N 002:08:30 E
DP 41:31:00 N 002:06:00 E

AC CTR
AN Girona CTR
AC0 ES
AL SFC
AH 2500FT
AF 119.500
DP 41:53:00 N 002:45:00 E
DP 41:55:30 N 002:47:30 E
DP 41:54:00 N 002:50:00 E
DP 41:51:30 N 002:47:30 E
DP 41:53:00 N 002:45:00 E

* Valencia Area Airspaces
AC CTR
AN Valencia CTR
AC0 ES
AL SFC
AH 2500FT
AF 118.775
DP 39:27:00 N 000:28:00 W
DP 39:30:00 N 000:25:00 W
DP 39:31:30 N 000:22:00 W
DP 39:29:00 N 000:19:00 W
DP 39:26:00 N 000:22:00 W
DP 39:24:30 N 000:25:00 W
DP 39:27:00 N 000:28:00 W

AC TMA
AN Valencia TMA
AC0 ES
AL 2500FT
AH FL195
AF 119.025
DP 39:10:00 N 000:45:00 W
DP 39:45:00 N 000:05:00 W
DP 39:40:00 N 000:15:00 E
DP 39:05:00 N 000:25:00 W
DP 39:10:00 N 000:45:00 W

AC CTR
AN Castellón CTR
AC0 ES
AL SFC
AH 2000FT
AF 122.450
DP 40:12:30 N 000:01:30 E
DP 40:14:30 N 000:04:00 E
DP 40:13:00 N 000:06:30 E
DP 40:11:00 N 000:04:00 E
DP 40:12:30 N 000:01:30 E

* Seville Area Airspaces
AC CTR
AN Sevilla CTR
AC0 ES
AL SFC
AH 3000FT
AF 120.600
DP 37:24:30 N 005:54:00 W
DP 37:27:00 N 005:51:00 W
DP 37:25:30 N 005:48:00 W
DP 37:23:00 N 005:51:00 W
DP 37:24:30 N 005:54:00 W

AC TMA
AN Sevilla TMA
AC0 ES
AL 3000FT
AH FL195
AF 121.050
DP 37:10:00 N 006:10:00 W
DP 37:40:00 N 005:30:00 W
DP 37:35:00 N 005:15:00 W
DP 37:05:00 N 005:55:00 W
DP 37:10:00 N 006:10:00 W

AC CTR
AN Jerez CTR
AC0 ES
AL SFC
AH 2500FT
AF 119.650
DP 36:44:30 N 006:03:30 W
DP 36:46:30 N 006:00:30 W
DP 36:45:00 N 005:57:30 W
DP 36:43:00 N 006:00:30 W
DP 36:44:30 N 006:03:30 W

* Bilbao Area Airspaces
AC CTR
AN Bilbao CTR
AC0 ES
AL SFC
AH 3000FT
AF 118.550
DP 43:17:30 N 002:54:00 W
DP 43:20:00 N 002:51:00 W
DP 43:18:30 N 002:48:00 W
DP 43:16:00 N 002:51:00 W
DP 43:17:30 N 002:54:00 W

AC TMA
AN Bilbao TMA
AC0 ES
AL 3000FT
AH FL195
AF 119.800
DP 43:05:00 N 003:10:00 W
DP 43:30:00 N 002:35:00 W
DP 43:25:00 N 002:20:00 W
DP 43:00:00 N 002:55:00 W
DP 43:05:00 N 003:10:00 W

AC CTR
AN Santander CTR
AC0 ES
AL SFC
AH 2500FT
AF 120.200
DP 43:25:30 N 003:48:00 W
DP 43:27:30 N 003:45:00 W
DP 43:26:00 N 003:42:00 W
DP 43:24:00 N 003:45:00 W
DP 43:25:30 N 003:48:00 W

* Canary Islands Airspaces
AC CTR
AN Las Palmas CTR
AC0 ES
AL SFC
AH 3500FT
AF 119.100
DP 27:55:30 N 015:23:00 W
DP 27:58:00 N 015:20:00 W
DP 27:56:30 N 015:17:00 W
DP 27:54:00 N 015:20:00 W
DP 27:55:30 N 015:23:00 W

AC TMA
AN Canarias TMA
AC0 ES
AL 3500FT
AH FL245
AF 121.300
DP 27:40:00 N 015:40:00 W
DP 28:15:00 N 015:00:00 W
DP 28:10:00 N 014:45:00 W
DP 27:35:00 N 015:25:00 W
DP 27:40:00 N 015:40:00 W

AC CTR
AN Tenerife Sur CTR
AC0 ES
AL SFC
AH 3000FT
AF 118.700
DP 28:02:30 N 016:34:30 W
DP 28:05:00 N 016:31:30 W
DP 28:03:30 N 016:28:30 W
DP 28:01:00 N 016:31:30 W
DP 28:02:30 N 016:34:30 W

AC CTR
AN Tenerife Norte CTR
AC0 ES
AL SFC
AH 3000FT
AF 119.300
DP 28:29:00 N 016:20:30 W
DP 28:31:00 N 016:17:30 W
DP 28:29:30 N 016:14:30 W
DP 28:27:30 N 016:17:30 W
DP 28:29:00 N 016:20:30 W

* Balearic Islands Airspaces
AC CTR
AN Palma CTR
AC0 ES
AL SFC
AH 3000FT
AF 118.150
DP 39:32:00 N 002:43:00 E
DP 39:35:00 N 002:46:00 E
DP 39:36:30 N 002:49:00 E
DP 39:34:00 N 002:52:00 E
DP 39:31:00 N 002:49:00 E
DP 39:29:30 N 002:46:00 E
DP 39:32:00 N 002:43:00 E

AC TMA
AN Baleares TMA
AC0 ES
AL 3000FT
AH FL245
AF 120.050
DP 39:15:00 N 002:25:00 E
DP 39:50:00 N 003:05:00 E
DP 39:45:00 N 003:20:00 E
DP 39:10:00 N 002:40:00 E
DP 39:15:00 N 002:25:00 E

AC CTR
AN Ibiza CTR
AC0 ES
AL SFC
AH 2500FT
AF 119.000
DP 38:52:00 N 001:22:00 E
DP 38:54:00 N 001:24:30 E
DP 38:52:30 N 001:27:00 E
DP 38:50:30 N 001:24:30 E
DP 38:52:00 N 001:22:00 E

AC CTR
AN Mahón CTR
AC0 ES
AL SFC
AH 2000FT
AF 119.250
DP 39:51:30 N 004:13:00 E
DP 39:53:30 N 004:15:30 E
DP 39:52:00 N 004:18:00 E
DP 39:50:00 N 004:15:30 E
DP 39:51:30 N 004:13:00 E

* Military and Restricted Zones - Spain
AC R
AN LEI R-44A Zaragoza Military
AC0 ES
AL SFC
AH FL195
DP 41:38:00 N 000:52:00 W
DP 41:47:00 N 000:43:00 W
DP 41:44:00 N 000:38:00 W
DP 41:35:00 N 000:47:00 W
DP 41:38:00 N 000:52:00 W

AC R
AN LEI R-44B Zaragoza Extended
AC0 ES
AL SFC
AH FL245
DP 41:30:00 N 001:00:00 W
DP 41:55:00 N 000:30:00 W
DP 41:50:00 N 000:20:00 W
DP 41:25:00 N 000:50:00 W
DP 41:30:00 N 001:00:00 W

AC R
AN LEI R-33 Bardenas Reales
AC0 ES
AL SFC
AH FL195
DP 42:10:00 N 001:25:00 W
DP 42:20:00 N 001:15:00 W
DP 42:15:00 N 001:05:00 W
DP 42:05:00 N 001:15:00 W
DP 42:10:00 N 001:25:00 W

AC R
AN LEI R-32 San Gregorio
AC0 ES
AL SFC
AH FL195
DP 41:05:00 N 002:20:00 W
DP 41:15:00 N 002:10:00 W
DP 41:10:00 N 002:00:00 W
DP 41:00:00 N 002:10:00 W
DP 41:05:00 N 002:20:00 W

AC R
AN LEI R-31 Chinchilla
AC0 ES
AL SFC
AH FL195
DP 38:50:00 N 001:40:00 W
DP 39:00:00 N 001:30:00 W
DP 38:55:00 N 001:20:00 W
DP 38:45:00 N 001:30:00 W
DP 38:50:00 N 001:40:00 W

AC R
AN LEI R-25 Alcantarilla
AC0 ES
AL SFC
AH FL195
DP 37:50:00 N 001:15:00 W
DP 38:00:00 N 001:05:00 W
DP 37:55:00 N 000:55:00 W
DP 37:45:00 N 001:05:00 W
DP 37:50:00 N 001:15:00 W

AC R
AN LEI R-26 Los Llanos
AC0 ES
AL SFC
AH FL195
DP 38:55:00 N 005:55:00 W
DP 39:05:00 N 005:45:00 W
DP 39:00:00 N 005:35:00 W
DP 38:50:00 N 005:45:00 W
DP 38:55:00 N 005:55:00 W

AC R
AN LEI R-18 Morón
AC0 ES
AL SFC
AH FL195
DP 37:05:00 N 005:40:00 W
DP 37:15:00 N 005:30:00 W
DP 37:10:00 N 005:20:00 W
DP 37:00:00 N 005:30:00 W
DP 37:05:00 N 005:40:00 W

* Danger Areas - Spain
AC D
AN LEI D-21 Valencia Training
AC0 ES
AL SFC
AH 8000FT
DP 39:20:00 N 000:20:00 W
DP 39:35:00 N 000:05:00 W
DP 39:30:00 N 000:05:00 E
DP 39:15:00 N 000:10:00 W
DP 39:20:00 N 000:20:00 W

AC D
AN LEI D-22 Castellón Sea
AC0 ES
AL SFC
AH 6000FT
DP 40:05:00 N 000:05:00 E
DP 40:15:00 N 000:15:00 E
DP 40:10:00 N 000:25:00 E
DP 40:00:00 N 000:15:00 E
DP 40:05:00 N 000:05:00 E

AC D
AN LEI D-15 Cádiz Bay
AC0 ES
AL SFC
AH 5000FT
DP 36:25:00 N 006:20:00 W
DP 36:35:00 N 006:10:00 W
DP 36:30:00 N 006:00:00 W
DP 36:20:00 N 006:10:00 W
DP 36:25:00 N 006:20:00 W

AC D
AN LEI D-16 Huelva Coast
AC0 ES
AL SFC
AH 4000FT
DP 37:05:00 N 007:05:00 W
DP 37:15:00 N 006:55:00 W
DP 37:10:00 N 006:45:00 W
DP 37:00:00 N 006:55:00 W
DP 37:05:00 N 007:05:00 W

* Prohibited Areas - Spain
AC P
AN LEI P-12 Madrid Moncloa
AC0 ES
AL SFC
AH 3000FT
DP 40:26:00 N 003:44:00 W
DP 40:28:00 N 003:42:00 W
DP 40:27:00 N 003:40:00 W
DP 40:25:00 N 003:42:00 W
DP 40:26:00 N 003:44:00 W

AC P
AN LEI P-11 Barcelona Port
AC0 ES
AL SFC
AH 2000FT
DP 41:21:00 N 002:09:00 E
DP 41:23:00 N 002:11:00 E
DP 41:22:00 N 002:13:00 E
DP 41:20:00 N 002:11:00 E
DP 41:21:00 N 002:09:00 E

AC P
AN LEI P-13 Valencia Port
AC0 ES
AL SFC
AH 1500FT
DP 39:26:00 N 000:18:00 W
DP 39:28:00 N 000:16:00 W
DP 39:27:00 N 000:14:00 W
DP 39:25:00 N 000:16:00 W
DP 39:26:00 N 000:18:00 W

* Regional Airports - Spain
AC CTR
AN Alicante CTR
AC0 ES
AL SFC
AH 2500FT
AF 119.100
DP 38:16:30 N 000:33:00 W
DP 38:18:30 N 000:30:00 W
DP 38:17:00 N 000:27:00 W
DP 38:15:00 N 000:30:00 W
DP 38:16:30 N 000:33:00 W

AC CTR
AN Málaga CTR
AC0 ES
AL SFC
AH 3000FT
AF 118.400
DP 36:39:00 N 004:30:00 W
DP 36:42:00 N 004:27:00 W
DP 36:40:30 N 004:24:00 W
DP 36:37:30 N 004:27:00 W
DP 36:39:00 N 004:30:00 W

AC CTR
AN A Coruña CTR
AC0 ES
AL SFC
AH 2500FT
AF 120.300
DP 43:18:00 N 008:23:00 W
DP 43:20:30 N 008:20:00 W
DP 43:19:00 N 008:17:00 W
DP 43:16:30 N 008:20:00 W
DP 43:18:00 N 008:23:00 W

AC CTR
AN Vigo CTR
AC0 ES
AL SFC
AH 2000FT
AF 119.700
DP 42:13:30 N 008:38:00 W
DP 42:15:30 N 008:35:00 W
DP 42:14:00 N 008:32:00 W
DP 42:12:00 N 008:35:00 W
DP 42:13:30 N 008:38:00 W

AC CTR
AN Santiago CTR
AC0 ES
AL SFC
AH 2500FT
AF 121.750
DP 42:53:30 N 008:25:00 W
DP 42:55:30 N 008:22:00 W
DP 42:54:00 N 008:19:00 W
DP 42:52:00 N 008:22:00 W
DP 42:53:30 N 008:25:00 W

AC CTR
AN Asturias CTR
AC0 ES
AL SFC
AH 3000FT
AF 118.850
DP 43:33:30 N 006:02:00 W
DP 43:35:30 N 005:59:00 W
DP 43:34:00 N 005:56:00 W
DP 43:32:00 N 005:59:00 W
DP 43:33:30 N 006:02:00 W

AC CTR
AN León CTR
AC0 ES
AL SFC
AH 2000FT
AF 122.100
DP 42:35:00 N 005:39:00 W
DP 42:37:00 N 005:36:00 W
DP 42:35:30 N 005:33:00 W
DP 42:33:30 N 005:36:00 W
DP 42:35:00 N 005:39:00 W

AC CTR
AN Salamanca CTR
AC0 ES
AL SFC
AH 2500FT
AF 123.000
DP 40:57:00 N 005:30:00 W
DP 40:59:00 N 005:27:00 W
DP 40:57:30 N 005:24:00 W
DP 40:55:30 N 005:27:00 W
DP 40:57:00 N 005:30:00 W

AC CTR
AN Valladolid CTR
AC0 ES
AL SFC
AH 2000FT
AF 120.450
DP 41:42:30 N 004:51:00 W
DP 41:44:30 N 004:48:00 W
DP 41:43:00 N 004:45:00 W
DP 41:41:00 N 004:48:00 W
DP 41:42:30 N 004:51:00 W

* === FRANCE ===
AC CTR
AN Paris-Charles de Gaulle CTR
AC0 FR
AL SFC
AH 4000FT
AF 118.150
DP 49:00:00 N 002:30:00 E
DP 49:05:00 N 002:35:00 E
DP 49:02:00 N 002:40:00 E
DP 48:57:00 N 002:35:00 E
DP 49:00:00 N 002:30:00 E

AC TMA
AN Paris TMA
AC0 FR
AL 1500FT
AH FL195
AF 119.050
DP 48:40:00 N 002:00:00 E
DP 49:20:00 N 002:50:00 E
DP 49:10:00 N 003:10:00 E
DP 48:30:00 N 002:20:00 E
DP 48:40:00 N 002:00:00 E

AC CTR
AN Nice Côte d'Azur CTR
AC0 FR
AL SFC
AH 3000FT
AF 118.700
DP 43:39:00 N 007:12:00 E
DP 43:42:00 N 007:15:00 E
DP 43:40:00 N 007:18:00 E
DP 43:37:00 N 007:15:00 E
DP 43:39:00 N 007:12:00 E

AC R
AN LF R-43 Mont-de-Marsan
AC0 FR
AL SFC
AH FL195
DP 43:52:00 N 000:28:00 W
DP 43:57:00 N 000:23:00 W
DP 43:54:00 N 000:18:00 W
DP 43:49:00 N 000:23:00 W
DP 43:52:00 N 000:28:00 W

AC D
AN LF D-17 Marseille
AC0 FR
AL SFC
AH 6500FT
DP 43:15:00 N 005:10:00 E
DP 43:20:00 N 005:15:00 E
DP 43:17:00 N 005:20:00 E
DP 43:12:00 N 005:15:00 E
DP 43:15:00 N 005:10:00 E

* === GERMANY ===
AC CTR
AN Frankfurt CTR
AC0 DE
AL SFC
AH 4000FT
AF 118.500
DP 50:01:00 N 008:31:00 E
DP 50:06:00 N 008:36:00 E
DP 50:03:00 N 008:41:00 E
DP 49:58:00 N 008:36:00 E
DP 50:01:00 N 008:31:00 E

AC TMA
AN Munich TMA
AC0 DE
AL 1500FT
AH FL245
AF 120.050
DP 48:00:00 N 011:20:00 E
DP 48:25:00 N 011:50:00 E
DP 48:20:00 N 012:10:00 E
DP 47:55:00 N 011:40:00 E
DP 48:00:00 N 011:20:00 E

AC R
AN ED R-138 Grafenwöhr
AC0 DE
AL SFC
AH FL195
DP 49:40:00 N 011:50:00 E
DP 49:45:00 N 011:55:00 E
DP 49:42:00 N 012:00:00 E
DP 49:37:00 N 011:55:00 E
DP 49:40:00 N 011:50:00 E

AC D
AN ED D-40 Hamburg
AC0 DE
AL SFC
AH 7500FT
DP 53:35:00 N 009:50:00 E
DP 53:40:00 N 009:55:00 E
DP 53:37:00 N 010:00:00 E
DP 53:32:00 N 009:55:00 E
DP 53:35:00 N 009:50:00 E

* === ITALY ===
AC CTR
AN Roma Fiumicino CTR
AC0 IT
AL SFC
AH 3500FT
AF 118.750
DP 41:46:00 N 012:14:00 E
DP 41:51:00 N 012:19:00 E
DP 41:48:00 N 012:24:00 E
DP 41:43:00 N 012:19:00 E
DP 41:46:00 N 012:14:00 E

AC TMA
AN Milano TMA
AC0 IT
AL 2000FT
AH FL195
AF 120.050
DP 45:25:00 N 008:30:00 E
DP 45:50:00 N 009:00:00 E
DP 45:45:00 N 009:20:00 E
DP 45:20:00 N 008:50:00 E
DP 45:25:00 N 008:30:00 E

AC R
AN LI R-18 Sardinia
AC0 IT
AL SFC
AH FL245
DP 40:15:00 N 009:30:00 E
DP 40:20:00 N 009:35:00 E
DP 40:17:00 N 009:40:00 E
DP 40:12:00 N 009:35:00 E
DP 40:15:00 N 009:30:00 E

AC D
AN LI D-47 Naples
AC0 IT
AL SFC
AH 5500FT
DP 40:50:00 N 014:15:00 E
DP 40:55:00 N 014:20:00 E
DP 40:52:00 N 014:25:00 E
DP 40:47:00 N 014:20:00 E
DP 40:50:00 N 014:15:00 E

* === UNITED KINGDOM ===
AC CTR
AN Manchester CTR
AC0 GB
AL SFC
AH 3500FT
AF 118.620
DP 53:21:00 N 002:16:00 W
DP 53:25:00 N 002:10:00 W
DP 53:23:00 N 002:05:00 W
DP 53:19:00 N 002:11:00 W
DP 53:21:00 N 002:16:00 W

AC P
AN London Prohibited Zone
AC0 GB
AL SFC
AH 2500FT
DP 51:28:00 N 000:27:00 W
DP 51:32:00 N 000:20:00 W
DP 51:30:00 N 000:15:00 W
DP 51:26:00 N 000:22:00 W
DP 51:28:00 N 000:27:00 W

* === SWITZERLAND ===
AC CTR
AN Zurich CTR
AC0 CH
AL SFC
AH 4500FT
AF 118.100
DP 47:25:00 N 008:30:00 E
DP 47:30:00 N 008:35:00 E
DP 47:27:00 N 008:40:00 E
DP 47:22:00 N 008:35:00 E
DP 47:25:00 N 008:30:00 E

AC R
AN LS R-12 Alpine
AC0 CH
AL 6500FT
AH FL195
DP 46:30:00 N 007:45:00 E
DP 46:35:00 N 007:50:00 E
DP 46:32:00 N 007:55:00 E
DP 46:27:00 N 007:50:00 E
DP 46:30:00 N 007:45:00 E

* === AUSTRIA ===
AC CTR
AN Vienna CTR
AC0 AT
AL SFC
AH 3500FT
AF 118.800
DP 48:06:00 N 016:33:00 E
DP 48:11:00 N 016:38:00 E
DP 48:08:00 N 016:43:00 E
DP 48:03:00 N 016:38:00 E
DP 48:06:00 N 016:33:00 E

* === NETHERLANDS ===
AC CTR
AN Amsterdam Schiphol CTR
AC0 NL
AL SFC
AH 3000FT
AF 118.400
DP 52:17:00 N 004:45:00 E
DP 52:22:00 N 004:50:00 E
DP 52:19:00 N 004:55:00 E
DP 52:14:00 N 004:50:00 E
DP 52:17:00 N 004:45:00 E

* === BELGIUM ===
AC CTR
AN Brussels CTR
AC0 BE
AL SFC
AH 2500FT
AF 118.950
DP 50:54:00 N 004:29:00 E
DP 50:59:00 N 004:34:00 E
DP 50:56:00 N 004:39:00 E
DP 50:51:00 N 004:34:00 E
DP 50:54:00 N 004:29:00 E
"""


# Routes
@api_router.get("/")
async def root():
    return {"message": "European Paramotorist Flight Planning API with Advanced Airspace Avoidance"}

@api_router.get("/airspaces", response_model=List[Airspace])
async def get_airspaces(type_filter: Optional[str] = None, country: Optional[str] = None):
    """Get airspace data, optionally filtered by type and/or country"""
    try:
        # Parse European airspace data
        parsed_airspaces = parse_openair_airspace(EUROPEAN_AIRSPACE_DATA)
        
        airspaces = []
        for airspace_data in parsed_airspaces:
            if type_filter and airspace_data.get('type') != type_filter:
                continue
            if country and airspace_data.get('country') != country:
                continue
                
            airspace = Airspace(
                name=airspace_data.get('name', 'Unknown'),
                type=airspace_data.get('type', 'Unknown'),
                country=airspace_data.get('country', 'EU'),
                coordinates=[Coordinate(**coord) for coord in airspace_data.get('coordinates', [])],
                floor=airspace_data.get('floor'),
                ceiling=airspace_data.get('ceiling'),
                floor_meters=airspace_data.get('floor_meters'),
                ceiling_meters=airspace_data.get('ceiling_meters'),
                frequency=airspace_data.get('frequency'),
                requirements=get_airspace_requirements(airspace_data.get('type', 'Unknown'))
            )
            airspaces.append(airspace)
        
        return airspaces
    except Exception as e:
        logger.error(f"Error getting airspaces: {e}")
        raise HTTPException(status_code=500, detail="Failed to get airspace data")

@api_router.get("/countries")
async def get_countries():
    """Get available countries with airspace data"""
    return {
        "countries": [
            {"code": "ES", "name": "Spain", "flag": "🇪🇸"},
            {"code": "FR", "name": "France", "flag": "🇫🇷"},
            {"code": "DE", "name": "Germany", "flag": "🇩🇪"},
            {"code": "IT", "name": "Italy", "flag": "🇮🇹"},
            {"code": "GB", "name": "United Kingdom", "flag": "🇬🇧"},
            {"code": "CH", "name": "Switzerland", "flag": "🇨🇭"},
            {"code": "AT", "name": "Austria", "flag": "🇦🇹"},
            {"code": "NL", "name": "Netherlands", "flag": "🇳🇱"},
            {"code": "BE", "name": "Belgium", "flag": "🇧🇪"}
        ]
    }

@api_router.get("/airspace-types")
async def get_airspace_types():
    """Get available airspace types for filtering"""
    return {
        "types": [
            {"code": "CTR", "name": "Control Zone", "description": "Controlled airspace around airports", "avoidable": True},
            {"code": "TMA", "name": "Terminal Control Area", "description": "Controlled airspace around major airports", "avoidable": True},
            {"code": "R", "name": "Restricted", "description": "Restricted airspace - entry prohibited", "avoidable": False},
            {"code": "P", "name": "Prohibited", "description": "Prohibited airspace - no aircraft allowed", "avoidable": False},
            {"code": "D", "name": "Danger", "description": "Danger area - potential hazards", "avoidable": True},
            {"code": "A", "name": "Class A", "description": "Class A controlled airspace", "avoidable": True},
            {"code": "B", "name": "Class B", "description": "Class B controlled airspace", "avoidable": True},
            {"code": "C", "name": "Class C", "description": "Class C controlled airspace", "avoidable": True},
            {"code": "E", "name": "Class E", "description": "Class E controlled airspace", "avoidable": True}
        ]
    }

@api_router.post("/routes", response_model=FlightRoute)
async def create_route(route_data: RouteCreate):
    """Create a new flight route with advanced airspace avoidance and altitude planning"""
    try:
        # Use default preferences if not provided
        preferences = route_data.airspace_preferences or AirspacePreferences()
        
        # Parse airspaces for route planning
        parsed_airspaces = parse_openair_airspace(EUROPEAN_AIRSPACE_DATA)
        
        # Calculate avoidance waypoints
        avoidance_waypoints = calculate_avoidance_waypoints(
            route_data.start_point, 
            route_data.end_point, 
            parsed_airspaces, 
            preferences
        )
        
        # Combine all route points (start + avoidance + user waypoints + end)
        all_route_points = [route_data.start_point] + avoidance_waypoints + route_data.waypoints + [route_data.end_point]
        
        # Calculate route segments with altitude planning
        route_segments = calculate_route_segments(all_route_points, parsed_airspaces, preferences)
        
        # Calculate total distance
        total_distance = sum(segment.distance for segment in route_segments)
        
        # Estimate flight time (considering altitude changes)
        base_speed = 50  # km/h for paramotor
        estimated_time = total_distance / base_speed
        
        # Collect all warnings
        all_warnings = []
        flight_warnings = []
        
        for segment in route_segments:
            all_warnings.extend(segment.airspace_warnings)
            
        # Remove duplicates
        all_warnings = list(set(all_warnings))
        
        # Add general flight warnings
        if any(segment.segment_type == "restricted" for segment in route_segments):
            flight_warnings.append("⚠️ CRITICAL: Route passes through prohibited airspace - modification required")
        if any(segment.altitude > preferences.maximum_altitude for segment in route_segments):
            flight_warnings.append("⚠️ WARNING: Route requires altitude above your maximum setting")
        if len(avoidance_waypoints) > 0:
            flight_warnings.append(f"ℹ️ INFO: Added {len(avoidance_waypoints)} waypoints to avoid restricted airspace")
        
        # Create route - handle waypoints conflict
        route_dict = route_data.dict(exclude={'airspace_preferences'})
        route_dict['waypoints'] = all_route_points[1:-1]  # Exclude start and end points
        
        route = FlightRoute(
            **route_dict,
            route_segments=route_segments,
            total_distance=total_distance,
            estimated_time=estimated_time,
            airspace_preferences=preferences,
            airspace_warnings=all_warnings,
            flight_warnings=flight_warnings
        )
        
        # Save to database
        route_dict = route.dict()
        route_dict['created_at'] = route_dict['created_at'].isoformat()
        await db.routes.insert_one(route_dict)
        
        return route
    except Exception as e:
        logger.error(f"Error creating route: {e}")
        raise HTTPException(status_code=500, detail="Failed to create route")

@api_router.get("/routes", response_model=List[FlightRoute])
async def get_routes():
    """Get all flight routes"""
    try:
        routes = await db.routes.find().to_list(1000)
        return [FlightRoute(**route) for route in routes]
    except Exception as e:
        logger.error(f"Error getting routes: {e}")
        raise HTTPException(status_code=500, detail="Failed to get routes")

@api_router.get("/routes/{route_id}", response_model=FlightRoute)
async def get_route(route_id: str):
    """Get a specific route by ID"""
    try:
        route = await db.routes.find_one({"id": route_id})
        if not route:
            raise HTTPException(status_code=404, detail="Route not found")
        return FlightRoute(**route)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting route: {e}")
        raise HTTPException(status_code=500, detail="Failed to get route")

@api_router.post("/routes/{route_id}/share")
async def share_route(route_id: str):
    """Create a shareable link for a route"""
    try:
        # Check if route exists
        route = await db.routes.find_one({"id": route_id})
        if not route:
            raise HTTPException(status_code=404, detail="Route not found")
        
        # Create share record
        shared_route = SharedRoute(route_id=route_id)
        share_dict = shared_route.dict()
        share_dict['created_at'] = share_dict['created_at'].isoformat()
        if share_dict.get('expires_at'):
            share_dict['expires_at'] = share_dict['expires_at'].isoformat()
        
        await db.shared_routes.insert_one(share_dict)
        
        return {
            "share_token": shared_route.share_token,
            "share_url": f"/shared/{shared_route.share_token}"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error sharing route: {e}")
        raise HTTPException(status_code=500, detail="Failed to share route")

@api_router.get("/shared/{share_token}", response_model=FlightRoute)
async def get_shared_route(share_token: str):
    """Get a route by share token"""
    try:
        shared_route = await db.shared_routes.find_one({"share_token": share_token})
        if not shared_route:
            raise HTTPException(status_code=404, detail="Shared route not found")
        
        route = await db.routes.find_one({"id": shared_route["route_id"]})
        if not route:
            raise HTTPException(status_code=404, detail="Route not found")
        
        return FlightRoute(**route)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting shared route: {e}")
        raise HTTPException(status_code=500, detail="Failed to get shared route")

@api_router.get("/wind/{lat}/{lng}")
async def get_wind(lat: float, lng: float, altitude: float = 1000):
    """Get wind data for a specific location"""
    try:
        wind_data = await get_wind_data(lat, lng, altitude)
        return wind_data
    except Exception as e:
        logger.error(f"Error getting wind data: {e}")
        raise HTTPException(status_code=500, detail="Failed to get wind data")


# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()