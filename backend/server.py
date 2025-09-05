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

def calculate_avoidance_waypoints(start: RoutePoint, end: RoutePoint, airspaces: List[Dict], preferences: AirspacePreferences) -> List[RoutePoint]:
    """Calculate intermediate waypoints to avoid restricted airspaces"""
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
    
    # Simple avoidance algorithm - if direct line intersects restricted airspace, add waypoints
    direct_line = [(start.lat, start.lng), (end.lat, end.lng)]
    
    for airspace in restricted_airspaces:
        if line_intersects_polygon(direct_line, airspace.get('coordinates', [])):
            # Calculate avoidance waypoint
            center_lat = sum(c['lat'] for c in airspace['coordinates']) / len(airspace['coordinates'])
            center_lng = sum(c['lng'] for c in airspace['coordinates']) / len(airspace['coordinates'])
            
            # Calculate bearing from start to end, then offset perpendicular
            route_bearing = bearing(start.lat, start.lng, end.lat, end.lng)
            avoidance_bearing = (route_bearing + 90) % 360  # 90 degrees offset
            
            # Calculate avoidance point 10km away from airspace center
            avoidance_distance = 10  # km
            lat_offset = avoidance_distance * math.cos(math.radians(avoidance_bearing)) / 111.32
            lng_offset = avoidance_distance * math.sin(math.radians(avoidance_bearing)) / (111.32 * math.cos(math.radians(center_lat)))
            
            waypoint = RoutePoint(
                lat=center_lat + lat_offset,
                lng=center_lng + lng_offset,
                altitude=preferences.preferred_altitude,
                waypoint_name=f"Avoid {airspace.get('name', 'Unknown')}"
            )
            waypoints.append(waypoint)
    
    return waypoints

def calculate_route_segments(route_points: List[RoutePoint], airspaces: List[Dict], preferences: AirspacePreferences) -> List[RouteSegment]:
    """Calculate route segments with altitude planning and airspace analysis"""
    segments = []
    
    for i in range(len(route_points) - 1):
        start_point = route_points[i]
        end_point = route_points[i + 1]
        
        # Calculate segment distance
        segment_distance = haversine_distance(start_point.lat, start_point.lng, end_point.lat, end_point.lng)
        
        # Determine altitude for this segment
        segment_altitude = preferences.preferred_altitude  # Default altitude
        
        # Check for airspace conflicts along this segment
        segment_warnings = []
        segment_type = "safe"
        
        # Sample points along the segment for airspace checking
        num_samples = max(5, int(segment_distance))  # More samples for longer segments
        for sample in range(num_samples + 1):
            ratio = sample / num_samples if num_samples > 0 else 0
            sample_lat = start_point.lat + ratio * (end_point.lat - start_point.lat)
            sample_lng = start_point.lng + ratio * (end_point.lng - start_point.lng)
            
            # Check each airspace
            for airspace in airspaces:
                if point_in_polygon((sample_lat, sample_lng), airspace.get('coordinates', [])):
                    airspace_type = airspace.get('type', 'Unknown')
                    airspace_name = airspace.get('name', 'Unknown airspace')
                    
                    # Check if we can fly above/below this airspace
                    floor = airspace.get('floor_meters', 0)
                    ceiling = airspace.get('ceiling_meters', 10000)  # Default high ceiling
                    
                    if airspace_type in ['R', 'P']:
                        segment_type = "restricted"
                        segment_warnings.append(f"PROHIBITED: {airspace_name} - Entry forbidden")
                    elif airspace_type in ['CTR', 'TMA'] and (preferences.avoid_ctr or preferences.avoid_tma):
                        if segment_altitude > floor and segment_altitude < ceiling:
                            # Try to fly above
                            if ceiling < preferences.maximum_altitude:
                                segment_altitude = ceiling + 50  # 50m above
                                segment_type = "caution"
                                segment_warnings.append(f"ALTITUDE ADJUSTED: Flying at {segment_altitude}m to clear {airspace_name}")
                            else:
                                segment_type = "avoid"
                                segment_warnings.append(f"WARNING: Cannot clear {airspace_name} at safe altitude")
                    elif airspace_type == 'D':
                        segment_type = "caution"
                        segment_warnings.append(f"DANGER AREA: {airspace_name} - Monitor NOTAMs")
                    
                    # Add requirements
                    requirements = get_airspace_requirements(airspace_type)
                    for req in requirements:
                        if req not in segment_warnings:
                            segment_warnings.append(f"{airspace_name}: {req}")
        
        # Create segment
        segment = RouteSegment(
            start=Coordinate(lat=start_point.lat, lng=start_point.lng),
            end=Coordinate(lat=end_point.lat, lng=end_point.lng),
            altitude=segment_altitude,
            segment_type=segment_type,
            airspace_warnings=segment_warnings,
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


# Sample OpenAir data for demonstration with enhanced data
SAMPLE_OPENAIR_DATA = """
* Enhanced airspace data for paramotoring with altitude information
AC R
AN Bristol TMA
AL SFC
AH FL195
AF 118.850
DP 51:23:00 N 002:35:00 W
DP 51:30:00 N 002:25:00 W
DP 51:25:00 N 002:15:00 W
DP 51:18:00 N 002:20:00 W
DP 51:23:00 N 002:35:00 W

AC P
AN London Prohibited Zone
AL SFC
AH 2500FT
DP 51:28:00 N 000:27:00 W
DP 51:32:00 N 000:20:00 W
DP 51:30:00 N 000:15:00 W
DP 51:26:00 N 000:22:00 W
DP 51:28:00 N 000:27:00 W

AC CTR
AN Manchester CTR
AL SFC
AH 3500FT
AF 118.620
DP 53:21:00 N 002:16:00 W
DP 53:25:00 N 002:10:00 W
DP 53:23:00 N 002:05:00 W
DP 53:19:00 N 002:11:00 W
DP 53:21:00 N 002:16:00 W

AC TMA
AN Birmingham TMA
AL 1500FT
AH FL245
AF 121.200
DP 52:28:00 N 001:50:00 W
DP 52:35:00 N 001:40:00 W
DP 52:32:00 N 001:30:00 W
DP 52:25:00 N 001:35:00 W
DP 52:28:00 N 001:50:00 W

AC D
AN Danger Area EG D323
AL SFC
AH 8000FT
DP 52:10:00 N 002:00:00 W
DP 52:15:00 N 001:55:00 W
DP 52:12:00 N 001:50:00 W
DP 52:07:00 N 001:55:00 W
DP 52:10:00 N 002:00:00 W
"""


# Routes
@api_router.get("/")
async def root():
    return {"message": "Paramotorist Flight Planning API with Advanced Airspace Avoidance"}

@api_router.get("/airspaces", response_model=List[Airspace])
async def get_airspaces(type_filter: Optional[str] = None):
    """Get airspace data, optionally filtered by type"""
    try:
        # Parse sample OpenAir data
        parsed_airspaces = parse_openair_airspace(SAMPLE_OPENAIR_DATA)
        
        airspaces = []
        for airspace_data in parsed_airspaces:
            if type_filter and airspace_data.get('type') != type_filter:
                continue
                
            airspace = Airspace(
                name=airspace_data.get('name', 'Unknown'),
                type=airspace_data.get('type', 'Unknown'),
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
        parsed_airspaces = parse_openair_airspace(SAMPLE_OPENAIR_DATA)
        
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