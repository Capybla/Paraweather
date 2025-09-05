from fastapi import FastAPI, APIRouter, HTTPException, Depends
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timezone
import httpx
import asyncio
import json
import re
import math


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

class Airspace(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    type: str  # CTR, TMA, R, P, D, etc.
    coordinates: List[Coordinate]
    floor: Optional[str] = None
    ceiling: Optional[str] = None
    frequency: Optional[str] = None
    description: Optional[str] = None

class RoutePoint(BaseModel):
    lat: float
    lng: float
    elevation: Optional[float] = None
    waypoint_name: Optional[str] = None

class FlightRoute(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: Optional[str] = None
    start_point: RoutePoint
    end_point: RoutePoint
    waypoints: List[RoutePoint] = []
    total_distance: Optional[float] = None
    estimated_time: Optional[float] = None
    wind_consideration: bool = True
    airspace_warnings: List[str] = []
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class RouteCreate(BaseModel):
    name: str
    description: Optional[str] = None
    start_point: RoutePoint
    end_point: RoutePoint
    waypoints: List[RoutePoint] = []
    wind_consideration: bool = True

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

def parse_openair_airspace(openair_content: str) -> List[Dict]:
    """Parse OpenAir format airspace data"""
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
                airspaces.append(current_airspace)
            current_airspace = {'type': line[3:].strip()}
            coordinates = []
        elif line.startswith('AN '):  # Airspace Name
            current_airspace['name'] = line[3:].strip()
        elif line.startswith('AL '):  # Lower limit
            current_airspace['floor'] = line[3:].strip()
        elif line.startswith('AH '):  # Upper limit
            current_airspace['ceiling'] = line[3:].strip()
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
        airspaces.append(current_airspace)
    
    return airspaces

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

def check_airspace_conflicts(route_points: List[RoutePoint], airspaces: List[Dict]) -> List[str]:
    """Check if route conflicts with restricted airspaces"""
    warnings = []
    
    for airspace in airspaces:
        if airspace.get('type') in ['R', 'P', 'D']:  # Restricted, Prohibited, Danger
            # Simple point-in-polygon check (simplified)
            for point in route_points:
                # This is a simplified check - in reality you'd need proper polygon intersection
                if airspace.get('coordinates'):
                    first_coord = airspace['coordinates'][0]
                    distance = haversine_distance(
                        point.lat, point.lng,
                        first_coord['lat'], first_coord['lng']
                    )
                    if distance < 5:  # Within 5km
                        warnings.append(f"Route passes near {airspace.get('type')} airspace: {airspace.get('name', 'Unknown')}")
    
    return warnings


# Sample OpenAir data for demonstration
SAMPLE_OPENAIR_DATA = """
* Sample airspace data for paramotoring
AC R
AN Bristol TMA
AL SFC
AH FL195
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
DP 53:21:00 N 002:16:00 W
DP 53:25:00 N 002:10:00 W
DP 53:23:00 N 002:05:00 W
DP 53:19:00 N 002:11:00 W
DP 53:21:00 N 002:16:00 W
"""


# Routes
@api_router.get("/")
async def root():
    return {"message": "Paramotorist Flight Planning API"}

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
                ceiling=airspace_data.get('ceiling')
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
            {"code": "CTR", "name": "Control Zone", "description": "Controlled airspace around airports"},
            {"code": "TMA", "name": "Terminal Control Area", "description": "Controlled airspace around major airports"},
            {"code": "R", "name": "Restricted", "description": "Restricted airspace - entry prohibited"},
            {"code": "P", "name": "Prohibited", "description": "Prohibited airspace - no aircraft allowed"},
            {"code": "D", "name": "Danger", "description": "Danger area - potential hazards"},
            {"code": "A", "name": "Class A", "description": "Class A controlled airspace"},
            {"code": "B", "name": "Class B", "description": "Class B controlled airspace"},
            {"code": "C", "name": "Class C", "description": "Class C controlled airspace"},
            {"code": "D", "name": "Class D", "description": "Class D controlled airspace"},
            {"code": "E", "name": "Class E", "description": "Class E controlled airspace"}
        ]
    }

@api_router.post("/routes", response_model=FlightRoute)
async def create_route(route_data: RouteCreate):
    """Create a new flight route with airspace and wind considerations"""
    try:
        # Calculate distance
        distance = haversine_distance(
            route_data.start_point.lat, route_data.start_point.lng,
            route_data.end_point.lat, route_data.end_point.lng
        )
        
        # Get all route points for airspace checking
        all_points = [route_data.start_point] + route_data.waypoints + [route_data.end_point]
        
        # Parse airspaces for conflict checking
        parsed_airspaces = parse_openair_airspace(SAMPLE_OPENAIR_DATA)
        airspace_warnings = check_airspace_conflicts(all_points, parsed_airspaces)
        
        # Estimate flight time (assuming 50 km/h average speed for paramotor)
        estimated_time = distance / 50  # hours
        
        # Create route
        route = FlightRoute(
            **route_data.dict(),
            total_distance=distance,
            estimated_time=estimated_time,
            airspace_warnings=airspace_warnings
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