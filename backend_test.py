#!/usr/bin/env python3
"""
Comprehensive Backend API Tests for Paramotorist Flight Planning App
Tests all critical endpoints with realistic aviation scenarios
"""

import requests
import json
import uuid
from datetime import datetime
import sys

# Backend URL from environment
BACKEND_URL = "https://sky-route-finder-1.preview.emergentagent.com/api"

class ParametoristAPITester:
    def __init__(self):
        self.base_url = BACKEND_URL
        self.session = requests.Session()
        self.test_results = []
        self.created_routes = []  # Track created routes for cleanup
        
    def log_test(self, test_name, success, details=""):
        """Log test results"""
        status = "✅ PASS" if success else "❌ FAIL"
        self.test_results.append({
            "test": test_name,
            "status": status,
            "details": details
        })
        print(f"{status}: {test_name}")
        if details:
            print(f"   Details: {details}")
    
    def test_health_check(self):
        """Test GET /api/ - Basic health check"""
        try:
            response = self.session.get(f"{self.base_url}/")
            if response.status_code == 200:
                data = response.json()
                if "message" in data and "Paramotorist" in data["message"]:
                    self.log_test("Health Check", True, f"Response: {data['message']}")
                    return True
                else:
                    self.log_test("Health Check", False, f"Unexpected response: {data}")
                    return False
            else:
                self.log_test("Health Check", False, f"Status: {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Health Check", False, f"Exception: {str(e)}")
            return False
    
    def test_airspaces_endpoint(self):
        """Test GET /api/airspaces - Returns OpenAir parsed airspace data"""
        try:
            response = self.session.get(f"{self.base_url}/airspaces")
            if response.status_code == 200:
                airspaces = response.json()
                
                # Verify it's a list
                if not isinstance(airspaces, list):
                    self.log_test("Airspaces Endpoint", False, "Response is not a list")
                    return False
                
                # Check for expected UK airspaces
                expected_airspaces = ["Bristol TMA", "London Prohibited Zone", "Manchester CTR"]
                found_airspaces = [a.get("name", "") for a in airspaces]
                
                missing = [name for name in expected_airspaces if name not in found_airspaces]
                if missing:
                    self.log_test("Airspaces Endpoint", False, f"Missing airspaces: {missing}")
                    return False
                
                # Verify airspace structure
                for airspace in airspaces:
                    required_fields = ["id", "name", "type", "coordinates"]
                    missing_fields = [field for field in required_fields if field not in airspace]
                    if missing_fields:
                        self.log_test("Airspaces Endpoint", False, f"Missing fields in airspace: {missing_fields}")
                        return False
                    
                    # Check coordinates structure
                    if not isinstance(airspace["coordinates"], list) or len(airspace["coordinates"]) == 0:
                        self.log_test("Airspaces Endpoint", False, "Invalid coordinates structure")
                        return False
                    
                    # Verify coordinate format
                    for coord in airspace["coordinates"]:
                        if not isinstance(coord, dict) or "lat" not in coord or "lng" not in coord:
                            self.log_test("Airspaces Endpoint", False, "Invalid coordinate format")
                            return False
                
                self.log_test("Airspaces Endpoint", True, f"Found {len(airspaces)} airspaces with proper structure")
                return True
            else:
                self.log_test("Airspaces Endpoint", False, f"Status: {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Airspaces Endpoint", False, f"Exception: {str(e)}")
            return False
    
    def test_airspace_types_endpoint(self):
        """Test GET /api/airspace-types - Returns available airspace types"""
        try:
            response = self.session.get(f"{self.base_url}/airspace-types")
            if response.status_code == 200:
                data = response.json()
                
                # Verify structure
                if "types" not in data:
                    self.log_test("Airspace Types Endpoint", False, "Missing 'types' field")
                    return False
                
                types = data["types"]
                if not isinstance(types, list):
                    self.log_test("Airspace Types Endpoint", False, "'types' is not a list")
                    return False
                
                # Check for expected types
                expected_codes = ["CTR", "TMA", "R", "P", "D"]
                found_codes = [t.get("code") for t in types]
                
                missing = [code for code in expected_codes if code not in found_codes]
                if missing:
                    self.log_test("Airspace Types Endpoint", False, f"Missing airspace types: {missing}")
                    return False
                
                # Verify type structure
                for airspace_type in types:
                    required_fields = ["code", "name", "description"]
                    missing_fields = [field for field in required_fields if field not in airspace_type]
                    if missing_fields:
                        self.log_test("Airspace Types Endpoint", False, f"Missing fields: {missing_fields}")
                        return False
                
                self.log_test("Airspace Types Endpoint", True, f"Found {len(types)} airspace types")
                return True
            else:
                self.log_test("Airspace Types Endpoint", False, f"Status: {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Airspace Types Endpoint", False, f"Exception: {str(e)}")
            return False
    
    def test_route_creation(self):
        """Test POST /api/routes - Create new flight route with realistic coordinates"""
        try:
            # Bristol to Bath route (realistic paramotor flight)
            route_data = {
                "name": "Bristol to Bath Scenic Route",
                "description": "Beautiful countryside flight avoiding restricted zones",
                "start_point": {
                    "lat": 51.4545,  # Bristol
                    "lng": -2.5879,
                    "elevation": 100,
                    "waypoint_name": "Bristol Launch Site"
                },
                "end_point": {
                    "lat": 51.3758,  # Bath
                    "lng": -2.3599,
                    "elevation": 150,
                    "waypoint_name": "Bath Landing Field"
                },
                "waypoints": [
                    {
                        "lat": 51.4200,
                        "lng": -2.4500,
                        "elevation": 120,
                        "waypoint_name": "Countryside Checkpoint"
                    }
                ],
                "wind_consideration": True
            }
            
            response = self.session.post(
                f"{self.base_url}/routes",
                json=route_data,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                route = response.json()
                
                # Verify response structure
                required_fields = ["id", "name", "start_point", "end_point", "total_distance", "estimated_time"]
                missing_fields = [field for field in required_fields if field not in route]
                if missing_fields:
                    self.log_test("Route Creation", False, f"Missing fields: {missing_fields}")
                    return False
                
                # Verify distance calculation (Bristol to Bath ~20km)
                distance = route.get("total_distance", 0)
                if not (15 <= distance <= 25):  # Allow some variance
                    self.log_test("Route Creation", False, f"Unexpected distance: {distance}km (expected ~20km)")
                    return False
                
                # Verify time calculation (should be distance/50 for paramotor)
                expected_time = distance / 50
                actual_time = route.get("estimated_time", 0)
                if abs(actual_time - expected_time) > 0.1:
                    self.log_test("Route Creation", False, f"Time calculation error: {actual_time}h vs {expected_time}h")
                    return False
                
                # Store route ID for later tests
                self.created_routes.append(route["id"])
                
                self.log_test("Route Creation", True, f"Route created: {distance:.1f}km, {actual_time:.2f}h")
                return True
            else:
                self.log_test("Route Creation", False, f"Status: {response.status_code}, Response: {response.text}")
                return False
        except Exception as e:
            self.log_test("Route Creation", False, f"Exception: {str(e)}")
            return False
    
    def test_route_listing(self):
        """Test GET /api/routes - List all saved routes"""
        try:
            response = self.session.get(f"{self.base_url}/routes")
            if response.status_code == 200:
                routes = response.json()
                
                if not isinstance(routes, list):
                    self.log_test("Route Listing", False, "Response is not a list")
                    return False
                
                # Should have at least the route we created
                if len(routes) == 0:
                    self.log_test("Route Listing", False, "No routes found")
                    return False
                
                # Verify route structure
                for route in routes:
                    required_fields = ["id", "name", "start_point", "end_point"]
                    missing_fields = [field for field in required_fields if field not in route]
                    if missing_fields:
                        self.log_test("Route Listing", False, f"Missing fields: {missing_fields}")
                        return False
                
                self.log_test("Route Listing", True, f"Found {len(routes)} routes")
                return True
            else:
                self.log_test("Route Listing", False, f"Status: {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Route Listing", False, f"Exception: {str(e)}")
            return False
    
    def test_route_retrieval(self):
        """Test GET /api/routes/{route_id} - Get specific route details"""
        if not self.created_routes:
            self.log_test("Route Retrieval", False, "No routes available for testing")
            return False
        
        try:
            route_id = self.created_routes[0]
            response = self.session.get(f"{self.base_url}/routes/{route_id}")
            
            if response.status_code == 200:
                route = response.json()
                
                # Verify it's the correct route
                if route.get("id") != route_id:
                    self.log_test("Route Retrieval", False, f"Wrong route returned: {route.get('id')} vs {route_id}")
                    return False
                
                # Verify complete structure
                required_fields = ["id", "name", "start_point", "end_point", "total_distance", "estimated_time"]
                missing_fields = [field for field in required_fields if field not in route]
                if missing_fields:
                    self.log_test("Route Retrieval", False, f"Missing fields: {missing_fields}")
                    return False
                
                self.log_test("Route Retrieval", True, f"Retrieved route: {route['name']}")
                return True
            elif response.status_code == 404:
                self.log_test("Route Retrieval", False, "Route not found (404)")
                return False
            else:
                self.log_test("Route Retrieval", False, f"Status: {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Route Retrieval", False, f"Exception: {str(e)}")
            return False
    
    def test_route_sharing(self):
        """Test POST /api/routes/{route_id}/share - Create shareable link"""
        if not self.created_routes:
            self.log_test("Route Sharing", False, "No routes available for testing")
            return False
        
        try:
            route_id = self.created_routes[0]
            response = self.session.post(f"{self.base_url}/routes/{route_id}/share")
            
            if response.status_code == 200:
                share_data = response.json()
                
                # Verify response structure
                required_fields = ["share_token", "share_url"]
                missing_fields = [field for field in required_fields if field not in share_data]
                if missing_fields:
                    self.log_test("Route Sharing", False, f"Missing fields: {missing_fields}")
                    return False
                
                # Verify token is UUID format
                share_token = share_data["share_token"]
                try:
                    uuid.UUID(share_token)
                except ValueError:
                    self.log_test("Route Sharing", False, f"Invalid UUID format: {share_token}")
                    return False
                
                # Store token for shared route test
                self.share_token = share_token
                
                self.log_test("Route Sharing", True, f"Share token created: {share_token[:8]}...")
                return True
            else:
                self.log_test("Route Sharing", False, f"Status: {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Route Sharing", False, f"Exception: {str(e)}")
            return False
    
    def test_shared_route_access(self):
        """Test GET /api/shared/{share_token} - Access shared route"""
        if not hasattr(self, 'share_token'):
            self.log_test("Shared Route Access", False, "No share token available")
            return False
        
        try:
            response = self.session.get(f"{self.base_url}/shared/{self.share_token}")
            
            if response.status_code == 200:
                route = response.json()
                
                # Verify it's a complete route
                required_fields = ["id", "name", "start_point", "end_point"]
                missing_fields = [field for field in required_fields if field not in route]
                if missing_fields:
                    self.log_test("Shared Route Access", False, f"Missing fields: {missing_fields}")
                    return False
                
                # Should match our created route
                if route["id"] not in self.created_routes:
                    self.log_test("Shared Route Access", False, "Shared route doesn't match created route")
                    return False
                
                self.log_test("Shared Route Access", True, f"Accessed shared route: {route['name']}")
                return True
            elif response.status_code == 404:
                self.log_test("Shared Route Access", False, "Shared route not found (404)")
                return False
            else:
                self.log_test("Shared Route Access", False, f"Status: {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Shared Route Access", False, f"Exception: {str(e)}")
            return False
    
    def test_wind_data(self):
        """Test GET /api/wind/{lat}/{lng} - Get wind data for location"""
        try:
            # Test with Bristol coordinates
            lat, lng = 51.4545, -2.5879
            response = self.session.get(f"{self.base_url}/wind/{lat}/{lng}")
            
            if response.status_code == 200:
                wind_data = response.json()
                
                # Verify structure
                required_fields = ["lat", "lng", "speed", "direction"]
                missing_fields = [field for field in required_fields if field not in wind_data]
                if missing_fields:
                    self.log_test("Wind Data", False, f"Missing fields: {missing_fields}")
                    return False
                
                # Verify data ranges
                speed = wind_data.get("speed", 0)
                direction = wind_data.get("direction", 0)
                
                if not (0 <= speed <= 50):  # Reasonable wind speed range
                    self.log_test("Wind Data", False, f"Unrealistic wind speed: {speed} m/s")
                    return False
                
                if not (0 <= direction <= 360):  # Valid direction range
                    self.log_test("Wind Data", False, f"Invalid wind direction: {direction}°")
                    return False
                
                # Verify coordinates match
                if abs(wind_data["lat"] - lat) > 0.001 or abs(wind_data["lng"] - lng) > 0.001:
                    self.log_test("Wind Data", False, "Coordinates don't match request")
                    return False
                
                self.log_test("Wind Data", True, f"Wind: {speed:.1f}m/s @ {direction:.0f}°")
                return True
            else:
                self.log_test("Wind Data", False, f"Status: {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Wind Data", False, f"Exception: {str(e)}")
            return False
    
    def test_airspace_conflict_detection(self):
        """Test route creation with airspace conflict detection"""
        try:
            # Create route that should trigger airspace warnings (near London Prohibited Zone)
            route_data = {
                "name": "London Area Test Route",
                "description": "Route to test airspace conflict detection",
                "start_point": {
                    "lat": 51.47,  # Very close to London Prohibited Zone center
                    "lng": -0.45,
                    "elevation": 100
                },
                "end_point": {
                    "lat": 51.48,
                    "lng": -0.44,
                    "elevation": 100
                },
                "waypoints": [],
                "wind_consideration": True
            }
            
            response = self.session.post(
                f"{self.base_url}/routes",
                json=route_data,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                route = response.json()
                
                # Should have airspace warnings
                warnings = route.get("airspace_warnings", [])
                if not warnings:
                    self.log_test("Airspace Conflict Detection", False, "No airspace warnings generated for route near prohibited zone")
                    return False
                
                # Check if warning mentions prohibited zone
                has_prohibited_warning = any("Prohibited" in warning or "P" in warning for warning in warnings)
                if not has_prohibited_warning:
                    self.log_test("Airspace Conflict Detection", False, f"Expected prohibited zone warning, got: {warnings}")
                    return False
                
                self.created_routes.append(route["id"])
                self.log_test("Airspace Conflict Detection", True, f"Detected conflicts: {len(warnings)} warnings")
                return True
            else:
                self.log_test("Airspace Conflict Detection", False, f"Status: {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Airspace Conflict Detection", False, f"Exception: {str(e)}")
            return False
    
    def run_all_tests(self):
        """Run comprehensive test suite"""
        print("🚁 Starting Paramotorist Flight Planning API Tests")
        print(f"Backend URL: {self.base_url}")
        print("=" * 60)
        
        # Test in logical order
        tests = [
            self.test_health_check,
            self.test_airspaces_endpoint,
            self.test_airspace_types_endpoint,
            self.test_route_creation,
            self.test_route_listing,
            self.test_route_retrieval,
            self.test_route_sharing,
            self.test_shared_route_access,
            self.test_wind_data,
            self.test_airspace_conflict_detection
        ]
        
        passed = 0
        total = len(tests)
        
        for test in tests:
            try:
                if test():
                    passed += 1
            except Exception as e:
                print(f"❌ FAIL: {test.__name__} - Unexpected error: {str(e)}")
        
        print("=" * 60)
        print(f"🏁 Test Results: {passed}/{total} tests passed")
        
        if passed == total:
            print("✅ All tests passed! Backend API is working correctly.")
            return True
        else:
            print(f"❌ {total - passed} tests failed. Check the details above.")
            return False

def main():
    """Main test execution"""
    tester = ParametoristAPITester()
    success = tester.run_all_tests()
    
    # Print summary for easy parsing
    print("\n" + "=" * 60)
    print("DETAILED TEST RESULTS:")
    for result in tester.test_results:
        print(f"{result['status']}: {result['test']}")
        if result['details']:
            print(f"    {result['details']}")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())