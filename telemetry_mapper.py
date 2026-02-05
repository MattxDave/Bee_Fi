"""
TelemetryBridge Mapper
Converts external telemetry JSON to Bee Swarm model objects.

Usage:
    from telemetry_mapper import load_telemetry
    
    bees, flowers, metadata = load_telemetry("telemetrybridge.json", {
        "grid_size": 30,
        "km_per_unit": 500,
    })
"""

import json
import math
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

from bee_state import Bee, Flower


class TelemetryMapper:
    """Maps telemetrybridge.json data to Bee/Flower objects."""

    def __init__(self, config: Optional[dict] = None):
        config = config or {}
        self.grid_size = config.get("grid_size", 30)
        # ECI coordinates are in METERS, typical LEO orbit ~7000km = 7,000,000m
        # Scale so orbits fit nicely in grid: 7,000,000m / 500,000 = 14 grid units from center
        self.meters_per_unit = config.get("meters_per_unit", 500_000)  # 500km per grid unit
        self.km_per_unit = self.meters_per_unit / 1000  # For backwards compat
        # Time: 1 step = 1 second for responsive control
        self.seconds_per_step = config.get("seconds_per_step", 1.0)
        self.battery_capacity = config.get("battery_capacity", 100.0)
        self.pollen_capacity = config.get("pollen_capacity", 10.0)
        self.episode_start: float = 0.0
        # Max steps per episode (for deadline clamping)
        self.max_steps = config.get("max_steps", 500)

        # ID mappings: string → int
        self.sat_id_map: Dict[str, int] = {}
        self.task_id_map: Dict[str, int] = {}
        
        # Deadline normalization: map all deadlines to fit within episode
        self.deadline_min: float = 0
        self.deadline_max: float = 0
        self.normalize_deadlines: bool = config.get("normalize_deadlines", True)
        
        # Coordinate auto-scaling: fit all positions within grid
        self.coord_min_x: float = 0
        self.coord_max_x: float = 0
        self.coord_min_y: float = 0
        self.coord_max_y: float = 0
        self.auto_scale_coords: bool = config.get("auto_scale_coords", True)
    # ─────────────────────────────────────────────────────────────────────────
    # Loading
    # ─────────────────────────────────────────────────────────────────────────

    def load(self, filepath: str) -> dict:
        """Load and parse telemetry JSON."""
        with open(filepath, "r") as f:
            content = f.read()

        # Handle files with extra text after JSON (e.g., "Failed satellites: ...")
        # Find the last closing brace that ends the JSON object
        brace_count = 0
        json_end = 0
        in_string = False
        escape_next = False

        for i, char in enumerate(content):
            if escape_next:
                escape_next = False
                continue
            if char == "\\":
                escape_next = True
                continue
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    json_end = i + 1
                    break

        if json_end > 0:
            content = content[:json_end]

        return json.loads(content)

    # ─────────────────────────────────────────────────────────────────────────
    # ID Mapping
    # ─────────────────────────────────────────────────────────────────────────

    def build_sat_id_map(self, satellites: List[dict]) -> Dict[str, int]:
        """Map string satellite IDs to integers."""
        self.sat_id_map = {}
        for idx, sat in enumerate(satellites):
            sat_id = sat.get("satellite_id", f"SAT-{idx}")
            self.sat_id_map[sat_id] = idx
        return self.sat_id_map

    def build_task_id_map(self, all_tasks: List[dict]) -> Dict[str, int]:
        """Map string task IDs to integers."""
        self.task_id_map = {}
        seen = set()
        idx = 0
        for task in all_tasks:
            task_id = task.get("task_id", f"TASK-{idx}")
            if task_id not in seen:
                self.task_id_map[task_id] = idx
                seen.add(task_id)
                idx += 1
        return self.task_id_map

    # ─────────────────────────────────────────────────────────────────────────
    # Coordinate Transforms
    # ─────────────────────────────────────────────────────────────────────────

    def collect_all_coordinates(self, data: dict) -> List[Tuple[float, float]]:
        """Collect all ECI x,y coordinates from satellites and tasks."""
        coords = []
        
        # From controller satellites
        controller = data.get("telemetry-bridge", {}).get("controller", {})
        for sat_id, sat_data in controller.items():
            if not isinstance(sat_data, dict):
                continue
            pos = sat_data.get("position_eci", {})
            if pos:
                x = pos.get("x", 0) or 0
                y = pos.get("y", 0) or 0
                coords.append((x, y))
        
        # From tasks
        all_tasks = self.extract_all_tasks(data)
        for task in all_tasks:
            loc = task.get("location_task", {})
            if loc and "x" in loc:
                x = loc.get("x", 0) or 0
                y = loc.get("y", 0) or 0
                coords.append((x, y))
        
        return coords
    
    def setup_coordinate_scaling(self, data: dict):
        """Calculate scaling to fit all coordinates within grid."""
        coords = self.collect_all_coordinates(data)
        if not coords:
            return
        
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        
        self.coord_min_x = min(xs)
        self.coord_max_x = max(xs)
        self.coord_min_y = min(ys)
        self.coord_max_y = max(ys)
        
        # Add some margin (5% on each side)
        margin = 0.05 * self.grid_size
        self.grid_margin = margin
    
    def eci_to_grid(self, eci_meters: dict) -> Tuple[float, float, float]:
        """
        Convert ECI coordinates (in METERS) to grid units.
        
        If auto_scale_coords is True, scales all coordinates to fit within grid.
        Otherwise uses fixed meters_per_unit scaling centered on grid.
        """
        x_m = eci_meters.get("x", 0) or 0
        y_m = eci_meters.get("y", 0) or 0
        z_m = eci_meters.get("z", 0) or 0
        
        if self.auto_scale_coords and self.coord_max_x != self.coord_min_x:
            # Auto-scale: map [min, max] → [margin, grid_size - margin]
            margin = getattr(self, 'grid_margin', 2.0)
            range_x = self.coord_max_x - self.coord_min_x
            range_y = self.coord_max_y - self.coord_min_y
            
            # Use the larger range to maintain aspect ratio, or scale independently
            usable = self.grid_size - 2 * margin
            
            if range_x > 0:
                x = margin + ((x_m - self.coord_min_x) / range_x) * usable
            else:
                x = self.grid_size / 2.0
            
            if range_y > 0:
                y = margin + ((y_m - self.coord_min_y) / range_y) * usable
            else:
                y = self.grid_size / 2.0
            
            # Z is not constrained to grid, just scale
            max_range = max(range_x, range_y, 1)
            z = (z_m / max_range) * usable
        else:
            # Fixed scaling centered on grid
            center = self.grid_size / 2.0
            x = (x_m / self.meters_per_unit) + center
            y = (y_m / self.meters_per_unit) + center
            z = z_m / self.meters_per_unit
        
        # Clamp to grid bounds for x, y
        x = max(0, min(self.grid_size, x))
        y = max(0, min(self.grid_size, y))

        return (x, y, z)

    def latlon_to_grid(self, lat: float, lon: float) -> Tuple[float, float]:
        """Convert lat/lon to grid coordinates."""
        # Map lat [-90, 90] → [0, grid_size]
        # Map lon [-180, 180] → [0, grid_size]
        x = ((lon + 180) / 360) * self.grid_size
        y = ((lat + 90) / 180) * self.grid_size
        return (x, y)

    def velocity_eci_to_grid(self, vel_m_s: dict) -> Tuple[float, float, float]:
        """Convert ECI velocity (m/s) to grid units per step."""
        # Velocities in telemetry are m/s (e.g., 6284 m/s = 6.284 km/s)
        vx = vel_m_s.get("vx", 0) or 0
        vy = vel_m_s.get("vy", 0) or 0
        vz = vel_m_s.get("vz", 0) or 0

        # Convert m/s to grid_units/step
        # velocity_grid = velocity_m * seconds_per_step / meters_per_unit
        factor = self.seconds_per_step / self.meters_per_unit

        return (vx * factor, vy * factor, vz * factor)

    # ─────────────────────────────────────────────────────────────────────────
    # Time Conversion
    # ─────────────────────────────────────────────────────────────────────────

    def set_episode_start(self, unix_timestamp: float):
        """Set the episode start time for step calculations."""
        self.episode_start = unix_timestamp

    def collect_all_deadlines(self, data: dict) -> List[float]:
        """Collect all deadline timestamps from the telemetry data."""
        deadlines = []
        
        # From satellites' assigned_tasks
        controller = data.get("telemetry-bridge", {}).get("controller", {})
        for sat in controller.get("satellites", []):
            for task in sat.get("assigned_tasks", []):
                dl = task.get("Deadline") or task.get("deadline")
                if dl and isinstance(dl, (int, float)) and dl > 0:
                    deadlines.append(float(dl))
        
        # From gossiper's unassignedTasks
        gossiper = data.get("telemetry-bridge", {}).get("gossiper", {})
        for sat_id, gdata in gossiper.items():
            if isinstance(gdata, dict):
                for task in gdata.get("unassignedTasks", []):
                    dl = task.get("Deadline") or task.get("deadline")
                    if dl and isinstance(dl, (int, float)) and dl > 0:
                        deadlines.append(float(dl))
        
        return deadlines

    def setup_deadline_normalization(self, data: dict):
        """Setup deadline normalization based on all deadlines in the data."""
        deadlines = self.collect_all_deadlines(data)
        
        if not deadlines:
            self.normalize_deadlines = False
            return
        
        self.deadline_min = min(deadlines)
        self.deadline_max = max(deadlines)
        
        # If all deadlines are the same, spread them out
        if self.deadline_max == self.deadline_min:
            self.deadline_max = self.deadline_min + 1000

    def timestamp_to_step(self, timestamp: Any) -> int:
        """Convert Unix timestamp to step count, normalizing to fit episode."""
        if timestamp is None:
            return self.max_steps  # No deadline = end of episode

        if isinstance(timestamp, str):
            try:
                ts_clean = timestamp.replace("Z", "+00:00")
                dt = datetime.fromisoformat(ts_clean)
                unix_ts = dt.timestamp()
            except ValueError:
                return self.max_steps
        else:
            unix_ts = float(timestamp)

        # Normalize deadlines to fit within episode (steps 10 to max_steps)
        if self.normalize_deadlines and self.deadline_max > self.deadline_min:
            # Map [deadline_min, deadline_max] → [10, max_steps]
            # Leave some early steps (0-10) for setup
            normalized = (unix_ts - self.deadline_min) / (self.deadline_max - self.deadline_min)
            step = int(10 + normalized * (self.max_steps - 10))
            return max(10, min(step, self.max_steps))
        
        # Fallback: use raw time difference
        if unix_ts < self.episode_start:
            return 10  # Past deadline → early in episode
        
        elapsed = unix_ts - self.episode_start
        step = int(elapsed / self.seconds_per_step)
        return max(0, min(step, self.max_steps))

    # ─────────────────────────────────────────────────────────────────────────
    # Satellite → Bee Mapping
    # ─────────────────────────────────────────────────────────────────────────

    def map_satellite_to_bee(self, sat: dict, failed_list: List[str]) -> Bee:
        """Convert a satellite dict to a Bee object."""
        sat_id = sat.get("satellite_id", "UNKNOWN")
        bee_id = self.sat_id_map.get(sat_id, 0)

        # Position
        eci = sat.get("position_eci", {})
        if not eci:
            # Fallback to orbital_parameters if position_eci empty
            orbital = sat.get("orbital_parameters", {})
            eci = {
                "x": orbital.get("x_eci", 0),
                "y": orbital.get("y_eci", 0),
                "z": orbital.get("z", 0),
            }
        fx, fy, fz = self.eci_to_grid(eci)

        # Velocity
        vel = sat.get("velocity_eci", {})
        if not vel:
            orbital = sat.get("orbital_parameters", {})
            vel = {
                "vx": orbital.get("vx_eci", 0),
                "vy": orbital.get("vy_eci", 0),
                "vz": orbital.get("vz", 0),
            }
        vx, vy, vz = self.velocity_eci_to_grid(vel)

        # Status
        is_failed = sat_id in failed_list
        status = sat.get("status", "Active")
        comm_status = sat.get("communication_status", "Online")
        terminated = is_failed or status == "Failed" or comm_status == "Offline"

        # Battery
        battery = sat.get("battery_level", 100.0)
        if battery is None:
            battery = 100.0
        if terminated:
            battery = 0.0

        # Orbital parameters (if available)
        orbital = sat.get("orbital_parameters", {})
        semi_major_axis = orbital.get("semi_major_axis", 7000.0)
        eccentricity = orbital.get("eccentricity", 0.0)
        inclination = orbital.get("inclination", 0.0)

        # Create Bee with required parameters
        bee = Bee(
            bee_id=bee_id,
            grid_size=self.grid_size,
            capacity=self.pollen_capacity,
            battery_capacity=self.battery_capacity,
            initial_battery=battery,
        )

        # Set 3D position
        bee.fx = fx
        bee.fy = fy
        bee.fz = fz

        # Set velocity (if Bee supports it)
        if hasattr(bee, "vx"):
            bee.vx = vx
            bee.vy = vy
            bee.vz = vz

        # Set battery
        bee.battery = battery

        # Set orbital parameters
        if hasattr(bee, "a"):
            bee.a = semi_major_axis / self.km_per_unit
        if hasattr(bee, "e"):
            bee.e = eccentricity
        if hasattr(bee, "i"):
            bee.i = math.radians(inclination) if inclination < 10 else inclination

        # Set termination status
        bee.terminated = terminated
        bee.truncated = terminated

        # Store original ID for reference
        bee.original_sat_id = sat_id

        # Initialize retask board
        if not hasattr(bee, "retask_board"):
            bee.retask_board = []

        return bee

    def map_all_satellites(self, data: dict) -> List[Bee]:
        """Convert all satellites to Bee objects."""
        controller = data.get("telemetry-bridge", {}).get("controller", {})
        satellites = controller.get("satellites", [])
        failed_list = data.get("failed_satellites", [])

        # Build ID map first
        self.build_sat_id_map(satellites)

        # Set episode start from simulation_time or generated_at
        if satellites:
            first_sat = satellites[0]
            sim_time = first_sat.get("simulation_time", 0)
            if not sim_time:
                sim_time = controller.get("generated_at", 0)
            self.set_episode_start(sim_time)

        bees = []
        for sat in satellites:
            bee = self.map_satellite_to_bee(sat, failed_list)
            bees.append(bee)

        return bees

    # ─────────────────────────────────────────────────────────────────────────
    # Task → Flower Mapping
    # ─────────────────────────────────────────────────────────────────────────

    def extract_all_tasks(self, data: dict) -> List[dict]:
        """Gather all tasks from satellites and gossiper."""
        all_tasks = []
        seen_ids = set()

        # From satellites' assigned_tasks
        controller = data.get("telemetry-bridge", {}).get("controller", {})
        satellites = controller.get("satellites", [])

        for sat in satellites:
            sat_id = sat.get("satellite_id", "UNKNOWN")
            for task in sat.get("assigned_tasks", []):
                task_id = task.get("task_id", "")
                if task_id and task_id not in seen_ids:
                    task["_source_satellite"] = sat_id
                    task["_unassigned"] = False
                    all_tasks.append(task)
                    seen_ids.add(task_id)

        # From gossiper's unassignedTasks
        gossiper = data.get("telemetry-bridge", {}).get("gossiper", {})
        for sat_id, gdata in gossiper.items():
            if not isinstance(gdata, dict):
                continue
            for task in gdata.get("unassignedTasks", []):
                task_id = task.get("task_id", "")
                if task_id and task_id not in seen_ids:
                    task["_source_satellite"] = sat_id
                    task["_unassigned"] = True
                    all_tasks.append(task)
                    seen_ids.add(task_id)

        return all_tasks

    def map_task_to_flower(self, task: dict) -> Flower:
        """Convert a task dict to a Flower object."""
        task_id = task.get("task_id", "UNKNOWN")
        flower_id = self.task_id_map.get(task_id, 0)

        # Position from location_task (coordinates in METERS)
        loc = task.get("location_task", {})
        if not loc:
            loc = {"x": 0, "y": 0, "z": 0}

        if "latitude" in loc:
            x, y = self.latlon_to_grid(loc.get("latitude", 0), loc.get("longitude", 0))
        else:
            # Task locations are also in meters
            x, y, _ = self.eci_to_grid(loc)

        # Priority → pollen (1-5 scale to 0.2-1.0 for better differentiation)
        priority = task.get("priority", 3)
        if priority is None:
            priority = 3
        pollen = 0.2 + (float(priority) - 1) * 0.2  # Maps 1-5 to 0.2-1.0

        # Deadline → window_end
        # Handle various deadline scenarios:
        # - Past deadlines (before episode start): treat as SOFT window (no hard penalty)
        # - Far future deadlines (>1 day): clamp to reasonable episode length
        # - No deadline: can complete anytime
        deadline = task.get("Deadline") or task.get("deadline", 0)
        window_type = "NONE"
        
        if deadline:
            deadline_diff = deadline - self.episode_start
            
            if deadline_diff < 0:
                # Past deadline - treat as soft/no window (already "expired" in real world)
                window_end = self.max_steps
                window_type = "NONE"  # No penalty for missing
            elif deadline_diff > 86400:  # More than 1 day away
                # Far future - clamp to end of episode
                window_end = self.max_steps
                window_type = "SOFT"
            else:
                # Within next 24 hours - convert to steps
                window_end = int(deadline_diff / self.seconds_per_step)
                window_end = max(10, min(window_end, self.max_steps))  # At least 10 steps
                window_type = "HARD" if deadline_diff < 3600 else "SOFT"  # HARD if <1hr
        else:
            window_end = self.max_steps
            window_type = "NONE"

        # Status
        status = task.get("task_status", "pending")
        harvested = status.lower() in ["completed", "done", "finished"]

        # Assignment
        assigned_bee = None
        if not task.get("_unassigned", False):
            source_sat = task.get("_source_satellite")
            if source_sat and source_sat in self.sat_id_map:
                assigned_bee = self.sat_id_map[source_sat]

        # Create Flower
        flower = Flower(
            flower_id=flower_id,
            grid_size=self.grid_size,
            window_start=0,
            window_end=window_end,
            window_type=window_type,
        )

        # Set position
        flower.x = int(x)
        flower.y = int(y)

        # Set pollen
        flower.pollen = pollen
        flower.priority = pollen

        # Set status
        flower.harvested = harvested
        flower.assigned_bee = assigned_bee

        # Store original ID
        flower.original_task_id = task_id

        # Store reassignment info if present
        if task.get("reassignment_reason"):
            flower.reassignment_reason = task.get("reassignment_reason")
        if task.get("reassigned_at"):
            flower.reassigned_at = task.get("reassigned_at")

        return flower

    def map_all_tasks(self, data: dict) -> List[Flower]:
        """Convert all tasks to Flower objects."""
        all_tasks = self.extract_all_tasks(data)

        if not all_tasks:
            return []

        self.build_task_id_map(all_tasks)

        flowers = []
        for task in all_tasks:
            flower = self.map_task_to_flower(task)
            flowers.append(flower)

        return flowers

    # ─────────────────────────────────────────────────────────────────────────
    # Reassignment Handling
    # ─────────────────────────────────────────────────────────────────────────

    def build_retask_boards(
        self, data: dict, bees: List[Bee], flowers: List[Flower]
    ) -> None:
        """
        Populate retask_board for active bees based on reassignment data.
        """
        failed_list = data.get("failed_satellites", [])

        # Find unassigned flowers (from failed satellites or pool)
        unassigned_flowers = [
            f for f in flowers if f.assigned_bee is None and not f.harvested
        ]

        # Get active bees
        active_bees = [b for b in bees if not b.terminated]

        if not active_bees or not unassigned_flowers:
            return

        # Distribute unassigned tasks to nearest active bees
        for flower in unassigned_flowers:
            nearest_bee = self._find_nearest_bee(flower, active_bees)
            if nearest_bee:
                retask_entry = {
                    "flower_id": flower.id,
                    "source_bee": None,  # Original bee unknown or failed
                    "hops": 1,
                    "received_step": 0,
                    "pollen": flower.pollen,
                }
                nearest_bee.retask_board.append(retask_entry)

    def _find_nearest_bee(self, flower: Flower, bees: List[Bee]) -> Optional[Bee]:
        """Find the nearest active bee to a flower."""
        best_bee = None
        best_dist = float("inf")

        fx = getattr(flower, "cx", getattr(flower, "x", 0))
        fy = getattr(flower, "cy", getattr(flower, "y", 0))

        for bee in bees:
            bx = getattr(bee, "fx", getattr(bee, "x", 0))
            by = getattr(bee, "fy", getattr(bee, "y", 0))

            dist = math.sqrt((bx - fx) ** 2 + (by - fy) ** 2)
            if dist < best_dist:
                best_dist = dist
                best_bee = bee

        return best_bee

    # ─────────────────────────────────────────────────────────────────────────
    # Gossiper/Neighbor Mapping (optional)
    # ─────────────────────────────────────────────────────────────────────────

    def map_gossiper_neighbors(self, data: dict, bees: List[Bee]) -> Dict[int, List[int]]:
        """
        Extract neighbor relationships from gossiper data.

        Returns: {bee_id: [neighbor_bee_id, ...]}
        """
        gossiper = data.get("telemetry-bridge", {}).get("gossiper", {})
        neighbor_map: Dict[int, List[int]] = {}

        for sat_id, gdata in gossiper.items():
            if not isinstance(gdata, dict):
                continue

            bee_id = self.sat_id_map.get(sat_id)
            if bee_id is None:
                continue

            neighbors = gdata.get("neighbors", {})
            neighbor_ids = []

            for neighbor_sat_id in neighbors.keys():
                neighbor_bee_id = self.sat_id_map.get(neighbor_sat_id)
                if neighbor_bee_id is not None:
                    neighbor_ids.append(neighbor_bee_id)

            neighbor_map[bee_id] = neighbor_ids

        return neighbor_map

    # ─────────────────────────────────────────────────────────────────────────
    # Main Entry Point
    # ─────────────────────────────────────────────────────────────────────────

    def map_telemetry(self, filepath: str) -> Tuple[List[Bee], List[Flower], dict]:
        """
        Main entry point: load and map entire telemetry file.

        Returns:
            (bees, flowers, metadata)
        """
        data = self.load(filepath)

        # Setup coordinate scaling BEFORE mapping anything
        # This finds the bounding box of all satellite and task positions
        self.setup_coordinate_scaling(data)

        # Setup deadline normalization BEFORE mapping tasks
        self.setup_deadline_normalization(data)

        # Map satellites → bees
        bees = self.map_all_satellites(data)

        # Map tasks → flowers
        flowers = self.map_all_tasks(data)

        # Build retask boards for reassigned tasks
        self.build_retask_boards(data, bees, flowers)

        # Extract neighbor relationships
        neighbors = self.map_gossiper_neighbors(data, bees)

        # Assign flowers to bees' assigned_flowers list
        for flower in flowers:
            if flower.assigned_bee is not None and flower.assigned_bee < len(bees):
                bee = bees[flower.assigned_bee]
                if not hasattr(bee, "assigned_flowers"):
                    bee.assigned_flowers = []
                bee.assigned_flowers.append(flower.id)

        # Metadata for reference
        metadata = {
            "sat_id_map": self.sat_id_map,
            "task_id_map": self.task_id_map,
            "failed_satellites": data.get("failed_satellites", []),
            "total_tasks_moved": data.get("task_reassignment", {}).get(
                "total_tasks_moved", 0
            ),
            "episode_start": self.episode_start,
            "num_bees": len(bees),
            "num_flowers": len(flowers),
            "num_active_bees": len([b for b in bees if not b.terminated]),
            "num_unassigned_flowers": len(
                [f for f in flowers if f.assigned_bee is None]
            ),
            "neighbor_map": neighbors,
        }

        return bees, flowers, metadata


# ─────────────────────────────────────────────────────────────────────────────
# Convenience Functions
# ─────────────────────────────────────────────────────────────────────────────


def load_telemetry(
    filepath: str, config: Optional[dict] = None
) -> Tuple[List[Bee], List[Flower], dict]:
    """Load telemetry file and return mapped objects."""
    mapper = TelemetryMapper(config)
    return mapper.map_telemetry(filepath)


def print_telemetry_summary(bees: List[Bee], flowers: List[Flower], metadata: dict):
    """Print a summary of loaded telemetry data."""
    print("=" * 60)
    print("TELEMETRY BRIDGE SUMMARY")
    print("=" * 60)
    print(f"Bees (satellites): {metadata['num_bees']}")
    print(f"  Active: {metadata['num_active_bees']}")
    print(f"  Failed: {len(metadata['failed_satellites'])}")
    if metadata["failed_satellites"]:
        print(f"    → {', '.join(metadata['failed_satellites'])}")

    print(f"\nFlowers (tasks): {metadata['num_flowers']}")
    print(f"  Assigned: {metadata['num_flowers'] - metadata['num_unassigned_flowers']}")
    print(f"  Unassigned: {metadata['num_unassigned_flowers']}")
    print(f"  Tasks moved: {metadata['total_tasks_moved']}")

    print(f"\nEpisode start: {metadata['episode_start']}")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# CLI for testing
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    filepath = sys.argv[1] if len(sys.argv) > 1 else "telemetrybridge.json"

    print(f"Loading telemetry from: {filepath}")

    bees, flowers, metadata = load_telemetry(filepath, {
        "grid_size": 30,
        "meters_per_unit": 500_000,  # 500km per grid unit - fits LEO/MEO/GEO in grid
        "seconds_per_step": 1.0,     # 1 second per step for responsive control
        "max_steps": 500,
    })

    print_telemetry_summary(bees, flowers, metadata)

    # Print first few bees
    print("\nFirst 3 bees:")
    for bee in bees[:3]:
        print(f"  Bee {bee.id} ({getattr(bee, 'original_sat_id', '?')}): "
              f"pos=({bee.fx:.1f}, {bee.fy:.1f}, {bee.fz:.1f}), "
              f"battery={bee.battery:.1f}, "
              f"terminated={bee.terminated}")

    # Print first few flowers
    print("\nFirst 3 flowers:")
    for flower in flowers[:3]:
        fx = getattr(flower, "cx", getattr(flower, "x", 0))
        fy = getattr(flower, "cy", getattr(flower, "y", 0))
        print(f"  Flower {flower.id} ({getattr(flower, 'original_task_id', '?')}): "
              f"pos=({fx:.1f}, {fy:.1f}), "
              f"pollen={flower.pollen:.2f}, "
              f"assigned={flower.assigned_bee}")

    # Test environment integration
    print("\n" + "=" * 60)
    print("TESTING ENVIRONMENT INTEGRATION")
    print("=" * 60)

    try:
        from bees_env import BeeForagingEnv

        env = BeeForagingEnv(num_bees=5, num_flowers=12, verbose=False)
        env.reset()

        # Load from telemetry
        meta = env.load_from_telemetry(filepath)

        print(f"✓ Environment loaded: {env.num_bees} bees, {env.num_flowers} flowers")
        print(f"✓ Active bees: {sum(1 for b in env.bees if not b.terminated)}")
        print(f"✓ Terminated bees: {sum(1 for b in env.bees if b.terminated)}")
        print(f"✓ Unassigned flowers: {sum(1 for f in env.flowers if f.assigned_bee is None)}")

        # Try a step
        actions = {f"bee_{i}": 0 for i in range(env.num_bees)}
        result = env.step(actions)

        # Handle different return formats (5 or 6 values)
        if len(result) == 5:
            obs, rewards, terms, truncs, infos = result
        else:
            obs, rewards, terms, truncs, infos = result[:5]

        print(f"✓ Step executed successfully")
        print(f"✓ Total reward: {sum(rewards.values()):.4f}")

        print("\n✅ All tests passed!")

    except ImportError as e:
        print(f"⚠ Environment test skipped: {e}")
    except Exception as e:
        print(f"✗ Environment test failed: {e}")
