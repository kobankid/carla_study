"""Autonomous driving sample for CARLA.

Spawns an ego vehicle driven by CARLA's BehaviorAgent (which handles
waypoint following, traffic lights, and collision/lane-change avoidance),
plus a handful of background traffic managed by the Traffic Manager.

Requires a running CARLA server, e.g.:
    packages/CALRA/CarlaUE4.sh

Usage:
    python3 autonomous_driving_demo.py --town Town05 --num-vehicles 20
"""
import argparse
import random
import sys
from pathlib import Path

import carla

# The `agents` package (BehaviorAgent, Traffic Manager helpers, ...) ships
# inside the CARLA build under PythonAPI/carla but isn't pip-installed.
_CARLA_PYTHONAPI = Path(__file__).resolve().parent / "packages" / "CALRA" / "PythonAPI" / "carla"
sys.path.append(str(_CARLA_PYTHONAPI))

from agents.navigation.behavior_agent import BehaviorAgent  # noqa: E402


def spawn_ego_vehicle(world, blueprint_library, available_spawn_points):
    """Spawns the ego vehicle, consuming its spawn point from available_spawn_points
    so background traffic never spawns on top of it."""
    vehicle_bp = blueprint_library.find("vehicle.tesla.model3")
    for spawn_point in list(available_spawn_points):
        vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
        if vehicle is not None:
            available_spawn_points.remove(spawn_point)
            return vehicle, spawn_point
    raise RuntimeError("Could not find a free spawn point for the ego vehicle")


def spawn_background_traffic(world, blueprint_library, available_spawn_points, count):
    vehicle_bps = blueprint_library.filter("vehicle.*")
    # Exclude bikes/motorcycles which need special physics tuning to behave well.
    vehicle_bps = [bp for bp in vehicle_bps if int(bp.get_attribute("number_of_wheels")) == 4]

    actors = []
    for spawn_point in list(available_spawn_points)[:count]:
        vehicle = world.try_spawn_actor(random.choice(vehicle_bps), spawn_point)
        if vehicle is None:
            continue
        available_spawn_points.remove(spawn_point)
        actors.append(vehicle)
    return actors


def pick_destination(spawn_points, origin, min_distance=30.0):
    """Avoids picking a destination that is basically where the vehicle
    already is, which produces a degenerate route and erratic control."""
    candidates = [sp.location for sp in spawn_points if sp.location.distance(origin) > min_distance]
    return random.choice(candidates) if candidates else random.choice(spawn_points).location


def move_spectator_to(world, transform):
    spectator = world.get_spectator()
    chase_transform = carla.Transform(
        transform.location + carla.Location(z=25),
        carla.Rotation(pitch=-90),
    )
    spectator.set_transform(chase_transform)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--town", default="Town05")
    parser.add_argument("--num-vehicles", type=int, default=20, help="background traffic count")
    parser.add_argument(
        "--behavior",
        choices=["cautious", "normal", "aggressive"],
        default="normal",
        help="ego vehicle driving style",
    )
    args = parser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)
    world = client.load_world(args.town)

    traffic_manager = client.get_trafficmanager()
    traffic_manager.set_global_distance_to_leading_vehicle(2.5)
    traffic_manager.set_synchronous_mode(True)

    settings = world.get_settings()
    original_sync_mode = settings.synchronous_mode
    original_fixed_delta_seconds = settings.fixed_delta_seconds
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    blueprint_library = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    available_spawn_points = list(spawn_points)
    random.shuffle(available_spawn_points)

    ego_vehicle = None
    background_vehicles = []

    try:
        ego_vehicle, ego_spawn_point = spawn_ego_vehicle(world, blueprint_library, available_spawn_points)
        background_vehicles = spawn_background_traffic(
            world, blueprint_library, available_spawn_points, args.num_vehicles
        )

        # Freshly spawned actors are dropped in slightly above the road and take a
        # few physics steps to settle. Handing over full throttle/steering control
        # (or autopilot) before that happens can make a vehicle lurch or clip
        # nearby geometry right at spawn, so let everything settle first.
        for _ in range(10):
            world.tick()

        for vehicle in background_vehicles:
            vehicle.set_autopilot(True, traffic_manager.get_port())

        agent = BehaviorAgent(ego_vehicle, behavior=args.behavior)
        destination = pick_destination(spawn_points, ego_spawn_point.location)
        agent.set_destination(destination)

        print(f"Ego vehicle spawned: {ego_vehicle.type_id} (id={ego_vehicle.id})")
        print(f"Background traffic: {len(background_vehicles)} vehicles")
        print(f"Driving to destination: {destination}")

        while True:
            world.tick()

            if agent.done():
                print("Destination reached, picking a new one.")
                destination = pick_destination(spawn_points, ego_vehicle.get_location())
                agent.set_destination(destination)
                print(f"New destination: {destination}")

            control = agent.run_step()
            control.manual_gear_shift = False
            ego_vehicle.apply_control(control)

            move_spectator_to(world, ego_vehicle.get_transform())

    except KeyboardInterrupt:
        print("\nStopping simulation.")
    finally:
        print("Cleaning up actors and restoring world settings...")
        traffic_manager.set_synchronous_mode(False)
        settings.synchronous_mode = original_sync_mode
        settings.fixed_delta_seconds = original_fixed_delta_seconds
        world.apply_settings(settings)

        for actor in background_vehicles:
            if actor.is_alive:
                actor.set_autopilot(False)

        actors_to_destroy = list(background_vehicles)
        if ego_vehicle is not None:
            actors_to_destroy.append(ego_vehicle)
        client.apply_batch_sync(
            [carla.command.DestroyActor(actor) for actor in actors_to_destroy if actor.is_alive]
        )


if __name__ == "__main__":
    main()
