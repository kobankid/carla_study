"""11-camera sensor rig demo for CARLA.

Drives an ego vehicle with BehaviorAgent (same setup as
autonomous_driving_demo.py) and attaches an 11-camera perception rig
modeled after typical L2+/L4 ADAS camera suites:

  1. front_near    - front bumper, wide, short range
  2. front_far     - windshield, narrow, long range (telephoto)
  3. rear_mid      - rear bumper, mid range
  4. left_front    - left mirror, forward-left side view
  5. right_front   - right mirror, forward-right side view
  6. left_rear     - left rear pillar, backward-left side view
  7. right_rear    - right rear pillar, backward-right side view
  8. fisheye_front - front bumper, ~170 deg
  9. fisheye_rear  - rear bumper, ~170 deg
  10. fisheye_left  - left side, ~170 deg
  11. fisheye_right - right side, ~170 deg

CARLA's sensor.camera.rgb has no true fisheye projection; the fisheye
entries here approximate the look with a wide FOV plus the lens_k /
lens_kcube / lens_circle_* radial-distortion attributes.

Requires a running CARLA server, e.g.:
    packages/CALRA/CarlaUE4.sh

Usage:
    python3 multi_camera_demo.py --town Town05 --num-vehicles 20 \
        --output-dir _out --duration 30
"""
import argparse
import os
import random
import sys
import time
from pathlib import Path

import carla

from autonomous_driving_demo import (
    BehaviorAgent,
    move_spectator_to,
    pick_destination,
    spawn_background_traffic,
    spawn_ego_vehicle,
)


# Each entry: name, mount transform (relative to vehicle origin), fov,
# resolution, and optional fisheye lens-distortion attributes.
# Mount points are chosen to clear the Tesla Model 3 bounding box (roughly
# +/-2.4m long, +/-1.1m wide, 0 to 1.5m tall) with a small margin so the
# camera's near plane doesn't clip through the vehicle's own body mesh.
CAMERA_RIG = [
    dict(
        name="front_near",
        transform=carla.Transform(carla.Location(x=2.5, z=0.5), carla.Rotation(yaw=0)),
        fov=120,
        width=960,
        height=540,
    ),
    dict(
        name="front_far",
        transform=carla.Transform(carla.Location(x=1.0, z=1.45), carla.Rotation(yaw=0)),
        fov=35,
        width=960,
        height=540,
    ),
    dict(
        name="rear_mid",
        transform=carla.Transform(carla.Location(x=-2.5, z=1.0), carla.Rotation(yaw=180)),
        fov=70,
        width=960,
        height=540,
    ),
    dict(
        name="left_front",
        transform=carla.Transform(carla.Location(x=0.9, y=-1.2, z=1.1), carla.Rotation(yaw=-55)),
        fov=100,
        width=960,
        height=540,
    ),
    dict(
        name="right_front",
        transform=carla.Transform(carla.Location(x=0.9, y=1.2, z=1.1), carla.Rotation(yaw=55)),
        fov=100,
        width=960,
        height=540,
    ),
    dict(
        name="left_rear",
        transform=carla.Transform(carla.Location(x=-1.0, y=-1.2, z=1.1), carla.Rotation(yaw=-110)),
        fov=100,
        width=960,
        height=540,
    ),
    dict(
        name="right_rear",
        transform=carla.Transform(carla.Location(x=-1.0, y=1.2, z=1.1), carla.Rotation(yaw=110)),
        fov=100,
        width=960,
        height=540,
    ),
    dict(
        name="fisheye_front",
        transform=carla.Transform(carla.Location(x=2.5, z=0.35), carla.Rotation(yaw=0)),
        fov=170,
        width=800,
        height=800,
        fisheye=True,
    ),
    dict(
        name="fisheye_rear",
        transform=carla.Transform(carla.Location(x=-2.5, z=0.35), carla.Rotation(yaw=180)),
        fov=170,
        width=800,
        height=800,
        fisheye=True,
    ),
    dict(
        name="fisheye_left",
        transform=carla.Transform(carla.Location(y=-1.2, z=0.6), carla.Rotation(yaw=-90)),
        fov=170,
        width=800,
        height=800,
        fisheye=True,
    ),
    dict(
        name="fisheye_right",
        transform=carla.Transform(carla.Location(y=1.2, z=0.6), carla.Rotation(yaw=90)),
        fov=170,
        width=800,
        height=800,
        fisheye=True,
    ),
]


def build_camera_blueprint(blueprint_library, spec, sensor_tick):
    bp = blueprint_library.find("sensor.camera.rgb")
    bp.set_attribute("image_size_x", str(spec["width"]))
    bp.set_attribute("image_size_y", str(spec["height"]))
    bp.set_attribute("fov", str(spec["fov"]))
    bp.set_attribute("sensor_tick", str(sensor_tick))
    if spec.get("fisheye"):
        bp.set_attribute("lens_circle_multiplier", "3.0")
        bp.set_attribute("lens_circle_falloff", "3.0")
        bp.set_attribute("lens_k", "-1.0")
        bp.set_attribute("lens_kcube", "3.0")
    return bp


def spawn_camera_rig(world, blueprint_library, ego_vehicle, output_dir, sensor_tick):
    """Attaches all 11 cameras to the ego vehicle. Each camera writes its
    frames to output_dir/<camera_name>/%08d.png via its own listen callback."""
    cameras = []
    for spec in CAMERA_RIG:
        bp = build_camera_blueprint(blueprint_library, spec, sensor_tick)
        camera = world.spawn_actor(bp, spec["transform"], attach_to=ego_vehicle)
        camera_output_dir = output_dir / spec["name"]
        camera_output_dir.mkdir(parents=True, exist_ok=True)

        def make_callback(save_dir):
            def callback(image):
                image.save_to_disk(str(save_dir / f"{image.frame:08d}.png"))
            return callback

        camera.listen(make_callback(camera_output_dir))
        cameras.append(camera)
        print(f"  camera '{spec['name']}' attached (fov={spec['fov']}, "
              f"{spec['width']}x{spec['height']}{', fisheye' if spec.get('fisheye') else ''})")
    return cameras


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
    parser.add_argument("--output-dir", default="_out", help="directory to write camera frames to")
    parser.add_argument(
        "--sensor-tick",
        type=float,
        default=0.2,
        help="seconds between captures per camera (0 = every simulation tick)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="stop automatically after this many seconds of simulated time (default: run until Ctrl+C)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)
    world = client.load_world(args.town)

    traffic_manager = client.get_trafficmanager()
    traffic_manager.set_global_distance_to_leading_vehicle(2.5)
    traffic_manager.set_synchronous_mode(True)

    settings = world.get_settings()
    original_sync_mode = settings.synchronous_mode
    original_fixed_delta_seconds = settings.fixed_delta_seconds
    fixed_delta_seconds = 0.05
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = fixed_delta_seconds
    world.apply_settings(settings)

    blueprint_library = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    available_spawn_points = list(spawn_points)
    random.shuffle(available_spawn_points)

    ego_vehicle = None
    background_vehicles = []
    cameras = []

    try:
        ego_vehicle, ego_spawn_point = spawn_ego_vehicle(world, blueprint_library, available_spawn_points)
        background_vehicles = spawn_background_traffic(
            world, blueprint_library, available_spawn_points, args.num_vehicles
        )

        for _ in range(10):
            world.tick()

        for vehicle in background_vehicles:
            vehicle.set_autopilot(True, traffic_manager.get_port())

        print(f"Attaching {len(CAMERA_RIG)}-camera rig, saving frames under '{output_dir}/':")
        cameras = spawn_camera_rig(world, blueprint_library, ego_vehicle, output_dir, args.sensor_tick)

        agent = BehaviorAgent(ego_vehicle, behavior=args.behavior)
        destination = pick_destination(spawn_points, ego_spawn_point.location)
        agent.set_destination(destination)

        print(f"Ego vehicle spawned: {ego_vehicle.type_id} (id={ego_vehicle.id})")
        print(f"Background traffic: {len(background_vehicles)} vehicles")
        print(f"Driving to destination: {destination}")

        elapsed = 0.0
        while args.duration is None or elapsed < args.duration:
            world.tick()
            elapsed += fixed_delta_seconds

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
        # Each camera's listen() callback (including its save_to_disk write)
        # runs on a background thread that trails the simulation by up to
        # one frame. Stop the cameras, then give that last in-flight
        # callback time to finish writing before destroying the actors --
        # otherwise the final frame on disk can end up truncated.
        for camera in cameras:
            if camera.is_alive:
                camera.stop()
        time.sleep(1.0)
        for camera in cameras:
            if camera.is_alive:
                camera.destroy()

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
        print("Done.")


if __name__ == "__main__":
    main()
    # The CARLA 0.9.16 client library can abort during normal interpreter
    # teardown after sensors have been used (a background RPC thread isn't
    # joined cleanly), even though cleanup above has already completed
    # successfully. Skip Python's object-destruction shutdown sequence to
    # avoid that spurious crash.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
