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

Each camera is encoded straight to an MP4 (no intermediate PNG frames) at
output_dir/<camera_name>_<timestamp>.mp4. Two extra views are recorded the
same way: "main_view", a chase camera behind the car (the normal
third-person view of it driving), and "birdseye", a top-down camera
mirroring CARLA's default spectator behavior (straight down, following the
car). All 13 feeds are combined into output_dir/combined_demo_<timestamp>.mp4
as two side-by-side panels:
  - left:  main_view in the middle, surrounded by the 7 non-fisheye rig
           cameras (front_near/front_far/rear_mid/left_front/right_front/
           left_rear/right_rear).
  - right: birdseye in the middle, surrounded by the 4 fisheye cameras.

Usage:
    python3 multi_camera_demo.py --town Town05 --num-vehicles 20 \
        --output-dir _out --duration 30
"""
import argparse
import os
import random
import sys
import threading
import time
from pathlib import Path

import carla
import cv2
import numpy as np

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

# The "normal simulator view" for the combined demo video: a chase camera
# behind and above the car, looking forward -- not one of the 11 rig
# cameras, just what you'd normally watch while the sim runs.
MAIN_VIEW_CAMERA = dict(
    name="main_view",
    transform=carla.Transform(carla.Location(x=-6.0, z=3.0), carla.Rotation(pitch=-12, yaw=0)),
    fov=100,
    width=1280,
    height=720,
)

# The other combined-video center view: CARLA's default top-down spectator
# look (straight down, 25m up), mirroring move_spectator_to() below. Unlike
# every other camera here it is NOT attached to the vehicle -- attaching
# would make it spin with the car's heading -- it's a free-floating actor
# whose transform gets updated every tick from birdseye_transform().
BIRDSEYE_CAMERA = dict(
    name="birdseye",
    fov=90,
    width=720,
    height=540,
)

# ---------------------------------------------------------------------------
# Combined demo video layout: two side-by-side panels.
#   left panel  = main_view in the middle, ringed by the 7 non-fisheye rig
#                 cameras (front row / rear row / diagonal corners).
#   right panel = birdseye in the middle, ringed by the 4 fisheye cameras
#                 at N/E/S/W.
# All sizes are (width, height) in pixels; layout is built from explicit
# pixel rects rather than a uniform grid since the two panels don't share
# the same column count.
LEFT_TILE_SIZE = (240, 180)
LEFT_TOP_ROW = ["left_front", "front_far", "front_near", "right_front"]
LEFT_BOTTOM_ROW = ["left_rear", "rear_mid", None, "right_rear"]
LEFT_CENTER_SIZE = (LEFT_TILE_SIZE[0] * len(LEFT_TOP_ROW), 540)  # 960x540, matches main_view's 16:9
LEFT_PANEL_SIZE = (LEFT_CENTER_SIZE[0], LEFT_TILE_SIZE[1] * 2 + LEFT_CENTER_SIZE[1])  # 960x900

RIGHT_SIDE_TILE_SIZE = (240, 540)  # fisheye_left / fisheye_right
RIGHT_TOPBOTTOM_TILE_SIZE = (720, 180)  # fisheye_front / fisheye_rear
RIGHT_CENTER_SIZE = (720, 540)  # birdseye, matches BIRDSEYE_CAMERA's native size 1:1 (no resize)
RIGHT_PANEL_SIZE = (
    RIGHT_SIDE_TILE_SIZE[0] * 2 + RIGHT_CENTER_SIZE[0],
    RIGHT_TOPBOTTOM_TILE_SIZE[1] * 2 + RIGHT_CENTER_SIZE[1],
)  # 1200x900

assert LEFT_PANEL_SIZE[1] == RIGHT_PANEL_SIZE[1], "left/right panels must have equal height to sit side by side"


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


def image_to_bgr_array(image):
    """CARLA delivers raw_data as a flat BGRA buffer; drop the alpha channel
    to get the BGR layout cv2.VideoWriter expects."""
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    return array[:, :, :3]


def _draw_label(tile, text):
    cv2.rectangle(tile, (0, 0), (tile.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(tile, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def _place_tile(canvas, frame, label, x, y, size):
    """Resizes frame to size=(w, h) and stamps it into canvas at (x, y).
    Leaves the background (and label) alone if frame hasn't arrived yet."""
    if frame is None:
        return
    w, h = size
    tile = cv2.resize(frame, (w, h))
    _draw_label(tile, label)
    canvas[y:y + h, x:x + w] = tile


class CompositeRecorder:
    """Builds output_dir/combined_demo_<timestamp>.mp4 as two side-by-side
    panels (see the LEFT_*/RIGHT_* layout constants above):
      - left:  main_view in the middle, ringed by the 7 non-fisheye cameras.
      - right: birdseye in the middle, ringed by the 4 fisheye cameras.

    Every camera's frame just updates a "latest frame" cache as it arrives
    (they run on independent sensor threads, generally not in lockstep);
    the combined frame is assembled and written every time a fresh
    main_view frame arrives (notify_main), using whatever is currently
    cached for everything else (at most one capture interval stale).
    """

    def __init__(self, output_dir, fps, timestamp):
        canvas_w = LEFT_PANEL_SIZE[0] + RIGHT_PANEL_SIZE[0]
        canvas_h = LEFT_PANEL_SIZE[1]
        self.canvas_size = (canvas_w, canvas_h)
        self.lock = threading.Lock()
        self.latest_frames = {}
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_path = output_dir / f"combined_demo_{timestamp}.mp4"
        self.writer = cv2.VideoWriter(str(video_path), fourcc, fps, self.canvas_size)
        print(f"  combined demo video -> {video_path} ({canvas_w}x{canvas_h})")

    def update(self, name, bgr_array):
        with self.lock:
            self.latest_frames[name] = bgr_array

    def notify_main(self, main_bgr_array):
        self.update("main_view", main_bgr_array)
        self._render()

    def _render(self):
        canvas_w, canvas_h = self.canvas_size
        canvas = np.full((canvas_h, canvas_w, 3), 30, dtype=np.uint8)
        with self.lock:
            frames = dict(self.latest_frames)

        tw, th = LEFT_TILE_SIZE
        for col, name in enumerate(LEFT_TOP_ROW):
            _place_tile(canvas, frames.get(name), name, col * tw, 0, LEFT_TILE_SIZE)
        _place_tile(canvas, frames.get("main_view"), "main_view", 0, th, LEFT_CENTER_SIZE)
        for col, name in enumerate(LEFT_BOTTOM_ROW):
            if name is not None:
                _place_tile(canvas, frames.get(name), name, col * tw, th + LEFT_CENTER_SIZE[1], LEFT_TILE_SIZE)

        x_off = LEFT_PANEL_SIZE[0]
        sw, sh = RIGHT_SIDE_TILE_SIZE
        tbw, tbh = RIGHT_TOPBOTTOM_TILE_SIZE
        cw, ch = RIGHT_CENTER_SIZE
        _place_tile(canvas, frames.get("fisheye_front"), "fisheye_front", x_off + sw, 0, RIGHT_TOPBOTTOM_TILE_SIZE)
        _place_tile(canvas, frames.get("fisheye_left"), "fisheye_left", x_off, tbh, RIGHT_SIDE_TILE_SIZE)
        _place_tile(canvas, frames.get("birdseye"), "birdseye", x_off + sw, tbh, RIGHT_CENTER_SIZE)
        _place_tile(canvas, frames.get("fisheye_right"), "fisheye_right", x_off + sw + cw, tbh, RIGHT_SIDE_TILE_SIZE)
        _place_tile(canvas, frames.get("fisheye_rear"), "fisheye_rear", x_off + sw, tbh + ch, RIGHT_TOPBOTTOM_TILE_SIZE)

        self.writer.write(canvas)

    def release(self):
        self.writer.release()


def spawn_camera_rig(world, blueprint_library, ego_vehicle, output_dir, sensor_tick, fps, composite, timestamp):
    """Attaches all 11 rig cameras to the ego vehicle. Each camera encodes
    its frames directly to output_dir/<camera_name>_<timestamp>.mp4 via its
    own listen callback (no PNG frames ever hit disk), and also feeds its
    latest frame to `composite` for the combined demo video."""
    cameras = []
    writers = []
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    for spec in CAMERA_RIG:
        bp = build_camera_blueprint(blueprint_library, spec, sensor_tick)
        camera = world.spawn_actor(bp, spec["transform"], attach_to=ego_vehicle)
        video_path = output_dir / f"{spec['name']}_{timestamp}.mp4"
        writer = cv2.VideoWriter(str(video_path), fourcc, fps, (spec["width"], spec["height"]))

        def make_callback(video_writer, name):
            def callback(image):
                frame = image_to_bgr_array(image)
                video_writer.write(frame)
                composite.update(name, frame)
            return callback

        camera.listen(make_callback(writer, spec["name"]))
        cameras.append(camera)
        writers.append(writer)
        print(f"  camera '{spec['name']}' attached (fov={spec['fov']}, "
              f"{spec['width']}x{spec['height']}{', fisheye' if spec.get('fisheye') else ''}) "
              f"-> {video_path}")
    return cameras, writers


def spawn_main_view_camera(world, blueprint_library, ego_vehicle, output_dir, sensor_tick, fps, composite, timestamp):
    """Attaches the chase-view camera and wires it to drive `composite`'s
    combined-frame assembly (see CompositeRecorder)."""
    spec = MAIN_VIEW_CAMERA
    bp = build_camera_blueprint(blueprint_library, spec, sensor_tick)
    camera = world.spawn_actor(bp, spec["transform"], attach_to=ego_vehicle)
    video_path = output_dir / f"{spec['name']}_{timestamp}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (spec["width"], spec["height"]))

    def callback(image):
        frame = image_to_bgr_array(image)
        writer.write(frame)
        composite.notify_main(frame)

    camera.listen(callback)
    print(f"  camera '{spec['name']}' attached (fov={spec['fov']}, "
          f"{spec['width']}x{spec['height']}) -> {video_path}")
    return camera, writer


def birdseye_transform(vehicle_transform):
    """Same framing as move_spectator_to() in autonomous_driving_demo.py:
    straight down from 25m up, not rotated with the vehicle's heading."""
    return carla.Transform(
        vehicle_transform.location + carla.Location(z=25),
        carla.Rotation(pitch=-90),
    )


def spawn_birdseye_camera(world, blueprint_library, ego_vehicle, output_dir, sensor_tick, fps, composite, timestamp):
    """Spawns the top-down birdseye camera as a free-floating actor (not
    attached to the vehicle, so it doesn't spin with its heading). Callers
    must reposition it every tick via birdseye_transform(); see main()."""
    spec = BIRDSEYE_CAMERA
    bp = build_camera_blueprint(blueprint_library, spec, sensor_tick)
    camera = world.spawn_actor(bp, birdseye_transform(ego_vehicle.get_transform()))
    video_path = output_dir / f"{spec['name']}_{timestamp}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (spec["width"], spec["height"]))

    def callback(image):
        frame = image_to_bgr_array(image)
        writer.write(frame)
        composite.update(spec["name"], frame)

    camera.listen(callback)
    print(f"  camera '{spec['name']}' attached (fov={spec['fov']}, "
          f"{spec['width']}x{spec['height']}) -> {video_path}")
    return camera, writer


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
    parser.add_argument(
        "--output-dir",
        default="_out",
        help="base directory to write recordings to; each run gets its own "
        "<output-dir>/<YYYYmmdd_HHMMSS>/ subfolder, and every video file "
        "name carries that same timestamp as a suffix",
    )
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

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp
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
    writers = []
    composite = None

    try:
        ego_vehicle, ego_spawn_point = spawn_ego_vehicle(world, blueprint_library, available_spawn_points)
        background_vehicles = spawn_background_traffic(
            world, blueprint_library, available_spawn_points, args.num_vehicles
        )

        for _ in range(10):
            world.tick()

        for vehicle in background_vehicles:
            vehicle.set_autopilot(True, traffic_manager.get_port())

        fps = (1.0 / args.sensor_tick) if args.sensor_tick > 0 else (1.0 / fixed_delta_seconds)
        print(f"Attaching {len(CAMERA_RIG)}-camera rig + main_view + birdseye, encoding video "
              f"under '{output_dir}/' at {fps:.1f} fps:")
        composite = CompositeRecorder(output_dir, fps, timestamp)
        cameras, writers = spawn_camera_rig(
            world, blueprint_library, ego_vehicle, output_dir, args.sensor_tick, fps, composite, timestamp
        )
        main_camera, main_writer = spawn_main_view_camera(
            world, blueprint_library, ego_vehicle, output_dir, args.sensor_tick, fps, composite, timestamp
        )
        birdseye_camera, birdseye_writer = spawn_birdseye_camera(
            world, blueprint_library, ego_vehicle, output_dir, args.sensor_tick, fps, composite, timestamp
        )
        cameras.append(main_camera)
        writers.append(main_writer)
        cameras.append(birdseye_camera)
        writers.append(birdseye_writer)

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
            if birdseye_camera.is_alive:
                birdseye_camera.set_transform(birdseye_transform(ego_vehicle.get_transform()))

    except KeyboardInterrupt:
        print("\nStopping simulation.")
    finally:
        print("Cleaning up actors and restoring world settings...")
        # Each camera's listen() callback (including its VideoWriter.write)
        # runs on a background thread that trails the simulation by up to
        # one frame. Stop the cameras, then give that last in-flight
        # callback time to finish before releasing the writers -- otherwise
        # the last frame or two can end up missing from the video.
        for camera in cameras:
            if camera.is_alive:
                camera.stop()
        time.sleep(1.0)
        for camera in cameras:
            if camera.is_alive:
                camera.destroy()
        for writer in writers:
            writer.release()
        if composite is not None:
            composite.release()

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
