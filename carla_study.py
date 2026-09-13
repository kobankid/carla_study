import carla
import random

client = carla.Client('localhost', 2000)
world = client.get_world()

# Print available maps
client.get_available_maps()

# Load new map
client.load_world('Town05')

# Reload current map and reset state
client.reload_world()

# Get names of all objects 
print(world.get_names_of_all_objects())

# Filter the list of names for buildings
filter(lambda x: 'Building' in x, world.get_names_of_all_objects())

# Get a list of all actors, such as vehicles and pedestrians
print(world.get_actors())

# Filter the list to find the vehicles
print(world.get_actors().filter('*vehicle*'))

# Get the blueprint library and filter for the vehicle blueprints
vehicle_bps = world.get_blueprint_library().filter('*vehicle*')

# Randomly choose a vehicle blueprint to spawn
vehicle_bp = random.choice(vehicle_bps)

# We need a place to spawn the vehicle that will work so we will
# use the predefined spawn points for the map and randomly select one
spawn_point = random.choice(world.get_map().get_spawn_points())

# Now let's spawn the vehicle
vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)

# Retrieve the spectator object
spectator = world.get_spectator()

# Get the location and rotation of the spectator through its transform
transform = spectator.get_transform()

location = transform.location
rotation = transform.rotation

# Set the spectator with an empty transform
spectator.set_transform(carla.Transform())
# This will set the spectator at the origin of the map, with 0 degrees
# pitch, yaw and roll - a good way to orient yourself in the map

# Get the map's spawn points
spawn_points = world.get_map().get_spawn_points()

# Get the blueprint library and filter for the vehicle blueprints
vehicle_bps = world.get_blueprint_library().filter('*vehicle*')

# Spawn 50 vehicles randomly distributed throughout the map
for i in range(0,50):
    world.try_spawn_actor(random.choice(vehicle_bps), random.choice(spawn_points))
    
# Get the map spawn points
spawn_points = world.get_map().get_spawn_points()

for i, spawn_point in enumerate(spawn_points):
    # Draw in the spectator window the spawn point index
    world.debug.draw_string(spawn_point.location, str(i), life_time=100)
    # We can also draw an arrow to see the orientation of the spawn point
    # (i.e. which way the vehicle will be facing when spawned)
    world.debug.draw_arrow(spawn_point.location, spawn_point.location + spawn_point.get_forward_vector(), life_time=100)

for ind in range(0, 100):
    world.try_spawn_actor(random.choice(vehicle_bps), random.choice(spawn_points))

# Print all available blueprints
for actor in world.get_blueprint_library():
    print(actor)

# Print all available vehicle blueprints
for actor in world.get_blueprint_library().filter('vehicle'):
    print(actor)

vehicle_blueprint = world.get_blueprint_library().find('vehicle.audi.tt')