#Code is modified from Felippe Roza's implementation which uses a 
#Double DQN with Priority Experience Replay, credits: https://github.com/FelippeRoza/carla-rl
#Additionally referenced Simonini Thomas's Deep learning course
#Credits: https://github.com/simoninithomas/Deep_reinforcement_learning_Course

import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
from carla import ColorConverter as cc
import time
import random
import numpy as np
import argparse
import logging
import pygame
import tensorflow as tf
import queue
import datetime
from PIL import Image
from multiprocessing import Process
from collections import deque
import re
import weakref
import math
import collections
import pickle
import matplotlib.pyplot as plt 

#Define hyperparameters of network including possible actions,
#learning-rate for gradient descent, number of training episodes,
#batch_size per gradient descent training iteration, and memory buffer size

test_flag = 0
model_name = "DQNetwork"

state_size = [84, 84, 1]

# discrete action-space described as (throttle, steer, brake)
action_space = np.array([(0.0, 0.0, 1.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0),
                        (0.5, 0.25, 0.0), (0.5, -0.25, 0.0), (0.5, 0.5, 0.0), (0.5, -0.5, 0.0)])
action_size = len(action_space)

learning_rate= 0.00025

# Training parameters
total_episodes = 50
max_steps = 200
batch_size = 64

max_tau = 5000  # tau is number of iterations until target network updated

# exploration params for epsilon greedy action selection
explore_start = 1.0  # exploration probability at start
explore_stop = 0.01  # minimum exploration probability
decay_rate = 0.00005  # exponential decay rate for exploration prob


gamma = 0.95  #discount rate to priortize more recent rewards 
pretrain_length = 100 
memory_size = 10000  #mem buffer size
memory_save_path = "memory.pkl"


#The following are helper functions used to interface with Carla server

# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)


# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    """Helper class to handle text output using pygame"""
    def __init__(self, font, width, height):
        lines = __doc__.split('\n')
        self.font = font
        self.line_space = 18
        self.dim = (780, len(lines) * self.line_space + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for n, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, n * self.line_space))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        self._render = not self._render

    def render(self, display):
        if self._render:
            display.blit(self.surface, self.pos)


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]

class CollisionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)


# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================


class LaneInvasionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        self.hud.notification('Crossed line %s' % ' and '.join(text))


# ==============================================================================
# -- GnssSensor ----------------------------------------------------------------
# ==============================================================================


class GnssSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(bp, carla.Transform(carla.Location(x=1.0, z=2.8)), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude


# ==============================================================================
# -- IMUSensor -----------------------------------------------------------------
# ==============================================================================


class IMUSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.accelerometer = (0.0, 0.0, 0.0)
        self.gyroscope = (0.0, 0.0, 0.0)
        self.compass = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.imu')
        self.sensor = world.spawn_actor(
            bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda sensor_data: IMUSensor._IMU_callback(weak_self, sensor_data))

    @staticmethod
    def _IMU_callback(weak_self, sensor_data):
        self = weak_self()
        if not self:
            return
        limits = (-99.9, 99.9)
        self.accelerometer = (
            max(limits[0], min(limits[1], sensor_data.accelerometer.x)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.y)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.z)))
        self.gyroscope = (
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.x))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.y))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.z))))
        self.compass = math.degrees(sensor_data.compass)


# ==============================================================================
# -- RadarSensor ---------------------------------------------------------------
# ==============================================================================


class RadarSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.velocity_range = 7.5 # m/s
        world = self._parent.get_world()
        self.debug = world.debug
        bp = world.get_blueprint_library().find('sensor.other.radar')
        bp.set_attribute('horizontal_fov', str(35))
        bp.set_attribute('vertical_fov', str(20))
        self.sensor = world.spawn_actor(
            bp,
            carla.Transform(
                carla.Location(x=2.8, z=1.0),
                carla.Rotation(pitch=5)),
            attach_to=self._parent)
        # We need a weak reference to self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda radar_data: RadarSensor._Radar_callback(weak_self, radar_data))

    @staticmethod
    def _Radar_callback(weak_self, radar_data):
        self = weak_self()
        if not self:
            return
        # To get a numpy [[vel, altitude, azimuth, depth],...[,,,]]:
        # points = np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4'))
        # points = np.reshape(points, (len(radar_data), 4))

        current_rot = radar_data.transform.rotation
        for detect in radar_data:
            azi = math.degrees(detect.azimuth)
            alt = math.degrees(detect.altitude)
            # The 0.25 adjusts a bit the distance so the dots can
            # be properly seen
            fw_vec = carla.Vector3D(x=detect.depth - 0.25)
            carla.Transform(
                carla.Location(),
                carla.Rotation(
                    pitch=current_rot.pitch + alt,
                    yaw=current_rot.yaw + azi,
                    roll=current_rot.roll)).transform(fw_vec)

            def clamp(min_v, max_v, value):
                return max(min_v, min(value, max_v))

            norm_velocity = detect.velocity / self.velocity_range # range [-1, 1]
            r = int(clamp(0.0, 1.0, 1.0 - norm_velocity) * 255.0)
            g = int(clamp(0.0, 1.0, 1.0 - abs(norm_velocity)) * 255.0)
            b = int(abs(clamp(- 1.0, 0.0, - 1.0 - norm_velocity)) * 255.0)
            self.debug.draw_point(
                radar_data.transform.location + fw_vec,
                size=0.075,
                life_time=0.06,
                persistent_lines=False,
                color=carla.Color(r, g, b))

# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    def __init__(self, parent_actor, hud, gamma_correction):
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        Attachment = carla.AttachmentType
        self._camera_transforms = [
            (carla.Transform(carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=8.0)), Attachment.SpringArm),
            (carla.Transform(carla.Location(x=1.6, z=1.7)), Attachment.Rigid),
            (carla.Transform(carla.Location(x=5.5, y=1.5, z=1.5)), Attachment.SpringArm),
            (carla.Transform(carla.Location(x=-8.0, z=6.0), carla.Rotation(pitch=6.0)), Attachment.SpringArm),
            (carla.Transform(carla.Location(x=-1, y=-bound_y, z=0.5)), Attachment.Rigid)]
        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB', {}],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)', {}],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)', {}],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)', {}],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)', {}],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
                'Camera Semantic Segmentation (CityScapes Palette)', {}],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)', {'range': '50'}],
            ['sensor.camera.dvs', cc.Raw, 'Dynamic Vision Sensor', {}],
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB Distorted',
                {'lens_circle_multiplier': '3.0',
                'lens_circle_falloff': '3.0',
                'chromatic_aberration_intensity': '0.5',
                'chromatic_aberration_offset': '0'}]]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(hud.dim[0]))
                bp.set_attribute('image_size_y', str(hud.dim[1]))
                if bp.has_attribute('gamma'):
                    bp.set_attribute('gamma', str(gamma_correction))
                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
            elif item[0].startswith('sensor.lidar'):
                self.lidar_range = 50

                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
                    if attr_name == 'range':
                        self.lidar_range = float(attr_value)


            item.append(bp)
        self.index = None

    def toggle_camera(self):
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.set_sensor(self.index, notify=False, force_respawn=True)

    def set_sensor(self, index, notify=True, force_respawn=False):
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else \
            (force_respawn or (self.sensors[index][2] != self.sensors[self.index][2]))
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[self.transform_index][1])
            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        self.recording = not self.recording
        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 3), 3))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / (2.0 * self.lidar_range)
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        elif self.sensors[self.index][0].startswith('sensor.camera.dvs'):
            # Example of converting the raw_data from a carla.DVSEventArray
            # sensor into a NumPy array and using it as an image
            dvs_events = np.frombuffer(image.raw_data, dtype=np.dtype([
                ('x', np.uint16), ('y', np.uint16), ('t', np.int64), ('pol', np.bool)]))
            dvs_img = np.zeros((image.height, image.width, 3), dtype=np.uint8)
            # Blue is positive, red is negative
            dvs_img[dvs_events[:]['y'], dvs_events[:]['x'], dvs_events[:]['pol'] * 2] = 255
            self.surface = pygame.surfarray.make_surface(dvs_img.swapaxes(0, 1))
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self.recording:
            image.save_to_disk('_out/%08d' % image.frame)



class World(object):
    def __init__(self, carla_world, hud, args):
        self.world = carla_world
        self.actor_role_name = args.rolename
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print('  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)
        self.hud = hud
        self.player = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.imu_sensor = None
        self.radar_sensor = None
        self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = args.filter
        self._gamma = args.gamma
        self.restart()
        self.world.on_tick(hud.on_world_tick)
        self.recording_enabled = False
        self.recording_start = 0

    def restart(self):
        self.player_max_speed = 1.589
        self.player_max_speed_fast = 3.713
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0
        # Get a random blueprint.
        blueprint = random.choice(self.world.get_blueprint_library().filter(self._actor_filter))
        blueprint.set_attribute('role_name', self.actor_role_name)
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        if blueprint.has_attribute('driver_id'):
            driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', driver_id)
        if blueprint.has_attribute('is_invincible'):
            blueprint.set_attribute('is_invincible', 'true')
        # set the max speed
        if blueprint.has_attribute('speed'):
            self.player_max_speed = float(blueprint.get_attribute('speed').recommended_values[1])
            self.player_max_speed_fast = float(blueprint.get_attribute('speed').recommended_values[2])
        else:
            print("No recommended values for 'speed' attribute")
        # Spawn the player.
        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
        while self.player is None:
            if not self.map.get_spawn_points():
                print('There are no spawn points available in your map/town.')
                print('Please add some Vehicle Spawn Point to your UE4 scene.')
                sys.exit(1)
            spawn_points = self.map.get_spawn_points()
            spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.imu_sensor = IMUSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud, self._gamma)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.player.get_world().set_weather(preset[0])

    def toggle_radar(self):
        if self.radar_sensor is None:
            self.radar_sensor = RadarSensor(self.player)
        elif self.radar_sensor.sensor is not None:
            self.radar_sensor.sensor.destroy()
            self.radar_sensor = None

    def tick(self, clock):
        self.hud.tick(self, clock)

    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy_sensors(self):
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.index = None

    def destroy(self):
        if self.radar_sensor is not None:
            self.toggle_radar()
        actors = [
            self.camera_manager.sensor,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor,
            self.imu_sensor.sensor,
            self.player]
        for actor in actors:
            if actor is not None:
                actor.destroy()

class HUD(object):
    def __init__(self, width, height):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 16), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        t = world.player.get_transform()
        v = world.player.get_velocity()
        c = world.player.get_control()
        compass = world.imu_sensor.compass
        heading = 'N' if compass > 270.5 or compass < 89.5 else ''
        heading += 'S' if 90.5 < compass < 269.5 else ''
        heading += 'E' if 0.5 < compass < 179.5 else ''
        heading += 'W' if 180.5 < compass < 359.5 else ''
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')
        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(world.player, truncate=20),
            'Map:     % 20s' % world.map.name,
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)),
            u'Compass:% 17.0f\N{DEGREE SIGN} % 2s' % (compass, heading),
            'Accelero: (%5.1f,%5.1f,%5.1f)' % (world.imu_sensor.accelerometer),
            'Gyroscop: (%5.1f,%5.1f,%5.1f)' % (world.imu_sensor.gyroscope),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (t.location.x, t.location.y)),
            'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            'Height:  % 18.0f m' % t.location.z,
            '']
        if isinstance(c, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', c.throttle, 0.0, 1.0),
                ('Steer:', c.steer, -1.0, 1.0),
                ('Brake:', c.brake, 0.0, 1.0),
                ('Reverse:', c.reverse),
                ('Hand brake:', c.hand_brake),
                ('Manual:', c.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(c.gear, c.gear)]
        elif isinstance(c, carla.WalkerControl):
            self._info_text += [
                ('Speed:', c.speed, 0.0, 5.556),
                ('Jump:', c.jump)]
        self._info_text += [
            '',
            'Collision:',
            collision,
            '',
            'Number of vehicles: % 8d' % len(vehicles)]
        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']
            distance = lambda l: math.sqrt((l.x - t.location.x)**2 + (l.y - t.location.y)**2 + (l.z - t.location.z)**2)
            vehicles = [(distance(x.get_location()), x) for x in vehicles if x.id != world.player.id]
            for d, vehicle in sorted(vehicles):
                if d > 200.0:
                    break
                vehicle_type = get_actor_display_name(vehicle, truncate=22)
                self._info_text.append('% 4dm %s' % (d, vehicle_type))

    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect((bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)

#Define architecture for Deep Q-Network 

class DQNetwork():
    def __init__(self, state_size, action_size, learning_rate, name='DQNetwork'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.possible_actions = np.identity(self.action_size, dtype=int).tolist()
        
        with tf.variable_scope(name):
            #inputs define image fed into NN
            self.inputs_ = tf.placeholder(tf.float32, [None, *state_size], name="inputs")
            
            #actions define array containing tuple of actions taken by system
            self.actions_ = tf.placeholder(tf.float32, [None, self.action_size], name="actions_")
            
            
            self.target_Q = tf.placeholder(tf.float32, [None], name="target") 
            
            self.conv1 = tf.layers.conv2d(inputs=self.inputs_, 
                                          filters=32,
                                          kernel_size=[8,8],
                                          strides=[4,4],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv1")
            self.conv1_out = tf.nn.elu(self.conv1, name="conv1_out")
            
            self.conv2 = tf.layers.conv2d(inputs=self.conv1_out, 
                                          filters=64,
                                          kernel_size=[4,4],
                                          strides=[2,2],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv2")
            self.conv2_out = tf.nn.elu(self.conv2, name="conv2_out")

            
            self.conv3 = tf.layers.conv2d(inputs=self.conv2_out, 
                                          filters=64,
                                          kernel_size=[3,3],
                                          strides=[2,2],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv3")
            #After multiple convolutions, use exponential linear unit
            #Activation function since DQN predicts continuous set of q-vals
            self.conv3_out = tf.nn.elu(self.conv3, name="conv3_out")
            
            self.flatten = tf.contrib.layers.flatten(self.conv3_out)
            
            self.fc = tf.layers.dense(inputs=self.flatten,
                                      units=512,
                                      activation=tf.nn.elu,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                      name="fc1")
            
            self.output = tf.layers.dense(inputs=self.fc,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          units=self.action_size,
                                          activation=None)
            
            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_)) #predicted Q-value computed by DNN by associating output of DNN w/ action tuples
            self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q)) #compute loss per each action-val 
            
            #return gradients for each weight of NN (change in weights after minimizing loss)
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            

    def predict_action(self, sess, explore_start, explore_stop, decay_rate, decay_step, state):
        #Implement epsilon-greedy action selection as policy where majority
        #of time optimal (greedy) action chosen and sometimes random action
        #chosen, probability of choosing randomly defined by explore_prob
        exp_tradeoff = np.random.rand()
        explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)

        if (explore_probability > exp_tradeoff):
            action_int = np.random.choice(self.action_size)
            action = self.possible_actions[action_int]
        else:
            #optimal action selection using output from DQN
            Qs = sess.run(self.output, feed_dict={self.inputs_: state.reshape((1, *state.shape))})
            action_int = np.argmax(Qs)
            action = self.possible_actions[int(action_int)]

        return action_int, action, explore_probability

'''maps continuous control to discrete action values
    used to convert control from autopilot to discrete values that the Q-network is using
    It works by computing the discrete action with the smaller euclidian distance from the
    continuous actions'''
def map_from_control(control, action_space):
    control_vector = np.array([control.throttle, control.steer, control.brake])
    distances = [] # euclidian distance list
    for control in action_space:
        distances.append(np.linalg.norm(control-control_vector)) # compute euclidian distance
        
    return np.argmin(distances)

   
#Use experience replay to efficiently train agent on random samples, reduce correlation
class Memory():
    def __init__(self, max_size, pretrain_length, action_space):
        self.buffer = deque(maxlen = max_size)
        self.pretrain_length = pretrain_length
        self.action_space = action_space
        self.action_size = len(action_space)
        self.possible_actions = np.identity(self.action_size, dtype=int).tolist()
    
    #function to add experience tuple to mem buffer
    def add(self, experience):
        self.buffer.append(experience)
        
    #sample randomly from buffer to train, outputs an array
    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size),
                                 size = batch_size,
                                 replace = True) 
        return [self.buffer[i] for i in index]
    
    #set agent on autopilot to retrieve useful experiences to fill buffer with
    def fill_memory(self, map, vehicle, camera_queue, sensors, autopilot = False):
        print("Started to fill memory")
        reset_environment(map, vehicle, sensors)

        vehicle.set_autopilot()

        for i in range(self.pretrain_length):

            if i % 10 == 0:
                print(i, "experiences stored")
            state = process_image(camera_queue)
            control = vehicle.get_control()
            #discretize continuous actions from carla-server into possible actions
            action_int = map_from_control(control, self.action_space)
            #select action to compute reward with
            action = self.possible_actions[action_int]
           
            time.sleep(0.25)

            reward = compute_reward(vehicle, sensors)
            done = isDone(reward)
            next_state = process_image(camera_queue)
            experience = state, action, reward, next_state, done
            self.add(experience)

            if done:
                reset_environment(map, vehicle, sensors)
            else:
                state = next_state

        print('Finished filing memory. %s experiences stored.' % self.pretrain_length)
        vehicle.set_autopilot(enabled = False)

    def save_memory(self, filename, object):
        handle = open(filename, "wb")
        pickle.dump(object, handle)

    def load_memory(self, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)


#carla sensors special actors to measure & stream data using listen() method
#can retrieve data upon timestep/action; must be attached to parent actor (vehicle)
#listen() method employs lambda func which recursively callback

class Sensors(object):
    """Class to keep track of all sensors added to the vehicle"""

    def __init__(self, world, vehicle):
        super(Sensors, self).__init__()
        self.world = world
        self.vehicle = vehicle
        self.camera_queue = queue.Queue() # queue to store images from buffer
        self.collision_flag = False # Flag for colision detection
        self.lane_crossed = False # Flag for lane crossing detection
        self.lane_crossed_type = '' # Which type of lane was crossed

        self.camera_rgb = self.add_sensors(world, vehicle, 'sensor.camera.rgb')
        self.collision = self.add_sensors(world, vehicle, 'sensor.other.collision')
        self.lane_invasion = self.add_sensors(world, vehicle, 'sensor.other.lane_invasion', sensor_tick = '0.5')

        self.sensor_list = [self.camera_rgb, self.collision, self.lane_invasion]

        #sensor uses lambda func to constantly retrieve data and return it
        #secondary func that uses lambda func will then manipulate data for use
        #such as by setting collision flag on 
        
        self.collision.listen(lambda collisionEvent: self.track_collision(collisionEvent))
        self.camera_rgb.listen(lambda image: self.camera_queue.put(image))
        self.lane_invasion.listen(lambda event: self.on_invasion(event))

    def add_sensors(self, world, vehicle, type, sensor_tick = '0.0'):

        sensor_bp = self.world.get_blueprint_library().find(type)
        try:
            sensor_bp.set_attribute('sensor_tick', sensor_tick)
        except:
            pass
        if type == 'sensor.camera.rgb':
            sensor_bp.set_attribute('image_size_x', '100')
            sensor_bp.set_attribute('image_size_y', '100')

        sensor_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        sensor = self.world.spawn_actor(sensor_bp, sensor_transform, attach_to=vehicle)
        return sensor

    def track_collision(self, collisionEvent):
        '''Whenever a collision occurs, the flag is set to True'''
        self.collision_flag = True

    def reset_sensors(self):
        '''Sets all sensor flags to False'''
        self.collision_flag = False
        self.lane_crossed = False
        self.lane_crossed_type = ''

    def on_invasion(self, event):
        '''Whenever the car crosses the lane, the flag is set to True'''
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        self.lane_crossed_type = text[0]
        self.lane_crossed = True

    def destroy_sensors(self):
        '''Destroy all sensors (Carla actors)'''
        for sensor in self.sensor_list:
            sensor.destroy()


#randomly spawn vehicle somewhere in environment, reset sensor data 
def reset_environment(map, vehicle, sensors):
    vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
    time.sleep(1)
    spawn_points = map.get_spawn_points()
    spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
    vehicle.set_transform(spawn_point)
    time.sleep(2)
    sensors.reset_sensors()
    
'''get the image from the buffer and process it. It's the state for vision-based systems'''
def process_image(queue):
    image = queue.get()
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    image = Image.fromarray(array).convert('L') # grayscale conversion
    image = np.array(image.resize((84, 84))) # convert to numpy array
    image = np.reshape(image, (84, 84, 1)) # reshape image
    return image


""" maps discrete actions into actual values to control the car"""
def map_action(action, action_space):
    control = carla.VehicleControl()
    control_sequence = action_space[action]
    control.throttle = control_sequence[0]
    control.steer = control_sequence[1]
    control.brake = control_sequence[2]

    return control

#Reward system for car 
def compute_reward(vehicle, sensors):
    max_speed = 14
    min_speed = 2
    speed = vehicle.get_velocity()
    #take magnitude of velocity along all axes to combine into single val
    vehicle_speed = np.linalg.norm([speed.x, speed.y, speed.z])

    #reward vehicle based on speed relative to min, max limits
    speed_reward = (abs(vehicle_speed) - min_speed) / (max_speed - min_speed)
    lane_reward = 0

    if (vehicle_speed > max_speed) or (vehicle_speed < min_speed):
        speed_reward = -0.05

    #heavily penalize if vehicle crosses lane or crashes
    if sensors.lane_crossed:
        if sensors.lane_crossed_type == "'Broken'" or sensors.lane_crossed_type == "'NONE'":
            lane_reward = -0.5
            sensors.lane_crossed = False

    if sensors.collision_flag:
        return -1

    else:
        return speed_reward + lane_reward
    
def isDone(reward):
    '''Return True if the episode is finished'''
    if reward <= -1:
        return True
    else:
        return False



def render(clock, world, display):
    clock.tick_busy_loop(30) # this sets the maximum client fps
    world.tick(clock)
    world.render(display)
    pygame.display.flip()
    

def update_target_graph():
    # This function helps to copy one set of variables to another
    # In our case we use it when we want to copy the parameters of DQN to Target_network
    # Thanks to Arthur Juliani https://github.com/awjuliani

    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "DQNetwork")
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "TargetNetwork")
    op_holder = []

    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))

    return op_holder


#Apply DQN architecture to Carla environment for building experiences in mem,
#And adjusting parameters of evaluation network 
def training(map, vehicle, sensors):
    
    
    tf.reset_default_graph()
    agent = DQNetwork(state_size, action_size, learning_rate, name="Agent")
    target_agent = DQNetwork(state_size, action_size, learning_rate, name="Target")
    print("NN init")
    writer = tf.summary.FileWriter("summary")
    tf.summary.scalar("Loss", agent.loss)
    write_op = tf.summary.merge_all()
    dqn_scores = []
    eps_loss = []
    
    saver = tf.train.Saver()

    
    #init memory 
    print("memory init")

    #begin filling up memory by setting car on autopilot 
    memory = Memory(max_size = memory_size, pretrain_length = pretrain_length, action_space = action_space)
    memory.fill_memory(map, vehicle, sensors.camera_queue, sensors, autopilot=True)
    memory.save_memory(memory_save_path, memory) #save memory to sample from 
    
    with tf.Session() as sess:
        print("session beginning")
        sess.run(tf.global_variables_initializer())
        writer.add_graph(sess.graph)
        m = 0
        decay_step = 0
        tau = 0
        print("beginning training")
        for episode in range(1, total_episodes):
            #init episode
            print("env reset")
            #reset environment, process input state using camera sensor
            reset_environment(map, vehicle, sensors)
            state = process_image(sensors.camera_queue)
            done = False
            start = time.time()
            episode_reward = 0
        
            #step through episode & retrieve data from DNN
            for step in range(max_steps):
                #increment tau and decay to account for updating target 
                #features and improving epsilon-greedy policy
                
                #Logic is that as agent trains more in an episode
                #it will learn more optimal actions and thus does not need
                #to take random actions as often
                #Require lower explore_prob to drive agent towards choosing
                #greedy actions more often as it is trained more 
                
                tau += 1
                decay_step += 1
                #return optimal action index (action_int), action's one-hot encoding (action), and explore_prob
                action_int, action, explore_probability = agent.predict_action(sess, explore_start, explore_stop, decay_rate, decay_step, state)
                print("action from NN received")
                #pass in action index & all possible actions from space, map it to carla control
                car_controls = map_action(action_int, action_space)
                vehicle.apply_control(car_controls)
                print("action applied to car")
                time.sleep(0.25)
                next_state = process_image(sensors.camera_queue)
                reward = compute_reward(vehicle, sensors)
                print("reward computed: " + str(reward))
                
                #compute reward, add experience to memory and inc to next state
                episode_reward += reward
                done = isDone(reward)
                memory.add((state, action, reward, next_state, done))
                state = next_state
            
                #begin learning by sampling a batch from memory
                #remember every experience is an array
                #when sampling a batch, useful to split up experiences into
                #constituent arrays to access individual samples for q-learning
                batch = memory.sample(batch_size)
                
                #state samples for batch
                s_mb = np.array([each[0] for each in batch], ndmin = 3)
                #action samples for batch
                a_mb = np.array([each[1] for each in batch])
                #reward samples for batch
                r_mb = np.array([each[2] for each in batch])
                #next state samples for batch
                next_s_mb = np.array([each[3] for each in batch], ndmin = 3)
                #done flag samples for batch
                dones_mb = np.array([each[4] for each in batch])
                
                target_Qs_batch = []

                #q-val for all next states to compute target q-val for current state
                Qs_next_state = sess.run(agent.output, feed_dict={agent.inputs_: next_s_mb})
                Qs_target_next_state = sess.run(target_agent.output, feed_dict={target_agent.inputs_: next_s_mb})
                
                for i in range(0, len(batch)):
                    terminal = dones_mb[i] #check if on last state of eps
                    action = np.argmax(Qs_next_state[i]) #store index of optimal action
                    if terminal:
                        target_Qs_batch.append((r_mb[i])) #if last state, append reward
                    else:
                        #formulate target q-vals by feed-fwd in network, using old weights for comparison 
                        #choose optimal action & compute q-val via target net rather than use argmax & same net here
                        #reduces overestimation
                        target = r_mb[i] + gamma*Qs_target_next_state[i][action]
                        #target = r_mb[i] + gamma*np.max(Qs_next_state[i])
                        target_Qs_batch.append(target)
                targets_mb = np.array([each for each in target_Qs_batch])
                
                #run session to compute loss & change NN weights, "newly" learned weights compared against old ones in training
                #feed in state inputs, actions to associate q-vals w/, and target q-vals for loss computation
                loss, _  = sess.run([agent.loss, agent.optimizer], feed_dict={agent.inputs_: s_mb, agent.target_Q: targets_mb, agent.actions_:a_mb})
                summary = sess.run(write_op, feed_dict={agent.inputs_: s_mb, agent.target_Q: targets_mb, agent.actions_:a_mb})
                writer.add_summary(summary, episode)
                writer.flush
                
                if tau > max_tau: #update target net weights every 5000 steps/actions 
                    update_target = update_target_graph()
                    sess.run(update_target)
                    m += 1
                    tau = 0
                    print("model updated")
                
                if episode % 5 == 0:
                    save_path = saver.save(sess, "./models/model.ckpt")
                    print("Model Saved")
                
                print('Episode: {}'.format(episode),
                                  'Total reward: {}'.format(episode_reward),
                                  'Explore P: {:.4f}'.format(explore_probability),
                                'Training Loss {:.4f}'.format(loss))
                
                if sensors.collision_flag == True: #if vehicle collides, reset
                    break
                
            #track loss and rewards to analyze data 
            eps_loss.append(loss)
            dqn_scores.append(episode_reward)
        print("Loss per episode")
        print(eps_loss)
        print("Reward per episode")
        print(dqn_scores)

                
#Test loop to be run after training and model gets saved locally 
def testing(map, vehicle, sensors):
    tf.reset_default_graph()
    with tf.Session() as sess:

        graph = tf.get_default_graph()
        inputs_ = graph.get_tensor_by_name("DQNetwork" + "/inputs:0")
        output = graph.get_tensor_by_name("DQNetwork" + "/output:0")

        episode_reward = 0
        reset_environment(map, vehicle, sensors)

        while True:
            state = process_image(sensors.camera_queue)
            Qs = sess.run(output, feed_dict={inputs_: state.reshape((1, *state.shape))})
            action_int = np.argmax(Qs)
            #print(Qs)
            #print(action_int)

            car_controls = map_action(action_int, action_space)
            vehicle.apply_control(car_controls)
            reward = compute_reward(vehicle, sensors)
            episode_reward += reward
            done = isDone(reward)

            if done:
                print("EPISODE ended", "TOTAL REWARD {:.4f}".format(episode_reward))
                reset_environment(map, vehicle, sensors)
                episode_reward = 0

            else:
                time.sleep(0.25)
                
               
#Used to open carla server and determine if model is testing or training
def control_loop(vehicle_id, host, port):
    actor_list = []
    try:
        #setup Carla
        client = carla.Client(host, port)
        client.set_timeout(15.0)
        world = client.get_world()
        map = world.get_map()
        vehicle = next((x for x in world.get_actors() if x.id == vehicle_id), None) #get the vehicle actor according to its id
        sensors = Sensors(world, vehicle)
        print("beginning training loop")
        if test_flag:
            testing(map, vehicle, sensors)
        else:
            training(map, vehicle, sensors)  
         
        
    finally:
        print("done")
        sensors.destroy_sensors()       

#Use multithreading to run neural network training and local pygame window
#rendering at the same time to view the agent as it learns
def render_loop(args):
    #loop responsible for rendering the simulation client
    pygame.init()
    pygame.font.init()
    world = None
    
    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(15.0)
        
        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        hud = HUD(args.width, args.height)
        world = World(client.get_world(), hud, args)
        print("beginning process loop")
        p = Process(target=control_loop, args=(world.player.id, args.host, args.port))
        p.start()
        #control_loop(world.player.id, args.host, args.port)
        clock = pygame.time.Clock()
        
        
        
        while True:
            render(clock, world, display) #pygame output update
        
    finally:
        if (world and world.recording_enabled):
            client.stop_recorder()

        if world is not None:
            world.destroy()
        

        pygame.quit()

#Main loop will parse commands from here and begin execution 
#Can configure carla-server port, car model, window resolution as well
def main():
    argparser = argparse.ArgumentParser(
        description='CARLA RL')
    argparser.add_argument(
        '--test',
        action='store_true',
        dest='test',
        help='test a trained model')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=5000,
        type=int,
        help='TCP port to listen to (default: 5000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='800x600',
        help='window resolution (default: 800x600)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.audi.tt',
        help='actor filter (default: "vehicle.audi.tt")')
    argparser.add_argument(
        '--rolename',
        metavar='NAME',
        default='hero',
        help='actor role name (default: "hero")')
    argparser.add_argument(
        '--gamma',
        metavar='NAME',
        default='0',
        help='gamma correction')
    
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:

        render_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':

    main()