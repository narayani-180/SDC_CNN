"""
Welcome to CARLA manual control.

Use ARROWS or WASD keys for control.

    W            : throttle
    S            : brake
    AD           : steer
    Q            : toggle reverse
    Space        : hand-brake
    P            : toggle autopilot

    R            : restart level

STARTING in a moment...
"""

from __future__ import print_function

import argparse
import logging
import random
import time

######

import cv2   #for image
import numpy as np 
import os     # for changing dir
import scipy.io as io  ## for saving image file 
print(os.getcwd())     #print current dir

##### %
try:
    import pygame    ## used for keyword input and gamepad etc.
    from pygame.locals import K_DOWN
    from pygame.locals import K_LEFT
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SPACE
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_d
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_n
    from pygame.locals import K_m
    from pygame.locals import K_j
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_w
    from pygame.locals import K_f
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')



####CARLA 
from carla import image_converter
from carla import sensor
from carla.client import make_carla_client, VehicleControl
from carla.planner.map import CarlaMap
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError
from carla.util import print_over_same_line
####CARLA %

#Import load_model function from model.py
# from model import load_model

i=0
WINDOW_WIDTH = 320
WINDOW_HEIGHT = 240
MINI_WINDOW_WIDTH = 320
MINI_WINDOW_HEIGHT = 180

def make_carla_settings(args):                                       ## normal fucn in python
    """Make a CarlaSettings object with the settings we need."""
    settings = CarlaSettings()
    settings.set(
        SynchronousMode=False,
        SendNonPlayerAgentsInfo=True,
        NumberOfVehicles=0,
        NumberOfPedestrians=0,
        WeatherId=random.choice([1, 3, 7, 8, 14]),
        QualityLevel=args.quality_level)
    settings.randomize_seeds()
    camera0 = sensor.Camera('CameraRGB')                           # Set RGB Camera
    camera0.set_image_size(WINDOW_WIDTH, WINDOW_HEIGHT)            #
    camera0.set_position(2.0, 0.0, 1.4)                            #
    camera0.set_rotation(0.0, 0.0, 0.0)                            #
    settings.add_sensor(camera0)                                   # Adding camera in Carla
    
    return settings                     ##return settings


class Timer(object):
    def __init__(self):
        self.step = 0
        self._lap_step = 0
        self._lap_time = time.time()

    def tick(self):
        self.step += 1

    def lap(self):
        self._lap_step = self.step
        self._lap_time = time.time()

    def ticks_per_second(self):
        return float(self.step - self._lap_step) / self.elapsed_seconds_since_lap()

    def elapsed_seconds_since_lap(self):
        return time.time() - self._lap_time

class CarlaGame(object):
    def __init__(self, carla_client, args):
        self.client = carla_client
        self._carla_settings = make_carla_settings(args)
        self._timer = None
        self._display = None
        self._main_image = None
        self._mini_view_image1 = None
        self._enable_autopilot = args.autopilot
        self._map_view = None
        self._is_on_reverse = False
        self._display_map = args.map
        self._city_name = None
        self._map = None
        self._map_shape = None
        self._map_view = None
        self._position = None
        self._agent_positions = None
        
        self.Model = load_model()
        self.Model.load_weights('./img/model_th1.h5')
        
        

        ################################################################ 
        
        ## - i is for data image sequence 
        self._i = 0   ##frame no.
        ## - val1 is for steering angle
        self._val1 = 0
        ## - val2 is for throttle
        self._val2 = 0
        ## - val3 is for data recording start
        self._val3 = 0   ## 1 recording start and if 0 recording stop
        ## - velocity of car
        self._velocity = 0
        ##for break
        self._val4 = 0
        ## for commands
        self._val5 = 'S'
        ## - data
        self._data = {}   ## key "throttle ,angle " = value img in array form

        ################################################################@
    
    ##EXECUTE FUNCTION IS MAIN LOOP 
    def execute(self):
        """Launch the PyGame."""
        pygame.init()
        self._initialize_game()  ##calling initialize_game func
        try:
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT: #checking for end code
                        print("End")
                        return
                    # elif event.type == K_f:
                    #     self._val3 = 1
                    #     print("AAAA")
                self._on_loop()           #img form carla keyboard input   calling on_loop func  *
                self._on_render()         # data saving in self._data calling on_render func *
        finally:
            ## when event.type == pygame.QUIT is true then this execute
            #######################################      SAVING THE DATA ON MEMORY 
            print("Saving data to :")
            if not os.path.exists('img'):
                os.makedirs('img')
            os.chdir('D:\CarlaSimulator\PythonClient\img')     ## SET FOLDER FOR STORING IMAGE DATA  data - .mat file
            io.savemat('img\\data',self._data)   #SAVING....
            print("Data saved")
            pygame.quit()
            
            ####################################
    
    def _initialize_game(self):   ## initialize CARLA things
        self._on_new_episode()     # calling on_new_episode func
        
        if self._city_name is not None:
            self._map = CarlaMap(self._city_name, 0.1643, 50.0)
            self._map_shape = self._map.map_image.shape
            self._map_view = self._map.get_map(WINDOW_HEIGHT)

            extra_width = int((WINDOW_HEIGHT/float(self._map_shape[0]))*self._map_shape[1])
            self._display = pygame.display.set_mode(
                (WINDOW_WIDTH + extra_width, WINDOW_HEIGHT),
                pygame.HWSURFACE | pygame.DOUBLEBUF)
        else:
            self._display = pygame.display.set_mode(
                (WINDOW_WIDTH, WINDOW_HEIGHT),
                pygame.HWSURFACE | pygame.DOUBLEBUF)
        
        logging.debug('pygame started')
    
    def _on_new_episode(self):
        self._carla_settings.randomize_seeds()
        self._carla_settings.randomize_weather()
        scene = self.client.load_settings(self._carla_settings)
        if self._display_map:
            self._city_name = scene.map_name
        number_of_player_starts = len(scene.player_start_spots)
        player_start = np.random.randint(number_of_player_starts)
        print('Starting new episode...')
        self.client.start_episode(player_start)
        self._timer = Timer()
        self._is_on_reverse = False
    
    
    def _on_loop(self):    # called in execute func
        self._timer.tick()
        
        measurements, sensor_data = self.client.read_data()
        
        self._main_image = sensor_data.get('CameraRGB', None)      # getting current image from CARLA
        #self._mini_view_image1 = sensor_data.get('CameraDepth', None)
        
        ## Print measurements every second.
        if self._timer.elapsed_seconds_since_lap() > 1.0:
            if self._city_name is not None:
                # Function to get car position on map.
                map_position = self._map.convert_to_pixel([
                    measurements.player_measurements.transform.location.x,
                    measurements.player_measurements.transform.location.y,
                    measurements.player_measurements.transform.location.z])
                # Function to get orientation of the road car is in.
                lane_orientation = self._map.get_lane_orientation([
                    measurements.player_measurements.transform.location.x,
                    measurements.player_measurements.transform.location.y,
                    measurements.player_measurements.transform.location.z])
                    
                self._print_player_measurements_map(
                    measurements.player_measurements,
                    map_position,
                    lane_orientation)
            
            else:
                self._print_player_measurements(measurements.player_measurements)
    
            # Plot position on the map as well.

            self._timer.lap()
        ##get keyboard commands->
        control = self._get_keyboard_control(pygame.key.get_pressed()) #Returns object of VehicleControl class
        ##############  IMP (used get_keyboard control func)
        # Set the player position
        if self._city_name is not None:
            self._position = self._map.convert_to_pixel([
                measurements.player_measurements.transform.location.x,
                measurements.player_measurements.transform.location.y,
                measurements.player_measurements.transform.location.z])
            self._agent_positions = measurements.non_player_agents
        #print("huwelebvncbhergvhuierhfierb ", self._enable_autopilot)
        if control is None:
            self._on_new_episode()
        elif self._enable_autopilot: 
            ############################################
            ############################################CODE FOR TESTING THE CAR WITH INPUT OF COMMAND AND LOADING THE MODEL
            ############################################
        else:
            print("p again pressed")
            self.client.send_control(control)     ##sending control to CARLA # on_loop end 
 
    def _get_keyboard_control(self, keys):  ##called in on_loop fucnc
        """
        Return a VehicleControl message based on the pressed keys. Return None
        if a new episode was requested.
        """
        
        if keys[K_r]:
            return None
        
        

        control = VehicleControl()  # control = model.predict(self._main_image,self._depth_image)
        
        #return control
        ################################################### YOU CAN EDIT IT ACCORDING YOU....
        if keys[K_n]:
            self._val5 = 'L'
        if keys[K_m]:
            self._val5 = 'R'
        if keys[K_j] :
            self._val5 = 'S'
        if keys[K_LEFT] or keys[K_a]:
            self._val2 = 0.50
            control.throttle = self._val2
            if self._val1 >= 0 :
                self._val1 = -0.25
            elif self._val1 > -1 :
                self._val1 = self._val1 -0.1
            control.steer = self._val1
            print("Left")
        if keys[K_RIGHT] or keys[K_d]:
            self._val2 = 0.50
            control.throttle = self._val2
            if self._val1 <= 0 :
                self._val1 = 0.25
            elif self._val1 < 1 :
                self._val1 = self._val1 + 0.1
            control.steer =self._val1
            print("Right")
        # else:
        #     self._val1 = 0
        #     control.steer =self._val1

        if keys[K_UP] or keys[K_w]:
            if self._val2 == 0 :
               self._val2 = 0.35
            elif self._val2 < 2 :
               self._val2 = self._val2 + 0.25
            control.throttle = self._val2     #Imp Line

        if keys[K_DOWN] or keys[K_s]:
            self._val4 = 1
            control.brake = self._val4
            self._val2 = 0
            control.throttle = self._val2
            
        ###  
        
        if keys[K_SPACE]:
            #control.hand_brake = True
            self._enable_autopilot = False
        if keys[K_f]:#starting val3 = 1
            self._val3 = 1 - self._val3  # for starting  and stoping data collection 
        if keys[K_q]:
            self._is_on_reverse = not self._is_on_reverse
        if keys[K_p]:
            self._enable_autopilot = not self._enable_autopilot
        control.reverse = self._is_on_reverse
        ################################################################################
        #print(control)
        return control

    def _print_player_measurements_map(self, player_measurements, map_position, lane_orientation):
        message = 'Step {step} ({fps:.1f} FPS): '
        message += 'Map Position ({map_x:.1f},{map_y:.1f}) '
        message += 'Lane Orientation ({ori_x:.1f},{ori_y:.1f}) '
        message += '{speed:.2f} km/h, '
        message += '{other_lane:.0f}% other lane, {offroad:.0f}% off-road'
        message = message.format(
            map_x=map_position[0],
            map_y=map_position[1],
            ori_x=lane_orientation[0],
            ori_y=lane_orientation[1],
            step=self._timer.step,
            fps=self._timer.ticks_per_second(),
            speed=player_measurements.forward_speed * 3.6,
            other_lane=100 * player_measurements.intersection_otherlane,
            offroad=100 * player_measurements.intersection_offroad)
        print_over_same_line(message)
        # self._velocity = player_measurements.forward_speed *3.6

    def _print_player_measurements(self, player_measurements):
        message = 'Step {step} ({fps:.1f} FPS): '
        message += '{speed:.2f} km/h, '
        message += '{other_lane:.0f}% other lane, {offroad:.0f}% off-road'
        message = message.format(
            step=self._timer.step,
            fps=self._timer.ticks_per_second(),
            speed=player_measurements.forward_speed * 3.6,
            other_lane=100 * player_measurements.intersection_otherlane,
            offroad=100 * player_measurements.intersection_offroad)
        print_over_same_line(message)
        # self._velocity = player_measurements.forward_speed *3.6

    def _on_render(self):                                         ## called in execute
        gap_x = (WINDOW_WIDTH - 2 * MINI_WINDOW_WIDTH) / 3
        mini_image_y = WINDOW_HEIGHT - MINI_WINDOW_HEIGHT - gap_x

        if self._main_image is not None:
            array = image_converter.to_rgb_array(self._main_image)  #-
            self.curr_img = image_converter.to_rgb_array(self._main_image)
            ################################################################################################   SAVING IMAGE DATA IN %SELF._DATA%
            #print(self._velocity)
            #print("\n")
            ###self._data['data{}_angle_{}_throttle_{}_command_{}_brake_{}'.format(self._i,format(self._val1,'.2f'),format(self._val2,'.2f'),format(self._val5,''),format(self._val4,'.2f'))] = array
            self._i +=1   ##FRAME NUMBER INCREASE BY ONE
            
            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            self._display.blit(surface, (0, 0))

        if self._mini_view_image1 is not None:
            array = image_converter.depth_to_logarithmic_grayscale(self._mini_view_image1)
            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            self._display.blit(surface, (gap_x, mini_image_y))
        
        if self._map_view is not None:
            array = self._map_view
            array = array[:, :, :3]
            
            new_window_width = \
                (float(WINDOW_HEIGHT) / float(self._map_shape[0])) * \
                float(self._map_shape[1])
            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            
            w_pos = int(self._position[0]*(float(WINDOW_HEIGHT)/float(self._map_shape[0])))
            h_pos = int(self._position[1] *(new_window_width/float(self._map_shape[1])))
            
            pygame.draw.circle(surface, [255, 0, 0, 255], (w_pos, h_pos), 6, 0)
            for agent in self._agent_positions:
                if agent.HasField('vehicle'):
                    agent_position = self._map.convert_to_pixel([
                        agent.vehicle.transform.location.x,
                        agent.vehicle.transform.location.y,
                        agent.vehicle.transform.location.z])
                    
                    w_pos = int(agent_position[0]*(float(WINDOW_HEIGHT)/float(self._map_shape[0])))
                    h_pos = int(agent_position[1] *(new_window_width/float(self._map_shape[1])))
                    
                    pygame.draw.circle(surface, [255, 0, 255, 255], (w_pos, h_pos), 4, 0)
            
            self._display.blit(surface, (WINDOW_WIDTH, 0))
        
        pygame.display.flip()

def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='localhost',
        help='IP of the host server (default: localhost)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '-q', '--quality-level',
        choices=['Low', 'Epic'],
        type=lambda s: s.title(),
        default='Low',
        help='graphics quality level, a lower level makes the simulation run considerably faster')
    argparser.add_argument(
        '-m', '--map',
        action='store_true',
        help='plot the map of the current city')
    args = argparser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    while True:
        try:

            with make_carla_client(args.host, args.port) as client:
                game = CarlaGame(client, args)                                          ## create a object of CarlaGame Class
                ############main function - execute
                game.execute()                           ##calling execute method(func) of CarlaGame class
                break

        except TCPConnectionError as error:
            logging.error(error)
            time.sleep(1)


if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


################################################################
