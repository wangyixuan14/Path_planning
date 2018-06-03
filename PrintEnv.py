import numpy as np
import pyglet
import time
import sys
import matplotlib as plot
pyglet.clock.set_fps_limit(10000)

class PrintEnv():

    #define global variable,such as dimensionality
    n_features = 2
    n_actions = 4
    n_sensor = 2

    sensor_max = 2000
    start_point = [120,10]

    dt = 0.1
    dw = 5

    viewer = None
    viewer_xy =(200,600)

    def __init__(self,length = 10,hight = 5,velocity = 0.5,theta= np.pi/12,discrete_action = True):

        self.length = length
        # self.velocity = velocity
        self.hight = hight
        self.velocity = velocity
        self.theta = theta
        self.dw = 0.5

        self.is_discrete_action = discrete_action
        if discrete_action:
            self.actions = np.array([[1,1],[-1,-1],[1,-1],[-1,1]])
        else:
            self.action_bound = [-1,1]

        self.terminal = False

        self.print_info = np.array([0,0,0,self.length,self.hight],dtype = np.float64)
        # self.sequence_coords = np.array([[200,50],
        #     [300,50],
        #     [300,1000],
        #     [200,1000],
        #     [100,800],
        #     [200,600],
        #     [100,200],
        #     [200,50]])

        self.sequence_coords = np.array([[100, 10],[200, 10],[200, 500],[80,500]])
        self.sensor_info = self.sensor_max + np.zeros((self.n_sensor,3))

        #define constant variable

    def step(self,action):    #All the steps of the algorothm.
        s = self._get_state()
        # if self.is_discrete_action:
        action_c = self.actions[action]
        # else:
        #     action = np.clip(action,*self.action_bound)[0]

        if 0 <self.print_info[2] < np.pi :
            self.print_info[2] += action_c[0] * self.theta
            self.print_info[3] += action_c[1] * self.dw
            self.print_info[:2] = self.print_info[:2] +\
                              self.velocity * self.dt * np.array([np.cos(self.print_info[2]),np.sin(self.print_info[2])])
        else:
            self.print_info[:2] += [0,0]
            self.print_info[2] = np.pi/2
        self._update_sensor()
        s_ = self._get_state()

        if  s_ == -1:
            if action == 0:
                reward = 1
            elif action ==1:
                reward = 1
            elif action == 2:
                reward = -1
            elif action == 3:
                reward = 2
        else :
            if action == 0:
                reward = 2
            elif action == 1:
                reward = -1
            elif action == 2:
                reward = 1
            elif action == 3:
                reward = 1


        # if st % 50 == 0:
        print(action,action_c,reward)
        return s_,reward,self.terminal

    def reset(self):
        self.terminal = False
        self.print_info[:3] = np.array([*self.start_point,np.pi/2])
        self._update_sensor()
        return self._get_state()

    def render(self):
        if self.viewer is None:
            self.viewer = Viewer(*self.viewer_xy,self.print_info,self.sensor_info,self.sequence_coords)

        self.viewer.render()

    def sample_action(self):
        if self.is_discrete_action:
            a = np.random.choice(list(range(4)))
        else:
            a = np.random.uniform(*self.action_bound,size=self.n_actions)
        return a

    def set_fps(self,fps = 30):
        pyglet.clock.set_fps_limit(fps)

    def distance_sensor(self):

        pass
    def _get_state(self):
        d = self.sensor_info[:,0] - self.print_info[3]/2
        # s = self.sensor_info[:,0]
        if d[0]>=d[1]:
            s = 1
        else:
            s = -1
        return s

    def _update_sensor(self):
        cx,cy = self.print_info [:2]

        n_sensors = len(self.sensor_info)
        sensor_theta = np.linspace(-np.pi/2,np.pi/2,n_sensors)
        xs = cx + np.zeros(n_sensors,)+self.sensor_max * np.sin(sensor_theta)
        ys = cy + np.zeros((n_sensors,))

        self.sensor_info[:,-2:] = np.vstack([xs,ys]).T

        q = np.array([cx,cy])
        for si in range (len(self.sensor_info)):
            s = self.sensor_info[si,-2:] - q
            possible_sensor_distance = [self.sensor_max]
            possible_intersections = [self.sensor_info[si,-2:]]

            #sequence collision
            for oi in range (len(self.sequence_coords)):
                p = self.sequence_coords[oi]
                r = self.sequence_coords[(oi+1)%len(self.sequence_coords)] - self.sequence_coords[oi]

                if np.cross(r,s) != 0:
                    t = np.cross((q - p),s) / np.cross(r,s)
                    u = np.cross((q - p),r) / np.cross(r,s)
                    if 0<= t <= 1 and 0<= u <= 1:
                        intersection = q + u * s
                        possible_intersections.append(intersection)
                        possible_sensor_distance.append(np.linalg.norm(u*s))

            distance = np.min(possible_sensor_distance)
            distance_index = np.argmin(possible_sensor_distance)
            self.sensor_info[si, 0] = distance
            self.sensor_info[si, -2:] = possible_intersections[distance_index]
            if distance < self.print_info[-1]/2:
                self.terminal = True

class Viewer(pyglet.window.Window):
    color = {
        'background':[1]*3 + [1]
    }
    fps_display = pyglet.clock.ClockDisplay()
    bar_thc = 2

    def __init__(self,width,height,print_info,sensor_info,sequence_coords):
        super(Viewer, self).__init__(resizable= True,caption='Print Env',vsync=True,fullscreen=False)
        self.set_location(x = 200,y= 50)
        pyglet.gl.glClearColor(*self.color['background'])
        self.width = width
        self.height = height
        self.print_info =print_info
        self.sensor_info = sensor_info

        self.batch = pyglet.graphics.Batch()
        background = pyglet.graphics.OrderedGroup(0)
        foreground = pyglet.graphics.OrderedGroup(1)

        #sensor
        self.sensors = []
        line_coord = [0,0] * 2
        sequence_color = (78,78,78) *2
        for i in range (len(self.sensor_info)):
            self.sensors.append(self.batch.add(2,pyglet.gl.GL_LINES,foreground,('v2f',line_coord),('c3B',sequence_color)))

        #print
        print_box = [0,0] * 4
        print_color = (249,86,86) * 4
        self.print = self.batch.add(4,pyglet.gl.GL_QUADS,foreground,('v2f',print_box),('c3B',print_color))

        print_point = [0,0] * 4
        point_color = (0,0,256) * 4
        self.point = self.batch.add(4,pyglet.gl.GL_QUADS,foreground,('v2f',print_point),('c3B',point_color))

        #sequence
        sequence_color = (134,81,243) * 4
        self.sequence = self.batch.add(4, pyglet.gl.GL_QUADS, background, ('v2f', sequence_coords.flatten()),
                            ('c3B', sequence_color))

    def render(self):
            pyglet.clock.tick()
            self._update()
            self.switch_to()
            self.dispatch_events()
            self.dispatch_event('on_draw')
            self.flip()

    def on_draw(self):
            self.clear()
            self.batch.draw()

    def _update(self):
        cx,cy,r,l,w = self.print_info

            # sensors
        for i,sensor in enumerate(self.sensors):
            sensor.vertices = [cx,cy,*self.sensor_info[i,-2:]]

            # print
        xys = [
            [cx + l/2, cy + w/2],
            [cx - l/2, cy + w/2],
            [cx - l/2, cy - w/2],
            [cx + l/2, cy - w/2]
        ]
        r_xys = []
        for x,y in xys:
            r_xys += [x,y]
        self.print.vertices = r_xys


        point_xys = [
            [cx + 1, cy + 1],
            [cx - 1, cy + 1],
            [cx - 1, cy - 1],
            [cx + 1, cy - 1]
        ]
        point_r_xys = []
        for x, y in point_xys:
            point_r_xys += [x, y]
        self.point.vertices = point_r_xys


if __name__ == '__main__':
    np.random.seed(2)
    env = PrintEnv()
    env.set_fps(50)
    for ep in range(20):
        s = env.reset()

        while True:
            env.render()
            s,r,done = env.step(env.sample_action())
            if done:
                break