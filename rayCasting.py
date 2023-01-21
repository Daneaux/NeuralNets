import pygame as pg
import math
from settings import *

class Box:
    def __init__(self, game, tx, ty, bx, by):
        self.game = game

        

class PointAndLength:
    def __init__(self, game, x, y, length):
        self.x = x
        self.y = y
        self.length = length
        self.game = game

    def draw(self, p2):
        pg.draw.line(self.game.screen, "blue", self.Point, p2.Point, 3)

    def draw_dot(self, color):
        pg.draw.circle(self.game.screen, color, self.Point, 6)

    @property
    def Point(self):
        return (self.x, self.y)



class RayCasting:
    def __init__(self, game):
        self.game = game

    def ray_cast(self):
        ray_angle = self.game.player.angle # - HALF_FOV + 0.0001
        #temp
        #pg.draw.rect(self.game.screen, "red", self.game.player.world_rect)
        wrect = self.game.player.world_rect
        startPos = self.game.player.world_pos
        for ray in range(NUM_RAYS):
            ray_angle += DELTA_ANGLE
            depth = MAX_DEPTH
            isHit = False
            while not isHit and depth>0:
                # solve for y, given X
                p1, p2 = self.solveForNextHit(
                        ray_angle, 
                        startPos,
                        wrect)
                #p1.draw(p2)                
        
                if p1.length < p2.length:
                    p1 = p1
                else:
                    p1 = p2

                lp = (p1.x // 100, p1.y // 100)
                if lp in self.game.map.world_map:
                    isHit = True
                    #pg.draw.circle(self.game.screen, "magenta", p2.Point, 25)
                    pg.draw.line(self.game.screen, "purple", self.game.player.world_pos, (p1.x, p1.y), 5)
                else:
                    depth -= 1
                    startPos = (p1.x, p1.y)
                    foo = lp[0]
                    bar = lp[1]
                    wrect = pg.Rect(math.floor(foo) * 100, math.floor(bar) * 100, 100, 100)
        
    def solveForNextHit(self, angle, start_world_pos, world_rect):
        wx = start_world_pos[0]
        wy = start_world_pos[1]

        q1 = math.pi / 2
        q2 = math.pi
        q3 = math.pi + q1
        q4 = 2 * math.pi

        # bottom right quadrant
        if  angle < q1 and angle > 0.0 :
            dx = world_rect.right - wx
            dy = world_rect.bottom - wy

            wRayLength_dy = dy / (math.sin(angle))
            x_intersect = wRayLength_dy * (math.cos(angle))
            p1 = PointAndLength(self.game, wx + x_intersect, wy + dy, wRayLength_dy)

            wRayLength_dx = dx / (math.cos(angle))
            y_intersect = wRayLength_dx * math.sin(angle)
            p2 = PointAndLength(self.game, wx + dx, wy + y_intersect, wRayLength_dx)

            return p1, p2
        
        # top right quad
        if angle >= q3 and angle < q4:
            #angle = angle - q3 + q1 
            dx = world_rect.right - wx
            dy = wy - world_rect.top

            wRayLength_dy = abs(dy / (math.sin(angle)))
            x_intersect = abs(wRayLength_dy * (math.cos(angle)))
            p1 = PointAndLength(self.game, wx + x_intersect, wy - dy, wRayLength_dy)

            wRayLength_dx = abs(dx / (math.cos(angle)))
            y_intersect = abs(wRayLength_dx * math.sin(angle))
            p2 = PointAndLength(self.game, wx + dx, wy - y_intersect, wRayLength_dx)

            return p1, p2

        # top left quad
        if angle >= q2 and angle < q3:
            dx = wx - world_rect.left
            dy = wy - world_rect.top

            wRayLength_dy = abs(dy / (math.sin(angle)))
            x_intersect = abs(wRayLength_dy * (math.cos(angle)))
            p1 = PointAndLength(self.game, wx - x_intersect, wy - dy, wRayLength_dy)

            wRayLength_dx = abs(dx / (math.cos(angle)))
            y_intersect = abs(wRayLength_dx * math.sin(angle))
            p2 = PointAndLength(self.game, wx - dx, wy - y_intersect, wRayLength_dx)

            return p1, p2
        
        # bottom left quad
        if angle >= q1 and angle < q2:
            dx = wx - world_rect.left
            dy = world_rect.bottom - wy

            wRayLength_dy = abs(dy / (math.sin(angle)))
            x_intersect = abs(wRayLength_dy * (math.cos(angle)))
            p1 = PointAndLength(self.game, wx - x_intersect, wy + dy, wRayLength_dy)

            wRayLength_dx = abs(dx / (math.cos(angle)))
            y_intersect = abs(wRayLength_dx * math.sin(angle))
            p2 = PointAndLength(self.game, wx - dx, wy + y_intersect, wRayLength_dx)

            return p1, p2
        
        return PointAndLength(self.game, 0, 0, 1), PointAndLength(self.game, 0, 0, 1)

    def update(self):
        self.ray_cast()



"""         p1 = PointAndLength(self.game, 
            self.game.player.world_pos[0], 
            self.game.player.world_pos[1],
            100)

        p2 = PointAndLength(self.game, 
            self.game.player.world_pos[0] + math.cos(ray_angle)*100,
            self.game.player.world_pos[1] + math.sin(ray_angle)*100,
            100) """

        #p1.draw(p2)
