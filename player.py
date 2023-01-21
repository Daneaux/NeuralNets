from settings import *
import pygame as pg
import math

class Player:
    def __init__(self, game):
        self.game = game
        self.x, self.y = PLAYER_POS
        self.angle = PLAYER_ANGLE

    def movement(self):
        deltaY = math.sin(self.angle)
        deltaX = math.cos(self.angle)
        dx, dy = 0, 0
        speed = self.game.delta * PLAYER_SPEED
        deltaX *= speed
        deltaY *= speed 

        keys = pg.key.get_pressed()

        if keys[pg.K_w]:
            dx += deltaX
            dy += deltaY
        if keys[pg.K_s]:
            dx -= deltaX
            dy -= deltaY

        if keys[pg.K_a]:
            dx += deltaY
            dy -= deltaX
        if keys[pg.K_d]:
            dx -= deltaY
            dy += deltaX
 
        self.check_collision(dx, dy)

        scaled_rotation = PLAYER_ROT_SPEED * self.game.delta
        if keys[pg.K_LEFT]:
            self.angle -= scaled_rotation
        if keys[pg.K_RIGHT]:
            self.angle += scaled_rotation
        self.angle %= math.tau   # tau = 2 * pi

    def check_wall(self, x, y):
        return (x, y) not in self.game.map.world_map
    
    def check_collision(self, dx, dy):
        if self.check_wall(int(self.x + dx), int(self.y)):
            self.x += dx
        if self.check_wall(int(self.x), int(self.y + dy)):
            self.y += dy

    def draw(self):
        world_pos = (self.x * 100, self.y * 100)
        world_pos_end = (self.x * 100 + WIDTH * self.xyAngle[0], self.y * 100 + WIDTH * self.xyAngle[1])
        pg.draw.line(self.game.screen, 'yellow', world_pos, world_pos_end, 2)
        pg.draw.circle(self.game.screen, 'blue', world_pos, 15)

    def update(self):
        self.movement()

    @property
    def pos(self):
        return self.x, self.y

    @property
    def world_pos(self):
        return self.x * 100, self.y * 100

    @property
    def xyAngle(self):
        return math.cos(self.angle), math.sin(self.angle)

    @property
    def map_pos(self):
        return int(self.x), int(self.y)

    @property
    def world_rect(self):
        tlx, tly = self.map_pos
        tlx *= 100
        tly *= 100
        return pg.Rect(tlx, tly, 100, 100)

    

        
        
