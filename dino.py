import cv2
import pygame
import numpy as np
import random

pygame.init()
cap = cv2.VideoCapture(0)
frame_rate = 30

ret, frame = cap.read()
h, w, _ = frame.shape
h = h * 5 // 7
w = w * 5 // 7
screen = pygame.display.set_mode((w, h))

dino_y = h - 100
dino_vel = 0
gravity = 5
is_jumping = False

obstacles = []  # list of dicts: {'x', 'y', 'w', 'h'}
obstacle_speed = 17
spawn_timer = 0

def jump_fn():
    global dino_vel, is_jumping
    if not is_jumping:
        dino_vel = -34
        is_jumping = True

clock = pygame.time.Clock()
running = True

while running:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()
    if keys[pygame.K_SPACE]:  # space triggers jump
        jump_fn()
    
    ret, frame = cap.read()
    frame = cv2.resize(frame, (w, h))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = np.rot90(frame)
    frame_surface = pygame.surfarray.make_surface(frame)

    dino_y += dino_vel
    dino_vel += gravity
    if dino_y >= h - 100:  # ground level
        dino_y = h - 100
        is_jumping = False

    spawn_timer -= 1
    if spawn_timer <= 0:
        obs_h = random.randint(75, 167)
        obstacles.append({'x': w, 'y': h - obs_h, 'w': 20, 'h': obs_h})
        spawn_timer = random.randint(40, 90)  # next spawn

    for obs in obstacles:
        obs['x'] -= obstacle_speed

    obstacles = [o for o in obstacles if o['x'] + o['w'] > 0]
    
    screen.blit(frame_surface, (0,0))
    pygame.draw.rect(screen, (0,255,0), (50, dino_y, 50, 50))
    for obs in obstacles:
        pygame.draw.rect(screen, (255,0,0), (obs['x'], obs['y'], obs['w'], obs['h']))

    pygame.display.update()
    clock.tick(frame_rate)

cap.release()
pygame.quit()  
