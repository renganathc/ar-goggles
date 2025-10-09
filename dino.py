import pygame
import numpy as np
import random
import cv2

def dino_game():
    pygame.init()
    game_over = False

    # Get video feed size
    h, w = 450, 750
    screen = pygame.Surface((w, h))

    # Dino properties
    dino_y = h - 100
    dino_vel = 0
    gravity = 4
    is_jumping = False

    obstacles = []  # list of dicts: {'x', 'y', 'w', 'h'}
    obstacle_speed = 17
    spawn_timer = 0

    score = 0
    min_dist, max_dist = 30, 50 # next block
    max_obs_height = 120

    def jump_fn():
        nonlocal is_jumping, dino_vel
        if not is_jumping:
            dino_vel = -34   # jump strength
            is_jumping = True

    running = True

    while running and not game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        dino_y += dino_vel
        dino_vel += gravity
        if dino_y >= h - 100:  # ground level
            dino_y = h - 100
            is_jumping = False

        spawn_timer -= 1
        if spawn_timer <= 0:
            obs_h = random.randint(max_obs_height - 80, max_obs_height)
            obstacles.append({'x': w, 'y': h - obs_h, 'w': 20, 'h': obs_h})
            spawn_timer = random.randint(min_dist, max_dist)  # next spawn

        for obs in obstacles:
            obs['x'] -= obstacle_speed

        obstacles = [o for o in obstacles if o['x'] + o['w'] > 0]

        dino_rect = pygame.Rect(50, dino_y, 50, 50)

        for obs in obstacles:
            obs_rect = pygame.Rect(obs['x'], obs['y'], obs['w'], obs['h'])
            if dino_rect.colliderect(obs_rect):
                print("Game Over!")
                game_over = True
                break
        
        screen.fill((0, 0, 0))
        score_font = pygame.font.SysFont(None, 90)
        score_text = score_font.render(str(score), True, (100,0,100))
        screen.blit(score_text, (w - 140, 100))
        pygame.draw.rect(screen, (0,255,0), (50, dino_y, 50, 50))  # green dino
        for obs in obstacles:
            pygame.draw.rect(screen, (255,0,0), (obs['x'], obs['y'], obs['w'], obs['h']))

        #clock.tick(frame_rate)
        score += 1

        if score % 100 == 0:
            #obstacle_speed += 2
            max_obs_height += 10
            if min_dist > 16:
                min_dist -= 2
            if max_dist > 27:
                max_dist -= 2
        
        frame = pygame.surfarray.array3d(screen)
        frame = np.transpose(frame, (1,0,2))  # pygame stores arrays differently
        frame = np.flipud(frame)
        frame = cv2.resize(frame, (1000, 600))
        x = yield frame

        if x:
            jump_fn()
        
    game_over_frame_count = 0

    while game_over:
        game_over_frame_count += 1
        if game_over_frame_count > 115:
            break
        font1 = pygame.font.SysFont(None, 160)
        font2 = pygame.font.SysFont(None, 100)
        line1 = font1.render("Game Over!", True, (0,100,200))
        line2 = font2.render("Score: " + str(score - 1), True, (255,0,0))

        line1_rect = line1.get_rect(center=(w//2, h//2 - 50))
        screen.blit(line1, line1_rect)
        line2_rect = line2.get_rect(center=(w//2, h//2 + 60))
        screen.blit(line2, line2_rect)

        frame = pygame.surfarray.array3d(screen)
        frame = np.transpose(frame, (1,0,2))  # pygame stores arrays differently
        frame = np.flipud(frame)
        frame = cv2.resize(frame, (1000, 600))
        yield frame

    pygame.quit()