import cv2
import numpy as np
import mediapipe as mp
import random
import time

# --- Game Config ---
WIDTH, HEIGHT = 1000, 650
PLAYER_SIZE = 60
PLAYER_Y = HEIGHT - 70
BULLET_RADIUS = 9
BULLET_SPEED = 22
ENEMY_SIZE = 40
ENEMY_SPEED_BASE = 3
ENEMY_ROWS_START = 2        # Starting enemy rows (fixed)
MAX_ENEMY_ROWS = 5
ENEMY_COLS = 8
FIRE_DELAY = 0.28
LIVES_INIT = 3
PLAYER_MAX_HEALTH = 6
POWERUP_SIZE = 20
POWERUP_SPEED = 3
BOSS_SPAWN_SCORE = 150
ENEMY_FIRE_CHANCE = 0.015  # Slightly increased firing rate
BOSS_FIRE_RATE = 30        # Fires roughly every 30 frames (~0.5 seconds)

# --- Colors (BGR) ---
BG = (18, 8, 8)
WHITE = (255, 255, 255)
BLUE = (255, 160, 60)
CYAN = (240, 220, 80)
RED = (70, 40, 230)
YELLOW = (80, 230, 250)
GREEN = (120, 220, 80)
HUD_BG = (32, 8, 8)
PURPLE = (180, 50, 220)

# --- Mediapipe Hand Landmark Indices ---
WRIST = 0
INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP = 8, 12, 16, 20

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# --- State ---
class GameState:
    def __init__(self):
        self.player_x = WIDTH // 2
        self.player_health = PLAYER_MAX_HEALTH
        self.bullets = []
        self.enemy_bullets = []
        self.boss_bullets = []
        self.enemies = []
        self.powerups = []
        self.score = 0
        self.lives = LIVES_INIT
        self.last_fire = 0
        self.frame_counter = 0
        self.run = True
        self.state = "MENU"  # "MENU", "PLAY", "GAMEOVER"
        self.boss_spawned = False
        self.boss = None
        self.parallax_stars = [
            self.init_stars(44, 0.7, (40, 40, 60)),
            self.init_stars(24, 1.7, (90, 90, 120)),
            self.init_stars(10, 3.3, (220, 220, 245))
        ]
        self.enemy_wave_number = 0  # Track waves for formation changes
        self.enemy_speed_direction = 1  # Alternate horizontal direction each wave

    def reset(self):
        self.__init__()

    def init_stars(self, N, speed, color):
        return {"stars":
                    [[random.randint(0, WIDTH), random.randint(0, HEIGHT)]
                     for _ in range(N)],
                "speed": speed, "color": color}


game = GameState()

# --- Utilities ---
def is_palm(lm):
    return lm[INDEX_TIP].y < lm[WRIST].y and \
           lm[MIDDLE_TIP].y < lm[WRIST].y and \
           lm[RING_TIP].y < lm[WRIST].y and \
           lm[PINKY_TIP].y < lm[WRIST].y

def is_index_up(lm):
    return (lm[INDEX_TIP].y < lm[MIDDLE_TIP].y and
            lm[INDEX_TIP].y < lm[RING_TIP].y and
            lm[INDEX_TIP].y < lm[PINKY_TIP].y)

def move_stars(star_layer):
    for s in star_layer["stars"]:
        s[1] += star_layer["speed"]
        if s[1] > HEIGHT:
            s[0] = random.randint(0, WIDTH)
            s[1] = 0

def draw_stars(frame, star_layer):
    for s in star_layer["stars"]:
        cv2.circle(frame, (int(s[0]), int(s[1])), 2, star_layer["color"], -1)

def spawn_enemy_wave():
    game.enemy_wave_number += 1
    rows = min(MAX_ENEMY_ROWS, ENEMY_ROWS_START + (game.enemy_wave_number - 1) % (MAX_ENEMY_ROWS - ENEMY_ROWS_START + 1))
    direction = -game.enemy_speed_direction  # Alternate speed direction
    game.enemy_speed_direction = direction  # Save for next wave

    gap_x = (WIDTH - ENEMY_COLS * ENEMY_SIZE) // (ENEMY_COLS + 1)
    top_y = 70
    enemy_speed = ENEMY_SPEED_BASE * direction

    game.enemies.clear()
    for row in range(rows):
        for col in range(ENEMY_COLS):
            ex = gap_x + col * (ENEMY_SIZE + gap_x) + ENEMY_SIZE // 2
            ey = top_y + row * ENEMY_SIZE * 2
            vx = enemy_speed + random.uniform(-0.5, 0.5)  # small speed variability
            vy = ENEMY_SPEED_BASE
            game.enemies.append({'x': ex, 'y': ey, 'vx': vx, 'vy': vy, 'alive': True})

def spawn_powerup():
    ptype = random.choice(["heal"])
    px = random.randint(POWERUP_SIZE, WIDTH - POWERUP_SIZE)
    py = -POWERUP_SIZE
    game.powerups.append({'x': px, 'y': py, 'type': ptype, 'vy': POWERUP_SPEED})

def spawn_boss():
    boss = {
        'x': WIDTH // 2,
        'y': -150,
        'vx': 2,
        'vy': 1,
        'hp': 150,
        'width': 260,
        'height': 120,
        'entered': False,
        'timer': 0
    }
    game.boss_spawned = True
    game.boss = boss

def draw_boss(frame, boss):
    x, y, w, h = int(boss['x']), int(boss['y']), boss['width'], boss['height']
    # Boss body
    cv2.rectangle(frame, (x - w//2, y), (x + w//2, y + h), PURPLE, -1, cv2.LINE_AA)
    # Boss eyes
    cv2.circle(frame, (x - 70, y + 40), 20, WHITE, -1)
    cv2.circle(frame, (x + 70, y + 40), 20, WHITE, -1)
    # Boss HP bar (prominent above boss)
    bar_x, bar_y = x - w//2, y - 30
    bar_w = w
    bar_h = 18
    filled_w = int(bar_w * max(0, boss['hp']) / 150)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), RED, -1)  # background (missing HP)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled_w, bar_y + bar_h), GREEN, -1)  # current HP
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), WHITE, 2)  # border



def space_shooter_game():

    gesture_move = False
    gesture_fire = False
    palm_x = game.player_x
    
    while game.run:
        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        cv2.rectangle(frame, (0, 0), (WIDTH, HEIGHT), BG, -1)

        # Stars background
        for layer in game.parallax_stars:
            move_stars(layer)
            draw_stars(frame, layer)

        # Powerup spawning
        if game.state == "PLAY" and game.frame_counter % 800 == 0:
            spawn_powerup()

        # Hand input detection (no hand landmarks drawn on frame)
        
        # Game states
        if game.state == "MENU":
            cv2.putText(frame, "SPACE SHOOTER", (WIDTH//2 - 280, 170), cv2.FONT_HERSHEY_TRIPLEX, 2.1, WHITE, 6)
            cv2.putText(frame, "Show palm to START", (WIDTH//2 - 260, 270), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (180, 180, 255), 4)
            if gesture_move:
                game.state = "PLAY"
                spawn_enemy_wave()

        elif game.state == "GAMEOVER":
            cv2.putText(frame, "GAME OVER!", (WIDTH//2 - 185, HEIGHT//2), cv2.FONT_HERSHEY_TRIPLEX, 2, (50, 20, 240), 7)
            cv2.putText(frame, f"Score: {game.score}", (WIDTH//2 - 95, HEIGHT//2 + 62), cv2.FONT_HERSHEY_SIMPLEX, 1.6, WHITE, 2)
            cv2.putText(frame, "Palm + Index to Restart", (WIDTH//2 - 220, HEIGHT//2 + 145), cv2.FONT_HERSHEY_SIMPLEX, 1.3, CYAN, 2)
            if gesture_move and gesture_fire:
                game.reset()

        elif game.state == "PLAY":
            # Player movement smoothing and clamp
            if gesture_move:
                margin = PLAYER_SIZE // 2 + 25
                palm_x = np.clip(palm_x, margin, WIDTH - margin)
                game.player_x = int((game.player_x * 3 + palm_x) // 4)

            # Player firing with cooldown
            now = time.time()
            if gesture_fire and now - game.last_fire > FIRE_DELAY:
                game.bullets.append({'x': game.player_x, 'y': PLAYER_Y - PLAYER_SIZE // 2})
                game.last_fire = now

            # Player bullets update
            for b in game.bullets[:]:
                b['y'] -= BULLET_SPEED
                if b['y'] < -BULLET_RADIUS:
                    game.bullets.remove(b)
                else:
                    cv2.circle(frame, (int(b['x']), int(b['y'])), BULLET_RADIUS, YELLOW, -1)

            # Enemy random firing
            for e in game.enemies:
                if random.random() < ENEMY_FIRE_CHANCE:
                    game.enemy_bullets.append({'x': e['x'], 'y': e['y'] + ENEMY_SIZE // 2, 'vy': 15})

            # Enemy bullets update + collide with player
            for eb in game.enemy_bullets[:]:
                eb['y'] += eb['vy']
                if eb['y'] > HEIGHT + BULLET_RADIUS:
                    game.enemy_bullets.remove(eb)
                else:
                    cv2.circle(frame, (int(eb['x']), int(eb['y'])), BULLET_RADIUS, RED, -1)
                if abs(game.player_x - eb['x']) < PLAYER_SIZE // 2 and abs(PLAYER_Y - eb['y']) < PLAYER_SIZE // 2:
                    if eb in game.enemy_bullets:
                        game.enemy_bullets.remove(eb)
                    game.player_health -= 1

            # Enemies update & draw
            for e in game.enemies[:]:
                e['x'] += e['vx']
                e['y'] += e['vy']
                if not (ENEMY_SIZE // 2 < e['x'] < WIDTH - ENEMY_SIZE // 2):
                    e['vx'] *= -1
                if e['alive']:
                    cv2.circle(frame, (int(e['x']), int(e['y'])), ENEMY_SIZE // 2, RED, -1)
                    cv2.rectangle(frame, (int(e['x'] - ENEMY_SIZE // 3), int(e['y'])),
                                  (int(e['x'] + ENEMY_SIZE // 3), int(e['y'] + 10)), (180, 180, 255), -1)
                if e['y'] > HEIGHT + ENEMY_SIZE:
                    game.enemies.remove(e)
                    game.player_health -= 1

            # Powerups update & draw + collection
            for p in game.powerups[:]:
                p['y'] += p['vy']
                if p['y'] > HEIGHT + POWERUP_SIZE:
                    game.powerups.remove(p)
                else:
                    cv2.circle(frame, (int(p['x']), int(p['y'])), POWERUP_SIZE // 2, GREEN, -1)
                if abs(game.player_x - p['x']) < PLAYER_SIZE // 2 + POWERUP_SIZE // 2 and abs(PLAYER_Y - p['y']) < PLAYER_SIZE // 2 + POWERUP_SIZE // 2:
                    if p['type'] == "heal":
                        game.player_health = min(PLAYER_MAX_HEALTH, game.player_health + 2)
                    game.powerups.remove(p)

            # Player bullet vs enemy collision
            for b in game.bullets[:]:
                removed = False
                for e in game.enemies[:]:
                    dist = np.hypot(b['x'] - e['x'], b['y'] - e['y'])
                    if dist < (ENEMY_SIZE // 2 + BULLET_RADIUS - 3):
                        if e in game.enemies:
                            game.enemies.remove(e)
                        if b in game.bullets:
                            game.bullets.remove(b)
                            removed = True
                        game.score += 10
                        break
                if removed:
                    continue

            # Boss spawn & logic
            if not game.boss_spawned and game.score >= BOSS_SPAWN_SCORE:
                spawn_boss()

            if game.boss_spawned and game.boss is not None:
                boss = game.boss
                if not boss['entered']:
                    boss['y'] += boss['vy']
                    if boss['y'] >= 30:
                        boss['entered'] = True
                else:
                    boss['x'] += boss['vx']
                    if boss['x'] < boss['width'] // 2 or boss['x'] > WIDTH - boss['width'] // 2:
                        boss['vx'] *= -1
                    boss['timer'] += 1

                    # Boss fires 5 bullets spread horizontally every BOSS_FIRE_RATE frames
                    if boss['timer'] % BOSS_FIRE_RATE == 0:
                        offsets = [-80, -40, 0, 40, 80]
                        for offset in offsets:
                            game.boss_bullets.append({'x': boss['x'] + offset, 'y': boss['y'] + boss['height'], 'vy': 22})

                draw_boss(frame, boss)

                # Player bullets hit boss
                for b in game.bullets[:]:
                    if (boss['x'] - boss['width'] // 2 < b['x'] < boss['x'] + boss['width'] // 2 and
                        boss['y'] < b['y'] < boss['y'] + boss['height']):
                        if b in game.bullets:
                            game.bullets.remove(b)
                        boss['hp'] -= 5
                        game.score += 20
                        if boss['hp'] <= 0:
                            game.boss = None
                            game.boss_spawned = False
                            spawn_enemy_wave()

            # Boss bullets update/draw & collide with player
            for bb in game.boss_bullets[:]:
                bb['y'] += bb['vy']
                if bb['y'] > HEIGHT + BULLET_RADIUS:
                    game.boss_bullets.remove(bb)
                else:
                    cv2.circle(frame, (int(bb['x']), int(bb['y'])), BULLET_RADIUS, PURPLE, -1)
                if abs(game.player_x - bb['x']) < PLAYER_SIZE // 2 and abs(PLAYER_Y - bb['y']) < PLAYER_SIZE // 2:
                    if bb in game.boss_bullets:
                        game.boss_bullets.remove(bb)
                    game.player_health -= 2

            # Player collisions with enemies and boss
            for e in game.enemies[:]:
                if (abs(game.player_x - e['x']) < ENEMY_SIZE // 2 + PLAYER_SIZE // 2 - 8 and
                    abs(PLAYER_Y - e['y']) < ENEMY_SIZE // 2 + PLAYER_SIZE // 2):
                    if e in game.enemies:
                        game.enemies.remove(e)
                    game.player_health -= 1

            if game.boss_spawned and game.boss is not None:
                boss = game.boss
                if (abs(game.player_x - boss['x']) < boss['width'] // 2 + PLAYER_SIZE // 2 and
                    abs(PLAYER_Y - (boss['y'] + boss['height'] // 2)) < boss['height'] // 2 + PLAYER_SIZE // 2):
                    game.player_health -= 3

            # Draw player ship
            pt1 = (game.player_x, PLAYER_Y - PLAYER_SIZE // 2)
            pt2 = (game.player_x - PLAYER_SIZE // 2, PLAYER_Y + PLAYER_SIZE // 2)
            pt3 = (game.player_x + PLAYER_SIZE // 2, PLAYER_Y + PLAYER_SIZE // 2)
            pts = np.array([pt1, pt2, pt3], np.int32)
            cv2.fillPoly(frame, [pts], BLUE)
            fx = (game.player_x, PLAYER_Y + PLAYER_SIZE // 2 + 16)
            fx2 = (game.player_x - 12, PLAYER_Y + PLAYER_SIZE // 2)
            fx3 = (game.player_x + 12, PLAYER_Y + PLAYER_SIZE // 2)
            cv2.fillPoly(frame, [np.array([fx, fx2, fx3], np.int32)], CYAN)

            # Spawn next wave if no enemies/boss alive
            if not game.enemies and not game.boss_spawned:
                spawn_enemy_wave()

            # Player dead check
            if game.player_health <= 0:
                game.lives -= 1
                game.player_health = PLAYER_MAX_HEALTH
                if game.lives <= 0:
                    game.state = "GAMEOVER"

            # HUD
            cv2.rectangle(frame, (0, 0), (WIDTH, 44), HUD_BG, -1)
            cv2.putText(frame, f"SCORE: {game.score}", (22, 36), cv2.FONT_HERSHEY_DUPLEX, 1.2, WHITE, 2)
            cv2.putText(frame, f"LIVES: {game.lives}", (WIDTH - 260, 36), cv2.FONT_HERSHEY_DUPLEX, 1.2, WHITE, 2)
            cv2.putText(frame, f"HP: {game.player_health}", (WIDTH - 130, 36), cv2.FONT_HERSHEY_DUPLEX, 1.2, GREEN, 2)
            cv2.putText(frame, "(Palm=move, Palm+Index=fire, Q/ESC=quit)", (270, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.05, (230, 240, 240), 2)

        frame = np.flip(frame,axis=0)
        
        gesture_move, gesture_fire, palm_x=yield frame

        game.frame_counter += 1
