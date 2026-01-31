import random
import numpy as np
import pygame
import os

ASSETS = "game/assets/"

BIRD_IMGS = [
    pygame.image.load(os.path.join(ASSETS, "bird1.png")),
    pygame.image.load(os.path.join(ASSETS, "bird2.png")),
    pygame.image.load(os.path.join(ASSETS, "bird3.png")),
]

PIPE_IMG = pygame.image.load(os.path.join(ASSETS, "pipe-green.png"))
BG_IMG = pygame.image.load(os.path.join(ASSETS, "background-day.png"))
GROUND_IMG = pygame.image.load(os.path.join(ASSETS, "base.png"))

PIPE_IMG = pygame.transform.scale2x(PIPE_IMG)
GROUND_IMG = pygame.transform.scale2x(GROUND_IMG)


WIDTH = 400
HEIGHT = 600

PIPE_GAP = 160
PIPE_SPEED = 4
PIPE_WIDTH = PIPE_IMG.get_width()

GRAVITY = 0.6
JUMP_VELOCITY = -9.5

BIRD_X = 80
BIRD_RADIUS = 15


class FlappyGameAI:
    """
    Actions (one-hot):
      [1, 0] = no flap
      [0, 1] = flap
    """

    def __init__(self, *, render=True, fps=60, shaping=True):
        self.render = render
        self.fps = fps
        self.shaping = shaping

        if self.render:
            pygame.init()
            self.display = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("Flappy RL")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("arial", 32)
        else:
            self.display = None
            self.clock = None
            self.font = None

        self.reset()

    # --------------------
    # RESET
    # --------------------
    def reset(self):
        self.bird_y = HEIGHT // 2
        self.bird_vel = 0.0
        self.score = 0

        self.bird_img_idx = 0
        self.bird_angle = 0

        self.pipe_x = WIDTH + 50
        self.pipe_gap_y = random.randint(120, HEIGHT - 200)
        self.pipe_passed = False

        self.ground_x = 0

    # --------------------
    # STEP
    # --------------------
    def play_step(self, action_one_hot):
        if self.render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

        flap = int(np.argmax(action_one_hot)) == 1

        # bird physics
        if flap:
            self.bird_vel = JUMP_VELOCITY

        self.bird_vel += GRAVITY
        self.bird_y += self.bird_vel

        # bird animation + rotation
        self.bird_img_idx = (self.bird_img_idx + 1) % len(BIRD_IMGS)
        self.bird_angle = max(-25, min(25, -self.bird_vel * 3))

        # move pipe
        self.pipe_x -= PIPE_SPEED

        reward = 0.1
        done = False

        # scoring
        if not self.pipe_passed and self.pipe_x + PIPE_WIDTH < BIRD_X:
            self.pipe_passed = True
            self.score += 1
            reward = 5.0

        if self.pipe_x < -PIPE_WIDTH:
            self.pipe_x = WIDTH + 50
            self.pipe_gap_y = random.randint(120, HEIGHT - 200)
            self.pipe_passed = False

        # reward shaping
        if self.shaping:
            dist = abs(self.bird_y - self.pipe_gap_y)
            reward -= 0.001 * dist

        if self._check_collision():
            reward = -10
            done = True

        if self.render:
            self._update_ui()
            self.clock.tick(self.fps)

        return reward, done, self.score

    # --------------------
    # COLLISION
    # --------------------
    def _check_collision(self):
        if self.bird_y < 0 or self.bird_y > HEIGHT - GROUND_IMG.get_height():
            return True

        gap_top = self.pipe_gap_y - PIPE_GAP // 2
        gap_bottom = self.pipe_gap_y + PIPE_GAP // 2

        bird_rect = pygame.Rect(
            BIRD_X,
            int(self.bird_y),
            BIRD_IMGS[0].get_width(),
            BIRD_IMGS[0].get_height(),
        )

        pipe_top_rect = pygame.Rect(
            self.pipe_x,
            gap_top - PIPE_IMG.get_height(),
            PIPE_IMG.get_width(),
            PIPE_IMG.get_height(),
        )

        pipe_bottom_rect = pygame.Rect(
            self.pipe_x,
            gap_bottom,
            PIPE_IMG.get_width(),
            PIPE_IMG.get_height(),
        )

        return bird_rect.colliderect(pipe_top_rect) or bird_rect.colliderect(pipe_bottom_rect)

    # --------------------
    # RENDER
    # --------------------
    def _update_ui(self):
        self.display.blit(BG_IMG, (0, 0))

        # pipes
        gap_top = self.pipe_gap_y - PIPE_GAP // 2
        gap_bottom = self.pipe_gap_y + PIPE_GAP // 2

        top_pipe = pygame.transform.flip(PIPE_IMG, False, True)
        self.display.blit(top_pipe, (self.pipe_x, gap_top - PIPE_IMG.get_height()))
        self.display.blit(PIPE_IMG, (self.pipe_x, gap_bottom))

        # bird
        bird_img = BIRD_IMGS[self.bird_img_idx]
        rotated_bird = pygame.transform.rotate(bird_img, self.bird_angle)
        self.display.blit(rotated_bird, (BIRD_X, int(self.bird_y)))

        # ground scrolling
        self.ground_x -= PIPE_SPEED
        if self.ground_x <= -GROUND_IMG.get_width():
            self.ground_x = 0

        self.display.blit(GROUND_IMG, (self.ground_x, HEIGHT - GROUND_IMG.get_height()))
        self.display.blit(
            GROUND_IMG,
            (self.ground_x + GROUND_IMG.get_width(), HEIGHT - GROUND_IMG.get_height()),
        )

        # score
        score_text = self.font.render(str(self.score), True, (255, 255, 255))
        self.display.blit(score_text, (WIDTH // 2 - score_text.get_width() // 2, 20))

        pygame.display.update()

    # --------------------
    # STATE
    # --------------------
    def get_state(self):
        return np.array(
            [
                self.bird_y / HEIGHT,
                self.bird_vel / 10.0,
                (self.pipe_x - BIRD_X) / WIDTH,
                (self.pipe_gap_y - PIPE_GAP / 2) / HEIGHT,
                (self.pipe_gap_y + PIPE_GAP / 2) / HEIGHT,
                int(self.bird_y < self.pipe_gap_y),
                int(self.bird_y > self.pipe_gap_y),
                int(self.pipe_x + PIPE_WIDTH >= BIRD_X),
            ],
            dtype=np.float32,
        )
