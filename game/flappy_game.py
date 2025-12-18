import pygame
import random
import numpy as np

WIDTH = 400
HEIGHT = 600
PIPE_GAP = 150
PIPE_SPEED = 3
PIPE_WIDTH = 60
GRAVITY = 0.5
JUMP_VELOCITY = -8
BIRD_X = 80
BIRD_RADIUS = 15


class FlappyGameAI:
    def __init__(self):
        pygame.init()
        self.display = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Flappy RL")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("arial", 25)
        self.reset()

    def reset(self):
        self.bird_y = HEIGHT // 2
        self.bird_vel = 0.0
        self.score = 0

        self.pipe_x = WIDTH
        self.pipe_gap_y = random.randint(120, HEIGHT - 120)
        self.frame = 0

        self.pipe_passed = False  # ✅ correct flag

    def play_step(self, action_one_hot):
        self.frame += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        flap = int(np.argmax(action_one_hot)) == 1

        # physics
        if flap:
            self.bird_vel = JUMP_VELOCITY
        self.bird_vel += GRAVITY
        self.bird_y += self.bird_vel

        # move pipe
        self.pipe_x -= PIPE_SPEED

        # base reward: staying alive
        reward = 0.1
        done = False

        # ✅ score when bird passes the pipe
        if (not self.pipe_passed) and (self.pipe_x + PIPE_WIDTH < BIRD_X):
            self.pipe_passed = True
            self.score += 1
            reward = 5.0

        # reset pipe when it leaves screen
        if self.pipe_x < -PIPE_WIDTH:
            self.pipe_x = WIDTH
            self.pipe_gap_y = random.randint(120, HEIGHT - 120)
            self.pipe_passed = False

        # ✅ reward shaping (encourage staying near gap center)
        gap_center = self.pipe_gap_y
        dist = abs(self.bird_y - gap_center)

        if dist < PIPE_GAP / 2:
            reward += 0.2
        reward -= 0.001 * dist

        # collision ends episode
        if self._check_collision():
            reward = -10
            done = True

        self._update_ui()
        self.clock.tick(60)

        return reward, done, self.score

    def _check_collision(self):
        if self.bird_y - BIRD_RADIUS < 0 or self.bird_y + BIRD_RADIUS > HEIGHT:
            return True

        gap_top = self.pipe_gap_y - PIPE_GAP // 2
        gap_bottom = self.pipe_gap_y + PIPE_GAP // 2

        bird_top = self.bird_y - BIRD_RADIUS
        bird_bottom = self.bird_y + BIRD_RADIUS
        bird_left = BIRD_X - BIRD_RADIUS
        bird_right = BIRD_X + BIRD_RADIUS

        pipe_left = self.pipe_x
        pipe_right = self.pipe_x + PIPE_WIDTH

        horizontally_inside = (bird_right > pipe_left) and (bird_left < pipe_right)
        hits_top = horizontally_inside and (bird_top < gap_top)
        hits_bottom = horizontally_inside and (bird_bottom > gap_bottom)

        return hits_top or hits_bottom

    def _update_ui(self):
        self.display.fill((0, 0, 0))

        gap_top = self.pipe_gap_y - PIPE_GAP // 2
        gap_bottom = self.pipe_gap_y + PIPE_GAP // 2

        pygame.draw.rect(self.display, (0, 255, 0), (self.pipe_x, 0, PIPE_WIDTH, gap_top))
        pygame.draw.rect(self.display, (0, 255, 0), (self.pipe_x, gap_bottom, PIPE_WIDTH, HEIGHT - gap_bottom))

        pygame.draw.circle(self.display, (255, 255, 0), (BIRD_X, int(self.bird_y)), BIRD_RADIUS)

        text = self.font.render("Score: " + str(self.score), True, (255, 255, 255))
        self.display.blit(text, [0, 0])

        pygame.display.flip()

    def get_state(self):
        bird_y_norm = self.bird_y / HEIGHT
        bird_vel_norm = self.bird_vel / 10.0
        dist_to_pipe_norm = (self.pipe_x - BIRD_X) / WIDTH

        gap_top = self.pipe_gap_y - PIPE_GAP // 2
        gap_bottom = self.pipe_gap_y + PIPE_GAP // 2
        gap_top_norm = gap_top / HEIGHT
        gap_bottom_norm = gap_bottom / HEIGHT

        bird_above_gap = int(self.bird_y < gap_top)
        bird_below_gap = int(self.bird_y > gap_bottom)
        pipe_ahead = int(self.pipe_x + PIPE_WIDTH >= BIRD_X)

        return np.array(
            [
                bird_y_norm,
                bird_vel_norm,
                dist_to_pipe_norm,
                gap_top_norm,
                gap_bottom_norm,
                bird_above_gap,
                bird_below_gap,
                pipe_ahead,
            ],
            dtype=float,
        )
