import numpy as np
import itertools
import math
import pygame
import sys

def start(n,r1, r2, L):
    particles = np.empty((n, 5), float)

    x = L * np.random.uniform(0, 1, n)
    y = L * np.random.uniform(0, 1, n)

    particles[:, 0] = x
    particles[:, 1] = y
    particles[:, 3] = 0
    particles[:, 4] = 0

    particles[:int(n / 2), 2] = r1
    particles[int(n / 2):, 2] = r2

    return particles

def assingall(particles, L, extent, coef):
    grid = math.floor(L / (extent * coef))
    boxes = np.empty((grid, grid), dtype=object)

    Is = np.floor((particles[:, 0]) * grid / L)
    Js = np.floor((particles[:, 1]) * grid / L)

    for (i, j) in itertools.product(range(grid), range(grid)):
        boxes[i, j] = particles[(Is == i) & (Js == j), :]

    return (boxes)

def Fs(p, r1, r2, E):
    r_sum = r1 + r2
    mask = p < r_sum
    F = np.zeros_like(p)
    F[mask] = E / r_sum[mask] * (1 - p[mask] / r_sum[mask])
    return F

def force(r1, r2, extent, L):
    dx = r1[:, 0:1].T - r2[:, 0:1]
    dy = r1[:, 1:2].T - r2[:, 1:2]

    dx[:, :] = np.where(dx[:, :] < -0.8 * L, dx[:, :] + L, dx[:, :])
    dx[:, :] = np.where(dx[:, :] > 0.8 * L, dx[:, :] - L, dx[:, :])

    dy[:, :] = np.where(dy[:, :] < -0.8 * L, dy[:, :] + L, dy[:, :])
    dy[:, :] = np.where(dy[:, :] > 0.8 * L, dy[:, :] - L, dy[:, :])
    D = np.linalg.norm(np.array([dx, dy]), axis=0)

    D[D == 0] = np.inf
    force_mag = Fs(D, r1[:, 2:3].T, r2[:, 2:3], 1)
    force_mag[D > extent] = 0
    forcex = force_mag * dx / D
    forcey = force_mag * dy / D
    return np.sum(forcex, axis=0), np.sum(forcey, axis=0), -np.sum(forcex, axis=1), -np.sum(forcey, axis=1)

def evolve(boxes, a, L, extent, coef):
    grid = math.floor(L / (extent * coef))
    for (i, j) in itertools.product(range(grid), range(grid)):
        fx1, fy1, fx2, fy2 = force(boxes[i, j], boxes[(i - 1) % grid, (j - 1) % grid], extent, L)
        boxes[i, j][:, 3] += fx1
        boxes[i, j][:, 4] += fy1
        boxes[(i - 1) % grid, (j - 1) % grid][:, 3] += fx2
        boxes[(i - 1) % grid, (j - 1) % grid][:, 4] += fy2

        fx1, fy1, fx2, fy2 = force(boxes[i, j], boxes[(i) % grid, (j - 1) % grid], extent, L)
        boxes[i, j][:, 3] += fx1
        boxes[i, j][:, 4] += fy1
        boxes[(i) % grid, (j - 1) % grid][:, 3] += fx2
        boxes[(i) % grid, (j - 1) % grid][:, 4] += fy2

        fx1, fy1, fx2, fy2 = force(boxes[i, j], boxes[(i + 1) % grid, (j - 1) % grid], extent, L)
        boxes[i, j][:, 3] += fx1
        boxes[i, j][:, 4] += fy1
        boxes[(i + 1) % grid, (j - 1) % grid][:, 3] += fx2
        boxes[(i + 1) % grid, (j - 1) % grid][:, 4] += fy2

        fx1, fy1, fx2, fy2 = force(boxes[i, j], boxes[(i + 1) % grid, (j) % grid], extent, L)
        boxes[i, j][:, 3] += fx1
        boxes[i, j][:, 4] += fy1
        boxes[(i + 1) % grid, (j) % grid][:, 3] += fx2
        boxes[(i + 1) % grid, (j) % grid][:, 4] += fy2

        fx1, fy1, fx2, fy2 = force(boxes[i, j], boxes[i, j], extent, L)
        boxes[i, j][:, 3] += fx1
        boxes[i, j][:, 4] += fy1

    particles = particles_out(boxes, grid, 5)
    F_mags = dot_product_of_columns(particles)
    vmax = max(F_mags)
    dt = a / vmax

    particles[:, 0:2] += particles[:, 3:5] * dt
    particles[:, 0:2] = np.where(particles[:, 0:2] < 0, particles[:, 0:2] + L, particles[:, 0:2])
    particles[:, 0:2] = np.where(particles[:, 0:2] > L, particles[:, 0:2] - L, particles[:, 0:2])
    particles[:, 3:] = 0

    boxnew = assingall(particles, L, extent, coef)

    return (boxnew, particles, dt)

def particles_out(box, grid, size):
    p = np.empty((1, size))
    for (i, j) in itertools.product(range(grid), range(grid)):
        p = np.vstack((p, box[i, j]))

    return(p[1:])

def dot_product_of_columns(array):
    # Extract the 2nd and 3rd columns
    forces = array[:, 3:5]
    # Calculate dot product along the rows
    dot_product = np.linalg.norm(forces, axis = 1)

    return dot_product

def Length(thy, n):
    R = (n/2 * np.pi * (r1 ** 2 + r2 ** 2)/thy)**0.5
    return(R)

############## Fiddle with all of these ######################
n = 1000             # No. of Particles
r1 = 1               # Radius of half the particles
r2 = 1.4             # Radius of other half
thy = 0.9            # Packing Fraction
a = 0.01             # Parameter controling size of time steps
coeff = 2.5          # Semi-optamised parameter for speed
###############################################################


extent = r1 + r2
L = Length(thy, n)
grid = math.floor(2 * L / (extent * coeff))

translation = 50
scaling = 600/L


particles = start(1000, 1, 1.4, L)

box = [assingall(particles, L, extent, coeff), 0 ,0]


pygame.init()

# Constants
WIDTH, HEIGHT = 700, 700
BOX_COLOR = (255, 255, 255)
CIRCLE_COLOR = 'red'

# Create a Pygame window
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Box with Circles")

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Clear the screen
    screen.fill((0, 0, 0))

    pos = L * scaling
    pygame.draw.rect(screen, BOX_COLOR, (translation, translation, pos, pos), 5)

    box = evolve(box[0], a, L, extent, coeff)

    particles = box[1]

    for k in particles:
        pygame.draw.circle(screen, CIRCLE_COLOR, (k[0] * scaling + translation, k[1] * scaling + translation), k[2] * scaling)
        pygame.draw.circle(screen, 'red', (10, 10), 10)

    pygame.display.flip()

pygame.quit()
sys.exit()