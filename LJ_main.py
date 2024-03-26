import numpy as np
import math
import itertools
import pygame
import sys

def start(n,r1, r2, L):
    particles = np.empty((n, 7), float)

    x = L * np.random.uniform(0, 1, n)
    y = L * np.random.uniform(0, 1, n)

    particles[:, 0] = x
    particles[:, 1] = y
    particles[:, 3] = np.random.uniform(0, 2 * np.pi, n)
    particles[:, 4] = 0
    particles[:, 5] = 0
    particles[:, 6] = 0

    particles[:int(n / 2), 2] = r1
    particles[int(n / 2):, 2] = r2

    return particles   #

def assingall(particles, R, extent, cast, coeff):
    grid = math.floor(2 * cast * R / (extent * coeff))
    boxes = np.empty((grid, grid), dtype=object)

    Is = np.floor((particles[:, 0] + cast * R) / (extent * coeff))
    Js = np.floor((particles[:, 1] + cast * R) / (extent * coeff))

    for (i, j) in itertools.product(range(grid), range(grid)):
        boxes[i, j] = particles[(Is == i) & (Js == j), :]

    return (boxes)

def Flj(p, r1, r2, Elj):  # change R for average of first and second radius.
    F = -4 * Elj * (-12 * (r1 + r2) ** 12 * p ** (-13) + 6 * (r1 + r2) ** 6 * p ** (-7))
    return (F)


def forceLJ(r1, r2, extent):
    dx = r1[:, 0:1].T - r2[:, 0:1]
    dy = r1[:, 1:2].T - r2[:, 1:2]
    D = np.linalg.norm(np.array([dx, dy]), axis=0)

    D[D == 0] = np.inf
    force_mag = Flj(D, r1[:, 2:3].T, r2[:, 2:3], 1)
    force_mag[D > extent] = 0
    forcex = force_mag * dx / D
    forcey = force_mag * dy / D
    return np.sum(forcex, axis=0), np.sum(forcey, axis=0), -np.sum(forcex, axis=1), -np.sum(forcey, axis=1)

def evolveLJ(boxes, a, R, extent, cast, va, D, coef):
    grid = math.floor(2 * cast * R / (extent * coef))
    for (i, j) in itertools.product(range(grid), range(grid)):
        fx1, fy1, fx2, fy2 = forceLJ(boxes[i, j], boxes[(i - 1) % grid, (j - 1) % grid], extent)
        boxes[i, j][:, 4] += fx1
        boxes[i, j][:, 5] += fy1
        boxes[(i - 1) % grid, (j - 1) % grid][:, 4] += fx2
        boxes[(i - 1) % grid, (j - 1) % grid][:, 5] += fy2

        fx1, fy1, fx2, fy2 = forceLJ(boxes[i, j], boxes[(i) % grid, (j - 1) % grid], extent)
        boxes[i, j][:, 4] += fx1
        boxes[i, j][:, 5] += fy1
        boxes[(i) % grid, (j - 1) % grid][:, 4] += fx2
        boxes[(i) % grid, (j - 1) % grid][:, 5] += fy2

        fx1, fy1, fx2, fy2 = forceLJ(boxes[i, j], boxes[(i + 1) % grid, (j - 1) % grid], extent)
        boxes[i, j][:, 4] += fx1
        boxes[i, j][:, 5] += fy1
        boxes[(i + 1) % grid, (j - 1) % grid][:, 4] += fx2
        boxes[(i + 1) % grid, (j - 1) % grid][:, 5] += fy2

        fx1, fy1, fx2, fy2 = forceLJ(boxes[i, j], boxes[(i + 1) % grid, (j) % grid], extent)
        boxes[i, j][:, 4] += fx1
        boxes[i, j][:, 5] += fy1
        boxes[(i + 1) % grid, (j) % grid][:, 4] += fx2
        boxes[(i + 1) % grid, (j) % grid][:, 5] += fy2

        fx1, fy1, fx2, fy2 = forceLJ(boxes[i, j], boxes[i, j], extent)
        boxes[i, j][:, 4] += fx1
        boxes[i, j][:, 5] += fy1

        boxes[i, j][:, 4] += va * np.cos(boxes[i, j][:, 3])
        boxes[i, j][:, 5] += va * np.sin(boxes[i, j][:, 3])
        boxes[i, j][:, 6] = np.random.uniform(-0.5, 0.5, boxes[i, j].shape[0]) * (2 * D) ** 0.5

    particles = particles_out(boxes, grid, 7)
    F_mags = dot_product_of_columns(particles)
    vmax = max(F_mags)
    dt = a / vmax

    particles[:, 0:2] += particles[:, 4:6] * dt
    particles[:, 3] += particles[:, 6] * (dt ** 0.5)
    particles[:, 4:6] = 0

    boxnew = assingall(particles, R, extent, cast, coef)

    return (boxnew, particles, dt)

def particles_out(box, grid, size):
    p = np.empty((1, size))
    for (i, j) in itertools.product(range(grid), range(grid)):
        p = np.vstack((p, box[i, j]))

    return(p[1:])

def dot_product_of_columns(array):
    # Extract the 2nd and 3rd columns
    forces = array[:, 4:6]
    # Calculate dot product along the rows
    dot_product = np.linalg.norm(forces, axis = 1)
    return dot_product

def radius(thy, n):
    R = (n*c/thy)**0.5
    return(R)

n = 1000
with open('particles1000.npy', 'rb') as f: p = np.load(f)
print('No. of particles = ' + str(len(p)), flush = True)

particles = np.empty((n,7), float)
for i in range(len(p)):
    particles[i] = p[i]


############### Don't Fiddle with these #################
r1 = 1
r2 = 1.4
g = (2**(1/6)-1)*2
c = ((r1+g/2)**2 + (r2 + g/2)**2)/2
thy = 0.9
n = 1000
R = radius(thy, n)
translation = 350
cast = 1.2
scaling = 300/(R * cast)
#########################################################

############### Fiddle with these to see their effect ##############
a = 0.01      # Controls variable time step
extent = 5    # Maximum extent of the Force
va = 1        # Proplsion Velocity
D = 5         # Diffusion Coeficent
coeff = 2.5   # Adimesional Prameter semi-optamised for speed
####################################################################

grid = math.floor(2 * cast * R / (extent * coeff))

box = [assingall(particles, R, extent, cast, coeff), 0 ,0]

pygame.init()

# Constants
WIDTH, HEIGHT = 700, 700
BOX_COLOR = (255, 255, 255)
CIRCLE_COLOR = 'hotpink'

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

    # Draw the circe
    pygame.draw.circle(screen, BOX_COLOR, (translation, translation), R * scaling)
    box = evolveLJ(box[0], a, R, extent, cast, va, D, coeff)

    particles = box[1]

    for k in particles:
        pygame.draw.circle(screen, CIRCLE_COLOR, (k[0] * scaling + translation, k[1] * scaling + translation),
                           k[2] * scaling)
        pygame.draw.circle(screen, 'red', (10, 10), 10)

    pygame.display.flip()

pygame.quit()
sys.exit()


