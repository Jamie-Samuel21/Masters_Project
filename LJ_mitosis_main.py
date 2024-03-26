import numpy as np
import math
import itertools
import pygame
import sys

g = (2 ** (1 / 6) - 1) * 2

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

    return particles

def assingall(particles, L, extent, coef):
    grid = math.floor(L / (extent * coef))
    boxes = np.empty((grid, grid), dtype=object)

    Is = np.floor((particles[:, 0]) * grid / L)
    Js = np.floor((particles[:, 1]) * grid / L)

    for (i, j) in itertools.product(range(grid), range(grid)):
        boxes[i, j] = particles[(Is == i) & (Js == j), :]

    return (boxes)

def Flj(p, r1, r2, Elj):  # change R for average of first and second radius.
    F = -4 * Elj * (-12 * (r1 + r2) ** 12 * p ** (-13) + 6 * (r1 + r2) ** 6 * p ** (-7))
    return (F)

def transform_array(arr):
    # Create a copy of the input array
    transformed_arr = np.copy(arr)

    # Replace values according to conditions
    transformed_arr[(arr == 0)] = 4

    transformed_arr = transformed_arr/2

    return transformed_arr

def forceLJ(r1, r2, extent, L):
    dx = r1[:, 0:1].T - r2[:, 0:1]
    dy = r1[:, 1:2].T - r2[:, 1:2]

    dx[:, :] = np.where(dx[:, :] < -0.8 * L, dx[:, :] + L, dx[:, :])
    dx[:, :] = np.where(dx[:, :] > 0.8 * L, dx[:, :] - L, dx[:, :])

    dy[:, :] = np.where(dy[:, :] < -0.8 * L, dy[:, :] + L, dy[:, :])
    dy[:, :] = np.where(dy[:, :] > 0.8 * L, dy[:, :] - L, dy[:, :])

    D = np.sqrt(dx ** 2 + dy ** 2)

    Elj = r1[:, 7:8].T + r2[:, 7:8]
    Elj = transform_array(Elj)

    D[D == 0] = np.inf
    force_mag = Flj(D, r1[:, 2:3].T, r2[:, 2:3], Elj)
    force_mag[D > extent] = 0
    forcex = force_mag * dx / D
    forcey = force_mag * dy / D
    return np.sum(forcex, axis=0), np.sum(forcey, axis=0), -np.sum(forcex, axis=1), -np.sum(forcey, axis=1)

def mitosis(particles, distribution_mean):
    splits = np.random.poisson(distribution_mean)
    for i in range(splits):
        n = particles.shape[0]  # Get the number of rows in the array
        random_index = np.random.randint(0, n)  # Randomly select an index
        while particles[random_index, 7] != 1 or particles[random_index, 8] != 0:
            random_index = np.random.randint(0, n)  # esures particle is active
        #print(particles[random_index, 7])

        particle = particles[random_index]
        p1, p2 = new_particles(particle)
        particles = np.delete(particles, random_index, axis=0)
        particles = np.append(particles, [p1], axis=0)
        particles = np.append(particles, [p2], axis=0)

    return(particles)

def activate(particles, R, L):  #takes n by 7 array returns an n by 8 array with the final entry being 1 if inside r and 0 otherwise.
    p = np.copy(particles)
    p[:, 0:2] -= L/2
    norms = np.linalg.norm(p[:, :2], axis=1)
    new_column = norms <= R
    new_column = new_column[:, np.newaxis]  # Reshaping to make it a column vector
    return np.hstack((particles, new_column))

def mitosis_ready(particles):
    zeros_column = np.zeros((particles.shape[0], 1))
    return(np.hstack((particles, zeros_column)))


def evolveLJ_active(boxes, a, L, extent, va, D, coef, distribution_mean):
    grid = math.floor(L / (extent * coef))
    for (i, j) in itertools.product(range(grid), range(grid)):
        fx1, fy1, fx2, fy2 = forceLJ(boxes[i, j], boxes[(i - 1) % grid, (j - 1) % grid], extent, L)
        boxes[i, j][:, 4] += fx1
        boxes[i, j][:, 5] += fy1
        boxes[(i - 1) % grid, (j - 1) % grid][:, 4] += fx2
        boxes[(i - 1) % grid, (j - 1) % grid][:, 5] += fy2

        fx1, fy1, fx2, fy2 = forceLJ(boxes[i, j], boxes[(i) % grid, (j - 1) % grid], extent, L)
        boxes[i, j][:, 4] += fx1
        boxes[i, j][:, 5] += fy1
        boxes[(i) % grid, (j - 1) % grid][:, 4] += fx2
        boxes[(i) % grid, (j - 1) % grid][:, 5] += fy2

        fx1, fy1, fx2, fy2 = forceLJ(boxes[i, j], boxes[(i + 1) % grid, (j - 1) % grid], extent, L)
        boxes[i, j][:, 4] += fx1
        boxes[i, j][:, 5] += fy1
        boxes[(i + 1) % grid, (j - 1) % grid][:, 4] += fx2
        boxes[(i + 1) % grid, (j - 1) % grid][:, 5] += fy2

        fx1, fy1, fx2, fy2 = forceLJ(boxes[i, j], boxes[(i + 1) % grid, (j) % grid], extent, L)
        boxes[i, j][:, 4] += fx1
        boxes[i, j][:, 5] += fy1
        boxes[(i + 1) % grid, (j) % grid][:, 4] += fx2
        boxes[(i + 1) % grid, (j) % grid][:, 5] += fy2

        fx1, fy1, fx2, fy2 = forceLJ(boxes[i, j], boxes[i, j], extent, L)
        boxes[i, j][:, 4] += fx1
        boxes[i, j][:, 5] += fy1

        boxes[i, j][:, 4] += va * np.cos(boxes[i, j][:, 3]) * boxes[i,j][:, 7]  #multplies by 0 or 1 depeding on active or passive
        boxes[i, j][:, 5] += va * np.sin(boxes[i, j][:, 3]) * boxes[i,j][:, 7]
        boxes[i, j][:, 6] = np.random.uniform(-0.5, 0.5, boxes[i, j].shape[0]) * (2 * D) ** 0.5

    particles = particles_out(boxes, grid, 9)
    F_mags = dot_product_of_columns(particles)
    vmax = max(F_mags)
    dt = a / vmax

    particles[:, 0:2] += particles[:, 4:6] * dt
    particles[:, 0:2] = np.where(particles[:, 0:2] < 0, particles[:, 0:2] + L, particles[:, 0:2])
    particles[:, 0:2] = np.where(particles[:, 0:2] > L, particles[:, 0:2] - L, particles[:, 0:2])
    particles[:, 3] += particles[:, 6] * (dt ** 0.5)
    particles[:, 4:7] = 0
    mask = particles[:, 8] > 0
    particles[mask, 2] += 0.001
    particles[mask, 8] -= 0.001

    particles = mitosis(particles, distribution_mean)
    particles = death(particles, distribution_mean)

    boxnew = assingall(particles, L, extent, coef)

    return (boxnew, particles, dt)

def particles_out(box, grid, size):
    p = np.empty((1, size))
    for (i, j) in itertools.product(range(grid), range(grid)):
        p = np.vstack((p, box[i, j]))

    return(p[1:])

def pick_random_entry_index(array, grid):
    # Generate random indices
    i = np.random.randint(0, grid)
    j = np.random.randint(0, grid)

    # Get the list at the random indices
    random_list = array[i, j]

    # If the list is empty, pick again
    while len(random_list) == 0:
        i = np.random.randint(0, grid)
        j = np.random.randint(0, grid)
        random_list = array[i, j]

    # Pick a random index from the list
    random_index = np.random.randint(0, len(random_list))

    return (i, j, random_index)

def new_particles(input_list):
    r = input_list[2]
    angle = np.random.uniform(0, 2 * np.pi)
    unit_vector = np.array([np.cos(angle), np.sin(angle)])

    shifted_plus = np.copy(input_list)
    shifted_minus = np.copy(input_list)

    shifted_plus[0:2] += 0.6 * r * unit_vector
    shifted_plus[2] = r * 0.5
    shifted_plus[-1] += r * 0.5

    shifted_minus[0:2] -= 0.6 * r * unit_vector
    shifted_minus[2] = r * 0.5
    shifted_minus[-1] += r * 0.5

    return shifted_plus, shifted_minus

def death(particles, distribution_mean):
    splits = np.random.poisson(distribution_mean)
    for i in range(splits):
        n = particles.shape[0]  # Get the number of rows in the array
        random_index = np.random.randint(0, n)  # Randomly select an index

        particles = np.delete(particles, random_index, axis=0)

    return(particles)

def dot_product_of_columns(array):
    # Extract the 2nd and 3rd columns
    forces = array[:, 4:6]
    # Calculate dot product along the rows
    dot_product = np.linalg.norm(forces, axis = 1)

    return dot_product

def Length(thy, n):
    L = np.sqrt((n*math.pi/(2**(2/3)*thy))*(1+1.4**2))
    return(L)

with open('particles1000BC.npy', 'rb') as f: particles = np.load(f)

############### Don't Fiddle with these #################
r1 = 1
r2 = 1.4
g = (2**(1/6)-1)*2
c = ((r1+g/2)**2 + (r2 + g/2)**2)/2
thy = 0.9
n = 1000
L = Length(thy, n)
translation = 50
scaling = 600/L
#########################################################


################### Fiddle with these ###################
a = 0.01                   # Parameter for timestep
extent = 5                 # Extent of Force
va = 1                     # Propulsion Velcity
D = 5                      # Diffusion Coeficent
coef = 2.5                 # adimesional parameter semi-optamised for speed
active_radius = 30         # radius of iner active particles
distribution_mean = 1/25   # Probability of a new particle spawning or dying
#########################################################

grid = math.floor(L / (extent * coef))

particles = activate(particles, active_radius, L)
particles = mitosis_ready(particles)
box = [assingall(particles, L, extent, coef), 0 ,0]

pygame.init()

# Constants
WIDTH, HEIGHT = 700, 700
BOX_COLOR = (255, 255, 255)
CIRCLE_COLOR = ['blue', 'red']

# Create a Pygame window
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Box with Circles")

counter = 0
time = 0

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Clear the screen
    screen.fill((0, 0, 0))

    pos = L * scaling

    # Draw the circe
    pygame.draw.rect(screen, BOX_COLOR, (translation, translation, pos, pos), 5)

    box = evolveLJ_active(box[0], a, L, extent, va, D, coef, distribution_mean)

    particles = box[1]


    for k in particles:
        pygame.draw.circle(screen, CIRCLE_COLOR[round(k[7])], (k[0] * scaling + translation, k[1] * scaling + translation), k[2] * scaling)
        pygame.draw.circle(screen, 'red', (10, 10), 10)

    pygame.display.flip()

pygame.quit()
sys.exit()