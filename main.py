import time
import numpy as np
import matplotlib.pyplot as plt

from utils import viz_tools

L_x = 1E+6  # Length of domain in x-direction
L_y = 1E+6  # Length of domain in y-direction
g = 9.81  # Acceleration of gravity [m/s^2]
H = 100  # Depth of fluid [m]
f_0 = 1E-4  # Fixed part ofcoriolis parameter [1/s]
beta = 2E-11  # gradient of coriolis parameter [1/ms]
rho_0 = 1024.0  # Density of fluid [kg/m^3)]
tau_0 = 0.1  # Amplitude of wind stress [kg/ms^2]
use_coriolis = True  # True if you want coriolis force
use_friction = True  # True if you want bottom friction
use_wind = True  # True if you want wind stress
use_beta = True  # True if you want variation in coriolis
use_source = False  # True if you want mass source into the domain
use_sink = False  # True if you want mass sink out of the domain
param_string = "\n================================================================"
param_string += "\nuse_coriolis = {}\nuse_beta = {}".format(use_coriolis, use_beta)
param_string += "\nuse_friction = {}\nuse_wind = {}".format(use_friction, use_wind)
param_string += "\nuse_source = {}\nuse_sink = {}".format(use_source, use_sink)
param_string += "\ng = {:g}\nH = {:g}".format(g, H)

N_x = 500  # Number of grid points in x-direction
N_y = 500  # Number of grid points in y-direction
dx = L_x / (N_x - 1)
dy = L_y / (N_y - 1)
dt = 0.1 * min(dx, dy) / np.sqrt(g * H)
time_step = 1
max_time_step = 10000
x = np.linspace(-L_x / 2, L_x / 2, N_x)
y = np.linspace(-L_y / 2, L_y / 2, N_y)
X, Y = np.meshgrid(x, y)
X = np.transpose(X)
Y = np.transpose(Y)
param_string += "\ndx = {:.2f} km\ndy = {:.2f} km\ndt = {:.2f} s".format(dx, dy, dt)

if use_friction is True:
    kappa_0 = 1 / (5 * 24 * 3600)
    kappa = np.ones((N_x, N_y)) * kappa_0
    # kappa[0, :] = kappa_0
    # kappa[-1, :] = kappa_0
    # kappa[:, 0] = kappa_0
    # kappa[:, -1] = kappa_0
    # kappa[:int(N_x/15), :] = 0
    # kappa[int(14*N_x/15)+1:, :] = 0
    # kappa[:, :int(N_y/15)] = 0
    # kappa[:, int(14*N_y/15)+1:] = 0
    # kappa[int(N_x/15):int(2*N_x/15), int(N_y/15):int(14*N_y/15)+1] = 0
    # kappa[int(N_x/15):int(14*N_x/15)+1, int(N_y/15):int(2*N_y/15)] = 0
    # kappa[int(13*N_x/15)+1:int(14*N_x/15)+1, int(N_y/15):int(14*N_y/15)+1] = 0
    # kappa[int(N_x/15):int(14*N_x/15)+1, int(13*N_y/15)+1:int(14*N_y/15)+1] = 0
    param_string += "\nkappa = {:g}\nkappa/beta = {:g} km".format(kappa_0, kappa_0 / (beta * 1000))

if use_wind is True:
    tau_x = -tau_0 * np.cos(np.pi * y / L_y) * 0
    tau_y = np.zeros((1, len(x)))
    param_string += "\ntau_0 = {:g}\nrho_0 = {:g} km".format(tau_0, rho_0)

if use_coriolis is True:  # Stole this from stackoverflow
    if use_beta is True:
        f = f_0 + beta * y  # Varying coriolis parameter
        L_R = np.sqrt(g * H) / f_0  # Rossby deformation radius
        c_R = beta * g * H / f_0 ** 2  # Long Rossby wave speed
    else:
        f = f_0 * np.ones(len(y))  # Constant coriolis parameter

    alpha = dt * f
    beta_c = alpha ** 2 / 4

    param_string += "\nf_0 = {:g}".format(f_0)
    param_string += "\nMax alpha = {:g}\n".format(alpha.max())
    param_string += "\nRossby radius: {:.1f} km".format(L_R / 1000)
    param_string += "\nRossby number: {:g}".format(np.sqrt(g * H) / (f_0 * L_x))
    param_string += "\nLong Rossby wave speed: {:.3f} m/s".format(c_R)
    param_string += "\nLong Rossby transit time: {:.2f} days".format(L_x / (c_R * 24 * 3600))
    param_string += "\n================================================================\n"

if use_source:
    sigma = np.zeros((N_x, N_y))
    sigma = 0.0001 * np.exp(-((X - L_x / 2) ** 2 / (2 * (1E+5) ** 2) + (Y - L_y / 2) ** 2 / (2 * (1E+5) ** 2)))

if use_sink is True:
    w = np.ones((N_x, N_y)) * sigma.sum() / (N_x * N_y)

with open("param_output.txt", "w") as output_file:
    output_file.write(param_string)

print(param_string)

u_n = np.zeros((N_x, N_y))
u_np1 = np.zeros((N_x, N_y))
v_n = np.zeros((N_x, N_y))
v_np1 = np.zeros((N_x, N_y))
eta_n = np.zeros((N_x, N_y))
eta_np1 = np.zeros((N_x, N_y))

h_e = np.zeros((N_x, N_y))
h_w = np.zeros((N_x, N_y))
h_n = np.zeros((N_x, N_y))
h_s = np.zeros((N_x, N_y))
uhwe = np.zeros((N_x, N_y))
vhns = np.zeros((N_x, N_y))

u_n[:, :] = 0.0  # Initial condition for u
v_n[:, :] = 0.0  # Initial condition for u
u_n[-1, :] = 0.0  # Ensuring initial u satisfy BC
v_n[:, -1] = 0.0  # Ensuring initial v satisfy BC

# eta_n[:, :] = np.sin(4*np.pi*X/L_y) + np.sin(4*np.pi*Y/L_y)
# eta_n = np.exp(-((X-0)**2/(2*(L_R)**2) + (Y-0)**2/(2*(L_R)**2)))
eta_n = np.exp(-((X - L_x / 2.7) ** 2 / (2 * (0.05E+6) ** 2) + (Y - L_y / 4) ** 2 / (2 * (0.05E+6) ** 2)))
# eta_n[int(3*N_x/8):int(5*N_x/8),int(3*N_y/8):int(5*N_y/8)] = 1.0
# eta_n[int(6*N_x/8):int(7*N_x/8),int(6*N_y/8):int(7*N_y/8)] = 1.0
# eta_n[int(3*N_x/8):int(5*N_x/8), int(13*N_y/14):] = 1.0
# eta_n[:, :] = 0.0

viz_tools.surface_plot3D(X, Y, eta_n, (X.min(), X.max()), (Y.min(), Y.max()), (eta_n.min(), eta_n.max()))

eta_list = list()
u_list = list()
v_list = list()
hm_sample = list();
ts_sample = list();
t_sample = list()
hm_sample.append(eta_n[:, int(N_y / 2)])
ts_sample.append(eta_n[int(N_x / 2), int(N_y / 2)])
t_sample.append(0.0)
anim_interval = 20
sample_interval = 1000

t_0 = time.clock()

while time_step < max_time_step:
    u_np1[:-1, :] = u_n[:-1, :] - g * dt / dx * (eta_n[1:, :] - eta_n[:-1, :])
    v_np1[:, :-1] = v_n[:, :-1] - g * dt / dy * (eta_n[:, 1:] - eta_n[:, :-1])

    if use_friction is True:
        u_np1[:-1, :] -= dt * kappa[:-1, :] * u_n[:-1, :]
        v_np1[:-1, :] -= dt * kappa[:-1, :] * v_n[:-1, :]

    if use_wind is True:
        u_np1[:-1, :] += dt * tau_x[:] / (rho_0 * H)
        v_np1[:-1, :] += dt * tau_y[:] / (rho_0 * H)

    if use_coriolis is True:
        u_np1[:, :] = (u_np1[:, :] - beta_c * u_n[:, :] + alpha * v_n[:, :]) / (1 + beta_c)
        v_np1[:, :] = (v_np1[:, :] - beta_c * v_n[:, :] - alpha * u_n[:, :]) / (1 + beta_c)

    v_np1[:, -1] = 0.0
    u_np1[-1, :] = 0.0

    h_e[:-1, :] = np.where(u_np1[:-1, :] > 0, eta_n[:-1, :] + H, eta_n[1:, :] + H)
    h_e[-1, :] = eta_n[-1, :] + H

    h_w[0, :] = eta_n[0, :] + H
    h_w[1:, :] = np.where(u_np1[:-1, :] > 0, eta_n[:-1, :] + H, eta_n[1:, :] + H)

    h_n[:, :-1] = np.where(v_np1[:, :-1] > 0, eta_n[:, :-1] + H, eta_n[:, 1:] + H)
    h_n[:, -1] = eta_n[:, -1] + H

    h_s[:, 0] = eta_n[:, 0] + H
    h_s[:, 1:] = np.where(v_np1[:, :-1] > 0, eta_n[:, :-1] + H, eta_n[:, 1:] + H)

    uhwe[0, :] = u_np1[0, :] * h_e[0, :]
    uhwe[1:, :] = u_np1[1:, :] * h_e[1:, :] - u_np1[:-1, :] * h_w[1:, :]

    vhns[:, 0] = v_np1[:, 0] * h_n[:, 0]
    vhns[:, 1:] = v_np1[:, 1:] * h_n[:, 1:] - v_np1[:, :-1] * h_s[:, 1:]

    eta_np1[:, :] = eta_n[:, :] - dt * (uhwe[:, :] / dx + vhns[:, :] / dy)

    if use_source is True:
        eta_np1[:, :] += dt * sigma

    if (use_sink is True):
        eta_np1[:, :] -= dt * w

    u_n = np.copy(u_np1)
    v_n = np.copy(v_np1)
    eta_n = np.copy(eta_np1)

    time_step += 1

    if time_step % sample_interval == 0:
        hm_sample.append(eta_n[:, int(N_y / 2)])  # Sample middle of domain for Hovmuller
        ts_sample.append(eta_n[int(N_x / 2), int(N_y / 2)])
        t_sample.append(time_step * dt)

    if time_step % anim_interval == 0:
        print("Time: \t{:.2f} hours".format(time_step * dt / 3600))
        print("Step: \t{} / {}".format(time_step, max_time_step))
        print("Mass: \t{}\n".format(np.sum(eta_n)))
        u_list.append(u_n)
        v_list.append(v_n)
        eta_list.append(eta_n)

print("Main computation loop done!\nExecution time: {:.2f} s".format(time.clock() - t_0))
print("\nVisualizing results...")

viz_tools.pmesh_plot(X, Y, eta_n, "Final state of surface elevation $\eta$")
viz_tools.quiver_plot(X, Y, u_n, v_n, "Final state of velocity field $\mathbf{u}(x,y)$")
viz_tools.hovmuller_plot(x, t_sample, hm_sample)
viz_tools.plot_time_series_and_ft(t_sample, ts_sample)
eta_anim = viz_tools.eta_animation(X, Y, eta_list, anim_interval * dt, "eta")
eta_surf_anim = viz_tools.eta_animation3D(X, Y, eta_list, anim_interval * dt, "eta_surface")
quiv_anim = viz_tools.velocity_animation(X, Y, u_list, v_list, anim_interval * dt, "velocity")

print("\nVisualization done!")
plt.show()
