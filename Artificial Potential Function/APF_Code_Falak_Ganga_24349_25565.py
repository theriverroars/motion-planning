import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting
from IPython.display import HTML

# ---------------- Simulation Parameters ----------------
dT = 0.05         # time step [s]
t_max = 2000      # maximum iterations

# ---------------- Initial Conditions ----------------
x = -0.5; y = 0.5; theta = 0.0
z = 5.0; vz = 0.0  # robot is dropped from z = 5 m

# ---------------- Physical Parameters ----------------
g = 9.81          # gravitational acceleration [m/s^2]
epsilon = 0.8     # restitution coefficient (for a visible bounce)

v_max = 1.0       # maximum horizontal speed

# ---------------- Attractive Potential Parameters (x-y plane) ----------------
zeta = 1.1547; dstar = 0.3

# ---------------- Kinematic Controller Parameters ----------------
error_theta_max = np.deg2rad(45)
Kp_omega = 1.5; omega_max = 0.5 * np.pi

# ---------------- Stationary Goal ----------------
x_goal = 10.0; y_goal = 0.5
position_accuracy = 0.05

# ---------------- Additional TASK Parameters ----------------
# TASK 5: After exiting tunnel, goal moves in x-direction
vd = 0.1

# TASK 2: Static obstacle (circular) with virtual periphery
obs_center = np.array([1.0, 1.0])
obs_radius = 0.5
virtual_periphery_static = 0.75  # repulsion activation distance
eta_static = 0.1

# TASK 3: Moving obstacle (point-sized, head-on)
moving_obs_pos = np.array([4.0, 0])   # initial position (x-y)
moving_obs_vel = np.array([-0.2, 0.0])    # constant velocity (x-y)
virtual_periphery_moving = 0.75
eta_moving = 0.15

# TASK 4: Tunnel parameters (corridor in x-y)
tunnel_start = 6.0; tunnel_end = 8.0
tunnel_lower = 0.3; tunnel_upper = 0.7  # these define the tunnel width in y

# NEW: Place tunnel at a fixed height (floating horizontally)
tunnel_floor = 0.0      # floor of tunnel at z = 3.0 m
tunnel_ceiling = 1.0    # ceiling of tunnel at z = 4.0 m
eta_tunnel = 0.2         # repulsion gain for tunnel walls

# ---------------- Precompute Simulation States ----------------
# States: (x, y, z, theta, x_goal, y_goal)
states = []
moving_obs_positions_list = []

time_sim = 0.0
for t in range(t_max):
    # Vertical dynamics (free-fall under gravity, impact, and bounce)
    if z > 0:
        vz = vz - g * dT
        z = z + vz * dT
        if z < 0:
            z = 0
            vz = -epsilon * vz  # bounce with energy loss

    # TASK 5: Move goal once robot exits tunnel region
    if x > tunnel_end:
        x_goal += vd * dT

    pos = np.array([x, y])
    goal = np.array([x_goal, y_goal])
    dist_to_goal = np.linalg.norm(pos - goal)
    if dist_to_goal <= dstar:
        nablaU_att = zeta * (pos - goal)
    else:
        nablaU_att = (dstar / dist_to_goal) * zeta * (pos - goal)

    # Compute repulsive forces from obstacles and tunnel (computed in x-y)
    nablaU_rep = np.array([0.0, 0.0])
    # Static obstacle repulsion (TASK 2)
    dist_static = np.linalg.norm(pos - obs_center)
    if dist_static <= virtual_periphery_static:
        rep_static = eta_static * (1/virtual_periphery_static - 1/dist_static) * (1/dist_static**2) * (pos - obs_center)
        nablaU_rep += rep_static
    # Moving obstacle repulsion (TASK 3)
    dist_moving = np.linalg.norm(pos - moving_obs_pos)
    if dist_moving <= virtual_periphery_moving:
        rep_moving = eta_moving * (1/virtual_periphery_moving - 1/dist_moving) * (1/dist_moving**2) * (pos - moving_obs_pos)
        nablaU_rep += rep_moving
    # Tunnel sidewalls repulsion (TASK 4) – computed in x-y only
    if tunnel_start <= x <= tunnel_end:
        d_lower = y - tunnel_lower
        if d_lower < 0.15:
            rep_lower = eta_tunnel * (1/0.15 - 1/d_lower) * (1/d_lower**2) * np.array([0, 1])
            nablaU_rep += rep_lower
        d_upper = tunnel_upper - y
        if d_upper < 0.15:
            rep_upper = eta_tunnel * (1/0.15 - 1/d_upper) * (1/d_upper**2) * np.array([0, -1])
            nablaU_rep += rep_upper

    # Total potential field in the x-y plane
    nablaU = nablaU_att + nablaU_rep
    theta_ref = math.atan2(-nablaU[1], -nablaU[0])
    error_theta = theta_ref - theta
    error_theta = np.arctan2(np.sin(error_theta), np.cos(error_theta))
    if abs(error_theta) <= error_theta_max:
        alpha = (error_theta_max - abs(error_theta)) / error_theta_max
        v_ref = min(alpha * np.linalg.norm(-nablaU), v_max)
    else:
        v_ref = 0.0
    omega_ref = Kp_omega * error_theta
    omega_ref = np.clip(omega_ref, -omega_max, omega_max)
    
    # Update horizontal state (simple Euler integration)
    theta = theta + omega_ref * dT
    x = x + v_ref * math.cos(theta) * dT
    y = y + v_ref * math.sin(theta) * dT

    # Update moving obstacle position (in x-y)
    moving_obs_pos = moving_obs_pos + moving_obs_vel * dT

    states.append((x, y, z, theta, x_goal, y_goal))
    moving_obs_positions_list.append(moving_obs_pos.copy())
    time_sim += dT

    if np.linalg.norm(np.array([x, y]) - np.array([x_goal, y_goal])) < position_accuracy and z == 0:
        break

states = np.array(states)                # shape: (N, 6)
moving_obs_positions = np.array(moving_obs_positions_list)  # shape: (N, 2)

# ---------------- Create 3D Animation ----------------
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

def update(frame):
    ax.clear()
    # Set 3D axis limits
    ax.set_xlim([-1, 12])
    ax.set_ylim([-3, 3])
    ax.set_zlim([-1, 6])
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    t = frame * dT
    x_val, y_val, z_val, theta_val, x_goal_val, y_goal_val = states[frame]
    ax.set_title(f"Time: {t:.2f} s")

    # --- Plot Floor (Colored) ---
    X_floor, Y_floor = np.meshgrid(np.linspace(-1, 12, 20), np.linspace(-3, 3, 20))
    Z_floor = np.zeros_like(X_floor)
    ax.plot_surface(X_floor, Y_floor, Z_floor, color='lightgray', alpha=0.5)

    # --- Plot the Robot's 3D Trajectory and Current Position ---
    ax.plot(states[:frame, 0], states[:frame, 1], states[:frame, 2], '-b')
    ax.scatter(x_val, y_val, z_val, c='g', marker='s', s=50)

    # --- Plot Static Obstacle as a Cylinder ---
    z_cyl = np.linspace(0, 6, 30)
    theta_cyl = np.linspace(0, 2 * np.pi, 30)
    theta_cyl, Z_cyl = np.meshgrid(theta_cyl, z_cyl)
    X_cyl = obs_center[0] + obs_radius * np.cos(theta_cyl)
    Y_cyl = obs_center[1] + obs_radius * np.sin(theta_cyl)
    ax.plot_surface(X_cyl, Y_cyl, Z_cyl, color='r', alpha=0.3)
    # Virtual periphery
    X_vir = obs_center[0] + virtual_periphery_static * np.cos(theta_cyl)
    Y_vir = obs_center[1] + virtual_periphery_static * np.sin(theta_cyl)
    ax.plot_surface(X_vir, Y_vir, Z_cyl, color='r', alpha=0.1)

    # --- Plot Moving Obstacle as a Sphere ---
    mo_x = moving_obs_positions[frame, 0]
    mo_y = moving_obs_positions[frame, 1]
    mo_z = 0.5  # fixed z for the moving obstacle
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    u, v = np.meshgrid(u, v)
    r_sphere = 0.2
    X_sphere = mo_x + r_sphere * np.cos(u) * np.sin(v)
    Y_sphere = mo_y + r_sphere * np.sin(u) * np.sin(v)
    Z_sphere = mo_z + r_sphere * np.cos(v)
    ax.plot_surface(X_sphere, Y_sphere, Z_sphere, color='k', alpha=0.5)

    # --- Plot Tunnel as a Horizontally Floating Structure ---
    # Tunnel is now placed at a fixed height: from tunnel_floor to tunnel_ceiling.
    # We'll plot its floor, ceiling, and side walls.
    # Floor and ceiling:
    X_tunnel, Y_tunnel = np.meshgrid(np.linspace(tunnel_start, tunnel_end, 30),
                                     np.linspace(tunnel_lower, tunnel_upper, 30))
    Z_floor_tunnel = np.full_like(X_tunnel, tunnel_floor)
    Z_ceiling_tunnel = np.full_like(X_tunnel, tunnel_ceiling)
    ax.plot_surface(X_tunnel, Y_tunnel, Z_floor_tunnel, color='m', alpha=0.3)
    ax.plot_surface(X_tunnel, Y_tunnel, Z_ceiling_tunnel, color='m', alpha=0.3)
    # Side walls:
    # Left wall at y = tunnel_lower
    Z_side = np.linspace(tunnel_floor, tunnel_ceiling, 30)
    X_side, Z_side = np.meshgrid(np.linspace(tunnel_start, tunnel_end, 30), Z_side)
    Y_side_left = np.full_like(X_side, tunnel_lower)
    Y_side_right = np.full_like(X_side, tunnel_upper)
    ax.plot_surface(X_side, Y_side_left, Z_side, color='m', alpha=0.3)
    ax.plot_surface(X_side, Y_side_right, Z_side, color='m', alpha=0.3)

    # --- Plot Goal as a Blue Sphere (fixed z, e.g., 0.5 m) ---
    goal_z = 0.5
    ax.scatter(x_goal_val, y_goal_val, goal_z, c='b', marker='o', s=50)

    return []

ani = animation.FuncAnimation(fig, update, frames=len(states), interval=10, blit=False)
#HTML(ani.to_html5_video())
plt.show()