import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation

# Given parameters
a, b = 0.1, 0.02  # Major and minor axes
thetaA, vA = 40, 2  # Ellipse A: Tilt & Speed
thetaB, vB = 80, 1  # Ellipse B: Tilt & Speed

# Initial positions
xA0, yA0 = 0, 0  # A starts at origin
d = 1  # Distance between A and B
phi = np.radians(10)  # Angle of initial separation

# Compute B's initial position
xB0 = xA0 + d * np.cos(phi)
yB0 = yA0 + d * np.sin(phi)

# Convert velocities to components
thetaA_rad = np.radians(thetaA)
thetaB_rad = np.radians(thetaB)
vAx, vAy = vA * np.cos(thetaA_rad), vA * np.sin(thetaA_rad)
vBx, vBy = vB * np.cos(thetaB_rad), vB * np.sin(thetaB_rad)

# Generate boundary points of an ellipse
def get_ellipse_boundary(xc, yc, a, b, theta, num_points=100):
    t = np.linspace(0, 2*np.pi, num_points)
    x = a * np.cos(t)
    y = b * np.sin(t)
    
    # Apply rotation
    x_rot = xc + x * np.cos(theta) - y * np.sin(theta)
    y_rot = yc + x * np.sin(theta) + y * np.cos(theta)
    
    return np.column_stack((x_rot, y_rot))  # Return as Nx2 array

# Check if a point (px, py) is inside an ellipse centered at (xc, yc)
def is_inside_ellipse(px, py, xc, yc, a, b, theta):
    X = px - xc
    Y = py - yc
    term1 = ((X * np.cos(theta) + Y * np.sin(theta)) ** 2) / a**2
    term2 = ((-X * np.sin(theta) + Y * np.cos(theta)) ** 2) / b**2
    return term1 + term2 <= 1

# Animation setup
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-0.5, 2)
ax.set_ylim(-0.5, 2)
ax.set_aspect('equal')

ellipseA_patch = patches.Ellipse((xA0, yA0), 2*a, 2*b, angle=thetaA, fill=True, color='blue', alpha=0.5)
ellipseB_patch = patches.Ellipse((xB0, yB0), 2*a, 2*b, angle=thetaB, fill=True, color='red', alpha=0.5)
ax.add_patch(ellipseA_patch)
ax.add_patch(ellipseB_patch)

# Add a timer text
timer_text = ax.text(1.5, 1.8, 'Time: 0.00s', fontsize=12, color='black', bbox=dict(facecolor='white', alpha=0.6))

# Store first and last intersection times
first_intersect = None
last_intersect = None

# Animation function
def update(frame):
    global first_intersect, last_intersect
    
    t = frame / 60  # Convert frame to time (assuming 60 fps)
    xA, yA = xA0 + vAx * t, yA0 + vAy * t
    xB, yB = xB0 + vBx * t, yB0 + vBy * t

    ellipseA_patch.set_center((xA, yA))
    ellipseB_patch.set_center((xB, yB))

    # Get boundary points
    boundary_A = get_ellipse_boundary(xA, yA, a, b, thetaA_rad)
    boundary_B = get_ellipse_boundary(xB, yB, a, b, thetaB_rad)

    # Check for intersection
    intersects = any(is_inside_ellipse(px, py, xB, yB, a, b, thetaB_rad) for px, py in boundary_A) or \
                 any(is_inside_ellipse(px, py, xA, yA, a, b, thetaA_rad) for px, py in boundary_B)

    if intersects:
        if first_intersect is None:
            first_intersect = t
        last_intersect = t

    # Update timer text
    timer_text.set_text(f'Time: {t:.2f}s')

    return ellipseA_patch, ellipseB_patch, timer_text

# Close the figure after animation completes
def close_figure(_):
    plt.close(fig)

# Run animation
ani = animation.FuncAnimation(fig, update, frames=200, interval=50, blit=True)

# Set timer to close after animation duration
fig.canvas.manager.window.after(200 * 50, lambda: plt.close(fig))  # 200 frames * 50ms

plt.show()

# Print intersection times after animation completes
if first_intersect is not None:
    print(f"Intersection started at t = {first_intersect:.2f}s")
    print(f"Intersection ended at t = {last_intersect:.2f}s")
else:
    print("No intersection detected.")
