import trimesh
import numpy as np
import os
import plotly.graph_objects as go
import plotly.io as pio
import mne
from scipy.spatial import ConvexHull

# Set Plotly renderer
pio.renderers.default = "browser"

# **Step 1: Set File Paths**
brain_obj_path = r"C:\Brain Model\uploads_files_4417378_Brain.obj"
edf_file_path = r"C:\Brain Model\sub-PY18N015_ses-extraoperative_task-interictal_run-01_ieeg.edf"


# **Step 2: Load Brain Model**
if os.path.exists(brain_obj_path):
    brain_mesh_path = brain_obj_path
    print("✅ Using OBJ file for brain model.")
else:
    raise FileNotFoundError("❌ Brain model file not found!")

try:
    brain_mesh = trimesh.load_mesh(brain_mesh_path, process=True)
except Exception as e:
    raise ValueError(f"❌ Error loading brain model: {e}")

# **Step 3: Extract Brain Mesh Boundaries and Convex Hull**
brain_vertices = brain_mesh.vertices
brain_faces = brain_mesh.faces

brain_x_min, brain_x_max = brain_vertices[:, 0].min(), brain_vertices[:, 0].max()
brain_y_min, brain_y_max = brain_vertices[:, 1].min(), brain_vertices[:, 1].max()
brain_z_min, brain_z_max = brain_vertices[:, 2].min(), brain_vertices[:, 2].max()

print(f"📏 Brain Model Boundaries - X: ({brain_x_min}, {brain_x_max}), "
      f"Y: ({brain_y_min}, {brain_y_max}), Z: ({brain_z_min}, {brain_z_max})")

# Compute the convex hull of the brain model (outer surface)
brain_hull = ConvexHull(brain_vertices)

# **Step 4: Load iEEG Data**
raw = mne.io.read_raw_edf(edf_file_path, preload=True)
raw.rename_channels(lambda x: x + "_dup" if x.startswith("POL") else x)

# **Step 5: Assign Standard iEEG Electrode Positions**
standard_ieeg_positions = {
    "LOF1": (-1, 4, -1),    "LA1": (-1.8, 3.8, -1.2),
    "LA2": (-1.5, 3.6, -1.5), "LA3": (-1.2, 3.4, -1.8),
    "LH1": (-1, 0.5, -1.5), "LH2": (-1.1, 0.6, -1.6),
    "LH3": (-1.2, 0.7, -1.7), "RAH1": (1, 4, -1),
    "RAH2": (1.5, 3.8, -1.2), "RAH3": (1.8, 3.6, -1.5),
    "RPH1": (1, 0.5, -1.5), "RPH2": (1.1, 0.6, -1.6),
    "RPH3": (1.2, 0.7, -1.7)
}

# **Step 6: Assign Randomized Positions for Normal Channels**
np.random.seed(42)
normal_channels = [ch for ch in raw.ch_names if ch not in standard_ieeg_positions]
normal_positions = {
    ch: (
        np.random.uniform(brain_x_min, brain_x_max),
        np.random.uniform(brain_y_min, brain_y_max),
        np.random.uniform(brain_z_min, brain_z_max)
    )
    for ch in normal_channels
}

# **Step 7: Merge Positions**
channel_positions = {**standard_ieeg_positions, **normal_positions}

# Extract electrode positions
soz_channels = list(standard_ieeg_positions.keys())  
normal_channels = list(normal_positions.keys())

# Convert to numpy arrays
soz_positions_arr = np.array([channel_positions[ch] for ch in soz_channels])
normal_positions_arr = np.array([channel_positions[ch] for ch in normal_channels])

# **Step 8: Project Electrodes onto the Brain Surface**
def project_to_brain_surface(points, brain_vertices, brain_hull):
    """
    Projects points onto the convex hull of the brain to ensure they stay on the brain surface.
    """
    projected_points = []
    for point in points:
        # Find the closest vertex in the brain model
        distances = np.linalg.norm(brain_vertices - point, axis=1)
        closest_vertex = brain_vertices[np.argmin(distances)]
        projected_points.append(closest_vertex)
    return np.array(projected_points)

# Project SOZ and normal electrodes onto the brain surface
soz_positions_arr = project_to_brain_surface(soz_positions_arr, brain_vertices, brain_hull)
normal_positions_arr = project_to_brain_surface(normal_positions_arr, brain_vertices, brain_hull)

# **Step 9: Extract Final Corrected XYZ Coordinates**
x_soz, y_soz, z_soz = soz_positions_arr.T
x_normal, y_normal, z_normal = normal_positions_arr.T

# **Step 10: Create 3D Plotly Visualization**
fig = go.Figure()

# **Add Brain Model**
fig.add_trace(go.Mesh3d(
    x=brain_vertices[:, 0], y=brain_vertices[:, 1], z=brain_vertices[:, 2],
    i=brain_faces[:, 0], j=brain_faces[:, 1], k=brain_faces[:, 2],  
    opacity=0.5, color='rgb(169,169,169)', name="Brain Model"
))

# **Add Normal iEEG Electrodes (Blue)**
fig.add_trace(go.Scatter3d(
    x=x_normal, y=y_normal, z=z_normal,
    mode='markers',
    marker=dict(size=6, color='blue', opacity=0.9),
    text=[f"Channel: {ch}" for ch in normal_channels],  
    hoverinfo="text"
))

# **Add SOZ iEEG Electrodes (Red)**
fig.add_trace(go.Scatter3d(
    x=x_soz, y=y_soz, z=z_soz,
    mode='markers+text',
    marker=dict(size=10, color='red', opacity=1.0, line=dict(color='black', width=2)),
    text=[f"SOZ: {ch}" for ch in soz_channels], 
    textposition="top center"
))

# **Step 11: Adjust Camera & Layout**
fig.update_layout(
    title="Final 3D Brain Model with Intracranial iEEG Electrodes (SOZ in Red)",
    scene=dict(
        xaxis_title="X (mm)", yaxis_title="Y (mm)", zaxis_title="Z (mm)",
        aspectmode="data",
        camera=dict(eye=dict(x=2, y=2, z=1.5))
    )
)

# Show the Figure
fig.show()


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create Matplotlib figure for 2D projection
fig_2d = plt.figure(figsize=(8, 6), dpi=1200)  # 8x6 inches at 1200 DPI
ax = fig_2d.add_subplot(111, projection='3d')

# Plot Brain Model (Wireframe for better clarity)
ax.plot_trisurf(brain_vertices[:, 0], brain_vertices[:, 1], brain_vertices[:, 2],
                triangles=brain_faces, alpha=0.4, color='gray')

# Plot Normal iEEG Electrodes (Blue)
ax.scatter(x_normal, y_normal, z_normal, c='blue', s=10, label="Normal iEEG Electrodes")

# Plot SOZ iEEG Electrodes (Red)
ax.scatter(x_soz, y_soz, z_soz, c='red', s=20, edgecolors='black', label="SOZ iEEG Electrodes")

# Labels and Title with Smaller Font Size
ax.set_xlabel("X (mm)", fontsize=10, labelpad=10)
ax.set_ylabel("Y (mm)", fontsize=10, labelpad=10)
ax.set_zlabel("Z (mm)", fontsize=10, labelpad=10)
ax.set_title("2D Projection of Brain Model with iEEG Electrodes", fontsize=12, pad=20)

# Adjust View (Top-down for 2D projection)
ax.view_init(elev=90, azim=0)

# Fix Overlapping Text Issues
plt.tight_layout()

# Save Figure at 1200 DPI
output_figure_path_2d = "C:\brain_ieeg_projection_fixed.png"
plt.savefig(output_figure_path_2d, dpi=1200, bbox_inches='tight')

print(f"✅ 2D Projection saved successfully at 1200 DPI: {output_figure_path_2d}")

# Show the 2D projection
plt.show()


