import matplotlib.pyplot as plt
import math
import numpy as np
import trimesh
import tensorflow as tf
from scipy.ndimage import gaussian_filter

# draws a heatmap of real-valued pressure
def draw_pressure_amplitude(img):
    fig, ax = plt.subplots()
    ax_img = ax.imshow(img,cmap='viridis')
    plt.colorbar(ax_img)
    #ax_img.set_clim(0,2)
    #plt.xlim(100,400);plt.ylim(100,400)

# draws a heatmap of the magnitude of complex-valued pressure
def draw_pressure_complex(p):
    img=tf.sqrt(p*tf.math.conj(p)).numpy().real
    draw_pressure_amplitude(img)
    
# draws a heatmap of phase
def draw_phase(img):
    #img = (lambda x: x % (2*np.pi) )(img)
    fig, ax = plt.subplots()
    #ax_img = ax.imshow(img,cmap='viridis')
    ax_img = ax.imshow(img,cmap='twilight')
    plt.colorbar(ax_img)
    #plt.xlim(100,400);plt.ylim(100,400)

# simulate the effect of adding some noise to the phase map
def add_noise_to_phase(phase,fraction_noise):
    # phase -- matrix phase map
    # fraction_noise -- amt of noise to add, as a fraction of 2pi
    
    # generate noise in the interval [-2pi*fraction_noise, +2pi*fraction_noise)
    noise = (np.random.random(phase.shape)-0.5) * 2 * 2*np.pi * fraction_noise
    noisy_phase = phase + noise
    noisy_phase = noisy_phase % (2*np.pi)
    #print(noise)
    return noisy_phase

# simulate the effect of rounding off the phase to a given number of steps within 2pi
# eg if N_intervals = 4, then it will round off to (0, pi/2, pi, 3pi/2)
def round_off_phase(phase,N_intervals):
    # phase -- matrix phase map
    # N_intervals -- round off to N intervals within 2pi
    # so if N_intervals=4, then we round to 0, pi/2, pi, 3pi/2
    phase_rescaled = (phase % (2*np.pi)) / (2*np.pi) * N_intervals
    return np.round(phase_rescaled) * 2*np.pi / N_intervals
    
    
# creates a Trimesh Mesh object from a given heightmap for a holographic lens
# input_map: a matrix representing the topview shadow of the lens (ie the outline of the transducer). 1s where the lens should be, 0 elsewhere
# x_values, y_values: realspace (x,y) values over the area covered by input_map
# H, R: height and radius (in m) of a cylinder to wrap around the lens. if left 0, then no cylinder.
def heightmap_to_mesh(heightmap,input_map,x_values,y_values,smooth=False,H=0,R=0,center=None):
    #H = np.max(heightmap)
    #steps = 25
    #heightmap = np.round(heightmap/H*steps)*H/steps
    if smooth:
        heightmap = gaussian_filter(heightmap, sigma=1)

    # create a grid of (x,y) points
    X, Y = np.meshgrid(x_values, y_values)
    # 3d vertices: associate (x,y) points to heightmap values
    verts = np.column_stack([X.ravel(), Y.ravel(), heightmap.ravel()])

    # create the triangles that define the faces
    faces = []
    n = len(x_values)
    for i in range(n-1):
        for j in range(n-1):
            # Two triangles per quad
            a = i*n + j
            b = a + 1
            c = a + n
            d = c + 1
            # do not add triangles that connect to z=0 (the bottom of the hologram)
            #if True:
            if input_map[a]*input_map[b]*input_map[c] > 0:
                faces.append([a, b, c])
            #if True:
            if input_map[a]*input_map[b]*input_map[c] > 0:
                faces.append([b, d, c])
    
    top_mesh = trimesh.Trimesh(vertices=verts, faces=faces)

    #if smooth:
    #    #trimesh.smoothing.filter_laplacian(top_mesh)
    #    print('smoothing on')
    #    trimesh.smoothing.filter_humphrey(top_mesh,alpha=0.4,beta=1.0,iterations=50)

    # remove unused vertices so we get a circular hologram instead of a big square
    top_mesh.remove_unreferenced_vertices()

    # dumb claude addition
    full_mesh = _close_heightmap_mesh(top_mesh)

    # fill in the space underneath the triangles that define the mesh
    #myTriangles = top_mesh.triangles
    #full_mesh = trimesh.creation.truncated_prisms(tris=myTriangles)

    # Merge vertices that are shared between prisms so there are no internal gaps
    full_mesh.merge_vertices()
    full_mesh.update_faces(full_mesh.unique_faces())
    full_mesh.fill_holes()
    trimesh.repair.fix_normals(full_mesh)

    # add a ring
    if H > 0 and R > 0:
        center_x = np.mean(x_values)
        center_y = np.mean(y_values)
        if center is not None:
            center_x=center[0]
            center_y=center[1]
        bottom_ridge_height = 0
        annulus = trimesh.creation.annulus(R,R+0.0003,H+bottom_ridge_height,transform=[[1,0,0,center_x],[0,1,0,center_y],[0,0,1,H/2-bottom_ridge_height/2],[0,0,0,1]])
        #full_mesh = trimesh.util.concatenate(full_mesh,annulus)
        full_mesh = full_mesh.union(annulus, engine='manifold')

        base_height = 1e-4
        cylinder = trimesh.creation.cylinder(R,base_height,transform=[[1,0,0,center_x],[0,1,0,center_y],[0,0,1,-base_height/2],[0,0,0,1]])
        #full_mesh = trimesh.util.concatenate(full_mesh,cylinder)
        full_mesh = full_mesh.union(cylinder, engine='manifold')

    return full_mesh

def _close_heightmap_mesh(top_mesh):
    n_top = len(top_mesh.vertices)

    bottom_verts = np.copy(top_mesh.vertices)
    bottom_verts[:, 2] = 0.0
    all_verts = np.vstack([top_mesh.vertices, bottom_verts])
    # bottom index of top vertex i is i + n_top

    # Boundary edges: appear exactly once across all faces
    edges_sorted = np.sort(top_mesh.edges, axis=1)
    unique_edges, counts = np.unique(edges_sorted, axis=0, return_counts=True)
    boundary_edges = unique_edges[counts == 1]

    side_faces = []
    for a, b in boundary_edges:
        a_bot = a + n_top
        b_bot = b + n_top

        # After projecting to z=0, check if the four points are actually distinct
        va_top = top_mesh.vertices[a]
        vb_top = top_mesh.vertices[b]
        va_bot = bottom_verts[a]
        vb_bot = bottom_verts[b]

        top_edge_len = np.linalg.norm(vb_top - va_top)
        bot_edge_len = np.linalg.norm(vb_bot - va_bot)

        # Skip if projected edge collapses (vertices share x,y)
        if bot_edge_len < 1e-12:
            continue

        # If top vertex is already at z=0, it IS its own bottom vertex —
        # use the top index directly instead of the duplicate bottom index
        if np.isclose(va_top[2], 0.0, atol=1e-12):
            a_bot = a
        if np.isclose(vb_top[2], 0.0, atol=1e-12):
            b_bot = b

        # Skip degenerate triangles (any two indices are the same)
        quad = [a, b, a_bot, b_bot]
        tri1 = [a, b_bot, a_bot]
        tri2 = [a, b, b_bot]
        if len(set(tri1)) == 3:
            side_faces.append(tri1)
        if len(set(tri2)) == 3:
            side_faces.append(tri2)

    # Bottom cap: mirror top faces with flipped winding, offset to bottom layer
    bottom_faces = top_mesh.faces[:, ::-1] + n_top

    all_faces = np.vstack([
        top_mesh.faces,
        np.array(side_faces),
        bottom_faces
    ])

    mesh = trimesh.Trimesh(vertices=all_verts, faces=all_faces)
    mesh.merge_vertices()
    mesh.update_faces(mesh.unique_faces())

    # Remove any remaining degenerate faces (zero area)
    areas = mesh.area_faces
    mesh.update_faces(areas > 1e-12)
    mesh.remove_unreferenced_vertices()

    '''top_only = trimesh.Trimesh(vertices=top_mesh.vertices, faces=top_mesh.faces)
    diagnose_mesh(top_only, "top surface only")

    mesh_before_repair = trimesh.Trimesh(vertices=all_verts, faces=all_faces)
    diagnose_mesh(mesh_before_repair, "after closing (before repair)")
    diagnose_degenerate_faces(mesh_before_repair, top_mesh)

    mesh_before_repair.merge_vertices()
    mesh_before_repair.update_faces(mesh_before_repair.unique_faces())
    trimesh.repair.fix_normals(mesh_before_repair)
    diagnose_mesh(mesh_before_repair, "after repair")'''

    trimesh.repair.fix_normals(mesh)
    return mesh





def diagnose_degenerate_faces(mesh, top_mesh):
    areas = mesh.area_faces
    deg_idx = np.where(areas < 1e-12)[0]
    print(f"\n{len(deg_idx)} degenerate faces:")
    for i in deg_idx:
        f = mesh.faces[i]
        vs = mesh.vertices[f]
        print(f"  face {i}: verts {f}")
        print(f"    v0={vs[0]}")
        print(f"    v1={vs[1]}")
        print(f"    v2={vs[2]}")
        # Check which layer each vertex is from
        layers = ['top' if vi < len(top_mesh.vertices) else 'bottom' for vi in f]
        print(f"    layers: {layers}")


def diagnose_mesh(mesh, name="mesh"):
    print(f"\n--- {name} ---")
    print(f"Vertices: {len(mesh.vertices)}")
    print(f"Faces: {len(mesh.faces)}")
    print(f"Is watertight: {mesh.is_watertight}")
    print(f"Is winding consistent: {mesh.is_winding_consistent}")
    
    # Find open boundary edges (belonging to only 1 face)
    edges_unique, counts = np.unique(
        np.sort(mesh.edges, axis=1), axis=0, return_counts=True
    )
    boundary_edges = edges_unique[counts == 1]
    print(f"Boundary (open) edges: {len(boundary_edges)}")
    
    # Find non-manifold edges (belonging to 3+ faces)
    non_manifold_edges = edges_unique[counts > 2]
    print(f"Non-manifold edges: {len(non_manifold_edges)}")
    
    # Check for degenerate faces
    areas = mesh.area_faces
    degenerate = np.sum(areas < 1e-12)
    print(f"Degenerate faces (area~0): {degenerate}")
    
    # Show bounding box to sanity-check scale
    print(f"Bounds: {mesh.bounds}")

def heightmap_to_mesh_v2(heightmap,input_map,dx,H=0,R=0,center=None):
    n = heightmap.shape[0]
    
    center_x = n*dx/2
    center_y = n*dx/2
    if center!=None:
        center_x=center[0]
        center_y=center[1]

    # put down a base
    base_height = 1e-4
    mesh = trimesh.creation.cylinder(R,base_height,transform=[[1,0,0,center_x],[0,1,0,center_y],[0,0,1,-base_height/2],[0,0,0,1]])

    for i in range(n):
        for j in range(n):
            if input_map[i,j] != 0:
                h = heightmap[i,j]
                box = trimesh.creation.box(extents=[dx,dx,h],transform=[[1,0,0,i*dx],[0,1,0,j*dx],[0,0,1,h/2],[0,0,0,1]])
                mesh = trimesh.util.concatenate(mesh,box)
    
    ''' # add a ring
    if H > 0 and R > 0:
        center_x = np.mean(x_values)
        center_y = np.mean(y_values)
        if center!=None:
            center_x=center[0]
            center_y=center[1]
        bottom_ridge_height = 1e-3
        annulus = trimesh.creation.annulus(R,R+0.0003,H+bottom_ridge_height,transform=[[1,0,0,center_x],[0,1,0,center_y],[0,0,1,H/2-bottom_ridge_height/2],[0,0,0,1]])
        full_mesh = trimesh.util.concatenate(full_mesh,annulus)

        base_height = 1e-4
        cylinder = trimesh.creation.cylinder(R,base_height,transform=[[1,0,0,center_x],[0,1,0,center_y],[0,0,1,-base_height/2],[0,0,0,1]])
        full_mesh = trimesh.util.concatenate(full_mesh,cylinder)

    return full_mesh'''

    return mesh
    




