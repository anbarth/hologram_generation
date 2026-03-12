import matplotlib.pyplot as plt
import math
import numpy as np
import trimesh
import tensorflow as tf
from scipy.ndimage import gaussian_filter

def draw_pressure_amplitude(img):
    fig, ax = plt.subplots()
    ax_img = ax.imshow(img,cmap='viridis')
    plt.colorbar(ax_img)
    #plt.xlim(100,400);plt.ylim(100,400)

def draw_pressure_complex(p):
    img=tf.sqrt(p*tf.math.conj(p)).numpy().real
    draw_pressure_amplitude(img)
    

def draw_phase(img):
    img = (lambda x: x % (2*np.pi) )(img)
    fig, ax = plt.subplots()
    #ax_img = ax.imshow(img,cmap='viridis')
    ax_img = ax.imshow(img,cmap='twilight')
    plt.colorbar(ax_img)
    #plt.xlim(100,400);plt.ylim(100,400)

def heightmap_to_mesh(heightmap,input_map,x_values,y_values,smooth=False):
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
            if input_map[a]*input_map[b]*input_map[c] > 0:
                faces.append([a, b, c])
            if input_map[a]*input_map[b]*input_map[c] > 0:
                faces.append([b, d, c])
    top_mesh = trimesh.Trimesh(vertices=verts, faces=faces)

    #if smooth:
    #    #trimesh.smoothing.filter_laplacian(top_mesh)
    #    print('smoothing on')
    #    trimesh.smoothing.filter_humphrey(top_mesh,alpha=0.4,beta=1.0,iterations=50)


    # remove unused vertices so we get a circular hologram instead of a big square
    top_mesh.remove_unreferenced_vertices()


    # fill in the space underneath the triangles that define the mesh
    myTriangles = top_mesh.triangles
    full_mesh = trimesh.creation.truncated_prisms(tris=myTriangles)
    return full_mesh
