import tensorflow as tf
import numpy as np
import cv2

def increase_resolution(mat,N_new):
    N_old = mat.shape[0]
    if not (N_new % N_old == 0):
        return -1
    m = int(N_new / N_old)
    new_mat = mat.repeat(m, axis=0).repeat(m, axis=1)
    return new_mat

# N: square grid side length (px)
# R: circle radius (px)
def circle_matrix(N,R):
    mymat = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if (i-N/2)*(i-N/2) + (j-N/2)*(j-N/2) < R*R:
                mymat[i,j]=1

    # convert to tensor with appropriate datatype
    mymat = tf.constant(mymat)
    mymat = tf.dtypes.cast(mymat, tf.float64)
    return mymat

# N: square grid side length (px)
# R: circle radius (px)
# nSections: number of sections for pinwheel
def pinwheel_matrix(N,R,nSections):
    if nSections%2 != 0:
        print('n_sections must be even for pinwheel')

    theta = 2*np.pi/nSections
    mymat = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            myAngle = np.arctan2(j-N/2,i-N/2) + np.pi
            mySection = np.floor(myAngle/theta)
            if (i-N/2)*(i-N/2) + (j-N/2)*(j-N/2) < R*R and mySection % 2 == 0:
                mymat[i,j]=1

    # convert to tensor with appropriate datatype
    mymat = tf.constant(mymat)
    mymat = tf.dtypes.cast(mymat, tf.float64)
    return mymat

# N: square grid side length (px)
# R1, R2: inner, outer radii (px)
def bullseye_matrix(N,R1,R2):
    mymat = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            r2 = (i-N/2)*(i-N/2) + (j-N/2)*(j-N/2)
            if r2 < R2*R2 and r2 > R1*R1: 
                mymat[i,j]=1
    
    # convert to tensor with appropriate datatype
    mymat = tf.constant(mymat)
    mymat = tf.dtypes.cast(mymat, tf.float64)
    return mymat

# img_file_name: image to be read. black=high pressure; white=low pressure
# N: square grid side length for output (px)
def img_to_matrix(img_file_name,N):
    # read and resize image
    img_np = cv2.imread(img_file_name, cv2.IMREAD_GRAYSCALE)
    size = (N, N)
    img_np = cv2.resize(img_np, size)
    
    # normalize to 1
    target_amp=1
    # amp_img = tf.constant(amp_img_np / 255.0 * target_amp)
    #print(np.max(img_np))
    img_tf = tf.constant(target_amp - img_np / 255.0 * target_amp) # inverted

    # convert to tensor with appropriate datatype
    img_tf = tf.dtypes.cast(img_tf, tf.float64)
    return img_tf