import numpy as np
import cv2
from sklearn.feature_extraction.image import extract_patches_2d as extract
from sklearn.neighbors import NearestNeighbors
# sklearn.neighbors.LSHForest gives deprecation warning, switch to pyflann
import pyflann as pyflann
"""
To get pyflann to work I had to copy the x64 directory from Lib>site-packages>pyflann>lib>win32 
into envs>python>Lib>site-packages>pyflann>win32 
"""

def remap_y(imgA,imgB):
    meanA = np.mean(imgA)
    sdA = np.std(imgA)
    meanB = np.mean(imgB)
    sdB = np.std(imgB)

    imgA_remapped = sdB/sdA*(imgA-meanA)+meanB

    return imgA_remapped

def rgb2yiq(image):
    yiq_xform = np.array([[0.299, 0.587, 0.114],
                          [0.596, -0.275, -0.321],
                         [0.212, -0.523, 0.311]])
    yiq = np.dot(image, yiq_xform.T.copy())

    return yiq


def yiq2rgb(image):
    rgb_xform = np.array([[1., 0.956, 0.619],
                          [1., -0.272, -0.647],
                          [1., -1.106, 1.703]])
    rgb = np.dot(image, rgb_xform.T.copy())

    return rgb


def get_pyramid(image, levels):
    img = image.copy()
    pyr = [img]
    for i in range(levels):
        img = cv2.pyrDown(img)
        pyr.append(img)

    return pyr


def get_features(img, causal=False):

    image_features = []

    # for each level, also get level - 1
    current_level_features = []
    #create 5x5 neighborhood for L, pad so that feature list is correct dimensions
    patches = cv2.copyMakeBorder(img,2,2,2,2,cv2.BORDER_DEFAULT)
    patches = extract(patches, (5, 5))

    # for each pixel in pyramid current level create feature vector and concatenate level and level-1
    height, width = img.shape  # dimensions of the current level of the gaussian pyramid
    for i in range(height):
        for j in range(width):
            if causal:
                current_level_vector = patches[i * width + j].flatten()[0:12]
            else:
                current_level_vector = patches[i*width+j].flatten()
            current_level_features.append(current_level_vector)

    image_features = np.vstack(current_level_features)  # put vector in format for Flann
    #image_features.append(image_vector)  # append it to total feature list, create list for each level of pyr

    return image_features


def make_analogy(A_f, Ap, B, Bp, s, coh_fact, method='pyflann'):
    if method == 'pyflann_kmeans':
        flann = pyflann.FLANN()

        print("Building FLANN kmeans index for size:", A_f.size, "for A size", Ap.size)
        flann_p = flann.build_index(A_f, algorithm="kmeans", branching=32, iterations=-1, checks=16)
        print("FLANN kmeans index done...")

    elif method == 'pyflann_kdtree':
        flann = pyflann.FLANN()

        print("Building FLANN kdtree index for size:", A_f.size, "for A size", Ap.size)
        flann_p = flann.build_index(A_f, algorithm="kdtree")
        print("FLANN kdtree index done...")

    elif method == 'nn':
        print("Building Scikit Nearest Neighbors index for size:", A_f.size, "for A size", Ap.size)
        sknn = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(A_f)
        print("NN index done...")

    B_border = cv2.copyMakeBorder(B,2,2,2,2,cv2.BORDER_DEFAULT)
    Bp_border = cv2.copyMakeBorder(Bp, 2, 2, 2, 2, cv2.BORDER_DEFAULT)

    Ap_int = np.uint8(Ap.copy()*255)
    cv2.imshow("ap", Ap_int)
    cv2.waitKey(1)
    coh_chosen = 0

    #for x in range(2, B.shape[0]-2):\
    for x in range(2, B_border.shape[0]-2):
        Bp_int = np.uint8(Bp.copy()*255)
        cv2.imshow("bp", Bp_int)
        cv2.waitKey(1)
        #for y in range(2, B.shape[1]-2):
        for y in range(2, B_border.shape[1]-2):
            bx = x-2
            by = y-2

            B_patch = B_border[x-2:x+3,y-2:y+3].flatten()
            Bp_causal = Bp_border[x-2:x+1,y-2:y+3].flatten()[0:12]
            B_ff = np.concatenate((B_patch, Bp_causal))


            if method == 'nn':
                distance, neighbor = sknn.kneighbors(B_ff[None, :])
                neighbor = int(neighbor[0])
            else:
                neighbor, distance = flann.nn_index(B_ff, 1, checks=flann_p['checks'])
            distance = distance**2
            # get p
            neighbor_ = np.unravel_index(neighbor%Ap.size, (Ap.shape[0],Ap.shape[1]))  # turn number in neighbor to coordinate in Ap

            if coh_fact > 2:  # kappa > 0
                coh_neighbor, coh_distance = get_coherent(A_f, B_ff, x, y, Ap.shape[0], Ap.shape[1],s)
                if coh_distance <= distance*coh_fact:
                    neighbor_ = coh_neighbor
                    coh_chosen += 1
            #print("Value:", Ap[m,n],"written to", x, ",", y)
            Bp[bx, by] = Ap[neighbor_]  # move luminance value (Y of YIQ) into Bprime
            # save s
            s[bx, by, :] = neighbor_
    print("coherent pixel chosen", coh_chosen, "times.")
    return Bp

def get_coherent(A_f,B_ff,x,y,M,N,s):  # tuned for 5x5 patches only
    min_distance = np.inf
    cohxy = [-1, -1]
    for i in range(5):
        for j in range(5):
            if i == 3 and j == 3:  # only do causal portion
                break
            qx,qy = int(x+i-2), int(y+j-2)
            if qx < 0 or qx >= s.shape[0] or qy < 0 or qy >= s.shape[1]:
                continue
            sx,sy = s[qx,qy]
            sx,sy = int(sx),int(sy)
            if sx < 0 or sy < 0:
                continue
            index = sx + sy*N
            rstar = np.sum((A_f[index]-B_ff)**2)
            if rstar < min_distance:
                min_distance = rstar
                cohxy = sx,sy

    return cohxy, min_distance

