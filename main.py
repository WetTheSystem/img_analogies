import errno
import os
import sys
import numpy as np
import cv2
import analogy
import helpers

def readImages():
    imgA = cv2.imread("src/A.jpg", cv2.IMREAD_UNCHANGED)/255.0
    imgAp = cv2.imread("src/Ap.jpg", cv2.IMREAD_UNCHANGED)/255.0
    imgB = cv2.imread("src/B.jpg", cv2.IMREAD_UNCHANGED)/255.0

    return (imgA, imgAp, imgB)


pyr_levels = 5
kappa = 400
# method can be pyflann_kmeans, pyflann_kdtree or nn
method = 'pyflann_kdtree'
imgA, imgAp, imgB = readImages()

A_ = analogy.rgb2yiq(imgA)
Ap_ = analogy.rgb2yiq(imgAp)
B_ = analogy.rgb2yiq(imgB)
Bp_ = B_.copy()
A_ = analogy.remap_y(A_, B_)

A_L = analogy.get_pyramid(analogy.rgb2yiq(imgA)[:,:,0], pyr_levels)
B_L = analogy.get_pyramid(analogy.rgb2yiq(imgB)[:,:,0], pyr_levels)
Ap_L = analogy.get_pyramid(analogy.rgb2yiq(imgAp)[:,:,0], pyr_levels)
Bp_L = []
s = []
for i in range(len(B_L)):
    Bp_L.append(np.zeros(B_L[i].shape))
    s.append(np.zeros((B_L[i].shape[0],B_L[i].shape[1],2)))

A_f = []
B_f = []
cv2.imwrite("apl.jpg", Ap_L[pyr_levels])
cv2.imwrite("bpl.jpg", Bp_L[pyr_levels])
#Bp_L[pyr_levels] = cv2.resize(Ap_L[pyr_levels], dsize=(Bp_L[pyr_levels].shape[1],Bp_L[pyr_levels].shape[0]))
cv2.imwrite("bplreshaped.jpg", Bp_L[pyr_levels])

# process pyramid from coursest to finest
for lvl in range(pyr_levels, -1, -1):
    print("Starting Level: ", lvl, "of ", pyr_levels)
    A_f = analogy.get_features(A_L[lvl])
    Ap_f = analogy.get_features(Ap_L[lvl], causal=True)
    A_ff = np.concatenate((A_f, Ap_f),1)

    if lvl < pyr_levels:
        Ad_f = analogy.get_features(cv2.resize(A_L[lvl + 1], (A_L[lvl].shape[1], A_L[lvl].shape[0])))
        Apd_f = analogy.get_features(cv2.resize(Ap_L[lvl + 1], (Ap_L[lvl].shape[1], Ap_L[lvl].shape[0])), causal=True)
        Ad_ff = np.concatenate((Ad_f, Apd_f),1)
        A_ff = np.concatenate((A_ff, Ad_ff), 0)

    # take previous pyramid level and resize it to the same shape as the current level
    if lvl < pyr_levels:
        Bp_holder = cv2.resize(Bp_L[lvl+1], dsize=(Bp_L[lvl].shape[1], Bp_L[lvl].shape[0]))
        #coh_fact = 1 + 2.0 ** (lvl - pyr_levels) * kappa
    else:
        #Bp_holder = cv2.resize(Ap_L[lvl], dsize=(Bp_L[lvl].shape[1], Bp_L[lvl].shape[0]))
        Bp_holder = Bp_L[lvl].copy()
        #coh_fact = 0
    coh_fact = 1 + 2.0 ** (lvl - pyr_levels) * kappa
    Bp_L[lvl] = analogy.make_analogy(A_ff, Ap_L[lvl], B_L[lvl], Bp_holder, s[lvl], coh_fact, method)
    #cv2.imwrite("bpl_lvl.jpg", Bp_L[lvl])

# move luminance values of final pyramid into Bp_
Bp_[:,:,0] = Bp_L[0][:,:]

imgBp = analogy.yiq2rgb(Bp_)
imgBp = imgBp * 255.0
imgBp[imgBp > 255] = 255
imgBp[imgBp < 0] = 0

cv2.imwrite("out/Bp.jpg", imgBp)