import cv2
import numpy as np
import scipy.io
from numpy import matlib
from math import *


def setupParams():
    params = {}
    params['saliancy_map_maxsize'] = 32
    params['blur_fraction'] = 0.02

    params['feature_channels'] = 'DIO'  # DKL color, Intensity, Rotation
    params['intensityWeight'] = 1
    # params['colorWeight'] = 1
    params['intensityWeight'] = 1
    params['orientationWeight'] = 1
    # params['contrastWeight'] = 1
    # params['flickerWeight'] = 1
    # params['motionWeight'] = 1
    params['dklcolorWeight'] = 1
    params['gaborangles'] = [0, 45, 90, 135]
    # params['flickerNewFrameWt'] = 1
    params['motionAngles'] = [0, 45, 90, 135]

    params['unCenterBias'] = 0
    params['levels'] = [2, 3, 4]
    params['multilevels'] = []
    params['sigma_frac_act'] = 0.15
    params['sigma_frac_norm'] = 0.06
    params['num_norm_iters'] = 1
    params['tol'] = 0.0001
    params['cyclic_type'] = 2
    params['normalizationType'] = 1
    params['normalizeTopChannelMaps'] = 0

    return params


def show(img):
    # cv2.imshow("i", img), cv2.waitKey()
    pass


def C_operation_BY(img_L, img_B, img_G, img_R):
    min_rg = np.minimum(img_R, img_G)
    b_min_rg = abs(np.subtract(img_B, min_rg))
    op = np.divide(b_min_rg, img_L, out=np.zeros_like(img_L), where=img_L != 0)
    return op


def C_operation_RG(img_L, img_B, img_G, img_R):
    r_g = abs(np.subtract(img_R, img_G))
    op = np.divide(r_g, img_L, out=np.zeros_like(img_L), where=img_L != 0)
    return op


def getGaborFiterMap(gaborparams, angle, phase):
    gp = gaborparams
    major_sd = gp['stddev']
    minor_sd = major_sd*gp['elongation']
    max_sd = max(major_sd, minor_sd)

    sz = gp['filterSize']
    if sz==-1:
        sz = ceil(max_sd*sqrt(10))
    else:
        sz = floor(sz/2)

    psi = np.pi / 180 * phase
    rtDeg = np.pi / 180 * angle

    omega = 2 * np.pi / gp['filterPeriod']
    co = cos(rtDeg)
    si = -sin(rtDeg)
    major_sigq = 2 * pow(major_sd, 2)
    minor_sigq = 2 * pow(minor_sd, 2)

    vec = range(-int(sz), int(sz)+1)
    vlen = len(vec)
    vco = [i*co for i in vec]
    vsi = [i*si for i in vec]

    # major = np.matlib.repmat(np.asarray(vco).transpose(), 1, vlen) + np.matlib.repmat(vsi, vlen, 1)
    a = np.tile(np.asarray(vco).transpose(), (vlen, 1)).transpose()
    b = np.matlib.repmat(vsi, vlen, 1)
    major = a + b
    major2 = np.power(major, 2)

    # minor = np.matlib.repmat(np.asarray(vsi).transpose(), 1, vlen) - np.matlib.repmat(vco, vlen, 1)
    a = np.tile(np.asarray(vsi).transpose(), (vlen, 1)).transpose()
    b = np.matlib.repmat(vco, vlen, 1)
    minor = a + b
    minor2 = np.power(minor, 2)

    a = np.cos(omega * major + psi)
    b = np.exp(-major2/major_sigq - minor2/minor_sigq)
    # result = np.cos(omega * major + psi) * exp(-major2/major_sigq - minor2/minor_sigq)
    result = np.multiply(a, b)

    filter1 = np.subtract(result , np.mean(result.reshape(-1)))
    filter1 = np.divide(filter1 , np.sqrt(np.sum(np.power(filter1.reshape(-1), 2))))
    return filter1


def getGaborFiters(thetas):
    gaborparams = {}
    gaborparams['stddev'] = 2
    gaborparams['elongation'] = 2
    gaborparams['filterSize'] = -1
    gaborparams['filterPeriod'] = np.pi

    gaborFilters = {}
    for th in thetas:
        gaborFilters[th] = {}
        gaborFilters[th]['0'] = getGaborFiterMap(gaborparams, th, 0)
        gaborFilters[th]['90'] = getGaborFiterMap(gaborparams, th, 90)

    return gaborFilters


def loadGraphDistanceMatrixFor28x32():
    f = scipy.io.loadmat("./28__32__m__2.mat")
    distanceMat = np.array(f['grframe'])[0][0][0]
    lx = np.array(f['grframe'])[0][0][1]
    dim = np.array(f['grframe'])[0][0][2]
    return [distanceMat, lx, dim]


### step 1 : computing feature maps
def getFeatureMaps(img, params):
    maps = {}
    maps['org'] = {}
    maps['res'] = {}
    # creating image pyramids
    max_level = 4
    imgb = img[:, :, 0]
    imgg = img[:, :, 1]
    imgr = img[:, :, 2]
    imgi = np.maximum(imgr, imgb, imgg)

    img_R = [cv2.pyrDown(imgr)]
    img_G = [cv2.pyrDown(imgg)]
    img_B = [cv2.pyrDown(imgb)]
    img_L = [cv2.pyrDown(imgi)]

    for i in range(1, max_level):
        img_R.append(cv2.pyrDown(img_R[i - 1]))
        img_G.append(cv2.pyrDown(img_G[i - 1]))
        img_B.append(cv2.pyrDown(img_B[i - 1]))
        img_L.append(cv2.pyrDown(img_L[i - 1]))

    print len(img_B)
    # cv2.imshow("1", img_L[0]), cv2.waitKey()

    # computing feature maps
    # computing C - feature maps
    maps['org']['CBY'] = []
    maps['res']['CBY'] = []

    maps['org']['CRG'] = []
    maps['res']['CRG'] = []

    for i in range(1, max_level):
        op = C_operation_BY(img_L[i], img_B[i], img_G[i], img_R[i])
        show(op)

        maps['org']['CBY'].append(op)

        res = cv2.resize(op, (32, 28), interpolation = cv2.INTER_CUBIC)
        show(res)

        maps['res']['CBY'].append(res)

        op = C_operation_RG(img_L[i], img_B[i], img_G[i], img_R[i])
        show(op)

        maps['org']['CRG'].append(op)

        res = cv2.resize(op, (32, 28), interpolation=cv2.INTER_CUBIC)
        show(res)

        maps['res']['CRG'].append(res)


    # computing I- feature Map
    maps['org']['I'] =[]
    maps['res']['I'] = []
    for i in range(1, max_level):
        maps['org']['I'].append(img_L[i])
        res = cv2.resize(img_L[i], (32, 28), interpolation=cv2.INTER_CUBIC)
        maps['res']['I'].append(res)


    # computing Orientation Maps

    thetas = params['gaborangles']
    for th in thetas:
        maps['org'][th] = {}
        maps['org'][th][0] = []
        maps['org'][th][90] = []

        maps['res'][th] = {}
        maps['res'][th][0] = []
        maps['res'][th][90] = []

        for i in range(1, max_level):
            img = img_L[i]

            kernelPair=getGaborFiters([th])
            kernel_0 = kernelPair[th]['0']
            kernel_90 = kernelPair[th]['90']

            o1 = cv2.filter2D(img, -1, kernel_0, borderType=cv2.BORDER_REPLICATE)
            o2 = cv2.filter2D(img, -1, kernel_90, borderType=cv2.BORDER_REPLICATE)

            maps['org'][th][0].append(o1)
            maps['org'][th][90].append(o2)
            show(o1), show(o2)

            r1 = cv2.resize(o1, (32, 28), interpolation=cv2.INTER_CUBIC)
            r2 = cv2.resize(o2, (32, 28), interpolation=cv2.INTER_CUBIC)


            maps['res'][th][0].append(r1)
            maps['res'][th][90].append(r2)
            show(r1), show(r2)

    return maps




if __name__ == "__main__":
    img = cv2.imread("1.jpg")
    img = img / 255.0
    show(img)
    params = setupParams()
    # getFeatureMaps(img, params)
    # [a,b,c] = loadGraphDistanceMatrixFor28x32()
    pass