import cv2
import numpy as np
import scipy.io
from numpy import matlib
from math import *
from sklearn.preprocessing import normalize
from matplotlib import pyplot as plt


def setupParams():
    params = {}
    params['gaborangles'] = [0, 45, 90, 135]
    params['sigma_frac_act'] = 0.15
    params['sigma_frac_norm'] = 0.06

    return params

def show(img):
    plt.imshow(img)
    plt.show()
    # cv2.namedWindow("i", flags=cv2.WINDOW_KEEPRATIO)
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
    minor_sd = major_sd * gp['elongation']
    max_sd = max(major_sd, minor_sd)

    sz = gp['filterSize']
    if sz == -1:
        sz = ceil(max_sd * sqrt(10))
    else:
        sz = floor(sz / 2)

    psi = np.pi / 180 * phase
    rtDeg = np.pi / 180 * angle

    omega = 2 * np.pi / gp['filterPeriod']
    co = cos(rtDeg)
    si = -sin(rtDeg)
    major_sigq = 2 * pow(major_sd, 2)
    minor_sigq = 2 * pow(minor_sd, 2)

    vec = range(-int(sz), int(sz) + 1)
    vlen = len(vec)
    vco = [i * co for i in vec]
    vsi = [i * si for i in vec]

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
    b = np.exp(-major2 / major_sigq - minor2 / minor_sigq)
    # result = np.cos(omega * major + psi) * exp(-major2/major_sigq - minor2/minor_sigq)
    result = np.multiply(a, b)

    filter1 = np.subtract(result, np.mean(result.reshape(-1)))
    filter1 = np.divide(filter1, np.sqrt(np.sum(np.power(filter1.reshape(-1), 2))))
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

def extractAllFeatureMaps(featMaps):
    linedUpMaps = []
    [linedUpMaps.append(a) for a in featMaps['res']['CBY']]
    [linedUpMaps.append(a) for a in featMaps['res']['CRG']]
    [linedUpMaps.append(a) for a in featMaps['res']['I']]
    [linedUpMaps.append(a) for a in featMaps['res'][0]]
    [linedUpMaps.append(a) for a in featMaps['res'][45]]
    [linedUpMaps.append(a) for a in featMaps['res'][90]]
    [linedUpMaps.append(a) for a in featMaps['res'][135]]
    return linedUpMaps

def computeEigenVector(mat):
    w,h = mat.shape
    diff = 1
    v = np.divide(np.ones((w, 1), dtype=np.float32), w)
    oldv = v
    oldoldv = v

    while diff > 0.0001 :
        oldv = v
        oldoldv = oldv
        v = np.dot(mat,v)
        diff = np.linalg.norm(oldv - v, ord=2)
        s = sum(v)
        if s>=0 and s< np.inf:
            continue
        else:
            v = oldoldv
            break

    v = np.divide(v, sum(v))

    return v

def computeGraphSaliencyForAFeatMap(params, map):
    [distanceMat, lx, dims] = loadGraphDistanceMatrixFor28x32()
    sigma = params['sigma_frac_act'] * np.mean(map.shape) # Just 0.15 percent of width ( we took avg of both dims)
    denom = 2*pow(sigma, 2)
    expr = -np.divide(distanceMat, denom)
    Fab = np.exp(expr)

    map_linear = np.ravel(map, order='F') # column major
    # map_linear = map.reshape(-1)
    state_transition_matrix = np.zeros_like(distanceMat, dtype=np.float32)
    # calculating STM : w = d*Fab
    for i in xrange(distanceMat.shape[0]):
        for j in xrange(distanceMat.shape[1]):
            state_transition_matrix[i][j] = Fab[i][j]*abs(map_linear[i] - map_linear[j])
    # normalising outgoing weights of each node to sum to 1, using scikit normalize
    norm_STM = normalize(state_transition_matrix, axis=0, norm='l1')
    # print sum(norm_STM)

    # caomputing equilibrium state of a markv chain is same as computing eigen vector of its weight matrix
    # https://lps.lexingtonma.org/cms/lib2/MA01001631/Centricity/Domain/955/EigenApplications%20to%20Markov%20Chains.pdf
    eVec = computeEigenVector(norm_STM)
    processed_reshaped = np.reshape(eVec, map.shape, order='F')
    return processed_reshaped

def normaliseUsingGraphBasedSaliency(params, map):
    [distanceMat, lx, dims] = loadGraphDistanceMatrixFor28x32()
    sigma = params['sigma_frac_norm'] * np.mean(map.shape)  # Just 0.15 percent of width ( we took avg of both dims)
    denom = 2 * pow(sigma, 2)
    expr = -np.divide(distanceMat, denom)
    Fab = np.exp(expr)

    act_map_linear = np.ravel(map, order='F')
    STM = np.zeros_like(distanceMat, dtype=np.float32)

    # calculating STM : w = A*Fab
    for i in xrange(distanceMat.shape[0]):
        for j in xrange(distanceMat.shape[1]):
            STM[i][j] = Fab[i][j]*abs(act_map_linear[i])

    # normalising outgoing weights of each node to sum to 1, using scikit normalize
    norm_STM = normalize(STM, axis=0, norm='l1')
    # print sum(norm_STM)

    # caomputing equilibrium state of a markv chain is same as computing eigen vector of its weight matrix
    # https://lps.lexingtonma.org/cms/lib2/MA01001631/Centricity/Domain/955/EigenApplications%20to%20Markov%20Chains.pdf
    eVec = computeEigenVector(norm_STM)
    processed_reshaped = np.reshape(eVec, map.shape, order='F')
    return processed_reshaped


###===========================================================================================================
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
        #show((op)

        maps['org']['CBY'].append(op)

        res = cv2.resize(op, (32, 28), interpolation=cv2.INTER_CUBIC)
        #show((res)

        maps['res']['CBY'].append(res)

        op = C_operation_RG(img_L[i], img_B[i], img_G[i], img_R[i])
        #show((op)

        maps['org']['CRG'].append(op)

        res = cv2.resize(op, (32, 28), interpolation=cv2.INTER_CUBIC)
        #show((res)

        maps['res']['CRG'].append(res)

    # computing I- feature Map
    maps['org']['I'] = []
    maps['res']['I'] = []
    for i in range(1, max_level):
        maps['org']['I'].append(img_L[i])
        res = cv2.resize(img_L[i], (32, 28), interpolation=cv2.INTER_CUBIC)
        maps['res']['I'].append(res)

    # computing Orientation Maps

    thetas = params['gaborangles']
    for th in thetas:
        maps['org'][th] = []
        maps['res'][th] = []

        for i in range(1, max_level):
            img = img_L[i]

            kernelPair = getGaborFiters([th])
            kernel_0 = kernelPair[th]['0']
            kernel_90 = kernelPair[th]['90']

            o1 = cv2.filter2D(img, -1, kernel_0, borderType=cv2.BORDER_REPLICATE)
            o2 = cv2.filter2D(img, -1, kernel_90, borderType=cv2.BORDER_REPLICATE)
            o = np.add(abs(o1), abs(o2))

            maps['org'][th].append(o)
            #show((o)

            r = cv2.resize(o, (32, 28), interpolation=cv2.INTER_CUBIC)

            maps['res'][th].append(r)
            #show((r)

    return maps


### step 2 : computing activation maps
def getActivationMap(params, featMaps):
    featureMaps = extractAllFeatureMaps(featMaps)
    activationMaps=[]
    for map in featureMaps:
        # map = np.array(scipy.io.loadmat("./AA.mat")['A'])
        salmap = computeGraphSaliencyForAFeatMap(params, map)
        activationMaps.append(salmap)
        print "Processed activation map."
    return  activationMaps


### step 3 : normalize the activation maps using graph based normalising
def normaliseActMaps(params, actMaps):
    normActMaps = []
    for actmap in actMaps:
        normsalmap = normaliseUsingGraphBasedSaliency(params, actmap)
        normActMaps.append(normsalmap)
        print "normalised map"
    return normActMaps

### step 4 : combine normalised activation maps for each feature channel
def combineNormActMaps(maps):
    cmbMaps = {}
    [cCnt, iCnt, oCnt] = [1, 1, 1]
    cmbMaps['c'] = maps[0]
    cmbMaps['i'] = maps[6]
    cmbMaps['o'] = maps[8]

    for i in range(1,6):
        np.add(cmbMaps['c'], maps[i])
        cCnt=cCnt+1
    for i in range(7,8):
        np.add(cmbMaps['i'], maps[i])
        iCnt=iCnt+1
    for i in range(9,21):
        np.add(cmbMaps['o'], maps[i])
        oCnt = oCnt+1

    np.divide(cmbMaps['c'], cCnt)
    np.divide(cmbMaps['i'], iCnt)
    np.divide(cmbMaps['o'], oCnt)

    mastermap = np.add(np.add(cmbMaps['c'], cmbMaps['i']), cmbMaps['o'])
    return mastermap

### step 5 : postprocessing
def postprocess(mastermap, img):
    gray = cv2.normalize(mastermap, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    blurred = cv2.GaussianBlur(gray,(3,3), 2)
    gray2 = cv2.normalize(blurred, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    mastermap_res = cv2.resize(gray2, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
    return mastermap_res

if __name__ == "__main__":
    for i in range(1, 6):
        imname = str(i)+".jpg"
        img = cv2.imread(imname)
        img = img / 255.0
        params = setupParams()
        featMaps = getFeatureMaps(img, params)
        actMaps = getActivationMap(params, featMaps)
        normActMaps = normaliseActMaps(params, actMaps)
        mastermap = combineNormActMaps(normActMaps)
        finalres = postprocess(mastermap, img)

        fig = plt.figure()
        fig.add_subplot(1,2,1)
        plt.imshow(img, cmap='gray')
        fig.add_subplot(1,2,2)
        plt.imshow(finalres, cmap='gray')
        plt.show()
