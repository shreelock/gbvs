import cv2
import numpy as np


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
    cv2.imshow("i", img), cv2.waitKey()


def C_operation_BY(img_L, img_B, img_G, img_R):
    min_rg = np.minimum(img_R, img_G)
    b_min_rg = abs(np.subtract(img_B, min_rg))
    op = np.divide(b_min_rg, img_L, out=np.zeros_like(img_L), where=img_L != 0)
    return op


def C_operation_RG(img_L, img_B, img_G, img_R):
    r_g = abs(np.subtract(img_R, img_G))
    op = np.divide(r_g, img_L, out=np.zeros_like(img_L), where=img_L != 0)
    return op


### step 1 : computing feature maps
def getFeatureMaps(img, params):
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
    # computing BY
    for i in range(1, max_level):
        op = C_operation_BY(img_L[i], img_B[i], img_G[i], img_R[i])
        show(op)


if __name__ == "__main__":
    img = cv2.imread("1.jpg")
    img = img / 255.0
    # cv2.imshow("1",img), cv2.waitKey()
    params = setupParams()
    getFeatureMaps(img, params)
