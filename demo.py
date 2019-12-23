from saliency_models import gbvs, ittikochneibur
import cv2
import time
from matplotlib import pyplot as plt

if __name__ == '__main__':
    for i in range(1, 9):
        imname = "./images/{}.jpg".format(i)
        print("processing {}".format(imname))

        img = cv2.imread(imname)

        saliency_map_gbvs = gbvs.compute_saliency(img)
        saliency_map_ikn = ittikochneibur.compute_saliency(img)

        oname = "./outputs/{}_out{}.jpg".format(i, time.time())
        cv2.imwrite(oname, saliency_map_gbvs)

        fig = plt.figure(figsize=(10, 3))

        fig.add_subplot(1, 3, 1)
        plt.imshow(img, cmap='gray')
        plt.gca().set_title("Original Image")
        plt.axis('off')

        fig.add_subplot(1, 3, 2)
        plt.imshow(saliency_map_gbvs, cmap='gray')
        plt.gca().set_title("GBVS")
        plt.axis('off')

        fig.add_subplot(1, 3, 3)
        plt.imshow(saliency_map_ikn, cmap='gray')
        plt.gca().set_title("Itti Koch Neibur")
        plt.axis('off')

        plt.show()
