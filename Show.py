import matplotlib.pyplot as plt
import numpy as np
import glob
import struct
import Config
import Convert

curr_pos = 0


def show(config):
    images1 = Convert.loadImages(config.trainImages)
    images2 = Convert.loadImages(config.trainLabels)
    images3 = np.load(config.likelihood)[:, :, :, 1]
    images4 = Convert.loadMHA(glob.glob(config.getResultFile("final_*.mha")))

    def press(event):
        global curr_pos
        # use keyboard to change images
        if event.key == "right":
            curr_pos = (curr_pos + 1) % images3.shape[0]
        elif event.key == "left":
            curr_pos = (curr_pos - 1) % images3.shape[0]
        show()

    def show():
        fig.clear()

        plt.subplot(2, 2, 1)
        plt.title("Origin")
        plt.imshow(images1[curr_pos], cmap="Greys_r")
        plt.axis('off')

        plt.subplot(2, 2, 2)
        plt.title("Labels")
        plt.imshow(images2[curr_pos], cmap="Greys_r")
        plt.axis('off')

        plt.subplot(2, 2, 3)
        plt.title("CNN Result")
        plt.imshow(images3[curr_pos], cmap="Greys_r")
        plt.axis('off')

        plt.subplot(2, 2, 4)
        plt.title("Segment Result")
        plt.imshow(images4[curr_pos])
        plt.axis('off')

        fig.suptitle("Image " + str(curr_pos + 1))
        fig.canvas.draw()

    fig = plt.figure(figsize=(12, 8))
    fig.canvas.mpl_connect('key_press_event', press)
    show()

    plt.show()


def loadMHA(config, pattern):
    mhas = glob.glob(config.getResultFile(pattern))
    mhas.sort()

    arrs = []
    for f in mhas:
        with open(f, mode="rb") as mha:
            arr = []
            data = mha.read()
            index = data.index("ElementDataFile = LOCAL\n")
            data = data[index + len("ElementDataFile = LOCAL\n"):]
            for i in range(0, len(data), 4):
                arr.append(struct.unpack("f", data[i:i + 4])[0])

            size = int(len(arr) ** 0.5)
            arr = np.array(arr).reshape(1, size, size)
            arrs.append(arr)

    arrs = np.concatenate(arrs)
    return arrs


if __name__ == "__main__":
    show(Config.load(False))
