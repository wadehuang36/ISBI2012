import matplotlib.pyplot as plt
import numpy as np

import Config
import Convert

curr_pos = 0


def show(config):
    images1 = Convert.loadImages(config.deployImages)
    images2 = np.load(config.likelihood)[:, :, :, 1]
    images3 = np.load(config.segment)

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
        plt.axis('off');

        plt.subplot(2, 2, 3)
        plt.title("Likelihood")
        plt.imshow(images2[curr_pos], cmap="Greys_r")
        plt.axis('off');

        plt.subplot(2, 2, 4)
        plt.title("Segment")
        plt.imshow(images3[curr_pos])
        plt.axis('off');

        fig.suptitle("Image " + str(curr_pos + 1))
        fig.canvas.draw()

    fig = plt.figure(figsize=(12, 8))
    fig.canvas.mpl_connect('key_press_event', press)
    show()

    plt.show()


if __name__ == "__main__":
    show(Config.load())
