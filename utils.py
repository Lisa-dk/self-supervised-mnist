import numpy as np
import matplotlib.pylab as plt
import os

def preproc_mnist(x_train, x_test):
    def proc_imgs(imgs):
        # preprocess [0,1]
        imgs = imgs.astype("float32") / 255
        # ensure shape (28, 28, 1)
        return np.expand_dims(imgs, axis=-1)
    return proc_imgs(x_train), proc_imgs(x_test)

def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 6))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 6, i+1)
        plt.imshow(predictions[i, :, :, 0], cmap='gray')
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()

def save_images(epoch, imgs, labels, losses, pred_labels, true_imgs, loss_dir, plot=False):
    labels_numbers = np.argmax(labels, axis=1)
    pred_labels_numbers = np.argmax(pred_labels, axis=1)

    fig = plt.figure(figsize=(16, 8))
    print(true_imgs.shape)
    plot_idx = 1
    for i in range(imgs.shape[0]):
        # print(losses[i])
        loss = f"{losses:.3f}"  # Format the SSIM value with three decimals
        fig.add_subplot(4, 8, plot_idx )
        plt.title(f"gt: {labels_numbers[i]}, pred: {pred_labels_numbers[i]}, loss: {loss}")
        plt.imshow(true_imgs[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')
        fig.add_subplot(4, 8, plot_idx + 1)
        plt.imshow(imgs[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')
 
        plot_idx += 2
    dir_path = './synth_imgs/' + loss_dir +'/'
    os.makedirs(dir_path, exist_ok=True)
    plt.savefig(dir_path + 'images_at_epoch_{:04d}.png'.format(epoch))
    if plot:
        plt.show()
    plt.close()

