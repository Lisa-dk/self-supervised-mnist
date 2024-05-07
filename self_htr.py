import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix

class Self_HTR:
    def __init__(self, htr_model, generator, optimizer, loss_fn, layer, latent_dim, metric):
        self.htr_model = htr_model
        self.generator = generator
        self.optimizer = optimizer
        self.loss_fn = self.get_loss(loss_fn, layer)
        self.latent_dim = latent_dim
        self.metric = metric
    
    def get_vgg_model(self, n_layer, summary=False):
        model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3), pooling=None)
        # model.summary()
        # for num, layer in enumerate(model.layers):
        # 	print(num, layer)
        input_data = tf.keras.Input(shape=(28, 28, 3))
        resized_input = tf.keras.layers.Resizing(32, 32)(input_data)
        desired_layer_output = tf.keras.models.Sequential(model.layers[:n_layer])(resized_input)
        vgg_model = tf.keras.Model(inputs=input_data, outputs=desired_layer_output)
        if summary:
            vgg_model.summary()
        return vgg_model


    def get_loss(self, loss_name, layer=8):
        if loss_name == 'vgg':
            self.vgg_model = self.get_vgg_model(layer)
            return self.vgg_loss
        elif loss_name == 'bce':
            return tf.keras.losses.BinaryCrossentropy()
        elif loss_name == 'mse':
            return tf.keras.losses.MeanSquaredError()
    
    def vgg_loss(self, images_in, images_gen):
        images_in = tf.concat([images_in] * 3, axis=-1)
        images_gen = tf.concat([images_gen] * 3, axis=-1)
        return tf.reduce_mean((self.vgg_model(images_in, training=False) - self.vgg_model(images_gen, training=False))**2)

    def validate(self, data):
        val_losses = []
        accuracies = []
        for image_batch in data:
            x_val, y_val = image_batch
            # print(x_val.shape, y_val.shape, tf.shape(x_val))
            y_preds = self.htr_model.predict(x_val, verbose=0)
            y_preds_argmax = np.argmax(y_preds, axis=1)
            accuracy = accuracy_score(y_true=np.argmax(y_val, axis=1), y_pred=y_preds_argmax)
            accuracies.append(accuracy)

            random_latent_vectors = tf.random.normal(
                shape=(tf.shape(x_val)[0], self.latent_dim), seed=1337
            )
            # print(tf.shape(random_latent_vectors))

            random_vector_labels = tf.concat(
                    [random_latent_vectors, y_preds], axis=1
                )
            
            synth_imgs = self.generator(random_vector_labels, training=False)

            # val_loss = tf.reduce_mean((vgg_model(images_in, training=False) - vgg_model(images_gen, training=False))**2)
            val_loss = self.loss_fn(x_val, synth_imgs)
            # val_loss = tf.keras.losses.BinaryCrossentropy()(x_val, synth_imgs)
            val_losses.append(val_loss)
        np.set_printoptions(suppress=True, precision=4)
        # print(np.round(y_preds[:16], 4))
        return np.mean(val_losses), np.mean(accuracies), synth_imgs[:16], x_val[:16], y_val[:16], y_preds[:16]
    
    # @tf.function
    def train_step(self, data):
        images, one_hot_labels = data

        random_latent_vectors = tf.random.normal(
            shape=(tf.shape(images)[0], self.latent_dim), seed=1337
        )
        # random_label_noise = tf.random.normal(shape=(tf.shape(images)[0], 10), mean=0, stddev=0.2)

        with tf.GradientTape() as tape:
            pred_labels = self.htr_model(images, training=True)

            random_vector_labels = tf.concat(
                [random_latent_vectors, pred_labels], axis=-1
            )

            synth_imgs = self.generator(random_vector_labels, training=False)
            # loss_mse = tf.reduce_mean((gen_imgs - images) ** 2)
            # loss_ssim = 1 - tf.reduce_mean(tf.image.ssim(images, gen_imgs, 1, filter_size=7))
            # loss = 10*loss_mse + loss_ssim #- 0.1*tf.reduce_mean(tf.reduce_max(pred_labels, axis=1))

            # loss = tf.reduce_mean((vgg_model(images_in, training=False) - vgg_model(images_gen, training=False))**2)
            loss = self.loss_fn(images, synth_imgs)
            # loss = tf.keras.losses.BinaryCrossentropy()(images, gen_imgs)
            
        grads = tape.gradient(loss, self.htr_model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.htr_model.trainable_weights))

        self.metric.update_state(
                y_true=tf.argmax(one_hot_labels, axis=1),
                y_pred=tf.argmax(pred_labels, axis=1))

        accuracy = self.metric.result()
        
        return loss, accuracy