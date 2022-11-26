import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf


class Model(tf.keras.layers.Layer):
    def __init__(self, output_dim):
        self.output_dim = output_dim
        super(Model, self).__init__()

    def build(self, input_shape):
        self.weight = self.add_weight(
            name='weight',
            shape=(input_shape[1], self.output_dim),
            initializer='zeros',
            trainable=True
        )
        super(Model, self).build(input_shape)

    def call(self, inputs):
        
        output = tf.keras.activations.get('softmax')(
                tf.matmul(inputs, self.weight), axis=1
            )    # NOTE: this has _keras_logits
        
        delattr(output, '_keras_logits')

        # output = tf.nn.softmax(
        #         tf.matmul(inputs, self.weight), axis=1
        #     )      # NOTE: this does not have _keras_logits

        print(f'Has _keras_logits: ', hasattr(output, '_keras_logits'))
        return output


if __name__ == '__main__':
    model = Model(2)
    inputs = tf.constant([[0, 0, 1, 0, 0, 0, 0, 0]], dtype=tf.float32)
    y_true = tf.constant([[0, 1]], dtype=tf.float32)

    # `tf.keras.activations.get('softmax')` not affected by `from_logits` - always using logits?
        # if from_logits=True, grad=0.25
        # if from_logits=False, grad=0.25   (no printing `Not using logits`)

    # `tf.nn.softmax` affected by `from_logits`
        # if from_logits=True, grad=0.125
        # if from_logits=False, grad=0.5    (confirmed false in source code)
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
    with tf.GradientTape() as tape:
        y_pred = model(inputs)
        loss = loss_fn(y_true, y_pred)
        grads = tape.gradient(loss, model.trainable_weights)
        print(f'loss = {loss}')
        print(f'grads: {grads}')

    print(tf.keras.activations.get('softmax'))
    print(tf.nn.softmax)