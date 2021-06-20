import tensorflow as tf

class Model () :
    def __init__ (self, model_name='resnet18') :
        self.model_name = model_name
        self.base_model = tf.keras.applications.ResNet50 (input_shape=(227, 227, 3), include_top=False, weights='imagenet')
        self.base_model.trainable = False

    def build_model (self) :
        inputs = tf.keras.layers.Input (shape=(227, 227, 3)) 

        x = self.base_model (inputs, training=False)
        x = tf.keras.layers.Flatten () (x)
        x = tf.keras.layers.Dropout (0.4) (x)
        x = tf.keras.layers.Dense (7) (x)
        x = tf.keras.layers.Softmax () (x)

        return tf.keras.Model (inputs=inputs, outputs=x)
