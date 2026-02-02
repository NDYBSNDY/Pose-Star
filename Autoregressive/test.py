import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.activations import relu
from tensorflow.keras.models import Sequential

import tensorflow_datasets as tfds

import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')
print(tf.__version__)

(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['test', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True)
fig = tfds.show_examples(ds_train, ds_info)

def binarize(image, label):
    image = tf.cast(image, tf.float32)
    image = tf.math.round(image/255.)
    return image, tf.cast(image, tf.int32)
    
ds_train = ds_train.map(binarize)
ds_train = ds_train.cache() # put dataset into memory
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(128)
ds_test = ds_test.map(binarize).batch(128).prefetch(64)

class MaskedConv2D(layers.Layer):
    def __init__(self, mask_type, kernel=5, filters=1):
        super(MaskedConv2D, self).__init__()
        self.kernel = kernel
        self.filters = filters
        self.mask_type = mask_type
    
    def build(self, input_shape):
        
        self.w = self.add_weight(shape=[
            self.kernel, 
            self.kernel, 
            input_shape[-1],
            self.filters],
            initializer='glorot_normal',
            trainable=True)
        
        self.b = self.add_weight(shape=(self.filters,),
            initializer='zeros',
            trainable=True)
        
        # Create Mask
        mask = np.ones(self.kernel ** 2, dtype=np.float32)
        center = len(mask) // 2
        mask[center+1:] = 0
        if self.mask_type == 'A':
            mask[center] = 0
        
        mask = mask.reshape((self.kernel, self.kernel, 1, 1))

        self.mask = tf.constant(mask, dtype='float32')
    
    def call(self, inputs):
        # mask the convolution
        masked_w = tf.math.multiply(self.w, self.mask)

        # preform conv2d using low level API
        output = tf.nn.conv2d(inputs, masked_w, 1, 'SAME') + self.b

        return tf.nn.relu(output)

class ResidualBlock(layers.Layer):
    def __init__(self, h=32):
        super(ResidualBlock, self).__init__()

        self.forward = Sequential([
            MaskedConv2D('B', kernel=1, filters=h),
            MaskedConv2D('B', kernel=3, filters=h),
            MaskedConv2D('B', kernel=1, filters=2*h),
        ])
    
    def call(self, inputs):
        x = self.forward(inputs)
        return x + inputs

def SimplePixelCnn(
    hidden_features=64,
    output_features=64,
    resblocks_num=7):
    
    inputs = layers.Input(shape=[28, 28, 1])
    x = inputs
    
    x = MaskedConv2D('A', kernel=7, filters=2*hidden_features)(x)

    for _ in range(resblocks_num):
        x = ResidualBlock(hidden_features)(x)
    
    x = layers.Conv2D(output_features, (1,1), padding='same', activation='relu')(x)
    x = layers.Conv2D(1, (1,1), padding='same', activation='sigmoid')(x)

    return tf.keras.Model(inputs=inputs, outputs=x, name='PixelCnn')

pixel_cnn = SimplePixelCnn()
pixel_cnn.summary()

pixel_cnn.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
    metrics=[tf.keras.losses.BinaryCrossentropy()]
)

grid_row = 5
grid_col = 5
batch = grid_row * grid_col
h = w = 28
images = np.ones((batch,h,w,1), dtype=np.float32)

for row in range(h):
    for col in range(w):
        prob = pixel_cnn.predict(images)[:, row,col,0]
        pixel_samples = tf.random.categorical(
            tf.math.log(np.stack([1-prob,prob],1)),1
        )
        #print(pixel_samples.shape)
        images[:,row,col,0] = tf.reshape(pixel_samples, [batch])

# Display
f, axarr = plt.subplots(grid_row, grid_col, figsize=(grid_col*1.1,grid_row))

i = 0
for row in range(grid_row):
    for col in range(grid_col):
        axarr[row,col].imshow(images[i,:,:,0], cmap='gray')
        axarr[row,col].axis('off')
        i += 1
f.tight_layout(0.1, h_pad=0.2, w_pad=0.1)        
plt.show()
