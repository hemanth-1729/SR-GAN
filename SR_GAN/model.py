from tensorflow import keras
import tensorflow as tf
from tensorflow.python.keras.layers import Lambda


class FastSRGAN(object):
    """SRGAN for fast super resolution."""

    def __init__(self, args):
        """
        Initializes the Mobile SRGAN class.
        Args:
            args: CLI arguments that dictate how to build the model.
        Returns:
            None
        """
        self.hr_height = args.hr_size
        self.hr_width = args.hr_size
        self.lr_height = self.hr_height // 4  # Low resolution height
        self.lr_width = self.hr_width // 4  # Low resolution width
        self.lr_shape = (self.lr_height, self.lr_width, 3)
        self.hr_shape = (self.hr_height, self.hr_width, 3)
        self.iterations = 0

        # Number of inverted residual blocks in the mobilenet generator
        self.n_residual_blocks = 6

        # Define a learning rate decay schedule.
        self.gen_schedule = keras.optimizers.schedules.ExponentialDecay(
            args.lr,
            decay_steps=100000,
            decay_rate=0.1,
            staircase=True
        )

        self.disc_schedule = keras.optimizers.schedules.ExponentialDecay(
            args.lr * 5,  # TTUR - Two Time Scale Updates
            decay_steps=100000,
            decay_rate=0.1,
            staircase=True
        )

        self.gen_optimizer = keras.optimizers.Adam(learning_rate=self.gen_schedule)
        self.disc_optimizer = keras.optimizers.Adam(learning_rate=self.disc_schedule)

        # We use a pre-trained VGG19 model to extract image features from the high resolution
        # and the generated high resolution images and minimize the mse between them
        self.vgg = self.build_vgg()
        self.vgg.trainable = False

        # Calculate output shape of D (PatchGAN)
        patch = int(self.hr_height / 2 ** 4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 32  # Realtime Image Enhancement GAN Galteri et al.
        self.df = 32

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()

        # Build and compile the generator for pretraining.
        self.generator = self.build_generator()

    @tf.function
    def content_loss(self, hr, sr):
        sr = keras.applications.vgg19.preprocess_input(((sr + 1.0) * 255) / 2.0)
        hr = keras.applications.vgg19.preprocess_input(((hr + 1.0) * 255) / 2.0)
        sr_features = self.vgg(sr) / 12.75
        hr_features = self.vgg(hr) / 12.75
        return tf.keras.losses.MeanSquaredError()(hr_features, sr_features)

    def build_vgg(self):
        """
        Builds a pre-trained VGG19 model that outputs image features extracted at the
        third block of the model
        """
        # Get the vgg network. Extract features from Block 5, last convolution.
        vgg = keras.applications.VGG19(weights="imagenet", input_shape=self.hr_shape, include_top=False)
        vgg.trainable = False
        for layer in vgg.layers:
            layer.trainable = False

        # Create model and compile
        model = keras.models.Model(inputs=vgg.input, outputs=vgg.get_layer("block5_conv4").output)

        return model

    def build_generator(self):
        """Build the generator that will do the Super Resolution task.
        Based on the Mobilenet design. Idea from Galteri et al."""

        def d_block(layer_input, filters, strides=1, bn=True):
            """Discriminator layer block.
            Args:
                layer_input: Input feature map for the convolutional block.
                filters: Number of filters in the convolution.
                strides: The stride of the convolution.
                bn: Whether to use batch norm or not.
            """
            d = keras.layers.Conv2D(filters, kernel_size=3, strides=strides, padding='same')(layer_input)
            if bn:
                d = keras.layers.BatchNormalization(momentum=0.8)(d)
            d = keras.layers.LeakyReLU(alpha=0.2)(d)
                
            return d

        # Input img
        d0 = keras.layers.Input(shape=self.lr_shape)

        d1 = d_block(d0, self.gf, bn=False)
        d2 = d_block(d1, self.gf)
        d3 = d_block(d2, self.gf)
        d4 = d_block(d3, self.gf)
        d5 = d_block(d4, self.gf * 2)
        d6 = d_block(d5, self.gf * 2)
        d7 = d_block(d6, self.gf * 2)
        d8 = d_block(d7, self.gf * 2)
        
        
        def pixel_shuffle(scale):
            return lambda x: tf.nn.depth_to_space(x, scale)

        def deconv2d(layer_input):
            """Upsampling layer to increase height and width of the input.
            Uses PixelShuffle for upsampling.
            Args:
                layer_input: The input tensor to upsample.
            Returns:
                u: Upsampled input by a factor of 2.
            """
            #u = keras.layers.UpSampling2D(size=2, interpolation='bilinear')(layer_input)
            u = keras.layers.Conv2D(self.gf, kernel_size=3, strides=1, padding='same')(layer_input)
            u = Lambda(pixel_shuffle(scale=2))(u)
            u = keras.layers.PReLU(shared_axes=[1, 2])(u)
            return u

        # Upsampling
        u1 = deconv2d(d8)
        u2 = deconv2d(u1)

        # Generate high resolution output
        gen_hr = keras.layers.Conv2D(3, kernel_size=3, strides=1, padding='same', activation='tanh')(u2)

        return keras.models.Model(d0, gen_hr)

    def build_discriminator(self):
        """Builds a discriminator network based on the SRGAN design."""

        def d_block(layer_input, filters, strides=1, bn=True):
            """Discriminator layer block.
            Args:
                layer_input: Input feature map for the convolutional block.
                filters: Number of filters in the convolution.
                strides: The stride of the convolution.
                bn: Whether to use batch norm or not.
            """
            d = keras.layers.Conv2D(filters, kernel_size=3, strides=strides, padding='same')(layer_input)
            if bn:
                d = keras.layers.BatchNormalization(momentum=0.8)(d)
            d = keras.layers.LeakyReLU(alpha=0.2)(d)
                
            return d

        # Input img
        d0 = keras.layers.Input(shape=self.hr_shape)

        d1 = d_block(d0, self.df, bn=False)
        d2 = d_block(d1, self.df, strides=2)
        d3 = d_block(d2, self.df)
        d4 = d_block(d3, self.df, strides=2)
        d5 = d_block(d4, self.df * 2)
        d6 = d_block(d5, self.df * 2, strides=2)
        d7 = d_block(d6, self.df * 2)
        d8 = d_block(d7, self.df * 2, strides=2)

        validity = keras.layers.Conv2D(1, kernel_size=1, strides=1, activation='sigmoid', padding='same')(d8)

        return keras.models.Model(d0, validity)
