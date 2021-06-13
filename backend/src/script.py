import tensorflow
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from keras.engine.network import Network
from keras.layers import *
from keras import backend
from keras.models import Model
from keras.preprocessing import image
import keras.backend as K
import numpy as np
import os
from PIL import Image

### Constants ###
DATA_DIR = "../data"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

IMG_SHAPE = (64, 64)

def main(folder):


    def load_dataset_small():


        X_train = []
        
        # Create training set.
        img_i = image.load_img(os.path.join("./inputs/"+folder, "input1.jpeg"))
        img_i = img_i.resize((64, 64))
        x = image.img_to_array(img_i)
        X_train.append(x)
        img_i = image.load_img(os.path.join("./inputs/"+folder, "input2.jpeg"))
        img_i = img_i.resize((64, 64))
        x = image.img_to_array(img_i)
        X_train.append(x)
        img_i = image.load_img(os.path.join("./inputs/"+folder, "input3.jpeg"))
        img_i = img_i.resize((64, 64))
        x = image.img_to_array(img_i)
        X_train.append(x)
        img_i = image.load_img(os.path.join("./inputs/"+folder, "container.jpeg"))
        img_i = img_i.resize((64, 64))
        x = image.img_to_array(img_i)
        X_train.append(x)  
        return np.array(X_train)

    # Load dataset.
    X_train_orig = load_dataset_small()

    # Normalize image vectors.
    X_train = X_train_orig/255.


    # S1: secret image1
    input_S1 = X_train[0:1]
    # S2: secret image2
    input_S2 = X_train[1:2]
    # S3: secret image3
    input_S3 = X_train[2:3]

    # C: cover image
    input_C = X_train[3:4]


    """# **Model**

    The model is composed of three parts: The Preparation Network, Hiding Network (Encoder) and the Reveal Network. Its goal is to be able to encode information about the secret image S into the cover image C, generating C' that closely resembles C, while still being able to decode information from C' to generate the decoded secret image S', which should resemble S as closely as possible.

    The Preparation Network has the responsibility of preparing data from the secret image to be concatenated with the cover image and fed to the Hiding Network. The Hiding Network than transforms that input into the encoded cover image C'. Finally, the Reveal Network decodes the secret image S' from C'. For stability, we add noise before the Reveal Network, as suggested by the paper. Although the author of the paper didn't originally specify the architecture of the three networks, we discovered aggregated layers showed good results. For both the Hiding and Reveal networks, we use 5 layers of 65 filters (50 3x3 filters, 10 4x4 filters and 5 5x5 filters). For the preparation network, we use only 2 layers with the same structure.

    Note that the loss function for the Reveal Network is different from the loss function for the Preparation and Hiding Networks. In order to correctly implement the updates for the weights in the networks, we create stacked Keras models, one for the Preparation and Hiding Network (which share the same loss function) and one for the Reveal Network. To make sure weights are updated only once, we freeze the weights on the layers of the Reveal Network before adding it to the full model.

    # Define Loss
    Mean Square Loss has been used. Loss for reveal network and the full network are different.
    """

    # Variable used to weight the losses of the secret and cover images (See paper for more details)
    beta = 1.0
        
    # Loss for reveal network
    def rev_loss(s_true, s_pred):
        # Loss for reveal network is: beta * |S-S'|
        return beta * K.sum(K.square(s_true - s_pred))

    # Loss for the full model, used for preparation and hidding networks
    def full_loss(y_true, y_pred):
        # Loss for the full model is: |C-C'| + beta * |S-S'|
        s1_true, s2_true, s3_true, c_true = y_true[...,0:3], y_true[...,3:6], y_true[...,6:9], y_true[...,9:12]
        s1_pred, s2_pred, s3_pred, c_pred = y_pred[...,0:3], y_pred[...,3:6], y_pred[...,6:9], y_pred[...,9:12]

        s1_loss = beta * K.sum(K.square(s1_true - s1_pred))
        s2_loss = beta * K.sum(K.square(s2_true - s2_pred))
        s3_loss = beta * K.sum(K.square(s3_true - s3_pred))
        c_loss = K.sum(K.square(c_true - c_pred))
        
        return s1_loss + c_loss + s2_loss + s3_loss

    """# Define Encoder
    It has Prep and Hiding Network.
    """

    # Returns the encoder as a Keras model, composed by Preparation and Hiding Networks.
    def make_encoder(input_size):
        input_S1 = Input(shape=(input_size))
        input_S2 = Input(shape=(input_size))
        input_S3 = Input(shape=(input_size))
        input_C= Input(shape=(input_size))

        # Preparation Network for Secret Image 1
        x3 = Conv2D(50, (3, 3), strides = (1, 1), padding='same', activation='relu', name='conv_prep0_3x3_1')(input_S1)
        x4 = Conv2D(10, (4, 4), strides = (1, 1), padding='same', activation='relu', name='conv_prep0_4x4_1')(input_S1)
        x5 = Conv2D(5, (5, 5), strides = (1, 1), padding='same', activation='relu', name='conv_prep0_5x5_1')(input_S1)
        x = concatenate([x3, x4, x5])
        
        x3 = Conv2D(50, (3, 3), strides = (1, 1), padding='same', activation='relu', name='conv_prep1_3x3_1')(x)
        x4 = Conv2D(10, (4, 4), strides = (1, 1), padding='same', activation='relu', name='conv_prep1_4x4_1')(x)
        x5 = Conv2D(5, (5, 5), strides = (1, 1), padding='same', activation='relu', name='conv_prep1_5x5_1')(x)
        x1 = concatenate([x3, x4, x5])

        # Preparation Network for Secret Image 2
        x3 = Conv2D(50, (3, 3), strides = (1, 1), padding='same', activation='relu', name='conv_prep0_3x3_2')(input_S2)
        x4 = Conv2D(10, (4, 4), strides = (1, 1), padding='same', activation='relu', name='conv_prep0_4x4_2')(input_S2)
        x5 = Conv2D(5, (5, 5), strides = (1, 1), padding='same', activation='relu', name='conv_prep0_5x5_2')(input_S2)
        x = concatenate([x3, x4, x5])
        
        x3 = Conv2D(50, (3, 3), strides = (1, 1), padding='same', activation='relu', name='conv_prep1_3x3_2')(x)
        x4 = Conv2D(10, (4, 4), strides = (1, 1), padding='same', activation='relu', name='conv_prep1_4x4_2')(x)
        x5 = Conv2D(5, (5, 5), strides = (1, 1), padding='same', activation='relu', name='conv_prep1_5x5_2')(x)
        x2 = concatenate([x3, x4, x5])

        # Preparation Network for Secret Image 3
        x3 = Conv2D(50, (3, 3), strides = (1, 1), padding='same', activation='relu', name='conv_prep0_3x3_3')(input_S3)
        x4 = Conv2D(10, (4, 4), strides = (1, 1), padding='same', activation='relu', name='conv_prep0_4x4_3')(input_S3)
        x5 = Conv2D(5, (5, 5), strides = (1, 1), padding='same', activation='relu', name='conv_prep0_5x5_3')(input_S3)
        x = concatenate([x3, x4, x5])
        
        x3 = Conv2D(50, (3, 3), strides = (1, 1), padding='same', activation='relu', name='conv_prep1_3x3_3')(x)
        x4 = Conv2D(10, (4, 4), strides = (1, 1), padding='same', activation='relu', name='conv_prep1_4x4_3')(x)
        x5 = Conv2D(5, (5, 5), strides = (1, 1), padding='same', activation='relu', name='conv_prep1_5x5_3')(x)
        x3_1 = concatenate([x3, x4, x5])
        
        # Prep Network outputs concatenated to the encoded cover
        x = concatenate([input_C, x1, x2, x3_1])
        
        # Hiding network 
        x3 = Conv2D(50, (3, 3), strides = (1, 1), padding='same', activation='relu', name='conv_hid0_3x3')(x)
        x4 = Conv2D(10, (4, 4), strides = (1, 1), padding='same', activation='relu', name='conv_hid0_4x4')(x)
        x5 = Conv2D(5, (5, 5), strides = (1, 1), padding='same', activation='relu', name='conv_hid0_5x5')(x)
        x = concatenate([x3, x4, x5])
        
        x3 = Conv2D(50, (3, 3), strides = (1, 1), padding='same', activation='relu', name='conv_hid1_3x3')(x)
        x4 = Conv2D(10, (4, 4), strides = (1, 1), padding='same', activation='relu', name='conv_hid1_4x4')(x)
        x5 = Conv2D(5, (5, 5), strides = (1, 1), padding='same', activation='relu', name='conv_hid1_5x5')(x)
        x = concatenate([x3, x4, x5])
        
        x3 = Conv2D(50, (3, 3), strides = (1, 1), padding='same', activation='relu', name='conv_hid2_3x3')(x)
        x4 = Conv2D(10, (4, 4), strides = (1, 1), padding='same', activation='relu', name='conv_hid2_4x4')(x)
        x5 = Conv2D(5, (5, 5), strides = (1, 1), padding='same', activation='relu', name='conv_hid2_5x5')(x)
        x = concatenate([x3, x4, x5])
        
        x3 = Conv2D(50, (3, 3), strides = (1, 1), padding='same', activation='relu', name='conv_hid3_3x3')(x)
        x4 = Conv2D(10, (4, 4), strides = (1, 1), padding='same', activation='relu', name='conv_hid3_4x4')(x)
        x5 = Conv2D(5, (5, 5), strides = (1, 1), padding='same', activation='relu', name='conv_hid3_5x5')(x)
        x = concatenate([x3, x4, x5])
        
        x3 = Conv2D(50, (3, 3), strides = (1, 1), padding='same', activation='relu', name='conv_hid4_3x3')(x)
        x4 = Conv2D(10, (4, 4), strides = (1, 1), padding='same', activation='relu', name='conv_hid4_4x4')(x)
        x5 = Conv2D(5, (5, 5), strides = (1, 1), padding='same', activation='relu', name='conv_hid5_5x5')(x)
        x = concatenate([x3, x4, x5])
        
        output_Cprime = Conv2D(3, (3, 3), strides = (1, 1), padding='same', activation='relu', name='output_C')(x)
        
        return Model(inputs=[input_S1, input_S2, input_S3, input_C],
                    outputs=output_Cprime,
                    name = 'Encoder')

    """# Define all decoders.
    Currently 3 decoders for 3 secret image retrieval.
    """

    # Returns the decoder as a Keras model, composed by the Reveal Network
    def make_decoder1(input_size, fixed=False):
        
        # Reveal network
        reveal_input = Input(shape=(input_size))
        
        # Adding Gaussian noise with 0.01 standard deviation.
        input_with_noise = GaussianNoise(0.01, name='output_C_noise1')(reveal_input)
        
        x3 = Conv2D(50, (3, 3), strides = (1, 1), padding='same', activation='relu', name='conv_rev0_3x3_1')(input_with_noise)
        x4 = Conv2D(10, (4, 4), strides = (1, 1), padding='same', activation='relu', name='conv_rev0_4x4_1')(input_with_noise)
        x5 = Conv2D(5, (5, 5), strides = (1, 1), padding='same', activation='relu', name='conv_rev0_5x5_1')(input_with_noise)
        x = concatenate([x3, x4, x5])
        
        x3 = Conv2D(50, (3, 3), strides = (1, 1), padding='same', activation='relu', name='conv_rev1_3x3_1')(x)
        x4 = Conv2D(10, (4, 4), strides = (1, 1), padding='same', activation='relu', name='conv_rev1_4x4_1')(x)
        x5 = Conv2D(5, (5, 5), strides = (1, 1), padding='same', activation='relu', name='conv_rev1_5x5_1')(x)
        x = concatenate([x3, x4, x5])
        
        x3 = Conv2D(50, (3, 3), strides = (1, 1), padding='same', activation='relu', name='conv_rev2_3x3_1')(x)
        x4 = Conv2D(10, (4, 4), strides = (1, 1), padding='same', activation='relu', name='conv_rev2_4x4_1')(x)
        x5 = Conv2D(5, (5, 5), strides = (1, 1), padding='same', activation='relu', name='conv_rev2_5x5_1')(x)
        x = concatenate([x3, x4, x5])
        
        x3 = Conv2D(50, (3, 3), strides = (1, 1), padding='same', activation='relu', name='conv_rev3_3x3_1')(x)
        x4 = Conv2D(10, (4, 4), strides = (1, 1), padding='same', activation='relu', name='conv_rev3_4x4_1')(x)
        x5 = Conv2D(5, (5, 5), strides = (1, 1), padding='same', activation='relu', name='conv_rev3_5x5_1')(x)
        x = concatenate([x3, x4, x5])
        
        x3 = Conv2D(50, (3, 3), strides = (1, 1), padding='same', activation='relu', name='conv_rev4_3x3_1')(x)
        x4 = Conv2D(10, (4, 4), strides = (1, 1), padding='same', activation='relu', name='conv_rev4_4x4_1')(x)
        x5 = Conv2D(5, (5, 5), strides = (1, 1), padding='same', activation='relu', name='conv_rev5_5x5_1')(x)
        x = concatenate([x3, x4, x5])
        
        output_S1prime = Conv2D(3, (3, 3), strides = (1, 1), padding='same', activation='relu', name='output_S1')(x)
        
        if not fixed:
            return Model(inputs=reveal_input,
                        outputs=output_S1prime)
        else:
            return Network(inputs=reveal_input,
                            outputs=output_S1prime)
            
    # Returns the decoder as a Keras model, composed by the Reveal Network
    def make_decoder2(input_size, fixed=False):
        
        # Reveal network
        reveal_input = Input(shape=(input_size))
        
        # Adding Gaussian noise with 0.01 standard deviation.
        input_with_noise = GaussianNoise(0.01, name='output_C_noise2')(reveal_input)
        
        x3 = Conv2D(50, (3, 3), strides = (1, 1), padding='same', activation='relu', name='conv_rev0_3x3_2')(input_with_noise)
        x4 = Conv2D(10, (4, 4), strides = (1, 1), padding='same', activation='relu', name='conv_rev0_4x4_2')(input_with_noise)
        x5 = Conv2D(5, (5, 5), strides = (1, 1), padding='same', activation='relu', name='conv_rev0_5x5_2')(input_with_noise)
        x = concatenate([x3, x4, x5])
        
        x3 = Conv2D(50, (3, 3), strides = (1, 1), padding='same', activation='relu', name='conv_rev1_3x3_2')(x)
        x4 = Conv2D(10, (4, 4), strides = (1, 1), padding='same', activation='relu', name='conv_rev1_4x4_2')(x)
        x5 = Conv2D(5, (5, 5), strides = (1, 1), padding='same', activation='relu', name='conv_rev1_5x5_2')(x)
        x = concatenate([x3, x4, x5])
        
        x3 = Conv2D(50, (3, 3), strides = (1, 1), padding='same', activation='relu', name='conv_rev2_3x3_2')(x)
        x4 = Conv2D(10, (4, 4), strides = (1, 1), padding='same', activation='relu', name='conv_rev2_4x4_2')(x)
        x5 = Conv2D(5, (5, 5), strides = (1, 1), padding='same', activation='relu', name='conv_rev2_5x5_2')(x)
        x = concatenate([x3, x4, x5])
        
        x3 = Conv2D(50, (3, 3), strides = (1, 1), padding='same', activation='relu', name='conv_rev3_3x3_2')(x)
        x4 = Conv2D(10, (4, 4), strides = (1, 1), padding='same', activation='relu', name='conv_rev3_4x4_2')(x)
        x5 = Conv2D(5, (5, 5), strides = (1, 1), padding='same', activation='relu', name='conv_rev3_5x5_2')(x)
        x = concatenate([x3, x4, x5])
        
        x3 = Conv2D(50, (3, 3), strides = (1, 1), padding='same', activation='relu', name='conv_rev4_3x3_2')(x)
        x4 = Conv2D(10, (4, 4), strides = (1, 1), padding='same', activation='relu', name='conv_rev4_4x4_2')(x)
        x5 = Conv2D(5, (5, 5), strides = (1, 1), padding='same', activation='relu', name='conv_rev5_5x5_2')(x)
        x = concatenate([x3, x4, x5])
        
        output_S2prime = Conv2D(3, (3, 3), strides = (1, 1), padding='same', activation='relu', name='output_S2')(x)
        
        if not fixed:
            return Model(inputs=reveal_input,
                        outputs=output_S2prime)
        else:
            return Network(inputs=reveal_input,
                            outputs=output_S2prime)

    # Returns the decoder as a Keras model, composed by the Reveal Network
    def make_decoder3(input_size, fixed=False):
        
        # Reveal network
        reveal_input = Input(shape=(input_size))
        
        # Adding Gaussian noise with 0.01 standard deviation.
        input_with_noise = GaussianNoise(0.01, name='output_C_noise2')(reveal_input)
        
        x3 = Conv2D(50, (3, 3), strides = (1, 1), padding='same', activation='relu', name='conv_rev0_3x3')(input_with_noise)
        x4 = Conv2D(10, (4, 4), strides = (1, 1), padding='same', activation='relu', name='conv_rev0_4x4')(input_with_noise)
        x5 = Conv2D(5, (5, 5), strides = (1, 1), padding='same', activation='relu', name='conv_rev0_5x5')(input_with_noise)
        x = concatenate([x3, x4, x5])
        
        x3 = Conv2D(50, (3, 3), strides = (1, 1), padding='same', activation='relu', name='conv_rev1_3x3')(x)
        x4 = Conv2D(10, (4, 4), strides = (1, 1), padding='same', activation='relu', name='conv_rev1_4x4')(x)
        x5 = Conv2D(5, (5, 5), strides = (1, 1), padding='same', activation='relu', name='conv_rev1_5x5')(x)
        x = concatenate([x3, x4, x5])
        
        x3 = Conv2D(50, (3, 3), strides = (1, 1), padding='same', activation='relu', name='conv_rev2_3x3')(x)
        x4 = Conv2D(10, (4, 4), strides = (1, 1), padding='same', activation='relu', name='conv_rev2_4x4')(x)
        x5 = Conv2D(5, (5, 5), strides = (1, 1), padding='same', activation='relu', name='conv_rev2_5x5')(x)
        x = concatenate([x3, x4, x5])
        
        x3 = Conv2D(50, (3, 3), strides = (1, 1), padding='same', activation='relu', name='conv_rev3_3x3')(x)
        x4 = Conv2D(10, (4, 4), strides = (1, 1), padding='same', activation='relu', name='conv_rev3_4x4')(x)
        x5 = Conv2D(5, (5, 5), strides = (1, 1), padding='same', activation='relu', name='conv_rev3_5x5')(x)
        x = concatenate([x3, x4, x5])
        
        x3 = Conv2D(50, (3, 3), strides = (1, 1), padding='same', activation='relu', name='conv_rev4_3x3')(x)
        x4 = Conv2D(10, (4, 4), strides = (1, 1), padding='same', activation='relu', name='conv_rev4_4x4')(x)
        x5 = Conv2D(5, (5, 5), strides = (1, 1), padding='same', activation='relu', name='conv_rev5_5x5')(x)
        x = concatenate([x3, x4, x5])
        
        output_S3prime = Conv2D(3, (3, 3), strides = (1, 1), padding='same', activation='relu', name='output_S3')(x)
        
        if not fixed:
            return Model(inputs=reveal_input,
                        outputs=output_S3prime)
        else:
            """return Container(inputs=reveal_input,
                            outputs=output_S2prime,
                            name = 'DecoderFixed')"""
            return Network(inputs=reveal_input,
                            outputs=output_S3prime)

    """# Define Full Model.
    Assemble the encoders and decoders
    """

    # Full model.
    def make_model(input_size):
        input_S1 = Input(shape=(input_size))
        input_S2 = Input(shape=(input_size))
        input_S3 = Input(shape=(input_size))
        input_C= Input(shape=(input_size))
        
        encoder = make_encoder(input_size)
        
        decoder1 = make_decoder1(input_size)
        decoder1.compile(optimizer='adam', loss=rev_loss)
        decoder1.trainable = False

        decoder2 = make_decoder2(input_size)
        decoder2.compile(optimizer='adam', loss=rev_loss)
        decoder2.trainable = False

        decoder3 = make_decoder3(input_size)
        decoder3.compile(optimizer='adam', loss=rev_loss)
        decoder3.trainable = False
        
        output_Cprime = encoder([input_S1, input_S2, input_S3, input_C])
        output_S1prime = decoder1(output_Cprime)
        output_S2prime = decoder2(output_Cprime)
        output_S3prime = decoder3(output_Cprime)

        autoencoder1 = Model(inputs=[input_S1, input_S2, input_S3, input_C],
                            outputs=concatenate([output_S1prime, output_S2prime, output_S3prime, output_Cprime]))
        autoencoder1.compile(optimizer='adam', loss=full_loss)
        
        return encoder, decoder1, decoder2, decoder3, autoencoder1


    def run():

        encoder_model, reveal_model1, reveal_model2, reveal_model3, autoencoder_model = make_model(input_S1.shape[1:])



        """# Decode the model's output"""

        autoencoder_model.load_weights('../models/model_A21_999')



        # Retrieve decoded predictions.
        decoded = autoencoder_model.predict([input_S1, input_S2, input_S3, input_C])
        decoded_S1, decoded_S2, decoded_S3, decoded_C = decoded[...,0:3], decoded[...,3:6], decoded[...,6:9], decoded[...,9:12]


        """# Final Recovered Images. """

        os.mkdir("./outputs/"+folder)
        data=image.array_to_img(input_C[0])
        data.save("./outputs/"+folder+"/"+"container.jpeg")

        data=image.array_to_img(input_S1[0])
        data.save("./outputs/"+folder+"/"+"secret1.jpeg")

        data=image.array_to_img(input_S2[0])
        data.save("./outputs/"+folder+"/"+"secret2.jpeg")

        data=image.array_to_img(input_S3[0])
        data.save("./outputs/"+folder+"/"+"secret3.jpeg")

        data=image.array_to_img(decoded_C[0])
        data.save("./outputs/"+folder+"/"+"encoded.jpeg")

        data=image.array_to_img(decoded_S1[0])
        data.save("./outputs/"+folder+"/"+"decoded1.jpeg")

        data=image.array_to_img(decoded_S2[0])
        data.save("./outputs/"+folder+"/"+"decoded2.jpeg")

        data=image.array_to_img(decoded_S3[0])
        data.save("./outputs/"+folder+"/"+"decoded3.jpeg")
    run()

