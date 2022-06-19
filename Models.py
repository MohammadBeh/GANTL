from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras import backend as K
from Layers import *
from utils import *

def Binary(y_true,y_pred):
    #return tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    return K.mean(K.binary_crossentropy(y_true, y_pred, from_logits=True), axis=-1)


def Pre_Discriminator(source_shape, target_shape):

	# source image input
    in_src_image = Input(shape= source_shape)
	# target image input
    in_target_image = Input(shape= target_shape)
	# concatenate images channel-wise
    merged = Concat(in_src_image, in_target_image)
	# C64
    d = Conv_dis(merged, 64, (4,4),strides=(1, 1), padding='same')
	# C128
    d = Conv_dis(d, 128, (4,4), strides=(1,2), padding='same')
	# C256
    d = Conv_dis(d, 256, (4,4), strides=(2,2), padding='same')

	# C512
    d = Conv_dis(d, 512, (4,4), strides=(2,2), padding='same')
	# patch output
    patch_out = Dis_classifier(d)
	# define model
    model = Model([in_src_image, in_target_image], patch_out)
	# compile model
    opt = Adam(lr=0.0004, beta_1=0.5)
    #bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    model.compile(loss=Binary, optimizer=opt, loss_weights=[0.5])
    return model




# define the discriminator model
def Discriminator(source_shape, target_shape):
	# source image input
    in_src_image = Input(shape=source_shape)
    in_src = Conv_dis(in_src_image, 5, kernel_size = 2, strides=(1,1),padding='valid', batch = False, active = False)
    
    in_src = Dis_transpose(in_src, 5, kernel_size = 2, strides = (int((target_shape[0] - 1)/(source_shape[0] - 1)), int((target_shape[1] - 1)/(source_shape[1] - 1) ))  )
    in_src = Dis_transpose(in_src, 5, kernel_size = 2, strides = (1,1)  )
    
	# target image input
    in_target_image = Input(shape=target_shape)
    #x = Cropping2D(cropping=((80, 80), (160, 160)))(in_target_image)
	# concatenate images channel-wise
    merged = Concatenate()([in_src, in_target_image])
	# C64
    d = Conv_dis(merged, 64, (4,4),strides=(1, 1), padding='same')
	# C128
    d = Conv_dis(d, 128, (4,4), strides=(1,2), padding='same')
	# C256
    d = Conv_dis(d, 256, (4,4), strides=(2,2), padding='same')

	# C512
    d = Conv_dis(d, 512, (4,4), strides=(2,2), padding='same')
	# patch output
    patch_out = Dis_classifier(d)
    
	# define model
    model = Model([in_src_image, in_target_image], patch_out)
	# compile model
    opt = Adam(lr=0.0004, beta_1=0.5)
    #bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    model.compile(loss=Binary, optimizer=opt, loss_weights=[0.5])
    return model
 





def Pre_Generator(image_shape=(41,81,5)):
	# latent space
    lat_space = Input(shape = (100,))
	
	# image input
    in_image = Input(shape=image_shape)
	# encoder model
    x = Conv_gen(in_image, 32, 2, strides = (1,1),padding='valid')
     ### 20 x 40 x 16
    y = SE_Concat(64, x)
    y = Conv_gen(y, 64,kernel_size=3 , strides=2, padding='same')

     ### 10 x 20 x 32
    y1 = SE_Concat(128, y)
    y1 = Conv_gen(y1, 128,kernel_size=3 , strides=2, padding='same')
    
     ####  5 x 10 x 64 
    y2 = SE_Concat(256, y1)
    y2 = Conv_gen(y2, 128,kernel_size=3 , strides=2, padding='same')
    
    y2 = SE_Concat(256, y2)
    
    
    l1 = Gen_lat(lat_space, 512*5*10, (5,10,512))
 
    a = synthesis_block(l1, y2,512)
    a = upsampling(a, 256)
    
    a = synthesis_block(a, y1,128)
    a = upsampling(a, 128)
    
    a = synthesis_block(a, y,64)
    a = upsampling(a, 64)
    
    out_image = Gen_transpose(a, 32, kernel_size = 2, strides=(1,1),padding='valid')

	# define model
    model = Model([in_image,lat_space], out_image)
    return model





def Generator(args, pretrain_model, image_shape=(41,81,5)):
	# latent space
    lat_space = Input(shape = (100,))
	
	# image input
    in_image = Input(shape=image_shape)
	
    x = pretrain_model([in_image,lat_space], training=True)

    x = Gen_transpose(x, 128, kernel_size = 3, strides=(int(extract_res(args)[0]/int(image_shape[0] - 1)) ,int(extract_res(args)[1]/int(image_shape[1] - 1))),padding='valid', batch = False, last = False)
    
    x = Conv_gen(x, 128,kernel_size=3 , strides=1, padding='valid')
    
    x = Conv_gen(x, 64,kernel_size=3 , strides=1, padding='same')
    
    x = Conv_gen(x, 32,kernel_size=3 , strides=1, padding='same')
    
    out_image = Conv_gen(x, 1, kernel_size=3, strides= 1,padding='same',activation='sigmoid', batch = False)
        
	# define model
    model = Model([in_image,lat_space], out_image)
    return model





# define the combined generator and discriminator model, for updating the generator
def GAN(g_model, d_model, image_shape = (41,81,5)):
	# make weights in the discriminator not trainable
    d_model.trainable = False
	# define the source image
    lat_input = Input(shape=(100,))
    in_src = Input(shape=image_shape)
	# connect the source image to the generator input
    gen_out = g_model([in_src,lat_input])
	# connect the source input and generator output to the discriminator input
    dis_out = d_model([in_src, gen_out])
	# src image as input, generated image and classification output
    model = Model([in_src,lat_input], [dis_out, gen_out])
	# compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss=[Binary, 'bce'], optimizer=opt)
    return model

