from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Conv2D,Add
from tensorflow.keras.layers import Conv2DTranspose,Activation
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation,MaxPool2D
from tensorflow.keras.layers import Concatenate,Dense,Multiply,Flatten
from tensorflow.keras.layers import Dropout,Reshape,GlobalMaxPooling2D,GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization


def Concat(inp1,inp2):
    return Concatenate()([inp1, inp2])

def Conv_dis(inp, filters, kernel_size, strides,padding, batch = True, active = True):
    conved = Conv2D(filters, kernel_size, strides=strides, padding=padding)(inp)
    if batch:
        conved = BatchNormalization()(conved)
    if active:
        conved = LeakyReLU(alpha=0.2)(conved)
    return conved

def Conv_gen(inp, filters, kernel_size, strides,padding,activation='relu', batch = True):
    x = Conv2D(filters, kernel_size=kernel_size, strides=strides,padding=padding,activation='relu')(inp)
    if batch:
        x = BatchNormalization(axis=-1)(x)
    return x
    
def Dis_transpose(inp, filters, kernel_size, strides,padding = 'valid'):
    return Conv2DTranspose(filters, kernel_size=kernel_size, strides=strides, padding = padding)(inp)
    

def Dis_classifier(inp):
    d = GlobalAveragePooling2D()(inp)
    #d = GlobalMaxPooling2D()(d)
    d = Dense(1)(d)
    d = Reshape((1,1,1))(d)
    return Activation('sigmoid')(d)


def Gen_lat(lat_space, units, reshape):
    l1 = Dense(units,activation='relu')(lat_space)
    return Reshape(reshape)(l1)


def Gen_transpose(inp, filters, kernel_size, strides,padding, batch = True, last = True):
    x = Conv2DTranspose(filters, kernel_size=kernel_size, strides=strides,padding=padding,activation='relu')(inp)
    if batch:
        x = BatchNormalization(axis=-1)(x)
    if last:
        x =  Conv2D(1, kernel_size=7, padding='same',activation='sigmoid')(x)
    
    return x
    

def synthesis_block(latents, image,filters):
    g = Conv2D(2*filters, kernel_size=3, padding='same')(latents)
    g = BatchNormalization(axis=-1)(g)
    g = LeakyReLU(alpha=0.2)(g)
    g = Conv2D(2*filters,kernel_size=3 , strides=1, padding='same')(g)
    g = BatchNormalization(axis=-1)(g)
    
    g_lat = g[:,:,:,:filters]
    g_switch = Activation('sigmoid')(g[:,:,:,filters:])
    
    x = Multiply()([g_switch, image])
    
    g = Add()([x,g_lat])
    return g


def upsampling(inputs, filters):
    g = Conv2DTranspose(filters, kernel_size=2, strides=(2,2))(inputs)
    g = BatchNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    g = Conv2D(filters,kernel_size=3 , strides=1, padding='same')(g)
    g = BatchNormalization(axis=-1)(g)
    return g
    
def SE_block(input_shape):
    x = GlobalMaxPooling2D()(input_shape)
    x = Dense(int(int(input_shape.shape[-1])/2),activation='relu')(x)
    x = Dense(int(input_shape.shape[-1]),activation='sigmoid')(x)
    x = Reshape((-1,1,int(input_shape.shape[-1])))(x)
    #x = Concatenate()([x, input_shape])
    x = Multiply()([x, input_shape])
    return x
    



def SE_Concat(n_filters, input_layer):
    
    g = Conv2D(n_filters, (3,3), padding='same')(input_layer)
    g = BatchNormalization(axis=-1)(g)
    a = Activation('relu')(g)
	# 
    g = Conv2D(n_filters, (3,3), padding='same')(a)
    g = BatchNormalization(axis=-1)(g)
    g = Activation('relu')(g)
	# 
    g = Conv2D(n_filters, (3,3), padding='same')(g)
    g = SE_block(g)
    
	# 
    g = Concatenate()([g, a])
    return g