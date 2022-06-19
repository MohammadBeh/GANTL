from matplotlib import pyplot
import numpy as np
import h5py
import os
from numpy.random import randn
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
def ensure_dir(path):
    """
    create path by first checking its existence,
    :param paths: path
    :return:
    """
    if not os.path.exists(path):
        os.makedirs(path)


def ensure_dirs(paths):
    """
    create paths by first checking their existence
    :param paths: list of path
    :return:
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            ensure_dir(path)
    else:
        ensure_dir(paths)
        
        
def Delete_Nan(xtrain,ytrain):
    y = np.argwhere(np.isnan(ytrain))
    y = y[:,0]
    y = np.unique(y)
    for i in range(np.size(y)-1,-1,-1):
        ytrain = np.delete(ytrain, y[i], 0)
        xtrain = np.delete(xtrain, y[i], 0)
        
    return xtrain,ytrain

# load and prepare training images
def load_real_samples(data_dir, test = False):
    if test:
        X = h5py.File(data_dir+'/xtest.mat', 'r')['xtest']
        Y = h5py.File(data_dir+'/ytest.mat', 'r')['ytest']
    else:
        X = h5py.File(data_dir+'/xtrain.mat', 'r')['xtrain']
        Y = h5py.File(data_dir+'/ytrain.mat', 'r')['ytrain']
    input_image = np.transpose(X)
    input_image = input_image[:,:,:,0:5]
    
    
    Y = np.transpose(Y)
    if Y.shape[1]%40 == 0:
        real_image = np.zeros((np.size(input_image,0),Y.shape[1]+1,Y.shape[2]+1, 1))
        real_image[:,:Y.shape[1],:Y.shape[2], 0] = Y
    else:
        real_image = np.reshape(Y,(-1,Y.shape[1]+1,Y.shape[2]+1,1))
    input_image ,real_image= Delete_Nan(input_image,real_image)
    
    #input_image = input_image[0:10000,:,:,:]
    #real_image = real_image[0:10000,:,:,:]
    return [input_image, real_image]
 
# select a batch of random samples, returns images and target
def generate_real_samples(dataset, n_samples, patch_shape):
	# unpack dataset
	trainA, trainB = dataset
	# choose random instances
	ix = np.random.randint(0, trainA.shape[0], n_samples)
	# retrieve selected images
	X1, X2 = trainA[ix], trainB[ix]
	# generate 'real' class labels (1)
	y = 0.9*np.ones((n_samples, patch_shape, patch_shape, 1))
	return [X1, X2], y

    


def generate_latent_points(latent_dim, n_samples):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	x_input = x_input.reshape(n_samples, latent_dim)
	return x_input


# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, samples, patch_shape):
	# generate fake instance
    lat = generate_latent_points(100, samples.shape[0])
    X = g_model.predict([samples,lat])
	# create 'fake' class labels (0)
    y = np.zeros((len(X), patch_shape, patch_shape, 1))
    return X, y
 
# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, dataset,model_dir,log_dir, n_samples=3):
	# select a sample of input images
	[X_realA, X_realB], _ = generate_real_samples(dataset, n_samples, 1)
	# generate a batch of fake samples
	X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1)
	
	for i in range(n_samples):
		pyplot.subplot(3, n_samples, 1 + n_samples + i)
		pyplot.axis('off')
		pyplot.imshow(X_fakeB[i,:,:,0])
	# plot real target image
	for i in range(n_samples):
		pyplot.subplot(3, n_samples, 1 + n_samples*2 + i)
		pyplot.axis('off')
		pyplot.imshow(X_realB[i,:,:,0])
	# save plot to file
	filename1 = log_dir+'\\plot_%03d.png' % (step)
	pyplot.savefig(filename1)
	pyplot.close()
	# save the generator model
	filename2 = model_dir+'\\model_%03d.h5' % (step)
	g_model.save(filename2)
	print('>Saved: %s and %s' % (filename1, filename2))
    
    
def save_prediction(n_testsamples, pred_dir, ytest,ypred):
    for i in range((n_testsamples if n_testsamples<ypred.shape[0] else ypred.shape[0])):
        pyplot.figure(figsize=(10, 10))
        pyplot.axis('off')
        pyplot.imshow(ytest[i,:,:,0])
        filename1 = pred_dir+'\\gt_%d.png' % (i)
        pyplot.savefig(filename1)
        pyplot.close()
        
        pyplot.figure(figsize=(10, 10))
        pyplot.axis('off')
        pyplot.imshow(ypred[i,:,:,0])
        filename2 = pred_dir+'\\yp_%d.png' % (i)
        pyplot.savefig(filename2)
        pyplot.close()
    
    
def Load_pretrain_model(model_dir):
    base_model = load_model(model_dir)
    mmodel = Model(inputs=base_model.input,outputs= base_model.layers[-2].output)
    mmodel.trainable = True
    return mmodel


def extract_res(args):
    if int(args.res[2]) != 0:
        return (int(args.res[:2]), int(args.res[2:]))
    else:
        return (int(args.res[:3]), int(args.res[3:]))
    