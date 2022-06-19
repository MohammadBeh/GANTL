from Models import *
from utils import *

def train(cfg,d_model, g_model, gan_model, dataset):
	# determine the output square shape of the discriminator
    n_patch = d_model.output_shape[1]
	# unpack dataset
    trainA, trainB = dataset
	# calculate the number of batches per training epoch
    bat_per_epo = int(len(trainA) / cfg.batch_size)
	# calculate the number of training iterations
    n_steps = bat_per_epo * cfg.n_epoch

	# manually enumerate epochs
    for i in range(n_steps):
		# select a batch of real samples
        [X_realA, X_realB], y_real = generate_real_samples(dataset, cfg.batch_size, n_patch)
		# generate a batch of fake samples
        X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)
        
        late = generate_latent_points(100, cfg.batch_size)
		# update discriminator for real samples
        d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
		# update discriminator for generated samples
        d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
		# update the generator
        g_loss, _, _ = gan_model.train_on_batch([X_realA,late], [y_real, X_realB])
		# summarize performance
        print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i+1, d_loss1, d_loss2, g_loss))
		# summarize model performance
        if (i+1) % (bat_per_epo * cfg.save_step) == 0:
            summarize_performance((i+1) // (bat_per_epo * cfg.save_step), g_model, dataset, cfg.model_dir, cfg.log_dir)
    return g_model