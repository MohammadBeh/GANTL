from Models import *
from config import * 
from trainer import * 
from utils import * 
from tensorflow.keras.models import load_model

cfg = ConfigGAN()

if cfg.traintest == 'traintest':
    if cfg.res == '4080':
        dataset = load_real_samples(cfg.data_root, test = False)
        print('Dataset is loaded', dataset[0].shape, dataset[1].shape)
        d_model = Pre_Discriminator(dataset[0].shape[1:], dataset[1].shape[1:])
        g_model = Pre_Generator(dataset[0].shape[1:])
    elif cfg.res != '4080':
        pretrain_model = Load_pretrain_model(cfg.pretrain_dir)
        dataset = load_real_samples(cfg.data_root, test = False)
        print('Dataset is loaded', dataset[0].shape, dataset[1].shape)
        d_model = Discriminator(dataset[0].shape[1:], dataset[1].shape[1:])
        g_model = Generator(cfg, pretrain_model)
    # define the composite model
    gan_model = GAN(g_model, d_model, dataset[0].shape[1:])
    # train model
    model = train(cfg,d_model, g_model, gan_model, dataset)
    test_dataset = load_real_samples(cfg.data_root, test = True)
    lat_input = generate_latent_points(100, test_dataset[0].shape[0])
    print('Test Data is loaded', test_dataset[0].shape, test_dataset[1].shape)
    ypred = model.predict([test_dataset[0],lat_input])
    
    save_prediction(cfg.n_testsamples, cfg.pred_dir, test_dataset[1],ypred)
    print('The predicted images has been saved!')
    
    
if cfg.traintest == 'test':
    test_dataset = load_real_samples(cfg.data_root, test = True)
    lat_input = generate_latent_points(100, test_dataset[0].shape[0])
    print('Test Data is loaded', test_dataset[0].shape, test_dataset[1].shape)
    model = load_model(cfg.model_dir + '\\model_%03d.h5' % (cfg.load_point))
    ypred = model.predict([test_dataset[0],lat_input])
    
    save_prediction(cfg.n_testsamples, cfg.pred_dir, test_dataset[1],ypred)
    print('The predicted images has been saved!')