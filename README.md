# GANTL

This is the code and data for [GANTL: Toward Practical and Real-Time Topology Optimization With Conditional Generative Adversarial Networks and Transfer Learning](https://asmedigitalcollection.asme.org/mechanicaldesign/article/144/2/021711/1121902/GANTL-Toward-Practical-and-Real-Time-Topology) paper published in the Journal of Mechanical Design. 

## Code Usage

### Dependencies

- Tensorflow == 2.3.0
- H5Py == 2.10.0
- Matplotlib == 3.5.1
- Numpy == 1.21.6

### Commands

- `--res` : The resolution of the domain. 
  - Options: `4080` (default), `80160` , `120160`, `120240`, `160320`, `200400`
- `--traintest`: 
  - Options: 
    - `traintest` which train the model and test that
    - `test` which test the saved model
- `--batch_size`: The batch size for the model training (default = 16)
- `--n_epoch`: The number of epochs for training (default = 500)
- `--save_step` : Save models every x epoch (default = 50)
- `--lr` : Learning rate (default = 2e-4)
- `--TOloss` : Whether to test the TO model or not (default = False)
  - Note 1: it should be used only on the test mode
  - Note 2: the resolution must be 200400

### Using Example

To train a model you can use the following command:

```python
python main.py --res 4080 --traintest traintest
```

The code will train the model and save the models in `GANTL/'res'/model` folder. It also save the images obtained during the training in `GANTL/'res'/images`. The final predictions will be saved in `GANTL/'res'/prediction`.

To test the model, one can run the following command:

```python
python main.py --res 4080 --traintest test
```

The result will be saved in  `GANTL/'res'/prediction`.

Note: You can run these codes for any resolutions.

To test the model on TO loss, you have to run the following command:

```python
python main.py --res 200400 --traintest test --TOloss True
```

## Citation 

```
@article{behzadi2022gantl,
  title={GANTL: Toward Practical and Real-Time Topology Optimization With Conditional Generative Adversarial Networks and Transfer Learning},
  author={Behzadi, Mohammad Mahdi and Ilie{\c{s}}, Horea T},
  journal={Journal of Mechanical Design},
  volume={144},
  number={2},
  year={2022},
  publisher={American Society of Mechanical Engineers Digital Collection}
}
```