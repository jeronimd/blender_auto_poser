# Blender Auto Poser

A Blender addon that uses AI to predict poses from keypoints<br>
Based on [ProtoRes: Proto-Residual Network for Pose Authoring via Learned Inverse Kinematics](https://arxiv.org/abs/2106.01981)<br>
Original paper by Boris N. Oreshkin, Florent Bocquelet, Félix G. Harvey, Bay Raitt, Dominic Laflamme<br>
ProtoRes is licensed for non-commercial academic research purposes only<br>

## Download Addon

https://drive.google.com/file/d/1_b3j66wFmAjMd562oWCS3dd7EguTcosU/view?usp=sharing

## Download Trained Model

You can also download the github repository and the model separately for security reasons<br>
All you have to do then is place the \*.ckpt file in the `poser/models/` folder<br>
https://drive.google.com/file/d/1d4nbZ28tMvphEE6aFuggEFRlZaI3lLb5/view?usp=sharing

## Citation

```
@inproceedings{oreshkin2022protores:,
  title={ProtoRes: Proto-Residual Network for Pose Authoring via Learned Inverse Kinematics},
  author={Boris N. Oreshkin and Florent Bocquelet and F{\'{e}}lix G. Harvey and Bay Raitt and Dominic Laflamme},
  booktitle={International Conference on Learning Representations},
  year={2022}
}
```

## Training
This is not needed to use the addon, but if you want to train the model yourself, you can follow the instructions below

### Environment Setup


```bash
# Python version (Blender uses Python 3.11)
Python 3.11.11

# Check if torch is installed
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU device name: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'No GPU available')"

# Install dependencies
pip install numpy
pip install torch --index-url https://download.pytorch.org/whl/cu124 --force-reinstall
pip install pytorch-lightning
pip install tensorboard
pip install matplotlib
pip install debugpy
pip install msgpack
pip install mathutils
pip install fake-bpy-module
```

### Running the Model

```bash
# Start training
python -m poser.trainer

# Redirect output to log file
python -m poser.trainer > training_output.log 2>&1
```

### Monitoring

```bash
# Launch TensorBoard
tensorboard --logdir=poser\models

# Open TensorBoard in browser
http://localhost:6006/
http://127.0.0.1:6006/
```
