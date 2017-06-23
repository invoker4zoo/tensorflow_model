# TensorFlow Models
学习tensorflow过程中各种实例程序

## Models

- [minist_samples](minist_samples):针对minist手写数据集的各类识别方法，包括minist数据集和iunput_data数据应用，测试程序包括了
  * [normal neural network](minist_samples/normal_nn_minist.py):一层隐层，10个神经元的神经网络
  * [cnn](minist/cnn_minist.py)：2层卷积加池化层和一个全连接层组成的cnn网络

- [openAI gym](open_ai):openAI gym, 针对强化学习的游戏接口

- [reinforcement_learing](reinforcement_learning_pygame): 强化学习的一些实例，使用pygame编写游戏接口
  * [game_AI](reinforcement_learning_pygame/game_AI.py):接弹球游戏，将全屏图像作为输入图像，使用三层卷积网络确定action