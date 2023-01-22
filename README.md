# Hand Prosthesis Controller with Deep Learning and Tensorflow Lite

## Disclaimer

This work was created to be my final project from my college (Universidade Federal do Rio Grande do Norte - UFRN), it was made wih the help of my Professor Advisor Helton Maia.

## About and Objectives

This project has the objective to create a Hand Prosthesis Controller using Deep Learning techniques, in which the model created with those techniques were embedded in a low energy microcontroller with the porpouse to run in real time, and should be low cost and acessible. To solve this problem were used common tools, such as TensorFlow, TensorFlow Lite, Numpy, Librosa, and Scypy. The main idea Is to create a classifier that classify each spectrogram (an spectrogram is an imagem of a sound) of a surface EMG (sEMG) signal in five different hand gestures.

One of the main objectives of this project is to study how electromyography (EMG) signals works, and how Deep Learning techniques can learning its paterns. This is a very interesting topic in Health. The other main objective of this project is to use TinyML techniques to embed Deep Learning models, TinyML here is a set of techniques that enable use Machine Learning/Deep Learning Models in low energy microcontrolers.

For this project it was used Arduino nano 3 BLE sense, but is possible to do se same for raspberry pi pico. Unfortunately there was no raspberry pi pico available during the project. 

More Details can be found in: https://repositorio.ufrn.br/handle/123456789/50899

## Database

I've used "sEMG for Basic Hand movements Dataset" that can found in: https://archive.ics.uci.edu/ml/datasets/sEMG+for+Basic+Hand+movements#. And to process this dataset
i've created a python file to transform each matlab file in imagens of every class and channel, it was done following dataset's instructions.

## Main Results

### Hiperparameters

To create the best model were made a sweep using Weight&Bias framework to find the best hyperparameters for the model, and for a better pre-process of the images. All the results made can be found here: https://wandb.ai/erikaalves24/tcc_wout_palm?workspace=user-erikaalves.

After fiding the best hyperparameters, it were made some adjutments to better fit and a microcontroller and processes in real time.

### Model Created with TensorFlow

In the imagem bellow, we can see the confusion matrix of the results fo the best model. Were obtained accuracy of 0.93 with validation subset. And also had a f1-score, precision and recall vary from 0.91-0.94 for each class.

<p align="center">
  <img src="https://github.com/CommanderErika/Hand-Prosthesis-Controller-with-Deep-Learning-and-Tensorflow-Lite/blob/main/files/matrix_final_ajeitado.png" width="700" heigth = "800" title="Confusion Matrix before Quantization">
</p>

### Model after Quantization with TensorFlow Lite

After creating the model, It were necessary to quantizate the model to be suitable fr low power microcontrolers. In the imagem bellow w have the results after the quantization. It were also obtained accuracy of 0.93 with validation subset. And also had a f1-score, precision and recall vary from 0.91-0.94 for each class.

<p align="center">
  <img src="https://github.com/CommanderErika/Hand-Prosthesis-Controller-with-Deep-Learning-and-Tensorflow-Lite/blob/main/files/matrix_final.png" width="700" heigth = "800" title="Confusion Matrix after Quantization">
</p>
