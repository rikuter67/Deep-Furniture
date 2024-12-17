import re
import matplotlib.pyplot as plt
import numpy as np
import os

history = ["""
Epoch 2/200
195/195 [==============================] - ETA: 0s - loss: 2.2972 - Set_accuracy: 0.3555 
Epoch 2: val_Set_accuracy improved from 0.32763 to 0.36167, saving model to /data1/yamazono/setRetrieval/DeepFurniture/output/models/best_model.weights.h5
195/195 [==============================] - 10s 51ms/step - loss: 2.2972 - Set_accuracy: 0.3555 - val_loss: 1.2491 - val_Set_accuracy: 0.3617 - lr: 0.0010
Epoch 3/200
194/195 [============================>.] - ETA: 0s - loss: 0.8278 - Set_accuracy: 0.4144 
Epoch 3: val_Set_accuracy improved from 0.36167 to 0.38491, saving model to /data1/yamazono/setRetrieval/DeepFurniture/output/models/best_model.weights.h5
195/195 [==============================] - 10s 50ms/step - loss: 0.8276 - Set_accuracy: 0.4139 - val_loss: 0.6115 - val_Set_accuracy: 0.3849 - lr: 0.0010
Epoch 4/200
194/195 [============================>.] - ETA: 0s - loss: 0.5002 - Set_accuracy: 0.4389
Epoch 4: val_Set_accuracy improved from 0.38491 to 0.43210, saving model to /data1/yamazono/setRetrieval/DeepFurniture/output/models/best_model.weights.h5
195/195 [==============================] - 10s 50ms/step - loss: 0.4990 - Set_accuracy: 0.4393 - val_loss: 0.3558 - val_Set_accuracy: 0.4321 - lr: 0.0010
Epoch 5/200
195/195 [==============================] - ETA: 0s - loss: 0.3514 - Set_accuracy: 0.4428 
Epoch 5: val_Set_accuracy did not improve from 0.43210
195/195 [==============================] - 10s 50ms/step - loss: 0.3514 - Set_accuracy: 0.4428 - val_loss: 0.3062 - val_Set_accuracy: 0.3957 - lr: 0.0010
Epoch 6/200
195/195 [==============================] - ETA: 0s - loss: 0.2262 - Set_accuracy: 0.4467
Epoch 6: val_Set_accuracy improved from 0.43210 to 0.43426, saving model to /data1/yamazono/setRetrieval/DeepFurniture/output/models/best_model.weights.h5
195/195 [==============================] - 10s 49ms/step - loss: 0.2262 - Set_accuracy: 0.4467 - val_loss: 0.2031 - val_Set_accuracy: 0.4343 - lr: 0.0010
Epoch 7/200
194/195 [============================>.] - ETA: 0s - loss: 0.1814 - Set_accuracy: 0.4520 
Epoch 7: val_Set_accuracy did not improve from 0.43426
195/195 [==============================] - 10s 49ms/step - loss: 0.1816 - Set_accuracy: 0.4514 - val_loss: 0.2814 - val_Set_accuracy: 0.4143 - lr: 0.0010
Epoch 8/200
195/195 [==============================] - ETA: 0s - loss: 0.1626 - Set_accuracy: 0.4516 
Epoch 8: val_Set_accuracy did not improve from 0.43426
195/195 [==============================] - 9s 49ms/step - loss: 0.1626 - Set_accuracy: 0.4516 - val_loss: 0.1846 - val_Set_accuracy: 0.3656 - lr: 0.0010
Epoch 9/200
195/195 [==============================] - ETA: 0s - loss: 0.1378 - Set_accuracy: 0.4397 
Epoch 9: val_Set_accuracy improved from 0.43426 to 0.48019, saving model to /data1/yamazono/setRetrieval/DeepFurniture/output/models/best_model.weights.h5
195/195 [==============================] - 10s 49ms/step - loss: 0.1378 - Set_accuracy: 0.4397 - val_loss: 0.0915 - val_Set_accuracy: 0.4802 - lr: 0.0010
Epoch 10/200
195/195 [==============================] - ETA: 0s - loss: 0.0951 - Set_accuracy: 0.4707
Epoch 10: val_Set_accuracy did not improve from 0.48019
195/195 [==============================] - 9s 49ms/step - loss: 0.0951 - Set_accuracy: 0.4707 - val_loss: 0.1231 - val_Set_accuracy: 0.3896 - lr: 0.0010
Epoch 11/200
194/195 [============================>.] - ETA: 0s - loss: 0.0826 - Set_accuracy: 0.4816
Epoch 11: val_Set_accuracy improved from 0.48019 to 0.50378, saving model to /data1/yamazono/setRetrieval/DeepFurniture/output/models/best_model.weights.h5
195/195 [==============================] - 10s 52ms/step - loss: 0.0826 - Set_accuracy: 0.4819 - val_loss: 0.0753 - val_Set_accuracy: 0.5038 - lr: 0.0010
Epoch 12/200
194/195 [============================>.] - ETA: 0s - loss: 0.0623 - Set_accuracy: 0.5163 
Epoch 12: val_Set_accuracy did not improve from 0.50378
195/195 [==============================] - 10s 51ms/step - loss: 0.0623 - Set_accuracy: 0.5163 - val_loss: 0.0907 - val_Set_accuracy: 0.4379 - lr: 0.0010
Epoch 13/200
194/195 [============================>.] - ETA: 0s - loss: 0.0617 - Set_accuracy: 0.5209 
Epoch 13: val_Set_accuracy improved from 0.50378 to 0.53116, saving model to /data1/yamazono/setRetrieval/DeepFurniture/output/models/best_model.weights.h5
195/195 [==============================] - 10s 49ms/step - loss: 0.0617 - Set_accuracy: 0.5211 - val_loss: 0.0519 - val_Set_accuracy: 0.5312 - lr: 0.0010
Epoch 14/200
195/195 [==============================] - ETA: 0s - loss: 0.0597 - Set_accuracy: 0.5331 
Epoch 14: val_Set_accuracy did not improve from 0.53116
195/195 [==============================] - 10s 49ms/step - loss: 0.0597 - Set_accuracy: 0.5331 - val_loss: 0.0352 - val_Set_accuracy: 0.5189 - lr: 0.0010
Epoch 15/200
195/195 [==============================] - ETA: 0s - loss: 0.0496 - Set_accuracy: 0.5363 
Epoch 15: val_Set_accuracy did not improve from 0.53116
195/195 [==============================] - 10s 49ms/step - loss: 0.0496 - Set_accuracy: 0.5363 - val_loss: 0.0717 - val_Set_accuracy: 0.4730 - lr: 0.0010
Epoch 16/200
194/195 [============================>.] - ETA: 0s - loss: 0.0522 - Set_accuracy: 0.5239 
Epoch 16: val_Set_accuracy did not improve from 0.53116
195/195 [==============================] - 10s 49ms/step - loss: 0.0521 - Set_accuracy: 0.5245 - val_loss: 0.0432 - val_Set_accuracy: 0.5312 - lr: 0.0010
Epoch 17/200
194/195 [============================>.] - ETA: 0s - loss: 0.0484 - Set_accuracy: 0.5448 
Epoch 17: val_Set_accuracy did not improve from 0.53116
195/195 [==============================] - 10s 49ms/step - loss: 0.0483 - Set_accuracy: 0.5447 - val_loss: 0.0504 - val_Set_accuracy: 0.4984 - lr: 0.0010
Epoch 18/200
195/195 [==============================] - ETA: 0s - loss: 0.0436 - Set_accuracy: 0.5155
Epoch 18: val_Set_accuracy did not improve from 0.53116

Epoch 18: ReduceLROnPlateau reducing learning rate to 0.0005000000237487257.
195/195 [==============================] - 9s 48ms/step - loss: 0.0436 - Set_accuracy: 0.5155 - val_loss: 0.0226 - val_Set_accuracy: 0.5209 - lr: 0.0010
Epoch 19/200
195/195 [==============================] - ETA: 0s - loss: 0.0107 - Set_accuracy: 0.6691 
Epoch 19: val_Set_accuracy improved from 0.53116 to 0.65742, saving model to /data1/yamazono/setRetrieval/DeepFurniture/output/models/best_model.weights.h5
195/195 [==============================] - 10s 51ms/step - loss: 0.0107 - Set_accuracy: 0.6691 - val_loss: 0.0101 - val_Set_accuracy: 0.6574 - lr: 5.0000e-04
Epoch 20/200
194/195 [============================>.] - ETA: 0s - loss: 0.0082 - Set_accuracy: 0.7037 
Epoch 20: val_Set_accuracy did not improve from 0.65742
195/195 [==============================] - 10s 53ms/step - loss: 0.0082 - Set_accuracy: 0.7034 - val_loss: 0.0099 - val_Set_accuracy: 0.6436 - lr: 5.0000e-04
Epoch 21/200
195/195 [==============================] - ETA: 0s - loss: 0.0078 - Set_accuracy: 0.7119
Epoch 21: val_Set_accuracy did not improve from 0.65742
195/195 [==============================] - 10s 50ms/step - loss: 0.0078 - Set_accuracy: 0.7119 - val_loss: 0.0092 - val_Set_accuracy: 0.6572 - lr: 5.0000e-04
Epoch 22/200
194/195 [============================>.] - ETA: 0s - loss: 0.0073 - Set_accuracy: 0.7298 
Epoch 22: val_Set_accuracy improved from 0.65742 to 0.66282, saving model to /data1/yamazono/setRetrieval/DeepFurniture/output/models/best_model.weights.h5
195/195 [==============================] - 10s 52ms/step - loss: 0.0073 - Set_accuracy: 0.7301 - val_loss: 0.0092 - val_Set_accuracy: 0.6628 - lr: 5.0000e-04
Epoch 23/200
195/195 [==============================] - ETA: 0s - loss: 0.0071 - Set_accuracy: 0.7363 
Epoch 23: val_Set_accuracy improved from 0.66282 to 0.66895, saving model to /data1/yamazono/setRetrieval/DeepFurniture/output/models/best_model.weights.h5
195/195 [==============================] - 10s 49ms/step - loss: 0.0071 - Set_accuracy: 0.7363 - val_loss: 0.0093 - val_Set_accuracy: 0.6689 - lr: 5.0000e-04
Epoch 24/200
195/195 [==============================] - ETA: 0s - loss: 0.0072 - Set_accuracy: 0.7342 
Epoch 24: val_Set_accuracy improved from 0.66895 to 0.67417, saving model to /data1/yamazono/setRetrieval/DeepFurniture/output/models/best_model.weights.h5
195/195 [==============================] - 10s 49ms/step - loss: 0.0072 - Set_accuracy: 0.7342 - val_loss: 0.0085 - val_Set_accuracy: 0.6742 - lr: 5.0000e-04
Epoch 25/200
195/195 [==============================] - ETA: 0s - loss: 0.0071 - Set_accuracy: 0.7365 
Epoch 25: val_Set_accuracy did not improve from 0.67417
195/195 [==============================] - 9s 48ms/step - loss: 0.0071 - Set_accuracy: 0.7365 - val_loss: 0.0091 - val_Set_accuracy: 0.6704 - lr: 5.0000e-04
Epoch 26/200
195/195 [==============================] - ETA: 0s - loss: 0.0075 - Set_accuracy: 0.7261
Epoch 26: val_Set_accuracy did not improve from 0.67417
195/195 [==============================] - 9s 48ms/step - loss: 0.0075 - Set_accuracy: 0.7261 - val_loss: 0.0089 - val_Set_accuracy: 0.6738 - lr: 5.0000e-04
Epoch 27/200
194/195 [============================>.] - ETA: 0s - loss: 0.0076 - Set_accuracy: 0.7321 
Epoch 27: val_Set_accuracy improved from 0.67417 to 0.68876, saving model to /data1/yamazono/setRetrieval/DeepFurniture/output/models/best_model.weights.h5
195/195 [==============================] - 10s 49ms/step - loss: 0.0076 - Set_accuracy: 0.7321 - val_loss: 0.0091 - val_Set_accuracy: 0.6888 - lr: 5.0000e-04
Epoch 28/200
195/195 [==============================] - ETA: 0s - loss: 0.0074 - Set_accuracy: 0.7401
Epoch 28: val_Set_accuracy did not improve from 0.68876
195/195 [==============================] - 9s 49ms/step - loss: 0.0074 - Set_accuracy: 0.7401 - val_loss: 0.0087 - val_Set_accuracy: 0.6828 - lr: 5.0000e-04
Epoch 29/200
195/195 [==============================] - ETA: 0s - loss: 0.0083 - Set_accuracy: 0.7276
Epoch 29: val_Set_accuracy did not improve from 0.68876
195/195 [==============================] - 9s 49ms/step - loss: 0.0083 - Set_accuracy: 0.7276 - val_loss: 0.0095 - val_Set_accuracy: 0.6461 - lr: 5.0000e-04
Epoch 30/200
195/195 [==============================] - ETA: 0s - loss: 0.0075 - Set_accuracy: 0.7413
Epoch 30: val_Set_accuracy did not improve from 0.68876
195/195 [==============================] - 10s 49ms/step - loss: 0.0075 - Set_accuracy: 0.7413 - val_loss: 0.0098 - val_Set_accuracy: 0.6527 - lr: 5.0000e-04
Epoch 31/200
195/195 [==============================] - ETA: 0s - loss: 0.0077 - Set_accuracy: 0.7436
Epoch 31: val_Set_accuracy did not improve from 0.68876
195/195 [==============================] - 10s 49ms/step - loss: 0.0077 - Set_accuracy: 0.7436 - val_loss: 0.0107 - val_Set_accuracy: 0.6450 - lr: 5.0000e-04
Epoch 32/200
195/195 [==============================] - ETA: 0s - loss: 0.0091 - Set_accuracy: 0.7199 
Epoch 32: val_Set_accuracy did not improve from 0.68876

Epoch 32: ReduceLROnPlateau reducing learning rate to 0.0002500000118743628.
195/195 [==============================] - 9s 48ms/step - loss: 0.0091 - Set_accuracy: 0.7199 - val_loss: 0.0112 - val_Set_accuracy: 0.6576 - lr: 5.0000e-04
Epoch 33/200
195/195 [==============================] - ETA: 0s - loss: 0.0066 - Set_accuracy: 0.7742 
Epoch 33: val_Set_accuracy improved from 0.68876 to 0.72352, saving model to /data1/yamazono/setRetrieval/DeepFurniture/output/models/best_model.weights.h5
195/195 [==============================] - 10s 49ms/step - loss: 0.0066 - Set_accuracy: 0.7742 - val_loss: 0.0073 - val_Set_accuracy: 0.7235 - lr: 2.5000e-04
Epoch 34/200
195/195 [==============================] - ETA: 0s - loss: 0.0058 - Set_accuracy: 0.8002 
Epoch 34: val_Set_accuracy improved from 0.72352 to 0.72604, saving model to /data1/yamazono/setRetrieval/DeepFurniture/output/models/best_model.weights.h5
195/195 [==============================] - 10s 49ms/step - loss: 0.0058 - Set_accuracy: 0.8002 - val_loss: 0.0073 - val_Set_accuracy: 0.7260 - lr: 2.5000e-04
Epoch 35/200
195/195 [==============================] - ETA: 0s - loss: 0.0058 - Set_accuracy: 0.7974
Epoch 35: val_Set_accuracy improved from 0.72604 to 0.73685, saving model to /data1/yamazono/setRetrieval/DeepFurniture/output/models/best_model.weights.h5
195/195 [==============================] - 10s 50ms/step - loss: 0.0058 - Set_accuracy: 0.7974 - val_loss: 0.0070 - val_Set_accuracy: 0.7369 - lr: 2.5000e-04
Epoch 36/200
195/195 [==============================] - ETA: 0s - loss: 0.0058 - Set_accuracy: 0.8029 
Epoch 36: val_Set_accuracy did not improve from 0.73685
195/195 [==============================] - 10s 49ms/step - loss: 0.0058 - Set_accuracy: 0.8029 - val_loss: 0.0082 - val_Set_accuracy: 0.7169 - lr: 2.5000e-04
Epoch 37/200
195/195 [==============================] - ETA: 0s - loss: 0.0058 - Set_accuracy: 0.8057 
Epoch 37: val_Set_accuracy did not improve from 0.73685
195/195 [==============================] - 9s 48ms/step - loss: 0.0058 - Set_accuracy: 0.8057 - val_loss: 0.0073 - val_Set_accuracy: 0.7311 - lr: 2.5000e-04
Epoch 38/200
195/195 [==============================] - ETA: 0s - loss: 0.0060 - Set_accuracy: 0.7983
Epoch 38: val_Set_accuracy did not improve from 0.73685
195/195 [==============================] - 10s 49ms/step - loss: 0.0060 - Set_accuracy: 0.7983 - val_loss: 0.0080 - val_Set_accuracy: 0.7176 - lr: 2.5000e-04
Epoch 39/200
195/195 [==============================] - ETA: 0s - loss: 0.0059 - Set_accuracy: 0.8008
Epoch 39: val_Set_accuracy improved from 0.73685 to 0.73919, saving model to /data1/yamazono/setRetrieval/DeepFurniture/output/models/best_model.weights.h5
195/195 [==============================] - 9s 49ms/step - loss: 0.0059 - Set_accuracy: 0.8008 - val_loss: 0.0074 - val_Set_accuracy: 0.7392 - lr: 2.5000e-04
Epoch 40/200
195/195 [==============================] - ETA: 0s - loss: 0.0061 - Set_accuracy: 0.8040 
Epoch 40: val_Set_accuracy did not improve from 0.73919
195/195 [==============================] - 9s 49ms/step - loss: 0.0061 - Set_accuracy: 0.8040 - val_loss: 0.0077 - val_Set_accuracy: 0.7277 - lr: 2.5000e-04
Epoch 41/200
195/195 [==============================] - ETA: 0s - loss: 0.0063 - Set_accuracy: 0.7981
Epoch 41: val_Set_accuracy did not improve from 0.73919
195/195 [==============================] - 9s 48ms/step - loss: 0.0063 - Set_accuracy: 0.7981 - val_loss: 0.0076 - val_Set_accuracy: 0.7298 - lr: 2.5000e-04
Epoch 42/200
195/195 [==============================] - ETA: 0s - loss: 0.0060 - Set_accuracy: 0.8013 
Epoch 42: val_Set_accuracy did not improve from 0.73919
195/195 [==============================] - 10s 49ms/step - loss: 0.0060 - Set_accuracy: 0.8013 - val_loss: 0.0077 - val_Set_accuracy: 0.7392 - lr: 2.5000e-04
Epoch 43/200
195/195 [==============================] - ETA: 0s - loss: 0.0061 - Set_accuracy: 0.8121
Epoch 43: val_Set_accuracy did not improve from 0.73919
195/195 [==============================] - 10s 49ms/step - loss: 0.0061 - Set_accuracy: 0.8121 - val_loss: 0.0085 - val_Set_accuracy: 0.7300 - lr: 2.5000e-04
Epoch 44/200
195/195 [==============================] - ETA: 0s - loss: 0.0060 - Set_accuracy: 0.8096
Epoch 44: val_Set_accuracy did not improve from 0.73919

Epoch 44: ReduceLROnPlateau reducing learning rate to 0.0001250000059371814.
195/195 [==============================] - 10s 49ms/step - loss: 0.0060 - Set_accuracy: 0.8096 - val_loss: 0.0075 - val_Set_accuracy: 0.7372 - lr: 2.5000e-04
Epoch 45/200
195/195 [==============================] - ETA: 0s - loss: 0.0052 - Set_accuracy: 0.8309 
Epoch 45: val_Set_accuracy improved from 0.73919 to 0.75919, saving model to /data1/yamazono/setRetrieval/DeepFurniture/output/models/best_model.weights.h5
195/195 [==============================] - 10s 49ms/step - loss: 0.0052 - Set_accuracy: 0.8309 - val_loss: 0.0067 - val_Set_accuracy: 0.7592 - lr: 1.2500e-04
Epoch 46/200
194/195 [============================>.] - ETA: 0s - loss: 0.0048 - Set_accuracy: 0.8419 
Epoch 46: val_Set_accuracy improved from 0.75919 to 0.76549, saving model to /data1/yamazono/setRetrieval/DeepFurniture/output/models/best_model.weights.h5
195/195 [==============================] - 10s 49ms/step - loss: 0.0048 - Set_accuracy: 0.8420 - val_loss: 0.0066 - val_Set_accuracy: 0.7655 - lr: 1.2500e-04
Epoch 47/200
194/195 [============================>.] - ETA: 0s - loss: 0.0050 - Set_accuracy: 0.8413
Epoch 47: val_Set_accuracy improved from 0.76549 to 0.77341, saving model to /data1/yamazono/setRetrieval/DeepFurniture/output/models/best_model.weights.h5
195/195 [==============================] - 10s 50ms/step - loss: 0.0050 - Set_accuracy: 0.8413 - val_loss: 0.0064 - val_Set_accuracy: 0.7734 - lr: 1.2500e-04
Epoch 48/200
195/195 [==============================] - ETA: 0s - loss: 0.0050 - Set_accuracy: 0.8469
Epoch 48: val_Set_accuracy improved from 0.77341 to 0.77504, saving model to /data1/yamazono/setRetrieval/DeepFurniture/output/models/best_model.weights.h5
195/195 [==============================] - 10s 51ms/step - loss: 0.0050 - Set_accuracy: 0.8469 - val_loss: 0.0065 - val_Set_accuracy: 0.7750 - lr: 1.2500e-04
Epoch 49/200
195/195 [==============================] - ETA: 0s - loss: 0.0049 - Set_accuracy: 0.8463 
Epoch 49: val_Set_accuracy did not improve from 0.77504
195/195 [==============================] - 10s 49ms/step - loss: 0.0049 - Set_accuracy: 0.8463 - val_loss: 0.0064 - val_Set_accuracy: 0.7622 - lr: 1.2500e-04
Epoch 50/200
195/195 [==============================] - ETA: 0s - loss: 0.0049 - Set_accuracy: 0.8460
Epoch 50: val_Set_accuracy did not improve from 0.77504
195/195 [==============================] - 10s 50ms/step - loss: 0.0049 - Set_accuracy: 0.8460 - val_loss: 0.0066 - val_Set_accuracy: 0.7682 - lr: 1.2500e-04
Epoch 51/200
195/195 [==============================] - ETA: 0s - loss: 0.0050 - Set_accuracy: 0.8462
Epoch 51: val_Set_accuracy did not improve from 0.77504
195/195 [==============================] - 10s 49ms/step - loss: 0.0050 - Set_accuracy: 0.8462 - val_loss: 0.0066 - val_Set_accuracy: 0.7716 - lr: 1.2500e-04
Epoch 52/200
195/195 [==============================] - ETA: 0s - loss: 0.0048 - Set_accuracy: 0.8515
Epoch 52: val_Set_accuracy did not improve from 0.77504
195/195 [==============================] - 9s 49ms/step - loss: 0.0048 - Set_accuracy: 0.8515 - val_loss: 0.0066 - val_Set_accuracy: 0.7673 - lr: 1.2500e-04
Epoch 53/200
195/195 [==============================] - ETA: 0s - loss: 0.0049 - Set_accuracy: 0.8515 
Epoch 53: val_Set_accuracy did not improve from 0.77504

Epoch 53: ReduceLROnPlateau reducing learning rate to 6.25000029685907e-05.
195/195 [==============================] - 10s 49ms/step - loss: 0.0049 - Set_accuracy: 0.8515 - val_loss: 0.0072 - val_Set_accuracy: 0.7585 - lr: 1.2500e-04
Epoch 54/200
195/195 [==============================] - ETA: 0s - loss: 0.0045 - Set_accuracy: 0.8621 
Epoch 54: val_Set_accuracy improved from 0.77504 to 0.78368, saving model to /data1/yamazono/setRetrieval/DeepFurniture/output/models/best_model.weights.h5
195/195 [==============================] - 10s 50ms/step - loss: 0.0045 - Set_accuracy: 0.8621 - val_loss: 0.0060 - val_Set_accuracy: 0.7837 - lr: 6.2500e-05
Epoch 55/200
195/195 [==============================] - ETA: 0s - loss: 0.0043 - Set_accuracy: 0.8628 
Epoch 55: val_Set_accuracy improved from 0.78368 to 0.78620, saving model to /data1/yamazono/setRetrieval/DeepFurniture/output/models/best_model.weights.h5
195/195 [==============================] - 10s 49ms/step - loss: 0.0043 - Set_accuracy: 0.8628 - val_loss: 0.0059 - val_Set_accuracy: 0.7862 - lr: 6.2500e-05
Epoch 56/200
195/195 [==============================] - ETA: 0s - loss: 0.0043 - Set_accuracy: 0.8687 
Epoch 56: val_Set_accuracy did not improve from 0.78620
195/195 [==============================] - 10s 49ms/step - loss: 0.0043 - Set_accuracy: 0.8687 - val_loss: 0.0059 - val_Set_accuracy: 0.7837 - lr: 6.2500e-05
Epoch 57/200
194/195 [============================>.] - ETA: 0s - loss: 0.0043 - Set_accuracy: 0.8673 
Epoch 57: val_Set_accuracy did not improve from 0.78620
195/195 [==============================] - 10s 49ms/step - loss: 0.0043 - Set_accuracy: 0.8674 - val_loss: 0.0063 - val_Set_accuracy: 0.7810 - lr: 6.2500e-05
Epoch 58/200
194/195 [============================>.] - ETA: 0s - loss: 0.0043 - Set_accuracy: 0.8718 
Epoch 58: val_Set_accuracy improved from 0.78620 to 0.79089, saving model to /data1/yamazono/setRetrieval/DeepFurniture/output/models/best_model.weights.h5
195/195 [==============================] - 10s 50ms/step - loss: 0.0042 - Set_accuracy: 0.8717 - val_loss: 0.0059 - val_Set_accuracy: 0.7909 - lr: 6.2500e-05
Epoch 59/200
195/195 [==============================] - ETA: 0s - loss: 0.0043 - Set_accuracy: 0.8714 
Epoch 59: val_Set_accuracy did not improve from 0.79089
195/195 [==============================] - 10s 49ms/step - loss: 0.0043 - Set_accuracy: 0.8714 - val_loss: 0.0059 - val_Set_accuracy: 0.7887 - lr: 6.2500e-05
Epoch 60/200
195/195 [==============================] - ETA: 0s - loss: 0.0044 - Set_accuracy: 0.8655 
Epoch 60: val_Set_accuracy did not improve from 0.79089
195/195 [==============================] - 10s 49ms/step - loss: 0.0044 - Set_accuracy: 0.8655 - val_loss: 0.0060 - val_Set_accuracy: 0.7864 - lr: 6.2500e-05
Epoch 61/200
195/195 [==============================] - ETA: 0s - loss: 0.0043 - Set_accuracy: 0.8728 
Epoch 61: val_Set_accuracy did not improve from 0.79089
195/195 [==============================] - 10s 49ms/step - loss: 0.0043 - Set_accuracy: 0.8728 - val_loss: 0.0060 - val_Set_accuracy: 0.7905 - lr: 6.2500e-05
Epoch 62/200
195/195 [==============================] - ETA: 0s - loss: 0.0043 - Set_accuracy: 0.8729 
Epoch 62: val_Set_accuracy did not improve from 0.79089
195/195 [==============================] - 10s 50ms/step - loss: 0.0043 - Set_accuracy: 0.8729 - val_loss: 0.0058 - val_Set_accuracy: 0.7903 - lr: 6.2500e-05
Epoch 63/200
195/195 [==============================] - ETA: 0s - loss: 0.0042 - Set_accuracy: 0.8746
Epoch 63: val_Set_accuracy improved from 0.79089 to 0.79521, saving model to /data1/yamazono/setRetrieval/DeepFurniture/output/models/best_model.weights.h5
195/195 [==============================] - 10s 50ms/step - loss: 0.0042 - Set_accuracy: 0.8746 - val_loss: 0.0058 - val_Set_accuracy: 0.7952 - lr: 6.2500e-05
Epoch 64/200
195/195 [==============================] - ETA: 0s - loss: 0.0042 - Set_accuracy: 0.8777 
Epoch 64: val_Set_accuracy improved from 0.79521 to 0.79539, saving model to /data1/yamazono/setRetrieval/DeepFurniture/output/models/best_model.weights.h5
195/195 [==============================] - 10s 49ms/step - loss: 0.0042 - Set_accuracy: 0.8777 - val_loss: 0.0060 - val_Set_accuracy: 0.7954 - lr: 6.2500e-05
Epoch 65/200
194/195 [============================>.] - ETA: 0s - loss: 0.0042 - Set_accuracy: 0.8757
Epoch 65: val_Set_accuracy did not improve from 0.79539
195/195 [==============================] - 10s 49ms/step - loss: 0.0042 - Set_accuracy: 0.8759 - val_loss: 0.0060 - val_Set_accuracy: 0.7947 - lr: 6.2500e-05
Epoch 66/200
195/195 [==============================] - ETA: 0s - loss: 0.0041 - Set_accuracy: 0.8777 
Epoch 66: val_Set_accuracy improved from 0.79539 to 0.79809, saving model to /data1/yamazono/setRetrieval/DeepFurniture/output/models/best_model.weights.h5
195/195 [==============================] - 10s 52ms/step - loss: 0.0041 - Set_accuracy: 0.8777 - val_loss: 0.0059 - val_Set_accuracy: 0.7981 - lr: 6.2500e-05
Epoch 67/200
195/195 [==============================] - ETA: 0s - loss: 0.0041 - Set_accuracy: 0.8777
Epoch 67: val_Set_accuracy improved from 0.79809 to 0.79935, saving model to /data1/yamazono/setRetrieval/DeepFurniture/output/models/best_model.weights.h5
195/195 [==============================] - 9s 48ms/step - loss: 0.0041 - Set_accuracy: 0.8777 - val_loss: 0.0058 - val_Set_accuracy: 0.7994 - lr: 6.2500e-05
Epoch 68/200
195/195 [==============================] - ETA: 0s - loss: 0.0041 - Set_accuracy: 0.8821 
Epoch 68: val_Set_accuracy did not improve from 0.79935
195/195 [==============================] - 10s 50ms/step - loss: 0.0041 - Set_accuracy: 0.8821 - val_loss: 0.0059 - val_Set_accuracy: 0.7959 - lr: 6.2500e-05
Epoch 69/200
194/195 [============================>.] - ETA: 0s - loss: 0.0041 - Set_accuracy: 0.8806
Epoch 69: val_Set_accuracy did not improve from 0.79935
195/195 [==============================] - 10s 50ms/step - loss: 0.0041 - Set_accuracy: 0.8810 - val_loss: 0.0057 - val_Set_accuracy: 0.7972 - lr: 6.2500e-05
Epoch 70/200
195/195 [==============================] - ETA: 0s - loss: 0.0041 - Set_accuracy: 0.8797 
Epoch 70: val_Set_accuracy improved from 0.79935 to 0.80025, saving model to /data1/yamazono/setRetrieval/DeepFurniture/output/models/best_model.weights.h5
195/195 [==============================] - 10s 50ms/step - loss: 0.0041 - Set_accuracy: 0.8797 - val_loss: 0.0057 - val_Set_accuracy: 0.8003 - lr: 6.2500e-05
Epoch 71/200
195/195 [==============================] - ETA: 0s - loss: 0.0040 - Set_accuracy: 0.8812 
Epoch 71: val_Set_accuracy did not improve from 0.80025
195/195 [==============================] - 10s 49ms/step - loss: 0.0040 - Set_accuracy: 0.8812 - val_loss: 0.0059 - val_Set_accuracy: 0.7994 - lr: 6.2500e-05
Epoch 72/200
195/195 [==============================] - ETA: 0s - loss: 0.0041 - Set_accuracy: 0.8848
Epoch 72: val_Set_accuracy did not improve from 0.80025
195/195 [==============================] - 9s 48ms/step - loss: 0.0041 - Set_accuracy: 0.8848 - val_loss: 0.0059 - val_Set_accuracy: 0.7959 - lr: 6.2500e-05
Epoch 73/200
195/195 [==============================] - ETA: 0s - loss: 0.0040 - Set_accuracy: 0.8847 
Epoch 73: val_Set_accuracy improved from 0.80025 to 0.80548, saving model to /data1/yamazono/setRetrieval/DeepFurniture/output/models/best_model.weights.h5
195/195 [==============================] - 10s 50ms/step - loss: 0.0040 - Set_accuracy: 0.8847 - val_loss: 0.0060 - val_Set_accuracy: 0.8055 - lr: 6.2500e-05
Epoch 74/200
194/195 [============================>.] - ETA: 0s - loss: 0.0040 - Set_accuracy: 0.8856
Epoch 74: val_Set_accuracy improved from 0.80548 to 0.80944, saving model to /data1/yamazono/setRetrieval/DeepFurniture/output/models/best_model.weights.h5
195/195 [==============================] - 10s 49ms/step - loss: 0.0040 - Set_accuracy: 0.8856 - val_loss: 0.0057 - val_Set_accuracy: 0.8094 - lr: 6.2500e-05
Epoch 75/200
195/195 [==============================] - ETA: 0s - loss: 0.0039 - Set_accuracy: 0.8873 
Epoch 75: val_Set_accuracy did not improve from 0.80944
195/195 [==============================] - 10s 49ms/step - loss: 0.0039 - Set_accuracy: 0.8873 - val_loss: 0.0059 - val_Set_accuracy: 0.8035 - lr: 6.2500e-05
Epoch 76/200
195/195 [==============================] - ETA: 0s - loss: 0.0039 - Set_accuracy: 0.8909 
Epoch 76: val_Set_accuracy did not improve from 0.80944
195/195 [==============================] - 9s 49ms/step - loss: 0.0039 - Set_accuracy: 0.8909 - val_loss: 0.0057 - val_Set_accuracy: 0.8039 - lr: 6.2500e-05
Epoch 77/200
194/195 [============================>.] - ETA: 0s - loss: 0.0038 - Set_accuracy: 0.8905 
Epoch 77: val_Set_accuracy did not improve from 0.80944
195/195 [==============================] - 10s 49ms/step - loss: 0.0038 - Set_accuracy: 0.8903 - val_loss: 0.0059 - val_Set_accuracy: 0.8019 - lr: 6.2500e-05
Epoch 78/200
194/195 [============================>.] - ETA: 0s - loss: 0.0039 - Set_accuracy: 0.8905
Epoch 78: val_Set_accuracy did not improve from 0.80944
195/195 [==============================] - 10s 49ms/step - loss: 0.0039 - Set_accuracy: 0.8907 - val_loss: 0.0058 - val_Set_accuracy: 0.8028 - lr: 6.2500e-05
Epoch 79/200
195/195 [==============================] - ETA: 0s - loss: 0.0038 - Set_accuracy: 0.8936
Epoch 79: val_Set_accuracy improved from 0.80944 to 0.81124, saving model to /data1/yamazono/setRetrieval/DeepFurniture/output/models/best_model.weights.h5
195/195 [==============================] - 10s 50ms/step - loss: 0.0038 - Set_accuracy: 0.8936 - val_loss: 0.0057 - val_Set_accuracy: 0.8112 - lr: 6.2500e-05
Epoch 80/200
195/195 [==============================] - ETA: 0s - loss: 0.0039 - Set_accuracy: 0.8943 
Epoch 80: val_Set_accuracy did not improve from 0.81124
195/195 [==============================] - 9s 48ms/step - loss: 0.0039 - Set_accuracy: 0.8943 - val_loss: 0.0057 - val_Set_accuracy: 0.8087 - lr: 6.2500e-05
Epoch 81/200
194/195 [============================>.] - ETA: 0s - loss: 0.0037 - Set_accuracy: 0.8944 
Epoch 81: val_Set_accuracy improved from 0.81124 to 0.81340, saving model to /data1/yamazono/setRetrieval/DeepFurniture/output/models/best_model.weights.h5
195/195 [==============================] - 10s 50ms/step - loss: 0.0037 - Set_accuracy: 0.8944 - val_loss: 0.0058 - val_Set_accuracy: 0.8134 - lr: 6.2500e-05
Epoch 82/200
195/195 [==============================] - ETA: 0s - loss: 0.0037 - Set_accuracy: 0.8963 
Epoch 82: val_Set_accuracy did not improve from 0.81340
195/195 [==============================] - 10s 49ms/step - loss: 0.0037 - Set_accuracy: 0.8963 - val_loss: 0.0058 - val_Set_accuracy: 0.8094 - lr: 6.2500e-05
Epoch 83/200
195/195 [==============================] - ETA: 0s - loss: 0.0038 - Set_accuracy: 0.8969 
Epoch 83: val_Set_accuracy did not improve from 0.81340
195/195 [==============================] - 9s 48ms/step - loss: 0.0038 - Set_accuracy: 0.8969 - val_loss: 0.0056 - val_Set_accuracy: 0.8078 - lr: 6.2500e-05
Epoch 84/200
195/195 [==============================] - ETA: 0s - loss: 0.0038 - Set_accuracy: 0.8942
Epoch 84: val_Set_accuracy did not improve from 0.81340
195/195 [==============================] - 9s 49ms/step - loss: 0.0038 - Set_accuracy: 0.8942 - val_loss: 0.0056 - val_Set_accuracy: 0.8116 - lr: 6.2500e-05
Epoch 85/200
195/195 [==============================] - ETA: 0s - loss: 0.0037 - Set_accuracy: 0.8970 
Epoch 85: val_Set_accuracy did not improve from 0.81340
195/195 [==============================] - 10s 49ms/step - loss: 0.0037 - Set_accuracy: 0.8970 - val_loss: 0.0057 - val_Set_accuracy: 0.8123 - lr: 6.2500e-05
Epoch 86/200
195/195 [==============================] - ETA: 0s - loss: 0.0037 - Set_accuracy: 0.8970
Epoch 86: val_Set_accuracy did not improve from 0.81340

Epoch 86: ReduceLROnPlateau reducing learning rate to 3.125000148429535e-05.
195/195 [==============================] - 9s 48ms/step - loss: 0.0037 - Set_accuracy: 0.8970 - val_loss: 0.0056 - val_Set_accuracy: 0.8089 - lr: 6.2500e-05
Epoch 87/200
195/195 [==============================] - ETA: 0s - loss: 0.0034 - Set_accuracy: 0.9011
Epoch 87: val_Set_accuracy improved from 0.81340 to 0.82006, saving model to /data1/yamazono/setRetrieval/DeepFurniture/output/models/best_model.weights.h5
195/195 [==============================] - 10s 49ms/step - loss: 0.0034 - Set_accuracy: 0.9011 - val_loss: 0.0054 - val_Set_accuracy: 0.8201 - lr: 3.1250e-05
Epoch 88/200
195/195 [==============================] - ETA: 0s - loss: 0.0034 - Set_accuracy: 0.9050 
Epoch 88: val_Set_accuracy improved from 0.82006 to 0.82061, saving model to /data1/yamazono/setRetrieval/DeepFurniture/output/models/best_model.weights.h5
195/195 [==============================] - 10s 51ms/step - loss: 0.0034 - Set_accuracy: 0.9050 - val_loss: 0.0055 - val_Set_accuracy: 0.8206 - lr: 3.1250e-05
Epoch 89/200
194/195 [============================>.] - ETA: 0s - loss: 0.0034 - Set_accuracy: 0.9085
Epoch 89: val_Set_accuracy did not improve from 0.82061
195/195 [==============================] - 10s 49ms/step - loss: 0.0034 - Set_accuracy: 0.9083 - val_loss: 0.0053 - val_Set_accuracy: 0.8184 - lr: 3.1250e-05
Epoch 90/200
195/195 [==============================] - ETA: 0s - loss: 0.0034 - Set_accuracy: 0.9051
Epoch 90: val_Set_accuracy did not improve from 0.82061
195/195 [==============================] - 9s 49ms/step - loss: 0.0034 - Set_accuracy: 0.9051 - val_loss: 0.0054 - val_Set_accuracy: 0.8197 - lr: 3.1250e-05
Epoch 91/200
195/195 [==============================] - ETA: 0s - loss: 0.0033 - Set_accuracy: 0.9043
Epoch 91: val_Set_accuracy improved from 0.82061 to 0.82205, saving model to /data1/yamazono/setRetrieval/DeepFurniture/output/models/best_model.weights.h5
195/195 [==============================] - 10s 49ms/step - loss: 0.0033 - Set_accuracy: 0.9043 - val_loss: 0.0055 - val_Set_accuracy: 0.8220 - lr: 3.1250e-05
Epoch 92/200
195/195 [==============================] - ETA: 0s - loss: 0.0034 - Set_accuracy: 0.9045 
Epoch 92: val_Set_accuracy improved from 0.82205 to 0.82259, saving model to /data1/yamazono/setRetrieval/DeepFurniture/output/models/best_model.weights.h5
195/195 [==============================] - 10s 49ms/step - loss: 0.0034 - Set_accuracy: 0.9045 - val_loss: 0.0054 - val_Set_accuracy: 0.8226 - lr: 3.1250e-05
Epoch 93/200
194/195 [============================>.] - ETA: 0s - loss: 0.0034 - Set_accuracy: 0.9097 
Epoch 93: val_Set_accuracy did not improve from 0.82259
195/195 [==============================] - 10s 50ms/step - loss: 0.0034 - Set_accuracy: 0.9096 - val_loss: 0.0054 - val_Set_accuracy: 0.8184 - lr: 3.1250e-05
Epoch 94/200
195/195 [==============================] - ETA: 0s - loss: 0.0033 - Set_accuracy: 0.9122 
Epoch 94: val_Set_accuracy did not improve from 0.82259
195/195 [==============================] - 10s 50ms/step - loss: 0.0033 - Set_accuracy: 0.9122 - val_loss: 0.0054 - val_Set_accuracy: 0.8192 - lr: 3.1250e-05
Epoch 95/200
194/195 [============================>.] - ETA: 0s - loss: 0.0034 - Set_accuracy: 0.9081
Epoch 95: val_Set_accuracy improved from 0.82259 to 0.82583, saving model to /data1/yamazono/setRetrieval/DeepFurniture/output/models/best_model.weights.h5
195/195 [==============================] - 10s 50ms/step - loss: 0.0034 - Set_accuracy: 0.9082 - val_loss: 0.0054 - val_Set_accuracy: 0.8258 - lr: 3.1250e-05
Epoch 96/200
194/195 [============================>.] - ETA: 0s - loss: 0.0034 - Set_accuracy: 0.9065 
Epoch 96: val_Set_accuracy did not improve from 0.82583
195/195 [==============================] - 10s 50ms/step - loss: 0.0034 - Set_accuracy: 0.9064 - val_loss: 0.0053 - val_Set_accuracy: 0.8238 - lr: 3.1250e-05
Epoch 97/200
195/195 [==============================] - ETA: 0s - loss: 0.0033 - Set_accuracy: 0.9102
Epoch 97: val_Set_accuracy did not improve from 0.82583
195/195 [==============================] - 10s 49ms/step - loss: 0.0033 - Set_accuracy: 0.9102 - val_loss: 0.0054 - val_Set_accuracy: 0.8217 - lr: 3.1250e-05
Epoch 98/200
195/195 [==============================] - ETA: 0s - loss: 0.0033 - Set_accuracy: 0.9131 
Epoch 98: val_Set_accuracy did not improve from 0.82583
195/195 [==============================] - 9s 49ms/step - loss: 0.0033 - Set_accuracy: 0.9131 - val_loss: 0.0053 - val_Set_accuracy: 0.8242 - lr: 3.1250e-05
Epoch 99/200
195/195 [==============================] - ETA: 0s - loss: 0.0033 - Set_accuracy: 0.9117 
Epoch 99: val_Set_accuracy did not improve from 0.82583
195/195 [==============================] - 9s 49ms/step - loss: 0.0033 - Set_accuracy: 0.9117 - val_loss: 0.0053 - val_Set_accuracy: 0.8238 - lr: 3.1250e-05
Epoch 100/200
194/195 [============================>.] - ETA: 0s - loss: 0.0033 - Set_accuracy: 0.9119 
Epoch 100: val_Set_accuracy improved from 0.82583 to 0.82691, saving model to /data1/yamazono/setRetrieval/DeepFurniture/output/models/best_model.weights.h5
195/195 [==============================] - 10s 49ms/step - loss: 0.0033 - Set_accuracy: 0.9119 - val_loss: 0.0053 - val_Set_accuracy: 0.8269 - lr: 3.1250e-05
Epoch 101/200
194/195 [============================>.] - ETA: 0s - loss: 0.0033 - Set_accuracy: 0.9096 
Epoch 101: val_Set_accuracy did not improve from 0.82691
195/195 [==============================] - 10s 50ms/step - loss: 0.0033 - Set_accuracy: 0.9095 - val_loss: 0.0054 - val_Set_accuracy: 0.8237 - lr: 3.1250e-05
Epoch 102/200
195/195 [==============================] - ETA: 0s - loss: 0.0033 - Set_accuracy: 0.9127
Epoch 102: val_Set_accuracy did not improve from 0.82691
195/195 [==============================] - 10s 49ms/step - loss: 0.0033 - Set_accuracy: 0.9127 - val_loss: 0.0055 - val_Set_accuracy: 0.8183 - lr: 3.1250e-05
Epoch 103/200
195/195 [==============================] - ETA: 0s - loss: 0.0032 - Set_accuracy: 0.9128
Epoch 103: val_Set_accuracy improved from 0.82691 to 0.82745, saving model to /data1/yamazono/setRetrieval/DeepFurniture/output/models/best_model.weights.h5
195/195 [==============================] - 9s 49ms/step - loss: 0.0032 - Set_accuracy: 0.9128 - val_loss: 0.0056 - val_Set_accuracy: 0.8274 - lr: 3.1250e-05
Epoch 104/200
195/195 [==============================] - ETA: 0s - loss: 0.0033 - Set_accuracy: 0.9121
Epoch 104: val_Set_accuracy did not improve from 0.82745
195/195 [==============================] - 9s 48ms/step - loss: 0.0033 - Set_accuracy: 0.9121 - val_loss: 0.0055 - val_Set_accuracy: 0.8247 - lr: 3.1250e-05
Epoch 105/200
194/195 [============================>.] - ETA: 0s - loss: 0.0032 - Set_accuracy: 0.9128
Epoch 105: val_Set_accuracy did not improve from 0.82745
195/195 [==============================] - 10s 51ms/step - loss: 0.0032 - Set_accuracy: 0.9128 - val_loss: 0.0053 - val_Set_accuracy: 0.8258 - lr: 3.1250e-05
Epoch 106/200
194/195 [============================>.] - ETA: 0s - loss: 0.0033 - Set_accuracy: 0.9111 
Epoch 106: val_Set_accuracy did not improve from 0.82745
195/195 [==============================] - 10s 51ms/step - loss: 0.0033 - Set_accuracy: 0.9114 - val_loss: 0.0054 - val_Set_accuracy: 0.8242 - lr: 3.1250e-05
Epoch 107/200
195/195 [==============================] - ETA: 0s - loss: 0.0032 - Set_accuracy: 0.9124 
Epoch 107: val_Set_accuracy did not improve from 0.82745
195/195 [==============================] - 10s 49ms/step - loss: 0.0032 - Set_accuracy: 0.9124 - val_loss: 0.0054 - val_Set_accuracy: 0.8264 - lr: 3.1250e-05
Epoch 108/200
195/195 [==============================] - ETA: 0s - loss: 0.0032 - Set_accuracy: 0.9153 
Epoch 108: val_Set_accuracy improved from 0.82745 to 0.83231, saving model to /data1/yamazono/setRetrieval/DeepFurniture/output/models/best_model.weights.h5
195/195 [==============================] - 10s 50ms/step - loss: 0.0032 - Set_accuracy: 0.9153 - val_loss: 0.0053 - val_Set_accuracy: 0.8323 - lr: 3.1250e-05
Epoch 109/200
194/195 [============================>.] - ETA: 0s - loss: 0.0032 - Set_accuracy: 0.9139
Epoch 109: val_Set_accuracy did not improve from 0.83231
195/195 [==============================] - 10s 49ms/step - loss: 0.0032 - Set_accuracy: 0.9139 - val_loss: 0.0054 - val_Set_accuracy: 0.8298 - lr: 3.1250e-05
Epoch 110/200
195/195 [==============================] - ETA: 0s - loss: 0.0032 - Set_accuracy: 0.9142
Epoch 110: val_Set_accuracy did not improve from 0.83231
195/195 [==============================] - 10s 49ms/step - loss: 0.0032 - Set_accuracy: 0.9142 - val_loss: 0.0054 - val_Set_accuracy: 0.8296 - lr: 3.1250e-05
Epoch 111/200
195/195 [==============================] - ETA: 0s - loss: 0.0032 - Set_accuracy: 0.9130 
Epoch 111: val_Set_accuracy did not improve from 0.83231
195/195 [==============================] - 10s 49ms/step - loss: 0.0032 - Set_accuracy: 0.9130 - val_loss: 0.0053 - val_Set_accuracy: 0.8302 - lr: 3.1250e-05
Epoch 112/200
195/195 [==============================] - ETA: 0s - loss: 0.0032 - Set_accuracy: 0.9139 
Epoch 112: val_Set_accuracy did not improve from 0.83231
195/195 [==============================] - 10s 49ms/step - loss: 0.0032 - Set_accuracy: 0.9139 - val_loss: 0.0053 - val_Set_accuracy: 0.8269 - lr: 3.1250e-05
Epoch 113/200
195/195 [==============================] - ETA: 0s - loss: 0.0032 - Set_accuracy: 0.9160
Epoch 113: val_Set_accuracy did not improve from 0.83231

Epoch 113: ReduceLROnPlateau reducing learning rate to 1.5625000742147677e-05.
195/195 [==============================] - 10s 50ms/step - loss: 0.0032 - Set_accuracy: 0.9160 - val_loss: 0.0053 - val_Set_accuracy: 0.8314 - lr: 3.1250e-05
Epoch 114/200
195/195 [==============================] - ETA: 0s - loss: 0.0030 - Set_accuracy: 0.9168 
Epoch 114: val_Set_accuracy improved from 0.83231 to 0.83501, saving model to /data1/yamazono/setRetrieval/DeepFurniture/output/models/best_model.weights.h5
195/195 [==============================] - 10s 49ms/step - loss: 0.0030 - Set_accuracy: 0.9168 - val_loss: 0.0052 - val_Set_accuracy: 0.8350 - lr: 1.5625e-05
Epoch 115/200
195/195 [==============================] - ETA: 0s - loss: 0.0030 - Set_accuracy: 0.9156 
Epoch 115: val_Set_accuracy did not improve from 0.83501
195/195 [==============================] - 10s 49ms/step - loss: 0.0030 - Set_accuracy: 0.9156 - val_loss: 0.0052 - val_Set_accuracy: 0.8325 - lr: 1.5625e-05
Epoch 116/200
194/195 [============================>.] - ETA: 0s - loss: 0.0030 - Set_accuracy: 0.9208
Epoch 116: val_Set_accuracy improved from 0.83501 to 0.83573, saving model to /data1/yamazono/setRetrieval/DeepFurniture/output/models/best_model.weights.h5
195/195 [==============================] - 10s 50ms/step - loss: 0.0030 - Set_accuracy: 0.9206 - val_loss: 0.0052 - val_Set_accuracy: 0.8357 - lr: 1.5625e-05
Epoch 117/200
194/195 [============================>.] - ETA: 0s - loss: 0.0030 - Set_accuracy: 0.9203
Epoch 117: val_Set_accuracy improved from 0.83573 to 0.83591, saving model to /data1/yamazono/setRetrieval/DeepFurniture/output/models/best_model.weights.h5
195/195 [==============================] - 10s 49ms/step - loss: 0.0030 - Set_accuracy: 0.9204 - val_loss: 0.0052 - val_Set_accuracy: 0.8359 - lr: 1.5625e-05
Epoch 118/200
195/195 [==============================] - ETA: 0s - loss: 0.0030 - Set_accuracy: 0.9201 
Epoch 118: val_Set_accuracy did not improve from 0.83591
195/195 [==============================] - 10s 49ms/step - loss: 0.0030 - Set_accuracy: 0.9201 - val_loss: 0.0052 - val_Set_accuracy: 0.8345 - lr: 1.5625e-05
Epoch 119/200
194/195 [============================>.] - ETA: 0s - loss: 0.0031 - Set_accuracy: 0.9212
Epoch 119: val_Set_accuracy did not improve from 0.83591
195/195 [==============================] - 10s 50ms/step - loss: 0.0031 - Set_accuracy: 0.9213 - val_loss: 0.0052 - val_Set_accuracy: 0.8354 - lr: 1.5625e-05
Epoch 120/200
194/195 [============================>.] - ETA: 0s - loss: 0.0031 - Set_accuracy: 0.9180 
Epoch 120: val_Set_accuracy did not improve from 0.83591
195/195 [==============================] - 10s 52ms/step - loss: 0.0031 - Set_accuracy: 0.9179 - val_loss: 0.0052 - val_Set_accuracy: 0.8338 - lr: 1.5625e-05
Epoch 121/200
194/195 [============================>.] - ETA: 0s - loss: 0.0031 - Set_accuracy: 0.9181 
Epoch 121: val_Set_accuracy did not improve from 0.83591
195/195 [==============================] - 10s 50ms/step - loss: 0.0031 - Set_accuracy: 0.9181 - val_loss: 0.0052 - val_Set_accuracy: 0.8348 - lr: 1.5625e-05
Epoch 122/200
194/195 [============================>.] - ETA: 0s - loss: 0.0030 - Set_accuracy: 0.9181 
Epoch 122: val_Set_accuracy did not improve from 0.83591

Epoch 122: ReduceLROnPlateau reducing learning rate to 7.812500371073838e-06.
195/195 [==============================] - 10s 51ms/step - loss: 0.0030 - Set_accuracy: 0.9184 - val_loss: 0.0051 - val_Set_accuracy: 0.8336 - lr: 1.5625e-05
Epoch 123/200
194/195 [============================>.] - ETA: 0s - loss: 0.0029 - Set_accuracy: 0.9223 
Epoch 123: val_Set_accuracy did not improve from 0.83591
195/195 [==============================] - 10s 49ms/step - loss: 0.0029 - Set_accuracy: 0.9222 - val_loss: 0.0051 - val_Set_accuracy: 0.8332 - lr: 7.8125e-06
Epoch 124/200
195/195 [==============================] - ETA: 0s - loss: 0.0029 - Set_accuracy: 0.9214 
Epoch 124: val_Set_accuracy did not improve from 0.83591
195/195 [==============================] - 10s 49ms/step - loss: 0.0029 - Set_accuracy: 0.9214 - val_loss: 0.0051 - val_Set_accuracy: 0.8347 - lr: 7.8125e-06
Epoch 125/200
194/195 [============================>.] - ETA: 0s - loss: 0.0029 - Set_accuracy: 0.9204
Epoch 125: val_Set_accuracy improved from 0.83591 to 0.83682, saving model to /data1/yamazono/setRetrieval/DeepFurniture/output/models/best_model.weights.h5
195/195 [==============================] - 10s 49ms/step - loss: 0.0029 - Set_accuracy: 0.9202 - val_loss: 0.0051 - val_Set_accuracy: 0.8368 - lr: 7.8125e-06
Epoch 126/200
194/195 [============================>.] - ETA: 0s - loss: 0.0029 - Set_accuracy: 0.9224
Epoch 126: val_Set_accuracy did not improve from 0.83682
195/195 [==============================] - 10s 50ms/step - loss: 0.0029 - Set_accuracy: 0.9226 - val_loss: 0.0051 - val_Set_accuracy: 0.8343 - lr: 7.8125e-06
Epoch 127/200
195/195 [==============================] - ETA: 0s - loss: 0.0029 - Set_accuracy: 0.9207
Epoch 127: val_Set_accuracy did not improve from 0.83682
195/195 [==============================] - 10s 49ms/step - loss: 0.0029 - Set_accuracy: 0.9207 - val_loss: 0.0051 - val_Set_accuracy: 0.8363 - lr: 7.8125e-06
Epoch 128/200
194/195 [============================>.] - ETA: 0s - loss: 0.0029 - Set_accuracy: 0.9239
Epoch 128: val_Set_accuracy improved from 0.83682 to 0.83826, saving model to /data1/yamazono/setRetrieval/DeepFurniture/output/models/best_model.weights.h5
195/195 [==============================] - 10s 50ms/step - loss: 0.0029 - Set_accuracy: 0.9239 - val_loss: 0.0051 - val_Set_accuracy: 0.8383 - lr: 7.8125e-06
Epoch 129/200
195/195 [==============================] - ETA: 0s - loss: 0.0030 - Set_accuracy: 0.9196 
Epoch 129: val_Set_accuracy improved from 0.83826 to 0.83844, saving model to /data1/yamazono/setRetrieval/DeepFurniture/output/models/best_model.weights.h5
195/195 [==============================] - 10s 49ms/step - loss: 0.0030 - Set_accuracy: 0.9196 - val_loss: 0.0051 - val_Set_accuracy: 0.8384 - lr: 7.8125e-06
Epoch 130/200
195/195 [==============================] - ETA: 0s - loss: 0.0029 - Set_accuracy: 0.9232 
Epoch 130: val_Set_accuracy did not improve from 0.83844
195/195 [==============================] - 10s 49ms/step - loss: 0.0029 - Set_accuracy: 0.9232 - val_loss: 0.0051 - val_Set_accuracy: 0.8366 - lr: 7.8125e-06
Epoch 131/200
195/195 [==============================] - ETA: 0s - loss: 0.0029 - Set_accuracy: 0.9213 
Epoch 131: val_Set_accuracy did not improve from 0.83844
195/195 [==============================] - 10s 49ms/step - loss: 0.0029 - Set_accuracy: 0.9213 - val_loss: 0.0051 - val_Set_accuracy: 0.8365 - lr: 7.8125e-06
Epoch 132/200
195/195 [==============================] - ETA: 0s - loss: 0.0029 - Set_accuracy: 0.9217 
Epoch 132: val_Set_accuracy did not improve from 0.83844
195/195 [==============================] - 10s 49ms/step - loss: 0.0029 - Set_accuracy: 0.9217 - val_loss: 0.0051 - val_Set_accuracy: 0.8377 - lr: 7.8125e-06
Epoch 133/200
195/195 [==============================] - ETA: 0s - loss: 0.0029 - Set_accuracy: 0.9251 
Epoch 133: val_Set_accuracy did not improve from 0.83844
195/195 [==============================] - 10s 49ms/step - loss: 0.0029 - Set_accuracy: 0.9251 - val_loss: 0.0051 - val_Set_accuracy: 0.8383 - lr: 7.8125e-06
Epoch 134/200
194/195 [============================>.] - ETA: 0s - loss: 0.0029 - Set_accuracy: 0.9256
Epoch 134: val_Set_accuracy did not improve from 0.83844

Epoch 134: ReduceLROnPlateau reducing learning rate to 3.906250185536919e-06.
195/195 [==============================] - 10s 50ms/step - loss: 0.0029 - Set_accuracy: 0.9255 - val_loss: 0.0051 - val_Set_accuracy: 0.8379 - lr: 7.8125e-06
Epoch 135/200
194/195 [============================>.] - ETA: 0s - loss: 0.0029 - Set_accuracy: 0.9258 
Epoch 135: val_Set_accuracy did not improve from 0.83844
195/195 [==============================] - 10s 49ms/step - loss: 0.0029 - Set_accuracy: 0.9258 - val_loss: 0.0051 - val_Set_accuracy: 0.8368 - lr: 3.9063e-06
Epoch 136/200
194/195 [============================>.] - ETA: 0s - loss: 0.0029 - Set_accuracy: 0.9211
Epoch 136: val_Set_accuracy did not improve from 0.83844
195/195 [==============================] - 10s 50ms/step - loss: 0.0029 - Set_accuracy: 0.9211 - val_loss: 0.0051 - val_Set_accuracy: 0.8383 - lr: 3.9063e-06
Epoch 137/200
195/195 [==============================] - ETA: 0s - loss: 0.0029 - Set_accuracy: 0.9237 
Epoch 137: val_Set_accuracy did not improve from 0.83844
195/195 [==============================] - 10s 50ms/step - loss: 0.0029 - Set_accuracy: 0.9237 - val_loss: 0.0051 - val_Set_accuracy: 0.8374 - lr: 3.9063e-06
Epoch 138/200
195/195 [==============================] - ETA: 0s - loss: 0.0028 - Set_accuracy: 0.9254 
Epoch 138: val_Set_accuracy did not improve from 0.83844
195/195 [==============================] - 10s 49ms/step - loss: 0.0028 - Set_accuracy: 0.9254 - val_loss: 0.0051 - val_Set_accuracy: 0.8381 - lr: 3.9063e-06
Epoch 139/200
195/195 [==============================] - ETA: 0s - loss: 0.0028 - Set_accuracy: 0.9237Restoring model weights from the end of the best epoch: 129.

Epoch 139: val_Set_accuracy did not improve from 0.83844

Epoch 139: ReduceLROnPlateau reducing learning rate to 1.9531250927684596e-06.
195/195 [==============================] - 10s 49ms/step - loss: 0.0028 - Set_accuracy: 0.9237 - val_loss: 0.0051 - val_Set_accuracy: 0.8377 - lr: 3.9063e-06
Epoch 139: early stopping
"""
]


# Parse history and extract data
history_text = history[0]
epochs = []
loss = []
val_loss = []
accuracy = []
val_accuracy = []

for line in history_text.split('\n'):
    # Extract epoch number
    epoch_match = re.search(r'Epoch (\d+)/\d+', line)
    if epoch_match:
        current_epoch = int(epoch_match.group(1))
        epochs.append(current_epoch)
    
    # Extract metrics
    loss_match = re.search(r'loss: ([\d.]+)', line)
    val_loss_match = re.search(r'val_loss: ([\d.]+)', line)
    accuracy_match = re.search(r'Set_accuracy: ([\d.]+)', line)
    val_accuracy_match = re.search(r'val_Set_accuracy: ([\d.]+)', line)

    if loss_match:
        loss.append(float(loss_match.group(1)))
    if val_loss_match:
        val_loss.append(float(val_loss_match.group(1)))
    if accuracy_match:
        accuracy.append(float(accuracy_match.group(1)))
    if val_accuracy_match:
        val_accuracy.append(float(val_accuracy_match.group(1)))

# Remove duplicate entries
def remove_duplicates(lst):
    return [lst[i] for i in range(len(lst)) if i % 2 == 0]

# Apply the function to metrics
loss = remove_duplicates(loss)
val_loss = remove_duplicates(val_loss)
accuracy = remove_duplicates(accuracy)
val_accuracy = remove_duplicates(val_accuracy)

# Ensure epochs matches the adjusted metrics
epochs = epochs[:len(loss)]  # Align epochs with the corrected loss
val_epochs = epochs[:len(val_loss)]  # Align epochs with the corrected val_loss

# Plot the data
plt.figure(figsize=(14, 6))

# Loss Plot
plt.subplot(1, 2, 1)
plt.plot(epochs, loss, label='Training Loss')
plt.plot(val_epochs, val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

# Accuracy Plot
plt.subplot(1, 2, 2)
accuracy_epochs = epochs[:len(accuracy)]  # Align with corrected accuracy
val_accuracy_epochs = epochs[:len(val_accuracy)]  # Align with corrected val_accuracy
plt.plot(accuracy_epochs, accuracy, label='Training Accuracy')
plt.plot(val_accuracy_epochs, val_accuracy, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(f"loss_acc_50.png"))
plt.close()