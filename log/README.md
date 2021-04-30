# Experimental Logs

## Experiment #1: No. of Convolution Layers in CNN

| No. of Conv Blocks in CNN | Log File Name              |
| ------------------------- | -------------------------- |
| 2                         | `cnn_lstm_512_2_1_512.log` |
| 3                         | `cnn_lstm_512_3_1_512.log` |
| 4                         | `cnn_lstm_512_4_1_512.log` |
| 5                         | `cnn_lstm_512_5_1_512.log` |
| 6                         | `cnn_lstm_512_6_1_512.log` |

## Experiment #2: Dropout Rate of Latent Vector

| Dropout Rate | Log File Name                     |
| ------------ | --------------------------------- |
| 0.1          | `cnn_lstm_512_6_1_512.log`        |
| 0.25         | `cnn_lstm_512_6_1_512_drop25.log` |
| 0.50         | `cnn_lstm_512_6_1_512_drop50.log` |
| 0.80         | `cnn_lstm_512_6_1_512_drop80.log` |

## Experiment #3: Weight Decay of Optimizer

| Weight Decay | Log File Name                              |
| ------------ | ------------------------------------------ |
| 1e-5         | `cnn_lstm_512_6_1_512_drop80_weightd5.log` |
| 1e-6         | `cnn_lstm_512_6_1_512_drop80_weightd6.log` |
| 1e-7         | `cnn_lstm_512_6_1_512_drop80_weightd7.log` |
| 1e-8         | `cnn_lstm_512_6_1_512_drop80_weightd8.log` |
| 0            | `cnn_lstm_512_6_1_512_drop80.log`          |
