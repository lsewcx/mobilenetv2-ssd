# SSD

> [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325) 

<!-- [ALGORITHM] -->

## 摘要

我们提出了一种使用单一深度神经网络检测图像中的对象的方法。我们的方法，名为SSD，将边界框的输出空间离散化为在每个特征图位置上具有不同长宽比和尺度的一组默认框。在预测时，网络为每个默认框中每个对象类别的存在生成分数，并产生调整以更好地匹配对象形状。此外，网络结合了具有不同分辨率的多个特征图的预测，自然处理各种大小的对象。我们的SSD模型比需要对象提议的方法简单，因为它完全消除了提议生成和后续像素或特征重采样阶段，并将所有计算封装在单个网络中。这使得SSD易于训练，并且可以轻松集成到需要检测组件的系统中。在PASCAL VOC、MS COCO和ILSVRC数据集上的实验结果证实，SSD在准确性上与使用额外对象提议步骤的方法相当，并且速度快得多，同时为训练和推理提供了统一的框架。与其他单级方法相比，即使输入图像尺寸较小，SSD也具有更好的准确性。对于300×300的输入，在Nvidia Titan X上VOC2007测试的mAP达到72.1%，58 FPS；对于500×500的输入，mAP达到75.1%，超过了相当的最先进Faster R-CNN模型。

<div align=center>
<img src="https://user-images.githubusercontent.com/40661020/143998553-4e12f681-6025-46b4-8410-9e2e1e53a8ec.png"/ >
</div>

## SSD的结果和模型

| 骨干网络 | 尺寸 | 风格 | Lr schd | 内存 (GB) | 推理时间 (fps) | 框 AP |           配置           |                                                                                                            下载                                                                                                             |
| :------: | :--: | :---: | :-----: | :------: | :------------: | :----: | :------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|  VGG16   | 300  | caffe |  120e   |   9.9    |      43.7      |  25.5  | [配置](./ssd300_coco.py) | [模型](https://download.openmmlab.com/mmdetection/v2.0/ssd/ssd300_coco/ssd300_coco_20210803_015428-d231a06e.pth)  \| [日志](https://download.openmmlab.com/mmdetection/v2.0/ssd/ssd300_coco/ssd300_coco_20210803_015428.log.json)  |
|  VGG16   | 512  | caffe |  120e   |   19.4   |      30.7      |  29.5  | [配置](./ssd512_coco.py) | [模型](https://download.openmmlab.com/mmdetection/v2.0/ssd/ssd512_coco/ssd512_coco_20210803_022849-0a47a1ca.pth)  \| [日志](https://download.openmmlab.com/mmdetection/v2.0/ssd/ssd512_coco/ssd512_coco_20210803_022849.log.json)  |

## SSD-Lite的结果和模型

|  骨干网络   | 尺寸 | 从头开始训练 | Lr schd | 内存 (GB) | 推理时间 (fps) | 框 AP |                           配置                           |                                                                                                                                                                下载                                                                                                                                                                 |
| :---------: | :--: | :-------------------: | :-----: | :------: | :------------: | :----: | :--------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| MobileNetV2 | 320  |          yes          |  600e   |   4.0    |      69.9      |  21.3  | [配置](./ssdlite_mobilenetv2-scratch_8xb24-600e_coco.py) | [模型](https://download.openmmlab.com/mmdetection/v2.0/ssd/ssdlite_mobilenetv2_scratch_600e_coco/ssdlite_mobilenetv2_scratch_600e_coco_20210629_110627-974d9307.pth)  \| [日志](https://download.openmmlab.com/mmdetection/v2.0/ssd/ssdlite_mobilenetv2_scratch_600e_coco/ssdlite_mobilenetv2_scratch_600e_coco_20210629_110627.log.json)  |

## 注意

### 兼容性

在v2.14.0中，[PR5291](https://github.com/open-mmlab/mmdetection/pull/5291) 重构了SSD的颈部和头部，以便更灵活地使用。如果用户想使用旧版本中训练的SSD检查点，我们提供了脚本`tools/model_converters/upgrade_ssd_version.py`来转换模型权重。

```bash
python tools/model_converters/upgrade_ssd_version.py ${OLD_MODEL_PATH} ${NEW_MODEL_PATH}

```

- OLD_MODEL_PATH: 要加载的旧版本SSD模型的路径。
- NEW_MODEL_PATH: 要保存的转换后的模型权重的路径。

### SSD-Lite训练设置

我们的MobileNetV2 SSD-Lite实现与[TensorFlow 1.x检测模型动物园](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md)中的实现有一些不同。

1. 使用320x320作为输入尺寸，而不是300x300。
2. 锚点尺寸不同。
3. C4特征图取自阶段4的最后一层，而不是块的中间。
4. TensorFlow1.x中的模型是在coco 2014上训练的，并在coco minival2014上验证的，但我们的模型是在coco 2017上训练和验证的。val2017的mAP通常比minival2014略低（参考TensorFlow Object Detection API中的结果，例如，MobileNetV2 SSD在minival2014上得到22 mAP，但在val2017上为20.2 mAP）。

## 引用

```latex
@article{Liu_2016,
   title={SSD: Single Shot MultiBox Detector},
   journal={ECCV},
   author={Liu, Wei and Anguelov, Dragomir and Erhan, Dumitru and Szegedy, Christian and Reed, Scott and Fu, Cheng-Yang and Berg, Alexander C.},
   year={2016},
}
```
