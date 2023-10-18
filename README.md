# Face Detection and Recognition with [Paddle](https://github.com/PaddlePaddle)

<div align="center">

**|** 👀[**Demos**](#-demos-videos) **|** ⚡[**Usage**](#-quick-inference) **|** 🔧[**Install**](#-dependencies-and-installation) **|** 💻[**Train**](#-train) **|**

</div>

## Introduction

Face recognition is a way of identifying or confirming an individual’s identity using their face. Facial recognition systems can be used to identify people in photos, videos, or in real-time. It is a category of biometric security. Other forms of biometric software include voice recognition, fingerprint recognition, and eye retina or iris recognition. The technology is mostly used for security and law enforcement, though there is increasing interest in other areas of use.

## 👀 Demos Videos

### Streamlit

[Streamlit](https://github.com/streamlit/streamlit) lets you turn data scripts into shareable web apps in minutes, not weeks. It’s all Python, open-source, and free! And once you’ve created an app you can use Streamlit Community Cloud platform to deploy, manage, and share your app.

You can run following command to use streamlit in our case:

```bash
  streamlit run stream.py
```

The result is as follows:

<img src="face_rec.gif">

## ⚡ Quick Inference

First, you must index file to recognize and labeled people. To do this you can use following:

```console
  Usage: python inference/insightface.py --build_index index.bin --img_dir labels/images --label labels/images/label.txt [options]...

    --build_index           Path to index file.
    --img_dir               Path to label images folder.
    --label                 Path to labels text file.
```

In the above command --img_dir must contain images with any people to recognize. Images should contain one person. And --label file should contain as following:

```
path_to_image(tab)label

for example:
salom/1.jpg Assalom
valom/3.jpg Vassalom
...
```

Then, you recognized with following command:

```console
  Usage: python inference/insightface.py --index index.bin --rec --rec_model models/Recognition --det --det_model models/Detection --input some.jpg --output output/salom.jpg [options]...

    --index                 Path to index file.
    --rec                   Face Recognition if you want.
    --rec_model             Path to recognition model.
    --det                   Face Detection if you want.
    --det_model             Path to detection model.
    --input                 Path to input. It can be video or image.
    --output                Path to output file or folder.
```

For more information [link](https://github.com/littletomatodonkey/insight-face-paddle). We get codes from this repository.

## 🔧 Dependencies and Installation

### Installation

If you have GPU:

1. Install nvidia-docker and paddle as follows:

   ```bash
   nvidia-docker pull registry.baidubce.com/paddlepaddle/paddle:2.2.0rc0-gpu-cuda11.2-cudnn8
   nvidia-docker run --name paddleface -it -v $PWD:/paddle registry.baidubce.com/paddlepaddle/paddle:2.2.0rc0-gpu-cuda11.2-cudnn8 /bin/bash
   ```

2. Clone repo and install requirements:

   ```bash
   git clone https://github.com/shoxa0707/Face-Recognition-with-Paddle.git
   cd Face-Recognition-with-Paddle
   pip install -r requirements.txt
   ```

If you don't have GPU:

1. Install nvidia-docker and paddle as follows:

   ```bash
   docker pull paddlepaddle/paddle:2.2.2
   docker run --name paddle_docker -it -v $PWD:/paddle registry.baidubce.com/paddlepaddle/paddle:2.2.2 /bin/bash
   ```

2. Clone repo and install requirements:

   ```bash
   git clone https://github.com/shoxa0707/Face-Recognition-with-Paddle.git
   cd Face-Recognition-with-Paddle
   pip install -r requirements.txt
   ```

## 💻 Train

To train paddle face recognition this [link](https://github.com/deepinsight/insightface/blob/master/recognition/arcface_paddle/README_en.md) is helpful (we also trained according to this repository).

First, download dataset and prepare to train as following python command:

```console
  Usage: python train/extract_images.py --root_dir dataset --output_dir out_dataset

    --root_dir              Root directory to mxnet dataset.
    --output_dir            Output dataset path.
```

Note. Before using following commands, you should modify them to suit your situation.
Then train with following command:

1. Static mode:

   ```bash
     sh train/arcface_paddle/scripts/train_static.sh
   ```

2. Dynamic mode:

   ```bash
     sh train/arcface_paddle/scripts/train_dynamic.sh
   ```

To export to paddle model format:

1. Static mode:

   ```bash
     sh train/arcface_paddle/scripts/export_static.sh
   ```

2. Dynamic mode:

   ```bash
     sh train/arcface_paddle/scripts/export_dynamic.sh
   ```

You can

# Requirements

- Linux
- Python 3.7
- NVIDIA GPU
- Docker
