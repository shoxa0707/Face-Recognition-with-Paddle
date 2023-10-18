import streamlit as st
import os
import logging
import imghdr
import pickle
from functools import partial

import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import paddle
from paddle.inference import Config
from paddle.inference import create_predictor
import re
import tempfile
import time

########################################################################################################################
############################################# Recognition part #########################################################
########################################################################################################################

def check_model_file(model):
    """Check the model files exist and download and untar when no exist.
    """
    if os.path.isdir(model):
        model_file_path = os.path.join(model, "inference.pdmodel")
        params_file_path = os.path.join(model, "inference.pdiparams")
        if not os.path.exists(model_file_path) or not os.path.exists(
                params_file_path):
            raise Exception(f"The specifed model directory error. The drectory must include \"inference.pdmodel\" and \"inference.pdiparams\".")

    return model_file_path, params_file_path


def normalize_image(img, scale=None, mean=None, std=None, order="chw"):
    if isinstance(scale, str):
        scale = eval(scale)
    scale = np.float32(scale if scale is not None else 1.0 / 255.0)
    mean = mean if mean is not None else [0.485, 0.456, 0.406]
    std = std if std is not None else [0.229, 0.224, 0.225]

    shape = (3, 1, 1) if order == "chw" else (1, 1, 3)
    mean = np.array(mean).reshape(shape).astype("float32")
    std = np.array(std).reshape(shape).astype("float32")

    if isinstance(img, Image.Image):
        img = np.array(img)

    assert isinstance(img, np.ndarray), "invalid input \"img\" in NormalizeImage"
    return (img.astype("float32") * scale - mean) / std


def to_CHW_image(img):
    if isinstance(img, Image.Image):
        img = np.array(img)
    return img.transpose((2, 0, 1))


class ColorMap(object):
    def __init__(self, num):
        super().__init__()
        self.get_color_map_list(num)
        self.color_map = {}
        self.ptr = 0

    def __getitem__(self, key):
        return self.color_map[key]

    def update(self, keys):
        for key in keys:
            if key not in self.color_map:
                i = self.ptr % len(self.color_list)
                self.color_map[key] = self.color_list[i]
                self.ptr += 1

    def get_color_map_list(self, num_classes):
        color_map = num_classes * [0, 0, 0]
        for i in range(0, num_classes):
            j = 0
            lab = i
            while lab:
                color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
                color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
                color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
                j += 1
                lab >>= 3
        self.color_list = [
            color_map[i:i + 3] for i in range(0, len(color_map), 3)
        ]


class BasePredictor(object):
    def __init__(self, predictor_config):
        super().__init__()
        self.predictor_config = predictor_config
        self.predictor, self.input_names, self.output_names = self.load_predictor(
            predictor_config["model_file"], predictor_config["params_file"])

    def load_predictor(self, model_file, params_file):
        config = Config(model_file, params_file)
        if self.predictor_config["use_gpu"]:
            config.enable_use_gpu(200, 0)
            config.switch_ir_optim(True)
        else:
            config.disable_gpu()
            config.set_cpu_math_library_num_threads(self.predictor_config["cpu_threads"])

            if self.predictor_config["enable_mkldnn"]:
                try:
                    # cache 10 different shapes for mkldnn to avoid memory leak
                    config.set_mkldnn_cache_capacity(10)
                    config.enable_mkldnn()
                except Exception as e:
                    logging.error("The current environment does not support `mkldnn`, so disable mkldnn.")
        config.disable_glog_info()
        config.enable_memory_optim()
        # use zero copy
        config.switch_use_feed_fetch_ops(False)
        predictor = create_predictor(config)
        input_names = predictor.get_input_names()
        output_names = predictor.get_output_names()
        return predictor, input_names, output_names


class Detector(BasePredictor):
    def __init__(self, det_config, predictor_config):
        super().__init__(predictor_config)
        self.det_config = det_config
        self.target_size = self.det_config["target_size"]
        self.thresh = self.det_config["thresh"]

    def preprocess(self, img):
        resize_h, resize_w = self.target_size
        img_shape = img.shape
        img_scale_x = resize_w / img_shape[1]
        img_scale_y = resize_h / img_shape[0]
        img = cv2.resize(img, None, None, fx=img_scale_x, fy=img_scale_y, interpolation=1)
        img = normalize_image(
            img,
            scale=1 / 255.,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            order="hwc")
        img_info = {}
        img_info["im_shape"] = np.array(img.shape[:2], dtype=np.float32)[np.newaxis, :]
        img_info["scale_factor"] = np.array([img_scale_y, img_scale_x], dtype=np.float32)[np.newaxis, :]

        img = img.transpose((2, 0, 1)).copy()
        img_info["image"] = img[np.newaxis, :, :, :]
        return img_info

    def postprocess(self, np_boxes):
        expect_boxes = (np_boxes[:, 1] > self.thresh) & (np_boxes[:, 0] > -1)
        return np_boxes[expect_boxes, :]

    def predict(self, img):
        inputs = self.preprocess(img)
        for input_name in self.input_names:
            input_tensor = self.predictor.get_input_handle(input_name)
            input_tensor.copy_from_cpu(inputs[input_name])
        self.predictor.run()
        output_tensor = self.predictor.get_output_handle(self.output_names[0])
        np_boxes = output_tensor.copy_to_cpu()
        # boxes_num = self.detector.get_output_handle(self.detector_output_names[1])
        # np_boxes_num = boxes_num.copy_to_cpu()
        box_list = self.postprocess(np_boxes)
        return box_list


class Recognizer(BasePredictor):
    def __init__(self, rec_config, predictor_config):
        super().__init__(predictor_config)
        if rec_config["index"] is not None:
            self.load_index(rec_config["index"])
        self.rec_config = rec_config
        self.cdd_num = self.rec_config["cdd_num"]
        self.thresh = self.rec_config["thresh"]
        self.max_batch_size = self.rec_config["max_batch_size"]

    def preprocess(self, img, box_list=None):
        img = normalize_image(
            img,
            scale=1 / 255.,
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
            order="hwc")
        if box_list is None:
            height, width = img.shape[:2]
            box_list = [np.array([0, 0, 0, 0, width, height])]
        batch = []
        input_batches = []
        cnt = 0
        for idx, box in enumerate(box_list):
            box[box < 0] = 0
            xmin, ymin, xmax, ymax = list(map(int, box[2:]))
            face_img = img[ymin:ymax, xmin:xmax, :]
            face_img = cv2.resize(face_img, (112, 112)).transpose((2, 0, 1)).copy()
            batch.append(face_img)
            cnt += 1
            if cnt % self.max_batch_size == 0 or (idx + 1) == len(box_list):
                input_batches.append(np.array(batch))
                batch = []
        return input_batches

    def retrieval(self, np_feature):
        labels = []
        for feature in np_feature:
            similarity = cosine_similarity(self.index_feature, feature).squeeze()
            abs_similarity = np.abs(similarity)
            candidate_idx = np.argpartition(abs_similarity, -self.cdd_num)[-self.cdd_num:]
            remove_idx = np.where(abs_similarity[candidate_idx] < self.thresh)
            candidate_idx = np.delete(candidate_idx, remove_idx)
            candidate_label_list = list(np.array(self.label)[candidate_idx])
            if len(candidate_label_list) == 0:
                maxlabel = "unknown"
            else:
                maxlabel = max(candidate_label_list, key=candidate_label_list.count)
            labels.append(maxlabel)
        return labels

    def load_index(self, file_path):
        with open(file_path, "rb") as f:
            index = pickle.load(f)
        self.label = index["label"]
        self.index_feature = np.array(index["feature"]).squeeze()

    def predict(self, img, box_list=None):
        batch_list = self.preprocess(img, box_list)
        feature_list = []
        for batch in batch_list:
            for input_name in self.input_names:
                input_tensor = self.predictor.get_input_handle(input_name)
                input_tensor.copy_from_cpu(batch)
            self.predictor.run()
            output_tensor = self.predictor.get_output_handle(self.output_names[0])
            np_feature = output_tensor.copy_to_cpu()
            feature_list.append(np_feature)
        return np.array(feature_list)


class MyFace(object):
    def __init__(self, rec_path, det_path, index_path):
        super().__init__()
        self.rec_path = rec_path
        self.det_path = det_path
        self.index_path = index_path
        # detection
        self.init_det()
        # recognition
        self.init_rec()

    def init_rec(self):
        rec_config = {
            "max_batch_size": 1,
            "resize": 112,
            "thresh": 0.45,
            "index": self.index_path,
            "cdd_num": 5
        }
        rec_predictor_config = {
            "use_gpu": True,
            "enable_mkldnn": False,
            "cpu_threads": 1
        }
        model_file_path, params_file_path = check_model_file(self.rec_path)
        rec_predictor_config["model_file"] = model_file_path
        rec_predictor_config["params_file"] = params_file_path
        self.rec_predictor = Recognizer(rec_config, rec_predictor_config)

    def init_det(self):
        det_config = {"thresh": 0.8, "target_size": [640, 640]}
        det_predictor_config = {
            "use_gpu": True,
            "enable_mkldnn": False,
            "cpu_threads": 1
        }
        model_file_path, params_file_path = check_model_file(self.det_path)
        det_predictor_config["model_file"] = model_file_path
        det_predictor_config["params_file"] = params_file_path
        self.det_predictor = Detector(det_config, det_predictor_config)

        # TODO(gaotingquan): now only support fixed number of color
        self.color_map = ColorMap(100)

    def preprocess(self, img):
        img = img.astype(np.float32, copy=False)
        return img

    def draw(self, img, box_list, labels):
        self.color_map.update(labels)
        im = Image.fromarray(img)
        draw = ImageDraw.Draw(im)

        for i, dt in enumerate(box_list):
            bbox, score = dt[2:], dt[1]
            label = labels[i]
            color = tuple(self.color_map[label])

            xmin, ymin, xmax, ymax = bbox

            font_size = max(int((xmax - xmin) // 6), 10)
            font = ImageFont.truetype("SourceHanSansCN-Medium.otf", font_size)

            text = "{} {:.4f}".format(label, score)
            th = sum(font.getmetrics())
            tw = font.getsize(text)[0]
            start_y = max(0, ymin - th)

            draw.rectangle([(xmin, start_y), (xmin + tw + 1, start_y + th)], fill=color)
            draw.text(
                (xmin + 1, start_y),
                text,
                fill=(255, 255, 255),
                font=font,
                anchor="la")
            draw.rectangle([(xmin, ymin), (xmax, ymax)], width=2, outline=color)
        return np.array(im)

    def predict_np_img(self, img, rec):
        input_img = self.preprocess(img)
        box_list = None
        np_feature = None
        box_list = self.det_predictor.predict(input_img)
        if rec:
            np_feature = self.rec_predictor.predict(input_img, box_list)
        
        return box_list, np_feature
    

    def predict(self, input_data, rec):
        """Predict input_data.
        Args:
            input_data (numpy ndarray): image.
        """
        box_list, np_feature = self.predict_np_img(input_data, rec)
        if np_feature is not None:
            labels = self.rec_predictor.retrieval(np_feature)
        else:
            labels = ["face"] * len(box_list)
        if box_list is not None:
            result = self.draw(input_data, box_list, labels=labels)
        return result


########################################################################################################################
############################################# Streamlit part ###########################################################
########################################################################################################################

original_title = '<p style="color:Blue; font-size: 50px;"><strong>Face Detection and Recognition</strong></p>'
st.markdown(original_title+'<style>body{color: White; background-color: DarkSlateGrey}</style>', unsafe_allow_html=True)

option = st.selectbox(
    'What kind of service do you want?',
    ('Detection only', 'Detection and Recognition'))

predictor = MyFace("paddle_models/FresResNet50", "paddle_models/BlazeFace", "ronmes.bin")

if option == "Detection only":
    st.header(':green[Face Detection only]')
    option = st.selectbox(
        'What type of information do you want detected?',
        ('Video', 'Image'))
    if option == 'Video':
        video_data = st.file_uploader("upload", ['mp4','mov', 'avi'])
        if video_data:
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_filename = temp_file.name
                temp_file.write(video_data.read())

            my_bar = st.progress(0, text="Process in progress. Please wait...")

            cap = cv2.VideoCapture(temp_filename)
            nums = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            imagepl = st.empty()

            if (cap.isOpened()== False):
                st.write("Error opening video file! Upload another video.")
            fpsst = st.empty()
            num = 0
            while(cap.isOpened()):
                ret, frame = cap.read()
                if num == int(nums):
                    my_bar.progress(num / nums, text="100% done.")
                else:
                    my_bar.progress(num / nums, text=f"{int(100*num/nums)}% done. Please wait...")
                num += 1
                if ret == True:
                    start = time.time()
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result = predictor.predict(frame, rec=False)
                    imagepl.image(result)
                    end = time.time()
                    fps = round(1 / (end - start), 2)
                    fpsst.write("FPS : " + str(fps))
                else:
                    break
    else:
        img_file_buffer = st.file_uploader("upload", type=["png", "jpg", "jpeg"])
        if img_file_buffer is not None:
            start = time.time()
            image = Image.open(img_file_buffer).convert('RGB')
            img_array = np.array(image)

            result = predictor.predict(img_array, rec=False)
            fps = round(time.time() - start, 4)
            st.image(result)
            st.write("The process time: ", fps, ' s')
else:
    st.header(':yellow[Face Detection with Recognition]')
    option = st.selectbox(
        'What type of information do you want detected?',
        ('Video', 'Image'))
    if option == 'Video':
        video_data = st.file_uploader("upload", ['mp4','mov', 'avi'])
        if video_data:
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_filename = temp_file.name
                temp_file.write(video_data.read())

            my_bar = st.progress(0, text="Process in progress. Please wait...")

            cap = cv2.VideoCapture(temp_filename)
            nums = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            imagepl = st.empty()

            if (cap.isOpened()== False):
                st.write("Error opening video file! Upload another video.")
            fpsst = st.empty()
            num = 0
            while(cap.isOpened()):
                ret, frame = cap.read()
                if num == int(nums):
                    my_bar.progress(num / nums, text="100% done.")
                else:
                    my_bar.progress(num / nums, text=f"{int(100*num/nums)}% done. Please wait...")
                num += 1
                if ret == True:
                    start = time.time()
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result = predictor.predict(frame, rec=True)
                    imagepl.image(result)
                    end = time.time()
                    fps = round(1 / (end - start), 2)
                    fpsst.write("FPS : " + str(fps))
                else:
                    break
    else:
        img_file_buffer = st.file_uploader("upload", type=["png", "jpg", "jpeg"])
        if img_file_buffer is not None:
            start = time.time()
            image = Image.open(img_file_buffer).convert('RGB')
            img_array = np.array(image)

            result = predictor.predict(img_array, rec=True)
            fps = round(time.time() - start, 4)
            st.image(result)
            st.write("The process time: ", fps, ' s')
