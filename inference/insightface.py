# Copyright (c) 2021 and Modified (m) 2023 PaddlePaddle Authors and @littletomatodonkey repo. All Rights Reserved.

import os
import argparse
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


def parser(add_help=True):
    def str2bool(v):
        return v.lower() in ("true", "t", "1")
    
    def correct_directory(path):
        return re.sub(r"\\", r"/", path)

    parser = argparse.ArgumentParser(add_help=add_help)
    parser.add_argument(
        "--det_model",
        type=str,
        default="paddle_models/BlazeFace",
        help="The detection model.")
    parser.add_argument(
        "--rec_model",
        type=str,
        default="paddle_models/FresResNet100",
        help="The recognition model.")
    parser.add_argument(
        "--use_gpu",
        type=str2bool,
        default=True,
        help="Whether use GPU to predict. Default by True.")
    parser.add_argument(
        "--enable_mkldnn",
        type=str2bool,
        default=False,
        help="Whether use MKLDNN to predict, valid only when --use_gpu is False. Default by False."
    )
    parser.add_argument(
        "--cpu_threads",
        type=int,
        default=1,
        help="The num of threads with CPU, valid only when --use_gpu is False. Default by 1."
    )
    parser.add_argument(
        "--input",
        type=str,
        help="The path or directory of image(s) or video to be predicted.")
    parser.add_argument(
        "--output",
        type=correct_directory,
        default="output/",
        help="The directory of prediction result.")
    parser.add_argument(
        "--det", action="store_true", help="Whether to detect.")
    parser.add_argument(
        "--det_thresh",
        type=float,
        default=0.8,
        help="The threshold of detection postprocess. Default by 0.8.")
    parser.add_argument(
        "--rec",
        action="store_true", 
        help="Whether to recognize.")
    parser.add_argument(
        "--index", 
        type=str, 
        default=None, 
        help="The path of index file.")
    parser.add_argument(
        "--cdd_num",
        type=int,
        default=5,
        help="The number of candidates in the recognition retrieval. Default by 5."
    )
    parser.add_argument(
        "--rec_thresh",
        type=float,
        default=0.45,
        help="The threshold of recognition postprocess. Default by 0.45.")
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=1,
        help="The maxium of batch_size to recognize. Default by 1.")
    parser.add_argument(
        "--build_index",
        type=str,
        default=None,
        help="The path of index to be build.")
    parser.add_argument(
        "--img_dir",
        type=str,
        default=None,
        help="The img(s) dir used to build index.")
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="The label file path used to build index.")

    return parser


def check_model_file(model):
    """Check the model files exist and download and untar when no exist.
    """
    if os.path.isdir(model):
        model_file_path = os.path.join(model, "inference.pdmodel")
        params_file_path = os.path.join(model, "inference.pdiparams")
        if not os.path.exists(model_file_path) or not os.path.exists(params_file_path):
            raise Exception(f"The specifed model directory error. The drectory must include \"inference.pdmodel\" and \"inference.pdiparams\".")

        return model_file_path, params_file_path
    else:
        raise Exception(f'Model directory is incorrect. Check model path {model}.')


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
            if rec_config["build_index"] is not None:
                raise Exception("Only one of --index and --build_index can be set!")
            self.load_index(rec_config["index"])
        elif rec_config["build_index"] is None:
            raise Exception("One of --index and --build_index have to be set!")
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
    def __init__(self, args):
        super().__init__()

        self.args = args

        self.font_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "SourceHanSansCN-Medium.otf")

        # build index
        if args.build_index:
            if args.rec or args.det:
                warning_str = f"Only one of --rec (or --det) and --build_index can be set!"
                raise Exception(warning_str)
            if args.img_dir is None or args.label is None:
                raise Exception("Please specify the --img_dir and --label when build index.")
            self.init_rec(args)

        # build index
        if args.build_index:
            if args.rec or args.det:
                warning_str = f"Only one of --rec (or --det) and --build_index can be set!"
                raise Exception(warning_str)
            if args.img_dir is None or args.label is None:
                raise Exception(
                    "Please specify the --img_dir and --label when build index."
                )
            self.init_det(args)
            self.init_rec(args)
            
        # detection
        if args.det:
            self.init_det(args)

        # recognition
        if args.rec:
            if not args.index:
                warning_str = f"The index file must be specified when recognition! "
                if args.det:
                    logging.warning(warning_str + "Detection only!")
                else:
                    raise Exception(warning_str)
            elif not os.path.isfile(args.index):
                warning_str = f"The index file not found! Please check path of index: \"{args.index}\". "
                if args.det:
                    logging.warning(warning_str + "Detection only!")
                else:
                    raise Exception(warning_str)
            else:
                self.init_rec(args)

        if not args.build_index and not args.det and not args.rec:
            raise Exception("Specify at least the detection(--det) or recognition(--rec) or --build_index!")
        

    def init_rec(self, args):
        rec_config = {
            "max_batch_size": args.max_batch_size,
            "resize": 112,
            "thresh": args.rec_thresh,
            "index": args.index,
            "build_index": args.build_index,
            "cdd_num": args.cdd_num
        }
        rec_predictor_config = {
            "use_gpu": args.use_gpu,
            "enable_mkldnn": args.enable_mkldnn,
            "cpu_threads": args.cpu_threads
        }
        model_file_path, params_file_path = check_model_file(args.rec_model)
        rec_predictor_config["model_file"] = model_file_path
        rec_predictor_config["params_file"] = params_file_path
        self.rec_predictor = Recognizer(rec_config, rec_predictor_config)

    def init_det(self, args):
        det_config = {"thresh": args.det_thresh, "target_size": [640, 640]}
        det_predictor_config = {
            "use_gpu": args.use_gpu,
            "enable_mkldnn": args.enable_mkldnn,
            "cpu_threads": args.cpu_threads
        }
        model_file_path, params_file_path = check_model_file(args.det_model)
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
            font = ImageFont.truetype(self.font_path, font_size)

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

    def predict_np_img(self, img):
        input_img = self.preprocess(img)
        box_list = None
        np_feature = None
        if hasattr(self, "det_predictor"):
            box_list = self.det_predictor.predict(input_img)
        if hasattr(self, "rec_predictor"):
            np_feature = self.rec_predictor.predict(input_img, box_list)
        return box_list, np_feature
    

    def predict(self, input_data):
        """Predict input_data.
        Args:
            input_data (str): The path of image, or the path of video.
        """
        
        # check for image and predict image, save to output path
        if re.search(r"\.(?:(jpg|png|jpeg))$", input_data):
            img = cv2.imread(input_data)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            box_list, np_feature = self.predict_np_img(img)
            if np_feature is not None:
                labels = self.rec_predictor.retrieval(np_feature)
            else:
                labels = ["face"] * len(box_list)
            if box_list is not None:
                result = cv2.cvtColor(self.draw(img, box_list, labels=labels), cv2.COLOR_RGB2BGR)
            if re.search(r"\.(?:(jpg|png|jpeg))$", self.args.output):
                start = 0
                while True:
                    if re.search(r"/", self.args.output[start:]):
                        end = self.args.output[start:].index('/')+start
                        if not os.path.exists(self.args.output[:end]):
                            os.mkdir(self.args.output[:end])
                        start = end + 1
                    else:
                        break
                cv2.imwrite(self.args.output, result)
            else:
                start = 0
                while True:
                    if re.search(r"/", self.args.output[start:]):
                        end = self.args.output[start:].index('/')+start
                        if not os.path.exists(self.args.output[:end]):
                            os.mkdir(self.args.output[:end])
                        start = end + 1
                    else:
                        if start < len(self.args.output):
                            if not os.path.exists(self.args.output):
                                os.mkdir(self.args.output)
                        break
                self.args.output = os.path.join(self.args.output, 'result.jpg')
                cv2.imwrite(self.args.output, result)
            logging.info(f"Process done. Result image file has been saved in \"{self.args.output}\"")
                
        # check for video and predict image, save to output path
        elif re.search(r"\.(?:(mp4|mov|avi))$", input_data):
            cap = cv2.VideoCapture(input_data)
            nums = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            frame_width = int(cap.get(3))
            frame_height = int(cap.get(4))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not re.search(r"\.mp4$", self.args.output):
                if re.search(r"\.(?:(mov|avi))$", self.args.output):
                    args.output = re.sub(r'([A-Za-z0-9_.]*)(\.(?:(mov|avi)))$', r'\1.mp4', self.args.output)
            else:
                args.output = os.path.join('result.mp4')
            out = cv2.VideoWriter(self.args.output, fourcc, fps, (frame_width, frame_height))
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    box_list, np_feature = self.predict_np_img(input_data)
                    if np_feature is not None:
                        labels = self.rec_predictor.retrieval(np_feature)
                    else:
                        labels = ["face"] * len(box_list)
                    if box_list is not None:
                        frame = cv2.cvtColor(self.draw(img, box_list, labels=labels), cv2.COLOR_RGB2BGR)
                    out.write(frame)
                else:
                    break
            logging.info(f"Process done. Result video file has been saved in \"{self.args.output}\"")
            out.release()
            cap.release()

    def build_index(self):
        img_dir = self.args.img_dir
        label_path = self.args.label
        with open(label_path, "r") as f:
            sample_list = f.readlines()

        feature_list = []
        label_list = []

        for idx, sample in enumerate(sample_list):
            name, label = sample.strip().split("\t")
            img = cv2.imread(os.path.join(img_dir, name))
            if img is None:
                logging.warning(f"Error in reading img {name}! Ignored.")
                continue
            box_list, np_feature = self.predict_np_img(img)
            if box_list is None:
                logging.warning(f"Face not detected in {name}! Ignored.")
                continue
            try:
                feature_list.append(np_feature[0])
                label_list.append(label)
            except:
                logging.warning(f"Error with recognition in {name}! Ignored.")
                continue

            if idx % 100 == 0:
                logging.info(f"Build idx: {idx}")

        with open(self.args.build_index, "wb") as f:
            pickle.dump({"label": label_list, "feature": feature_list}, f)
        logging.info(f"Build done. Total {len(label_list)}. Index file has been saved in \"{self.args.build_index}\"")


# for CLI
def main():
    logging.basicConfig(level=logging.INFO)

    args = parser().parse_args()
    predictor = MyFace(args)
    if args.build_index:
        predictor.build_index()
    else:
        res = predictor.predict(args.input)


if __name__ == "__main__":
    main()
