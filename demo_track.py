import argparse
import os
import os.path as osp
import time
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import subprocess
from PIL import Image
import torch.nn.functional as F
import sys

sys.path.append("./ProPainter/")
sys.path.append("./PIDNet/")
sys.path.append("./HybridSORT/")

from HybridSORT.utils.args import make_parser, args_merge_params_form_exp

from loguru import logger

from HybridSORT.yolox.data.data_augment import preproc
from HybridSORT.yolox.exp import get_exp
from HybridSORT.yolox.utils import fuse_model, get_model_info, postprocess
from HybridSORT.yolox.utils.visualize import plot_tracking, plot_tracking_detection
from HybridSORT.trackers.ocsort_tracker.ocsort import OCSort
from HybridSORT.trackers.hybird_sort_tracker.hybird_sort import Hybird_Sort
from HybridSORT.trackers.hybird_sort_tracker.hybird_sort_reid import Hybird_Sort_ReID
from HybridSORT.trackers.tracking_utils.timer import Timer
from HybridSORT.fast_reid.fast_reid_interfece import FastReIDInterface
import copy

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

script_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)
from PIDNet.models import pidnet as model_pid


def get_masks(predictor, image, boxes):
    transformed_boxes = predictor.transform.apply_boxes_torch(boxes, image.shape[:2])
    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )
    return masks


def get_image_list(path):
    '''해당 path에 존재하는 모든 이미지 파일 명을 저장하여 반환'''
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = osp.join(maindir, filename)
            ext = osp.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


def mask_file_name_reset(folder):
    path = folder
    file_list = os.listdir(path)

    for old_name in file_list:
        dummy = '00000'
        new_name = list(old_name.split('_'))[1]
        new_name = dummy[:-len(list(new_name.split('.'))[0])] + new_name
        os.rename(os.path.join(path, old_name), os.path.join(path, new_name))


def load_pretrained(model, pretrained):
    pretrained_dict = torch.load(pretrained, map_location='cpu')
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if (k[6:] in model_dict and v.shape == model_dict[k[6:]].shape)}
    msg = 'Loaded {} parameters!'.format(len(pretrained_dict))
    print('Attention!!!')
    print(msg)
    print('Over!!!')
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict = False)
    
    return model

def input_transform(image):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    image = image.astype(np.float32)[:, :, ::-1]
    image = image / 255.0
    image -= mean
    image /= std
    return image


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        trt_file=None,
        decoder=None,
        device=torch.device("cpu"),
        fp16=False,
        with_reid=False,
        fast_reid_config=None,
        fast_reid_weights=None,
    ):
        self.model = model
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones((1, 3, exp.test_size[0], exp.test_size[1]), device=device)
            self.model(x)
            self.model = model_trt
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        self.with_reid = with_reid
        if self.with_reid:
            self.fast_reid_config = fast_reid_config
            self.fast_reid_weights = fast_reid_weights
            self.encoder = FastReIDInterface(self.fast_reid_config, self.fast_reid_weights, 'cuda')

    def inference(self, img, timer):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, ratio, raw_image = preproc(img, self.test_size, self.rgb_means, self.std)  # _ for raw_image
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img = img.half()  # to FP16

        with torch.no_grad():
            timer.tic()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )
            if self.with_reid:
                bbox_xyxy = copy.deepcopy(outputs[0][:, :4])  # [hgx0411]
                # we should save the detections here !
                # os.makedirs("dance_detections/{}".format(video_name), exist_ok=True)
                # torch.save(outputs[0], ckt_file)

                # [hgx0411] box rescale borrowed from convert_to_coco_format()
                scale = min(self.test_size[0] / float(img_info["height"]), self.test_size[1] / float(img_info["width"]))
                bbox_xyxy /= scale
                id_feature = self.encoder.inference(raw_image, bbox_xyxy.cpu().detach().numpy())  # normalization and numpy included
        if self.with_reid:
            return outputs, img_info, id_feature
        else:
            return outputs, img_info


def image_demo(predictor, vis_folder, current_time, args):
    if osp.isdir(args.path):
        files = get_image_list(args.path)
    else:
        files = [args.path]
    files.sort()
    if not args.hybird_sort_with_reid:
        tracker = Hybird_Sort(args, det_thresh=args.track_thresh,
                                    iou_threshold=args.iou_thresh,
                                    asso_func=args.asso,
                                    delta_t=args.deltat,
                                    inertia=args.inertia,
                                    use_byte=args.use_byte)
    else:
        tracker = Hybird_Sort_ReID(args, det_thresh=args.track_thresh,
                                    iou_threshold=args.iou_thresh,
                                    asso_func=args.asso,
                                    delta_t=args.deltat,
                                    inertia=args.inertia)
        # tracker = OCSort(det_thresh=args.track_thresh, iou_threshold=args.iou_thresh, use_byte=args.use_byte)

    timer = Timer()
    results = []

    for frame_id, img_path in enumerate(files, 1):
        if args.with_fastreid:
            outputs, img_info, id_feature = predictor.inference(img_path, timer)
        else:
            outputs, img_info = predictor.inference(img_path, timer)
        if outputs[0] is not None:
            if args.with_fastreid:
                online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size, id_feature=id_feature)
            else:
                online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size)
            online_tlwhs = []
            online_ids = []
            for t in online_targets:
                tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
                tid = t[4]
                vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                # if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                if tlwh[2] * tlwh[3] > args.min_box_area:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    results.append(
                        f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},1.0,-1,-1,-1\n"
                    )
            timer.toc()
            online_im = plot_tracking(
                img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id, fps=1. / timer.average_time
            )
            img_h, img_w = img_info['height'], img_info['width']
            scale = min(exp.test_size[0] / float(img_h), exp.test_size[1] / float(img_w))
            online_im_detection = plot_tracking_detection(
                img_info['raw_img'], outputs[0][:, :4]/scale, (outputs[0][:, 4]*outputs[0][:, 5]), frame_id=frame_id, fps=1. / timer.average_time
            )
        else:
            timer.toc()
            online_im = img_info['raw_img']

        # result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
        if args.save_result:
            if not args.demo_dancetrack:
                timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
                save_folder = osp.join(vis_folder, timestamp)
            else:
                timestamp = args.path[-19:]
                save_folder = osp.join(vis_folder, timestamp)
            os.makedirs(save_folder, exist_ok=True)
            cv2.imwrite(osp.join(save_folder, osp.basename(img_path)), online_im)
            save_folder_detection = osp.join(save_folder , "detection")
            os.makedirs(save_folder_detection, exist_ok=True)
            cv2.imwrite(osp.join(save_folder_detection, osp.basename(img_path)), online_im_detection)

        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

        ch = cv2.waitKey(0)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break

    if args.save_result:
        res_file = osp.join(vis_folder, f"{timestamp}.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")


def imageflow_demo(predictor, pid_model, vis_folder, current_time, args):
    cap = cv2.VideoCapture(args.path if args.demo_type == "video" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    save_folder = osp.join(vis_folder, timestamp)
    os.makedirs(save_folder, exist_ok=True)
    if args.demo_type == "video":
        save_path = args.out_path
    else:
        save_path = osp.join(save_folder, "camera.mp4")
    logger.info(f"video save_path is {save_path}")
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )
    if not args.hybird_sort_with_reid:
        tracker = Hybird_Sort(args, det_thresh=args.track_thresh,
                                    iou_threshold=args.iou_thresh,
                                    asso_func=args.asso,
                                    delta_t=args.deltat,
                                    inertia=args.inertia,
                                    use_byte=args.use_byte)
    else:
        tracker = Hybird_Sort_ReID(args, det_thresh=args.track_thresh,
                                    iou_threshold=args.iou_thresh,
                                    asso_func=args.asso,
                                    delta_t=args.deltat,
                                    inertia=args.inertia)
    # tracker = OCSort(det_thresh=args.track_thresh, iou_threshold=args.iou_thresh, use_byte=args.use_byte)
    timer = Timer()
    frame_id = 0
    results = []
    while True:
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
        ret_val, frame = cap.read()
        if ret_val:
            if args.hybird_sort_with_reid:
                outputs, img_info, id_feature = predictor.inference(frame, timer)
            else:
                outputs, img_info = predictor.inference(frame, timer)
            # sam_predictor.set_image(frame)
            if outputs[0] is not None:
                if args.hybird_sort_with_reid:
                    online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size, id_feature=id_feature)
                else:
                    online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size)
                online_tlwhs = []
                online_ids = []
                sam_boxes = []
                for t in online_targets:
                    tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
                    tid = int(t[4])
                    vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                    # if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                    if tlwh[2] * tlwh[3] > args.min_box_area:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        results.append(
                            f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},1.0,-1,-1,-1\n"
                        )
                        sam_boxes.append([t[0], t[1], t[2], t[3]])


                if frame_id == 0:
                    input_point = list(map(float, args.input_bbox.split(',')))
                    input_x, input_y = input_point[0], input_point[1]
                    min_dist = np.inf
                    target_id = None
                    for id, co in zip(online_ids, sam_boxes):
                        cur_x, cur_y = (co[0] + co[2]) / 2, (co[1] + co[3]) / 2
                        cur_dist = (cur_x - input_x) ** 2 + (cur_y - input_y) ** 2

                        if min_dist > cur_dist:
                            min_dist = cur_dist
                            target_id = id
                target_idx = online_ids.index(target_id)

                if sam_boxes != []:
                    # masks = get_masks(sam_predictor, frame, torch.tensor(sam_boxes, device=sam_predictor.device))
                    # masks = masks.cpu().numpy()
                    # masks = np.sum(masks, axis=0).clip(0, 1)
                    
                    with torch.no_grad():
                        img = frame.copy()
                        img = input_transform(img)
                        img = img.transpose((2, 0, 1)).copy()
                        img = torch.from_numpy(img).unsqueeze(0).cuda()
                        pred = pid_model(img)
                        pred = F.interpolate(pred, size=img.size()[-2:], 
                                                mode='bilinear', align_corners=True)
                        pred = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()
                        masks = (pred == 11).astype(np.uint8)
                        masks[int(sam_boxes[target_idx][1]):int(sam_boxes[target_idx][3]), int(sam_boxes[target_idx][0]):int(sam_boxes[target_idx][2])] = 0

                else:
                    masks = None

                # if frame_id % 20 == 0:
                #     image_data = masks * 255
                #     image = Image.fromarray(image_data.astype('uint8'), 'L')
                #     image.show()

                timer.toc()
                online_im = plot_tracking(
                    img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id + 1, fps=1. / timer.average_time, masks=masks
                )
            else:
                timer.toc()
                online_im = img_info['raw_img']
            if args.save_result:
                vid_writer.write(online_im)
                if masks is not None:
                    masks = np.array(masks * 255).astype('uint8')
                    cv2.imwrite(osp.join(save_folder, f'mask_{frame_id}.jpg'), masks)
                else:
                    non_mask = np.zeros((img_info['height'], img_info['width']))
                    non_mask = np.array(non_mask * 255).astype('uint8')
                    cv2.imwrite(osp.join(save_folder, f'mask_{frame_id}.jpg'), non_mask)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break
        frame_id += 1

    if args.save_result:
        res_file = osp.join(vis_folder, f"{timestamp}.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")

    return save_folder


def main(exp, args):
    if not args.expn:
        args.expn = exp.exp_name

    output_dir = osp.join(exp.output_dir, args.expn)
    os.makedirs(output_dir, exist_ok=True)

    if args.save_result:
        output_dir = 'ProPainter/inputs/object_removal/'
        vis_folder = osp.join(output_dir, str(args.path.split('/')[-1].split('.')[0]))
        os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"
    args.device = torch.device("cuda" if args.device == "gpu" else "cpu")

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model().to(args.device)
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    model.eval()

    if not args.trt:
        ckpt_file = 'HybridSORT/pretrained/bytetrack_x_mot20.tar'
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.fp16:
        model = model.half()  # to FP16

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = osp.join(output_dir, "model_trt.pth")
        assert osp.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    # sam_checkpoint = "pretrained/sam_vit_h_4b8939.pth"
    # model_type = "vit_h"

    # sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    # sam.to(device=args.device)

    # sam_predictor = SamPredictor(sam)

    pid_model = model_pid.get_pred_model('pidnet-l', 19)
    pid_model = load_pretrained(pid_model, 'HybridSORT/pretrained/PIDNet_L_Cityscapes_test.pt').cuda()
    pid_model.eval()

    predictor = Predictor(model, exp, trt_file, decoder, args.device, args.fp16,
                          with_reid=args.with_fastreid, fast_reid_config=args.fast_reid_config, fast_reid_weights=args.fast_reid_weights)      # [hgx0427] with_fastreid
    current_time = time.localtime()
    if args.demo_type == "image":
        image_demo(predictor, vis_folder, current_time, args)
    elif args.demo_type == "video" or args.demo_type == "webcam":
        folder = imageflow_demo(predictor, pid_model, vis_folder, current_time, args)
        mask_file_name_reset(folder)

        subprocess.run(["python", "ProPainter/inference_propainter.py", "--video", args.path, 
                "--mask", folder, "--resize_ratio", "0.5", "--fp16"])


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    args_merge_params_form_exp(args, exp)

    main(exp, args)
