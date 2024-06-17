import torch
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
import time
import os
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
import cv2
from PIL import Image
from classes import DetectionObj



class DETREngine:
    def __init__(self, detr_version="base"):
        print(torch.__version__, torch.cuda.is_available())
        torch.set_grad_enabled(False)

        self.threshold = 0.8
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.detr_version = detr_version

        if (self.detr_version == "ems"):
            self.finetuned_classes = [
                'IV needle', 'bp monitor', 'bvm', 'defib pads', 'dummy', 'hands'
            ]
            checkpoint = torch.load(
                './EMS_Vision/weights/ems_finetuned_detr_checkpoint.pth', map_location='cpu')

        else:
            self.finetuned_classes = [
                'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
                'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
                'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
                'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
                'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
                'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
                'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
                'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
                'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
                'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
                'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
                'toothbrush'
            ]
            checkpoint = torch.load(
                './EMS_Vision/weights/detr-r50-e632da11.pth', map_location='cpu')

        self.num_classes = len(self.finetuned_classes)

        self.COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
                       [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

        self.transform = T.Compose([
            # T.Resize((512, 512)),  # Resize the image to 224x224 pixels
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.model = torch.hub.load(
            'facebookresearch/detr', 'detr_resnet50', pretrained=False, num_classes=self.num_classes)
        self.model.load_state_dict(checkpoint['model'], strict=True)
        print("[DETR_Engine] DETR Model loaded")

        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def cv2_to_pil(cv2_image):
        """Convert a cv2 image (numpy array in BGR) to a PIL Image in RGB."""
        rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        return pil_image

    @staticmethod
    def box_cxcywh_to_xyxy(x):
        """Convert bbox coordinates from (center_x, center_y, w, h) to [(x1, y1), (x2, y2)]"""
        x_c, y_c, w, h = x.unbind(1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
             (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1)

    @staticmethod
    def rescale_bboxes(out_bbox, size):
        """Rescale bounding boxes from ratio values to pixel values"""
        img_w, img_h = size
        b = DETREngine.box_cxcywh_to_xyxy(out_bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32).to(b.device)
        return b

    def filter_bboxes_from_outputs(self, outputs, threshold=0.7):
        """Recover bounding-boxes with prediction confidence above threshold (default=0.7)"""
        probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > threshold
        probas_to_keep = probas[keep]
        bboxes_scaled = self.rescale_bboxes(
            outputs['pred_boxes'][0, keep], (640, 480))
        return probas_to_keep, bboxes_scaled

    def create_bbox_objs(self, prob=None, boxes=None):
        """ Takes in outputs from model and creates DetectionObjs

        Args:
            prob (_type_): Probabilities of each detection
            boxes (_type_): min and max corner of each detection

        Returns
            list: list of dictionaries in following format {name: x, boxcoords: [(xmin, ymin), (xmax, ymax)]}
        """
        detection_objects = []
        if prob is not None and boxes is not None:
            for p, box in zip(prob, boxes):
                detection_object = dict()
                cl = p.argmax().item()
                name = f'{self.finetuned_classes[cl]}: {p[cl]:.2f}'
                xmin, ymin, xmax, ymax = box
                boxcoords = [(xmin.item(), ymin.item()),
                             (xmax.item(), ymax.item())]

                detection_object = DetectionObj(
                    box_coords=boxcoords, obj_name=name)
                detection_objects.append(detection_object)
        return detection_objects

    def plot_finetuned_results(self, cv2_img, prob=None, boxes=None):
        """ Given confidences and bounding box coordinates plot the bounding boxes and assign labels.
        Also returns DetectionObjs with data from the probabilities and box coordinates.

        Args:
            cv2_img (_type_): image in cv2 format
            prob (_type_): #TODO: fill in
            boxes (torch.Tensor): tensor for bounding box coordinates

        Returns:
            np.array, list: annotated image, list of DetectionObjs
        """
        detection_objects = []
        if prob is not None and boxes is not None:
            for p, box in zip(prob, boxes):
                xmin, ymin, xmax, ymax = box
                box_coordinates = [(int(xmin), int(ymin)),
                                   (int(xmax), int(ymax))]
                cl = p.argmax().item()
                name = self.finetuned_classes[cl]
                confidence = f'{p[cl]:.2f}'
                label = f'{name}: {confidence}'

                detection_objects.append(DetectionObj(box_coords=box_coordinates,
                                                      name=name,
                                                      confidence=confidence))

                color = [int(x * 255)
                         for x in self.COLORS[cl % len(self.COLORS)]]

                # Draw rectangle
                cv2.rectangle(cv2_img, (int(xmin), int(ymin)),
                              (int(xmax), int(ymax)), color, 2)

                # Draw label background
                label_size, _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(cv2_img, (int(xmin), int(ymin) - label_size[1] - 10),
                              (int(xmin) + label_size[0], int(ymin)), color, cv2.FILLED)

                # Draw text
                cv2.putText(cv2_img, label, (int(xmin), int(ymin) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                # for obj in detection_objects:
                #     print("These are the finetuned results: " + str(obj.obj_name))

        return cv2_img, detection_objects

    def run_workflow(self, my_image):
        start_t = time.time()
        img = self.cv2_to_pil(my_image)
        # print(f"[DETR Engine] Image conversion time: {time.time() - start_t}")
        img = self.transform(img).unsqueeze(0)
        img = img.to(self.device)
        outputs = self.model(img)
        # print(f"[DETR Engine] Inference time: {time.time() - start_t}")
        probas_to_keep, bboxes_scaled = self.filter_bboxes_from_outputs(
            outputs, threshold=self.threshold)

        # plot bboxes on image
        result_image, detection_objects = self.plot_finetuned_results(
            my_image, probas_to_keep, bboxes_scaled)

        detection_results_serialized = [
            vars(detection_object) for detection_object in detection_objects]
        

        return result_image, detection_results_serialized
