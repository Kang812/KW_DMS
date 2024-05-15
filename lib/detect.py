import numpy as np
import time
import cv2
import sys
import os
import torch
import torchvision
import matplotlib.pyplot as plt
from scipy.spatial import distance as dist
sys.path.append(os.path.join(os.getcwd(),"lib/Pytorch_Retinaface/"))
sys.path.append(os.path.join(os.getcwd(),"lib/"))
from PFLD_pytorch.models.pfld import PFLDInference, AuxiliaryNet

from models.retinaface import RetinaFace
from data import cfg_mnet
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
from utils.box_utils import decode, decode_landm

def ear(pts):
    A = dist.euclidean(pts[1], pts[7])
    B = dist.euclidean(pts[3], pts[5])
    C = dist.euclidean(pts[0], pts[4])
    return (A+B)/(2.0 * C)

def head_rate(pts):
    A = dist.euclidean(pts[51], pts[54])
    B = dist.euclidean(pts[54], pts[16])
    return A/B

def stir_rate(pts):
    A = dist.euclidean(pts[4], pts[54])
    B = dist.euclidean(pts[28], pts[54])
    return A/B

def face_metric(pts):
    left_EAR = ear(pts[60:68])
    right_EAR = ear(pts[68:76])
    EAR_avg = (left_EAR + right_EAR) / 2
    HR = head_rate(pts)
    SR = stir_rate(pts)
    return EAR_avg, HR, SR

def face_detection(test_img, face_detection_weight):
    device = "cpu"
    cfg_mnet['pretrain_path'] = os.path.join(os.getcwd(), "models/face_detection/mobilenetV1X0.25_pretrain.tar")
    model = RetinaFace(cfg_mnet, phase = 'test').to(device)
    
    resize = 1
    confidence_threshold = 0.02
    top_k = 5000
    nms_threshold = 0.4
    keep_top_k = 750
    
    model.load_state_dict(torch.load(face_detection_weight, map_location=device)) # map_location 어느 디바이스로 학습했던, 핸재 device를 사용한다는 의미
    model.eval()

    img = np.float32(test_img)
    im_height, im_width, _ = img.shape
    
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img -= (104, 117, 123)
    
    img = img.transpose(2, 0, 1) #(c, w, h)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)
    scale = scale.to(device)

    loc, conf, landms = model(img)  # forward pass
    
    priorbox = PriorBox(cfg_mnet, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg_mnet['variance'])
    boxes = boxes * scale / resize
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

    # ignore low scores
    inds = np.where(scores > confidence_threshold)[0] # confidence_threshold 값보다 높은 box의 index
    boxes = boxes[inds] # confidence_threshold 값보다 높은 box 추출(인덱싱 수행)
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:top_k] # top-k만큼 추출하기 위한 정렬 -> scores를 기준으로 정렬(내림차순으로 정렬)
    boxes = boxes[order] # 인텍싱 수행
    scores = scores[order] # 인텍싱 수행

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, nms_threshold)
    dets = dets[keep, :]

    # keep top-K faster NMS
    dets = dets[:keep_top_k, :]
    return dets

def landmark_detection(img, det, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    pfld_backbone = PFLDInference().to(device)
    pfld_backbone.load_state_dict(checkpoint['pfld_backbone'])
    pfld_backbone.eval()
    pfld_backbone = pfld_backbone.to(device)
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        
    # 2. 얼굴 영역 찾아 모델 input으로 지정
    # a. 이미지 크기 계산
    height, width = img.shape[:2]
        
    # b. 얼굴 검출 위치 정보 계산
    x1, y1, x2, y2 = (det[:4] + 0.5).astype(np.int32)
    
    w = x2 - x1 + 1
    h = y2 - y1 + 1
    cx = x1 + w // 2
    cy = y1 + h // 2
    
    # c. 얼굴 인식 크기를 정사각형 크기로 확대한 후, 잘린영역 길이 계산
    size = int(max([w, h]) * 1.1)
    x1 = cx - size // 2
    x2 = x1 + size
    y1 = cy - size // 2
    y2 = y1 + size
    
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width, x2)
    y2 = min(height, y2)
    
    edx1 = max(0, -x1)
    edy1 = max(0, -y1)
    edx2 = max(0, x2 - width)
    edy2 = max(0, y2 - height)

    # d. 얼굴 부분만 가져오기
    cropped = img[y1:y2, x1:x2]
        
    # e. 잘린 부분 padding
    if (edx1 > 0 or edy1 > 0 or edx2 > 0 or edy2 > 0):
        cropped = cv2.copyMakeBorder(cropped, edy1, edy2, edx1, edx2, cv2.BORDER_CONSTANT, 0)
        
    # f. resize(112, 112), unsqueeze
    input = cv2.resize(cropped, (112, 112))
    input = transform(input).unsqueeze(0).to(device)
        
    # 3. 모델 Inference 후 원래 이미지에서의 특징점 위치 조정
    _, landmarks = pfld_backbone(input)
    pre_landmark = landmarks[0]
    pre_landmark = pre_landmark.cpu().detach().numpy().reshape(
            -1, 2) * [size, size] - [edx1, edy1]
    
    result = []
    # 4. 검출한 특징점 result에 추가하기 [x,y] 쌍으로
    result = []
    for p in pre_landmark:
        x = p[0] + x1 
        y = p[1] + y1
        result.append([int(x), int(y)])
    
    return result