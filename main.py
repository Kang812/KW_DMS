import argparse
import os
import cv2
import time
from tqdm import tqdm
from lib.detect import face_detection
from lib.detect import landmark_detection
from lib.detect import face_metric

def dms_main(data_folder, face_detection_model, face_landmark_detection_model, vis_thresh, time_th, save_folder):
    ear_status = False # 눈의 떠져있을 경우에 False
    head_status = False # 정면 주시시 False 
    face_detection_n = False # 모델이 얼굴을 찾거나 vis_thresh 보다 큰 경우, False
    ear_status_warning = False # 눈의 떠져있을 경우에 False 이지만, 2.5초 또는 3초 동안 감겨있을 시에 True로 위험하다는 경고 
    head_status_warning = False # 정면을 주시시 False 이지만, 2.5초 또는 3초 동안 미주시에 True로 위험하다는 경고
    face_detection_n_warning = False # 정면을 미주시시한 시간이 2.5초 또는 3초가 됐고, 정면을 보는 얼굴이 없을 경우 위험 신호를 반환하고 EAR, SR, HR을 계산하지 않음
    
    # 정면 또는 눈을 감은지 경과된 시간을 계산하기 위한 변수
    fd_time = 0 
    head_time = 0
    ear_time = 0
    idx = 0
    file_list = os.listdir(data_folder)
    
    for i in tqdm(range(len(file_list)), bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:10}{r_bar}'):
        # test image 읽어오기
        file = file_list[i]
        img = cv2.imread(os.path.join(data_folder, file))

        #Face detection 진행
        dets = face_detection(img, face_detection_model)
        cr = time.time()
        
        if len(dets) == 0 or dets[0][4] < vis_thresh:
            if face_detection_n == False:
                fd_time = cr
                face_detection_n = True
            if face_detection == True and cr - fd_time > time_th:
                face_detection_n_warning = True
                if face_detection_n_warning :
                    cv2.rectangle(img, (0,0), (300, 100), (255,0,0), -1, cv2.LINE_AA)  # filled
                    cv2.putText(img, 'Warning!', (5,70),cv2.FONT_HERSHEY_DUPLEX, 2,(255,255,255), thickness=3, lineType=cv2.LINE_AA)
            filename = f'm_{idx:04d}.png'
            cv2.imwrite(os.path.join(save_folder, filename), cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            idx += 1
            continue
        
        face_detection_n = False
        face_detection_n_warning = False
        points = landmark_detection(img, dets[0], face_landmark_detection_model)
        EAR, HR, SR = face_metric(points)

        # EAR 이상 확인
        if EAR < EAR_Thresh:
            cr = time.time()
            if ear_status == False:
                ear_time = cr
                ear_status = True
            if ear_status == True and cr - ear_time >time_th:
                ear_status_warning = True
        else:
            ear_status = False
            ear_status_warning = False

        # 고개 이상 확인
        if (HR_Thresh[0] <HR and HR < HR_Thresh[1]) or (SR > SR_Thresh[0] and SR_Thresh[1] < SR):
            cr = time.time()
            if head_status == False:
                head_time = cr
                head_status = True
            if head_status == True and cr - head_time >time_th:
                head_status_warning = True
        else:
            head_status = False
            head_status_warning = False
        
        if ear_status_warning or head_status_warning:
            cv2.rectangle(img, (0,0), (300, 100), (255, 0, 0), -1, cv2.LINE_AA)
            cv2.putText(img, 'Warning!', (5, 70), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 255), thickness = 3, lineType = cv2.LINE_AA)
        
        cv2.putText(img, f'EAR:{EAR:.2f} ', (600,100),cv2.FONT_HERSHEY_DUPLEX, 1,(0,0,255), thickness=3, lineType=cv2.LINE_AA)
        cv2.putText(img, f'SR:{SR:.2f}', (600,150),cv2.FONT_HERSHEY_DUPLEX, 1,(0,0,255), thickness=3, lineType=cv2.LINE_AA)
        cv2.putText(img, f'HR:{HR:.2f}', (600,200),cv2.FONT_HERSHEY_DUPLEX, 1,(0,0,255), thickness=3, lineType=cv2.LINE_AA)
        filename = f'm_{idx:04d}.png'
        cv2.imwrite(os.path.join(save_folder, filename), cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        idx += 1

parser = argparse.ArgumentParser(description='DMS(Drive Monitoring System)')
parser.add_argument('--face_detection_model', type = str, default="/workspace/autonomous_driving/Part_1_Perception/8_DMS/models/face_detection/mobilenet0.25_epoch_245.pth")
parser.add_argument('--face_landmark_detection_model', type = str, default="/workspace/autonomous_driving/Part_1_Perception/8_DMS/models/landmark_detection/face_landmark_detection.tar")
parser.add_argument('--data_folder', type = str, default="/workspace/autonomous_driving/Part_1_Perception/8_DMS/dataset/test_img/267_G1/")
parser.add_argument('--time_th', type = float, default=0.25)
parser.add_argument('--vis_thresh', type = float, default=0.3)
parser.add_argument('--save_folder', type = str, default="/workspace/autonomous_driving/Part_1_Perception/8_DMS/dataset/test_img/267_G1_result/")

if __name__ == "__main__":
    args = parser.parse_args()
    EAR_Thresh = 0.2        # 이것보다 작으면 문제
    HR_Thresh = (0.5, 0.95) # 이구간만 문제
    SR_Thresh = (0.5, 2)    # 이사이만 정상

    face_detection_model = args.face_detection_model
    face_landmark_detection_model = args.face_landmark_detection_model
    data_folder = args.data_folder
    time_th = args.time_th
    vis_thresh = args.vis_thresh
    save_folder = args.save_folder

    dms_main(data_folder, face_detection_model, face_landmark_detection_model, vis_thresh, time_th, save_folder)
