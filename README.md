# KW_DMS
DMS(Drive Monitoring System)
- 운전자가 전방 주시를 하는지 또는 졸음 운전중인지를 딥러닝 기반으로 모니터링

## 환경
- ubuntu 20.04
- cpu AMD Ryzen 4650g
- ram 64gb
- gpu RTX 3060 12GB x2

## 사용모델
- [retinanet(mobilenet0.25)](https://github.com/biubug6/Pytorch_Retinaface)
  * mAP0.97
- [PFLD](https://github.com/polarisZhao/PFLD-pytorch)
- 모델 학습 관련된 내용은 각 깃헙 참고

## DMS 실행
```
git clone https://github.com/Kang812/KW_DMS.git\
cd ./KW_DMS
./main.sh
```

## DMS 시각화 결과
- [사용한 데이터 셋](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=173)
