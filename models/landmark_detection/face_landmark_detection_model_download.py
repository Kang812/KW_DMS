import gdown

file_id = "1QTfRGk3y7QtfKeJwnhB98uFCG4Vhs3Ty"
output = "./face_landmark_detection.tar"
gdown.download(id=file_id, output=output, quiet=False)
