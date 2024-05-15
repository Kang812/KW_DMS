import gdown
import zipfile

file_id = "1jogArNuf_lOw3_rw1TSdIrZ5UHQmHrCO"
output = "./face_detection.zip"
gdown.download(id=file_id, output=output, quiet=False)

output_dir = "./"
zip_file = zipfile.ZipFile(output)
zip_file.extractall(path=output_dir)