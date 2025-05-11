import zipFile
import os


def unzip_data(zip_path="data.zip", extract_to="data"):
  if not os.path.exist(extract_to):
    os.makedirs(extract_to)
  with zipFile.zipFile(zip_path, "r") as zip_file:
    zip_file.extractall(extract_to)
  print(f"unzipped to {extract_to}")
