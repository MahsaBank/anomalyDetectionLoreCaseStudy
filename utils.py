import zipfile
import os


def unzip_data(zip_path="data.zip", extract_to="data"):
  if not os.path.exists(extract_to):
    os.makedirs(extract_to)
  with zipfile.ZipFile(zip_path, "r") as zip_file:
    zip_file.extractall(extract_to)
  print(f"unzipped to {extract_to}")
