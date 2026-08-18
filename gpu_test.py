import os
import sys

# 지금 실행 중인 python.exe 경로에서 환경 루트를 역산
python_path = sys.executable
print("현재 파이썬 경로:", python_path)

env_root = os.path.dirname(python_path)  # ...\envs\kmap_1_GPU
dll_path = os.path.join(env_root, 'Library', 'bin')
print("DLL 경로:", dll_path)

print("cudart64_110.dll 존재?:", os.path.exists(os.path.join(dll_path, 'cudart64_110.dll')))
print("cudnn64_8.dll 존재?:", os.path.exists(os.path.join(dll_path, 'cudnn64_8.dll')))

os.add_dll_directory(dll_path)

import tensorflow as tf
print("TF 버전:", tf.__version__)
print("GPU 목록:", tf.config.list_physical_devices('GPU'))