wget https://gitee.com/ascend/pytorch/releases/download/v5.0.0-pytorch2.1.0/torch_npu-2.1.0-cp39-cp39-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple
pip3 install torch==2.1.0
pip3 install setuptools
pip3 install torch_npu-2.1.0-cp39-cp39-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
yum install libX11-devel -y
pip3 install -r requirements.txt
MAX_JOBS=8 pip install praat-parselmouth --no-build-isolation
