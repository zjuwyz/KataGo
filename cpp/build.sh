#/bin/bash
#sudo rm CMakeCache.txt
#docker run -it --rm -v /home/wangyize/.katago:/root/.katago --gpus all katago-dev bash -c "cd /root/.katago/KataGo/cpp && cmake . -DUSE_BACKEND=TensorRT -DUSE_CACHE_TENSORRT_PLAN=1 -DUSE_TENSORRT_CUDA_GRAPH=1 -DTENSORRT_ROOT_DIR=/TensorRT && make -j8"
cd /home/wangyize/.katago/KataGo/cpp && cmake . -DUSE_BACKEND=TensorRT -DUSE_CACHE_TENSORRT_PLAN=1 -DUSE_TENSORRT_CUDA_GRAPH=1 -DTENSORRT_ROOT_DIR=/home/wangyize/3rdparty/TensorRT-10.13.3.9 && make -j
