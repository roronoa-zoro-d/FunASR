
ffmpeg_dir=${HOME}/tools/ffmpeg-master-latest-linux64-gpl-shared



export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.8/site-packages/torch/lib:${ffmpeg_dir}/lib
