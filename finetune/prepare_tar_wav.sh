#! /bin/bash

# 将压缩的pcm文件 转换成 wav


stage=0
stop_stage=2
nj=32



# data_dir=$1     # 包含rar文件
# if [ $# -ne 1 ]; then
#     echo "Usage:  <data_dir>"
#     echo " data_dir contain rar files"
#     exit 1
# fi

data_dir=/data/nas/dataset/asr/kefu/huaian2
data_dir=/data/nas/dataset/asr/kefu/huaian3
data_dir=/data/nas/dataset/asr/kefu/huaian_testset/badcase/
data_dir=/data/nas/dataset/asr/kefu/huaian_testset/data2/
data_dir=/data/nas/dataset/asr/kefu/huaian4
data_dir=/data/nas/dataset/asr/kefu/huaian3

pcm_dir=${data_dir}/pcms
raw_dir=${data_dir}/raws
wav_dir=${data_dir}/wavs
temp_dir=${data_dir}/temps

. utils/parse_options.sh || exit 1;




if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    
    echo "stage 0: unrar files"
    [ ! -d ${pcm_dir} ] && rm -rf ${pcm_dir} && mkdir -p ${pcm_dir}

    echo "unrar files `date`"
    echo "$(ls ${data_dir}/*.rar)"
    for file in $(ls ${data_dir}/*.rar); do
        echo "unrar ${file}"
        unrar x ${file} ${pcm_dir} > /dev/null 2>&1 &
    done
    wait
    echo "unrar files done `date`"
fi




if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: mv pcm to raw"
    [ ! -d ${raw_dir} ] && rm -rf ${raw_dir} && mkdir -p ${raw_dir}
    [ ! -d ${temp_dir} ] && rm -rf ${temp_dir} && mkdir -p ${temp_dir}
    
    find ${pcm_dir}/ -name "*.pcm" > ${temp_dir}/pcm_list.txt

    # 生成命令 mv a.pcm a.raw
    cat ${temp_dir}/pcm_list.txt | awk '{print  substr($1, 1, length($1)-4) ".raw",  $1}' | sed "s|${pcm_dir}|${raw_dir}|" | awk '{print "mv", $2, $1}' > ${temp_dir}/pcm_to_raw.sh
    # 创建输出文件夹
    cat ${temp_dir}/pcm_to_raw.sh | awk '{print $3}' | awk -F'/' '{for(i=1;i<NF;i++) printf("%s/", $i); print ""}' | sort -u | awk '{print "mkdir", $1}' | awk '{system($0)}'
    cat ${temp_dir}/pcm_to_raw.sh | xargs -P $nj -I {} sh -c "{}"


    echo "mv to wav `date`"

fi 

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: sox raw to wav"

    # [ ! -d ${wav_dir} ] && rm -rf ${wav_dir} && mkdir -p ${wav_dir}

    find ${raw_dir}/ -name "*.raw" > ${temp_dir}/raw_list.txt
    # 生成输入输出 a/a.raw b/b.wav
    cat ${temp_dir}/raw_list.txt | awk '{print  substr($1, 1, length($1)-4) ".wav",  $1}' | sed "s|${raw_dir}|${wav_dir}|" > ${temp_dir}/wav_raw.txt
    # 创建输出文件夹
    cat  ${temp_dir}/wav_raw.txt | awk '{print $1}' | awk -F'/' '{for(i=1;i<NF;i++) printf("%s/", $i); print ""}' | sort -u | awk '{print "mkdir -p", $1}' | awk '{system($0)}'
    # 打印 sox 执行命令
    cat  ${temp_dir}/wav_raw.txt | awk '{print "sox -r 8000 -e signed-integer -b 16 -c 1", $2, "-r 16000", $1}' > ${temp_dir}/raw_to_wav.sh
    cat ${temp_dir}/raw_to_wav.sh | xargs -P $nj -I {} sh -c "{}" > /dev/null 2>&1
    find ${wav_dir} -name "*.wav"  | awk -F"/" '{print substr($NF, 1, length($NF)-4), $0}' >  ${data_dir}/wav.scp
fi