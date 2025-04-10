#!/bin/bash

# 检查参数
if [ $# -lt 2 ]; then
  echo "Usage: $0 <output_dir> <dataset_dir1> [<dataset_dir2> ...]"
  exit 1
fi

output_dir="$1"
shift  # 移除output_dir参数，剩下的都是数据集目录

# 创建输出目录
mkdir -p "$output_dir"/{train,dev,test}

# 主合并逻辑
for subset in train dev test; do
  echo "merge for $subset "
  for file in wav.scp text ; do
    echo "    merge file ${file}"
    output_file="$output_dir/$subset/$file"
    > "$output_file"  # 清空或创建文件
    
    # 合并所有数据集目录中的对应文件
    for dataset_dir in "$@"; do
      src_file="$dataset_dir/$subset/$file"
      if [ -f "$src_file" ]; then
        echo "        Merging $src_file -> $output_file"
	name=`basename ${dataset_dir}`
        #head -n 1 "$src_file" | awk -v prefix=${name} '{print prefix "-" $0}'
        cat "$src_file" | awk -v prefix=${name} '{print prefix "-" $0}' >> "$output_file"
      fi
    done
  done
done

echo "合并完成！输出目录: $output_dir"
