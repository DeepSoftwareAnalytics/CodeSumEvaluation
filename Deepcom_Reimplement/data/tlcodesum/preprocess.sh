data_dir=original

output_dir=processed
java_files_dir=$HOME/tl_codesum/java_files
mkdir -p ${output_dir}

python preprocess.py -clc -sbt_type 2 \
    -data_dir $data_dir \
    -java_files_dir $java_files_dir \
    -output_dir $output_dir 2>&1 | tee preprocess.log