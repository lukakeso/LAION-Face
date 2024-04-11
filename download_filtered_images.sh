#!/bin/bash  

folder="$2"
echo $folder
mkdir $folder

echo "start download"
img2dataset --url_list $1/laion_face_$3.parquet --input_format "parquet" \
    --url_col "URL" --caption_col "TEXT" --output_format webdataset\
    --output_folder $folder/glasses_$3 --processes_count 16 --thread_count 128 --resize_mode no \
        --save_additional_columns '["NSFW","similarity","LICENSE","SAMPLE_ID"]'
echo "end download"