wget -l1 -r --no-parent https://the-eye.eu/public/AI/cah/laion400m-met-release/laion400m-meta/

mv the-eye.eu/public/AI/cah/laion400m-met-release/laion400m-meta/ .

python convert_parquet.py ./laion_face_ids.pth ./laion400m-meta ./laion_face_meta