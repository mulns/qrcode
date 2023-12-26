CUDA_VISIBLE_DEVICES=0 python t2qr.py \
               -c configs/config_qrloss.yaml \
               -ct data/paper_before_talk/01319/pic2.png \
               -cd data/paper_before_talk/01319/qr2_margin5.png \
               -dir ./out/paper/crossMarker/01319 \
               -mg 5 \
               -e 400 