python3 export_onnx.py \
--weights=/home/troegubov.aiu/work/repos/pytorch-image-models/output/train/good_bad_plates/convnext_pico.d1_in1k-50_100_pretrained_pad_around_3_cls_with_night-v20-color/plate_model_with_softmax.pt \
--imgsz 50 100 \
--opset 15

#python3 export_onnx.py \
#--weights=/media/user/SSD2TB/Disk/Projects/CarTrack/det_seg_weights/server_weights/vehicle_type/convnext_base/model.pt \
#--imgsz 256 256