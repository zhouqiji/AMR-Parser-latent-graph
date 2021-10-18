python run.py predict --archive-file ../ckpts/ckpt-amr-2.0-rec --weights-file ../ckpts/ckpt-amr-2.0-rec/best.th --input-file data/AMR/amr_2.0/test.txt.features.preproc --batch-size 32 --use-dataset-reader --cuda-device 0 --output-file test.pred.txt --silent --beam-size 5 --predictor PARSER --visual


