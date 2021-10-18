python run.py predict --archive-file checkpoints/ckpt-amr-1.0-rec --weights-file checkpoints/ckpt-amr-1.0-rec/best.th --input-file data/AMR/amr_1.0/test.txt.features.preproc --batch-size 32 --use-dataset-reader --cuda-device 0 --output-file test.pred.txt --silent --beam-size 5 --predictor PARSER


