@echo off
REM Set environment variables for your project (if necessary)
set PYTHONPATH=.

REM Activate your virtual environment if you are using one
REM call C:\path\to\your\virtualenv\Scripts\activate.bat

REM Run the Python script with the required arguments
python train/transformerTrain.py ^
    --train-raw data/Har/train.pt ^
    --validation-raw data/Har/val.pt ^
    --eval-raw data/Har/test.pt ^
    --batch-size 256 ^
    --input-size 9 ^
    --d-model 16 ^
    --nhead 4 ^
    --n-layers 2 ^
    --output-size 6 ^
    --dropout 0.1 ^
    --learning-rate 0.0001 ^
    --epochs 100 ^
    --save-model ^
    --checkpoint-dir ../trained_models

REM Deactivate the virtual environment if necessary
REM deactivate
