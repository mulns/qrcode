# T2QR: Text to QRCode Generation
### Introduction:
Stage 1:
Given prompts and messages, we encode the messages into a qrcode image which is generated according to prompts. The generated qrcode is not scannable.

Stage2:
Given the generated qrcode and its corresponding blueprint image (from stage 1), we fine-tune it to be scanning robust.


---
### Instruction:
Stage 1: please refer to codes in `T2QR_preprocess`.


Stage 2:

Environment setup: 

```
conda env create -f environment.yml
```

Running script:

```
python t2qr.py \
    -c configs/config_qrloss.yaml \
    -ct path_to_generated_qrcode_img \
    -cd path_to_blueprint_img \
    -dir output_directory \
    -mg margin_size \
    -e 400 
```

The `-mg` defines the margin size of a qrcode, which is set to 5 by default. Please note that the generated_qrcode_img, blueprint_img and their margin_size is obtained from stage 1.

**Note:**
If the script gets stuck loading the VAE parameters, check the proxy as we load the pre-trained parameters remotely by default. If no proxy available, please pre-download the checkpoint and load it locally.