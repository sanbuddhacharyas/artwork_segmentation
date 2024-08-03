import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import cv2
import torch
from segment_anything import sam_model_registry, SamPredictor




def Segment_Image(sam, image_bgr, bbox):

    
    mask_predictor = SamPredictor(sam)


    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mask_predictor.set_image(image_rgb)

    masks, scores, logits = mask_predictor.predict(
        box=bbox,
        multimask_output=False
    )

    return masks, scores

def load_sam_model(CHECKPOINT_PATH):

    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    MODEL_TYPE = "vit_h"

    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
    sam.to(device=DEVICE)

    return sam
