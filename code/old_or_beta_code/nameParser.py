import re

from transformers import DonutProcessor, VisionEncoderDecoderModel
from datasets import load_dataset
import torch
import cv2

processor = DonutProcessor.from_pretrained('D:/Programmierung/TCG_Card_Detection/models/Donut_processor')
model = VisionEncoderDecoderModel.from_pretrained('D:/Programmierung/TCG_Card_Detection/models/Donut_model')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

image = cv2.imread('D:/Programmierung/TCG_Card_Detection/data/save_name.png')

task_prompt = "<s_cord-v2>"
decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids

pixel_values = processor(image, return_tensors='pt').pixel_values

outputs = model.generate(
    pixel_values.to(device),
    decoder_input_ids=decoder_input_ids.to(device),
    max_length=model.decoder.config.max_position_embeddings,
    pad_token_id=processor.tokenizer.pad_token_id,
    eos_token_id=processor.tokenizer.eos_token_id,
    use_cache=True,
    bad_words_ids=[[processor.tokenizer.unk_token_id]],
    return_dict_in_generate=True,
)

sequence = processor.batch_decode(outputs.sequences)[0]
sequence = sequence.replace(processor.tokenizer.eos_token, '').replace(processor.tokenizer.pad_token, '')
sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()

print(processor.token2json(sequence))