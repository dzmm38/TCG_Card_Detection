import cv2
import os
from tqdm import tqdm

base_path = '../data/set_1/'
width = 250
height = 350

x1_art, y1_art = 12,15
x2_art, y2_art = 233,15
x3_art, y3_art = 12,178
x4_art, y4_art = 233,178

for image_name in tqdm(os.listdir(base_path)):
    image_path = os.path.join(base_path, image_name)
    image = cv2.imread(image_path)
    image = image[y1_art:y3_art, x1_art:x2_art]
    # image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    cv2.imwrite(image_path, image)
