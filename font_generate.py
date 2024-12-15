from PIL import Image, ImageDraw, ImageFont
import os
import re

def generate_img(text, font_url, output_url):
    img = Image.new('RGB', (500, 200), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    font_size = int(0.6 * img.height)
    font = ImageFont.truetype(font_url, size=font_size)
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    x = (img.width - text_width) // 2
    y = (img.height - text_height) // 2
    draw.text((x, y), text, fill=(0, 0, 0), font=font)
    img.save(output_url)

folder_path = './targeted_image/'
file_names = os.listdir(folder_path)

for file in file_names:
    match = re.match(r'(.+)\.png$', file)
    if match:
        text = match.group(1)
        output_url = f'./generated_image/{text}.png' 
        font_path = './fonts/FangZhengFangSong-GBK-1.ttf'
        generate_img(text, font_path, output_url)



