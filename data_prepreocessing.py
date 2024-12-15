from PIL import Image

def ConvertGrayImage(url_input, url_output):
    image = Image.open(url_input)
    gray_image = image.convert('L')
    gray_image.save(url_output)
    
import os
folder_path = './data'
file_names = os.listdir(folder_path)
for file_name in file_names:
    url_input = './data/'+ file_name
    url_output = './targeted_gray_image/'+ file_name
    ConvertGrayImage(url_input, url_output)