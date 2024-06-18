import json
import numpy as np
from PIL import Image, ImageDraw


def json_to_mask(json_file, output_mask_file, image_size):
    # Load the JSON data
    with open(json_file) as f:
        data = json.load(f)

    # Create a blank image for the mask
    mask = Image.new('L', (image_size[0], image_size[1]), 0)
    draw = ImageDraw.Draw(mask)

    # Iterate through all shapes in the JSON file
    for shape in data['shapes']:
        points = shape['points']
        polygon = [(point[0], point[1]) for point in points]
        draw.polygon(polygon, outline=1, fill=1)

    # Save the mask image
    mask.save(output_mask_file)


# Example usage
json_file = 'D:/pycharm_project/intrusion-detection/mask.json'
output_mask_file = 'D:/pycharm_project/intrusion-detection/mask.png'
image_size = (1920, 1080)  # Replace with the actual size of your images

json_to_mask(json_file, output_mask_file, image_size)
