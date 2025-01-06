from PIL import Image, ImageDraw, ImageFont
import random
import os

# Path to the text file containing message examples
TEXT_FILE_PATH = 'message_examples.txt'

# Read messages from the file
def load_message_examples(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file.readlines()]

# Load the list of message examples
MESSAGE_EXAMPLES = load_message_examples(TEXT_FILE_PATH)

# Set the font and size
FONT_PATH = "/Library/Fonts/Arial.ttf"  # Path to the font file
FONT_SIZE = 20
font = ImageFont.truetype(FONT_PATH, FONT_SIZE)

# Function to create a simple placeholder avatar image
def create_placeholder_avatar(output_path="images/placeholder_avatar.png"):
    # Create a simple placeholder image (e.g., a blue circle on a white background)
    img = Image.new("RGB", (40, 40), "white")
    draw = ImageDraw.Draw(img)
    draw.ellipse((5, 5, 35, 35), fill="blue", outline="black")
    img.save(output_path)

# Call this function to generate a placeholder avatar
create_placeholder_avatar()

# Function to generate synthetic Discord message images
def generate_message_image(message, avatar_path="images/placeholder_avatar.png", output_path=None):
    # Create a new blank image (adjust size as needed)
    img_width = 800
    img_height = 200
    img = Image.new("RGB", (img_width, img_height), (255, 255, 255))  # White background
    draw = ImageDraw.Draw(img)

    # Load avatar image and resize it
    avatar = Image.open(avatar_path)
    avatar = avatar.resize((40, 40))  # Size of avatar

    # Paste the avatar onto the image
    img.paste(avatar, (20, 20))

    # Draw the message text
    text_x = 70  # Starting position for the text
    text_y = 20
    draw.text((text_x, text_y), message, fill="black", font=font)

    # Optionally, add other elements like timestamps or user names (e.g. User#1234)
    draw.text((text_x, text_y + 30), "User#1234", fill="gray", font=font)

    # Save the generated image
    img.save(output_path)

# Generate a synthetic image for each message in the loaded examples
for idx, message in enumerate(MESSAGE_EXAMPLES):
    output_image_path = f"images/generated/generated_image_{idx + 1}.png"
    generate_message_image(message, output_path=output_image_path)
    print(f"Generated image saved to {output_image_path}")