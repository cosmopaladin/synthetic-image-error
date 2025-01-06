import os
import random
from PIL import Image, ImageDraw, ImageFont
import textwrap
import shutil

# Configurations
FONT_PATH = "path/to/font/Roboto-Regular.ttf"
AVATAR_PATH = "path/to/avatar.jpg"
EMOJI_PATH = "path/to/emoji.png"
OUTPUT_DIR = "synthetic_data"
BACKGROUND_COLORS = [(54, 57, 63), (255, 255, 255)]
TEXT_COLORS = [(255, 255, 255), (0, 0, 0)]
MESSAGE_EXAMPLES = [
    "Hello, how are you?",
    "This is a sample Discord message.",
    "Random text for dataset generation.",
    "Machine learning is amazing!",
    "This message is unaltered.",
]
NUM_MESSAGES = 500

# Create output directories
os.makedirs(f"{OUTPUT_DIR}/train/unaltered", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/train/altered", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/val/unaltered", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/val/altered", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/test/unaltered", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/test/altered", exist_ok=True)

def generate_synthetic_message(output_path, altered=False):
    width, height = 800, random.randint(100, 150)
    background_color = random.choice(BACKGROUND_COLORS)
    text_color = TEXT_COLORS[BACKGROUND_COLORS.index(background_color)]
    image = Image.new("RGB", (width, height), color=background_color)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(FONT_PATH, size=24)

    # Add avatar
    avatar = Image.open(AVATAR_PATH).resize((50, 50))
    image.paste(avatar, (10, 25))

    # Add username and timestamp
    username = f"User_{random.randint(1000, 9999)}"
    draw.text((70, 10), username, font=font, fill=(114, 137, 218))
    timestamp = "Today at " + f"{random.randint(1, 12)}:{random.randint(0, 59):02d} PM"
    draw.text((70 + draw.textsize(username, font=font)[0] + 10, 10), timestamp, font=font, fill=(142, 146, 151))

    # Add message text
    message_text = random.choice(MESSAGE_EXAMPLES)
    wrapped_text = textwrap.fill(message_text, width=60)
    draw.text((70, 50), wrapped_text, font=font, fill=text_color)

    if altered:
        # Apply alteration
        alteration_type = random.choice(["edit_text", "remove_avatar", "add_noise", "add_element"])
        if alteration_type == "edit_text":
            draw.rectangle((70, 50, width - 10, height - 10), fill=background_color)
            new_text = random.choice(["Fake text", "Altered message!", "Tampered!"])
            draw.text((70, 50), new_text, font=font, fill=text_color)
        elif alteration_type == "remove_avatar":
            draw.rectangle((10, 25, 60, 75), fill=background_color)
        elif alteration_type == "add_noise":
            for _ in range(500):
                x, y = random.randint(0, width - 1), random.randint(0, height - 1)
                draw.point((x, y), fill=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
        elif alteration_type == "add_element":
            emoji = Image.open(EMOJI_PATH).resize((20, 20))
            image.paste(emoji, (random.randint(70, width - 30), height - 30))

    image.save(output_path)

# Generate dataset
def generate_dataset(split_ratios=(0.7, 0.2, 0.1)):
    all_images = []
    for i in range(NUM_MESSAGES):
        altered = random.random() < 0.5
        subfolder = "altered" if altered else "unaltered"
        all_images.append((f"{OUTPUT_DIR}/{subfolder}/message_{i}.png", altered))

    random.shuffle(all_images)
    train_split = int(split_ratios[0] * NUM_MESSAGES)
    val_split = int(split_ratios[1] * NUM_MESSAGES)

    for i, (image_path, altered) in enumerate(all_images):
        split = "train" if i < train_split else "val" if i < train_split + val_split else "test"
        subfolder = "altered" if altered else "unaltered"
        generate_synthetic_message(f"{OUTPUT_DIR}/{split}/{subfolder}/message_{i}.png", altered)

generate_dataset()