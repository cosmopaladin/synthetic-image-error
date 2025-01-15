# synthetic-image-error
Project 4 of data science bootcamp. 

Synthetic-image-error (Pronounced, Sigh). Named for the feeling you will have everytime you encounter a problem this is meant to solve. 

This project generate synthetic discord message screenshots. Creates artificial errors in them. With the intention to train a model that can evaluate if a discord message has been artificially altered.

## Generating Synthetic data
1. Make the folders the script expects. Which is a discord_chats folder with an altered, gen, to_alter subfolder
2. Make an images folder with avatars subfolder
3. Put avatars of 128x128 pixel generate how you see fit in this folder. I used stabile diffusion 1.5
4. Run scripts in this order
    1. text_generate.py 
    2. improved_generator.py
    3. move_images.py
    4. distort.py
    5. run pytorch_training.py, command line options described below

## Command line training
### Train new model (default)
python pytorch_training.py

### Train from checkpoint
python pytorch_training.py --model checkpoint.pth --mode train

### Predict single image
python pytorch_training.py --model checkpoint.pth --mode predict --image test.jpg

### Tuning
python pytorch_training.py --mode tune --n-trials 10
The default for number of trials is 10