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
### Tuning
python pytorch_training.py --mode tune --n-trials 10  
The default for number of trials is 10  
All runs with a saved checkpoint are logged in training_report.txt. Epoch stats are also logged to checkpoint_history.txt     
Also logs what it considers the single best run to hyperparameter_study.txt, but I encourage exploring the other logs as well while tuning.    

### Train new model (default)
python pytorch_training.py  
All runs with a saved checkpoint are logged in checkpoint_history.txt as well as a record of each epoch

### Train from checkpoint
python pytorch_training.py --mode train --model checkpoint.pth
Logs to checkpoint_history.txt

### Single image prediction
python pytorch_training.py --mode predict --model model.pth --image test.jpg  
Logs to training_report.txt

### Directory prediction
python pytorch_training.py --mode predict --model model.pth --dir test_images/  
Logs to training_report.txt

### Run all models in a directory against test images
python pytorch_training.py --mode predict --models-dir models/ --dir test_images/
Logs to multi_model_predictions.txt