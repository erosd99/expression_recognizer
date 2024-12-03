# expression_recognizer
Simple facial expression recognizer written in python

# How to run

1. Install dependencies

    ```
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```
2. Either download the pretrained model, or train it yourself
    1. Pretrained model: https://drive.google.com/file/d/1pWIoF5frgDLc7kvFg4gj67qtMoRJkp5F/view?usp=sharing
        Copy the pretrained model into the **data** folder
    2. Train yourself: 
        1. Download and extract the (https://www.kaggle.com/code/gauravsharma99/facial-emotion-recognition/input?select=fer2013)[fer2013.tar.gz] dataset into the **data** directory
        2. Run the **src/cnn.py** script. **WARNING:** The training is pretty slow, for me it took ~5 hours.
    
3. After the model is trained, run the main script. 
    1. Put your desired video in the **data** folder.
    2. At the bottom of the **src/face_tracker.py** file, replace "happy_man.mp4" with the name of your video clip.
    3. Run **src/face_tracker.py**
