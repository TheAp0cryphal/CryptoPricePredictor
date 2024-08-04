# Cryptocurrency Analysis

![image](https://github.com/user-attachments/assets/3fff3f4f-8485-46ea-9c8a-00f5d8edbe5e)

## How to Run Programs

### run reddit.py

Required Libraries: Pandas, Numpy, psaw, tqdm

This is the API interface we used to collect the reddit data. This can be run with:

  python3 reddit.py out_dir
  
When run this will collect all reddit comments that contain the word "bitcoin" between 2009 and 2012. 
This can be changed to find data about other coins or over a larger time span.

The directory produced by this code will not be used in later parts of this assignment as we will include relevant csv files


### run spark.py

Required Libraries: nltk

This program is what we used to run on our coin_Bitcoin.csv file which is quite large (~2.3Gb) we included a smaller 
reduced_sample.csv to run. 

This program is what runs the sentiment analysis. It will apply the sentiment analysis to the body of the comments. 
Afterwards the program will output directories for the data by average daily sentiment, and a directory of unaggregated data.

To run spark-submit spark.py reduced_sample.csv out_directory_average out_directory_all

The csv file in out_directory_average will be used for the regression training models, and the classification training models

The out_directory_all run with the Neural Net 

### run models.py

Required Libraries: Pandas, Numpy, sys, matplotlib, sklearn, re, seaborn, datetime, requests

BEFORE YOU RUN: create a folder called images

if you run:

python3 models.py coin_Bitcoin.csv DailyAverages.csv images

it will run all the models and put the images into the images folder

### neural.py

Required Libraries: Pandas, Numpy, sys, matplotlib, pathlib, sklearn, keras

This program will run in two different ways depending on how you call it

if you run:

python3 neural.py out_directory_all coin_Bitcoin.csv

then this will run the neural net on the provided sample data

python3 neural.py NeuralNetInput.csv

then this will run the neural net on the full dataset

NeuralNetInput.csv can be found at the provided google drive
https://drive.google.com/file/d/1RjXFVNA2X0g04F0SKgs4whEvMlNGnnwh/view?usp=sharing

### Run correlation.py
Required Libraries: Pandas, Numpy, sys, matplotlib, pathlib, scipy.stats

This program requires the coins .csv files to be located in the same directory level as the program.

Run the program by executing "python correlation.py" in the terminal.
Both the heatmap and bitcoin correlation visuals will be saved as jpgs.

Please note that the bitcoin correlation graph will also show when running the program. This is because the x-label is cut-off while saving it as a figure.

### Run precent_change.py
Required Libraries: Pandas, sys, matplotlib, pathlib, scipy.stats

This program requires the coins .csv files to be located in the same directory level as the program.

Run the program by executing "python percent_change.py [firstCoin] [secondCoin]" in the terminal. Ensure to indicate the two coins that are desired as sys arguments. The visuals will be saved as .jpgs.

Example: >python percent_change.py bitcoin polkadot



# Paper
![cryptoProjectreport-01](https://user-images.githubusercontent.com/70075553/180358452-2c5d4e85-758a-4ab8-8ca0-61a9ebed1149.jpg)
![cryptoProjectreport-02](https://user-images.githubusercontent.com/70075553/180358455-c2c0fdee-5e65-4113-9883-20de29dc5c0c.jpg)
![cryptoProjectreport-03](https://user-images.githubusercontent.com/70075553/180358459-53f85c87-f989-4af1-87ce-f0eb3c9451f3.jpg)
![cryptoProjectreport-04](https://user-images.githubusercontent.com/70075553/180358460-bc122a92-1f4b-46ba-959d-ba07a8874649.jpg)
![cryptoProjectreport-05](https://user-images.githubusercontent.com/70075553/180358461-8d41705c-16ff-49d0-b4c7-4eea3a6eb800.jpg)
![cryptoProjectreport-06](https://user-images.githubusercontent.com/70075553/180358464-c9df12b5-187d-437d-8994-eed3f043813d.jpg)
![cryptoProjectreport-07](https://user-images.githubusercontent.com/70075553/180358465-93722210-d354-446f-ae00-0a8258860985.jpg)
![cryptoProjectreport-08](https://user-images.githubusercontent.com/70075553/180358467-dba990bf-7637-440a-aa7e-b28a9eb933e1.jpg)
![cryptoProjectreport-09](https://user-images.githubusercontent.com/70075553/180358468-cb1919e2-1026-4c05-89ee-e12503ce47c3.jpg)



