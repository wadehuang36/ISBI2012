# My research project for ISBI 2012 challenge

This project use CNN to classify each pixel is border or not (binary classifier) with train images, then use the trained models to find the likelihood with test images, then pass the likelihood to do segmentation.

## Pre-requirements

1. [Caffe](http://caffe.berkeleyvision.org/), follow the install instruction, if it isn't in your machine. Notice, you have to add the path of the folder of caffe to PATH(system variable) and the path of of the folder of pycaffe to PYTHONPATH
2. [The images of ISBI 2012 challenges](http://brainiac2.mit.edu/isbi_challenge/home), download the images and place them in ./data folder
3. Python 2.7 with protobuf, numpy, tifffile, sklearn and matplotlib packages
4. Build the segment project and add the path of the folder to PATH. I am sorry, I cannot provide the source code because it is a private project by others.
5. JRE 8, if you want to rebuild Evaluation code, you have to install JDK 8.

To start, you can just type 
```bash
python All.py
```

the program will ask you which model do you want to run. Or you can give the specific model. 
```bash
python All.py --model={modelName}
```

the rule of finding model is ./models/{modelName}/config.json where {modelName} is the value you input.

## Tasks
1. Train.py, run Caffe train with pre-processed images.
2. Deploy.py, run Caffe classifer with trained-model generated from task 1.
3. Segment.py, run segmentation with likelihood generated from task 2.
4. Evaluation.py, run a java library to evaluate results from task 3.

All.py is just run all tasks together.

The execution log will be written in place of /models/{model}/results/{time}.log 

## Show Results
you can run
```bash
python Show.py --model={modelName}
```
to visualize results and use key ← and key → to switch images.
![Alt sample](sample.png)

## Configurations And Files
Each model must have four files
1. config.json, the configurations for the model.
2. solver.prototxt, the configurations of the solver for Caffe, the file name can be any but it must be matched in config.json.
3. train.prototxt, the configurations of train for Caffe, the file name can be any but it must be matched in solver.prototxt.
4. deploy.prototxt, the configurations of deploy for Caffe, the file name can be any but it must be matched in config.json.

### Properties of config.json
* solver, thd path of solver.prototxt
* modelPrototxt, thd path of deploy.prototxt
* trainedModel, thd path of the trained result by Caffe. The path can be any but it must be matched in solver.prototxt. 
* subImageSize, the size of each sub image. the sub image is a pixel with its neighbors, it will be floor(subImageSize/2) + a pixel + floor(subImageSize/2) for height and width. this value is used in Convert.py and Deploy.py, and it must be matched the shape in deploy.prototxt.  
* trainImages, the path of images for training.
* trainLabels, the path of labels for training.
* trainRange, how many images should be used for training. The format is Python format, for example, "range(0,15)" which means takes images from 0 to 15, but no 15, so it is 0 to 14 (equals to images\[0:15\])
* trainData, the path of the output LMDB database. The path can be any but it must be matched in train.prototxt.
* testImages, the path of images for testing. It can be null, if you don't want Caffe to run testing whiling training(solver.prototxt and train.prototxt have to be configured, too).
* testLabels, the path of labels for testing. It is nullable.
* testRange, how many images should be used for testing. The format is the same as trainRange. It is nullable.
* testData, the path of the output LMDB database. The path can be any but it must be matched in train.prototxt if tt is not null.
* deployImages, the path of images for deploying.
* deployLabels, the path of labels for deploying.
* deployRange, how many images should be used for deploying. The format is the same as trainRange.
* likelihood, the path of the result of deploying which is in numpy array format. 
* randomForestRange, how many images should be used for random forest training and it should be the sub range of deployRange. The format is the same as trainRange.
* segmentRange, how many images should be used for segmentRange and it should be the sub range of deployRange. The format is the same as trainRange.

For example,
```javascript
{
  "subImageSize" : 65,
  "trainImages" : "data/train-volume.tif",
  "trainLabels" : "data/train-labels.tif",
  "trainRange" : "range(0,15)",
  "trainData" : "data/train_65",
  "testImages" : null,
  "testLabels" : null,
  "testRange" : null,
  "testData" : null,
  "deployImages" : "data/train-volume.tif",
  "deployLabels" : "data/train-labels.tif",
  "deployRange" : "range(15,30)",
  "randomForestRange" : "range(15,25)",
  "segmentRange" : "range(25,30)",
  "solver" : "models/B/solver.prototxt",
  "modelPrototxt" : "models/B/deploy.prototxt",
  "trainedModel" : "models/B/results/B_iter_3932160.caffemodel.h5",
  "likelihood" : "models/B/results/likelihood.npy",
  "segment" : "models/B/results/segment.npy"
}
```
The every pixel of frame 1 to frame 15 of train-volume.tif will becomes 65x65 images and write into database train_65. No testing while training. Calculate likelihood with the same file train-volume.tif, but uses frame 16 to frame 30. And the results of frame 16 to frame 25 to be used for random forest training. And the results of frame 26 to frame 30 to be used for segmentation with the result of random forest training.  
 
P.S. There model A,B,C,D are referred this paper [Deep models for brain EM image segmentation: novel insights and improved performance.](https://www.ncbi.nlm.nih.gov/pubmed/27153603).

## Caution
The Segment code is writen by another team in C++, so it doesn't include in this repo(I might or might not change it to python). Also, this project is for personal learning, so the quality might not be good. 
 
## References

1. https://github.com/mjpekala/bio-segmentation
2. [Deep models for brain EM image segmentation: novel insights and improved performance.](https://www.ncbi.nlm.nih.gov/pubmed/27153603)
3. [Segmentation evaluation after border thinning](http://imagej.net/Segmentation_evaluation_after_border_thinning_-_Script)
