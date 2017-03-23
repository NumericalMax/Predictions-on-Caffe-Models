# Predicting test samples with a Caffe model

Making predictions on a bunch of test images in Nvidia Digits can easily achieved by the contained function Classify Many Images. However the ouput format is frequently not in the supposed format. For instance taking part in machine learning competations require frequently a comma-seperated format as submission. According, this small python code creates such a format based on a trained Caffe model in Nvidia Digits.

Assume you have trained a Caffe Model in Nvidia Digits, then the following steps have to be performed:

  •	Download the Model at a suitable Epoch
  
  •	Execute the python code with suitable FLAGS.
    Exemplary:
    python createPrediction.py --image_dir=/home/user/data/dataset/test/*.jpg --prediction_file=/home/user/submissions/solution.txt --model_dir=/home/user/downloads/model/ --caffe_model=snapshot_iter_100.caffemodel

Note, that the prediction file has to be already existent by executing the code. Furthermore it is likely that adaptions in the python file has to be accomplished, especially in the part, in which the data transformer is defined.

Even though there is a FLAG for the batch size by now the batch size is set constantly to 1. In the next release this will be fixed, such that more than one testsample can be passed through the network at the same time.

For an overview of all FLAGS you just have to type python createPrediction.py -h
