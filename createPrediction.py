import caffe
import glob
import numpy as np
import argparse
import sys
import os

def main(args):
	
	# Create required paths
	protoPath = FLAGS.model_dir + FLAGS.prototxt_file
	modelPath = FLAGS.model_dir + FLAGS.caffe_model
	testFiles = glob.glob(FLAGS.image_dir)
	testNames = [os.path.basename(x) for x in glob.glob(FLAGS.image_dir)]
	meanFile = FLAGS.model_dir + FLAGS.mean_file	
	labelPath = FLAGS.model_dir + FLAGS.label_file	
	
	# Write header to prediction file

	solutionFile = open(FLAGS.prediction_file, 'w')
	solutionFile.write('image_name,')
	labels = open(labelPath, 'r')
	with open(labelPath) as f:
    		labels = f.readlines()
	labels = [x.strip() for x in labels] 
	l = len(labels)
	i = 0
	while (i < l-1):
		solutionFile.write(labels[i] + ',')
		i = i + 1
	solutionFile.write(labels[l-1] + '\n')	
	
	# Prepare test data	
	if (FLAGS.GPU_mode == 1):
		caffe.set_mode_gpu()
	else:
		caffe.set_mode_cpu();
	
	# Load mean file
	blob = caffe.proto.caffe_pb2.BlobProto()
	data = open(meanFile, 'rb').read()
	blob.ParseFromString(data)
	arr = np.array(caffe.io.blobproto_to_array(blob))
	mu = arr[0]
	mu = mu.mean(1).mean(1)

	# Load Network
	net = caffe.Net(protoPath, modelPath, caffe.TEST)

	# Preprocessing Parameters / have to be adjusted
	transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
	transformer.set_transpose('data', (2,0,1))
	transformer.set_raw_scale('data', 255)	
	transformer.set_channel_swap('data', (2,1,0))	
	transformer.set_mean('data', mu)

	#net.blobs['data'].reshape(3, 224, 224)
	net.blobs['data'].reshape(1, 3, 227, 227)
	# Predict Images
	ll = len(testFiles) #/ FLAGS.batch_size;
	print(str(ll) + ' Files are tested!')
	for k in range(0,ll):
		
		input_image = caffe.io.load_image(testFiles[k])
		transformed_image = transformer.preprocess('data', input_image)
		net.blobs['data'].data[...] = transformed_image
		output = net.forward()
		print(testNames[k])
		print(output)
		prediction = output['softmax'][0]
		j = 0
		solutionFile.write(testNames[k] + ',')
		while (j < l-1):
			solutionFile.write(str(prediction[j]) + ',')
			j = j + 1
		solutionFile.write(str(prediction[l-1]) + '\n')	

	solutionFile.close();
	

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--image_dir',
		type=str,
		help='Path to image folder'
	)
	parser.add_argument(
		'--prediction_file',
		type=str,
		help='Path destination of prediction file (.txt format)'	
	)
	parser.add_argument(
		'--model_dir',
		type=str,
		help='Path to model folder'	
	)
	parser.add_argument(
		'--prototxt_file',
		type=str,
		default='deploy.prototxt',
		help='Name of prototxt file'	
	)
	parser.add_argument(
		'--mean_file',
		type=str,
		default='mean.binaryproto',
		help='Name of mean Image file'	
	)
	parser.add_argument(
		'--label_file',
		type=str,
		default='labels.txt',
		help='Name of label file'	
	)
	parser.add_argument(
		'--caffe_model',
		type=str,
		default='snapshot_iter_264.caffemodel',
		help='Name of caffe model'	
	)
	parser.add_argument(
		'--batch_size',
		type=int,
		default=1,
		help='Number of simultanous evaluation.'	
	)
	parser.add_argument(
		'--GPU_mode',
		type=int,
		default=0,
		help='GPU (1) or CPU (0) mode.'	
	)
	FLAGS, unparsed = parser.parse_known_args()
	main([sys.argv[0]] + unparsed)
