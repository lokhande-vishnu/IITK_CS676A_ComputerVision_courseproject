Topic: Weakly supervised object detection using CNN

Papers followed:
1) Oquab, Maxime, et al. "Is object localization for free?-
weakly-supervised learning with convolutional neural
networks." Proceedings of the IEEE Conference on
Computer Vision and Pattern Recognition. 2015.
2) Oquab, Maxime, et al. "Learning and transferring mid-level
image representations using convolutional neural
networks." Proceedings of the IEEE Conference on
Computer Vision and Pattern Recognition. 2014.
3) Lempitsky, Victor, and Andrew Zisserman. "Learning to
count objects in images." Advances in Neural Information
Processing Systems. 2010.

Group members:
L S Vishnu Sai Rao (12376) and Saurabh Kataria (12637)

Platform and tools used:
Torch7 - https://github.com/torch/torch7
Linux
MATLAB
Nvidia GPU and its CUDA

Datasets used:
PASCAL VOC 2012
PASCAL VOC 2007

Dependencies of code:
Common libraries of Torch7
OpenBLAS
Nvidia CUDA latest 6.5+

Directions to execute the code:
1) Run 'generatedataset-trainval-full-weak-500.lua' and 'generatedataset-test-full-weak-500.lua' to generate training and testing data in t7 format.
2) Run 'post-train.lua' on a system with capable GPU and memory to train the CNN model.
3) Run 'run-on-test-set.lua' to create maxpooledscores (output) by testing the trained model on test data (in t7 format).
4) Run 'confusionmatrix.lua' to generate the accuracies of all classes calculated from the confusion matrices obtained for each class.

Index of brief description of each script:
1) confusionmatrix.lua - generates the confusion matrix for each class and calculate the accuracies too.
2) featureextractor.lua - extracts the features from an image and store it into t7 format
3) generatedataset-test-full-weak-500.lua - converts test image set into batches of t7 format
4) generatedataset-trainval-full-weak-500.lua - converts training image set into batches of t7 format
5) post-train.lua - trains the CNN
6) run-on-test-set.lua - create maxpooledscores (output) by testing the trained model on test data (in t7 format)
