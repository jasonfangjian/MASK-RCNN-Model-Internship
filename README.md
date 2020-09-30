# MASK-RCNN-Model-Internship

## Summary
* Implemented the Mask RCNN model on Python 3, Keras and TensorFlow to extract the building footprint automatically (instance segmentation) from historical maps.
* Labeled the buildings (target objects) in the historical maps and generated a training data set containing approximately 18,000 images.
* Adjusted and trained the Mask RCNN with the average precision: 0.964 and extracted more than 121,000 building footprints from 14 maps.
* Implemented a polyline compression algorithm in Python to correct distortions in building footprint extracted from historical maps

## Test Result
1. Picture (Map) before detection <br>
/[picture1](./assets/test.png)
2. Pciture (Building footprints) extracted after detection <br>
