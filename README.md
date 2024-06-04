# Image Captioning Project

This project aims to develop image captioning models using recurrent neural networks (RNNs) and attention mechanisms used in LSTM. The models are trained on the COCO dataset, a large-scale dataset of images with corresponding captions.

## Table of Contents

- [Project Description](#project-description)
- [How to Run the Project](#how-to-run-the-project)
- [Badges](#badges)
- [Conclusion](#conclusion)

## Project Description

The project consists of the following components:

- Data downloading and preprocessing
- Minibatch visualization
- Overfitting test
- RNN model training
- Attention LSTM model training
- Result visualization

## How to Run the Project

To run the project, follow these steps:

1. Install the required dependencies.
2. Clone the repository.
3. Download [coco.py](http://web.eecs.umich.edu/~justincj/teaching/eecs498/coco.pt) and save in datasets directory
4. Use the ipynb notebook to train the models and show the result.

## Results

![image](https://github.com/Shnekels/Caption_RNN_Study/assets/146712800/e74b824f-0bee-482f-ab2d-90042cd04179)
![image](https://github.com/Shnekels/Caption_RNN_Study/assets/146712800/d37bac00-6195-4c5d-9874-8090beb56357)

![image](https://github.com/Shnekels/Caption_RNN_Study/assets/146712800/9e07963b-6462-4ea5-95e6-da5e651decf8)
![image](https://github.com/Shnekels/Caption_RNN_Study/assets/146712800/1a5b5a5b-a922-4eec-a913-d161ec614e75)

We can clearly see that the results on the train data are much better then on the val data.


## Badges

- [![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/aleju/image-captioning/blob/master/LICENSE)

## Conclusion

This project provided a comprehensive overview of image captioning models and their implementation. The user gained hands-on experience with data preprocessing, model training, and result visualization. The project also highlighted the importance of attention mechanisms in improving the performance of image captioning models.
