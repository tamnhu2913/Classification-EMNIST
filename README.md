## Classify the handwriting of EMNIST
- The EMNIST dataset is an extension of the MNIST dataset, which includes more handwriting of letters and digits. This project focuses on classifying letters from the EMNIST using a CNN model.
- The datasets are downloaded from *Kaggle*: https://www.kaggle.com/datasets/crawford/emnist/data. The main dataset is the letters EMINST subset, which consist 103600 characters with 26 classes. Each image is 28x28 pixels, grayscale.
- The CNN model used in this project included the following layes: Convolutional layers with ReLU activation, Max-pooling layers, Fully connected (dense) layers and Dropout layers for regularization. The model is trained using the Adam optimizer and categorical cross-entropy loss.
