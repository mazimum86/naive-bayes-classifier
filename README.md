# Naive Bayes Classifier

This repository contains an implementation of the Naive Bayes classifier, a probabilistic machine learning model used for classification tasks. Naive Bayes is based on applying Bayes' theorem with the assumption of independence between every pair of features.

## Dataset

The dataset used for this implementation is designed for classification tasks, where each data point belongs to a specific category. The target variable is categorical, and the features are either continuous or categorical.

## Contents

- **naive_bayes_classifier.py**: The main script that implements the Naive Bayes classifier, including data preprocessing, model training, and prediction.
- **data.csv**: The dataset file used for training and testing the model.
- **requirements.txt**: A list of Python dependencies required to run the code.

## Implementation Details

The implementation follows these steps:

1. **Data Loading**: The dataset is loaded from `data.csv` and split into features and target variables.
2. **Data Preprocessing**: Preprocessing steps such as handling missing values and encoding categorical features are applied.
3. **Model Training**: The Naive Bayes classifier is trained using the preprocessed data. Various types of Naive Bayes models, such as Gaussian, Multinomial, and Bernoulli, can be used depending on the nature of the features.
4. **Prediction**: The trained model is used to predict the class labels on the test dataset.
5. **Evaluation**: The model's performance is evaluated using metrics such as accuracy, precision, recall, and F1-score. The confusion matrix is also provided to give insight into the classification results.

## Usage

To use this code, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/mazimum86/naive-bayes-classifier.git
    ```
2. Navigate to the repository directory:
    ```bash
    cd naive-bayes-classifier
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. Run the Naive Bayes Classifier script:
    ```bash
    python naive_bayes_classifier.py
    ```

## Dependencies

The following Python packages are required to run the code:

- pandas
- scikit-learn
- numpy
- matplotlib

These dependencies are listed in the `requirements.txt` file and can be installed using `pip`.

## Results

The output includes:

- **Predicted Labels**: The class labels predicted by the model on the test dataset.
- **Evaluation Metrics**: Metrics like accuracy, precision, recall, and F1-score to assess the model's performance.
- **Confusion Matrix**: A confusion matrix that provides a detailed breakdown of correct and incorrect classifications.

## Contributing

Contributions are welcome! If you have any ideas for improvements or additional features, feel free to submit a pull request or open an issue.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
