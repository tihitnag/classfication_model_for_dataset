# Classification of Titanic Dataset using Neural Network, Logistic Regression, and Decision Tree 

Classification is a fundamental task in machine learning, and it involves categorizing data into predefined classes or categories. In this repositoryI have tried to implementthe classification of the famous Titanic dataset using three different algorithms: Neural Network, Logistic Regression, and Decision Tree. The Titanic dataset contains information about passengers on the Titanic ship, including their demographics and whether they survived the disaster. The goal is to build models that can accurately predict the survival outcome based on the given features.
# Understanding the Dataset
The Titanic dataset consists of various attributes such as age, gender, ticket class, fare, cabin, and others. The target variable is the "Survived" column, which indicates whether a passenger survived (1) or not (0).
# Preprocessing 
Before applying any classification algorithm, it is crucial to perform data preprocessing, including handling missing values, changing the data into one hot encoding, and data normalization
### Data preprocessing
x_data = data.drop(['Name', 'Ticket', 'Cabin'], axis=1)
x_data['Sex'] = x_data['Sex'].replace(['male', 'female'], [1, 0])
x_data['Embarked'].fillna('S', inplace=True)
x_data['Age'].fillna(30, inplace=True)
x_data['Embarked'] = x_data['Embarked'].replace(['S', 'C', 'Q'], [0, 1, 2])

### Split the data into training and testing sets
train_data, test_data = train_test_split(x_data, test_size=0.2, random_state=25)

### Convert data to numpy arrays
y_train = train_data['Survived'].values
x_train = train_data.drop(['Survived'], axis=1).values
y_test = test_data['Survived'].values
x_test = test_data.drop(['Survived'], axis=1).values
# Logistic Regression 
Logistic Regression is a popular classification algorithm that works well for binary classification problems. It models the relationship between the input features and the binary target variable using a logistic function. Logistic regression calculates the probability of an instance belonging to a particular class and then applies a threshold to make the final classification decision.
To apply logistic regression to the Titanic dataset, we preprocess the data, splitting it into a training set and a test setas we did in the above example.Then We train the logistic regression model using the training set and evaluate its performance using metrics such as accuracy, presission recall and F1-score.In this repository I have used accuracy as an evaluation metrics. The model predicts the survival outcome of passengers based on the given features.
# Neural Network
Neural networks, specifically deep learning models, have gained significant popularity in recent years due to their ability to learn complex patterns in data. For the Titanic dataset classification, we can design a neural network architecture with multiple layers, including input, hidden, and output layers. Each layer consists of nodes or neurons that perform computations and pass information to the next layer. The final layer uses an activation function to produce the predicted survival outcome.
Similar to logistic regression, we preprocess the Titanic dataset and split it into training and test sets. We then train the neural network model using the training data, tuning hyperparameters such as the number of layers, the number of neurons in each layer, and the learning rate. Here I have used 2 layers and 8 neurons to train the data.  After training, we evaluate the model's performance on the test set and analyze metrics such as accuracy and score.
# Decision Tree
A Decision Tree is a simple yet powerful classification algorithm that uses a tree-like model of decisions and their possible consequences. Each internal node represents a feature or attribute, each branch represents a decision rule, and each leaf node represents the outcome or class label. Decision trees are known for their interpretability and the ability to handle both numerical and categorical data.
To apply a decision tree to the Titanic dataset, we preprocess the data and split it into training and test sets like the above two then We train the decision tree model using the training set and evaluate its performance using metrics such as accuracy, and score. The model predicts the survival outcome based on the given features and the decision rules learned from the training data.
# Comparing The Results
Once we have trained and evaluated the logistic regression, neural network, and decision tree models, we can compare their performance on the Titanic dataset. We consider accuracy to determine which algorithm performs better in predicting the survival outcome. 
##### Neural Network - 0.8221% accuracy
##### decision tree - 0.815% accuracy
##### Logistic Regression - 0.815% accuracy
 Based on the accuracy metric, the neural network model appears to be the most effective in predicting the survival outcome on the Titanic dataset. It achieved a slightly higher accuracy than the other models.
