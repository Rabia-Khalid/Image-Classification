!unzip clf-data.zip
import os
import pickle
from skimage.io import imread
from skimage.transform import resize
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Prepare data
input_dir = r'./clf-data'
categories = ['empty', 'not_empty']

data = []
labels = []
for category_idx, category in enumerate(categories):
    for file in os.listdir(os.path.join(input_dir, category)):
        img_path = os.path.join(input_dir, category, file)
        img = imread(img_path)
        img = resize(img, (15, 15))
        data.append(img.flatten())
        labels.append(category_idx)

data = np.asarray(data)
labels = np.asarray(labels)

# Count the number of samples in each category
unique, counts = np.unique(labels, return_counts=True)
print("Number of samples per category:")
for category, count in zip(categories, counts):
    print(f"{category}: {count}")

# Train/test split
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

print("\nData shapes:")
print("data:", data.shape)
print("labels:", labels.shape)
print("x_train:", x_train.shape)
print("x_test:", x_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

# Train classifier
classifier = SVC()
parameters = [{'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100, 1000]}]
grid_search = GridSearchCV(classifier, parameters, cv=5, verbose=1)

print("\nTraining classifier...")
grid_search.fit(x_train, y_train)
print("Training complete.")

# Test performance
best_estimator = grid_search.best_estimator_
y_prediction = best_estimator.predict(x_test)

score = accuracy_score(y_prediction, y_test)
print(f'\nOverall accuracy: {score * 100:.2f}%')

# Classification report
report = classification_report(y_test, y_prediction, target_names=categories)
print("\nClassification Report:")
print(report)

# Count correctly classified samples per category
correct_counts = np.sum(y_prediction == y_test)
print(f"\nNumber of correctly classified samples: {correct_counts}/{len(y_test)}")

correct_empty = np.sum((y_test == 0) & (y_prediction == 0))
correct_not_empty = np.sum((y_test == 1) & (y_prediction == 1))

print("\nCorrectly classified samples per category:")
print(f"empty: {correct_empty}")
print(f"not_empty: {correct_not_empty}")

# Save the model
pickle.dump(best_estimator, open('./model.p', 'wb'))

# Plot Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_prediction)
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(categories))
plt.xticks(tick_marks, categories, rotation=45)
plt.yticks(tick_marks, categories)

# Add the numbers inside the boxes
thresh = conf_matrix.max() / 2.
for i, j in np.ndindex(conf_matrix.shape):
    plt.text(j, i, format(conf_matrix[i, j], 'd'),
             horizontalalignment="center",
             color="white" if conf_matrix[i, j] > thresh else "black")

plt.ylabel('True Labels')
plt.xlabel('Predicted Labels')
plt.tight_layout()
plt.show()

# Plot Grid Search results
results = grid_search.cv_results_
plt.figure(figsize=(12, 8))
plt.title("Grid Search Scores", fontsize=16)
plt.xlabel("Parameter Combination Index")
plt.ylabel("Mean Test Score")

# Visualize scores
scores = results['mean_test_score']
indices = range(len(scores))
plt.plot(indices, scores, marker='o', linestyle='--')

# Annotate each point with the corresponding parameter combination
for i, (mean_score, params) in enumerate(zip(scores, results['params'])):
    plt.annotate(f"{params}", (indices[i], scores[i]), fontsize=8)

plt.show()
