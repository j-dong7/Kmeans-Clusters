import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


with open('behavior-performance.txt', 'r') as f:
    raw_data = [x.strip().split('\t') for x in f.readlines()]
df = pd.DataFrame.from_records(raw_data[1:], columns=raw_data[0])
df['VidID']       = pd.to_numeric(df['VidID'])
df['fracSpent']   = pd.to_numeric(df['fracSpent'])
df['fracComp']    = pd.to_numeric(df['fracComp'])
df['fracPlayed']  = pd.to_numeric(df['fracPlayed'])
df['fracPaused']  = pd.to_numeric(df['fracPaused'])
df['numPauses']   = pd.to_numeric(df['numPauses'])
df['avgPBR']      = pd.to_numeric(df['avgPBR'])
df['stdPBR']      = pd.to_numeric(df['stdPBR'])
df['numRWs']      = pd.to_numeric(df['numRWs'])
df['numFFs']      = pd.to_numeric(df['numFFs'])
df['s']           = pd.to_numeric(df['s'])
dataset_1 = df

# Question 1
# Filter data for students who completed at least five videos
video_counts = df['userID'].value_counts()
valid_students = video_counts[video_counts >= 5].index
df_filtered = df[df['userID'].isin(valid_students)]

# Select relevant features
features = ['fracSpent', 'fracComp', 'fracPaused', 'numPauses', 'avgPBR', 'numRWs', 'numFFs']
X = df_filtered[features]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine the optimal number of clusters using the Elbow method (k = 8 because significant bend)
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot Elbow method graph
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Apply KMeans with the chosen number of clusters
kmeans = KMeans(n_clusters=8, random_state=42)
df_filtered['cluster'] = kmeans.fit_predict(X_scaled)

# Calculate the mean and standard deviation of each feature within each cluster
cluster_summary = df_filtered.groupby('cluster').agg({feature: ['mean', 'std'] for feature in features})
print(cluster_summary)

print(df_filtered.head(35))


# Question 2
video_counts = df['userID'].value_counts()
valid_students = video_counts[video_counts >= 5].index
df_filtered = df[df['userID'].isin(valid_students)]

# Agg data for each student
df_aggregated = df_filtered.groupby('userID').agg({
    'fracSpent': 'mean',
    'fracComp': 'mean',
    'fracPaused': 'mean',
    'numPauses': 'mean',
    'avgPBR': 'mean',
    'numRWs': 'mean',
    'numFFs': 'mean',
    's': 'mean'  # Average score
}).reset_index()

# Prepare features and target
features = ['fracSpent', 'fracComp', 'fracPaused', 'numPauses', 'avgPBR', 'numRWs', 'numFFs']
X_agg = df_aggregated[features]
y_agg = df_aggregated['s']

# Standardize the features
scaler = StandardScaler()
X_agg_scaled = scaler.fit_transform(X_agg)

# Split the data into training and testing sets
X_train_agg, X_test_agg, y_train_agg, y_test_agg = train_test_split(X_agg_scaled, y_agg, test_size=0.3, random_state=42)

# Trains model
reg = LinearRegression()
reg.fit(X_train_agg, y_train_agg)

# Predict test
y_pred_agg = reg.predict(X_test_agg)

# Evaluate model
mse = mean_squared_error(y_test_agg, y_pred_agg)
r2 = r2_score(y_test_agg, y_pred_agg)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

#Question 3
# Prepare features and target
features = ['fracSpent', 'fracComp', 'fracPaused', 'numPauses', 'avgPBR', 'numRWs', 'numFFs']
X = df[features]
y = df['s']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a logistic regression model
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

# Predict on the test set
y_pred = log_reg.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print the evaluation metrics
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# Print confusion matrix and classification report
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

