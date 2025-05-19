# Install sklearn if not already installed (Colab usually has it)
!pip install -q scikit-learn pandas

# Import required libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn import preprocessing

# Load the dataset from an online source
url = "https://raw.githubusercontent.com/krishnaik06/Decision-Tree-Classification/master/play_tennis.csv"
df = pd.read_csv(url)

# Display dataset
print("Dataset:\n", df)

# Convert categorical variables to numbers using LabelEncoder
le = preprocessing.LabelEncoder()
for column in df.columns:
    df[column] = le.fit_transform(df[column])

# Separate features and target
X = df.drop('PlayTennis', axis=1)
y = df['PlayTennis']

# Train ID3 (using entropy)
model = DecisionTreeClassifier(criterion='entropy')
model.fit(X, y)

# Show the tree as text
tree_rules = export_text(model, feature_names=list(X.columns))
print("\nDecision Tree (ID3) Rules:\n")
print(tree_rules)
