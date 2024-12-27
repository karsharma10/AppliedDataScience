from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    df = pd.read_csv("../training/breast-cancer.csv")
    # Separate features and target
    y_label = df['diagnosis'].map({'M': 1, 'B': 0})  # Map diagnosis to binary labels
    x_labels = df.drop(columns=['diagnosis', 'id'])  # Drop target and unnecessary columns

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(x_labels, y_label, test_size=0.2, random_state=42)

    # Train a Random Forest
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)

    # Extract feature importances
    importances = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)

    # Plot feature importances
    plt.figure(figsize=(12, 6))
    importances.plot(kind='bar', title='Feature Importance')
    plt.xlabel("Features")
    plt.ylabel("Importance Score")

    # Save the plot to a file
    plt.savefig("feature_importance.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Select top features
    top_features = importances.nlargest(10).index
    X_train_top = X_train[top_features]
    X_test_top = X_test[top_features]

    print(f"Top selected features: {list(top_features)}")