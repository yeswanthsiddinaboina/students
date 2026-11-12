import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


class StudentPerformance:

    def __init__(self):
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.y_pred = None

        self.base_path = Path(__file__).resolve().parent
        self.data_path = self.base_path / "Data" / "Student_Performance.csv"
        self.model_path = self.base_path / "model.pkl"

    # Step 1: Load Dataset
    def load_dataset(self):
        try:
            self.data = pd.read_csv(self.data_path)
            print("\nDataset Loaded Successfully!")
            print(self.data.head())

        except Exception as e:
            print("Dataset Loading Error:", e)

    # Step 2: Explore Dataset
    def explore_data(self):
        try:
            print("\n--- Dataset Information ---")

            print("\nFirst 5 Rows:")
            print(self.data.head())

            print("\nDataset Shape:")
            print(self.data.shape)

            print("\nNull Values:")
            print(self.data.isnull().sum())

            print("\nData Types:")
            print(self.data.dtypes)

            print("\nStatistical Description:")
            print(self.data.describe())

        except Exception as e:
            print("Data Exploration Error:", e)

    # Step 3: Data Preprocessing
    def preprocess_data(self):
        try:
            self.data["Extracurricular Activities"] = (
                self.data["Extracurricular Activities"]
                .astype(str)
                .str.strip()
                .map({"Yes": 1, "No": 0})
            )

            if self.data.isnull().sum().any():
                raise ValueError("Missing or invalid values found!")

            self.X = self.data.drop("Performance Index", axis=1)
            self.y = self.data["Performance Index"]

            print("\n--- Data Preprocessing ---")

            print("\nIndependent Variables (X):")
            print(self.X.head())

            print("\nDependent Variable (y):")
            print(self.y.head())

            print("\nX Shape:", self.X.shape)
            print("y Shape:", self.y.shape)

            print("\nPreprocessing Completed!")

        except Exception as e:
            print("Preprocessing Error:", e)

    # Step 4: Data Visualization
    def visualize_data(self):
        try:
            print("\n--- Data Visualization ---")

            # Graph 1: Correlation Heatmap
            plt.figure(figsize=(10, 7))

            correlation = self.data.corr(numeric_only=True)

            sns.heatmap(
                correlation,
                annot=True,
                cmap="coolwarm",
                fmt=".2f"
            )

            plt.title("Student Performance Correlation Heatmap")
            plt.tight_layout()

            plt.savefig(
                self.base_path / "correlation_heatmap.png"
            )

            plt.show()

            # Graph 2: Study Hours vs Performance
            plt.figure(figsize=(8, 5))

            sns.scatterplot(
                data=self.data,
                x="Hours Studied",
                y="Performance Index"
            )

            plt.title("Study Hours vs Performance Index")
            plt.xlabel("Hours Studied")
            plt.ylabel("Performance Index")
            plt.tight_layout()

            plt.savefig(
                self.base_path / "study_hours_performance.png"
            )

            plt.show()

            # Graph 3: Performance Index Distribution
            plt.figure(figsize=(8, 5))

            sns.histplot(
                data=self.data,
                x="Performance Index",
                bins=20,
                kde=True
            )

            plt.title("Performance Index Distribution")
            plt.xlabel("Performance Index")
            plt.ylabel("Number of Students")
            plt.tight_layout()

            plt.savefig(
                self.base_path / "performance_distribution.png"
            )

            plt.show()

            # Observations
            print("\n--- Data Analysis Observations ---")

            print(
                "Average Performance Index:",
                self.data["Performance Index"].mean()
            )

            print(
                "Hours Studied Correlation:",
                correlation.loc[
                    "Hours Studied", "Performance Index"
                ]
            )

            print(
                "Previous Scores Correlation:",
                correlation.loc[
                    "Previous Scores", "Performance Index"
                ]
            )

            print("\nAll Visualizations Generated Successfully!")

        except Exception as e:
            print("Visualization Error:", e)

    # Step 5: Train Test Split
    def split_data(self):
        try:
            (
                self.X_train,
                self.X_test,
                self.y_train,
                self.y_test
            ) = train_test_split(
                self.X,
                self.y,
                test_size=0.2,
                random_state=42
            )

            print("\n--- Train Test Split ---")
            print("Training Data:", self.X_train.shape)
            print("Testing Data:", self.X_test.shape)

        except Exception as e:
            print("Split Error:", e)

    # Step 6: Model Training
    def train_data(self):
        try:
            self.model = LinearRegression()

            self.model.fit(
                self.X_train,
                self.y_train
            )

            self.y_pred = self.model.predict(self.X_test)

            print("\nModel Trained Successfully!")

        except Exception as e:
            print("Training Error:", e)

    # Step 7: Model Evaluation
    def evaluate_model(self):
        try:
            r2 = r2_score(self.y_test, self.y_pred)

            mse = mean_squared_error(
                self.y_test,
                self.y_pred
            )

            rmse = np.sqrt(mse)

            print("\n--- Model Evaluation ---")
            print("R2 Score:", r2)
            print("Mean Squared Error:", mse)
            print("Root Mean Squared Error:", rmse)

        except Exception as e:
            print("Evaluation Error:", e)

    # Step 8: Sample Prediction
    def sample_prediction(self):
        try:
            sample = pd.DataFrame(
                [[7, 99, 1, 9, 1]],
                columns=self.X.columns
            )

            prediction = self.model.predict(sample)

            print("\n--- Sample Prediction ---")
            print("Input Values:")
            print(sample)

            print(
                "Predicted Performance Index:",
                prediction[0]
            )

        except Exception as e:
            print("Prediction Error:", e)

    # Step 9: Save Model
    def save_model(self):
        try:
            with open(self.model_path, "wb") as file:
                pickle.dump(self.model, file)

            print("\nModel Saved Successfully!")
            print("File:", self.model_path)

        except Exception as e:
            print("Saving Error:", e)

    # Step 10: Load Model
    def load_model(self):
        try:
            with open(self.model_path, "rb") as file:
                loaded_model = pickle.load(file)

            print("\nModel Loaded Successfully!")

            sample = pd.DataFrame(
                [[7, 99, 1, 9, 1]],
                columns=self.X.columns
            )

            original_prediction = self.model.predict(sample)
            loaded_prediction = loaded_model.predict(sample)

            print(
                "Original Prediction:",
                original_prediction[0]
            )

            print(
                "Loaded Prediction:",
                loaded_prediction[0]
            )

            if np.allclose(
                original_prediction,
                loaded_prediction
            ):
                print("Both Predictions Matched Successfully!")

        except Exception as e:
            print("Loading Error:", e)


# Main Program
if __name__ == "__main__":

    project = StudentPerformance()

    project.load_dataset()

    if project.data is not None:

        project.explore_data()

        project.preprocess_data()

        if project.X is not None and project.y is not None:

            project.visualize_data()

            project.split_data()

            if project.X_train is not None:

                project.train_data()

                if project.model is not None:

                    project.evaluate_model()
                    project.sample_prediction()
                    project.save_model()
                    project.load_model()