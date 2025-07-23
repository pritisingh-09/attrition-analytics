
"""
Employee Attrition Prediction with Action Simulation
====================================================
Author: Data Analyst Portfolio Project
Description: Predicts employee attrition and simulates the impact of retention interventions

This project goes beyond traditional churn prediction by:
1. Predicting which employees are likely to leave
2. Simulating the impact of different retention actions
3. Calculating ROI of retention strategies
4. Providing interactive dashboard for stakeholder decision-making
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class EmployeeAttritionPredictor:
    """
    A comprehensive employee attrition prediction system with action simulation capabilities
    """

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_importance = None
        self.attrition_cost_per_employee = 50000  # Average cost of replacing an employee

    def load_and_preprocess_data(self, filepath=None):
        """
        Load and preprocess the IBM HR Analytics dataset
        """
        if filepath:
            self.df = pd.read_csv(filepath)
        else:
            # For demo purposes, we'll create sample data structure
            # In real implementation, download from: https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset
            print("Please download the IBM HR Analytics dataset from Kaggle")
            print("URL: https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset")
            return None

        # Basic preprocessing
        self.df['Attrition_Binary'] = self.df['Attrition'].map({'Yes': 1, 'No': 0})

        # Identify categorical and numerical columns
        self.categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        self.categorical_cols.remove('Attrition')  # Remove target variable

        self.numerical_cols = self.df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.numerical_cols.remove('Attrition_Binary')  # Remove target variable

        # Encode categorical variables
        for col in self.categorical_cols:
            le = LabelEncoder()
            self.df[col + '_Encoded'] = le.fit_transform(self.df[col])
            self.label_encoders[col] = le

        print(f"Dataset loaded successfully. Shape: {self.df.shape}")
        print(f"Attrition rate: {self.df['Attrition_Binary'].mean():.2%}")

        return self.df

    def train_model(self):
        """
        Train the attrition prediction model using ensemble methods
        """
        # Prepare features
        feature_cols = [col for col in self.df.columns 
                       if col.endswith('_Encoded') or col in self.numerical_cols]

        X = self.df[feature_cols]
        y = self.df['Attrition_Binary']

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale numerical features
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()

        numerical_feature_cols = [col for col in feature_cols if col in self.numerical_cols]
        X_train_scaled[numerical_feature_cols] = self.scaler.fit_transform(X_train[numerical_feature_cols])
        X_test_scaled[numerical_feature_cols] = self.scaler.transform(X_test[numerical_feature_cols])

        # Train ensemble model
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        gb_model = GradientBoostingClassifier(random_state=42)

        # Use GridSearch for hyperparameter tuning
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5]
        }

        self.model = GridSearchCV(rf_model, param_grid, cv=5, scoring='roc_auc')
        self.model.fit(X_train_scaled, y_train)

        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]

        print("Model Performance:")
        print(f"AUC-ROC Score: {roc_auc_score(y_test, y_pred_proba):.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.model.best_estimator_.feature_importances_
        }).sort_values('importance', ascending=False)

        # Store for simulation
        self.X_train = X_train_scaled
        self.X_test = X_test_scaled
        self.y_train = y_train
        self.y_test = y_test
        self.feature_cols = feature_cols

        return self.model

    def simulate_retention_actions(self, action_scenarios):
        """
        Simulate the impact of different retention actions on attrition probability

        Parameters:
        action_scenarios (dict): Dictionary defining different intervention scenarios
        """
        if self.model is None:
            print("Please train the model first!")
            return

        simulation_results = {}

        # Get current predictions for all employees
        all_features = self.df[self.feature_cols].copy()

        # Scale numerical features
        numerical_feature_cols = [col for col in self.feature_cols if col in self.numerical_cols]
        all_features[numerical_feature_cols] = self.scaler.transform(all_features[numerical_feature_cols])

        baseline_probabilities = self.model.predict_proba(all_features)[:, 1]
        high_risk_employees = self.df[baseline_probabilities > 0.5].copy()
        high_risk_features = all_features[baseline_probabilities > 0.5].copy()

        print(f"Identified {len(high_risk_employees)} high-risk employees")

        for scenario_name, actions in action_scenarios.items():
            modified_features = high_risk_features.copy()

            # Apply actions (this is simplified - in reality you'd map actions to feature changes)
            if 'salary_increase' in actions:
                # Increase monthly income by specified percentage
                if 'MonthlyIncome' in modified_features.columns:
                    modified_features['MonthlyIncome'] *= (1 + actions['salary_increase'])

            if 'work_life_balance_improvement' in actions:
                # Improve work-life balance score
                if 'WorkLifeBalance' in modified_features.columns:
                    modified_features['WorkLifeBalance'] = np.minimum(
                        modified_features['WorkLifeBalance'] + 1, 4
                    )

            if 'job_satisfaction_program' in actions:
                # Improve job satisfaction
                if 'JobSatisfaction' in modified_features.columns:
                    modified_features['JobSatisfaction'] = np.minimum(
                        modified_features['JobSatisfaction'] + 1, 4
                    )

            # Get new predictions
            new_probabilities = self.model.predict_proba(modified_features)[:, 1]

            # Calculate impact
            probability_reduction = baseline_probabilities[baseline_probabilities > 0.5] - new_probabilities
            employees_retained = (probability_reduction > 0.2).sum()  # Significant reduction threshold

            simulation_results[scenario_name] = {
                'employees_at_risk': len(high_risk_employees),
                'employees_retained': employees_retained,
                'retention_rate_improvement': employees_retained / len(high_risk_employees),
                'cost_savings': employees_retained * self.attrition_cost_per_employee,
                'average_probability_reduction': probability_reduction.mean(),
                'action_cost': self._calculate_action_cost(actions, len(high_risk_employees))
            }

        return simulation_results

    def _calculate_action_cost(self, actions, num_employees):
        """
        Calculate the cost of implementing retention actions
        """
        total_cost = 0

        if 'salary_increase' in actions:
            # Assume average salary of $60,000 and calculate yearly increase cost
            avg_salary = 60000
            total_cost += num_employees * avg_salary * actions['salary_increase']

        if 'work_life_balance_improvement' in actions:
            # Cost of implementing work-life balance programs
            total_cost += 2000 * num_employees  # $2000 per employee for programs

        if 'job_satisfaction_program' in actions:
            # Cost of job satisfaction improvement programs
            total_cost += 1500 * num_employees  # $1500 per employee

        return total_cost

    def calculate_roi(self, simulation_results):
        """
        Calculate ROI for each retention scenario
        """
        roi_results = {}

        for scenario, results in simulation_results.items():
            cost_savings = results['cost_savings']
            action_cost = results['action_cost']

            if action_cost > 0:
                roi = ((cost_savings - action_cost) / action_cost) * 100
            else:
                roi = float('inf') if cost_savings > 0 else 0

            roi_results[scenario] = {
                'roi_percentage': roi,
                'net_benefit': cost_savings - action_cost,
                'cost_benefit_ratio': cost_savings / action_cost if action_cost > 0 else float('inf')
            }

        return roi_results

    def create_dashboard_data(self):
        """
        Prepare data for the interactive dashboard
        """
        if self.model is None:
            print("Please train the model first!")
            return

        # Get predictions for all employees
        all_features = self.df[self.feature_cols].copy()
        numerical_feature_cols = [col for col in self.feature_cols if col in self.numerical_cols]
        all_features[numerical_feature_cols] = self.scaler.transform(all_features[numerical_feature_cols])

        probabilities = self.model.predict_proba(all_features)[:, 1]

        # Add predictions to dataframe
        dashboard_df = self.df.copy()
        dashboard_df['AttritionProbability'] = probabilities
        dashboard_df['RiskCategory'] = pd.cut(
            probabilities, 
            bins=[0, 0.3, 0.7, 1.0], 
            labels=['Low Risk', 'Medium Risk', 'High Risk']
        )

        return dashboard_df

# Example usage and simulation scenarios
def main():
    """
    Main function demonstrating the Employee Attrition Prediction system
    """
    # Initialize predictor
    predictor = EmployeeAttritionPredictor()

    # Load and preprocess data (you need to download the dataset first)
    # df = predictor.load_and_preprocess_data('IBM_HR_Analytics_Employee_Attrition.csv')

    # For demo, we'll show the structure
    print("Employee Attrition Prediction with Action Simulation")
    print("=" * 60)
    print()
    print("This project includes:")
    print("1. Machine Learning model for attrition prediction")
    print("2. Action simulation engine")
    print("3. ROI calculations for retention strategies")
    print("4. Interactive dashboard for stakeholder use")
    print()

    # Define simulation scenarios
    action_scenarios = {
        'Salary_Increase_10%': {
            'salary_increase': 0.10
        },
        'Work_Life_Balance_Program': {
            'work_life_balance_improvement': True
        },
        'Job_Satisfaction_Initiative': {
            'job_satisfaction_program': True
        },
        'Comprehensive_Package': {
            'salary_increase': 0.05,
            'work_life_balance_improvement': True,
            'job_satisfaction_program': True
        }
    }

    print("Sample Action Scenarios:")
    for scenario, actions in action_scenarios.items():
        print(f"- {scenario}: {actions}")

    print("\nTo run this project:")
    print("1. Download the IBM HR Analytics dataset from Kaggle")
    print("2. Run the training pipeline")
    print("3. Execute action simulations")
    print("4. Generate ROI analysis")
    print("5. Deploy interactive dashboard")

if __name__ == "__main__":
    main()
