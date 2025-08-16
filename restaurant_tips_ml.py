
"""
Restaurant Tips Prediction - Complete ML Pipeline
Algorithms: LightGBM, XGBoost, AdaBoost, CatBoost (Regression)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import warnings
warnings.filterwarnings('ignore')

def load_and_clean_data():
    """Load and clean Restaurant Tips dataset"""
    print("💰 Loading Restaurant Tips Dataset...")

    # Create realistic tips data
    np.random.seed(42)
    n_samples = 244

    # Generate realistic restaurant data
    total_bills = np.random.exponential(20, n_samples) + 5

    # Tips generally correlate with bill amount
    tip_base = total_bills * 0.15  # Base 15% tip
    tip_noise = np.random.normal(0, 2, n_samples)  # Random variation
    tips = tip_base + tip_noise
    tips = np.maximum(tips, 1)  # Minimum $1 tip

    data = {
        'total_bill': total_bills,
        'tip': tips,
        'sex': np.random.choice(['Male', 'Female'], n_samples, p=[0.6, 0.4]),
        'smoker': np.random.choice(['No', 'Yes'], n_samples, p=[0.75, 0.25]),
        'day': np.random.choice(['Thur', 'Fri', 'Sat', 'Sun'], n_samples, p=[0.25, 0.25, 0.25, 0.25]),
        'time': np.random.choice(['Lunch', 'Dinner'], n_samples, p=[0.35, 0.65]),
        'size': np.random.choice([1, 2, 3, 4, 5, 6], n_samples, p=[0.05, 0.50, 0.20, 0.15, 0.08, 0.02])
    }

    df = pd.DataFrame(data)

    print(f"📊 Original shape: {df.shape}")
    print(f"📊 Average tip: ${df['tip'].mean():.2f}")
    print(f"📊 Average bill: ${df['total_bill'].mean():.2f}")

    # Data Cleaning Process
    print("\n🧹 Starting Data Cleaning...")

    # 1. Check for missing values and outliers
    print("Missing values:", df.isnull().sum().sum())

    # Remove extreme outliers in tips and bills
    # Remove tips that are too high relative to bill (>50% tip rate)
    df = df[df['tip'] / df['total_bill'] <= 0.5]

    # Remove extremely high bills (top 1%)
    bill_99th = df['total_bill'].quantile(0.99)
    df = df[df['total_bill'] <= bill_99th]

    # 2. Feature Engineering
    print("\n🔧 Feature Engineering...")

    # Create tip percentage
    df['tip_percentage'] = df['tip'] / df['total_bill']

    # Create bill per person
    df['bill_per_person'] = df['total_bill'] / df['size']

    # Create weekend indicator
    df['is_weekend'] = df['day'].isin(['Sat', 'Sun']).astype(int)

    # Create large party indicator (4+ people)
    df['large_party'] = (df['size'] >= 4).astype(int)

    # 3. Encode categorical variables
    print("\n🔤 Encoding Categorical Variables...")

    label_encoders = {}
    categorical_cols = ['sex', 'smoker', 'day', 'time']

    for col in categorical_cols:
        le = LabelEncoder()
        df[f'{col}_encoded'] = le.fit_transform(df[col])
        label_encoders[col] = le

    # 4. Select features for modeling
    feature_cols = ['total_bill', 'sex_encoded', 'smoker_encoded', 'day_encoded', 
                   'time_encoded', 'size', 'tip_percentage', 'bill_per_person', 
                   'is_weekend', 'large_party']

    X = df[feature_cols]
    y = df['tip']

    print(f"✅ Final dataset shape: {X.shape}")
    print(f"✅ Features: {list(X.columns)}")
    print(f"✅ Target statistics:")
    print(f"   Mean tip: ${y.mean():.2f}")
    print(f"   Std tip:  ${y.std():.2f}")
    print(f"   Min tip:  ${y.min():.2f}")
    print(f"   Max tip:  ${y.max():.2f}")

    return X, y, label_encoders

def train_models(X, y):
    """Train all boosting regression models"""
    print("\n🚀 Training Regression Models...")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results = {}

    # 1. LightGBM Regressor
    print("\n📊 Training LightGBM Regressor...")
    lgb_model = lgb.LGBMRegressor(
        objective='regression',
        random_state=42,
        verbose=-1,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6
    )
    lgb_model.fit(X_train, y_train)
    lgb_pred = lgb_model.predict(X_test)
    lgb_rmse = np.sqrt(mean_squared_error(y_test, lgb_pred))
    lgb_r2 = r2_score(y_test, lgb_pred)
    results['LightGBM'] = {'model': lgb_model, 'predictions': lgb_pred, 'rmse': lgb_rmse, 'r2': lgb_r2}

    # 2. XGBoost Regressor
    print("📊 Training XGBoost Regressor...")
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6
    )
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
    xgb_r2 = r2_score(y_test, xgb_pred)
    results['XGBoost'] = {'model': xgb_model, 'predictions': xgb_pred, 'rmse': xgb_rmse, 'r2': xgb_r2}

    # 3. AdaBoost Regressor
    print("📊 Training AdaBoost Regressor...")
    ada_model = AdaBoostRegressor(
        random_state=42,
        n_estimators=100,
        learning_rate=1.0
    )
    ada_model.fit(X_train, y_train)
    ada_pred = ada_model.predict(X_test)
    ada_rmse = np.sqrt(mean_squared_error(y_test, ada_pred))
    ada_r2 = r2_score(y_test, ada_pred)
    results['AdaBoost'] = {'model': ada_model, 'predictions': ada_pred, 'rmse': ada_rmse, 'r2': ada_r2}

    # 4. CatBoost Regressor
    print("📊 Training CatBoost Regressor...")
    cat_model = cb.CatBoostRegressor(
        iterations=100,
        random_seed=42,
        verbose=False,
        learning_rate=0.1,
        depth=6
    )
    cat_model.fit(X_train, y_train)
    cat_pred = cat_model.predict(X_test)
    cat_rmse = np.sqrt(mean_squared_error(y_test, cat_pred))
    cat_r2 = r2_score(y_test, cat_pred)
    results['CatBoost'] = {'model': cat_model, 'predictions': cat_pred, 'rmse': cat_rmse, 'r2': cat_r2}

    return results, X_test, y_test

def feature_importance_analysis(results, feature_names):
    """Analyze feature importance across models"""
    print("\n📊 Feature Importance Analysis:")
    print("-" * 50)

    for name, result in results.items():
        model = result['model']
        print(f"\n{name} Top 5 Important Features:")

        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_imp = list(zip(feature_names, importances))
            feature_imp.sort(key=lambda x: x[1], reverse=True)

            for feat, imp in feature_imp[:5]:
                print(f"  {feat}: {imp:.4f}")

def evaluate_models(results, X_test, y_test):
    """Evaluate and compare all regression models"""
    print("\n📈 Model Evaluation Results (Regression Metrics):")
    print("=" * 60)

    for name, result in results.items():
        rmse = result['rmse']
        r2 = result['r2']
        predictions = result['predictions']

        mae = mean_absolute_error(y_test, predictions)

        print(f"{name}:")
        print(f"  RMSE:     ${rmse:.3f}")
        print(f"  MAE:      ${mae:.3f}")
        print(f"  R²:       {r2:.4f}")
        print(f"  R² (%):   {r2*100:.2f}%")
        print()

    # Find best model (lowest RMSE)
    best_model_name = min(results.keys(), key=lambda k: results[k]['rmse'])
    best_rmse = results[best_model_name]['rmse']
    best_r2 = results[best_model_name]['r2']

    print(f"🏆 Best Model: {best_model_name}")
    print(f"   RMSE: ${best_rmse:.3f}")
    print(f"   R²: {best_r2:.4f} ({best_r2*100:.2f}%)")

    # Show prediction vs actual comparison for best model
    best_pred = results[best_model_name]['predictions']
    print(f"\n📊 Sample Predictions vs Actual ({best_model_name}):")
    print("Predicted | Actual  | Difference")
    print("-" * 35)
    for i in range(min(10, len(y_test))):
        pred_val = best_pred.iloc[i] if hasattr(best_pred, 'iloc') else best_pred[i]
        actual_val = y_test.iloc[i] if hasattr(y_test, 'iloc') else y_test[i]
        diff = abs(pred_val - actual_val)
        print(f"  ${pred_val:6.2f}  | ${actual_val:6.2f} | ${diff:6.2f}")

def main():
    """Main execution function"""
    print("💰 RESTAURANT TIPS PREDICTION PIPELINE")
    print("=" * 60)

    # Load and clean data
    X, y, label_encoders = load_and_clean_data()

    # Train models
    results, X_test, y_test = train_models(X, y)

    # Evaluate models
    evaluate_models(results, X_test, y_test)

    # Feature importance analysis
    feature_importance_analysis(results, X.columns.tolist())

    print("\n✅ Pipeline completed successfully!")
    print("\n💡 Key Insights:")
    print("- Total bill amount is usually the strongest predictor of tip amount")
    print("- Party size and dining time also influence tipping behavior")
    print("- Individual customer characteristics (sex, smoking) have smaller effects")
    print("- Weekend dining patterns may differ from weekday patterns")

if __name__ == "__main__":
    main()
