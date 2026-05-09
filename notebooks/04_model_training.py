import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src.logger import get_logger

logger = get_logger("ML_Engine")


def train_infrastructure_model():
    logger.info("🧠 Initializing Random Forest Model Training...")

    data_path = ROOT / "data" / "schools_priority_ranked.csv"
    if not data_path.exists():
        logger.error("Ranked data not found.")
        return

    df = pd.read_csv(data_path)
    df.columns = df.columns.str.lower()

    # --- CRITICAL FIX: DROP ROWS WITH MISSING TARGETS ---
    original_count = len(df)
    df = df.dropna(subset=['priority_score'])
    new_count = len(df)

    if original_count > new_count:
        logger.warning(f"🧹 Cleaned {original_count - new_count} rows with missing priority scores.")
    # ----------------------------------------------------

    features = ['pov_norm', 'lit_norm']

    # Also ensure features themselves don't have NaNs
    X = df[features].fillna(0)
    y = df['priority_score']

    logger.info(f"📈 Training on {len(df)} valid school samples...")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    logger.info(f"✅ Model Validation: MAE: {mae:.4f} | R-squared: {r2:.4f}")

    # Feature Importance Chart
    importance = pd.Series(model.feature_importances_, index=['Poverty Intensity', 'Youth Literacy']).sort_values()
    plt.figure(figsize=(10, 6))
    importance.plot(kind='barh', color=['#C8960C', '#006B3F'])
    plt.title('Which Factors Drive School Infrastructure Deprivation Most?', fontweight='bold')

    chart_path = ROOT / "outputs" / "feature_importance.png"
    plt.tight_layout()
    plt.savefig(chart_path, dpi=150)

    # Save Model
    model_dir = ROOT / "models"
    model_dir.mkdir(exist_ok=True)
    joblib.dump(model, model_dir / "eduinfra_v1.pkl")
    logger.info("💾 Model and Insights saved successfully.")


if __name__ == "__main__":
    train_infrastructure_model()