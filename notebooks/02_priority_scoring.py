import pandas as pd
import os
import sys

# Ensure Python can find the 'src' folder for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.logger import get_logger
from src.quality import QualityController

logger = get_logger("ScoringPipeline")


def run_scored_pipeline():
    logger.info("🚀 Initializing Priority Scoring...")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "master_infrastructure_dataset.csv")

    if not os.path.exists(data_path):
        logger.error(f"Missing input data: {data_path}")
        return

    df = pd.read_csv(data_path)

    # 1. Normalize metrics (Scaling to 0-1)
    # Higher MPI (Poverty) = Higher Priority
    df['pov_norm'] = (df['mpi_score'] - df['mpi_score'].min()) / (df['mpi_score'].max() - df['mpi_score'].min())

    # Lower Literacy = Higher Priority (Inverting the scale)
    df['lit_norm'] = 1 - ((df['youth_literacy_count'] - df['youth_literacy_count'].min()) /
                          (df['youth_literacy_count'].max() - df['youth_literacy_count'].min()))

    # 2. AI Weighted Score Calculation
    # 60% Weight on Poverty / 40% Weight on Literacy
    df['priority_score'] = (df['pov_norm'] * 0.6) + (df['lit_norm'] * 0.4)
    df = df.sort_values(by='priority_score', ascending=False)

    # 3. Export Ranked Dataset
    output_path = os.path.join(base_dir, "data", "schools_priority_ranked.csv")
    df.to_csv(output_path, index=False)
    logger.info(f"💾 Rankings saved to {output_path}")

    # 4. Trigger Quality Control Validation
    qc = QualityController()
    success = qc.run_all_checks(df)

    if success:
        logger.info("🎯 PIPELINE SUCCESS: Data is ready for the Innovation Challenge.")
    else:
        logger.warning("🚨 PIPELINE COMPLETED WITH QUALITY WARNINGS.")


if __name__ == "__main__":
    run_scored_pipeline()