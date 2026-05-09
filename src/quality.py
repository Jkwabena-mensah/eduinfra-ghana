from src.logger import get_logger

logger = get_logger(__name__)


class QualityController:
    """Inspects priority scores for geographic and statistical plausibility."""

    # Priority regions per Ghana's Poverty Maps (GSS/MPI)
    NORTHERN_BELT = ['NORTHERN', 'SAVANNAH', 'NORTH EAST', 'UPPER WEST', 'UPPER EAST']

    def run_all_checks(self, df):
        logger.info("🧪 Starting AI Quality Control Lab...")

        # Check 1: Geographic Plausibility
        # We expect at least 70% of the top 10 schools to be in the northern belt.
        top_10 = df.nlargest(10, 'priority_score')
        northern_count = top_10['region'].str.upper().isin(self.NORTHERN_BELT).sum()

        passed_geo = northern_count >= 7
        if passed_geo:
            logger.info(f"✅ Geographic Check: PASSED ({northern_count}/10 in Northern Belt).")
        else:
            logger.warning(f"⚠️ Geographic Check: FAILED ({northern_count}/10). Investigate weights.")

        # Check 2: Statistical Variance
        spread = df['priority_score'].max() - df['priority_score'].min()
        passed_spread = spread > 0.4
        if passed_spread:
            logger.info(f"✅ Data Variance: PASSED ({spread:.2f} spread).")
        else:
            logger.warning("⚠️ Data Variance: FAILED. Scores are too clustered.")

        return passed_geo and passed_spread