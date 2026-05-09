import pandas as pd
import os
import re


def run_foundational_pipeline():
    print("🚀 Finalizing Master Data Pipeline (100% Coverage Mode)...")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")

    schools_path = os.path.join(data_dir, "ghana_schools_final_2025_COMPLETE.csv")
    ref_path = os.path.join(data_dir, "schools_clean.csv")

    schools_df = pd.read_csv(schools_path)
    ref_df = pd.read_csv(ref_path)

    # 1. Standardize helper
    def universal_clean(text):
        if pd.isna(text): return ""
        text = str(text).upper()
        noise = ['SENIOR HIGH', 'TECHNICAL', 'SHTS', 'SHS', 'COMMUNITY', 'SCHOOL', 'VOCATIONAL', 'MUNICIPAL',
                 'DISTRICT', 'ASSEMBLY']
        for word in noise:
            text = text.replace(word, "")
        return re.sub(r'[^A-Z0-9]', '', text).strip()

    # 2. Build Keys
    schools_df['s_key'] = schools_df['School_Name'].apply(universal_clean)
    ref_df['s_key'] = ref_df['school_name'].apply(universal_clean)
    schools_df['d_key'] = schools_df['District'].apply(universal_clean)
    ref_df['d_key'] = ref_df['District'].apply(universal_clean)

    schools_df['final_key'] = schools_df['s_key'] + schools_df['d_key']
    ref_df['final_key'] = ref_df['s_key'] + ref_df['d_key']

    # 3. Perform Primary Merge
    ref_subset = ref_df[['final_key', 'youth_literacy_count', 'mpi_score']].drop_duplicates('final_key')
    master_df = schools_df.merge(ref_subset, on='final_key', how='left')

    # 4. Fallback 1: School Name Alone
    missing_mask = master_df['youth_literacy_count'].isna()
    ref_name_only = ref_df[['s_key', 'youth_literacy_count', 'mpi_score']].drop_duplicates('s_key')
    fallback_data = master_df[missing_mask][['s_key']].merge(ref_name_only, on='s_key', how='left')
    master_df.loc[missing_mask, 'youth_literacy_count'] = fallback_data['youth_literacy_count'].values
    master_df.loc[missing_mask, 'mpi_score'] = fallback_data['mpi_score'].values

    # 5. Fallback 2: Regional Imputation (The Final 100% Gap Filler)
    # If we still have NaNs, use the average for that Region from the reference file
    print("Filling remaining gaps with Regional Averages...")
    regional_stats = ref_df.groupby('Region')[['youth_literacy_count', 'mpi_score']].mean().reset_index()
    regional_stats['Region'] = regional_stats['Region'].str.upper().str.strip()

    # Merge on Region
    master_df['Region_Upper'] = master_df['Region'].str.upper().str.strip()
    master_df = master_df.merge(regional_stats, left_on='Region_Upper', right_on='Region', how='left',
                                suffixes=('', '_reg'))

    # Fill only the missing ones
    master_df['youth_literacy_count'] = master_df['youth_literacy_count'].fillna(master_df['youth_literacy_count_reg'])
    master_df['mpi_score'] = master_df['mpi_score'].fillna(master_df['mpi_score_reg'])

    # 6. Final Clean up
    cols_to_drop = ['s_key', 'd_key', 'final_key', 'Region_Upper', 'Region_reg', 'youth_literacy_count_reg',
                    'mpi_score_reg']
    master_df = master_df.drop(columns=[c for c in cols_to_drop if c in master_df.columns])

    # 7. Export
    output_path = os.path.join(data_dir, "master_infrastructure_dataset.csv")
    master_df.to_csv(output_path, index=False)

    print(f"\n✅ Pipeline Finalized!")
    print(f"   Final Dataset: {len(master_df)} schools")
    print(f"   Literacy Coverage: {master_df['youth_literacy_count'].notna().sum()}/{len(master_df)}")
    print(f"   Poverty Coverage: {master_df['mpi_score'].notna().sum()}/{len(master_df)}")
    print(f"💾 File ready for AI analysis: {output_path}")


if __name__ == "__main__":
    run_foundational_pipeline()