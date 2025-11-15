#!/usr/bin/env python3
"""
Advanced Feature Engineering for Credit Default Prediction
Goal: Improve AUC from 0.7889 to > 0.80

Author: Data Science Specialist
Date: 2025-11-15
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import KBinsDiscretizer, PolynomialFeatures
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class AdvancedFeatureEngineer:
    """
    Comprehensive feature engineering pipeline for credit default prediction
    """

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.feature_importance = {}
        self.transformation_log = []

    def log(self, message):
        """Log processing steps"""
        if self.verbose:
            print(f"[FE] {message}")
        self.transformation_log.append(message)

    def load_data(self, train_path, test_path):
        """Load train and test datasets"""
        self.log("Loading training and test data...")
        self.X_train = pd.read_parquet(train_path)
        self.X_test = pd.read_parquet(test_path)
        self.log(f"Train shape: {self.X_train.shape}, Test shape: {self.X_test.shape}")

        # Store original feature names
        self.original_features = self.X_train.columns.tolist()

        return self

    def load_target(self, y_train_path):
        """Load target variable"""
        self.y_train = pd.read_parquet(y_train_path).values.ravel()
        self.log(f"Target loaded: {len(self.y_train)} samples, {self.y_train.sum()} defaults ({self.y_train.mean():.2%})")
        return self

    def load_analysis_results(self):
        """Load pre-computed analysis results"""
        self.log("Loading analysis results...")

        # Load key statistics (feature names are in index)
        self.numeric_stats = pd.read_csv('/home/dr/cbu/numeric_features_statistics.csv', index_col=0)
        self.class_separation = pd.read_csv('/home/dr/cbu/class_separation_analysis.csv', index_col=0)
        self.interactions = pd.read_csv('/home/dr/cbu/interaction_candidates.csv')
        self.target_corrs = pd.read_csv('/home/dr/cbu/target_correlations.csv', index_col=0)

        # Reset index to have feature as column
        self.numeric_stats = self.numeric_stats.reset_index().rename(columns={'index': 'feature'})
        self.class_separation = self.class_separation.reset_index().rename(columns={'index': 'feature'})
        self.target_corrs = self.target_corrs.reset_index().rename(columns={'index': 'feature'})

        # Identify feature types
        self.high_skew_features = self.numeric_stats[
            self.numeric_stats['skewness'].abs() > 3
        ]['feature'].tolist()

        self.low_variance_features = self.numeric_stats[
            self.numeric_stats['std'] < 0.01
        ]['feature'].tolist()

        self.sparse_features = self.numeric_stats[
            self.numeric_stats['zeros_pct'] > 50
        ]['feature'].tolist()

        # Top predictors by KS statistic
        self.top_predictors = self.class_separation.nlargest(10, 'ks_statistic')['feature'].tolist()

        self.log(f"Identified {len(self.high_skew_features)} high-skew features")
        self.log(f"Identified {len(self.sparse_features)} sparse features (>50% zeros)")
        self.log(f"Top predictors: {self.top_predictors[:5]}")

        return self

    def create_interaction_features(self):
        """Create interaction features based on analysis"""
        self.log("Creating interaction features...")

        new_features_train = pd.DataFrame()
        new_features_test = pd.DataFrame()

        # Top interaction candidates from analysis
        interaction_pairs = [
            ('debt_service_ratio', 'payment_to_income_ratio'),
            ('debt_service_ratio', 'payment_burden'),
            ('payment_to_income_ratio', 'total_debt_to_income'),
            ('annual_income', 'age'),
            ('credit_score', 'debt_to_income_ratio'),
            ('credit_score', 'annual_income'),
            ('income_vs_regional', 'debt_service_ratio'),
            ('age', 'employment_length'),
            ('num_delinquencies_2yrs', 'credit_score'),
            ('credit_utilization', 'revolving_balance')
        ]

        for feat1, feat2 in interaction_pairs:
            if feat1 in self.X_train.columns and feat2 in self.X_train.columns:
                # Multiplicative interaction
                interaction_name = f"{feat1}_X_{feat2}"
                new_features_train[interaction_name] = self.X_train[feat1] * self.X_train[feat2]
                new_features_test[interaction_name] = self.X_test[feat1] * self.X_test[feat2]

                # Ratio interaction (with safety for division by zero)
                if not (self.X_train[feat2] == 0).any():
                    ratio_name = f"{feat1}_div_{feat2}"
                    new_features_train[ratio_name] = self.X_train[feat1] / (self.X_train[feat2] + 1e-10)
                    new_features_test[ratio_name] = self.X_test[feat1] / (self.X_test[feat2] + 1e-10)

        # Add new features to datasets
        self.X_train = pd.concat([self.X_train, new_features_train], axis=1)
        self.X_test = pd.concat([self.X_test, new_features_test], axis=1)

        self.log(f"Created {len(new_features_train.columns)} interaction features")
        return self

    def create_polynomial_features(self):
        """Create polynomial features for top predictors"""
        self.log("Creating polynomial features for top predictors...")

        # Select top 5 predictors that exist in our data
        top_features = [f for f in self.top_predictors[:5] if f in self.X_train.columns]

        for feature in top_features:
            # Square term
            self.X_train[f"{feature}_squared"] = self.X_train[feature] ** 2
            self.X_test[f"{feature}_squared"] = self.X_test[feature] ** 2

            # Cube term (only for features with reasonable range)
            if self.X_train[feature].abs().max() < 100:
                self.X_train[f"{feature}_cubed"] = self.X_train[feature] ** 3
                self.X_test[f"{feature}_cubed"] = self.X_test[feature] ** 3

            # Square root (for non-negative features)
            if (self.X_train[feature] >= 0).all():
                self.X_train[f"{feature}_sqrt"] = np.sqrt(self.X_train[feature])
                self.X_test[f"{feature}_sqrt"] = np.sqrt(self.X_test[feature])

            # Log transformation (for positive features)
            if (self.X_train[feature] > 0).all():
                self.X_train[f"{feature}_log"] = np.log1p(self.X_train[feature])
                self.X_test[f"{feature}_log"] = np.log1p(self.X_test[feature])

        self.log(f"Created polynomial features for {len(top_features)} top predictors")
        return self

    def create_binned_features(self):
        """Create binned features for high-skew and continuous variables"""
        self.log("Creating binned features...")

        # Features to bin
        features_to_bin = list(set(self.high_skew_features) & set(self.X_train.columns))[:10]

        for feature in features_to_bin:
            if feature in self.X_train.columns:
                # Quantile-based binning
                kbd = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')

                # Fit on train, transform both
                feature_values_train = self.X_train[feature].values.reshape(-1, 1)
                feature_values_test = self.X_test[feature].values.reshape(-1, 1)

                try:
                    binned_train = kbd.fit_transform(feature_values_train)
                    binned_test = kbd.transform(feature_values_test)

                    self.X_train[f"{feature}_binned"] = binned_train
                    self.X_test[f"{feature}_binned"] = binned_test
                except:
                    # Skip if binning fails
                    continue

        self.log(f"Created binned versions of {len(features_to_bin)} features")
        return self

    def create_sparse_indicators(self):
        """Create binary indicators for sparse features"""
        self.log("Creating sparse feature indicators...")

        sparse_to_process = [f for f in self.sparse_features if f in self.X_train.columns][:20]

        for feature in sparse_to_process:
            # Binary indicator: has non-zero value
            self.X_train[f"{feature}_has_value"] = (self.X_train[feature] != 0).astype(int)
            self.X_test[f"{feature}_has_value"] = (self.X_test[feature] != 0).astype(int)

            # For very sparse features, create magnitude indicator
            if self.numeric_stats[self.numeric_stats['feature'] == feature]['zeros_pct'].values[0] > 80:
                # Categorize magnitude when non-zero
                non_zero_train = self.X_train[feature][self.X_train[feature] != 0]
                if len(non_zero_train) > 10:
                    q1, q3 = non_zero_train.quantile([0.25, 0.75])
                    # Only create bins if q1 and q3 are different
                    if q1 != q3 and q1 != 0:
                        try:
                            self.X_train[f"{feature}_magnitude"] = pd.cut(
                                self.X_train[feature],
                                bins=[-np.inf, 0, q1, q3, np.inf],
                                labels=[0, 1, 2, 3],
                                duplicates='drop'
                            ).astype(float)
                            self.X_test[f"{feature}_magnitude"] = pd.cut(
                                self.X_test[feature],
                                bins=[-np.inf, 0, q1, q3, np.inf],
                                labels=[0, 1, 2, 3],
                                duplicates='drop'
                            ).astype(float)
                            sparse_count += 1
                        except:
                            # Skip if binning still fails
                            pass

        self.log(f"Created indicators for {len(sparse_to_process)} sparse features")
        return self

    def create_domain_specific_features(self):
        """Create credit-specific domain features"""
        self.log("Creating domain-specific features...")

        # Financial health scores
        if all(f in self.X_train.columns for f in ['annual_income', 'total_debt_amount', 'credit_score']):
            # Debt burden score
            self.X_train['debt_burden_score'] = (
                self.X_train['total_debt_amount'] / (self.X_train['annual_income'] + 1) *
                (1000 - self.X_train['credit_score']) / 1000
            )
            self.X_test['debt_burden_score'] = (
                self.X_test['total_debt_amount'] / (self.X_test['annual_income'] + 1) *
                (1000 - self.X_test['credit_score']) / 1000
            )

        # Credit utilization categories
        if 'credit_utilization' in self.X_train.columns:
            self.X_train['credit_util_category'] = pd.cut(
                self.X_train['credit_utilization'],
                bins=[0, 0.3, 0.5, 0.7, 1.0],
                labels=[1, 2, 3, 4]
            ).astype(float).fillna(0)
            self.X_test['credit_util_category'] = pd.cut(
                self.X_test['credit_utilization'],
                bins=[0, 0.3, 0.5, 0.7, 1.0],
                labels=[1, 2, 3, 4]
            ).astype(float).fillna(0)

        # Age-income interaction score
        if all(f in self.X_train.columns for f in ['age', 'annual_income']):
            # Expected income based on age (simplified career progression model)
            expected_income_train = 20000 + (self.X_train['age'] - 18) * 2000
            expected_income_test = 20000 + (self.X_test['age'] - 18) * 2000

            self.X_train['income_vs_age_expected'] = (
                self.X_train['annual_income'] / expected_income_train
            )
            self.X_test['income_vs_age_expected'] = (
                self.X_test['annual_income'] / expected_income_test
            )

        # Payment capacity score
        if all(f in self.X_train.columns for f in ['monthly_free_cash_flow', 'monthly_payment']):
            self.X_train['payment_capacity'] = (
                self.X_train['monthly_free_cash_flow'] /
                (self.X_train['monthly_payment'] + 100)  # Add small constant to avoid division by zero
            )
            self.X_test['payment_capacity'] = (
                self.X_test['monthly_free_cash_flow'] /
                (self.X_test['monthly_payment'] + 100)
            )

        # Risk score combining multiple factors
        risk_factors = ['num_delinquencies_2yrs', 'debt_service_ratio', 'payment_to_income_ratio']
        available_risk_factors = [f for f in risk_factors if f in self.X_train.columns]

        if available_risk_factors:
            # Normalize and combine
            risk_scores_train = pd.DataFrame()
            risk_scores_test = pd.DataFrame()

            for factor in available_risk_factors:
                # Normalize to 0-1 range
                min_val = self.X_train[factor].min()
                max_val = self.X_train[factor].max()
                risk_scores_train[factor] = (self.X_train[factor] - min_val) / (max_val - min_val + 1e-10)
                risk_scores_test[factor] = (self.X_test[factor] - min_val) / (max_val - min_val + 1e-10)

            self.X_train['combined_risk_score'] = risk_scores_train.mean(axis=1)
            self.X_test['combined_risk_score'] = risk_scores_test.mean(axis=1)

        self.log("Created domain-specific credit risk features")
        return self

    def create_ratio_features(self):
        """Create additional financial ratio features"""
        self.log("Creating additional ratio features...")

        ratio_pairs = [
            ('revolving_balance', 'available_credit'),
            ('monthly_payment', 'monthly_income'),
            ('existing_monthly_debt', 'monthly_income'),
            ('total_debt_amount', 'annual_income'),
            ('num_customer_service_calls', 'account_tenure_years'),
            ('num_login_sessions', 'account_tenure_years')
        ]

        for numerator, denominator in ratio_pairs:
            if numerator in self.X_train.columns and denominator in self.X_train.columns:
                ratio_name = f"ratio_{numerator}_to_{denominator}"
                # Add small constant to avoid division by zero
                self.X_train[ratio_name] = self.X_train[numerator] / (self.X_train[denominator] + 1e-10)
                self.X_test[ratio_name] = self.X_test[numerator] / (self.X_test[denominator] + 1e-10)

                # Cap extreme values at 99th percentile
                cap_value = self.X_train[ratio_name].quantile(0.99)
                self.X_train[ratio_name] = self.X_train[ratio_name].clip(upper=cap_value)
                self.X_test[ratio_name] = self.X_test[ratio_name].clip(upper=cap_value)

        self.log("Created additional ratio features")
        return self

    def create_aggregation_features(self):
        """Create aggregation features across feature groups"""
        self.log("Creating aggregation features...")

        # Group features by theme
        debt_features = [f for f in self.X_train.columns if 'debt' in f.lower()]
        payment_features = [f for f in self.X_train.columns if 'payment' in f.lower()]
        income_features = [f for f in self.X_train.columns if 'income' in f.lower()]

        # Create aggregations
        if len(debt_features) > 1:
            self.X_train['debt_features_mean'] = self.X_train[debt_features].mean(axis=1)
            self.X_train['debt_features_std'] = self.X_train[debt_features].std(axis=1)
            self.X_train['debt_features_max'] = self.X_train[debt_features].max(axis=1)

            self.X_test['debt_features_mean'] = self.X_test[debt_features].mean(axis=1)
            self.X_test['debt_features_std'] = self.X_test[debt_features].std(axis=1)
            self.X_test['debt_features_max'] = self.X_test[debt_features].max(axis=1)

        if len(payment_features) > 1:
            self.X_train['payment_features_mean'] = self.X_train[payment_features].mean(axis=1)
            self.X_train['payment_features_sum'] = self.X_train[payment_features].sum(axis=1)

            self.X_test['payment_features_mean'] = self.X_test[payment_features].mean(axis=1)
            self.X_test['payment_features_sum'] = self.X_test[payment_features].sum(axis=1)

        self.log("Created aggregation features across feature groups")
        return self

    def remove_low_value_features(self):
        """Remove features with very low predictive value"""
        self.log("Removing low-value features...")

        # Features to remove
        features_to_remove = []

        # 1. Very low variance features
        features_to_remove.extend(self.low_variance_features)

        # 2. Features with near-zero correlation with target (if available)
        if hasattr(self, 'y_train'):
            correlations = {}
            for col in self.X_train.columns:
                if col not in self.original_features:
                    # Only check new features
                    corr = np.abs(np.corrcoef(self.X_train[col], self.y_train)[0, 1])
                    if not np.isnan(corr) and corr < 0.001:
                        features_to_remove.append(col)

        # Remove duplicates from removal list
        features_to_remove = list(set(features_to_remove))

        # Keep features that exist in both datasets
        features_to_remove = [f for f in features_to_remove if f in self.X_train.columns]

        if features_to_remove:
            self.X_train = self.X_train.drop(columns=features_to_remove)
            self.X_test = self.X_test.drop(columns=features_to_remove)
            self.log(f"Removed {len(features_to_remove)} low-value features")

        return self

    def validate_features(self):
        """Validate feature quality and check for issues"""
        self.log("Validating engineered features...")

        # Check for infinite values
        inf_cols_train = self.X_train.columns[np.isinf(self.X_train).any()].tolist()
        inf_cols_test = self.X_test.columns[np.isinf(self.X_test).any()].tolist()

        if inf_cols_train or inf_cols_test:
            self.log(f"WARNING: Found infinite values in {len(set(inf_cols_train + inf_cols_test))} features")
            # Replace infinite values with large numbers
            self.X_train = self.X_train.replace([np.inf, -np.inf], np.nan)
            self.X_test = self.X_test.replace([np.inf, -np.inf], np.nan)

            # Fill NaN with median
            for col in set(inf_cols_train + inf_cols_test):
                if col in self.X_train.columns:
                    median_val = self.X_train[col].median()
                    self.X_train[col] = self.X_train[col].fillna(median_val)
                    self.X_test[col] = self.X_test[col].fillna(median_val)

        # Check for NaN values
        nan_cols_train = self.X_train.columns[self.X_train.isna().any()].tolist()
        nan_cols_test = self.X_test.columns[self.X_test.isna().any()].tolist()

        if nan_cols_train or nan_cols_test:
            self.log(f"WARNING: Found NaN values in {len(set(nan_cols_train + nan_cols_test))} features")
            # Fill with 0 for now (could use more sophisticated imputation)
            self.X_train = self.X_train.fillna(0)
            self.X_test = self.X_test.fillna(0)

        # Final shape
        self.log(f"Final shape - Train: {self.X_train.shape}, Test: {self.X_test.shape}")
        self.log(f"New features created: {self.X_train.shape[1] - len(self.original_features)}")

        return self

    def save_engineered_features(self, train_output_path, test_output_path):
        """Save engineered features to parquet files"""
        self.log(f"Saving engineered features...")

        self.X_train.to_parquet(train_output_path, engine='pyarrow', compression='snappy')
        self.X_test.to_parquet(test_output_path, engine='pyarrow', compression='snappy')

        self.log(f"Saved train to: {train_output_path}")
        self.log(f"Saved test to: {test_output_path}")

        return self

    def generate_feature_report(self):
        """Generate comprehensive report on feature engineering"""
        report = []
        report.append("="*80)
        report.append("FEATURE ENGINEERING REPORT")
        report.append("="*80)
        report.append("")

        report.append("DATASET INFORMATION:")
        report.append(f"- Original features: {len(self.original_features)}")
        report.append(f"- Final features: {self.X_train.shape[1]}")
        report.append(f"- New features created: {self.X_train.shape[1] - len(self.original_features)}")
        report.append(f"- Training samples: {self.X_train.shape[0]}")
        report.append(f"- Test samples: {self.X_test.shape[0]}")
        report.append("")

        report.append("FEATURE ENGINEERING STEPS:")
        for i, step in enumerate(self.transformation_log, 1):
            report.append(f"{i}. {step}")
        report.append("")

        # Feature categories
        report.append("NEW FEATURE CATEGORIES:")
        new_features = [f for f in self.X_train.columns if f not in self.original_features]

        interaction_features = [f for f in new_features if '_X_' in f or '_div_' in f]
        polynomial_features = [f for f in new_features if any(suffix in f for suffix in ['_squared', '_cubed', '_sqrt', '_log'])]
        binned_features = [f for f in new_features if '_binned' in f]
        indicator_features = [f for f in new_features if '_has_value' in f or '_magnitude' in f]
        ratio_features = [f for f in new_features if f.startswith('ratio_')]
        domain_features = [f for f in new_features if any(term in f for term in ['score', 'capacity', 'category'])]
        aggregation_features = [f for f in new_features if any(term in f for term in ['_mean', '_std', '_max', '_sum'])]

        report.append(f"- Interaction features: {len(interaction_features)}")
        report.append(f"- Polynomial features: {len(polynomial_features)}")
        report.append(f"- Binned features: {len(binned_features)}")
        report.append(f"- Sparse indicators: {len(indicator_features)}")
        report.append(f"- Ratio features: {len(ratio_features)}")
        report.append(f"- Domain-specific features: {len(domain_features)}")
        report.append(f"- Aggregation features: {len(aggregation_features)}")
        report.append("")

        # Top new features by correlation with target (if available)
        if hasattr(self, 'y_train'):
            report.append("TOP NEW FEATURES BY TARGET CORRELATION:")
            correlations = {}
            for col in new_features[:50]:  # Check first 50 to avoid long computation
                if col in self.X_train.columns:
                    corr = np.abs(np.corrcoef(self.X_train[col], self.y_train)[0, 1])
                    if not np.isnan(corr):
                        correlations[col] = corr

            top_corrs = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:10]
            for feat, corr in top_corrs:
                report.append(f"- {feat}: {corr:.4f}")
            report.append("")

        report.append("RECOMMENDATIONS:")
        report.append("1. Train XGBoost/LightGBM with the engineered features")
        report.append("2. Use feature importance to identify most valuable new features")
        report.append("3. Consider ensemble methods combining multiple models")
        report.append("4. Fine-tune hyperparameters with focus on:")
        report.append("   - scale_pos_weight for class imbalance")
        report.append("   - max_depth and min_child_weight for overfitting control")
        report.append("   - learning_rate and n_estimators for convergence")
        report.append("5. Validate with stratified k-fold cross-validation")
        report.append("")

        report.append("EXPECTED IMPROVEMENTS:")
        report.append("- Baseline AUC: 0.7889")
        report.append("- Expected AUC with new features: 0.80-0.82")
        report.append("- Key improvements from:")
        report.append("  * Interaction features capturing non-linear relationships")
        report.append("  * Domain-specific risk scores")
        report.append("  * Better handling of sparse features")
        report.append("  * Polynomial transformations of top predictors")

        return "\n".join(report)


def main():
    """Main execution function"""
    print("="*80)
    print("ADVANCED FEATURE ENGINEERING FOR CREDIT DEFAULT PREDICTION")
    print("="*80)
    print()

    # Initialize feature engineer
    fe = AdvancedFeatureEngineer(verbose=True)

    # Execute feature engineering pipeline
    fe.load_data(
        train_path='/home/dr/cbu/X_train.parquet',
        test_path='/home/dr/cbu/X_test.parquet'
    )

    fe.load_target('/home/dr/cbu/y_train.parquet')
    fe.load_analysis_results()

    # Apply all feature engineering techniques
    fe.create_interaction_features()
    fe.create_polynomial_features()
    fe.create_binned_features()
    fe.create_sparse_indicators()
    fe.create_domain_specific_features()
    fe.create_ratio_features()
    fe.create_aggregation_features()
    fe.remove_low_value_features()
    fe.validate_features()

    # Save engineered features
    fe.save_engineered_features(
        train_output_path='/home/dr/cbu/X_train_engineered.parquet',
        test_output_path='/home/dr/cbu/X_test_engineered.parquet'
    )

    # Generate and save report
    report = fe.generate_feature_report()
    print("\n" + report)

    with open('/home/dr/cbu/feature_engineering_report.txt', 'w') as f:
        f.write(report)

    print("\n" + "="*80)
    print("FEATURE ENGINEERING COMPLETED SUCCESSFULLY")
    print("="*80)


if __name__ == "__main__":
    main()