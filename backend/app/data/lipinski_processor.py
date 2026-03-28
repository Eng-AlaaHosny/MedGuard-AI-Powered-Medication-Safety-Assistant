import os
import pandas as pd
import numpy as np
from typing import Dict, Optional

LIPINSKI_FEATURES = ['molecular_weight', 'n_hba', 'n_hbd', 'logp', 'ro5_fulfilled']


class LipinskiProcessor:
    """
    Processes DrugBank Lipinski compounds CSV.
    Blueprint Section 4.3:
    - 2,647 drug compounds with physicochemical data
    - Features: molecular_weight, n_hba, n_hbd, logp, ro5_fulfilled
    - Enriches drug node embeddings with molecular-level features
    - Missing drugs get column mean imputation (training only)
    - Unknown experimental drugs trigger Data Unavailable warning
    """

    def __init__(self):
        self.df = None
        self.column_means = {}
        self.drug_id_to_features = {}

    def load(self, csv_path: str):
        """Load and process the Lipinski CSV file."""
        print(f"Loading Lipinski data from {csv_path}...")

        self.df = pd.read_csv(csv_path)
        print(f"Loaded {len(self.df)} compounds")
        print(f"Columns: {list(self.df.columns)}")

        
        id_col = self._find_id_column()
        print(f"Using ID column: {id_col}")

        for feature in LIPINSKI_FEATURES:
            if feature in self.df.columns:
                self.column_means[feature] = self.df[feature].mean()
            else:
                self.column_means[feature] = 0.0

        print(f"Column means computed: {self.column_means}")

       
        for _, row in self.df.iterrows():
            drug_id = str(row[id_col])
            features = self._extract_features(row)
            self.drug_id_to_features[drug_id] = features

        print(f"Indexed {len(self.drug_id_to_features)} drugs")

    def _find_id_column(self) -> str:
        """Find the DrugBank ID column in the CSV."""
        possible_names = [
            'drugbank_id', 'DrugBank ID', 'drugbank-id',
            'id', 'ID', 'drug_id', 'DrugBankID'
        ]
        for name in possible_names:
            if name in self.df.columns:
                return name
     
        return self.df.columns[0]

    def _extract_features(self, row) -> np.ndarray:
        """Extract Lipinski features from a row."""
        features = []
        for feature in LIPINSKI_FEATURES:
            if feature in self.df.columns:
                val = row[feature]
                if pd.isna(val):
                    val = self.column_means.get(feature, 0.0)
                
                if feature == 'ro5_fulfilled':
                    val = float(bool(val))
                features.append(float(val))
            else:
                features.append(0.0)
        return np.array(features, dtype=np.float32)

    def get_features(self, drug_id: str) -> Optional[np.ndarray]:
        """
        Get Lipinski features for a drug by DrugBank ID.
        Returns None if drug not found (triggers Data Unavailable).
        Blueprint Section 4.3
        """
        return self.drug_id_to_features.get(drug_id)

    def get_features_or_mean(self, drug_id: str) -> np.ndarray:
        """
        Get features or column means for missing training drugs.
        Blueprint Section 4.3: standard imputation for training samples.
        Only use for training — not for unknown user queries.
        """
        features = self.get_features(drug_id)
        if features is not None:
            return features
        
        return np.array(
            [self.column_means.get(f, 0.0) for f in LIPINSKI_FEATURES],
            dtype=np.float32
        )

    def is_drug_available(self, drug_id: str) -> bool:
        """Check if drug exists in Lipinski dataset."""
        return drug_id in self.drug_id_to_features

    def get_feature_dim(self) -> int:
        """Return number of Lipinski features."""
        return len(LIPINSKI_FEATURES)

    def normalize_features(self):
        """
        Normalize features to zero mean unit variance.
        Should be called after loading all data.
        """
        if self.df is None:
            return

        for feature in LIPINSKI_FEATURES:
            if feature in self.df.columns and feature != 'ro5_fulfilled':
                mean = self.df[feature].mean()
                std = self.df[feature].std()
                if std > 0:
                    for drug_id in self.drug_id_to_features:
                        self.drug_id_to_features[drug_id] = \
                            self.drug_id_to_features[drug_id].copy()
                        idx = LIPINSKI_FEATURES.index(feature)
                        self.drug_id_to_features[drug_id][idx] = \
                            (self.drug_id_to_features[drug_id][idx] - mean) / std

        print("Features normalized")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, 'DB_compounds_lipinski.csv')

    if not os.path.exists(csv_path):
        print(f"File not found at {csv_path}")
    else:
        processor = LipinskiProcessor()
        processor.load(csv_path)
        processor.normalize_features()

        print(f"\nFeature dimension: {processor.get_feature_dim()}")
        print(f"Total drugs indexed: {len(processor.drug_id_to_features)}")

        first_id = list(processor.drug_id_to_features.keys())[0]
        features = processor.get_features(first_id)
        print(f"\nSample drug ID: {first_id}")
        print(f"Sample features: {features}")
        print(f"\nLipinski processor working correctly!")
