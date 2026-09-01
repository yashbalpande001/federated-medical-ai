import sys
from pathlib import Path
import pytest
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.eda_rsna_splits import (
    generate_synthetic_rsna_metadata,
    perform_class_distribution_analysis,
    inspect_dicom_quality,
    split_patients_grouped,
    check_set_intersection,
    run_eda_and_splits,
)
from src.utils.env_config import get_paths


def test_generate_synthetic_rsna_metadata():
    df = generate_synthetic_rsna_metadata(num_patients=100, seed=42)
    assert len(df) == 100
    assert "patientId" in df.columns
    assert "class" in df.columns
    assert "Target" in df.columns


def test_class_distribution_analysis():
    df = generate_synthetic_rsna_metadata(num_patients=100, seed=42)
    dist_df = perform_class_distribution_analysis(df)
    assert "Class" in dist_df.columns
    assert "Count" in dist_df.columns
    assert "Percentage (%)" in dist_df.columns
    assert dist_df["Count"].sum() == 100


def test_set_intersection_check():
    df = generate_synthetic_rsna_metadata(num_patients=100, seed=42)
    tr, val, te = split_patients_grouped(df, subset_size=100, seed=42)
    # Check passes
    assert check_set_intersection(tr, val, te) is True

    # Test error raising when leakage is intentionally introduced
    leaked_val = pd.concat([val, tr.iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError, match="DATA LEAKAGE DETECTED"):
        check_set_intersection(tr, leaked_val, te)


def test_expected_output_files():
    tr_df, val_df, test_df, eda_summary_path = run_eda_and_splits()

    paths = get_paths()
    splits_dir = paths.output_root / "splits"

    assert (splits_dir / "train.csv").exists()
    assert (splits_dir / "val.csv").exists()
    assert (splits_dir / "test.csv").exists()
    assert eda_summary_path.exists()

    # Load and verify CSV columns
    train_csv = pd.read_csv(splits_dir / "train.csv")
    val_csv = pd.read_csv(splits_dir / "val.csv")
    test_csv = pd.read_csv(splits_dir / "test.csv")

    assert "patientId" in train_csv.columns
    assert "patientId" in val_csv.columns
    assert "patientId" in test_csv.columns

    # Verify empty set intersection
    s_tr = set(train_csv["patientId"])
    s_val = set(val_csv["patientId"])
    s_te = set(test_csv["patient_id"] if "patient_id" in test_csv.columns else test_csv["patientId"])

    assert len(s_tr.intersection(s_val)) == 0
    assert len(s_tr.intersection(s_te)) == 0
    assert len(s_val.intersection(s_te)) == 0
