import os
import pandas as pd
import pandera.pandas as pa
from pandera import Column, Check

from target import RAW_PATH


def get_phone_schema():
    """Strict schema specification for the raw mobile phone dataset."""
    binary = Check.isin([0, 1])
    return pa.DataFrameSchema({
        "id": Column(pa.Int, Check.ge(1), unique=True),
        "battery_power": Column(pa.Int, Check.ge(0)),
        "blue": Column(pa.Int, binary),
        "clock_speed": Column(pa.Float, Check.ge(0.0)),
        "dual_sim": Column(pa.Int, binary),
        "fc": Column(pa.Int, Check.ge(0)),
        "four_g": Column(pa.Int, binary),
        "int_memory": Column(pa.Int, Check.ge(1)),
        "m_dep": Column(pa.Float, Check.in_range(0.0, 1.0)),
        "mobile_wt": Column(pa.Int, Check.ge(1)),
        "n_cores": Column(pa.Int, Check.in_range(1, 8)),
        "pc": Column(pa.Int, Check.ge(0)),
        "px_height": Column(pa.Int, Check.ge(0)),
        "px_width": Column(pa.Int, Check.ge(0)),
        "ram": Column(pa.Int, Check.ge(1)),
        "sc_h": Column(pa.Int, Check.ge(1)),
        "sc_w": Column(pa.Int, Check.ge(0)),
        "talk_time": Column(pa.Int, Check.ge(0)),
        "three_g": Column(pa.Int, binary),
        "touch_screen": Column(pa.Int, binary),
        "wifi": Column(pa.Int, binary),
    }, strict=True)


def validate_schema(df, output_report_name="schema_validation_errors.csv"):
    """Validates the input DataFrame against the defined schema."""
    print(f"[INFO] Validating schema (Records: {len(df)})...")
    schema = get_phone_schema()

    try:
        schema.validate(df, lazy=True)
        print("[SUCCESS] Schema Validation PASSED. Dataset is clean.")
        return True

    except pa.errors.SchemaErrors as err:
        print("[ERROR] Schema Validation FAILED. Corruptions detected.")

        failures = err.failure_cases[["schema_context", "column", "check", "failure_case", "index"]]
        print(failures.to_string())

        os.makedirs("artifacts", exist_ok=True)
        report_path = os.path.join("artifacts", output_report_name)
        failures.to_csv(report_path, index=False)
        print(f"[INFO] Detailed failure report saved to '{report_path}'.")
        return False


if __name__ == "__main__":
    if os.path.exists(RAW_PATH):
        raw_df = pd.read_csv(RAW_PATH)
        ok = validate_schema(raw_df, output_report_name="baseline_validation.csv")
        raise SystemExit(0 if ok else 1)
    else:
        print(f"[ERROR] Target file not found at: {RAW_PATH}")
        raise SystemExit(1)
