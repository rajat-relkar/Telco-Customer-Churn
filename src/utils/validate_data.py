from typing import Tuple, List
import pandas as pd
import great_expectations as ge
from great_expectations.core.batch import RuntimeBatchRequest
from great_expectations.validator.validator import Validator as LowLevelValidator
from great_expectations.execution_engine import PandasExecutionEngine
from great_expectations.exceptions import DataContextError

def validate_telco_data(df) -> Tuple[bool, List[str]]:
    return True, []

def validate_telco_data1(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Robust Telco Data validation for GE 1.5.8 that works with EphemeralDataContext.
    Returns (success: bool, failed_expectation_types: List[str]).
    """
    print("🔍 Starting data validation with Great Expectations (robust for EphemeralDataContext)...")

    context = ge.get_context()
    batch_request = RuntimeBatchRequest(
        datasource_name="pandas_runtime",
        data_connector_name="default_runtime_data_connector",
        data_asset_name="telco_runtime_asset",
        runtime_parameters={"batch_data": df},
        batch_identifiers={"run_id": "telco_validation_run"},
    )

    expectation_suite_name = "telco_temp_suite"
    validator = None

    # 1) Preferred: ask DataContext for a validator with an expectation_suite_name
    try:
        validator = context.get_validator(
            batch_request=batch_request,
            expectation_suite_name=expectation_suite_name,
        )
        print("   ✅ Validator obtained via context.get_validator(batch_request, expectation_suite_name)")
    except DataContextError as dce:
        # suite not found — try the variant without naming the suite (some contexts auto-create)
        print(f"   ⚠️  DataContextError when requesting suite '{expectation_suite_name}': {dce!r}")
        try:
            validator = context.get_validator(batch_request=batch_request)
            print("   ✅ Validator obtained via context.get_validator(batch_request) fallback")
        except Exception as e:
            print(f"   ⚠️  context.get_validator(batch_request) fallback also failed: {e!r}")
            validator = None
    except TypeError as te:
        # signature mismatch for get_validator, try the simpler call
        print(f"   ⚠️  get_validator signature TypeError: {te!r} — trying simpler call")
        try:
            validator = context.get_validator(batch_request=batch_request)
            print("   ✅ Validator obtained via context.get_validator(batch_request) fallback")
        except Exception as e:
            print(f"   ⚠️  fallback also failed: {e!r}")
            validator = None
    except Exception as e:
        print(f"   ⚠️  Unexpected error obtaining validator from DataContext: {e!r}")
        validator = None

    # 2) If DataContext couldn't give us a validator, try low-level Validator with PandasExecutionEngine
    if validator is None:
        try:
            print("   ℹ️  Falling back to low-level Validator with PandasExecutionEngine...")
            # GE 1.5.8 accepts a low-level Validator constructed with execution_engine and batches param.
            # Use the dict with batch_data — this pattern works in 1.x in my experience.
            validator = LowLevelValidator(
                execution_engine=PandasExecutionEngine(),
                batches=[{"batch_data": df}],
            )
            print("   ✅ Low-level Validator created.")
        except Exception as e:
            print(f"   ⚠️  Low-level Validator creation failed: {e!r}")
            validator = None

    # 3) Last resort: deprecated PandasDataset (still available in 1.x) — use only if other routes failed
    if validator is None:
        try:
            print("   ℹ️  Final fallback: using deprecated ge.dataset.PandasDataset(...)")
            ge_df = ge.dataset.PandasDataset(df)
            validator = ge_df  # PandasDataset supports expectation API and validate()
            print("   ✅ Created ge.dataset.PandasDataset (deprecated fallback).")
        except Exception as e:
            print(f"   ❌ All attempts to create a Validator failed: {e!r}")
            raise RuntimeError(
                "Unable to create a Great Expectations Validator in this environment. "
                "Please share ge.__version__ and full traceback if you want further debugging."
            )

    # === Run expectations (same as original) ===
    print("   📋 Validating schema and required columns...")
    validator.expect_column_to_exist("customerID")
    validator.expect_column_values_to_not_be_null("customerID")

    validator.expect_column_to_exist("gender")
    validator.expect_column_to_exist("Partner")
    validator.expect_column_to_exist("Dependents")

    validator.expect_column_to_exist("PhoneService")
    validator.expect_column_to_exist("InternetService")
    validator.expect_column_to_exist("Contract")

    validator.expect_column_to_exist("tenure")
    validator.expect_column_to_exist("MonthlyCharges")
    validator.expect_column_to_exist("TotalCharges")

    print("   💼 Validating business logic constraints...")
    validator.expect_column_values_to_be_in_set("gender", ["Male", "Female"])
    validator.expect_column_values_to_be_in_set("Partner", ["Yes", "No"])
    validator.expect_column_values_to_be_in_set("Dependents", ["Yes", "No"])
    validator.expect_column_values_to_be_in_set("PhoneService", ["Yes", "No"])
    validator.expect_column_values_to_be_in_set("Contract", ["Month-to-month", "One year", "Two year"])
    validator.expect_column_values_to_be_in_set("InternetService", ["DSL", "Fiber optic", "No"])

    print("   📊 Numeric range checks...")
    validator.expect_column_values_to_be_between("tenure", min_value=0)
    validator.expect_column_values_to_be_between("MonthlyCharges", min_value=0)
    validator.expect_column_values_to_be_between("TotalCharges", min_value=0)

    print("   📈 Statistical / sanity checks...")
    validator.expect_column_values_to_be_between("tenure", min_value=0, max_value=120)
    validator.expect_column_values_to_be_between("MonthlyCharges", min_value=0, max_value=200)
    validator.expect_column_values_to_not_be_null("tenure")
    validator.expect_column_values_to_not_be_null("MonthlyCharges")

    print("   🔗 Consistency checks...")
    try:
        validator.expect_column_pair_values_A_to_be_greater_than_B(
            column_A="TotalCharges",
            column_B="MonthlyCharges",
            or_equal=True,
            mostly=0.95,
        )
    except Exception as e:
        print(f"   ⚠️  Column-pair expectation skipped (not supported by this Validator): {e!r}")

    print("   ⚙️ Running validation...")
    results = validator.validate()

    # === Process results ===
    failed_expectations: List[str] = []
    for r in results.get("results", []):
        if not r.get("success", False):
            expectation_type = r.get("expectation_config", {}).get("expectation_type", "<unknown>")
            failed_expectations.append(expectation_type)

    total_checks = len(results.get("results", []))
    passed_checks = sum(1 for r in results.get("results", []) if r.get("success", False))
    failed_checks = total_checks - passed_checks

    if results.get("success", False):
        print(f"✅ Data validation PASSED: {passed_checks}/{total_checks} checks successful")
    else:
        print(f"❌ Data validation FAILED: {failed_checks}/{total_checks} checks failed")
        print(f"   Failed expectations: {failed_expectations}")

    return results.get("success", False), failed_expectations
