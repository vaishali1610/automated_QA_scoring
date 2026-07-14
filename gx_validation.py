import great_expectations as ge


COLUMN_ALIASES = {
    "id": ["id", "emp_id", "customer_id"],
    "name": ["name", "customer_name", "employee_name"],
    "email": ["email", "email_id", "mail"],
    "phone": ["phone", "mobile", "contact", "contact_number"],
    "gender": ["gender", "sex"],
    "city": ["city", "location"],
    "age": ["age"],
    "amount": ["salary", "amount", "price", "income", "balance", "cost"],
    "date": ["last_updated", "updated_at", "created_date", "date", "dob"]
}


def find_column(df, aliases):
    for alias in aliases:
        if alias in df.columns:
            return alias
    return None


def validate_dataset(df):

    checks = {}

    def add_check(check_name, column_name, validation_function):

        if column_name:

            try:
                checks[check_name] = validation_function()["success"]

            except Exception:
                checks[check_name] = False

        else:

            checks[check_name] = None

    id_col = find_column(df, COLUMN_ALIASES["id"])
    name_col = find_column(df, COLUMN_ALIASES["name"])
    email_col = find_column(df, COLUMN_ALIASES["email"])
    phone_col = find_column(df, COLUMN_ALIASES["phone"])
    gender_col = find_column(df, COLUMN_ALIASES["gender"])
    city_col = find_column(df, COLUMN_ALIASES["city"])
    age_col = find_column(df, COLUMN_ALIASES["age"])
    amount_col = find_column(df, COLUMN_ALIASES["amount"])
    date_col = find_column(df, COLUMN_ALIASES["date"])

    add_check(
        "ID Not Null",
        id_col,
        lambda: df.expect_column_values_to_not_be_null(id_col)
    )

    add_check(
        "Name Not Null",
        name_col,
        lambda: df.expect_column_values_to_not_be_null(name_col)
    )

    add_check(
        "Email Not Null",
        email_col,
        lambda: df.expect_column_values_to_not_be_null(email_col)
    )

    add_check(
        "City Not Null",
        city_col,
        lambda: df.expect_column_values_to_not_be_null(city_col)
    )

    add_check(
        "Date Not Null",
        date_col,
        lambda: df.expect_column_values_to_not_be_null(date_col)
    )

    add_check(
        "ID Unique",
        id_col,
        lambda: df.expect_column_values_to_be_unique(id_col)
    )

    add_check(
        "Age Datatype",
        age_col,
        lambda: df.expect_column_values_to_be_of_type(
            age_col,
            "int64"
        )
    )

    add_check(
        "Age Between 0-120",
        age_col,
        lambda: df.expect_column_values_to_be_between(
            age_col,
            min_value=0,
            max_value=120
        )
    )

    add_check(
        "Positive Numeric",
        amount_col,
        lambda: df.expect_column_values_to_be_between(
            amount_col,
            min_value=0
        )
    )

    add_check(
        "Email Format",
        email_col,
        lambda: df.expect_column_values_to_match_regex(
            email_col,
            r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"
        )
    )

    add_check(
        "Phone Format",
        phone_col,
        lambda: df.expect_column_values_to_match_regex(
            phone_col,
            r"^[6-9]\d{9}$"
        )
    )

    add_check(
        "Gender Valid",
        gender_col,
        lambda: df.expect_column_values_to_be_in_set(
            gender_col,
            ["Male", "Female", "Other"]
        )
    )

    return checks


def print_validation_report(checks):

    print("\n" + "=" * 60)
    print("               DATA VALIDATION REPORT")
    print("=" * 60)

    for rule, status in checks.items():

        if status is True:
            result = "PASS"

        elif status is False:
            result = "FAIL"

        else:
            result = "SKIPPED"

        print(f"{rule:<35}: {result}")

    executed = sum(v is not None for v in checks.values())
    passed = sum(v is True for v in checks.values())
    failed = sum(v is False for v in checks.values())
    skipped = sum(v is None for v in checks.values())

    success_rate = (
        (passed / executed) * 100
        if executed
        else 0
    )

    print("\n" + "=" * 60)
    print("                 VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Executed Rules          : {executed}")
    print(f"Passed Rules            : {passed}")
    print(f"Failed Rules            : {failed}")
    print(f"Skipped Rules           : {skipped}")
    print(f"Validation Success Rate : {success_rate:.2f}%")
    print("=" * 60)