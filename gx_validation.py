import great_expectations as ge

# Load dataset
df = ge.read_csv("data/bad.csv")

print("=" * 65)
print("                 DATA VALIDATION REPORT")
print("=" * 65)

checks = {}

def add_check(check_name, column_name, validation_function):
    if column_name in df.columns:
        try:
            checks[check_name] = validation_function()["success"]
        except Exception:
            checks[check_name] = False
    else:
        checks[check_name] = None


#-------- COMPLETENESS ---------
add_check(
    "ID Not Null",
    "id",
    lambda: df.expect_column_values_to_not_be_null("id")
)

add_check(
    "Name Not Null",
    "name",
    lambda: df.expect_column_values_to_not_be_null("name")
)

add_check(
    "Email Not Null",
    "email",
    lambda: df.expect_column_values_to_not_be_null("email")
)

add_check(
    "City Not Null",
    "city",
    lambda: df.expect_column_values_to_not_be_null("city")
)

add_check(
    "DOB Not Null",
    "dob",
    lambda: df.expect_column_values_to_not_be_null("dob")
)

add_check(
    "Last Updated Not Null",
    "last_updated",
    lambda: df.expect_column_values_to_not_be_null("last_updated")
)

#---------- CONSISTENCY-------------
add_check(
    "ID Unique",
    "id",
    lambda: df.expect_column_values_to_be_unique("id")
)

add_check(
    "Age Datatype",
    "age",
    lambda: df.expect_column_values_to_be_of_type("age", "int64")
)

# ---------ACCURACY----------------
add_check(
    "Age Between 0-120",
    "age",
    lambda: df.expect_column_values_to_be_between(
        "age",
        min_value=0,
        max_value=120
    )
)

add_check(
    "Salary Positive",
    "salary",
    lambda: df.expect_column_values_to_be_between(
        "salary",
        min_value=0
    )
)

# -----------VALIDITY---------------------
add_check(
    "Email Format",
    "email",
    lambda: df.expect_column_values_to_match_regex(
        "email",
        r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"
    )
)

add_check(
    "Phone Format",
    "phone",
    lambda: df.expect_column_values_to_match_regex(
        "phone",
        r"^[6-9]\d{9}$"
    )
)

add_check(
    "Gender Valid",
    "gender",
    lambda: df.expect_column_values_to_be_in_set(
        "gender",
        ["Male", "Female", "Other"]
    )
)

# Display Function
def display_section(title, rule_names):
    print(f"\n{title}")
    print("-" * 65)

    for rule in rule_names:

        status = checks[rule]

        if status is True:
            result = "PASS"
        elif status is False:
            result = "FAIL"
        else:
            result = "SKIPPED"

        print(f"{rule:<35}: {result}")


display_section(
    "COMPLETENESS",
    [
        "ID Not Null",
        "Name Not Null",
        "Email Not Null",
        "City Not Null",
        "DOB Not Null",
        "Last Updated Not Null"
    ]
)

display_section(
    "CONSISTENCY",
    [
        "ID Unique",
        "Age Datatype"
    ]
)

display_section(
    "ACCURACY",
    [
        "Age Between 0-120",
        "Salary Positive"
    ]
)

display_section(
    "VALIDITY",
    [
        "Email Format",
        "Phone Format",
        "Gender Valid"
    ]
)

# Summary
executed = sum(1 for value in checks.values() if value is not None)
passed = sum(1 for value in checks.values() if value is True)
failed = sum(1 for value in checks.values() if value is False)
skipped = sum(1 for value in checks.values() if value is None)

success_rate = (passed / executed) * 100 if executed else 0

print("\n" + "=" * 65)
print("                    VALIDATION SUMMARY")
print("=" * 65)
print(f"Total Rules              : {len(checks)}")
print(f"Executed Rules           : {executed}")
print(f"Passed Rules             : {passed}")
print(f"Failed Rules             : {failed}")
print(f"Skipped Rules            : {skipped}")
print(f"Validation Success Rate  : {success_rate:.2f}%")
print("=" * 65)