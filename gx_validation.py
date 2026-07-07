import great_expectations as ge

# Load dataset
df = ge.read_csv("data/bad.csv")

print("\nChecking dataset quality...\n")

# Validation Rules
result1 = df.expect_column_values_to_not_be_null("id")

result2 = df.expect_column_values_to_be_unique("id")

result3 = df.expect_column_values_to_not_be_null("email")

result4 = df.expect_column_values_to_be_between(
    "age",
    min_value=0,
    max_value=120
)

# Validation Report
print("VALIDATION REPORT")
print("-" * 30)

checks = {
    "ID Not Null": result1["success"],
    "ID Unique": result2["success"],
    "Email Not Null": result3["success"],
    "Age Valid (0-120)": result4["success"]
}

for rule, status in checks.items():
    print(f"{rule}: {'PASS' if status else 'FAIL'}")

# Summary
passed = sum(checks.values())
total = len(checks)

print("\nSUMMARY")
print("-" * 30)
print(f"Passed Checks : {passed}")
print(f"Failed Checks : {total - passed}")
print(f"Total Checks  : {total}")
print(f"Quality Score : {(passed/total)*100:.2f}%")