from pycaret.classification import load_model

model = load_model("dataset_quality_model")

print(type(model))
print(model)