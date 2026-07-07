def test_dataset_loads():
    df = load_dataset("data/good.csv")
    assert len(df) > 0