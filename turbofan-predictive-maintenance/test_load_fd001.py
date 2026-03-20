from src.data_loading import load_fd001_prepared

if __name__ == "__main__":
    train_df, test_df = load_fd001_prepared(save_processed=False)

    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)
    print("\nTrain head:")
    print(train_df.head())
    print("\nTest head:")
    print(test_df.head())
