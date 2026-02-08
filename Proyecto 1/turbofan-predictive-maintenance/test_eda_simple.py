from src.data_loading import load_fd001_prepared

print("Cargando datos...")
train_df, test_df = load_fd001_prepared(save_processed=False)

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

print("\nRUL stats:")
print(train_df["RUL"].describe())

print("\nMemoria usada:", train_df.memory_usage(deep=True).sum() / 1024**2, "MB")
