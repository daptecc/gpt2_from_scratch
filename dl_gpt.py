from gpt_download import download_and_load_gpt2
settings, params = download_and_load_gpt2(
    model_size="124M", models_dir="gpt2"
)

print("settings:", settings)
print("parameter dictionary keys:", params.keys())