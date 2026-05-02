# Inside data/dawnload.py
import kagglehub

def get_dataset_path():
    path = kagglehub.dataset_download("gowrishankarp/newspaper-text-summarization-cnn-dailymail")
    return path


if __name__ == "__main__":
    print(get_dataset_path())
