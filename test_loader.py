from preprocessing.load_asvspoof import load_asvspoof

X_train, y_train = load_asvspoof(
    "data/asvspoof/train",
    "data/asvspoof/train/labels.txt",
    max_files=100
)

print(X_train.shape, y_train.shape)
