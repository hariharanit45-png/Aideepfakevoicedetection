from utils import load_data, train_model, save_model

X, y = load_data('data/dataset.csv')
model = train_model(X, y)
save_model(model, 'model.pkl')
print("Model trained and saved as model.pkl")