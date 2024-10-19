from flask import Flask, render_template, request, jsonify
from PIL import Image
import torch
from torchvision import transforms
import torch.nn as nn
from bot import chatbot_response, load_nutritional_info

# Load the food classification model (PyTorch)
class FoodClassificationModel(nn.Module):
    def __init__(self):
        super(FoodClassificationModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 56 * 56, 512)
        self.fc2 = nn.Linear(512, 5)  # Assuming 5 output classes

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 56 * 56)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
food_classification_model = FoodClassificationModel().to(device)
food_classification_model.load_state_dict(torch.load('C:/Users/sharr/OneDrive/Desktop/NLP pytorch/best_food_model.pth', map_location=device))
food_classification_model.eval()

# Define the image transformations (for preprocessing)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to the model's expected input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize the image
])

# Initialize Flask app
app = Flask(__name__)

# Home route to serve the HTML file
@app.route('/')
def index():
    return render_template('index.html')

# Route to predict food from image
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    img = Image.open(file.stream).convert('RGB')  # Ensure image is in RGB format
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = food_classification_model(img_tensor)
        predicted_class_index = torch.argmax(predictions, dim=1).item()
    
    # List of food class names
    class_names = ["pizza", "pasta", "pancake", "French fries", "donut"]
    predicted_food = class_names[predicted_class_index]

    # Load the respective nutritional information
    nutrition_info = load_nutritional_info(predicted_food)
    
    return jsonify({
        'prediction': predicted_food,
        'food_info': nutrition_info
    })

# Route to handle chatbot interactions
@app.route('/chatbot', methods=['POST'])
def chatbot():
    data = request.get_json()
    question = data.get('question')
    predicted_food = data.get('predicted_food')

    # Load the respective food information file
    food_info = load_nutritional_info(predicted_food)

    if not food_info:
        return jsonify({'error': 'Food information not found'}), 400

    # Generate chatbot response
    response = chatbot_response(question, food_info, predicted_food)
    
    return jsonify({'response': response})

# Run Flask server
if __name__ == '__main__':
    app.run(debug=True)
