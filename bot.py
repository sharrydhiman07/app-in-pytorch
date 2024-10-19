import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load the Flan-T5 model and tokenizer
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
chatbot_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
chatbot_model.to(device)

# Dictionary of text files corresponding to each food class
food_info_files = {
    "pizza": "C:/Users/sharr/OneDrive/Desktop/NLP pytorch/pizza_info.txt",
    "pasta": "C:/Users/sharr/OneDrive/Desktop/NLP pytorch/pasta_info.txt",
    "pancake": "C:/Users/sharr/OneDrive/Desktop/NLP pytorch/pancake_info.txt",
    "French fries": "C:/Users/sharr/OneDrive/Desktop/NLP pytorch/fries_info.txt",
    "donut": "C:/Users/sharr/OneDrive/Desktop/NLP pytorch/donut_info.txt"
}

# Function to load and extract only the relevant nutritional information
def load_nutritional_info(predicted_food):
    file_path = food_info_files.get(predicted_food)
    if file_path:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            nutritional_info = ""
            for line in lines:
                if "calories" in line.lower() or "protein" in line.lower() or "fat" in line.lower():
                    nutritional_info += line.strip() + "\n"
            return nutritional_info if nutritional_info else "Nutritional information not found."
    else:
        return "No information available."

# Function to generate chatbot responses
def chatbot_response(question, food_context, predicted_food):
    if not question.strip():
        return f"Can you ask something specific about {predicted_food}? 😄"

    if predicted_food.lower() not in question.lower():
        return f"Oops! I can only help you with questions about {predicted_food}. Please try asking something else! 😊"

    # Example of a more detailed response for making French fries
    if "how to make french fries" in question.lower():
        return ("Sure! Here's how to make crispy French fries at home 🍟:\n"
                "1. Slice the potatoes into thin strips.\n"
                "2. Soak them in cold water for 30 minutes to remove excess starch.\n"
                "3. Heat oil to 325°F for the first fry to cook them.\n"
                "4. Drain and let them cool.\n"
                "5. Heat oil again to 375°F for the second fry until golden and crispy.\n"
                "6. Sprinkle with salt and enjoy! 🍽️")

    # Prepare the prompt for the Flan-T5 model
    prompt = f"Question: {question}\nContext: {food_context}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    
    # Generate a response
    output = chatbot_model.generate(**inputs, max_new_tokens=100, do_sample=True, top_p=0.95, temperature=0.7)
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return answer.strip() + " 🍽️"
