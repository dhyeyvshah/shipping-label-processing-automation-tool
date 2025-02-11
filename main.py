from flask import Flask, request, jsonify, render_template
import cv2
import pytesseract
from openai import OpenAI
import json

app = Flask(__name__)

api_client = OpenAI(
    api_key="YOUR API KEY"
)

def preprocess_image(image):
    image = cv2.imread(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return threshold

def extract_text(image):
    return pytesseract.image_to_string(image)

def ai_extract(text_content):
    prompt = """You are a shipping label parser AI. I am going to provide you with text extracted from an image of a shipping label. I need you to return a JSON object with this structure: {“recipient_name”, “street_address”}. Only return the JSON object. Do not return anything else. Here is the text extracted from the shipping label: """ + text_content

    response = ai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    return response.choices[0].message.content[response.choices[0].message.content.find('{'):response.choices[0].message.content.rfind('}')+1]

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/process-image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    image_file = request.files['image']
    temp_path = "temp_image.jpg"
    image_file.save(temp_path)
    preprocessed_image = preprocess_image(temp_path)
    text_content = extract_text(preprocessed_image)
    try:
        json_data = ai_extract(text_content)
        return jsonify({"data": json_data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)