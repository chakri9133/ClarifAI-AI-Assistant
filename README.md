# 📚 ClarifAI – AI that Listens, Sees and Explains

ClarifAI is a real-time AI-powered educational assistant designed to support students through multimodal inputs and adaptive, emotion-aware responses. It can handle **text**, **voice**, and **image-based queries**, and responds using a **local LLM (Gemma 2B via Ollama)**. The system detects student emotions using webcam analysis and simplifies explanations based on real-time feedback, creating a responsive, classroom-friendly AI tutor.

---

## 🔧 How It Works

### 🔹 Text Interaction
- Users type a question into the chat.
- Input is filtered for educational relevance and passed to the local LLM.
- AI generates a contextual, academic response with optional visual aid.

### 🔹 Voice Input
- Web Speech API captures user speech and transcribes it in real-time.
- Transcribed query is treated the same as a typed input.
- Responses are streamed live to the user interface.

### 🔹 Image Input
- Users upload an image (diagram, handwritten question, etc.).
- System uses **EasyOCR** for text extraction or **BLIP** for image captioning.
- Extracted or interpreted content is semantically matched and answered.

### 🔹 Emotion Detection
- Webcam captures frames in the background.
- OpenCV processes and detects facial expressions (happy, confused, sad, etc.).
- If negative emotion is detected, the assistant suggests a simplified explanation.

### 🔹 Visual Aid Generation
- If the AI response includes steps or structured data, the system triggers:
  - **Graphviz** to generate flowcharts.
  - **Matplotlib** for charts/graphs.

---

## 🗂️ Project Structure

```
clarifai-edu-assistant/
│
├── app.py                      # Main Flask application with routing and SSE
├── requirements.txt            # Python dependencies
├── README.md                   # Project overview
│
├── static/                     # Static assets (style.css, images, etc.)
│   └── style.css               # UI styling
│
├── templates/
│   └── index.html              # Frontend HTML file
│
├── edu_ollama_assistant.py     # LLM interaction and query handling
├── edu_voice_assistant.py      # Voice input processing (Web Speech API)
├── edu_image_assistant.py      # OCR + captioning + image query processing
├── visual_generator.py         # Flowchart/chart generation logic
```

---

## 💻 Tech Stack

| Component          | Technology                            |
|--------------------|----------------------------------------|
| Backend            | Python (Flask)                         |
| Language Model     | Gemma 2B (via Ollama)                  |
| Voice Input        | Web Speech API                         |
| Image Processing   | EasyOCR, BLIP (Hugging Face)           |
| Emotion Detection  | OpenCV                                 |
| Visuals            | Graphviz, Matplotlib                   |
| Frontend           | HTML, CSS, JavaScript                  |

---

## 🚀 Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/chakri9133/ClarifAI-AI-Assistant.git
cd ClarifAI-AI-Assistant
```

### 2. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 3. Start Ollama and load the LLM
```bash
ollama run gemma:2b
```

### 4. Run the Flask web server
```bash
python app.py
```

Visit `http://localhost:5050` in your browser.

---

## 🧪 Features Summary

- ✅ Text, voice, and image input handling  
- ✅ Local LLM integration for offline AI support  
- ✅ Live emotion detection and simplified response logic  
- ✅ Flowchart and chart generation based on response type  
- ✅ Educational keyword filtering for safe AI use  

---

## 🧠 Example Use Cases

- Ask academic questions like “Explain Newton’s Second Law” (text/voice).  
- Upload diagrams to get explanations (e.g., circuits, biology flowcharts).  
- Speak a complex question; if the system detects confusion, it offers a simpler version.  
- Get visual aids for process-based topics like food chains or life cycles.  

---

## 🔗 GitHub Repository

Project Source Code: [ClarifAI GitHub](https://github.com/chakri9133/ClarifAI-AI-Assistant)

---

## 📽️ Demo Video

Watch the project demo video here:  
▶️ [ClarifAI Demo – Google Drive](https://drive.google.com/file/d/1O16NL2WnBiTnAWRbJlajOMuEaR2oP_RI/view?usp=sharing)

---

## 🤝 Contributors

- [**Hasya**](https://github.com/Chavva-HasyaReddy) – Voice input, image captioning, OCR, visual rendering  
- [**Chakri**](https://github.com/chakri9133) – Backend logic, LLM integration, emotion detection  
- [**Hima Sree**](https://github.com/Himasree08) – Frontend UI, voice + emotion display, visual rendering  

---

## 📄 License

This project is part of the Intel Unnati Internship and is licensed for academic use.
