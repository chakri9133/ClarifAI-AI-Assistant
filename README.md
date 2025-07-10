
# 📚 ClarifAI – Multimodal, Emotion-Aware AI Learning Assistant

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
git clone https://github.com/chakri9133/clarifai-edu-assistant.git
cd clarifai-edu-assistant
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

Project Source Code: [ClarifAI GitHub](https://github.com/chakri9133/ClarifAI-Project-)

---

## 🤝 Contributors

- **Hasya** – Voice input, image captioning, OCR, visual rendering
- **Chakri** – Backend logic, LLM integration, emotion detection
- **Hima Sree** – Frontend UI, voice + emotion display, visual rendering

---

## 📄 License

This project is part of the Intel Unnati Internship and is licensed for academic use.

