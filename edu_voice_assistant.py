import speech_recognition as sr
import pyttsx3
from edu_ollama_assistant import is_educational, handle_math

# Initialize TTS
engine = pyttsx3.init()
engine.setProperty('rate', 170)

def speak(text):
    engine.say(text)
    engine.runAndWait()

# STT setup
recognizer = sr.Recognizer()

def listen():
    with sr.Microphone() as source:
        print("\n🎙️ Listening... Speak now.")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
        try:
            query = recognizer.recognize_google(audio)
            print("You said:", query)
            return query
        except sr.UnknownValueError:
            print("❗ Sorry, I didn't catch that. Please try again.")
            return ""
        except sr.RequestError:
            print("❗ Speech service is unavailable.")
            return ""

# Generate answer with streaming + sentence-based TTS
def generate_with_ollama_streaming(query, speak_along=False):
    import requests
    import json

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "gemma:2b", "prompt": query,  "num_predict": 60, "temperature": 0.3, "top_p": 0.9},
        stream=True
    )

    print("\nAssistant: ", end="", flush=True)
    full_response = ""
    buffer = ""

    for line in response.iter_lines():
        if line:
            try:
                data = json.loads(line.decode("utf-8"))
                word = data.get("response", "")
                print(word, end="", flush=True)
                full_response += word
                buffer += word

                # Speak complete sentences
                if speak_along and buffer.count('.') + buffer.count('!') + buffer.count('?') >= 1:
                    sentences = [s.strip() for s in buffer.replace('\n', '.').split('.') if s.strip()]
                    for sentence in sentences:
                        speak(sentence + ".")
                    buffer = ""
            except json.JSONDecodeError:
                continue

    if speak_along and buffer.strip():
        speak(buffer.strip())

    return full_response.strip()

# Main loop
if __name__ == "__main__":
    while True:
        query = listen()
        if not query:
            continue

        # ✅ Flexible exit detection
        if any(exit_word in query.lower() for exit_word in ["exit", "quit", "stop"]):
            print("\nAssistant: Goodbye!")
            speak("Goodbye!")
            break

        # 🗣️ TTS check
        speak_enabled = "speak" in query.lower().split()
        query_clean = " ".join(w for w in query.lower().split() if w != "speak").strip()

        # ➗ Math handling
        math_result = handle_math(query_clean)
        if math_result:
            print("\nAssistant:", math_result)
            if speak_enabled:
                speak(math_result)
            continue

        # 📚 Educational filter
        if not is_educational(query_clean):
            fallback = "Let's focus on something educational."
            print("\nAssistant:", fallback)
            if speak_enabled:
                speak(fallback)
            continue

        # 🧠 Generate + speak
        try:
            generate_with_ollama_streaming(query_clean, speak_along=speak_enabled)
        except Exception as e:
            error_msg = "There was an issue connecting to the model."
            print("\nAssistant:", error_msg)
            if speak_enabled:
                speak(error_msg)