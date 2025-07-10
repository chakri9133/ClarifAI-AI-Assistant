# app.py
from flask import Flask, render_template, request, jsonify, Response, stream_with_context, session
from flask_session import Session

from edu_ollama_assistant import handle_math, is_educational
from edu_image_assistant import analyze_image_file
from deepface import DeepFace # type: ignore
from visual_generator import needs_visual_aid, extract_steps_from_text, generate_flowchart, generate_chart

import cv2
import json
import threading
import time
import requests
import re
import os
import warnings
import random

# Suppress TensorFlow and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["TORCH_ELASTIC_ERROR_FILE"] = "NUL"
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = 'clarifai-secret'  # Change in production
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

current_emotion = {"value": "neutral"}

# Realtime formatting (chunk-wise)
def format_chunk(chunk):
    chunk = re.sub(r"\*\*(.+?)\*\*:", r"<b>\1:</b>", chunk)
    chunk = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", chunk)
    chunk = re.sub(r"[\r\n]*[-*]\s+", "<br>• ", chunk)
    chunk = re.sub(r"(?<!<br>)\s*(\d+\.\s+)", r"<br>\1", chunk)
    return chunk

# Emotion Detection Thread
# 📦 Import this if not already done
from deepface import DeepFace # type: ignore
import cv2
import time

# Global emotion state
current_emotion = {"value": "neutral"}

# Optimized Emotion Detection Loop
def detect_emotion_loop():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use CAP_DSHOW for Windows, else drop the flag
    last_detection = time.time()

    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                time.sleep(1)
                continue

            # Only run detection every 5 seconds
            if time.time() - last_detection >= 5:
                frame = cv2.resize(frame, (480, 360))
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Run DeepFace with faster backend
                result = DeepFace.analyze(
                    rgb,
                    actions=['emotion'],
                    enforce_detection=False,
                    detector_backend='opencv'  # 🔁 Switch to opencv for speed
                )

                emotion = result[0]['dominant_emotion'].lower()
                current_emotion["value"] = emotion
                print(f"[Emotion] Detected: {emotion}")

                last_detection = time.time()

        except Exception as e:
            current_emotion["value"] = "neutral"
            print(f"[Emotion] Detection failed: {e}")
            time.sleep(2)


threading.Thread(target=detect_emotion_loop, daemon=True).start()

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/get_emotion')
def get_emotion():
    try:
        return jsonify({"emotion": current_emotion["value"]})
    except:
        return jsonify({"emotion": "neutral"})

@app.route('/ask_stream', methods=["POST"])
def ask_stream():
    query = request.json.get("query", "").strip()

    def generate():
        if not query:
            yield "Please enter a valid question."
            return

        memory = session.get("chat_memory", [])
        memory_prompt = "\n".join([f"User: {q}\nAssistant: {a}" for q, a in memory[-3:]])
        full_prompt = f"{memory_prompt}\nUser: {query}\nAssistant:"

        math_result = handle_math(query)
        if math_result:
            session["chat_memory"] = memory + [(query, math_result)]
            yield format_chunk(math_result)
            return

        if not is_educational(query):
            warning = "Let's focus on academic topics. Please ask something educational."
            session["chat_memory"] = memory + [(query, warning)]
            yield format_chunk(warning)
            return

        buffer = ""
        try:
            with requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "gemma:2b", "prompt": full_prompt, "temperature": 0.3, "top_p": 0.9, "stream": True},
                stream=True,
                timeout=60
            ) as response:
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line.decode("utf-8"))
                            chunk = data.get("response", "")
                            buffer += chunk
                            yield format_chunk(chunk)
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            error_msg = f"⚠️ Model connection error: {str(e)}"
            yield format_chunk(error_msg)

        memory.append((query, buffer.strip()))
        session["chat_memory"] = memory

        # Emotion-based intervention suggestion
        emotion = current_emotion["value"]
        if emotion in ["confused", "sad", "angry", "frustrated"]:
            suggestion = "\n\n🤖 It looks like you're having some trouble. Would you like me to show an example, a diagram, or explain it in a simpler way?"
            buffer += suggestion
            yield format_chunk(suggestion)
        # ✅ Flowchart generation only when explicitly asked
        if "flowchart" in query.lower():
            steps = extract_steps_from_text(buffer)
            if steps and len(steps) >= 2:
                image_path = generate_flowchart("Flowchart", steps)
                if image_path:
                    yield f'<br><img src="{image_path}" alt="Flowchart" style="margin-top:10px; max-width:100%; display:block; margin:auto;"><br>'


        # Chart detection first
        # 📊 Chart detection in assistant response
        combined_text = buffer + "\n" + query

        # 1. Look for explicit [CHART:...] tags in AI response
        chart_match = re.search(r"\[CHART:(\w+)\]\s*(.*?)\s*\[DATA:(.*?)\]", combined_text, re.DOTALL)
        if chart_match:
            chart_type = chart_match.group(1).lower()
            chart_title = chart_match.group(2).strip()
            data_raw = chart_match.group(3).strip()
            try:
                pairs = [item.strip() for item in data_raw.split(",")]
                labels, values = zip(*[(p.split(":")[0].strip(), float(p.split(":")[1].strip())) for p in pairs])
                image_path = generate_chart(chart_type, labels, values, chart_title)
                if image_path:
                    yield f'<br><img src="{image_path}" alt="{chart_type.capitalize()} Chart" style="margin-top:10px; max-width:100%; display:block; margin:auto;"><br>'
                return
            except Exception as e:
                yield f"<br>⚠️ Chart generation failed: {str(e)}"

        # 2. Auto chart detection from AI response even without tags
        auto_data_match = re.findall(r"(\w+)\s*[:=]\s*(\d+\.?\d*)", buffer)
        if len(auto_data_match) >= 3:  # at least 3 data points
            labels, values = zip(*[(label.title(), float(val)) for label, val in auto_data_match])
            chart_type = "pie" if "percentage" in query.lower() or "distribution" in query.lower() else "bar"
            chart_title = "Auto-Generated Chart Based on Response"
            image_path = generate_chart(chart_type, labels, values, chart_title)
            if image_path:
                yield f"<br><b>{chart_type.capitalize()} Chart - {chart_title}:</b><br>"
                yield f'<img src="{image_path}" alt="{chart_type.capitalize()} Chart" style="margin-top:10px; max-width:100%; display:block; margin:auto;"><br>'
            return
                # 2.5: Try parsing markdown table as chart data
        markdown_match = re.findall(r"(\w[\w\s]*)\s*\|\s*(\d+\.?\d*)", buffer)
        if len(markdown_match) >= 3:
            try:
                labels, values = zip(*[(label.strip(), float(val)) for label, val in markdown_match])
                chart_type = "pie" if "pie" in query.lower() else "bar"
                chart_title = "Subject-wise Student Marks"
                image_path = generate_chart(chart_type, labels, values, title=chart_title)
                if image_path:
                    yield f"<br><b>{chart_type.capitalize()} Chart - {chart_title}:</b><br>"
                    yield f'<img src="{image_path}" alt="{chart_type.capitalize()} Chart" style="margin-top:10px; max-width:100%; display:block; margin:auto;"><br>'
                return
            except Exception as e:
                yield f"<br>⚠️ Failed to parse markdown chart: {str(e)}"


        # 3. Fallback detection from query only
        fallback_match = re.search(r"(bar|pie|line|histogram|scatter)[ -]?(chart|graph).*?(?:of|showing)?\s*(.*?)(?:\:|$)", query, re.IGNORECASE)
        if fallback_match:
            chart_type = fallback_match.group(1).lower()
            chart_title = fallback_match.group(3).strip().title()

            # Try data in prompt
            data_match = re.search(r":\s*([A-Za-z0-9\s:,]+)", query)
            if data_match:
                try:
                    data_raw = data_match.group(1)
                    pairs = [p.strip() for p in data_raw.split(",")]
                    labels, values = zip(*[(p.split(":")[0].strip(), float(p.split(":")[1].strip())) for p in pairs])
                    image_path = generate_chart(chart_type, labels, values, chart_title)
                    if image_path:
                        yield f"<br><b>{chart_type.capitalize()} Chart - {chart_title}:</b><br>"
                        yield f'<img src="{image_path}" alt="{chart_type.capitalize()} Chart" style="margin-top:10px; max-width:100%; display:block; margin:auto;"><br>'
                    return
                except Exception as e:
                    yield f"<br>⚠️ Failed to generate chart from provided data: {str(e)}"
                    return

            # Use dummy data
            dummy_labels = ["Category A", "Category B", "Category C"]
            dummy_values = random.sample(range(40, 90), 3)
            image_path = generate_chart(chart_type, dummy_labels, dummy_values, title=chart_title)
            if image_path:
                yield f"<br><b>Here's a sample {chart_type} chart for {chart_title}:</b><br>"
                yield f'<img src="{image_path}" alt="{chart_type.capitalize()} Chart" style="margin-top:10px; max-width:100%; display:block; margin:auto;"><br>'
            return


        # Auto chart generation even without proper tag
        # Auto chart generation even without proper tag
# Auto chart or graph generation even without proper tag
        fallback_match = re.search(r"(bar|pie|line|histogram|scatter)[ -]?(chart|graph).*?(?:of|showing)?\s*(.*?)(?:\:|$)", query, re.IGNORECASE)

        if fallback_match:
            chart_type = fallback_match.group(1).lower()
            chart_title = fallback_match.group(3).strip().title()

            # Try to extract data from the rest of the query (after ":")
            data_match = re.search(r":\s*([A-Za-z0-9\s:,]+)", query)
            if data_match:
                try:
                    data_raw = data_match.group(1)
                    pairs = [p.strip() for p in data_raw.split(",")]
                    labels, values = zip(*[(p.split(":")[0].strip(), float(p.split(":")[1].strip())) for p in pairs])
                    image_path = generate_chart(chart_type, labels, values, chart_title)
                    if image_path:
                        yield f"<br><b>{chart_type.capitalize()} Chart - {chart_title}:</b><br>"
                        yield f'<img src="{image_path}" alt="{chart_type.capitalize()} Chart" style="margin-top:10px; max-width:100%; display:block; margin:auto;"><br>'
                    return
                except Exception as e:
                    yield f"<br>⚠️ Failed to generate chart from provided data: {str(e)}"
                    return

            # If no values given, use dummy data
            dummy_labels = ["Category A", "Category B", "Category C"]
            dummy_values = random.sample(range(40, 90), 3)
            image_path = generate_chart(chart_type, dummy_labels, dummy_values, title=chart_title)
            if image_path:
                yield f"<br><b>Here's a sample {chart_type} chart for {chart_title}:</b><br>"
                yield f'<img src="{image_path}" alt="{chart_type.capitalize()} Chart" style="margin-top:10px; max-width:100%; display:block; margin:auto;"><br>'
            return


        # Fallback to flowchart ONLY if not chart-related
        if not re.search(r"(bar|pie|line|histogram|scatter)[ -]?chart", query, re.IGNORECASE):
            if needs_visual_aid(buffer) and "⚠️" not in buffer:
                steps = extract_steps_from_text(buffer)
                steps = [s for s in steps if 3 <= len(s) <= 180]
                unique_steps = list(dict.fromkeys(steps))
                if len(unique_steps) >= 3:
                    image_path = generate_flowchart("Generated Flowchart", unique_steps)
                    yield f'<br><img src="{image_path}" alt="Flowchart" style="margin-top:10px; max-width:100%; display:block; margin:auto;"><br>'

    return Response(stream_with_context(generate()), content_type='text/html')

@app.route("/analyze_image", methods=["POST"])
def analyze_image():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image_file = request.files["image"]
    image_path = "./temp_image.jpg"
    image_file.save(image_path)

    explanation = analyze_image_file(image_path)
    return jsonify({"caption": format_chunk(explanation)})

if __name__ == '__main__':
    import webbrowser
    webbrowser.open("http://127.0.0.1:5050")
    app.run(debug=False, port=5050)