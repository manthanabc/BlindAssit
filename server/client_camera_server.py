#!/usr/bin/env python3
"""
BlindAssit Navigation Server
Uses YOLO11 for sidewalk/obstacle detection with server-side TTS and OpenRouter.
"""

import sys
import os
import time
import json
import base64
import requests
from io import BytesIO
from collections import deque

# Import from original repo
import cv2
import argparse
import torch
import numpy as np

# YOLO model
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️  YOLO not installed. Install with: pip install ultralytics")

# Server-side TTS
try:
    import gtts
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("⚠️  gTTS not installed. Install with: pip install gtts")

# OpenAI client for OpenRouter
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️  OpenAI not installed. Install with: pip install openai")


# Model classes for YOLO11 sidewalk segmentation
CLASS_NAMES = {
    0: "cover",
    1: "curb",
    2: "hole",
    3: "lane",
    4: "sidewalk"
}


# OpenRouter API configuration
OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY', '')
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"


class NavigationState:
    def __init__(self):
        self.last_guidance = ""
        self.guidance_count = 0
        self.obstacle_detected = False
        self.on_sidewalk = False
        self.on_left_side = False

        # Stability improvements
        self.last_spoken_time = 0
        self.min_speak_interval = 3.0  # Minimum seconds between announcements
        self.detection_history = deque(maxlen=10)  # Track last 10 frames
        self.stable_state_count = 0  # Count frames in stable state
        self.last_state_change_time = time.time()
        self.min_state_change_interval = 5.0  # Minimum seconds between state changes


nav_state = NavigationState()


def analyze_frame(frame, model):
    """Analyze frame using YOLO model and return detection results."""
    results = model(frame, verbose=False)
    result = results[0]

    detections = []
    sidewalk_detected = False
    curb_detected = False
    lane_detected = False
    hole_detected = False
    cover_detected = False

    if result.boxes is not None:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = CLASS_NAMES[cls_id]

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            # Check position relative to frame center
            frame_width = frame.shape[1]
            box_center_x = (x1 + x2) / 2
            position = "left" if box_center_x < frame_width / 2 else "right"

            detections.append({
                "class": cls_name,
                "confidence": conf,
                "bbox": [x1, y1, x2, y2],
                "position": position
            })

            # Track detections
            if cls_name == "sidewalk":
                sidewalk_detected = True
            elif cls_name == "curb":
                curb_detected = True
            elif cls_name == "lane":
                lane_detected = True
            elif cls_name == "hole":
                hole_detected = True
            elif cls_name == "cover":
                cover_detected = True

    # Update navigation state
    nav_state.on_sidewalk = sidewalk_detected
    nav_state.on_left_side = curb_detected or lane_detected
    nav_state.obstacle_detected = hole_detected or cover_detected

    return {
        "detections": detections,
        "on_sidewalk": sidewalk_detected,
        "on_left_side": curb_detected or lane_detected,
        "obstacle_detected": hole_detected or cover_detected,
        "obstacle_type": "hole" if hole_detected else ("cover" if cover_detected else None)
    }


def get_stable_state(analysis):
    """Get stable state string from analysis."""
    if analysis["obstacle_detected"]:
        return f"obstacle_{analysis['obstacle_type']}"
    elif analysis["on_sidewalk"]:
        return "sidewalk"
    elif analysis["on_left_side"]:
        return "left_side"
    else:
        return "road"


def should_announce_guidance(guidance):
    """Determine if guidance should be announced based on timing and state stability."""
    current_time = time.time()

    # Always announce obstacles immediately
    if "Warning" in guidance:
        return True

    # Check minimum time since last announcement
    time_since_last_speak = current_time - nav_state.last_spoken_time
    if time_since_last_speak < nav_state.min_speak_interval:
        return False

    # Check if guidance is different from last
    if guidance == nav_state.last_guidance:
        return False

    # Check minimum time since last state change
    time_since_state_change = current_time - nav_state.last_state_change_time
    if time_since_state_change < nav_state.min_state_change_interval:
        return False

    return True


def generate_guidance(analysis):
    """Generate voice guidance based on frame analysis with stability."""
    current_state = get_stable_state(analysis)

    # Track detection history for stability
    nav_state.detection_history.append(current_state)

    # Count how many frames have been in current state
    nav_state.stable_state_count = sum(1 for s in nav_state.detection_history if s == current_state)

    # Determine if state has changed
    last_state = get_stable_state({
        "on_sidewalk": nav_state.on_sidewalk,
        "on_left_side": nav_state.on_left_side,
        "obstacle_detected": nav_state.obstacle_detected,
        "obstacle_type": "hole" if nav_state.obstacle_detected else None
    })
    state_changed = current_state != last_state

    if state_changed:
        nav_state.last_state_change_time = time.time()
        nav_state.stable_state_count = 0

    guidance = ""
    priority = "normal"  # normal, high, urgent

    # Obstacle detection (highest priority)
    if analysis["obstacle_detected"]:
        obstacle = analysis["obstacle_type"]
        position = None
        for det in analysis["detections"]:
            if det["class"] == obstacle:
                position = det["position"]
                break

        direction = "on your left" if position == "left" else "on your right"
        guidance = f"Warning! {obstacle.capitalize()} detected {direction}. Please stop and navigate around it carefully."
        priority = "urgent"
        nav_state.obstacle_detected = True

    # Only announce stable states (require at least 3 frames of same state)
    elif nav_state.stable_state_count >= 3:
        # No sidewalk and no left side
        if not analysis["on_sidewalk"] and not analysis["on_left_side"]:
            guidance = "No sidewalk detected. Stay on left side of road and proceed carefully."
            priority = "high"

        # On sidewalk (less frequent)
        elif analysis["on_sidewalk"]:
            # Only announce if we weren't on sidewalk before
            if "sidewalk" not in nav_state.detection_history or nav_state.stable_state_count == 3:
                guidance = "You are on sidewalk. Continue walking safely."

        # On left side (less frequent)
        elif analysis["on_left_side"]:
            if "left_side" not in nav_state.detection_history or nav_state.stable_state_count == 3:
                guidance = "Good. You are on left side of road. Continue carefully."

    nav_state.guidance_count += 1

    # Only update last guidance if we actually have something to say
    if guidance:
        nav_state.last_guidance = guidance
        nav_state.last_spoken_time = time.time()

    return {
        "guidance": guidance,
        "priority": priority,
        "should_speak": should_announce_guidance(guidance),
        "state_stable": nav_state.stable_state_count >= 3,
        "current_state": current_state
    }


def text_to_speech(text):
    """
    Convert text to speech using server-side TTS.
    Returns base64 encoded audio.
    """
    if not TTS_AVAILABLE:
        print("⚠️  TTS not available")
        return None

    try:
        # Create TTS object
        tts = gtts.gTTS(text=text, lang='en', slow=False)

        # Save to BytesIO
        audio_buffer = BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)

        # Encode to base64
        audio_base64 = base64.b64encode(audio_buffer.read()).decode('utf-8')
        return audio_base64

    except Exception as e:
        print(f"❌ TTS error: {e}")
        return None


def openrouter_query(prompt, context=None):
    """
    Query OpenRouter API for audio/text response.
    """
    if not OPENROUTER_API_KEY:
        print("⚠️  OpenRouter API key not set")
        return None

    try:
        # Prepare messages
        messages = [
            {
                "role": "system",
                "content": "You are a helpful navigation assistant for blind users. Provide concise, clear navigation instructions."
            }
        ]

        if context:
            messages.append({
                "role": "system",
                "content": f"Current context: {context}"
            })

        messages.append({
            "role": "user",
            "content": prompt
        })

        # Make API request
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }

        data = {
            "model": "openai/gpt-3.5-turbo",
            "messages": messages,
            "max_tokens": 150,
            "temperature": 0.7
        }

        response = requests.post(OPENROUTER_API_URL, headers=headers, json=data, timeout=10)
        response.raise_for_status()

        result = response.json()

        # Extract response text
        if 'choices' in result and len(result['choices']) > 0:
            return result['choices'][0]['message']['content']
        else:
            print("⚠️  No response from OpenRouter")
            return None

    except Exception as e:
        print(f"❌ OpenRouter error: {e}")
        return None


def process_frame_with_navigation(frame, model):
    """Process a frame and return detection with navigation analysis."""
    # Perform inference
    analysis = analyze_frame(frame, model)
    guidance_result = generate_guidance(analysis)

    # Draw detections on frame
    annotated_frame = frame.copy()
    for det in analysis["detections"]:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        cls_name = det["class"]
        conf = det["confidence"]

        # Draw bounding box
        color = (0, 255, 0) if cls_name == "sidewalk" else (0, 0, 255)
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

        # Draw label
        label = f"{cls_name}: {conf:.2f}"
        cv2.putText(annotated_frame, label, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return annotated_frame, analysis, guidance_result


# Flask web app
from flask import Flask, render_template, Response, jsonify, request

# Get directory of current script
current_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(current_dir, 'templates')

app = Flask(__name__, template_folder=template_dir)

# Global state
model = None
inference_times = []


def init_model():
    """Initialize the YOLO model."""
    global model

    print("Initializing YOLO11 model...")
    try:
        model_path = '/mnt/b/local_projects/SeaFormer/yolo11m-sidewalk-seg.pt'

        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            return False

        model = YOLO(model_path)
        print("✅ Model initialized successfully!")
        print(f"   Model classes: {model.names}")
        return True
    except Exception as e:
        print(f"❌ Error initializing model: {e}")
        return False


@app.route('/')
def index():
    """Main page."""
    return render_template('client_camera.html')


@app.route('/infer', methods=['POST'])
def infer():
    """Process client-side camera frame for inference."""
    if model is None:
        return jsonify({'error': 'Model not initialized'}), 500

    try:
        # Get frame data from request
        data = request.get_json()
        image_data = data.get('image', '')

        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400

        # Remove data URL prefix
        image_data = image_data.split(',')[1]

        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image_np = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({'error': 'Invalid image data'}), 400

        # Process frame
        start_time = time.time()
        annotated_frame, analysis, guidance_result = process_frame_with_navigation(frame, model)
        inference_time = time.time() - start_time
        inference_times.append(inference_time)

        # Generate TTS audio for instruction
        audio_base64 = None
        if guidance_result["should_speak"] and guidance_result["guidance"]:
            audio_base64 = text_to_speech(guidance_result["guidance"])

        # Encode annotated frame back to base64
        _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        annotated_base64 = base64.b64encode(buffer).decode('utf-8')

        # Calculate average metrics
        avg_inference_time = np.mean(inference_times[-30:]) if len(inference_times) >= 30 else np.mean(inference_times)

        return jsonify({
            'annotated_image': f'data:image/jpeg;base64,{annotated_base64}',
            'on_sidewalk': analysis["on_sidewalk"],
            'on_left_side': analysis["on_left_side"],
            'obstacle_detected': analysis["obstacle_detected"],
            'obstacle_type': analysis.get("obstacle_type"),
            'guidance': guidance_result["guidance"],
            'priority': guidance_result["priority"],
            'should_speak': guidance_result["should_speak"],
            'state_stable': guidance_result["state_stable"],
            'current_state': guidance_result["current_state"],
            'audio': f'data:audio/mp3;base64,{audio_base64}' if audio_base64 else None,
            'inference_time_ms': f"{inference_time * 1000:.0f}",
            'avg_inference_time_ms': f"{avg_inference_time * 1000:.0f}",
            'fps': f"{1.0 / avg_inference_time:.1f}" if avg_inference_time > 0 else "0.0"
        })

    except Exception as e:
        print(f"Inference error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/tts', methods=['POST'])
def tts():
    """Server-side text-to-speech endpoint."""
    try:
        data = request.get_json()
        text = data.get('text', '')

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        # Generate TTS audio
        audio_base64 = text_to_speech(text)

        if not audio_base64:
            return jsonify({'error': 'TTS generation failed'}), 500

        return jsonify({
            'audio': f'data:audio/mp3;base64,{audio_base64}'
        })

    except Exception as e:
        print(f"TTS error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/query', methods=['POST'])
def query():
    """
    Query OpenRouter for navigation assistance.
    Returns both text and audio response.
    """
    try:
        data = request.get_json()
        prompt = data.get('prompt', '')
        context = data.get('context', None)

        if not prompt:
            return jsonify({'error': 'No prompt provided'}), 400

        # Query OpenRouter
        response_text = openrouter_query(prompt, context)

        if not response_text:
            return jsonify({'error': 'Failed to get response from OpenRouter'}), 500

        # Generate TTS audio for response
        audio_base64 = text_to_speech(response_text)

        return jsonify({
            'response': response_text,
            'audio': f'data:audio/mp3;base64,{audio_base64}' if audio_base64 else None
        })

    except Exception as e:
        print(f"Query error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok' if model is not None else 'model_not_loaded',
        'yolo_available': YOLO_AVAILABLE,
        'tts_available': TTS_AVAILABLE,
        'openrouter_available': bool(OPENROUTER_API_KEY),
        'inference_count': len(inference_times),
        'avg_inference_time_ms': f"{np.mean(inference_times) * 1000:.0f}" if inference_times else "0.0"
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    args = parser.parse_args()

    if init_model():
        print(f"\n🌐 BlindAssit Navigation Server")
        print(f"📍 http://{args.host}:{args.port}")
        print("\n📱 Features:")
        print("   • Client-side camera capture")
        print("   • Real-time sidewalk/obstacle detection")
        print("   • Navigation assistance")
        print("   • Server-side TTS (no browser API)")
        print("   • OpenRouter AI queries")
        print(f"   • YOLO Available: {'✅' if YOLO_AVAILABLE else '❌'}")
        print(f"   • TTS Available: {'✅' if TTS_AVAILABLE else '❌'}")
        print(f"   • OpenRouter Available: {'✅' if OPENROUTER_API_KEY else '❌ (Set OPENROUTER_API_KEY)'}")
        print("\nPress Ctrl+C to stop\n")
        app.run(host=args.host, port=args.port, debug=False)
    else:
        print("\n❌ Failed to initialize model. Exiting.")
        sys.exit(1)
