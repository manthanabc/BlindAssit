#!/usr/bin/env python3
"""
Flask server for SeaFormer segmentation with client-side camera processing.
Uses server-side TTS and OpenRouter for audio queries.
"""
import sys
import os
import time
import json
import base64
import requests
from io import BytesIO

# Add original repo to path
sys.path.insert(0, '/mnt/b/local_projects/SeaFormer/seaformer-seg')

# Import from original repo
import cv2
import argparse
import torch
import numpy as np

from mmseg.apis import init_segmentor, inference_segmentor
from mmseg.core.evaluation import get_palette

# Server-side TTS
try:
    import gtts
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("⚠️  gTTS not installed. Install with: pip install gtts")

# Cityscapes class labels
CITYSCAPES_CLASSES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
    'person', 'rider', 'car', 'truck', 'bus', 'train',
    'motorcycle', 'bicycle'
]


# OpenRouter API configuration
OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY', '')
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"


def analyze_navigation_path(segmentation_mask):
    """
    Analyzes a Cityscapes segmentation mask to provide navigation logic.
    """
    height, width = segmentation_mask.shape

    # Define Regions of Interest (Vertical Sectors)
    col_split = width // 3
    sectors = {
        'Left': segmentation_mask[:, :col_split],
        'Center': segmentation_mask[:, col_split:2*col_split],
        'Right': segmentation_mask[:, 2*col_split:]
    }

    # Define Target Classes (Cityscapes IDs)
    SIDEWALK_ID = 8
    ROAD_ID = 7
    HAZARD_IDS = [11, 12, 13, 17, 18, 21, 22]

    navigation_report = {}

    for name, region in sectors.items():
        total_pixels = region.size

        # Calculate densities
        sidewalk_pixels = np.sum(region == SIDEWALK_ID)
        hazard_pixels = np.sum(np.isin(region, HAZARD_IDS))

        sidewalk_density = (sidewalk_pixels / total_pixels) * 100
        hazard_density = (hazard_pixels / total_pixels) * 100

        navigation_report[name] = {
            'sidewalk_density': sidewalk_density,
            'hazard_detected': hazard_density > 5.0,
            'hazard_density': hazard_density
        }

    return navigation_report


def get_navigation_instruction(report):
    """Translates density report into a simple text/audio instruction."""
    center = report['Center']
    left = report['Left']
    right = report['Right']

    if center['sidewalk_density'] > 40 and not center['hazard_detected']:
        return "Path clear. Continue straight."
    elif left['sidewalk_density'] > right['sidewalk_density'] and not left['hazard_detected']:
        return "Obstacle ahead. Veer slight left."
    elif right['sidewalk_density'] > left['sidewalk_density'] and not right['hazard_detected']:
        return "Obstacle ahead. Veer slight right."
    else:
        return "Caution: Path obstructed. Slow down."


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


def process_frame_with_navigation(frame, model, palette):
    """Process a frame and return segmentation with navigation analysis."""
    # Perform inference
    result = inference_segmentor(model, frame)
    seg_map = result[0]

    # Analyze navigation
    navigation_report = analyze_navigation_path(seg_map)
    instruction = get_navigation_instruction(navigation_report)

    # Create colored segmentation map
    height, width = frame.shape[:2]
    color_seg = np.zeros((height, width, 3), dtype=np.uint8)
    for idx, color in enumerate(palette):
        if idx < len(palette):
            color_seg[seg_map == idx, :] = color

    # Blend with original image (30% original, 70% segmentation for better visibility)
    blend = (frame * 0.3 + color_seg * 0.7).astype(np.uint8)

    return blend, navigation_report, instruction


# Flask web app
from flask import Flask, render_template, Response, jsonify, request

# Get directory of current script
current_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(current_dir, 'templates')

app = Flask(__name__, template_folder=template_dir)

# Global state
model = None
palette = None
inference_times = []


def init_model():
    """Initialize the SeaFormer model."""
    global model, palette

    print("Initializing SeaFormer model...")
    try:
        config_path = '/mnt/b/local_projects/SeaFormer/seaformer-seg/local_configs/seaformer/seaformer_small_1024x1024_160k_1x8city.py'
        checkpoint_path = '/home/manth/Downloads/SeaFormer_S_cityf_76.4.pth'

        model = init_segmentor(config_path, checkpoint_path, device='cpu')
        model.eval()
        palette = get_palette('cityscapes')
        print("✅ Model initialized successfully!")
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
        annotated_frame, navigation_report, instruction = process_frame_with_navigation(frame, model, palette)
        inference_time = time.time() - start_time
        inference_times.append(inference_time)

        # Generate TTS audio for instruction
        audio_base64 = text_to_speech(instruction)

        # Encode annotated frame back to base64
        _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        annotated_base64 = base64.b64encode(buffer).decode('utf-8')

        # Calculate average metrics
        avg_inference_time = np.mean(inference_times[-30:]) if len(inference_times) >= 30 else np.mean(inference_times)

        return jsonify({
            'annotated_image': f'data:image/jpeg;base64,{annotated_base64}',
            'left_sidewalk': f"{navigation_report['Left']['sidewalk_density']:.1f}%",
            'center_sidewalk': f"{navigation_report['Center']['sidewalk_density']:.1f}%",
            'right_sidewalk': f"{navigation_report['Right']['sidewalk_density']:.1f}%",
            'hazard_status': 'hazard' if any(s['hazard_detected'] for s in navigation_report.values()) else 'clear',
            'instruction': instruction,
            'audio': f'data:audio/mp3;base64,{audio_base64}' if audio_base64 else None,
            'inference_time_ms': f"{inference_time * 1000:.0f}",
            'avg_inference_time_ms': f"{avg_inference_time * 1000:.0f}",
            'fps': f"{1.0 / avg_inference_time:.1f}" if avg_inference_time > 0 else "0.0"
        })

    except Exception as e:
        print(f"Inference error: {e}")
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
        print(f"\n🌐 SeaFormer Navigation Server")
        print(f"📍 http://{args.host}:{args.port}")
        print("\n📱 Features:")
        print("   • Client-side camera capture")
        print("   • Real-time semantic segmentation")
        print("   • Navigation assistance")
        print("   • Server-side TTS (no browser API)")
        print("   • OpenRouter AI queries")
        print(f"   • TTS Available: {'✅' if TTS_AVAILABLE else '❌'}")
        print(f"   • OpenRouter Available: {'✅' if OPENROUTER_API_KEY else '❌ (Set OPENROUTER_API_KEY)'}")
        print("\nPress Ctrl+C to stop\n")
        app.run(host=args.host, port=args.port, debug=False)
    else:
        print("\n❌ Failed to initialize model. Exiting.")
        sys.exit(1)
