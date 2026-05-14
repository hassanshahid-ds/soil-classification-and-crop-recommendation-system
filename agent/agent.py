import os
import json
from groq import Groq
from dotenv import load_dotenv
from agent.tools import predict_soil, recommend_crop

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ─── Sirf 2 Tools — No Hardcoded Compatibility ────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "predict_soil",
            "description": "Analyzes soil image using MobileNetV2 model. Returns soil type and confidence.",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "Full path to the uploaded soil image"
                    }
                },
                "required": ["image_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "recommend_crop",
            "description": "Uses crop recommendation model with NPK and environmental parameters. Returns best crop and alternatives.",
            "parameters": {
                "type": "object",
                "properties": {
                    "N":           {"type": "number", "description": "Nitrogen content"},
                    "P":           {"type": "number", "description": "Phosphorus content"},
                    "K":           {"type": "number", "description": "Potassium content"},
                    "temperature": {"type": "number", "description": "Temperature in Celsius"},
                    "humidity":    {"type": "number", "description": "Humidity percentage"},
                    "ph":          {"type": "number", "description": "Soil pH value"},
                    "rainfall":    {"type": "number", "description": "Rainfall in mm"}
                },
                "required": ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
            }
        }
    }
]

# ─── Tool Execute ──────────────────────────────────────────────
def execute_tool(tool_name: str, tool_input: dict):
    if tool_name == "predict_soil":
        return predict_soil(**tool_input)
    elif tool_name == "recommend_crop":
        return recommend_crop(**tool_input)

# ─── Main Agentic Loop ─────────────────────────────────────────
def run_agent(image_path: str, parameters: dict) -> str:

    # ── Step 1: Soil Predict ──
    print("Agent calling tool: predict_soil")
    soil_result = predict_soil(image_path)
    print(f"Tool result: {soil_result}")

    # ── Step 2: Crop Recommend ──
    print("Agent calling tool: recommend_crop")
    crop_result = recommend_crop(
        parameters['N'], parameters['P'], parameters['K'],
        parameters['temperature'], parameters['humidity'],
        parameters['ph'], parameters['rainfall']
    )
    print(f"Tool result: {crop_result}")

    # ── Step 3: Agent Reasoning — Detailed Report ──
    analysis_prompt = f"""You are an expert agricultural scientist. 
Based on the ML model results below, write a detailed agricultural analysis report.

ML Model Results:
- Detected Soil Type: {soil_result['soil_type']} (Confidence: {soil_result['confidence']}%)
- Recommended Crop: {crop_result['recommended_crop']} (Confidence: {crop_result['confidence']}%)
- Alternative Crops: {', '.join(crop_result['alternative_crops'])}

Field Parameters:
- Nitrogen (N): {parameters['N']}, Phosphorus (P): {parameters['P']}, Potassium (K): {parameters['K']}
- Temperature: {parameters['temperature']}°C, Humidity: {parameters['humidity']}%
- pH: {parameters['ph']}, Rainfall: {parameters['rainfall']}mm

Start the report with these two lines EXACTLY at the very top, before any sections:

🌱 Soil Type: {soil_result['soil_type']} — Confidence: {soil_result['confidence']}%
🌾 Recommended Crop: {crop_result['recommended_crop']} — Confidence: {crop_result['confidence']}%

Then write a detailed report with these 4 sections. Each section must have 4-5 sentences:

**Soil Analysis:**
Describe {soil_result['soil_type']} properties, texture, water retention, nutrient content, and farming characteristics.

**Crop Recommendation:**
Analyze if {crop_result['recommended_crop']} is suitable for {soil_result['soil_type']}. Explain the agricultural reasoning using soil properties and field parameters. Mention any challenges.

**Alternative Crops:**
Evaluate {', '.join(crop_result['alternative_crops'])} one by one for {soil_result['soil_type']}. Which is most suitable and why?

**Farmer Advice:**
Give 4-5 practical tips specific to farming on {soil_result['soil_type']}. Mention soil improvements, irrigation, and fertilizer advice based on the NPK values."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": analysis_prompt}],
        max_tokens=2048,
        timeout=60
    )

    return response.choices[0].message.content