"""Smart Waste Vision — Enhanced Gradio web demo (Railway-ready).

Tabs:
  1. Classify: upload image -> CNN + Grad-CAM + AI advice
  2. Live Webcam: capture-based detection with auto-detect
  3. Batch Processing: classify multiple images, export CSV
  4. AI Chat: ask the EcoBot about waste & recycling
  5. History & Stats: session dashboard with charts
  6. Waste Guide: searchable educational reference
"""

import datetime
import os
import sys
import tempfile

import cv2
import gradio as gr
import numpy as np
import pandas as pd
import tensorflow as tf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import IMG_SIZE, preprocess_single_image
from src.gradcam import find_last_conv_layer, get_gradcam_heatmap, overlay_gradcam
import src.models  # registers BackbonePreprocess

from waste_knowledge import WASTE_KNOWLEDGE, BIN_DISPLAY, SAFETY_DISPLAY
from ai_advisor import get_ai_advice, chat_with_advisor

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
/* Overall app */
.gradio-container { max-width: 1200px !important; }

/* Header */
.app-header {
    text-align: center;
    padding: 20px 0 10px 0;
}
.app-header h1 {
    font-size: 2.2em !important;
    background: linear-gradient(135deg, #2E7D32, #1565C0);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 4px !important;
}
.app-header p { color: #666; font-size: 0.95em; }

/* Result card */
.result-card {
    border-radius: 12px;
    padding: 20px;
    margin: 8px 0;
    border: 1px solid #e0e0e0;
    background: #fafafa;
}
.result-card h3 { margin-top: 0 !important; }

/* Bin badge */
.bin-badge {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 20px;
    color: white;
    font-weight: 600;
    font-size: 0.95em;
    margin: 4px 0 12px 0;
}
.bin-recyclable { background: #1976D2; }
.bin-compostable { background: #388E3C; }
.bin-hazardous { background: #D32F2F; }
.bin-landfill { background: #616161; }

/* Confidence bar */
.conf-bar-wrap {
    background: #e0e0e0;
    border-radius: 8px;
    height: 10px;
    margin: 6px 0 12px 0;
    overflow: hidden;
}
.conf-bar {
    height: 100%;
    border-radius: 8px;
    transition: width 0.5s;
}
.conf-high { background: linear-gradient(90deg, #43A047, #2E7D32); }
.conf-mid { background: linear-gradient(90deg, #FFA726, #F57C00); }
.conf-low { background: linear-gradient(90deg, #EF5350, #C62828); }

/* Warning box */
.warning-box {
    padding: 10px 14px;
    border-radius: 8px;
    margin: 8px 0;
    font-size: 0.9em;
}
.warning-yellow { background: #FFF8E1; border-left: 4px solid #FFA726; }
.warning-red { background: #FFEBEE; border-left: 4px solid #EF5350; }

/* Safety alert */
.safety-alert {
    padding: 10px 14px;
    border-radius: 8px;
    margin: 8px 0;
    font-size: 0.9em;
}
.safety-danger { background: #FFEBEE; border-left: 4px solid #D32F2F; color: #B71C1C; }
.safety-warning { background: #FFF8E1; border-left: 4px solid #F9A825; color: #E65100; }
.safety-info { background: #E3F2FD; border-left: 4px solid #1976D2; color: #0D47A1; }

/* AI advice panel */
.ai-advice {
    background: linear-gradient(135deg, #E8F5E9, #E3F2FD);
    border-radius: 12px;
    padding: 16px;
    margin: 10px 0;
    border: 1px solid #C8E6C9;
}
.ai-advice h4 { margin-top: 0 !important; color: #2E7D32; }

/* Steps list */
.steps-list { padding-left: 0; list-style: none; }
.steps-list li {
    padding: 6px 0 6px 28px;
    position: relative;
    border-bottom: 1px solid #f0f0f0;
}
.steps-list li::before {
    content: attr(data-step);
    position: absolute;
    left: 0;
    width: 22px; height: 22px;
    background: #1976D2;
    color: white;
    border-radius: 50%;
    text-align: center;
    font-size: 0.75em;
    line-height: 22px;
    font-weight: 600;
}

/* Chatbot */
.chatbot-container .message { border-radius: 12px !important; }

/* Tab styling */
.tab-nav button { font-weight: 600 !important; }
"""


# ---------------------------------------------------------------------------
# Constants & Model Loading
# ---------------------------------------------------------------------------
_app_models = os.path.join(os.path.dirname(__file__), "models")
_output_models = os.path.join(os.path.dirname(__file__), "..", "outputs", "models")
MODELS_DIR = _app_models if os.path.isdir(_app_models) else _output_models

CLASS_NAMES = [
    "battery", "biological", "cardboard", "clothes", "glass",
    "metal", "paper", "plastic", "shoes", "trash",
]

MODEL_INFO = {
    "efficientnetb3": {"label": "EfficientNet-B3 (93.96%)", "accuracy": 0.9396},
    "resnet50":       {"label": "ResNet50 (82.93%)",         "accuracy": 0.8293},
    "mobilenetv2":    {"label": "MobileNetV2 (81.73%)",      "accuracy": 0.8173},
    "vgg16":          {"label": "VGG16 (80.33%)",            "accuracy": 0.8033},
}

CLASS_COLORS = {
    "battery": (0, 0, 255),    "biological": (0, 180, 0),   "cardboard": (139, 90, 43),
    "clothes": (255, 0, 255),  "glass": (255, 255, 0),      "metal": (128, 128, 128),
    "paper": (200, 200, 200),  "plastic": (0, 165, 255),    "shoes": (80, 50, 20),
    "trash": (100, 100, 100),
}

print("Loading models...")
loaded_models = {}
loaded_conv_layers = {}

for name in MODEL_INFO:
    path = os.path.join(MODELS_DIR, f"{name}.keras")
    if os.path.exists(path):
        print(f"  Loading {name}...")
        loaded_models[name] = tf.keras.models.load_model(path)
        loaded_conv_layers[name] = find_last_conv_layer(loaded_models[name])
        print(f"  {name} ready")

if not loaded_models:
    raise FileNotFoundError(f"No models found in {MODELS_DIR}")

default_model = next(iter(loaded_models))
model_choices = [MODEL_INFO[n]["label"] for n in loaded_models]
print(f"{len(loaded_models)} models loaded. Default: {default_model}")


def _get_model_name(label):
    for name, info in MODEL_INFO.items():
        if info["label"] == label:
            return name
    return default_model


# ---------------------------------------------------------------------------
# HTML builders for rich result display
# ---------------------------------------------------------------------------
def build_result_html(class_name, confidence, all_confidences, model_label):
    """Build a styled HTML result card with inline styles for reliable rendering."""
    info = WASTE_KNOWLEDGE.get(class_name, {})
    disposal = info.get("disposal", {})
    impact = info.get("environmental_impact", {})
    safety = info.get("safety", {})

    status = disposal.get("status", "unknown")
    bin_info = BIN_DISPLAY.get(status, {})
    bin_emoji = bin_info.get("emoji", "")

    # Bin badge color
    bin_colors = {
        "recyclable": "#1976D2", "compostable": "#388E3C",
        "hazardous": "#D32F2F", "landfill": "#616161",
    }
    bin_color = bin_colors.get(status, "#757575")

    # Confidence bar color
    conf_pct = confidence * 100
    if confidence >= 0.70:
        conf_color = "linear-gradient(90deg, #43A047, #2E7D32)"
    elif confidence >= 0.40:
        conf_color = "linear-gradient(90deg, #FFA726, #F57C00)"
    else:
        conf_color = "linear-gradient(90deg, #EF5350, #C62828)"

    # Warning
    warning_html = ""
    if confidence < 0.70:
        sorted_c = sorted(all_confidences.items(), key=lambda x: x[1], reverse=True)
        alts = ", ".join(f"<b>{n}</b> ({v:.0%})" for n, v in sorted_c[1:3])
        if confidence < 0.40:
            warning_html = (
                f'<div style="padding:10px 14px;border-radius:8px;margin:8px 0;'
                f'background:#FFEBEE;border-left:4px solid #EF5350;font-size:0.9em;">'
                f'<b>Very uncertain ({confidence:.0%})</b> &mdash; Consider: {alts}</div>'
            )
        else:
            warning_html = (
                f'<div style="padding:10px 14px;border-radius:8px;margin:8px 0;'
                f'background:#FFF8E1;border-left:4px solid #FFA726;font-size:0.9em;">'
                f'<b>Low confidence ({confidence:.0%})</b> &mdash; Consider: {alts}</div>'
            )

    # Safety alert colors
    safety_level = safety.get("level", "info")
    safety_styles = {
        "danger": "background:#FFEBEE;border-left:4px solid #D32F2F;color:#B71C1C;",
        "warning": "background:#FFF8E1;border-left:4px solid #F9A825;color:#E65100;",
        "info": "background:#E3F2FD;border-left:4px solid #1976D2;color:#0D47A1;",
    }
    safety_style = safety_styles.get(safety_level, safety_styles["info"])
    safety_icon = SAFETY_DISPLAY.get(safety_level, {}).get("icon", "")
    safety_html = (
        f'<div style="padding:10px 14px;border-radius:8px;margin:8px 0;'
        f'font-size:0.9em;{safety_style}">'
        f'{safety_icon} <b>Safety:</b> {safety.get("alert", "")}</div>'
    )

    # Steps
    steps_items = ""
    for i, step in enumerate(disposal.get("steps", []), 1):
        steps_items += (
            f'<li style="padding:6px 0;border-bottom:1px solid #f0f0f0;">'
            f'<span style="display:inline-block;width:22px;height:22px;'
            f'background:#1976D2;color:white;border-radius:50%;text-align:center;'
            f'font-size:0.75em;line-height:22px;font-weight:600;margin-right:8px;">'
            f'{i}</span>{step}</li>'
        )

    # Mistakes
    mistakes_items = "".join(
        f'<li style="padding:3px 0;">{m}</li>'
        for m in disposal.get("common_mistakes", [])
    )

    html = f"""
    <div style="border-radius:12px;padding:20px;margin:8px 0;border:1px solid #e0e0e0;
                background:#fafafa;font-family:sans-serif;">
        <h3 style="margin:0 0 8px 0;font-size:1.4em;">{bin_emoji} {class_name.upper()}</h3>
        <span style="display:inline-block;padding:6px 16px;border-radius:20px;color:white;
                     font-weight:600;font-size:0.95em;margin:0 0 12px 0;
                     background:{bin_color};">
            {disposal.get('bin_label', status)}
        </span>

        <div style="margin:10px 0;">
            <small style="color:#666;">Confidence: {confidence:.1%}</small>
            <div style="background:#e0e0e0;border-radius:8px;height:10px;margin:4px 0 12px 0;
                        overflow:hidden;">
                <div style="height:100%;border-radius:8px;width:{conf_pct:.0f}%;
                            background:{conf_color};"></div>
            </div>
        </div>

        {warning_html}

        <h4 style="margin:12px 0 6px 0;color:#333;">How to Dispose</h4>
        <ol style="padding-left:0;list-style:none;margin:0;">{steps_items}</ol>

        <details style="margin:10px 0;">
            <summary style="cursor:pointer;font-weight:600;color:#D84315;padding:4px 0;">
                Common Mistakes to Avoid
            </summary>
            <ul style="margin-top:6px;padding-left:20px;">{mistakes_items}</ul>
        </details>

        <details style="margin:10px 0;" open>
            <summary style="cursor:pointer;font-weight:600;color:#1565C0;padding:4px 0;">
                Environmental Impact
            </summary>
            <ul style="margin-top:6px;padding-left:20px;">
                <li style="padding:3px 0;"><b>Decomposition:</b> {impact.get('decomposition_time', 'N/A')}</li>
                <li style="padding:3px 0;"><b>Recycling benefit:</b> {impact.get('recycling_benefit', 'N/A')}</li>
                <li style="padding:3px 0;"><b>Did you know?</b> {impact.get('fun_fact', '')}</li>
            </ul>
        </details>

        {safety_html}

        <div style="margin-top:10px;font-size:0.8em;color:#999;">Model: {model_label}</div>
    </div>
    """
    return html


def build_ai_advice_html(advice_text):
    """Wrap AI advice in a styled panel with inline styles."""
    if not advice_text:
        return ""
    import re
    # Convert **bold** to <b>bold</b>
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', advice_text)
    text = text.replace("\n- ", "\n&bull; ")
    text = text.replace("\n", "<br>")
    return f"""
    <div style="background:linear-gradient(135deg, #E8F5E9, #E3F2FD);
                border-radius:12px;padding:16px;margin:10px 0;
                border:1px solid #C8E6C9;font-family:sans-serif;">
        <h4 style="margin:0 0 8px 0;color:#2E7D32;">EcoBot AI Advice</h4>
        <div style="font-size:0.95em;line-height:1.6;color:#333;">{text}</div>
    </div>
    """


# ---------------------------------------------------------------------------
# Helper: History management
# ---------------------------------------------------------------------------
def add_to_history(history, class_name, confidence, source="upload"):
    info = WASTE_KNOWLEDGE.get(class_name, {})
    disposal_status = info.get("disposal", {}).get("status", "unknown")
    bin_label = BIN_DISPLAY.get(disposal_status, {}).get("bin", "Unknown")

    if source == "webcam" and history:
        last = history[-1]
        if last["class"] == class_name:
            try:
                last_time = datetime.datetime.strptime(last["timestamp"], "%H:%M:%S").time()
                now_time = datetime.datetime.now().time()
                last_s = last_time.hour * 3600 + last_time.minute * 60 + last_time.second
                now_s = now_time.hour * 3600 + now_time.minute * 60 + now_time.second
                if abs(now_s - last_s) < 10:
                    return history
            except ValueError:
                pass

    return history + [{
        "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
        "class": class_name,
        "confidence": round(confidence, 3),
        "disposal_type": disposal_status,
        "bin": bin_label,
    }]


def make_pie_chart(data, title):
    fig, ax = plt.subplots(figsize=(4, 4))
    if not data:
        ax.text(0.5, 0.5, "No data yet", ha="center", va="center",
                fontsize=14, color="gray")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        return fig
    disposal_colors = {
        "recyclable": "#2196F3", "compostable": "#4CAF50",
        "hazardous": "#F44336", "landfill": "#616161",
    }
    colors = None
    if title == "By Disposal Type":
        colors = [disposal_colors.get(k, "#9E9E9E") for k in data.keys()]
    ax.pie(data.values(), labels=data.keys(), autopct="%1.0f%%",
           startangle=90, colors=colors)
    ax.set_title(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig


def render_history_dashboard(history):
    if not history:
        empty_fig = make_pie_chart({}, "")
        return (
            "*No classifications yet.*",
            empty_fig, empty_fig, "",
            pd.DataFrame(columns=["Time", "Class", "Confidence", "Disposal", "Bin"]),
        )
    df = pd.DataFrame(history)
    total = len(df)
    cat_counts = df["class"].value_counts().to_dict()
    disp_counts = df["disposal_type"].value_counts().to_dict()
    landfill = disp_counts.get("landfill", 0)
    diverted = total - landfill

    stats_md = (
        f"### Session Statistics\n\n"
        f"- **Total classified:** {total}\n"
        f"- **Diverted from landfill:** {diverted} ({diverted / total:.0%})\n"
        f"- **Most common:** {max(cat_counts, key=cat_counts.get)}\n"
    )
    env_md = (
        f"### Environmental Impact\n\n"
        f"- **{disp_counts.get('recyclable', 0)}** recycled\n"
        f"- **{disp_counts.get('compostable', 0)}** composted\n"
        f"- **{disp_counts.get('hazardous', 0)}** hazardous identified\n"
        f"- **{diverted}** total diverted from landfill\n"
    )
    display_df = pd.DataFrame({
        "Time": df["timestamp"],
        "Class": df["class"].str.capitalize(),
        "Confidence": df["confidence"].apply(lambda x: f"{x:.1%}"),
        "Disposal": df["disposal_type"].str.capitalize(),
        "Bin": df["bin"],
    })
    return (
        stats_md,
        make_pie_chart(cat_counts, "By Category"),
        make_pie_chart(disp_counts, "By Disposal Type"),
        env_md,
        display_df,
    )


# ---------------------------------------------------------------------------
# Helper: Batch processing
# ---------------------------------------------------------------------------
def batch_classify(files, model_label, history):
    empty_df = pd.DataFrame(columns=["#", "Filename", "Class", "Confidence", "Bin", "Warning"])
    if not files:
        return empty_df, "*No files uploaded.*", None, history

    mname = _get_model_name(model_label)
    mdl = loaded_models[mname]

    images, filenames = [], []
    for f in files:
        fp = f if isinstance(f, str) else f.name
        img = cv2.imread(fp)
        if img is None:
            continue
        images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        filenames.append(os.path.basename(fp))

    if not images:
        return empty_df, "*No valid images found.*", None, history

    batch = np.concatenate([preprocess_single_image(im) for im in images], axis=0)
    all_preds = mdl.predict(batch, verbose=0)

    rows = []
    for i, (fname, preds) in enumerate(zip(filenames, all_preds)):
        idx = int(np.argmax(preds))
        cls = CLASS_NAMES[idx]
        conf = float(preds[idx])
        info = WASTE_KNOWLEDGE.get(cls, {})
        status = info.get("disposal", {}).get("status", "unknown")
        blab = BIN_DISPLAY.get(status, {}).get("bin", "Unknown")
        warn = "Very uncertain" if conf < 0.40 else ("Low confidence" if conf < 0.70 else "")
        rows.append({"#": i + 1, "Filename": fname, "Class": cls.capitalize(),
                      "Confidence": f"{conf:.1%}", "Bin": blab, "Warning": warn})
        history = add_to_history(history, cls, conf, source="batch")

    rdf = pd.DataFrame(rows)
    disp_c = {}
    for r in rows:
        disp_c[r["Bin"]] = disp_c.get(r["Bin"], 0) + 1
    summary = f"### {len(rows)} items classified\n\n"
    summary += " | ".join(f"**{v}** {k}" for k, v in disp_c.items())
    wc = sum(1 for r in rows if r["Warning"])
    if wc:
        summary += f"\n\n*{wc} item(s) low confidence*"
    summary += f"\n\n*Model: {model_label}*"
    return rdf, summary, rdf, history


def export_batch_csv(state):
    if state is None or (isinstance(state, pd.DataFrame) and state.empty):
        return None
    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w", prefix="waste_")
    if isinstance(state, pd.DataFrame):
        state.to_csv(tmp.name, index=False)
    tmp.close()
    return tmp.name


# ---------------------------------------------------------------------------
# Helper: Waste guide
# ---------------------------------------------------------------------------
def render_waste_guide(search_query=""):
    query = (search_query or "").strip().lower()
    sections = []
    for cname in sorted(WASTE_KNOWLEDGE.keys()):
        info = WASTE_KNOWLEDGE[cname]
        if query:
            searchable = " ".join([
                cname, info["education"]["description"],
                " ".join(info["education"]["example_items"]),
                info["disposal"]["status"], info["disposal"]["bin_label"],
            ]).lower()
            if query not in searchable:
                continue

        d = info["disposal"]
        im = info["environmental_impact"]
        ed = info["education"]
        sf = info["safety"]
        bi = BIN_DISPLAY.get(d["status"], {})
        si = SAFETY_DISPLAY.get(sf["level"], {})

        steps = "\n".join(f"{i}. {s}" for i, s in enumerate(d["steps"], 1))
        examples = ", ".join(ed["example_items"])

        sections.append(
            f"---\n"
            f"### {bi.get('emoji', '')} {cname.capitalize()} — {bi.get('bin', '')} ({d['status'].capitalize()})\n\n"
            f"*{ed['description']}*\n\n"
            f"**Examples:** {examples}\n\n"
            f"**How to dispose:**\n{steps}\n\n"
            f"**Environmental facts:**\n"
            f"- Decomposition: {im['decomposition_time']}\n"
            f"- {im['recycling_benefit']}\n"
            f"- {im['fun_fact']}\n\n"
            f"> {si.get('icon', '')} **Safety:** {sf['alert']}\n"
        )

    return "\n\n".join(sections) if sections else f'*No match for "{search_query}".*'


# ---------------------------------------------------------------------------
# Core: Upload prediction
# ---------------------------------------------------------------------------
def predict(image, model_label, history):
    if image is None:
        return {}, None, "<p>Please upload an image.</p>", "", history

    mname = _get_model_name(model_label)
    mdl = loaded_models[mname]
    conv_layer = loaded_conv_layers[mname]

    img_pre = preprocess_single_image(image)
    preds = mdl.predict(img_pre, verbose=0)[0]
    confidences = {CLASS_NAMES[i]: float(preds[i]) for i in range(len(CLASS_NAMES))}

    heatmap, pred_idx = get_gradcam_heatmap(mdl, img_pre, conv_layer)
    disp_img = tf.image.resize(image, [IMG_SIZE, IMG_SIZE]).numpy()
    disp_img = disp_img.astype(np.float32) / 255.0 if disp_img.max() > 1 else disp_img
    gcam_img = overlay_gradcam(disp_img, heatmap)

    pred_class = CLASS_NAMES[pred_idx]
    conf = float(preds[pred_idx])

    # Build HTML result card
    result_html = build_result_html(pred_class, conf, confidences, model_label)

    # Get AI advice (non-blocking attempt)
    disposal_info = WASTE_KNOWLEDGE.get(pred_class, {}).get("disposal", {})
    ai_text = get_ai_advice(pred_class, conf, disposal_info)
    ai_html = build_ai_advice_html(ai_text)

    history = add_to_history(history, pred_class, conf, source="upload")

    return confidences, gcam_img, result_html, ai_html, history


# ---------------------------------------------------------------------------
# Core: Live webcam
# ---------------------------------------------------------------------------
def predict_live(frame, model_label):
    if frame is None:
        return None

    mname = _get_model_name(model_label)
    mdl = loaded_models[mname]

    img_pre = preprocess_single_image(frame)
    preds = mdl.predict(img_pre, verbose=0)[0]
    pred_idx = int(np.argmax(preds))
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(preds[pred_idx])

    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # Banner color based on confidence
    if confidence < 0.40:
        banner_color = (0, 0, 180)
    elif confidence < 0.70:
        banner_color = (0, 180, 180)
    else:
        banner_color = (0, 0, 0)

    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), banner_color, -1)
    annotated = cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0)

    color = CLASS_COLORS.get(pred_class, (255, 255, 255))
    color_rgb = (color[2], color[1], color[0])
    cv2.putText(annotated, f"{pred_class.upper()} ({confidence:.1%})", (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_rgb, 2, cv2.LINE_AA)

    # Tip or warning
    if confidence < 0.40:
        tip = "UNCERTAIN - inspect manually"
    elif confidence < 0.70:
        tip = "Low confidence - check manually"
    else:
        tip = WASTE_KNOWLEDGE.get(pred_class, {}).get("disposal", {}).get("bin_label", "")
    cv2.putText(annotated, tip[:60], (15, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # Top-3 bars
    for i, idx in enumerate(np.argsort(preds)[::-1][:3]):
        y = 100 + i * 30
        c = CLASS_COLORS.get(CLASS_NAMES[idx], (255, 255, 255))
        cv2.rectangle(annotated, (w - 200, y),
                       (w - 200 + int(preds[idx] * 150), y + 20),
                       (c[2], c[1], c[0]), -1)
        cv2.putText(annotated, f"{CLASS_NAMES[idx]} {preds[idx]:.0%}",
                     (w - 195, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                     (255, 255, 255), 1, cv2.LINE_AA)
    return annotated


def detect_and_annotate(image, model_label, history):
    if image is None:
        return None, "*Capture a photo first.*", history
    annotated = predict_live(image, model_label)

    mname = _get_model_name(model_label)
    preds = loaded_models[mname].predict(preprocess_single_image(image), verbose=0)[0]
    pred_idx = int(np.argmax(preds))
    pred_class = CLASS_NAMES[pred_idx]
    conf = float(preds[pred_idx])
    confidences = {CLASS_NAMES[i]: float(preds[i]) for i in range(len(CLASS_NAMES))}

    winfo = WASTE_KNOWLEDGE.get(pred_class, {})
    disposal = winfo.get("disposal", {})
    bi = BIN_DISPLAY.get(disposal.get("status", ""), {})
    safety = winfo.get("safety", {})

    # Build rich text
    warning = ""
    if conf < 0.40:
        sorted_c = sorted(confidences.items(), key=lambda x: x[1], reverse=True)
        alts = ", ".join(f"**{n}** ({v:.0%})" for n, v in sorted_c[1:3])
        warning = f"> **Very uncertain** — Consider: {alts}\n\n"
    elif conf < 0.70:
        sorted_c = sorted(confidences.items(), key=lambda x: x[1], reverse=True)
        alts = ", ".join(f"**{n}** ({v:.0%})" for n, v in sorted_c[1:3])
        warning = f"> **Low confidence** — Consider: {alts}\n\n"

    safety_text = ""
    if safety.get("level") in ("danger", "warning"):
        si = SAFETY_DISPLAY.get(safety["level"], {})
        safety_text = f"\n\n> {si.get('icon', '')} {safety.get('alert', '')}"

    text = (
        f"### {bi.get('emoji', '')} {pred_class.upper()} — {conf:.1%}\n"
        f"**{disposal.get('bin_label', '')}** | {bi.get('bin', '')}\n\n"
        f"{warning}"
        f"**Dispose:** {disposal.get('steps', [''])[0]}"
        f"{safety_text}\n\n"
        f"*{model_label}*"
    )

    history = add_to_history(history, pred_class, conf, source="webcam")
    return annotated, text, history


# ---------------------------------------------------------------------------
# AI Chat handler
# ---------------------------------------------------------------------------
def ai_chat_respond(message, chat_history, waste_ctx):
    if not message.strip():
        return "", chat_history
    bot_reply = chat_with_advisor(message, chat_history, waste_ctx)
    chat_history = chat_history + [[message, bot_reply]]
    return "", chat_history


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="Smart Waste Vision") as demo:
    history_state = gr.State(value=[])
    batch_results_state = gr.State(value=None)
    waste_context_state = gr.State(value="")

    # Header
    gr.HTML(
        '<div style="text-align:center;padding:20px 0 10px 0;">'
        '<h1 style="font-size:2.2em;background:linear-gradient(135deg,#2E7D32,#1565C0);'
        '-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:4px;">'
        'Smart Waste Vision</h1>'
        f'<p style="color:#666;font-size:0.95em;">'
        f'AI-powered waste classification &bull; {len(loaded_models)} models &bull; '
        f'Gemini AI advisor &bull; 10 categories</p>'
        '</div>'
    )

    with gr.Tabs():
        # ========== Tab 1: Classify ==========
        with gr.TabItem("Classify"):
            with gr.Row():
                with gr.Column(scale=1):
                    upload_model = gr.Dropdown(
                        choices=model_choices, value=model_choices[0],
                        label="Model",
                    )
                    input_image = gr.Image(
                        label="Upload Waste Image", type="numpy",
                        sources=["upload", "webcam"],
                    )
                    submit_btn = gr.Button("Classify", variant="primary", size="lg")

                with gr.Column(scale=1):
                    label_output = gr.Label(label="Confidence", num_top_classes=5)
                    gradcam_output = gr.Image(label="Grad-CAM — What the model sees")

            # Result sections below the image row
            with gr.Row():
                with gr.Column(scale=1):
                    guidance_output = gr.HTML(label="Disposal Guidance")
                with gr.Column(scale=1):
                    ai_advice_output = gr.HTML(label="AI Advice")

            submit_btn.click(
                fn=predict,
                inputs=[input_image, upload_model, history_state],
                outputs=[label_output, gradcam_output, guidance_output,
                         ai_advice_output, history_state],
            )

        # ========== Tab 2: Live Webcam ==========
        with gr.TabItem("Live Webcam"):
            gr.Markdown(
                "### How to use:\n"
                "1. Click the webcam area to open your camera\n"
                "2. Point at a waste item and **click the camera button**\n"
                "3. Detection runs automatically after each capture\n"
                "4. Check **Auto-detect** for continuous detection"
            )
            with gr.Row():
                with gr.Column(scale=1):
                    live_model = gr.Dropdown(
                        choices=model_choices, value=model_choices[0],
                        label="Model",
                    )
                    webcam_input = gr.Image(
                        label="Webcam", type="numpy", sources=["webcam"],
                    )
                    with gr.Row():
                        detect_btn = gr.Button("Detect", variant="primary")
                        auto_detect = gr.Checkbox(label="Auto-detect every 2s", value=False)

                with gr.Column(scale=1):
                    webcam_output = gr.Image(label="Detection Result")
                    webcam_label = gr.Markdown("*Waiting for detection...*")

            detect_btn.click(
                fn=detect_and_annotate,
                inputs=[webcam_input, live_model, history_state],
                outputs=[webcam_output, webcam_label, history_state],
            )
            webcam_input.change(
                fn=detect_and_annotate,
                inputs=[webcam_input, live_model, history_state],
                outputs=[webcam_output, webcam_label, history_state],
            )
            timer = gr.Timer(value=2, active=False)
            auto_detect.change(
                fn=lambda x: gr.Timer(active=x),
                inputs=auto_detect, outputs=timer,
            )
            timer.tick(
                fn=detect_and_annotate,
                inputs=[webcam_input, live_model, history_state],
                outputs=[webcam_output, webcam_label, history_state],
            )

        # ========== Tab 3: Batch Processing ==========
        with gr.TabItem("Batch"):
            gr.Markdown("### Bulk Classification\nUpload multiple images for batch classification.")
            with gr.Row():
                with gr.Column(scale=1):
                    batch_model = gr.Dropdown(
                        choices=model_choices, value=model_choices[0],
                        label="Model",
                    )
                    batch_input = gr.File(
                        file_count="multiple", file_types=["image"],
                        label="Upload Images",
                    )
                    batch_btn = gr.Button("Classify All", variant="primary")
                with gr.Column(scale=2):
                    batch_results = gr.Dataframe(
                        headers=["#", "Filename", "Class", "Confidence", "Bin", "Warning"],
                        datatype=["number", "str", "str", "str", "str", "str"],
                        label="Results", interactive=False,
                    )
                    batch_summary = gr.Markdown()
                    batch_download = gr.DownloadButton("Export CSV", variant="secondary")

            batch_btn.click(
                fn=batch_classify,
                inputs=[batch_input, batch_model, history_state],
                outputs=[batch_results, batch_summary, batch_results_state, history_state],
            )
            batch_download.click(
                fn=export_batch_csv,
                inputs=[batch_results_state], outputs=[batch_download],
            )

        # ========== Tab 4: AI Chat ==========
        with gr.TabItem("AI Chat"):
            gr.Markdown(
                "### EcoBot — Your Waste & Recycling Advisor\n"
                "Ask anything about waste disposal, recycling rules, "
                "environmental impact, or creative reuse ideas."
            )
            chatbot = gr.Chatbot(
                label="EcoBot",
                height=400,
                placeholder="Ask me about waste sorting, recycling, or sustainability...",
                elem_classes=["chatbot-container"],
            )
            with gr.Row():
                chat_input = gr.Textbox(
                    placeholder="e.g. Can I recycle pizza boxes? How do I dispose of paint?",
                    label="Your question",
                    scale=4,
                )
                chat_send = gr.Button("Send", variant="primary", scale=1)

            chat_send.click(
                fn=ai_chat_respond,
                inputs=[chat_input, chatbot, waste_context_state],
                outputs=[chat_input, chatbot],
            )
            chat_input.submit(
                fn=ai_chat_respond,
                inputs=[chat_input, chatbot, waste_context_state],
                outputs=[chat_input, chatbot],
            )

        # ========== Tab 5: History & Stats ==========
        with gr.TabItem("History"):
            gr.Markdown("### Classification Dashboard")
            with gr.Row():
                with gr.Column():
                    stats_summary = gr.Markdown("*No classifications yet.*")
                    category_plot = gr.Plot(label="By Category")
                with gr.Column():
                    env_impact = gr.Markdown()
                    disposal_plot = gr.Plot(label="By Disposal Type")
            history_table = gr.Dataframe(
                headers=["Time", "Class", "Confidence", "Disposal", "Bin"],
                label="Log", interactive=False,
            )
            with gr.Row():
                refresh_btn = gr.Button("Refresh", variant="primary")
                clear_btn = gr.Button("Clear", variant="stop")

            refresh_btn.click(
                fn=render_history_dashboard, inputs=[history_state],
                outputs=[stats_summary, category_plot, disposal_plot, env_impact, history_table],
            )

            def clear_history():
                ef = make_pie_chart({}, "")
                return ([], "*Cleared.*", ef, ef, "",
                        pd.DataFrame(columns=["Time", "Class", "Confidence", "Disposal", "Bin"]))

            clear_btn.click(
                fn=clear_history,
                outputs=[history_state, stats_summary, category_plot,
                         disposal_plot, env_impact, history_table],
            )

        # ========== Tab 6: Waste Guide ==========
        with gr.TabItem("Guide"):
            gr.Markdown(
                "### Waste Classification Reference\n"
                "Search for any waste category, item, or disposal type."
            )
            guide_search = gr.Textbox(
                label="Search",
                placeholder="e.g. plastic, recycle, hazardous...",
            )
            guide_content = gr.Markdown(value=render_waste_guide(""))
            guide_search.change(
                fn=render_waste_guide,
                inputs=[guide_search], outputs=[guide_content],
            )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        css=CUSTOM_CSS,
    )
