import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import pandas as pd

# ==============================================================================
# 1. PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Resistor Lead Inspection",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS for UI styling
st.markdown("""
    <style>
        .metric-card { background-color: #f0f2f6; padding: 15px; border-radius: 10px; text-align: center; }
        .pass-badge { color: #ffffff; background-color: #28a745; padding: 10px; border-radius: 5px; font-weight: bold; text-align: center; font-size: 20px;}
        .fail-badge { color: #ffffff; background-color: #dc3545; padding: 10px; border-radius: 5px; font-weight: bold; text-align: center; font-size: 20px;}
        .warning-text { color: #ffc107; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. MODEL LOADING (CACHED TO PREVENT RELOADING)
# ==============================================================================
@st.cache_resource
def load_yolo_model(model_path):
    try:
        return YOLO(model_path)
    except Exception as e:
        return None

MODEL_FILENAME = 'best_nano_seed_1301.pt' 
model = load_yolo_model(MODEL_FILENAME)

# ==============================================================================
# 3. SIDEBAR CONFIGURATION
# ==============================================================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=100)
    st.title("⚙️ Configuration")
    
    st.info("💡 Tip: If components are not detected, lower the 'Confidence' threshold (0.1 - 0.15).")
    
    # Default lowered to 0.15 for better sensitivity with webcams
    conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.15, 0.05) 
    iou_threshold = st.slider("IoU Threshold (Overlap)", 0.0, 1.0, 0.45, 0.05)
    
    st.markdown("---")
    st.write("Display Settings:")
    show_labels = st.checkbox("Show Defect Names", value=True)
    show_conf = st.checkbox("Show Confidence %", value=True)

# ==============================================================================
# 4. CORE PROCESSING FUNCTIONS
# ==============================================================================
def process_prediction(model, image, conf, iou):
    """Run AI inference with high resolution settings"""
    results = model.predict(
        source=image,
        conf=conf,
        iou=iou,
        imgsz=1280,    
        max_det=1000,   # Allow up to 1000 detections
        verbose=False
    )
    return results[0]

def crop_defects(original_image_pil, boxes, names):
    """Crop defect areas for detailed inspection"""
    crops = []
    # Convert PIL to Numpy for cropping
    img_np = np.array(original_image_pil)
    
    for box in boxes:
        cls_id = int(box.cls[0])
        cls_name = names[cls_id]
        
        # Logic: Crop if class name contains 'bad' or 'ng'
        if "bad" in cls_name.lower() or "ng" in cls_name.lower():
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Expand crop area by 30px for better context
            h, w, _ = img_np.shape
            x1 = max(0, x1 - 30); y1 = max(0, y1 - 30)
            x2 = min(w, x2 + 30); y2 = min(h, y2 + 30)
            
            crop_img = img_np[y1:y2, x1:x2]
            conf_score = float(box.conf[0])
            crops.append((crop_img, cls_name, conf_score))
            
    return crops

# ==============================================================================
# 5. MAIN USER INTERFACE (UI)
# ==============================================================================
st.title("🛡️ Resistor Inspection")
st.markdown("Visual inspection system for resistor leads using **Fine-tuned YOLOv8n-seg**.")

if model is None:
    st.error(f"❌ Model file `{MODEL_FILENAME}` not found. Please upload it to the same directory as this script.")
    st.stop()

# Tabs for input sources
tab1, tab2 = st.tabs(["📁 Upload Image (High Quality)", "📹 Webcam (Quick Test)"])

image_source = None
source_type = ""

with tab1:
    uploaded_file = st.file_uploader("Upload component image (JPG, PNG)", type=['jpg', 'png', 'jpeg'])
    if uploaded_file:
        image_source = Image.open(uploaded_file)
        source_type = "upload"

with tab2:
    st.warning("⚠️ Note: Webcam images may be blurry. Keep the component close (5-10cm) and ensure good lighting.")
    camera_shot = st.camera_input("Capture live photo")
    if camera_shot:
        image_source = Image.open(camera_shot)
        source_type = "webcam"

# ==============================================================================
# 6. EXECUTION & RESULTS DISPLAY
# ==============================================================================
if image_source is not None:
    st.markdown("---")
    
    # Layout columns
    col_orig, col_res = st.columns(2)
    
    with col_orig:
        st.subheader("1. Original Image")
        st.image(image_source, use_column_width=True)

    with col_res:
        st.subheader("2. Analysis")
        with st.spinner('Inspecting...'):
            # Run prediction
            result = process_prediction(model, image_source, conf_threshold, iou_threshold)
            
            # Plot results
            res_plot = result.plot(labels=show_labels, conf=show_conf, line_width=2)
            # Convert BGR (OpenCV) -> RGB (Streamlit/PIL)
            res_plot_rgb = cv2.cvtColor(res_plot, cv2.COLOR_BGR2RGB)
            
            st.image(res_plot_rgb, use_column_width=True)

    # --- ASSESSMENT LOGIC (PASS/FAIL) ---
    boxes = result.boxes
    names = result.names
    
    total_detections = len(boxes)
    defect_count = 0
    defect_list = []
    
    for box in boxes:
        c_name = names[int(box.cls[0])]
        if "bad" in c_name.lower():
            defect_count += 1
            defect_list.append(c_name)

    # Display Result Banner
    st.markdown("### 📝 Inspection Conclusion")
    if total_detections == 0:
        st.warning("⚠️ Cannot detect any component leads! (Check lighting or camera focus)")
    elif defect_count > 0:
        st.markdown(f'<div class="fail-badge">❌ {defect_count} DEFECTS DETECTED (NG)</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="pass-badge">✅ PRODUCT PASSED (OK)</div>', unsafe_allow_html=True)

    # --- DEFECT ZOOM FEATURE (AUTO-CROP) ---
    if defect_count > 0:
        st.markdown("---")
        st.subheader("🔍 Defect Details (Zoomed)")
        
        crops = crop_defects(image_source, boxes, names)
        
        # Display in Grid
        cols = st.columns(4)
        for idx, (crop, name, conf) in enumerate(crops):
            with cols[idx % 4]:
                st.image(crop, caption=f"{name} ({conf:.2f})", width=150)
                st.error(f"Defect #{idx+1}")

    # --- STATISTICS TABLE ---
    if total_detections > 0:
        with st.expander("📊 View Detailed Statistics"):
            # Count occurrences of each class
            from collections import Counter
            all_labels = [names[int(b.cls[0])] for b in boxes]
            counts = Counter(all_labels)
            
            df = pd.DataFrame.from_dict(counts, orient='index', columns=['Quantity'])
            st.dataframe(df)