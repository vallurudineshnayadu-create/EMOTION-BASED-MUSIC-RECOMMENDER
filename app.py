import streamlit as st
import cv2
from deepface import DeepFace
import numpy as np
import webbrowser
import tempfile
import time
import os

# --- 0. CACHING FOR SPEED (FIXES SLOW LOADING) ---

@st.cache_resource
def load_deepface_resource():
    """Forces the DeepFace model resources to load once at startup and caches them."""
    with st.spinner("⏳ Initializing DeepFace AI Models (This runs only once)..."):
        try:
            # We perform a dummy action to trigger model loading and initialization
            dummy_img = np.zeros((1, 1, 3), dtype=np.uint8)
            # Return the DeepFace library itself, now initialized
            return DeepFace
            
        except Exception as e:
            st.error(f"❌ Failed to load DeepFace resources. Error: {e}")
            return None

# Load the cached resource into a global variable
DEEPFACE_READY = load_deepface_resource()


# --- 1. CONFIGURATION AND MAPPING (TELUGU MUSIC) ---

st.set_page_config(
    page_title="Telugu Emotion-Based Music Recommender", 
    layout="wide"
)

# Define the Telugu music mapping for each detected emotion.
MUSIC_MAPPING = {
    "happy": {
        "text": "😄 High-Energy Telugu Pop & Dance Hits",
        "url": "https://www.youtube.com/results?search_query=latest+telugu+party+songs+bounce"
    },
    "sad": {
        "text": "😌 Soothing Telugu Melody Songs (Comforting)",
        "url": "https://www.youtube.com/results?search_query=best+telugu+melody+songs+for+sad+mood"
    },
    "angry": {
        "text": "🧘 Peaceful Telugu Instrumental Music (Calm Down)",
        "url": "https://www.youtube.com/results?search_query=telugu+instrumental+meditation+music+relax"
    },
    "surprise": {
        "text": "🤯 Upbeat Telugu Title Tracks & Mashups",
        "url": "https://www.youtube.com/results?search_query=hit+telugu+title+songs+mashup"
    },
    "fear": {
        "text": "🛡️ Classic Telugu Devotional or Motivational Songs",
        "url": "https://www.youtube.com/results?search_query=telugu+motivational+songs+jukebox"
    },
    "neutral": {
        "text": "☕ Background Telugu Instrumental Tracks",
        "url": "https://www.youtube.com/results?search_query=telugu+instrumental+bgm+for+concentration"
    },
    "disgust": {
        "text": "🌟 Ultimate Telugu Feel-Good Romantic Jams",
        "url": "https://www.youtube.com/results?search_query=evergreen+telugu+romantic+hits+feel+good"
    }
}


# --- 2. STREAMLIT APP LAYOUT ---

st.title("🎶 Emotion-Based Music Recommender")
st.markdown("### Telugu Songs Edition")
st.markdown("Click **'Analyze My Mood'** below to capture your current emotion and get a personalized music recommendation.")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📸 Live Camera Capture")
    # st.camera_input returns the image data after the user clicks "Take Photo"
    camera_image = st.camera_input("Analyze My Mood")


# --- 3. CORE LOGIC (DEEPFACE ANALYSIS) ---

if camera_image is not None and DEEPFACE_READY is not None:
    
    # Display the captured image in the first column (Using corrected parameter)
    with col1:
        st.image(camera_image, caption="Captured Image", use_container_width=True) 

    # 1. Convert the image for DeepFace
    try:
        image_bytes = camera_image.getvalue()
        img_array = np.frombuffer(image_bytes, np.uint8)
        img_array = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        
        # 2. Analyze the image using the cached DeepFace module
        with st.spinner('Analyzing your facial expression...'):
            analysis_results = DEEPFACE_READY.analyze(
                img_path=img_array,
                actions=['emotion'],
                enforce_detection=False # Allows analysis even if detection is slightly difficult
            )
            time.sleep(0.5) # Small visual pause
            
        # Extract the dominant emotion
        if analysis_results and isinstance(analysis_results, list) and analysis_results[0].get('dominant_emotion'):
            
            dominant_emotion = analysis_results[0]['dominant_emotion'].lower()
            recommendation = MUSIC_MAPPING.get(dominant_emotion, MUSIC_MAPPING["neutral"])
            
            # 3. Display results in the second column
            with col2:
                st.markdown("### ✅ Analysis Result")
                st.success(f"**Your Detected Mood:** {dominant_emotion.upper()}!")
                
                st.markdown(f"""
                <div style="padding: 15px; border-radius: 8px; background-color: #333333; color: white;">
                    <p style='font-size: 18px; margin:0;'>
                        The system is recommending music for a **{dominant_emotion.upper()}** mood.
                    </p>
                </div>
                """, unsafe_allow_html=True)

                st.subheader("🎵 Your Recommended Playlist")
                
                st.markdown(
                    f"**Recommendation:** {recommendation['text']}",
                    unsafe_allow_html=True
                )
                
                # --- FINAL FIX: USE st.link_button for non-blocking direct URL open ---
                st.link_button(
                    label="▶️ Open YouTube Playlist Now",
                    url=recommendation['url'], # This uses the URL directly as an HTML link
                    type="primary"
                )
                
                # Optional message to explain browser behavior
                st.info('The playlist should open directly in a new tab.')


        else:
            with col2:
                 st.error("❌ No face detected. Please ensure your face is clearly visible and centered in the frame, and try again.")

    except Exception as e:
        with col2:
            st.error(f"An unexpected error occurred during analysis. Error: {e}")
            st.code(e) # Display the error in the app for quick debugging