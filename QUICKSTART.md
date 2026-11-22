# 🚀 Quick Start Guide - Streamlit Audio Event Detection UI

## ✅ Your Streamlit App is Running!

The application is now live at: **http://localhost:8501**

## 🎨 What You'll See

The UI features a **modern, premium design** with:

### 🎵 Main Features

1. **Real-Time Detection Tab** 🎤
   - Start/Stop listening buttons
   - Live confidence gauge meter
   - Current event display with animated badges
   - Event history table with timestamps
   - Adjustable confidence threshold

2. **File Upload Tab** 📁
   - Drag & drop audio files (WAV, MP3, OGG, FLAC)
   - Waveform visualization
   - Audio playback
   - Top 5 predictions with confidence bars
   - Instant analysis

3. **Analytics Tab** 📊
   - Event distribution pie chart
   - Confidence trends over time
   - Summary statistics (total detections, unique events, avg confidence)
   - Visual insights into detection patterns

### 🎨 Design Highlights

- **Purple Gradient Theme**: Beautiful gradient from #667eea to #764ba2
- **Glassmorphism Cards**: Frosted glass effect with backdrop blur
- **Smooth Animations**: Pulse effects, hover transitions
- **Interactive Charts**: Plotly visualizations with real-time updates
- **Responsive Layout**: Works on all screen sizes
- **Modern Typography**: Inter font family

## 🎯 How to Use

### Real-Time Detection
1. Open http://localhost:8501 in your browser
2. Go to the "🎤 Real-Time Detection" tab
3. Click "🎙️ Start Listening"
4. Make sounds or play audio near your microphone
5. Watch events appear in real-time!
6. Adjust the confidence threshold in the sidebar

### File Analysis
1. Go to the "📁 File Upload" tab
2. Upload an audio file
3. View the waveform
4. Click "🔍 Analyze Audio"
5. See the top 5 predictions

### View Analytics
1. Go to the "📊 Analytics" tab
2. View event distribution
3. Track confidence trends
4. See summary statistics

## ⚙️ Sidebar Settings

- **Confidence Threshold**: 0.0 - 1.0 (default: 0.6)
  - Higher = fewer but more confident detections
  - Lower = more detections with varying confidence

- **History Length**: 5 - 50 events (default: 20)
  - Number of recent events to display

## 🎵 Supported Events (50 Classes)

The model can detect:
- **Animals**: dog, cat, bird, crow, frog, etc.
- **Natural**: rain, wind, water drops, thunderstorm, etc.
- **Human**: coughing, sneezing, laughing, crying, etc.
- **Indoor**: door knock, clock tick, keyboard typing, etc.
- **Outdoor**: car horn, siren, engine, helicopter, etc.

## 🔧 Troubleshooting

### Microphone Not Working?
- Check browser permissions (allow microphone access)
- Ensure your microphone is connected
- Check system audio settings

### No Events Detected?
- Lower the confidence threshold
- Ensure audio is clear and loud enough
- Check that the model file exists (model_50class.pth)

### App Not Loading?
- Refresh the browser (Ctrl+R or Cmd+R)
- Check that the command is still running
- Restart the app: `python -m streamlit run app.py`

## 🛑 Stopping the App

To stop the Streamlit server:
- Press `Ctrl+C` in the terminal
- Or close the terminal window

## 📝 Tips for Best Results

1. **Clear Audio**: Use clear audio with minimal background noise
2. **Proper Volume**: Ensure audio is at a reasonable volume
3. **Microphone Quality**: Better microphones = better detection
4. **Threshold Tuning**: Adjust confidence threshold based on your needs
5. **File Format**: WAV files typically work best for analysis

## 🎉 Enjoy!

Your audio event detection system is ready to use with a beautiful, modern interface!

For more information, see STREAMLIT_README.md
