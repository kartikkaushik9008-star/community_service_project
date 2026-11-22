# 🎵 Audio Event Detection - Streamlit UI

A beautiful, modern web interface for real-time audio event detection using the ESC-50 dataset and deep learning.

## ✨ Features

### 🎤 Real-Time Detection
- Live microphone monitoring
- Real-time event classification
- Confidence meter with visual gauge
- Event history tracking
- Adjustable confidence threshold

### 📁 File Upload & Analysis
- Support for WAV, MP3, OGG, FLAC formats
- Audio waveform visualization
- Top 5 predictions with confidence scores
- Audio playback

### 📊 Analytics Dashboard
- Event distribution pie chart
- Confidence trends over time
- Summary statistics
- Unique event tracking

## 🚀 How to Run

1. **Install dependencies** (if not already installed):
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

3. **Open your browser** to the URL shown (usually http://localhost:8501)

## 🎨 UI Features

- **Modern Design**: Gradient backgrounds, glassmorphism effects
- **Responsive Layout**: Works on desktop and mobile
- **Interactive Visualizations**: Plotly charts for real-time data
- **Smooth Animations**: Pulse effects, hover transitions
- **Dark Theme**: Premium purple gradient theme

## 📱 Usage

### Real-Time Detection Tab
1. Click "Start Listening" to begin monitoring
2. Speak or play sounds near your microphone
3. Watch detected events appear in real-time
4. View confidence scores and event history
5. Click "Stop Listening" when done

### File Upload Tab
1. Upload an audio file (WAV, MP3, etc.)
2. View the waveform visualization
3. Click "Analyze Audio" to detect events
4. See top 5 predictions with confidence scores

### Analytics Tab
- View event distribution
- Track confidence trends
- See summary statistics

## ⚙️ Settings

- **Confidence Threshold**: Minimum confidence to display events (0.0 - 1.0)
- **History Length**: Number of events to keep in history (5 - 50)

## 🎯 Model Information

- **Architecture**: AudioCNN (3-layer fully connected network)
- **Input**: MFCC features (128 coefficients)
- **Output**: 50 ESC-50 environmental sound classes
- **Sample Rate**: 22,050 Hz
- **Block Duration**: 2.0 seconds

## 🎨 Design Highlights

- Custom CSS with Inter font family
- Gradient backgrounds (purple theme)
- Glassmorphism cards with backdrop blur
- Animated status indicators
- Hover effects and transitions
- Responsive gauge charts
- Interactive plotly visualizations

## 🔧 Technical Stack

- **Frontend**: Streamlit
- **Visualization**: Plotly
- **Audio Processing**: librosa, sounddevice
- **Deep Learning**: PyTorch
- **Data**: Pandas, NumPy

## 📝 Notes

- Ensure your microphone is connected and working
- The model file `model_50class.pth` must be in the project directory
- For best results, use clear audio with minimal background noise
- Confidence threshold can be adjusted based on your needs

## 🎵 Supported Event Classes

The model can detect 50 different environmental sounds from the ESC-50 dataset, including:
- Animals (dog, cat, bird, etc.)
- Natural soundscapes (rain, wind, water, etc.)
- Human sounds (coughing, sneezing, laughing, etc.)
- Interior/domestic sounds (door knock, clock tick, keyboard, etc.)
- Exterior/urban sounds (car horn, siren, engine, etc.)

Enjoy detecting audio events! 🎉
