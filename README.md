# BeyondSight

Beyond Sight is an AI-powered assistive application designed to help vision-impaired individuals to navigate their surroundings with ease and also to know who is coming towards them with the help of facial recognition using their saved contact database. The application provides real-time voice-guided navigation, object detection, facial recognition, and auditory assistance to improve accessibility and independence.

🚀 Features

Voice-Guided Navigation: Provides step-by-step audio guidance to help users move safely.

Object Detection: Identifies objects in the environment and describes them audibly.

![image](https://github.com/user-attachments/assets/761ab099-bb75-4c8b-902b-a99936dc53af)


Facial Recognition: Recognizes and speaks out names of familiar individuals based on a CSV database.

![image](https://github.com/user-attachments/assets/3886843c-1fe4-4ac6-a55e-4c63dad80610)


Audio Assistance: Converts detected text and recognized names into speech using ElevenLabs TTS API.

Depth Detection: Detects how far is the object



🛠️ Technologies Used

Python (OpenCV, Face Recognition, TensorFlow, NumPy, Pandas)

Text-to-Speech (TTS) via ElevenLabs API

Computer Vision for facial and object recognition

Machine Learning for facial detection

CSV Database for storing recognized faces

📂 Project Structure

BeyondSight/
│── models/                  # Trained AI models for detection & recognition
│── data/                    # CSV database with names and images
│── audio/                   # Audio files for speech synthesis
│── src/
│   │── facial_recognition.py # Recognizes faces and speaks names
│   │── object_detection.py   # Detects and announces objects
│   │── speech.py            # Text-to-speech module
│   │── main.py              # Main application file
│── requirements.txt         # Dependencies
│── README.md                # Documentation

🔧 Installation

1️⃣ Clone the Repository

git clone https://github.com/KrishnaLodha/BeyondSight.git
cd BeyondSight

2️⃣ Install Dependencies

pip install -r requirements.txt

3️⃣ Set Up ElevenLabs API

Sign up at ElevenLabs

Get your API key.

Open speech.py and replace your_api_key_here with your actual key.


📌 How It Works

The camera detects faces and objects in real-time.

If a face is recognized, the system speaks the person's name.

Objects in the surroundings are identified and described audibly.

The system provides navigation assistance using voice commands.

⚡ Future Enhancements

Multi-Language Support: Enable support for multiple languages in TTS.

Gesture Recognition: Allow users to give commands via hand gestures.

Cloud Integration: Store user profiles and history for personalized assistance.

📜 License

This project is licensed under the MIT License. Feel free to modify and use it!



