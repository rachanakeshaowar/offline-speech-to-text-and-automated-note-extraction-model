# 📢 Whisper Fine-Tuning

This project involves fine-tuning OpenAI's Whisper model for Automatic Speech Recognition (ASR). Below is the list of Python packages used, along with their purpose.


## 📦 Python Packages

- **`datasets[audio]`**  
  Used to download and prepare our training data.

- **`transformers`** and **`accelerate`**  
  Used to load and train our Whisper model efficiently.

- **`soundfile`**  
  Required to pre-process audio files.

- **`evaluate`** and **`jiwer`**  
  Used to assess the performance of our model by calculating evaluation metrics like Word Error Rate (WER).

- **`tensorboard`**  
  Used to log and visualize training metrics such as loss, accuracy, and learning rate.

- **`gradio`**  
  Used to build a simple and flashy demo of our fine-tuned model, allowing users to upload audio and get real-time transcriptions.

---
# 📥 Data Preparation

 We have downloaded and prepare the Common Voice splits.

---


