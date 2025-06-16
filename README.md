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
# 📥 Load Dataset (Data Preparation)

 We have downloaded and prepare the Common Voice splits.

 
 Since Hindi is a very low-resource language, we'll combine the `train` and `validation` splits to create our training data:

- **Training Data**: `train + validation` → ~8 hours of data
- **Test Data**: `test` → ~4 hours of data used as held-out test set
  
For fine-tuning the Whisper model, we only consider the **input audio** and corresponding **transcribed text** from the dataset.  
All additional metadata fields (such as speaker ID, age, gender, accent, etc.) are discarded during pre-processing.

---

# 📥 Load Whisper Feature Extractor (Data Preprocessing)

The `WhisperFeatureExtractor` performs two important preprocessing operations before passing the audio to the model:

1️⃣ **Padding / Truncating**  

- Pads audio inputs shorter than 30 seconds with silence (zeros).
   
- Truncates audio inputs longer than 30 seconds to fit 30 seconds.  

2️⃣ **Log-Mel Spectrogram Conversion**  

- Converts 1D raw audio signals into log-Mel spectrograms.
  
- Log-Mel spectrograms are visual representations of audio and serve as input features for Whisper.

 - We load the feature extractor from the pre-trained Whisper checkpoint

## 🔤 Load Whisper Tokenizer

- The Whisper model outputs a sequence of token ids.
  
- The tokenizer maps each of these token ids to their corresponding text string.
  
- For Hindi, we can load the pre-trained tokenizer and use it for fine-tuning without any further modifications.
  
- We simply have to specify the target language and the task.
  
- These arguments inform the tokenizer to prefix the language and task tokens to the start of encoded label sequences.

## 🔤 Combine To Create A WhisperProcessor

- To simplify using the feature extractor and tokenizer, we can wrap both into a single WhisperProcessor class.
  
- This processor object inherits from the WhisperFeatureExtractor and WhisperProcessor, and can be used on the audio inputs and model predictions as required.
  
- In doing so, we only need to keep track of two objects during training: the processor and the model.

## 🔤Prepare Data

- Print the first example of the Common Voice dataset to see what form the data is in.

- Since our input audio is sampled at 48kHz, we need to downsample it to 16kHz prior to passing it to the Whisper feature extractor, 16kHz being the sampling rate expected by the Whisper model.

- We'll set the audio inputs to the correct sampling rate using dataset's cast_column method.
  
- This operation does not change the audio in-place, but rather signals to datasets to resample audio samples on the fly the first time that they are loaded.

- Re-loading the first audio sample in the Common Voice dataset will resample it to the desired sampling rate.

- Now we implemented function to prepare our data ready for the model:

- We load and resample the audio data by calling batch["audio"]. As explained above, Datasets performs any necessary resampling operations on the fly.
  
- We use the feature extractor to compute the log-Mel spectrogram input features from our 1-dimensional audio array.
  
- We encode the transcriptions to label ids through the use of the tokenizer.

- We can apply the data preparation function to all of our training examples using dataset's .map method.
  
- The argument num_proc specifies how many CPU cores to use.
  
- Setting num_proc > 1 will enable multiprocessing.
  
- If the .map method hangs with multiprocessing, set num_proc=1 and process the dataset sequentially.

---
# 📥Training and Evaluation

- Now that we've prepared our data, we're ready to dive into the training pipeline. The Trainer will do much of the heavy lifting for us. All we have to do is:

- Load a pre-trained checkpoint: we need to load a pre-trained checkpoint and configure it correctly for training.

- Define a data collator: the data collator takes our pre-processed data and prepares PyTorch tensors ready for the model.

- Evaluation metrics: during evaluation, we want to evaluate the model using the word error rate (WER) metric. We need to define a compute_metrics function that handles this computation.

- Define the training configuration: this will be used by the Trainer to define the training schedule.

- Once we've fine-tuned the model, we will evaluate it on the test data to verify that we have correctly trained it to transcribe speech in Hindi.

## 🔤 Load a Pre-Trained Checkpoint

- We'll start our fine-tuning run from the pre-trained Whisper small checkpoint, the weights for which we need to load from the Hugging Face Hub. Again, this is trivial through use of 🤗 Transformers!
  
- We can disable the automatic language detection task performed during inference, and force the model to generate in Hindi.
  
- To do so, we set the langauge and task arguments to the generation config.
  
- We'll also set any forced_decoder_ids to None, since this was the legacy way of setting the language and task arguments.

 ## 🔤 Define a Data Collator
 
- The data collator for a sequence-to-sequence speech model is unique in the sense that it treats the input_features and labels independently: the input_features must be handled by the feature extractor and the labels by the tokenizer.

- The input_features are already padded to 30s and converted to a log-Mel spectrogram of fixed dimension by action of the feature extractor, so all we have to do is convert the input_features to batched PyTorch tensors.
  
- We do this using the feature extractor's .pad method with return_tensors=pt.

- The labels on the other hand are un-padded. We first pad the sequences to the maximum length in the batch using the tokenizer's .pad method.
  
- The padding tokens are then replaced by -100 so that these tokens are not taken into account when computing the loss.
  
- We then cut the BOS token from the start of the label sequence as we append it later during training.

- We can leverage the WhisperProcessor we defined earlier to perform both the feature extractor and the tokenizer operations.
  
- Initialise the data collator we've just defined.

---

 # 📥 Evaluation Metrics
 
- We'll use the word error rate (WER) metric, the 'de-facto' metric for assessing ASR systems.
  
- For more information, refer to the WER docs. We'll load the WER metric from Evaluate.
  
- We then simply have to define a function that takes our model predictions and returns the WER metric.
  
- This function, called compute_metrics, first replaces -100 with the pad_token_id in the label_ids (undoing the step we applied in the data collator to ignore padded tokens correctly in the loss).
  
- It then decodes the predicted and label ids to strings.
  
- Finally, it computes the WER between the predictions and reference labels.

 ## 🔤 Define the Training Configuration
- In the final step, we define all the parameters related to training.
  
- We'll save the processor object once before starting training.
  
- Since the processor is not trainable, it won't change over the course of training.

  ---

# 📥 Training
- Training will take approximately 5-10 hours depending on your GPU or the one allocated to this Google Colab.
  
- If using this Google Colab directly to fine-tune a Whisper model, you should make sure that training isn't interrupted due to inactivity.
  
- The peak GPU memory for the given training configuration is approximately 15.8GB.

- Depending on the GPU allocated to the Google Colab, it is possible that you will encounter a CUDA "out-of-memory" error when you launch training.
  
- In this case, can reduce the per_device_train_batch_size incrementally by factors of 2 and employ gradient_accumulation_steps to compensate.
  
- Our best WER is 32.0% -  for 8h of training data! We can make our model more accessible on the Hub with appropriate tags and README information.
  
- We can change these values to match your dataset, language and model name accordingly.
  
- The training results uploaded to the Hub. To do so, we execute the push_to_hub command and save the preprocessor object we created.

  ---
# 📥 Building A Demo
  - Now that we've fine-tuned our model we can build a demo to show off its ASR capabilities!
    
  - We'll make use of Transformers pipeline, which will take care of the entire ASR pipeline, right from pre-processing the audio inputs to decoding the model predictions.

  - Running the example below will generate a Gradio demo where we can record speech through the microphone of our computer and input it to our fine-tuned Whisper model to transcribe the corresponding text:

  ---
# 📥 Summary
  - In this Project, we completed a step-by-step fine-tuning Whisper for multilingual ASR using Datasets, Transformers and the Hugging Face Hub.. 
  
