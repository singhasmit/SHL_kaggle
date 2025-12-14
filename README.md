# Grammar Scoring Engine from Spoken Audio

This project implements an end-to-end grammar scoring system for spoken English audio. 
The goal is to predict a continuous grammar score (0–5) for each audio sample based on speech fluency and grammatical structure.

## Dataset
The dataset consists of short spoken audio recordings (45–60 seconds each) provided as part of a Kaggle assessment.
Each training audio file is associated with a grammar score, while test samples contain only audio files without labels.

## Approach
1. Audio samples are transcribed into text using the Whisper speech-to-text model.
2. Linguistic and fluency-based features are extracted from the transcripts, including word count, speaking rate (WPM), sentence length statistics, repetition ratio, and fragment ratio.
3. A Ridge Regression model is trained using these features to predict grammar scores.
4. Model performance is evaluated using RMSE and Pearson correlation, along with visual analysis of predictions.

## Model Selection
Multiple regression approaches were explored. Ridge Regression was chosen as the final model due to its ability to generalize better on unseen data through L2 regularization.

## Evaluation
The final model achieved reasonable performance on validation data, capturing relative differences in grammatical proficiency while avoiding overfitting.

## Submission
Predictions for the test dataset were generated using the trained model, clipped to the valid score range (0–5), and saved in the required `submission.csv` format.

## Tools & Libraries
- Python
- Whisper (speech-to-text)
- Scikit-learn
- NumPy, Pandas
- Matplotlib

## Author
Asmit Singh
