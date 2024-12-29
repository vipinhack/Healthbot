# HealthBot

**HealthBot** is an AI-powered chatbot designed to assist patients and caregivers by providing relevant health-related information and helping them navigate medical resources. This project leverages **PyTorch** to implement a deep learning-based text classification model, integrated with a comprehensive **NLP pipeline** to process and respond to user queries efficiently. The chatbot also addresses data imbalance in text classification tasks and is trained using advanced techniques to ensure accuracy and performance.

## Key Features

- **AI-Powered Chatbot**: The chatbot is capable of understanding and responding to various health-related queries, providing users with accurate and actionable information.
- **Text Classification**: Uses deep learning techniques to classify user inputs and respond with the most appropriate information.
- **NLP Pipeline**: Includes text preprocessing, tokenization, and embedding layers to efficiently process user queries.
- **Data Imbalance Handling**: The model addresses class imbalance issues using oversampling, undersampling, and class weighting techniques to improve accuracy and avoid bias.
- **Fast and Accurate Response**: The chatbot has been optimized to deliver accurate responses in real-time by experimenting with different architectures, hyperparameters, and pre-trained embeddings.

## Backend

- **PyTorch**: Utilized for building and training the neural network model for text classification. The model incorporates multiple layers for robust feature extraction and prediction.
- **Natural Language Processing (NLP)**: Techniques such as tokenization, word embeddings (e.g., Word2Vec or GloVe), and sequence processing are employed to handle the textual data.
- **Data Handling**: The system uses oversampling of the minority class and undersampling of the majority class to address the data imbalance during model training.

## Model Architecture

- **Neural Network**: The chatbot uses a multi-layered PyTorch neural network model for text classification, which is fine-tuned to maximize accuracy and minimize prediction errors.
- **Pre-trained Embeddings**: Used pre-trained embeddings to represent words in a dense vector space, capturing their semantic meaning and improving the model's understanding of text.

## Challenges & Solutions

- **Data Imbalance**: To ensure the model does not favor majority classes, techniques like oversampling, undersampling, and class weighting were implemented. This improved accuracy by 79%.
- **Text Preprocessing**: Implemented a robust NLP pipeline with tokenization, stopword removal, and stemming/lemmatization to ensure clean input for the model.
- **Model Tuning**: Extensively tuned hyperparameters (e.g., learning rate, batch size) and experimented with different neural network architectures to enhance the model's performance.

## My Role

As the lead developer, I was responsible for designing and implementing the **PyTorch** neural network model, setting up the **NLP pipeline**, and addressing data imbalance issues. I spearheaded the entire development cycle, including model tuning, architecture selection, and hyperparameter optimization. I also managed the deployment of the chatbot and collaborated with teammates for testing and improvement.

## Technologies Used

- **PyTorch**: Deep learning framework used for building and training the text classification model.
- **NLP Libraries**: Libraries such as **NLTK** and **spaCy** for text processing.
- **Pre-trained Embeddings**: Word embeddings like **Word2Vec** and **GloVe** to enhance the model's understanding of words.
- **Python**: Backend language for implementing the AI model and chatbot logic.

