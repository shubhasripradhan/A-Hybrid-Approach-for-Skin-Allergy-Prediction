import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Embedding, LSTM, Concatenate, Dropout
from tensorflow.keras.models import Model
import json
import ast
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# Load and preprocess the text datasets
try:
    skin_disease_df = pd.read_csv('skin_disease_classifier.csv')
    skin_text_df = pd.read_csv('Skin_text_classifier.csv')
    
    # Combine text data from both datasets
    text_data = []
    
    # Process first dataset
    for _, row in skin_disease_df.iterrows():
        try:
            # Try to safely evaluate the string as a dictionary
            data = ast.literal_eval(row['Skin_disease_classification'])
            if isinstance(data, dict) and 'query' in data:
                text_data.append(data['query'])
        except:
            continue
    
    # Process second dataset
    for _, row in skin_text_df.iterrows():
        try:
            data = ast.literal_eval(row['Skin_disease_classification'])
            if isinstance(data, dict) and 'query' in data:
                text_data.append(data['query'])
        except:
            continue
    
except Exception as e:
    print(f"Warning: Could not load CSV files. Using default text data. Error: {str(e)}")
    # Fallback text data
    text_data = [
        "Red, itchy rash on skin that appears after exposure",
        "Dark, scaly patches on face and neck",
        "Small, flesh-colored bumps on arms",
        "Painful, swollen areas with redness",
        "White, flaky patches on scalp",
        "Circular, red patches with clear center",
        "Brown spots that darken with sun exposure",
        "Rough, scaly patches on sun-exposed areas",
        "Tiny red dots that don't fade under pressure"
    ]

# Initialize tokenizer for text processing
tokenizer = Tokenizer(num_words=1000, oov_token='<OOV>')
tokenizer.fit_on_texts(text_data)

# Disease descriptions for additional context
disease_descriptions = {
    'Actinic keratosis': 'Rough, scaly patches on skin caused by years of sun exposure. Usually found on face, lips, ears, hands, forearms. Red or pink colored lesions.',
    'Atopic Dermatitis': 'Chronic inflammatory skin condition causing red, itchy rashes. Common in skin folds, face, neck. Often appears dry and scaly.',
    'Benign keratosis': 'Non-cancerous growth on skin, light brown to black in color. Waxy, scaly, slightly raised appearance. Common in older adults.',
    'Dermatofibroma': 'Harmless round growth in skin, usually brown to reddish. Firm to touch, may dimple when pressed. Common on legs.',
    'Melanocytic nevus': 'Common mole, usually brown or black. Round or oval shaped with distinct borders. Can be flat or raised.',
    'Melanoma': 'Serious form of skin cancer. Irregular borders, varying colors, asymmetrical shape. May change over time. Often larger than 6mm.',
    'Squamous cell carcinoma': 'Type of skin cancer. Red, scaly patches or rough, thickened skin. May bleed easily. Often on sun-exposed areas.',
    'Tinea Ringworm Candidiasis': 'Fungal infection causing red, circular rash with clearer center. Scaly, itchy patches. Can occur anywhere on body.',
    'Vascular lesion': 'Abnormal cluster of blood vessels. Red or purple in color. May be flat or raised. Various sizes and shapes.'
}

# Symptom categories and options
SYMPTOM_OPTIONS = {
    'Location': [
        'Face',
        'Neck',
        'Arms',
        'Legs',
        'Chest',
        'Back',
        'Hands',
        'Feet',
        'Scalp',
        'Whole body'
    ],
    'Appearance': [
        'Red',
        'White',
        'Brown',
        'Black',
        'Purple',
        'Scaly',
        'Rough',
        'Smooth',
        'Raised',
        'Flat',
        'Bumpy',
        'Blistered',
        'Crusty',
        'Flaky'
    ],
    'Size': [
        'Tiny (< 2mm)',
        'Small (2-5mm)',
        'Medium (5-10mm)',
        'Large (>10mm)',
        'Varying sizes'
    ],
    'Symptoms': [
        'Itchy',
        'Painful',
        'Burning',
        'Stinging',
        'Numb',
        'No sensation',
        'Bleeding',
        'Oozing',
        'Dry',
        'Warm to touch'
    ],
    'Duration': [
        'New (< 1 week)',
        'Recent (1-4 weeks)',
        'Persistent (1-6 months)',
        'Chronic (> 6 months)'
    ],
    'Pattern': [
        'Constant',
        'Comes and goes',
        'Getting worse',
        'Getting better',
        'Changes with weather',
        'Worse at night',
        'Worse during day'
    ],
    'Triggers': [
        'Sun exposure',
        'Heat',
        'Cold',
        'Stress',
        'Exercise',
        'Food',
        'Medications',
        'No clear trigger',
        'Contact with substances'
    ]
}

def create_multimodal_model():
    # Image input branch
    image_input = Input(shape=(150, 150, 3), name='image_input')
    x = Conv2D(64, (3, 3), activation='relu')(image_input)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    image_features = Dropout(0.5)(x)

    # Text input branch
    text_input = Input(shape=(100,), name='text_input')
    y = Embedding(1000, 64)(text_input)
    y = LSTM(128, return_sequences=True)(y)
    y = LSTM(64)(y)
    text_features = Dense(64, activation='relu')(y)

    # Combine image and text features
    combined = Concatenate()([image_features, text_features])
    z = Dense(128, activation='relu')(combined)
    z = Dropout(0.5)(z)
    z = Dense(64, activation='relu')(z)
    output = Dense(len(disease_descriptions), activation='softmax')(z)

    model = Model(inputs=[image_input, text_input], outputs=output)
    model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    return model

def preprocess_image(img_array):
    try:
        # Convert to BGR for OpenCV
        img_bgr = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB2BGR)
        
        # 1. Denoise
        img_denoised = cv2.fastNlMeansDenoisingColored(img_bgr)
        
        # 2. Enhance contrast using CLAHE
        lab = cv2.cvtColor(img_denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        enhanced = cv2.merge((cl,a,b))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # 3. Sharpen
        kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        # Convert back to RGB
        img_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        
        # 4. Normalize to [0,1]
        img_normalized = img_rgb.astype(np.float32) / 255.0
        
        return img_normalized
    except Exception as e:
        print(f"Warning: Error in image preprocessing. Using basic normalization. Error: {str(e)}")
        return img_array.astype(np.float32) / 255.0

def prepare_text_data(text):
    try:
        # Convert text to sequence
        sequence = tokenizer.texts_to_sequences([text])
        # Pad sequence
        padded = pad_sequences(sequence, maxlen=100, padding='post', truncating='post')
        return padded[0]
    except Exception as e:
        print(f"Warning: Error in text preprocessing. Using zero padding. Error: {str(e)}")
        return np.zeros(100)

def analyze_text_similarity(user_text, disease_name):
    try:
        # Get all queries for this disease from both datasets
        disease_queries = []
        
        # Process first dataset
        try:
            disease_data = skin_disease_df[skin_disease_df['Skin_disease_classification'].str.contains(disease_name, case=False)]
            for _, row in disease_data.iterrows():
                data = ast.literal_eval(row['Skin_disease_classification'])
                if isinstance(data, dict) and 'query' in data:
                    disease_queries.append(data['query'].lower())
        except:
            pass
        
        # Process second dataset
        try:
            disease_data = skin_text_df[skin_text_df['Skin_disease_classification'].str.contains(disease_name, case=False)]
            for _, row in disease_data.iterrows():
                data = ast.literal_eval(row['Skin_disease_classification'])
                if isinstance(data, dict) and 'query' in data:
                    disease_queries.append(data['query'].lower())
        except:
            pass
        
        # Add disease description
        if disease_name in disease_descriptions:
            disease_queries.append(disease_descriptions[disease_name].lower())
        
        # Calculate similarity score based on common words
        user_words = set(user_text.lower().split())
        max_similarity = 0
        
        for query in disease_queries:
            query_words = set(query.split())
            intersection = len(user_words.intersection(query_words))
            union = len(user_words.union(query_words))
            similarity = intersection / union if union > 0 else 0
            max_similarity = max(max_similarity, similarity)
        
        return max_similarity
    except Exception as e:
        print(f"Warning: Error in text similarity analysis. Using default similarity. Error: {str(e)}")
        return 0.1

def predict_skin_disease(image_path, user_symptoms):
    try:
        # Load and preprocess image
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(150, 150))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = preprocess_image(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Prepare text data
        text_data = prepare_text_data(user_symptoms)
        text_data = np.expand_dims(text_data, axis=0)
        
        # Get model predictions
        predictions = model.predict({'image_input': img_array, 'text_input': text_data}, verbose=0)
        
        # Calculate text similarity scores for each disease
        similarity_scores = []
        for disease in disease_descriptions.keys():
            similarity = analyze_text_similarity(user_symptoms, disease)
            similarity_scores.append(similarity)
        
        # Combine model predictions with text similarity
        combined_scores = 0.7 * predictions[0] + 0.3 * np.array(similarity_scores)
        
        # Get top 3 predictions
        top_3_indices = np.argsort(combined_scores)[-3:][::-1]
        disease_names = list(disease_descriptions.keys())
        
        results = []
        for idx in top_3_indices:
            disease_name = disease_names[idx]
            confidence = combined_scores[idx]
            results.append((disease_name, confidence))
        
        # Display results
        print("\nPrediction Results:")
        print("-" * 50)
        for i, (disease, confidence) in enumerate(results, 1):
            print(f"{i}. {disease}")
            print(f"   Confidence: {confidence:.2%}")
            print(f"   Description: {disease_descriptions[disease]}\n")
        
        # Display the image
        plt.figure(figsize=(8, 8))
        plt.imshow(img)
        plt.title(f"Top Prediction: {results[0][0]}\nConfidence: {results[0][1]:.2%}")
        plt.axis('off')
        plt.show()
        
        return results[0][0], results[0][1]
    
    except Exception as e:
        print(f"Error processing input: {str(e)}")
        return None, None

def create_cnn_model():
    """Create a CNN-only model for comparison"""
    inputs = Input(shape=(150, 150, 3))
    x = Conv2D(64, (3, 3), activation='relu')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(len(disease_descriptions), activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    return model

def predict_with_cnn(image_array, cnn_model):
    """Make prediction using CNN-only model"""
    predictions = cnn_model.predict(np.expand_dims(image_array, axis=0), verbose=0)
    predicted_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_idx]
    disease_names = list(disease_descriptions.keys())
    return disease_names[predicted_idx], confidence

def compare_predictions(image_path, symptoms, selections):
    """Compare predictions between CNN-only and multimodal models"""
    try:
        # Load and preprocess image
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(150, 150))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = preprocess_image(img_array)
        
        # Get CNN-only prediction
        cnn_prediction, cnn_confidence = predict_with_cnn(img_array, cnn_model)
        
        # Get multimodal prediction
        text_data = prepare_text_data(symptoms)
        text_data = np.expand_dims(text_data, axis=0)
        img_array_batch = np.expand_dims(img_array, axis=0)
        
        multimodal_predictions = model.predict(
            {'image_input': img_array_batch, 'text_input': text_data}, 
            verbose=0
        )
        
        # Calculate text similarity scores
        similarity_scores = []
        for disease in disease_descriptions.keys():
            similarity = analyze_text_similarity(symptoms, disease)
            similarity_scores.append(similarity)
        
        # Combine predictions with text similarity
        combined_scores = 0.7 * multimodal_predictions[0] + 0.3 * np.array(similarity_scores)
        
        # Get multimodal prediction
        disease_names = list(disease_descriptions.keys())
        multimodal_prediction = disease_names[np.argmax(combined_scores)]
        multimodal_confidence = np.max(combined_scores)
        
        # Display comparison
        print("\nPrediction Comparison:")
        print("-" * 50)
        
        print("\nCNN-only Model:")
        print(f"Prediction: {cnn_prediction}")
        print(f"Confidence: {cnn_confidence:.2%}")
        
        print("\nMultimodal Model:")
        print(f"Prediction: {multimodal_prediction}")
        print(f"Confidence: {multimodal_confidence:.2%}")
        
        # Display top 3 predictions for each model
        print("\nTop 3 Predictions:")
        print("-" * 50)
        
        print("\nCNN-only Model:")
        top_3_cnn = np.argsort(predictions[0])[-3:][::-1]
        for idx in top_3_cnn:
            print(f"{disease_names[idx]}: {predictions[0][idx]:.2%}")
        
        print("\nMultimodal Model:")
        top_3_multimodal = np.argsort(combined_scores)[-3:][::-1]
        for idx in top_3_multimodal:
            print(f"{disease_names[idx]}: {combined_scores[idx]:.2%}")
        
        # Display the image with both predictions
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title(f"CNN Prediction:\n{cnn_prediction}\nConfidence: {cnn_confidence:.2%}")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(img)
        plt.title(f"Multimodal Prediction:\n{multimodal_prediction}\nConfidence: {multimodal_confidence:.2%}")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        return {
            'cnn': (cnn_prediction, cnn_confidence),
            'multimodal': (multimodal_prediction, multimodal_confidence)
        }
        
    except Exception as e:
        print(f"Error in prediction comparison: {str(e)}")
        return None

def calculate_accuracy_metrics(results):
    """Calculate accuracy metrics for both models"""
    if not results:
        return
    
    cnn_predictions = []
    multimodal_predictions = []
    true_classes = []
    
    for result in results:
        if result['true_class'] and result['cnn'] and result['multimodal']:
            true_classes.append(result['true_class'])
            cnn_predictions.append(result['cnn'][0])
            multimodal_predictions.append(result['multimodal'][0])
    
    if not true_classes:
        print("No valid predictions to calculate accuracy metrics.")
        return
    
    print("\nAccuracy Metrics:")
    print("-" * 50)
    
    # CNN Model Metrics
    print("\nCNN-only Model:")
    cnn_accuracy = accuracy_score(true_classes, cnn_predictions)
    print(f"Accuracy: {cnn_accuracy:.2%}")
    print("\nClassification Report:")
    print(classification_report(true_classes, cnn_predictions, target_names=disease_descriptions.keys()))
    
    # Multimodal Model Metrics
    print("\nMultimodal Model:")
    multimodal_accuracy = accuracy_score(true_classes, multimodal_predictions)
    print(f"Accuracy: {multimodal_accuracy:.2%}")
    print("\nClassification Report:")
    print(classification_report(true_classes, multimodal_predictions, target_names=disease_descriptions.keys()))
    
    # Plot confusion matrices
    plt.figure(figsize=(20, 8))
    
    plt.subplot(1, 2, 1)
    cm_cnn = confusion_matrix(true_classes, cnn_predictions, labels=list(disease_descriptions.keys()))
    sns.heatmap(cm_cnn, annot=True, fmt='d', cmap='Blues', 
                xticklabels=disease_descriptions.keys(), 
                yticklabels=disease_descriptions.keys())
    plt.title('CNN Model Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha='right')
    
    plt.subplot(1, 2, 2)
    cm_multimodal = confusion_matrix(true_classes, multimodal_predictions, labels=list(disease_descriptions.keys()))
    sns.heatmap(cm_multimodal, annot=True, fmt='d', cmap='Blues', 
                xticklabels=disease_descriptions.keys(), 
                yticklabels=disease_descriptions.keys())
    plt.title('Multimodal Model Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()

# Load both models
print("Loading models...")
try:
    model = tf.keras.models.load_model("skin_disease/multimodal_model.h5")
    cnn_model = tf.keras.models.load_model("skin_disease/skin_disease_cnn_model.h5")
except Exception as e:
    print(f"Error loading models: {str(e)}")
    print("Please ensure both model files exist and are accessible.")
    exit(1)

def get_symptom_selection(category, options):
    """Helper function to get user selection from a list of options"""
    print(f"\n{category}:")
    for i, option in enumerate(options, 1):
        print(f"{i}. {option}")
    
    selections = []
    while True:
        choice = input("\nEnter the numbers of all that apply (comma-separated) or 0 to skip: ").strip()
        if choice == '0':
            break
            
        try:
            # Parse the comma-separated input
            choices = [int(x.strip()) for x in choice.split(',')]
            # Validate choices
            valid_choices = [c for c in choices if 1 <= c <= len(options)]
            if valid_choices:
                selections.extend([options[i-1] for i in valid_choices])
                break
            else:
                print("Invalid selection. Please try again.")
        except:
            print("Invalid input. Please enter numbers separated by commas.")
    
    return selections

def format_symptom_description(selections):
    """Convert symptom selections into a natural language description"""
    description_parts = []
    
    for category, selected in selections.items():
        if selected:
            if category == 'Location':
                description_parts.append(f"The condition appears on my {', '.join(selected).lower()}")
            elif category == 'Appearance':
                description_parts.append(f"It looks {' and '.join(selected).lower()}")
            elif category == 'Size':
                description_parts.append(f"The affected areas are {', '.join(selected).lower()}")
            elif category == 'Symptoms':
                description_parts.append(f"It feels {' and '.join(selected).lower()}")
            elif category == 'Duration':
                description_parts.append(f"The condition has been {selected[0].lower()}")
            elif category == 'Pattern':
                description_parts.append(f"The condition is {' and '.join(selected).lower()}")
            elif category == 'Triggers':
                if len(selected) == 1 and selected[0] == 'No clear trigger':
                    description_parts.append("There is no clear trigger")
                else:
                    description_parts.append(f"It gets worse with {' and '.join(selected).lower()}")
    
    return '. '.join(description_parts) + '.'

def get_user_input():
    """Get image path and structured symptom description from user"""
    print("\nSkin Disease Diagnosis System")
    print("-" * 50)
    
    # Get image path
    while True:
        image_path = input("\nEnter the path to your skin image: ").strip()
        if os.path.exists(image_path):
            break
        print(f"\nError: Image file '{image_path}' not found. Please try again.")
    
    # Get structured symptom description
    print("\nPlease describe your symptoms by selecting from the following categories.")
    print("For each category, you can select multiple options or skip if not applicable.")
    
    selections = {}
    for category, options in SYMPTOM_OPTIONS.items():
        selections[category] = get_symptom_selection(category, options)
    
    # Convert selections to natural language description
    symptoms = format_symptom_description(selections)
    
    # Show final description and allow for additional notes
    print("\nBased on your selections, here's the symptom description:")
    print(symptoms)
    
    additional_notes = input("\nWould you like to add any additional notes about your symptoms? (Enter to skip): ").strip()
    if additional_notes:
        symptoms += f" Additional notes: {additional_notes}"
    
    return image_path, symptoms, selections

# Main execution
if __name__ == "__main__":
    print("\nWelcome to the Skin Disease Diagnosis System!")
    print("This system will compare predictions between CNN-only and multimodal approaches.")
    print("You will be guided through a series of questions to describe your symptoms.")
    
    results = []
    while True:
        image_path, symptoms, selections = get_user_input()
        
        # Get predictions from both models
        predictions = compare_predictions(image_path, symptoms, selections)
        
        if predictions:
            result = {
                'image_path': image_path,
                'symptoms': symptoms,
                'selections': selections,
                'cnn': predictions['cnn'],
                'multimodal': predictions['multimodal']
            }
            
            # Ask for true class if known
            true_class = input("\nIf you know the true condition, please enter it (or press Enter to skip): ").strip()
            if true_class:
                result['true_class'] = true_class
            
            results.append(result)
            
            # Calculate accuracy metrics if we have enough data
            if len(results) > 0:
                calculate_accuracy_metrics(results)
            
            print("\nWould you like to analyze another image? (yes/no)")
            response = input().lower().strip()
            if response != 'yes':
                break
        else:
            print("\nWould you like to try again? (yes/no)")
            response = input().lower().strip()
            if response != 'yes':
                break
    
    # Final accuracy calculation
    if results:
        print("\nFinal Accuracy Metrics:")
        calculate_accuracy_metrics(results)

print("\nThank you for using the Skin Disease Diagnosis System!") 