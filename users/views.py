from ast import alias
from concurrent.futures import process
from django.shortcuts import render

# Create your views here.
from django.shortcuts import render, HttpResponse
from django.contrib import messages

import brain_disease_and_age

from .forms import UserRegistrationForm
from .models import UserRegistrationModel
from django.conf import settings
import os


def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})


def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(
                loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHomePage.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})


def UserHome(request):
    return render(request, 'users/UserHomePage.html', {})

# def DatasetView(request):
#     path = settings.MEDIA_ROOT + "//" + 'alzheimers_prediction_dataset.csv'
#        df = pd.read_csv(path)
#     df = df.to_html
#     return render(request, 'users/viewdataset.html', {'data': df})

import pandas as pd
# df = pd.read_csv(r'C:\Users\Ramaa\OneDrive\Desktop\project\Brain age and  disease Classifcation\CODE\brain_disease_and_age\brain_disease_and_age\your_file.csv')
df = pd.read_csv(r'C:\Users\Ramaa\OneDrive\Desktop\project\Brain age and  disease Classifcation\CODE\brain_disease_and_age\media\alzheimers_prediction_dataset.csv')
x = df.drop('Alzheimer_Diagnosis', axis=1)
y=df['Alzheimer_Diagnosis']

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report , confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, log_loss, roc_curve, auc
from sklearn.linear_model import LogisticRegression
import numpy as np
import seaborn as sns

path = r'balanced_data.csv'
df = pd.read_csv(path)
df.fillna(0, inplace=True)

    
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)
model = LogisticRegression(random_state=42)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
# Create your views here.
from django.shortcuts import render
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


def Training(request):
    # Define model at the start of the function
    model = LogisticRegression(max_iter=1000, random_state=42)

    # Train/test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Fit the model on the training data
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy:.2f}")      
    # Make predictions on the test data
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # Confusion matrix plot
    # plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=['No Alzheimer\'s', 'Alzheimer\'s'], 
                yticklabels=['No Alzheimer\'s', 'Alzheimer\'s'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

    # Train/test accuracies
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    print(f"Train Accuracy: {train_acc * 100:.2f}%")
    print(f"Test Accuracy: {test_acc * 100:.2f}%")

    # Cross-validation
    train_accuracies = []
    val_accuracies = []
    train_losses = []
    val_losses = []
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for train_index, val_index in kf.split(x, y):
        x_train_fold, x_val_fold = x.iloc[train_index], x.iloc[val_index]
        y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]
        
        # Train the model on each fold
        model.fit(x_train_fold, y_train_fold)
        
        # Train and validation metrics
        y_train_pred_fold = model.predict(x_train_fold)
        y_train_prob_fold = model.predict_proba(x_train_fold)
        train_accuracies.append(accuracy_score(y_train_fold, y_train_pred_fold))
        train_losses.append(log_loss(y_train_fold, y_train_prob_fold))

        y_val_pred_fold = model.predict(x_val_fold)
        y_val_prob_fold = model.predict_proba(x_val_fold)
        val_accuracies.append(accuracy_score(y_val_fold, y_val_pred_fold))
        val_losses.append(log_loss(y_val_fold, y_val_prob_fold))

    # Plot Training vs Validation Accuracy
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Train Accuracy', marker='o')
    plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Validation Accuracy', marker='o')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training vs Validation Accuracy')

    # Plot Training vs Validation Loss
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', marker='o')
    plt.xlabel('Fold')
    plt.ylabel('Log Loss')
    plt.legend()
    plt.title('Training vs Validation Loss')
    plt.tight_layout()
    plt.show()

    # ROC curve and AUC
    y_test_prob = model.predict_proba(x_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_test_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

    # Return context to template
    context = {
        'accuracy':accuracy,
        'train_acc': train_acc,
        'test_acc': test_acc
    }
    return render(request, 'users/training.html', context)

def training_brain(request):
    import os
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    from io import BytesIO
    import base64

    # Set image size and batch size
    img_size = (224, 224)
    batch_size = 32
    
    # Load age data
    age_csv_path = os.path.join(settings.MEDIA_ROOT, 'Brain_tumor_MRI_Dataset', 'age.csv')
    age_df = pd.read_csv(age_csv_path)
    # age_df = pd.read_csv(r'C:\Users\satti\OneDrive\Desktop\project\Brain age and  disease Classifcation\CODE\brain_disease_and_age\media\Brain_tumor_MRI_Dataset\age.csv')
    
    # Organize data directories
    # data_dir = os.path.join(settings.MEDIA_ROOT, 'Brain_tumor_MRI_Dataset')
    # data_dir = r'C:\Users\satti\OneDrive\Desktop\project\Brain age and  disease Classifcation\CODE\brain_disease_and_age\media\media\Brain_tumor_MRI_Dataset'
    # train_dir = os.path.join(data_dir, 'Training')
    # test_dir = os.path.join(data_dir, 'Testing')
    data_dir = os.path.join(settings.MEDIA_ROOT, 'Brain_tumor_MRI_Dataset')
    train_dir = os.path.join(data_dir, 'Training')
    test_dir = os.path.join(data_dir, 'Testing')

 # Now print these directories to verify
    print(f"Data directory: {data_dir}")
    print(f"Training directory: {train_dir}")
    print(f"Testing directory: {test_dir}")
    
    # Image data generators
    tr_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    ts_gen = ImageDataGenerator(rescale=1./255)
    
    # Load training, validation, and test data
    gen_train = tr_gen.flow_from_directory(
        train_dir, target_size=img_size, class_mode='categorical', batch_size=batch_size, shuffle=True)
    
    gen_valid = tr_gen.flow_from_directory(
        train_dir, target_size=img_size, class_mode='categorical', batch_size=batch_size, subset='validation')
    
    gen_test = ts_gen.flow_from_directory(
        test_dir, target_size=img_size, class_mode='categorical', batch_size=batch_size, shuffle=False)
    
    # Model architecture
    input_layer = tf.keras.layers.Input(shape=(224, 224, 3))
    x = Conv2D(64, (3,3), activation='relu')(input_layer)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3,3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    
    # Outputs
    classification_output = Dense(4, activation='softmax', name='classification')(x)
    age_output = Dense(1, activation='linear', name='age_estimation')(x)
    
    # Define and compile model
    model = tf.keras.Model(inputs=input_layer, outputs=[classification_output, age_output])
    model.compile(
        optimizer='adam',
        loss={'classification': 'categorical_crossentropy', 'age_estimation': 'mse'},
        metrics={'classification': 'accuracy', 'age_estimation': 'mae'}
    )
    
    # Train model
    history = model.fit(
        gen_train, epochs=5, validation_data=gen_valid, verbose=1,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
    )
    
    # Save model
    model.save("Brain_age_Estimator_model.h5")
    
    # Evaluate model
    scores = model.evaluate(gen_test)
    
    # Predictions
    preds = model.predict(gen_test)
    y_pred_labels = np.argmax(preds[0], axis=1)
    y_pred_ages = preds[1]
    
    # Confusion matrix
    cm = confusion_matrix(gen_test.classes, y_pred_labels)
    labels = list(gen_train.class_indices.keys())
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    cm_buffer = BytesIO()
    plt.savefig(cm_buffer, format='png')
    cm_buffer.seek(0)
    cm_image = base64.b64encode(cm_buffer.getvalue()).decode('utf-8')
    
    # Plot age estimation
    plt.figure()
    plt.hist(y_pred_ages, bins=20, color='blue', alpha=0.6, label='Predicted Ages')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.title('Age Estimation Distribution')
    plt.legend()
    age_buffer = BytesIO()
    plt.savefig(age_buffer, format='png')
    age_buffer.seek(0)
    age_image = base64.b64encode(age_buffer.getvalue()).decode('utf-8')
    
    context = {
        'cm_image': cm_image,
        'age_image': age_image,
        'classification_accuracy': scores[3],  # Assuming classification accuracy is at index 3
        'age_mae': scores[4],  # Assuming age MAE is at index 4
    }
    
    return render(request, 'users/training_brain.html', context)

from django.shortcuts import render
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from io import BytesIO
import base64
from PIL import Image
import matplotlib.pyplot as plt

def predict_brain(request):
    model_dir = os.path.join(settings.MEDIA_ROOT, 'Brain_age_Estimator_model.h5')

    # Load trained model
    model = tf.keras.models.load_model(model_dir)

    # Load and preprocess the uploaded image
    if request.method == 'POST' and request.FILES['image']:
        uploaded_image = request.FILES['image']
        img = Image.open(uploaded_image)
        img = img.resize((224, 224))  # Resize image to match model input
        img_array = np.array(img) / 255.0  # Rescale the image

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        # Make predictions
        preds = model.predict(img_array)
        classification_preds = np.argmax(preds[0], axis=1)  # Class prediction
        age_prediction = preds[1]  # Age prediction

        # Display classification prediction
        class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']  # Replace with your class names
        predicted_class = class_labels[classification_preds[0]]

        # Convert the predicted age to an integer between 1 and 100
        predicted_age = int(age_prediction[0][0] * 100)  # Scale to 100 and convert to integer
        predicted_age = min(max(predicted_age, 1), 100)  # Ensure it's between 1 and 100

        # Prepare results for display
        context = {
            'predicted_class': predicted_class,
            'predicted_age': predicted_age,
        }

        # Display the uploaded image
        img_buffer = BytesIO()
        img.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        img_data = base64.b64encode(img_buffer.getvalue()).decode('utf-8')

        context['image_data'] = img_data

        return render(request, 'users/prediction_brain.html', context)

    return render(request, 'users/prediction_brain.html')


def Prediction(request):
    if request.method == "POST":
        ag = request.POST.get('age')
        cl = request.POST.get('cholesterol_levels')
        cd = request.POST.get('chest_discomfort')
        ohf1 = request.POST.get('other_health_factor_1')
        ohf2 = request.POST.get('other_health_factor_2')
        # ag = int(ag) if ag else 0
        # cl = int(cl) if cl else 0
        cd = 1 if cd == 'Yes' else 0
        # ohf1 = float(ohf1) if ohf1 else 0
        # ohf2 = float(ohf2) if ohf2 else 0
        
        input_df = pd.DataFrame({
            'Age': [ag],
            'Cholesterol_Levels': [cl],
            'Chest_Discomfort': [cd],
            'Other_Health_Factor_1': [ohf1],
            'Other_Health_Factor_2': [ohf2]
        })
        print(input_df)
        print("Input DataFrame before encoding:", input_df)
        op = model.predict(input_df)
        prediction_label = "Patient contain Alzheimer's" if int(op[0]) == 1 else "Patient does not contain Alzheimer's"
        print("Prediction label:", prediction_label)
        context = {
        'prediction': prediction_label
        }

        return render(request, 'users/predict_form.html', context)

    return render(request, 'users/predict_form.html')

