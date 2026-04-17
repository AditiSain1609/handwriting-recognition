# Handwriting Recognition (A–Z)

##  Project Description
This project is a **Handwriting Recognition System** that can identify English alphabets (A–Z) from a given image.
The user provides a **black and white clean image of a single alphabet**, and the model predicts which alphabet it is using a **CNN (Convolutional Neural Network)**.

---
##  Features
* Predicts handwritten alphabets (A–Z)
* Accepts user-uploaded image
* Can also test using sample images
* Simple web interface using Flask
* Deep Learning model (CNN) for accurate prediction

---

##  Technologies Used
* Python 
* CNN (Convolutional Neural Network)
* Flask (Web Framework)
* HTML (Frontend)

---

## How to Run the Project

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/handwriting-recognition.git
cd handwriting-recognition
```
### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

*(If requirements.txt not available, install manually: Flask, numpy, tensorflow, etc.)*

---

### Step 3: Train the Model (Optional)

```bash
python train.py
```

---

### Step 4: Run the Application

```bash
python app.py
```

---

### Step 5: Open in Browser

Go to:

```
http://127.0.0.1:5000/
```

---

## Input

* User can upload:
 * Black & white image
  * Single alphabet (A–Z)
* OR use sample images for testing

---

## Output

* Model predicts the alphabet (A–Z)
* Displays result on the screen

---

##  Example

Input: Image of handwritten "B"
Output: **Predicted Alphabet → B**

---

##  Future Improvements

* Add support for lowercase letters
* Improve accuracy with more data
* Add real-time drawing input (canvas)
* Deploy project online

---

## Author

Aditi Saini
