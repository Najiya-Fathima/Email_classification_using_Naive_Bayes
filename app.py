import streamlit as st
import joblib

# Load the pre-trained pipeline:
pipeline_filename = "email_classifier_naive.joblib" 
loaded_pipeline = joblib.load(pipeline_filename)

def classify_email(email_text):
    """Classifies an email and returns probabilities."""
    prediction = loaded_pipeline.predict([email_text])
    probabilities = loaded_pipeline.predict_proba([email_text])
    
    result = "The given email is HAM" if prediction[0] == 0 else "The given email is SPAM"
    ham_prob = probabilities[0][0]
    spam_prob = probabilities[0][1]
    
    return result, ham_prob, spam_prob # Return all three values


# Streamlit app:
st.title("Spam Email Classifier")

email_input = st.text_area("Enter email text here:")

if st.button("Classify"):
    if email_input:
        result, ham_prob, spam_prob = classify_email(email_input)  # Get all returned values
        st.write(f"**Classification:** {result}")
        st.write(f"**Probability of HAM:** {ham_prob:.2f}")  # Format to 4 decimal places
        st.write(f"**Probability of SPAM:** {spam_prob:.2f}")
    else:
        st.write("Please enter some email text.")
