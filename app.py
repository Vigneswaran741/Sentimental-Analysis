# importing required libraries
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import streamlit as st


def remove_digit(text):
    """
    Removes numbers from text
    """
    return "".join(word for word in text if not word.isdigit())


# preprocess and make prediction with the model
def preprocess(text):
    """
    Makes predictions on given text and returns whether it is a real news or not
    """

    # preprocess text
    text = np.array(text)
    custom_text_modified = []
    for i in range(len(text)):
        custom_text_modified.append(remove_digit(text[i]))

    # loading model
    model = tf.keras.models.load_model("Model_3_GRU_with_USE.h5", custom_objects={'KerasLayer':hub.KerasLayer})

    # making predictions
    preds = model.predict(custom_text_modified)

    # rounding off the predictions
    preds = int(np.round(np.argmax(preds)))

    # returning the predictions
    if preds == 0:
        return "Negative"
    elif preds == 1:
        return "Neutral"

    return "Positive"


def main():
        
    st.title("Sentimental Analysis")
    text = st.text_area("Enter the Review you saw:", placeholder="Copy & Paste the Review here...",
                        help="Provide the Review", height=350)

    if st.button("Submit"):
        st.snow()
        if text == "":
            st.warning(" Please Enter some Review", icon="⚠️")

        else:
            result = preprocess([text])
            if result == "Positive":
                st.success(result, icon="✅")
            elif result == "Neutral":
                st.info(result, icon="✅")
            else:
                st.error("Negative Review", icon="❌")

    

if __name__ == "__main__":
    # call main function
    main()