import streamlit as st
import requests

st.title("Wine Prediction App")

# تعريف أسماء الفيتشرز حسب بيانات الـ Wine
wine_features = [
    "alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium",
    "total_phenols", "flavanoids", "nonflavanoid_phenols", "proanthocyanins",
    "color_intensity", "hue", "od280/od315_of_diluted_wines", "proline"
]

# نخلي المستخدم يدخل كل القيم
values = []
for feature in wine_features:
    val = st.number_input(feature, 0.0, 100.0, 10.0)
    values.append(val)

if st.button("Predict"):
    response = requests.post(
        "http://127.0.0.1:8000/predict",
        json={"values": values}
    )
    st.write("Status Code:", response.status_code)
    st.write("Raw Response:", response.text)
    if response.status_code == 200:
        result = response.json()
        st.success(f"Prediction: {result.get('prediction')}")
