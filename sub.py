{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "bc041f8d-a03b-40d0-9157-8195e508f810",
   "metadata": {},
   "source": [
    "# To predict on Streamlit code (LR)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9436ac36-9127-4dff-ba10-3402cb05d84b",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "import streamlit as st\n",
    "import pandas as pd\n",
    "import joblib\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b0d475c2-39f9-4acf-90a7-f25e6d7c5908",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load the trained model\n",
    "model = joblib.load('titanic_logreg_model_retrained.pkl')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "57a626d4-a8aa-4d7b-8f1a-081d77af14ba",
   "metadata": {},
   "outputs": [],
   "source": [
    "def main():\n",
    "    st.title(\"Titanic Survival Prediction\")  # Title should always be visible\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cfe537a1-179d-4439-9c08-c1e8bebf84d5",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Inputs\n",
    "    passenger_id = st.text_input(\"Passenger ID\", \"\")\n",
    "    pclass = st.selectbox(\"Passenger Class (Pclass)\", [1, 2, 3], index=2)\n",
    "    sex = st.selectbox(\"Sex\", [\"male\", \"female\"])\n",
    "    age = st.number_input(\"Age (in years)\", min_value=0, max_value=120, value=30)\n",
    "    sibsp = st.number_input(\"Number of Siblings/Spouses Aboard (SibSp)\", min_value=0, max_value=10, value=0)\n",
    "    parch = st.number_input(\"Number of Parents/Children Aboard (Parch)\", min_value=0, max_value=10, value=0)\n",
    "    fare = st.number_input(\"Fare (Ticket Price)\", min_value=0.0, value=50.0)\n",
    "    embarked = st.selectbox(\"Port of Embarkation (Embarked)\", [\"C\", \"Q\", \"S\"])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fa669631-5216-49ae-8f19-b0348e87d3fc",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Preprocess inputs\n",
    "    input_data = pd.DataFrame({\n",
    "        'Pclass': [pclass],\n",
    "        'Sex': [sex],\n",
    "        'Age': [age],\n",
    "        'SibSp': [sibsp],\n",
    "        'Parch': [parch],\n",
    "        'Fare': [fare],\n",
    "        'Embarked': [embarked]\n",
    "    })"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "59442166-8878-4079-aeea-47b0558f94c0",
   "metadata": {},
   "outputs": [],
   "source": [
    " # Use the model's internal pipeline for preprocessing\n",
    "    if st.button(\"Predict\"):\n",
    "        try:\n",
    "            prediction = model.predict(input_data)[0]\n",
    "            survival_probability = model.predict_proba(input_data)[0][1]\n",
    "\n",
    "            if prediction == 1:\n",
    "                st.success(f\"Survived (Probability: {survival_probability:.2f})\")\n",
    "            else:\n",
    "                st.error(f\"Did Not Survive (Probability: {survival_probability:.2f})\")\n",
    "        except ValueError as e:\n",
    "            st.error(f\"Error during prediction: {str(e)}\")\n",
    "\n",
    "if __name__ == '__main__':\n",
    "    main()\n",
    "\n",
    "st.write(\"Debug: Model loaded successfully\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5976a65d-f0ae-4910-b6e2-c8c2063f2378",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "aa6ec642-beed-4eb8-939f-a55a7ecc7d0f",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
