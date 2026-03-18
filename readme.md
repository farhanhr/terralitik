# 🌱 Terralitik — Java Drought Risk EWS

**Terralitik** is a smart, AI-powered dashboard designed to help farmers and local governments in Java, Indonesia, monitor, predict, and mitigate the risks of agricultural drought and crop failure. 


- **🗺️ Interactive Heatmap:** Visualizes the current drought risk level across all regencies and cities in Java.
- **📈 14-Day AI Forecast:** Uses a Machine Learning model to predict the drought risk score for the next two weeks based on real-time satellite data.
- **💼 Economic Risk Analysis:** Calculates the estimated crop loss percentage and potential financial loss (in Rupiah) per hectare.

---

## 🚀 How to Run the Project Locally

Follow these simple steps to run Terralitik on your own computer.

### Step 1: Clone the Repository
Download the project to your computer:
```bash
git clone [https://github.com/your-username/terralitik.git](https://github.com/your-username/terralitik.git)
cd terralitik
```

### Step 2: Install Dependencies
It is highly recommended to use a virtual environment. Install all the required libraries:
```bash
pip install -r requirements.txt
```

### Step 3: Set Up Your AI API Key
The AI Assistant needs a Google Gemini API Key to work. 
1. Get a free API key from [Google AI Studio](https://aistudio.google.com/).
2. Inside the project folder, create a new hidden folder named `.streamlit`.
3. Inside the `.streamlit` folder, create a file named `secrets.toml`.
4. Open the file and paste your API key like this:
   ```toml
   GEMINI_API_KEY = "put_your_api_key_here"
   ```

### Step 4: Run the Dashboard!
Start the Streamlit application:
```bash
streamlit run src/dashboard/app.py
```
*The application will automatically open in your default web browser.*

---

## ⚙️ How the Automated Data Works 
You do not need to update the data manually! This project includes an automated pipeline. Every midnight, **GitHub Actions** will automatically run the `daily_update.yaml` script. It will:
1. Fetch the 14-day weather forecast from Open-Meteo.
2. Update the `.csv` database using the upsert method (preventing duplicates).
3. Push the new data to this repository so the live Streamlit dashboard is always up-to-date.

*If you want to pull the data manually on your computer, simply run:*
```bash
python src/data_pipeline/fetch_weather.py
```