import http.client
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
from datetime import datetime

# Fungsi untuk konversi suhu dari Kelvin ke Celsius
def kelvin_to_celsius(kelvin):
    return kelvin - 273.15

# Fungsi untuk konversi suhu dari Kelvin ke Fahrenheit
def kelvin_to_fahrenheit(kelvin):
    return (kelvin - 273.15) * 9/5 + 32

# Step 1: Mendapatkan data cuaca dari API
def get_weather_data(city=""):
    conn = http.client.HTTPSConnection("weather-api167.p.rapidapi.com")

    headers = {
        'x-rapidapi-key': "b70903ee8dmsha552142825bc669p12c254jsnebaba604ce7c",
        'x-rapidapi-host': "weather-api167.p.rapidapi.com",
        'Accept': "application/json"
    }

    # Gantilah "sorong" dengan nama kota yang ingin Anda gunakan
    conn.request("GET", f"/api/weather/forecast?place={city}&cnt=3&units=standard&type=three_hour&mode=json&lang=id", headers=headers)

    res = conn.getresponse()
    data = res.read()

    # Mengonversi respons JSON ke dictionary Python
    weather_data = json.loads(data.decode("utf-8"))

    # Debugging: Tampilkan sebagian data untuk memeriksa struktur
    print(json.dumps(weather_data, indent=4))  # Menampilkan dengan format yang lebih mudah dibaca
    
    return weather_data

# Step 2: Mengolah data cuaca untuk digunakan dalam pelatihan model
def prepare_data(weather_data):
    data = {
        "Datetime": [],
        "Temperature (°C)": [],  # Menggunakan Celcius
        "Feels Like (°C)": [],
        "Humidity (%)": [],
        "Cloudiness (%)": [],
        "Wind Speed (m/s)": [],
        "Pressure (hPa)": [],
        "Rain (mm)": []  # Target untuk klasifikasi hujan
    }

    # Menyusun data dari JSON
    for forecast in weather_data["list"]:
        try:
            # Mengambil data cuaca
            datetime = forecast["dt_txt"]
            temperature_k = forecast["main"].get("temp", None)  # Menggunakan .get() untuk menghindari KeyError
            feels_like_k = forecast["main"].get("feels_like", None)
            humidity = forecast["main"].get("humidity", None)
            cloudiness = forecast["clouds"].get("all", None)
            wind_speed = forecast["wind"].get("speed", None)
            pressure = forecast["main"].get("pressure", None)
            rain = forecast["rain"].get("3h", 0) if "rain" in forecast else 0  # Hujan dalam 3 jam

            # Memeriksa jika data yang diperlukan ada (jika tidak, lanjutkan ke iterasi berikutnya)
            if None in [temperature_k, feels_like_k, humidity, cloudiness, wind_speed, pressure]:
                print(f"Data tidak lengkap untuk waktu {datetime}, melewati data ini...")
                continue

            # Mengonversi suhu ke Celsius
            temperature_c = kelvin_to_celsius(temperature_k)
            feels_like_c = kelvin_to_celsius(feels_like_k)

            # Menambahkan data ke dictionary
            data["Datetime"].append(datetime)
            data["Temperature (°C)"].append(temperature_c)
            data["Feels Like (°C)"].append(feels_like_c)
            data["Humidity (%)"].append(humidity)
            data["Cloudiness (%)"].append(cloudiness)
            data["Wind Speed (m/s)"].append(wind_speed)
            data["Pressure (hPa)"].append(pressure)
            data["Rain (mm)"].append(1 if rain > 0 else 0)  # Jika ada hujan, target = 1, jika tidak = 0

        except KeyError as e:
            print(f"KeyError: {e} pada forecast {forecast}")
            continue

    # Membuat DataFrame
    df = pd.DataFrame(data)
    return df

# Step 3: Melatih model
def train_model(df):
    # Menyiapkan fitur (X) dan target (y)
    X = df[["Temperature (°C)", "Feels Like (°C)", "Humidity (%)", "Cloudiness (%)", "Wind Speed (m/s)", "Pressure (hPa)"]]
    y = df["Rain (mm)"]  # Target (hujan: 0 atau 1)
    
    # Membagi dataset menjadi data latih dan data uji
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Membuat model regresi logistik
    model = LogisticRegression()

    # Melatih model
    model.fit(X_train, y_train)

    # Memprediksi hasil untuk data uji
    y_pred = model.predict(X_test)

    # Mengukur akurasi
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.2f}")

    return model

# Step 4: Menyimpan model
def save_model(model, filename="rain_predictor.pkl"):
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

# Step 5: Menampilkan kisaran waktu perkiraan hujan
def predict_rain_range(df, model):
    rain_times = []
    for index, row in df.iterrows():
        datetime = row["Datetime"]
        rain_pred = model.predict([row[["Temperature (°C)", "Feels Like (°C)", "Humidity (%)", "Cloudiness (%)", "Wind Speed (m/s)", "Pressure (hPa)"]]])[0]
        
        # Jika model memprediksi hujan
        if rain_pred == 1:
            time = datetime.split()[1]  # Mengambil waktu (jam:menit:detik)
            rain_times.append(time)
    
    # Menampilkan kisaran waktu hujan
    if rain_times:
        print("Perkiraan hujan pada kisaran waktu: ")
        print(f"Jam sekitar {min(rain_times)} - {max(rain_times)}")
    else:
        print("Tidak ada perkiraan hujan dalam rentang waktu yang tersedia.")

# Main program
if __name__ == "__main__":
    # Mendapatkan data cuaca dari API untuk kota yang dimasukkan
    city = input("Masukkan nama kota untuk ramalan cuaca: ").strip().lower() or "sorong"
    weather_data = get_weather_data(city)

    # Mengolah data untuk pelatihan
    df = prepare_data(weather_data)
    
    # Menampilkan data yang lebih mudah dibaca
    print("\nData yang diproses (dalam satuan yang lebih mudah dibaca): ")
    print(df.to_string(index=False))

    # Melatih model
    model = train_model(df)

    # Menyimpan model
    save_model(model)

    # Menampilkan kisaran waktu perkiraan hujan
    predict_rain_range(df, model)
