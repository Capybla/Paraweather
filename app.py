from flask import Flask, render_template
import requests
from geopy.geocoders import Nominatim

app = Flask(__name__)

API_KEY = "bc52655a59ac00cac12cb4faa0e90044"  # Clave de OpenWeather

# Función para obtener la ubicación del dispositivo (por ahora fija en Paiporta)
def obtener_ubicacion_actual():
    lat, lon = 39.4258, -0.4183  # Coordenadas de Paiporta
    geolocator = Nominatim(user_agent="geoapiExercises")
    location = geolocator.reverse((lat, lon), language="es")
    if location and location.raw.get('address'):
        ciudad = location.raw['address'].get('city', 'Desconocido')
        return lat, lon, ciudad
    return lat, lon, "Desconocido"

# Función para obtener datos meteorológicos
def obtener_clima(lat, lon):
    url = f"https://api.openweathermap.org/data/2.5/weather"
    params = {
        "lat": lat,
        "lon": lon,
        "appid": API_KEY,
        "units": "metric",
        "lang": "es"
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        return None

# Función para calcular el nivel de advertencia
def calcular_nivel(clima):
    viento = clima['wind']['speed'] * 3.6  # Convertir m/s a km/h
    temperatura = clima['main']['temp']
    humedad = clima['main']['humidity']

    if 5 <= viento <= 10 and 15 <= temperatura <= 25 and humedad <= 70:
        return {"nivel": 1, "color": "verde", "mensaje": "Perfecto para volar", "simbolo": "✅"}
    elif 10 < viento <= 15 and 10 <= temperatura <= 28:
        return {"nivel": 2, "color": "verde claro", "mensaje": "Excelente", "simbolo": "✅"}
    elif 15 < viento <= 20 and 8 <= temperatura <= 30:
        return {"nivel": 3, "color": "amarillo claro", "mensaje": "Muy bueno", "simbolo": "⚠️"}
    elif 20 < viento <= 25 and 5 <= temperatura <= 35:
        return {"nivel": 4, "color": "amarillo", "mensaje": "Bueno", "simbolo": "⚠️"}
    elif 25 < viento <= 30:
        return {"nivel": 5, "color": "naranja", "mensaje": "Aceptable", "simbolo": "⚠️"}
    elif 30 < viento <= 35:
        return {"nivel": 6, "color": "rojo claro", "mensaje": "Subóptimo", "simbolo": "⚠️"}
    elif 35 < viento <= 40:
        return {"nivel": 7, "color": "rojo", "mensaje": "Riesgoso", "simbolo": "⛔"}
    elif 40 < viento <= 50:
        return {"nivel": 8, "color": "rojo oscuro", "mensaje": "Muy peligroso", "simbolo": "⛔"}
    elif viento > 50:
        return {"nivel": 9, "color": "negro", "mensaje": "Prohibido volar", "simbolo": "⛔"}
    else:
        return {"nivel": 10, "color": "gris", "mensaje": "Datos insuficientes", "simbolo": "❓"}

@app.route("/")
def index():
    lat, lon, ciudad = obtener_ubicacion_actual()
    clima = obtener_clima(lat, lon)
    if not clima:
        return render_template("error.html", mensaje="No se pudo obtener el clima.")
    nivel = calcular_nivel(clima)
    return render_template("index.html", ciudad=ciudad, clima=clima, nivel=nivel)

if __name__ == "__main__":
    app.run(debug=True)