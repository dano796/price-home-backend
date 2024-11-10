from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pickle

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

with open("modeloVentasRealState.pkl", "rb") as file:
    model, variables, min_max_scaler = pickle.load(file)

@app.get('/saludar')
async def saludar():
    return {"message": "Hola mundo"}

@app.post('/predict')
async def predict_price(house_data: dict):
    try:
        # Extraer datos del diccionario de entrada y estructurar
        input_data = {
            "area": house_data.get("area", 0),
            "baths": house_data.get("baths", 0),
            "garages": house_data.get("garages", 0),
            "rooms": house_data.get("rooms", 0),
            "is_new": f"is_new_{house_data.get('is_new', 0)}",
            "stratum": f"stratum_{house_data.get('stratum', '1')}",
            "property_type": f"property_type_{house_data.get('property_type', '')}",
            "neighbourhood": f"neighbourhood_{house_data.get('neighbourhood', '')}",
            "city": f"city_{house_data.get('city', '')}"
        }

        # Crear un vector de entrada con ceros, y llenar según las variables
        data_preparada = {var: 0 for var in variables}
        for key, value in input_data.items():
            if value in data_preparada:
                data_preparada[value] = 1
            elif key in data_preparada:
                data_preparada[key] = value

        # Transformar diccionario en lista según el orden de 'variables' para la predicción
        data_list = [data_preparada.get(var, 0) for var in variables]

        # Realizar la predicción
        prediction = model.predict([data_list])
        return {"predicted_price": prediction[0]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al predecir el precio: {str(e)}")
