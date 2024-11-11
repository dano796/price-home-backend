from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pickle
from fastapi.middleware.cors import CORSMiddleware

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


class HouseData(BaseModel):
    area: int
    baths: int
    city: str
    garages: int
    is_new: int
    neighbourhood: str
    property_type: str
    rooms: int
    stratum: str


@app.get('/saludar')
async def saludar():
    return "Hola mundo"

@app.post("/predict")
async def predict_price(house_data: HouseData):
    df = pd.DataFrame([house_data.model_dump()])

    data_preparada = pd.get_dummies(
        df,
        columns=["is_new", "stratum", "property_type", "neighbourhood", "city"],
        drop_first=False,
    )
    data_preparada = data_preparada.reindex(columns=variables, fill_value=0)

    try:
        prediction = model.predict(data_preparada)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"predicted_price": prediction[0]}