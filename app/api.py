from fastapi import FastAPI

app = FastAPI()

@app.get('/saludar')
def home():
    return {'msg': 'Todo en orden'}