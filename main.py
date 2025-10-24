from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np

# Carrega o modelo treinado
modelo = joblib.load("lr_otimizada.pkl")

# Inicializa o app FastAPI
app = FastAPI(
    title="API de Previsão de Diabetes",
    description="API que utiliza um modelo de Machine Learning para prever diabetes",
    version="1.0.0"
)

# Define o esquema de entrada (validação automática)
class InputData(BaseModel):
    gravidez: int = Field(..., ge=0, description="Número de gravidezes")
    glicose: float = Field(..., ge=0, description="Nível de glicose no sangue")
    imc: float = Field(..., ge=0, description="Índice de Massa Corporal (IMC)")
    idade: int = Field(..., ge=0, description="Idade em anos")

@app.get("/")
def root():
    return {"mensagem": "API de Previsão de Diabetes está online!"}

@app.post("/predict")
def prever_diabetes(dados: InputData):
    try:
        # Define o threshold fixo (valor de corte)
        threshold = 0.42

        # Prepara a entrada no formato esperado pelo modelo
        entrada = np.array([[dados.gravidez, dados.glicose, 0, 0, 0, dados.imc, 0, dados.idade]])

        # Calcula a probabilidade
        prob = modelo.predict_proba(entrada)[:, 1][0]
        resultado = 1 if prob >= threshold else 0

        return {
            "probabilidade": round(float(prob), 4),
            "classe": resultado
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
