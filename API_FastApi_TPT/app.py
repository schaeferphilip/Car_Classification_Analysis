import uvicorn
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse 
from fastapi.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates 
import pandas as pd
import numpy as np
from model_wrapper import maclass 

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

model = maclass('config.ini') 

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    # return {"Hello": "World"}
    return templates.TemplateResponse("home.html", {"request": request})

@app.post('/predict', response_class=HTMLResponse)
async def predict(request: Request, age: float = Form(), sexe: str = Form(), taux: float = Form(), 
                            situationFamiliale: str = Form(),  nbEnfantsAcharge: float = Form(), 
                            VoitureN2: str = Form(),  ): 
    
    tranche_age = 0 if age < 35 else (1 if age>=35 and age<60 else 2)
    data = [age, sexe, taux, situationFamiliale, nbEnfantsAcharge, VoitureN2, tranche_age] 
    X= pd.DataFrame(data=[data], columns=model.data_preprocessor.feature_names_in_)
    catgeorie, vehicles =  model.predict(X)
    # prediction_name = target_names[prediction]
    return templates.TemplateResponse('home.html', {
                           "request": request, 
                           "pred_target": f'La catégorie adéquate est celle des <b class="text-decoration-underline">{catgeorie}</b>',
                           "vehicles": vehicles.to_html(col_space=25, index=False, 
                                                justify="center", max_rows=7, 
                                                classes=["table", "table-striped", ])
                           })

if(__name__) == '__main__':
    uvicorn.run(
        "app:app",
        host    = "0.0.0.0",
        port    = 5000, 
        reload  = True
    )