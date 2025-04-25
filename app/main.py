from fastapi import FastAPI, File, HTTPException, UploadFile
import spacy
from collections import Counter
from nltk.tokenize import sent_tokenize
from .resumo import resumir_local
from .utils import extrair_texto_pdf
from dotenv import load_dotenv
import os
import requests 


load_dotenv() 

API_KEY = os.getenv("API_KEY")
API_URL = os.getenv("API_URL")

headers = {
    'Authorization': f'Bearer {API_KEY}',
    'Content-Type': 'application/json'
}

# Carregar o modelo spaCy
nlp_spacy = spacy.load("pt_core_news_sm")

# Criar uma instância FastAPI
app = FastAPI()

@app.post("/resumir-pdf/")
async def resumir_pdf(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Somente arquivos PDF são aceitos.")

        conteudo = await file.read()
        texto_extraido = extrair_texto_pdf(conteudo)

        if not texto_extraido.strip():
            raise HTTPException(status_code=400, detail="Não foi possível extrair texto do PDF.")

        texto_limitado = texto_extraido[:10000]  

        resumo = resumir_local(texto_limitado)
        return {"resumo": resumo}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao gerar resumo: {str(e)}")
    

@app.post("/resumir-pdf-ia/")
async def resumir_pdf_ia(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Somente arquivos PDF são aceitos.")

        conteudo = await file.read()
        texto_extraido = extrair_texto_pdf(conteudo)

        if not texto_extraido.strip():
            raise HTTPException(status_code=400, detail="Não foi possível extrair texto do PDF.")

        prompt = f"Resuma o seguinte texto em português de forma clara e objetiva:\n\n{texto_extraido[:10000]}"  # Limitando tamanho

        data = {
            "model": "deepseek/deepseek-chat:free",
            "messages": [
                {"role": "system", "content": "Você é um assistente especializado em leitura e resumo de artigos científicos."},
                {"role": "user", "content": prompt}
            ]
        }

        response = requests.post(API_URL, headers=headers, json=data)

        if response.status_code == 200:
            resposta_json = response.json()
            resumo = resposta_json['choices'][0]['message']['content'].strip()
            return {"resumo": resumo}
        else:
            print("Erro:", response.text)
            raise HTTPException(status_code=500, detail="Erro ao chamar a API do OpenRouter.")

    except Exception as e:
        print("Erro:", e)
        raise HTTPException(status_code=500, detail="Erro ao processar o PDF.")

@app.get("/")
def read_root():
    return {"message": "API de resumo de artigos científicos está funcionando!"}
