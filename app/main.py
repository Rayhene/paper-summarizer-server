from fastapi import FastAPI, File, HTTPException, UploadFile
import spacy
from collections import Counter
from nltk.tokenize import sent_tokenize
from .resumo import resumir_local
from .utils import extrair_texto_pdf
from dotenv import load_dotenv
import os
import requests 
import re
from fastapi.middleware.cors import CORSMiddleware


load_dotenv() 

API_KEY = os.getenv("API_KEY")
API_URL = os.getenv("API_URL")

headers = {
    'Authorization': f'Bearer {API_KEY}',
    'Content-Type': 'application/json'
}

nlp_spacy = spacy.load("pt_core_news_sm")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/resumir-pdf-ia/")
async def resumir_pdf_ia(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Somente arquivos PDF são aceitos.")

        conteudo = await file.read()
        texto_extraido = extrair_texto_pdf(conteudo)

        if not texto_extraido.strip():
            raise HTTPException(status_code=400, detail="Não foi possível extrair texto do PDF.")


        def dividir_em_chunks(texto, tamanho=4000):
            return [texto[i:i + tamanho] for i in range(0, len(texto), tamanho)]
        
        chunks = dividir_em_chunks(texto_extraido)

        print(f"🔹 Total de chunks criados: {len(chunks)}")

        texto_completo = "\n\n".join(chunks)


        prompt = (
            f"Você é um assistente especializado em leitura e resumo de artigos científicos.\n\n"
            f"A seguir está o conteúdo completo de um artigo dividido em partes:\n\n{texto_completo}\n\n"
            f"Gere um resumo estruturado com as seções:\n"
            f"- Introdução\n- Metodologia\n- Resultados\n- Discussão\n- Conclusão\n\n"
            f"O resumo deve ser objetivo e conter apenas as informações mais relevantes."
        )
        
        print(f"📤 Enviando prompt para o modelo... Tamanho: {len(prompt)} caracteres")

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

            resumo_estruturado = resumo.replace("### Resumo Estruturado", "").strip()
            print("Texto markdown:", resumo_estruturado)
            return {"resumo": resumo_estruturado}

        else:
            print("Erro:", response.text)
            raise HTTPException(status_code=500, detail="Erro ao chamar a API do OpenRouter.")

    except Exception as e:
        print("Erro:", e)
        raise HTTPException(status_code=500, detail="Erro ao processar o PDF.")

@app.get("/")
def read_root():
    return {"message": "API de resumo de artigos científicos está funcionando!"}


def parse_resumo_estruturado(texto_markdown):
    print(f"🔹 Texto markdown: {texto_markdown}")
    secoes = {}
    
    matches = re.split(r"###\s+(.*?)\n", texto_markdown)

    for i in range(1, len(matches), 2):
        titulo = matches[i].strip()
        conteudo = matches[i + 1].strip() if i + 1 < len(matches) else ""
        secoes[titulo.lower()] = conteudo

    return secoes