import fitz  # PyMuPDF

def extrair_texto_pdf(file_bytes: bytes) -> str:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    texto = ""
    for pagina in doc:
        texto += pagina.get_text()  # Extrai o texto de cada página
    return texto
