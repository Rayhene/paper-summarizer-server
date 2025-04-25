import nltk
import spacy
from collections import Counter
from nltk.tokenize import sent_tokenize

# Carregar o modelo spaCy para o português
nlp_spacy = spacy.load("pt_core_news_sm")
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

def resumir_local(texto: str, num_frases: int = 5) -> str:
    # Tokenização das sentenças
    sentencas = nltk.sent_tokenize(texto, language='portuguese')

    # Processamento de texto com spaCy
    doc = nlp_spacy(texto)

    # Filtrando as palavras relevantes
    palavras_importantes = [token.text.lower() for token in doc if token.is_alpha and not token.is_stop]
    frequencia = Counter(palavras_importantes)

    # Rankear as sentenças com base nas palavras mais frequentes
    ranking = {}
    for i, sent in enumerate(sentencas):
        for palavra in nltk.word_tokenize(sent.lower(), language='portuguese'):
            if palavra in frequencia:
                ranking[i] = ranking.get(i, 0) + frequencia[palavra]

    # Selecionar as frases mais importantes
    indices = sorted(ranking, key=ranking.get, reverse=True)[:num_frases]
    frases_resumo = [sentencas[i] for i in sorted(indices)]
    
    return ' '.join(frases_resumo)
