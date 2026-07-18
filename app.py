# ======================================================
#               Bibliotecas Utilizadas
# ======================================================

import streamlit as st
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import FreqDist
from nltk.sentiment import SentimentIntensityAnalyzer
from deep_translator import GoogleTranslator

# =====================================================
#      DOWNLOAD DOS RECURSOS NECESSÁRIOS DO NLTK
# =====================================================
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("vader_lexicon")

# =====================================================
#               TÍTULO DA APLICAÇÃO
# =====================================================
st.title("🧠 Análise de Sentimentos e Processamento de Linguagem Natural (NLTK)")

# Campo para o usuário informar o texto
texto = st.text_area("Digite um texto em português:")

# Botão para iniciar as análises
if st.button("Executar"):

    # Normaliza o texto deixando todas as letras minúsculas
    texto = texto.lower()

    # =====================================================
    # ATIVIDADE 1 - TOKENIZAÇÃO
    # Divide o texto em palavras individuais
    # =====================================================
    st.header("1. Tokenização")

    tokens = word_tokenize(texto)
    st.write(tokens)

    # =====================================================
    # ATIVIDADE 2 - FREQUÊNCIA DAS PALAVRAS
    # Identifica quais palavras aparecem mais vezes
    # =====================================================
    st.header("2. Frequência das Palavras")

    freq = FreqDist(tokens)
    st.write(freq.most_common())

    # =====================================================
    # ATIVIDADE 3 - REGRA CONDICIONAL
    # Procura palavras negativas para priorizar atendimento
    # =====================================================
    st.header("3. Mensagem Prioritária")

    if "ruim" in texto or "péssimo" in texto or "erro" in texto:
        st.write("Mensagem prioritária.")
    else:
        st.write("Mensagem normal.")

    # =====================================================
    # ATIVIDADE 4 - STOPWORDS
    # Remove palavras que não ajudam na análise
    # =====================================================
    st.header("4. Remoção de Stopwords")

    stop = stopwords.words("portuguese")

    palavras_filtradas = [
        palavra for palavra in tokens
        if palavra not in stop
    ]

    st.write(palavras_filtradas)

    # =====================================================
    # ATIVIDADE 5 - SENTIMENTO POR CONDICIONAL
    # Analisa sentimento usando palavras-chave
    # =====================================================
    st.header("5. Sentimento por Condicional")

    positivas = [
        "bom",
        "ótimo",
        "excelente",
        "gostei",
        "maravilhoso"
    ]

    negativas = [
        "ruim",
        "péssimo",
        "horrível",
        "terrível"
    ]

    if any(palavra in texto for palavra in positivas):
        st.write("Sentimento: Positivo")

    elif any(palavra in texto for palavra in negativas):
        st.write("Sentimento: Negativo")

    else:
        st.write("Sentimento: Neutro")

    # =====================================================
    # ATIVIDADE 6 - CHATBOT
    # Identifica o setor responsável pelo atendimento
    # =====================================================
    st.header("6. Direcionamento do Atendimento")


    if "cancelar" in texto:
        st.write("Setor: Cancelamento")

    elif "erro" in texto:
        st.write("Setor: Suporte Técnico")

    elif "pagamento" in texto:
        st.write("Setor: Financeiro")

    else:
        st.write("Setor: Atendimento Geral")

    # =====================================================
    # ATIVIDADE 7 - PALAVRAS MAIS FREQUENTES
    # Mostra as cinco palavras mais utilizadas
    # =====================================================
    st.header("7. Top 5 Palavras")

    st.write(freq.most_common(5))

    # =====================================================
    # ATIVIDADE 8 - CLASSIFICAÇÃO DE MENSAGENS
    # Classifica mensagens por palavras específicas
    # =====================================================
    st.header("8. Classificação")


    if "erro" in texto or "sistema" in texto:
        st.write("Categoria: Suporte Técnico")

    elif "boleto" in texto or "pagamento" in texto:
        st.write("Categoria: Financeiro")

    else:
        st.write("Categoria: Outro Setor")

    # =====================================================
    # ATIVIDADE 9 - LIMPEZA DO TEXTO
    # Remove pontuação e mantém apenas palavras
    # =====================================================
    st.header("9. Texto Limpo")

    texto_limpo = texto

    for simbolo in string.punctuation:
        texto_limpo = texto_limpo.replace(simbolo, "")

    st.write(texto_limpo)

    # =====================================================
    # ATIVIDADE 10 - TOKENIZAÇÃO + CONDICIONAL
    # Analisa sentimento usando os tokens gerados
    # =====================================================
    st.header("10. Sentimento com Tokens")


    if "ruim" in tokens or "péssimo" in tokens:
        st.write("Sentimento: Negativo")

    else:
        st.write("Sentimento: Positivo")

    # =====================================================
    # ANÁLISE DE SENTIMENTOS COM VADER + TRADUÇÃO
    # Traduz português para inglês para melhorar a análise
    # =====================================================
    st.header("Análise de Sentimentos com VADER")


    sia = SentimentIntensityAnalyzer()


    # Tradução automática português -> inglês
    texto_traduzido = GoogleTranslator(
        source="pt",
        target="en"
    ).translate(texto)

    # Análise do sentimento usando o texto traduzido
    resultado = sia.polarity_scores(texto_traduzido)
    compound = resultado["compound"]

    # Define o sentimento final
    if compound >= 0.05:
        sentimento = "Positivo"

    elif compound <= -0.05:
        sentimento = "Negativo"

    else:
        sentimento = "Neutro"

    # Exibição dos resultados
    st.write("Texto traduzido:", texto_traduzido)
    st.write("Sentimento:", sentimento)
    st.write("Compound:", compound)
    st.write(resultado)
