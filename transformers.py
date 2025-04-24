import re
import unicodedata
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.corpus import stopwords

# Transformador personalizado para limpiar texto
class CleanText(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.stop_words = set(stopwords.words("spanish"))

    def clean_text(self, texto):
        if not isinstance(texto, str):
            return ""

        # Lista de términos compuestos
        terminos_compuestos = [
            "acido salicilico",
            "acido hialuronico",
            "vitamina c",
            "aloe vera",
            "manteca de karite",
            "aceite de almendras",
            "aceite de argan",
            "aceite de coco",
            "aceite de ricino",
            "aceite de jojoba",
            "aceite de romero",
            "crecimiento capilar",
            "caida del cabello",
            "sal rosa"
        ]

        # Unir términos compuestos con guiones
        for termino in terminos_compuestos:
            texto = re.sub(
                r'\b' + re.escape(termino) + r'\b',
                termino.replace(" ", "-"),
                texto,
                flags=re.IGNORECASE
            )

        # Limpieza estándar (conserva guiones)
        texto = re.sub(r'https?://\S+|www\.\S+', '', texto)  # URLs
        texto = unicodedata.normalize("NFD", texto).encode("ascii", "ignore").decode("utf-8")  # Tildes
        texto = re.sub(r'[^\w\s-]', '', texto)  # Conserva guiones
        texto = texto.lower().strip()  # Minúsculas y espacios

        return texto

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [self.clean_text(texto) for texto in X]

from sklearn.base import BaseEstimator, TransformerMixin
import nltk
from nltk.corpus import stopwords

# Descargar stopwords de NLTK
nltk.download('stopwords')

class TokenizerText(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.stop_words = set(stopwords.words("spanish"))
        self.terminos_compuestos = {
            "acido-salicilico",
            "acido-hialuronico",
            "vitamina-c",
            "aloe-vera",
            "manteca-de-karite",
            "aceite-de-almendras",
            "aceite-de-argan",
            "aceite-de-coco",
            "aceite-de-ricino",
            "aceite-de-jojoba",
            "aceite-de-romero",
            "crecimiento-capilar",
            "caida-del-cabello"
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Debug: Verificar que terminos_compuestos está definido
        # print(f"terminos_compuestos: {self.terminos_compuestos}")

        # Dividir texto en tokens y filtrar stopwords
        return [
            ' '.join([
                token for token in texto.split()
                if token not in self.stop_words or token in self.terminos_compuestos
            ])
            for texto in X
        ]