from ragas.metrics import faithfulness, answer_correctness, context_precision
from ragas.evaluation import evaluate
from langchain.schema import Document
import os

"""
RAGAS предлагает следующие оценочные баллы:

Верность: измеряет фактическую согласованность сгенерированного ответа по отношению к заданному контексту.
    Она рассчитывается на основе ответа и извлеченного контекста. Ответ масштабируется в диапазоне (0,1). Чем выше, тем лучше.

Релевантность ответа: фокусируется на оценке того, насколько сгенерированный ответ релевантен заданной подсказке. 
    Более низкий балл присваивается ответам, которые являются неполными или содержат избыточную информацию, 
    а более высокие баллы указывают на более высокую релевантность. Эта метрика вычисляется с использованием вопроса, контекста и ответа.

Полнота контекста: измеряет степень, в которой извлеченный контекст соответствует аннотированному ответу, рассматриваемому как истина. 
    Он вычисляется на основе истины и извлеченного контекста, а значения находятся в диапазоне от 0 до 1, причем более высокие значения
    указывают на лучшую производительность.

Точность контекста: оценивает, все ли элементы, соответствующие истине и присутствующие в контекстах, ранжированы выше. 
    В идеале все соответствующие фрагменты должны отображаться в верхних рангах. Эта метрика вычисляется с использованием 
    вопроса, истины и контекстов со значениями в диапазоне от 0 до 1, где более высокие баллы указывают на лучшую точность.

Релевантность контекста: измеряет релевантность извлеченного контекста, вычисляемую на основе как вопроса, так и контекстов. 
    Значения находятся в диапазоне (0, 1), причем более высокие значения указывают на лучшую релевантность.

Возврат объекта контекста: предоставляет меру возврата извлеченного контекста на основе количества объектов, 
    присутствующих как в истине, так и в контекстах, относительно количества объектов, присутствующих только в истине.

Метрики сквозной оценки
Также RAGAS предлагает две метрики сквозной оценки производительности конвейера RAG.

Семантическое сходство ответа: оценивает семантическое сходство между сгенерированным ответом и истинным значением. 
    Эта оценка основана на истинном значении и ответе, значения которого находятся в диапазоне от 0 до 1.

Корректность ответа: включает в себя измерение точности сгенерированного ответа по сравнению с истинным значением. 
    Эта оценка основана на истинном значении и ответе, значения которого находятся в диапазоне от 0 до 1.

"""
os.environ["OPENAI_API_KEY"] = "your api key"

# Пример документов (контекст RAG)
documents = [
    Document(page_content="RAG (Retrieval-Augmented Generation) - это метод, который улучшает генеративные модели, добавляя внешний контекст из базы данных."),
    Document(page_content="Библиотека ragas позволяет оценивать качество системы RAG с помощью различных метрик.")
]

# Вопросы и ответы модели (симуляция работы RAG)
test_samples = [
    {"question": "Что такое RAG?", "answer": "RAG - это метод, который использует внешний контекст для генерации ответов.", "contexts": [documents[0]]},
    {"question": "Какие метрики использует ragas?", "answer": "Он использует метрики верности, точности ответа и релевантности контекста.", "contexts": [documents[1]]}
]

# Оценка качества RAG
results = evaluate(
    test_samples,
    [faithfulness, answer_correctness, context_precision]
)

print(results)
