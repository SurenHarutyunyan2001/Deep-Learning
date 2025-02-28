import openai

# Устанавливаем ваш API-ключ
openai.api_key = "YOUR_API_KEY"  # Замените на ваш реальный ключ

# Статический набор данных знаний
knowledge_base = """ 
Эйфелева башня находится в Париже, Франция. 
Она была завершена в 1889 году и имеет высоту 1083 фута. 
""" 

# Функция запроса с кэшированным контекстом 
def query_with_cag(context, query): 
    prompt = f"Context:\n {context} \n\nQuery: {query} \nAnswer:"
    response = openai.ChatCompletion.create( 
        model="gpt-4", 
        messages = [ {"role": "system", "content": "Вы помощник на основе искусственного интеллекта с экспертными знаниями."}, 
                     {"role": "user", "content": prompt}], 
        max_tokens = 50, 
        temperature = 0.2
    ) 
    return response['choices'][0]['message']['content'].strip() 

# Пример запроса 
print(query_with_cag(knowledge_base, "Когда была достроена Эйфелева башня?"))
