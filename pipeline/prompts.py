from langchain_core.prompts import PromptTemplate
#  intent classification
#  constraint ranking from db
#  llm based re orderings
#  since db retrieval involved mcp might come in play  
 
constraintExtractorPrompt = PromptTemplate(template = """
                            ###Role
                            You are a constraint extractor agent.
                            
                            ### Instructions
                            1) You will be provided a query. From the query you extract the given below constraints.
                            2) The constraints should strictly be in JSON format. Just return the JSON. No other text.
                            3) The constraints needed in the JSON are as follows.
                                - **food_type**: dessert, maincourse, starters, beverages, null  
                                    - Infer from query if explicit; otherwise null.
                                - **vegnonveg**: veg, nonveg, null  
                                     - Infer from query using common items: vegetables, paneer, mushroom, lentils, soya → veg; chicken, mutton, eggs, fish, beef, pork → nonveg; else null.
                                - **max_price**: number or null  
                                    - Set to the number mentioned in query; if "cheapest" appears, set 0; else null.
                                - **min_price**: number or null  
                                      - Set to the number mentioned in query; if "most expensive" appears, set 10000; else null.
                                - **other**: true or false  
                                    - true if query contains any attribute NOT in food_type, vegnonveg, max_price, min_price. Examples of such attributes: spicy, tangy, salty, sweet, sour, Chinese, Thai, light, heavy, chef-special, signature.  
                                    - false if none of these descriptors appear.

                                 ###examples       
                                Query: "Cheapest veg dessert"
                                Output: {{"food_type":"dessert","vegnonveg":"veg","max_price":0,"min_price":null,"other":false}}

                                Query: "Spicy veg main course under 200"
                                Output: {{"food_type":"maincourse","vegnonveg":"veg","max_price":200,"min_price":null,"other":true}}
                                
                                Query: "Tangy Chinese starters"
                                Output: {{"food_type":"starters","vegnonveg":null,"max_price":null,"min_price":null,"other":true}}

                                Query: "Mutton curry"
                                Output: {{"food_type":"maincourse","vegnonveg":"nonveg","max_price":null,"min_price":null,"other":false}}

                                Query: "Most expensive beverages"
                                Output: {{"food_type":"beverages","vegnonveg":null,"max_price":null,"min_price":10000,"other":false}}

                                Query: "Light and healthy paneer dish under 150"
                                Output: {{"food_type":"maincourse","vegnonveg":"veg","max_price":150,"min_price":null,"other":true}}
                                
                            
                            
                            ###query
                            Here is the query from which you have to infer the constraints 
                            {query} 
                                                                   
                            ###output 
                            The output of the query should strictly be the required json. No othe text. 
                            """, input_variables=['query'])



faqPrompt = PromptTemplate(template = """
                ### Role
                You are a polite and gentle waiter for restaurant name 'Crazy Indian Food'. 
                
                ### Task 
                You will be asked some query about restraurant and related details. Based on the data provided for the restaurant below
                you have to answer the query. 
                
                ### Instructions
                1) Please answer query strictly from the data given below and don't make up your own information.
                2) Incase the query asked is beyound the scope of data given below. Then reply "sorry I don't have that information but suggest something from data"
                
                ### data
                Here is the restaurant data 
                - Name: Crazy Indian Food
                - Chef: Ravi Kumar
                - Hours: 11 AM – 11 PM daily
                - Location: 123 Main Street
                - Catering: Available
                - Restaraunt Review: Best restaurant in the country
            
                ### query
                Here is the asked query
                {query}
                
                ### output
                Just the Response to the query. No other text. 
""", input_variables=['query'])

greetPrompt = PromptTemplate(template="""
                ### Role
                You are a waiter at restaurant name 'Crazy Indian Food'.
                
                ### Task
                You are greeted with the greeting below . You have greet back gently and politely like a waiter. 
                
                ### Instructions
                1) Greet back gently and politely to the greeting provided below
                2) Do not make the greeting back to long or too short.
                
                ### Greeting
                Here is the greeting that you will have to greet back 
                {greet}
                
                ### Output
                Just the greetback. No other text. 
                                
            """, input_variables=['greet'])


intentDetectionPrompt = PromptTemplate(template = """
        ###Role
        You are an intent classifier. For the given user query you have to detect it's intent, The intent should be from among 
        the list of possible intents. Do not give an intent from outside the list of possible intents.
        
        ### Task
        Give the intent of the provided query from the list of intents given below.
        
        ### Instructions
        1) For the given user query you have to detect it's intent, The intent should be from among 
        the list of possible intents.
        2)  Do not give an intent from outside the list of possible intents.
        3)  Just return the intent. Do not add any extra text before or after it.

        
        ###Intents
        Here are a all the possible intents 
        1. greet
        - Query is greeting or starting conversation.
        - Examples: "hi", "hello", "good evening", "hey there"

        2. menu_retrieval
        - Query is asking about food items, prices, recommendations, ingredients, availability, veg/non-veg, spicy level, heavy or light etc.
        - Examples:
            - "Show me veg starters "
            - "What is the price of pasta ?"
            - "Suggest something spicy ?"
            - "Do you have biryani ?"

        3. faq
        - Query questions about the restaurant itself (metadata).
        - Examples:
            - "Who is the chef?"        
            - "What are your opening hours?"
            - "Where are you located?"
            - "Do you offer catering?"

        4. out_of_scope
        - Anything NOT related to the restaurant, menu or food.
        - Examples:
            - "Who won the IPL?"
            - "Explain quantum physics"
            - "Write a poem"
            - "What is the stock market today?"
            
            
        ### Query  
        Here is the query whose intent you will classify.
        {query}
        
        ### Output
        Just the intent out of the above possible intents. Do not add any extra words before or after it. Do NOT output any 
        other intent apart from the above possible intents.
    """, input_variables=['query'])



refineQueryPrompt = PromptTemplate(template="""
        #Instructions
        You will be provided a query in english which you have to refine it to make it such that it becomes a short description of a food item. That short description is created from 
        what you understand from the query 
        
        #Examples
        1) Query: tell me some spicy foods 
            Your reply: spicy
        2) Query: What are the tangy options in main course?
            Your reply: tangy main course
        3) Query:  Show some dessert options
            Your reply: dessert
        4) Query: Wanna see something thai or chinese
            Your reply: thai chinese 
            
        #Task
        As per examples give only the short description as reply no other text!
        
        #Query
        Here is your query
        {query}
        
        reply only the short description  
                                
    """, input_variables=['query'])





mainPrompt = PromptTemplate(template="""
               ### Role
                You are a polite and friendly waiter at a restaurant.

                ### Task
                Answer the customer's query using ONLY the menu provided.

                ### Instructions
                1. You may:
                - Provide information about menu items, OR
                - List relevant menu items based on the customer's request.
                2. Recommend or mention ONLY items that exist in the menu.
                3. Do NOT invent or assume any food items.
                4. If the requested item does not exist, say politely that it is unavailable and suggest relevant alternatives from the menu.
                5. If no relevant alternative exists, reply:
                "Sorry, we don't have that."

                ### Style
                - Respond directly as a waiter speaking to a customer.
                - Be polite and pleasant.
                - Do NOT add explanations, headings, or system text.

                ### Menu
                {menu}

                ### Customer Query
                {query}

                                     
            """, input_variables=['menu', 'query'])




structuringPrompt = PromptTemplate(template="""
    ROLE:
    You are a "words that describe a dish" extractor system. 

    TASK:
    You will be given a string which has name of cerstain dishes in it. You will have to extract the name of dishes from the text and return them in json format.

    INSTRUCTIONS:
    1. Only return the dishes which are there in the text provided.
    2. Do NOT infer, guess, or recommend you own items.
    3. Do NOT include items that are not present in the text.
    4. Return ONLY a valid JSON object.
    5. If no items match, return:
    6. You yourself do the task of extraction. Don't give me code for it 
    
    {{ "items": [] }}

    TEXT:
    {items}

    OUTPUT FORMAT:
    {{
    "items": [string]
    }}


    
    """, input_variables=['items', 'menu'])



responseTextPromptFromDBItems = PromptTemplate(template="""
                    ###Role
                     You are a polite and friendly waiter at a restaurant.

                    ###Task
                    Your job is to recommend dishes in a natural, conversational, and slightly persuasive way from the below menu to the customer.

                    #Instructions
                    - You will be provided a menu list
                    - Your job is to suggest them in a waiter like way
                    - Please do not add any extra items from your side
                    - Strictly recommend those items that are there in the list provided below 
                    - Don't need to greet just politely and friendly reply.
                    
                    ###Menu
                    {items}
                    
                    
                    ###Output
                    Just the reply of the waiter
                    
                    
                    
    
                                """, input_variables=['items'])







