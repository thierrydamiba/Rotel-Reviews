from openai import OpenAI
import requests
import streamlit as st
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import json

def getResponse(query):

    url = "https://api-ares.traversaal.ai/live/predict"


    payload = { "query": [query] }
    headers = {
      "x-api-key": st.secrets["TRAVERSAAL"],
      "content-type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        # Get the JSON content from the response
        json_content = response.json()

        # Specify the file path where you want to save the JSON content
        return json_content
    else:
        print(response.status_code)
        return " "


class NeuralSearcher:
    def __init__(self, collection_name):
        self.collection_name = collection_name
        # Initialize encoder model
        self.model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        # initialize Qdrant client
         # self.qdrant_client = QdrantClient("http://localhost:6333")
        self.qdrant_client = QdrantClient(
            url="https://ed55d75f-bb54-4c09-8907-8d112e6278a1.us-east4-0.gcp.cloud.qdrant.io",
            api_key=st.secrets["QDRANT_API_KEY"],
        )

    def search(self, text: str):
        # Convert text query into vector
        vector = self.model.encode(text).tolist()

        #
        # city_of_interest = "Berlin"
        # # Define a filter for cities
        # city_filter = Filter(**{
        #     "must": [{
        #         "key": "country", # Store city information in a field of the same name 
        #         "match": { # This condition checks if payload field has the requested value
        #             "value": "London" 
        #         }
        #     }]
        # })
        #
        # Use `vector` for search for closest vectors in the collection
        search_result = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=vector,
            query_filter=None,  # If you don't want any filters for now
            limit=3,  # 5 the most closest results is enough
        )
        # `search_result` contains found vector ids with similarity scores along with the stored payload
        # In this function you are interested in payload only
        payloads = [hit.payload for hit in search_result]
        return payloads




def initializeClient():
    return OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def decode(hotel_description, query):
    client = initializeClient();
    prompt = f"""
    this is the hotel descriptoin:

    \"{hotel_description}\"

     and these are my requirements

    \"{query}\"

    now tell me why the hotel might be a good fit for me given the requirements, make it consise.
    """

    stream = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    str = ""
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            str += (chunk.choices[0].delta.content)
    return str
    




def canAnswer(description, prompt):
    client = initializeClient();

    prompt = f"""
    \"{description}\"
    \n
    given the above information can you answer the following question : {prompt}
    \n
    answer in one word, "yes" or "no"
    """
    stream = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    str = ""
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            str += (chunk.choices[0].delta.content)
    return str.lower() == "yes";
    


def home_page():
    # st.title("TraverGo")

    st.markdown("<h1 style='text-align: center; color: white;'>TraverGo</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: white;'>Find any type of Hotel you want !</h2>", unsafe_allow_html=True)
    st.session_state["value"] = None

    def search_hotels():
        query = st.text_input("Enter your hotel preferences:", placeholder ="clean and cheap hotel with good food and gym")

        if "load_state" not in st.session_state:
            st.session_state.load_state = False;

        # Perform semantic search when user submits query
        if query or st.session_state.load_state:
            st.session_state.load_state=True;
            neural_searcher = NeuralSearcher(collection_name="hotel_descriptions")
            results = sorted(neural_searcher.search(query), key=lambda d: d['sentiment_rate_average'])
            st.subheader("Hotels")
            for hotel in results:
                explore_hotel(hotel, query)  # Call a separate function for each hotel

    def explore_hotel(hotel, query):
        if "decoder" not in st.session_state:
            st.session_state['decoder'] = [0];

        button = st.checkbox(hotel['hotel_name'])


        if not button:
            if st.session_state.decoder == [0]:
                x = (decode(hotel['hotel_description'][:1000], query))
                st.session_state['value_1'] = x
                st.session_state.decoder = [st.session_state.decoder[0] + 1]
                st.write(x)

            elif (st.session_state.decoder == [1]):
                x = (decode(hotel['hotel_description'][:1000], query))
                st.session_state['value_2'] = x

                st.session_state.decoder = [st.session_state.decoder[0] + 1];
                st.write(x);

            elif st.session_state.decoder == [2]:
                x = (decode(hotel['hotel_description'][:1000], query))
                st.session_state['value_3'] = x;
                st.session_state.decoder = [st.session_state.decoder[0] + 1];
                st.write(x);


            if (st.session_state.decoder[0] >= 3):
                i = st.session_state.decoder[0] % 3
                l = ['value_1', 'value_2', 'value_3']
                st.session_state[l[i - 1]];
                st.session_state.decoder = [st.session_state.decoder[0] + 1];

        if button:
            # if "messages" in st.session_state:
            #     st.session_state.messages = [];
            st.session_state["value"] = hotel


        # if (st.session_state.decoder[0] < 3):
        #     st.write(decode(hotel['hotel_description'][:1000], query))
        #     st.session_state.decoder = [st.session_state[0] + 1];
        #

        question = st.text_input(f"Enter a question about {hotel['hotel_name']}:");
            
        if question:
            st.write(ares_api(question + " - " + hotel['hotel_name'] + "located in" + hotel['country']))






    search_hotels()
    chat_page()


def ares_api(query):
    response_json = getResponse(query);
    # if response_json is not json:
    #     return "Could not find information"
    return (response_json['data']['response_text'])
def chat_page():
    hotel = st.session_state["value"]
    st.session_state.value = None
    if (hotel == None):
        return;

    st.write(hotel['hotel_name']);
    st.title("Conversation")

    # Set OpenAI API key from Streamlit secrets
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    # st.session_state.pop("messages")
    # Set a default model
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo"

    prompt = f"{hotel['hotel_description'][:2000]}\n\n you are a hotel advisor now, you should give the best response based on the above text. i will now ask you some questions get ready. If you don't know the answer then try to connect it with something that is given in the text above. don't ever mention anything about not having sufficient infromation"
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "user", "content": prompt}]



    for message in st.session_state.messages[1:]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What is up?"):
        if (not canAnswer(hotel['hotel_description'][:2000], prompt)):
            print("GOING TO ARES API")
            x = ares_api(prompt + "for" + hotel['hotel_name'] + "located in" + hotel['country'])
            st.session_state.messages[0]['content'] = x + "\n" + st.session_state.messages[0]['content'];
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)



    #Display assistant response in chat message container
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        )
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})



home_page()