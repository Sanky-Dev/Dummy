# # import streamlit as st

# # # Define custom HTML and CSS
# # html_temp = """
# # <!DOCTYPE html>
# # <html lang="en">
# # <head>
# # <meta charset="UTF-8">
# # <title>Sunrise Scene</title>
# # <style>
# #     body {
# #         margin: 0;
# #         height: 100vh;
# #         background-color: #87CEEB; /* light blue sky */
# #         display: flex;
# #         justify-content: center;
# #         align-items: center;
# #         position: relative;
# #         overflow: hidden;
# #     }
# #     svg {
# #         position: absolute;
# #         bottom: 0; /* Position the sun at the bottom of the viewport */
# #         width: 100%;
# #         height: auto;
# #     }
# # </style>
# # </head>
# # <body>
# # <svg height="300px" width="100%" viewBox="0 0 800 300">
# #     <!-- Background circle for the sun -->
# #     <circle cx="400" cy="250" r="100" fill="#FFD700" stroke="#FFA500" stroke-width="10"/>
# #     <!-- Sun rays -->
# #     <g>
# #         <path d="M 400 150 L 400 0" stroke="#FFD700" stroke-width="20" />
# #         <path d="M 400 150 L 550 50" stroke="#FFD700" stroke-width="20" />
# #         <path d="M 400 150 L 550 250" stroke="#FFD700" stroke-width="20" />
# #         <path d="M 400 150 L 250 50" stroke="#FFD700" stroke-width="20" />
# #         <path d="M 400 150 L 250 250" stroke="#FFD700" stroke-width="20" />
# #         <path d="M 400 150 L 600 150" stroke="#FFD700" stroke-width="20" />
# #         <path d="M 400 150 L 200 150" stroke="#FFD700" stroke-width="20" />
# #     </g>
# #     <!-- Text path for the curved message -->
# #     <text x="50%" y="50%" font-size="24" font-family="Arial" fill="#FF4500" text-anchor="middle">
# #         <textPath xlink:href="#curve" startOffset="50%">
# #             AWS IS THE BEST
# #         </textPath>
# #     </text>
# #     <!-- Invisible curve for the text path -->
# #     <path id="curve" fill="transparent" d="M200,250 Q400,100 600,250" />
# # </svg>
# # </body>
# # </html>

# # """

# # # Display custom HTML
# # st.markdown(html_temp, unsafe_allow_html=True)
# # '''
# # st.markdown("""
# # <style>
# # .banner {
# #   display: flex;
# #   justify-content: center;
# #   align-items: center;
# #   background-color: #ff9900; /* AWS orange color */
# #   height: 200px;
# #   border-radius: 10px; /* Optional: for rounded corners of the banner */
# # }

# # .curved-text {
# #   font-size: 24px;
# #   font-weight: bold;
# #   color: white;
# #   position: relative;
# # }

# # .curved-text::before {
# #   content: '';
# #   position: absolute;
# #   width: 100%;
# #   height: 100%;
# #   background: inherit;
# #   border-radius: 50%;
# #   z-index: -1;
# # }
# # </style>
# # """, unsafe_allow_html=True)
# # '''

# # # Sidebar
# # st.sidebar.header("Sidebar")
# # st.sidebar.selectbox("Choose Option", ["Home", "About", "Contact"])

# # # Main content area
# # st.write("This is a simple Streamlit app demonstrating HTML and CSS styling.")

# # # User input
# # name = st.text_input("Enter your name")

# # # Button and response
# # if st.button("Greet"):
# #     st.write(f"Hello, {name}!")

# # # Display custom CSS by injecting it


# # # Session state to store the counts
# # if 'likes' not in st.session_state:
# #     st.session_state['likes'] = 0
# # if 'dislikes' not in st.session_state:
# #     st.session_state['dislikes'] = 0

# # col1, col2 = st.columns(2)

# # with col1:
# #     # Like button with thumbs up emoji
# #     if st.button('👍 Like'):
# #         st.session_state['likes'] += 1

# # with col2:
# #     # Dislike button with thumbs down emoji
# #     if st.button('👎 Dislike'):
# #         st.session_state['dislikes'] += 1

# # # Display the counts
# # st.write(f"Likes: {st.session_state['likes']}, Dislikes: {st.session_state['dislikes']}")




# import streamlit as st

# # Option 1: Using CSS for Basic Styling (Simpler)
# def button_with_css(text, button_class):
#   """Creates a button with custom CSS styling.

#   Args:
#       text: The text to display on the button.
#       button_class: The CSS class name to apply for styling.

#   Returns:
#       A Streamlit button element with the specified text and styling.
#   """
#   st.write(f'<button class="{button_class}">{text}</button>', unsafe_allow_html=True)

# # Option 2: Using a Custom Component Library (More Advanced)
# # Install streamlit-elements if you prefer this approach:
# # pip install streamlit-elements

# # from streamlit_elements import Elements

# # def create_beautiful_button(text, button_type="primary"):
# #   """Creates a beautiful button using the streamlit-elements library.

# #   Args:
# #       text: The text to display on the button.
# #       button_type: The type of button (e.g., "primary", "secondary", "danger").

# #   Returns:
# #       A button element created using streamlit-elements.
# #   """
# #   mt = Elements()
# #   return mt.button(text, size="md", buttonType=button_type)

# # Choose the button creation method you prefer (comment out the other)
# button_creator = button_with_css  # Option 1 (CSS)
# # button_creator = create_beautiful_button  # Option 2 (streamlit-elements)

# # Define your CSS classes (for Option 1)
# st.markdown("""
# <style>
# .beautiful-button {
#   background-color: #4CAF50; /* Green */
#   border: none;
#   color: white;
#   padding: 15px 32px;
#   text-align: center;
#   text-decoration: none;
#   display: inline-block;
#   font-size: 16px;
#   margin: 4px 2px;
#   cursor: pointer;
#   border-radius: 5px;
# }

# .beautiful-button:hover {
#   background-color: #45A049;
# }
# </style>
# """, unsafe_allow_html=True)

# # Create your beautiful buttons
# st.title("Streamlit App with Beautiful Buttons")
# button_creator("Click Me!", "beautiful-button")

# # Additional options using streamlit-elements (if chosen)
# # if using streamlit-elements:
# #   button_creator("Another Beautiful Button", "secondary")
# #   button_creator("Danger Button!", "danger")

# st.write("This is some content after the buttons.")

###########################################################################################
# '''
# import streamlit as st

# # HTML content with inline CSS for styling
# html_content = """
# <!DOCTYPE html>
# <html>
# <head>
# <style>
#   .banner {
#     width: 100%; /* Full width */
#     height: auto; /* Maintain aspect ratio */
#     background-image: url(./'aws-rocks.jpg'); /* Replace with your image URL */
#     background-size: cover;
#     text-align: center;
#     padding: 50px 0; /* Spacing above and below text */
#   }
#   h1 {
#     color: white; /* White text */
#     font-size: 2em; /* Large text */
#   }
# </style>
# </head>
# <body>
#   <div class="banner">
#     <h1>AWS BEDROCK ROCKS</h1>
#   </div>
# </body>
# </html>
# """

# # Display the HTML in your Streamlit app
# st.markdown(html_content, unsafe_allow_html=True)
# '''

# '''
# import streamlit as st

# # Example function that mimics what conn_insert might do
# def conn_insert(question, answer):
#     # Simulate inserting data into a database
#     pass

# user_question = "Example question?"
# reply = "Example answer."
# print("L")
# # Initialize session state if not already initialized
# if 'Yes' not in st.session_state:
#     st.session_state.Yes = 0

# if st.button('👍 Would You Like to Have Same Answer the next time you ask similar query ?', on_click=conn_insert, args=(user_question, reply)):
#     st.session_state.Yes += 1
#     print("Hiiiiiiiii")  # This will display the message in the Streamlit app
# '''

# import streamlit as st

# # if st.button("Click Me", on_click=st.balloons):
# #    st.toast('Your edited image was saved!', icon='😍')
# uploaded_file = st.file_uploader("Choose a file",accept_multiple_files=True )
# print(uploaded_file[0])

# import streamlit as st
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
# from langchain_community.embeddings import BedrockEmbeddings
# from langchain_community.vectorstores.faiss import FAISS
# from langchain_community.llms import Bedrock
# from langchain.memory import ChatMessageHistory, ConversationBufferMemory
# from langchain.chains import RetrievalQA, ConversationalRetrievalChain
# from langchain_community.chat_models import BedrockChat

# from langchain_core.prompts import PromptTemplate
# import boto3
# import numpy as np
# import streamlit as st 
# import sqlite3

# from langchain_community.document_loaders import UnstructuredURLLoader
# urls = [
#     "https://www.freechildrenstories.com/the-stellar-one-1",
# ]

# main_placeholder = st.empty()
# process_url_clicked = st.sidebar.button("Process URLs")

# if process_url_clicked:
#     # load data
#     loader = UnstructuredURLLoader(urls=urls)
#     main_placeholder.text("Data Loading...Started...✅✅✅")
#     data = loader.load()
#     # split data
#     text_splitter = RecursiveCharacterTextSplitter(
#         separators=['\n\n', '\n', '.', ','],
#         chunk_size=1000
#     )
#     main_placeholder.text("Text Splitter...Started...✅✅✅")
#     docs = text_splitter.split_documents(data)
#     # create embeddings and save it to FAISS index
#     embeddings = OpenAIEmbeddings()
#     vectorstore_openai = FAISS.from_documents(docs, embeddings)
#     main_placeholder.text("Embedding Vector Started Building...✅✅✅")
#     time.sleep(2)

#     # Save the FAISS index to a pickle file
#     with open(file_path, "wb") as f:
#         pickle.dump(vectorstore_openai, f)


# loader = UnstructuredURLLoader(urls=urls)
# data = loader.load()
# print(type(data[0]))
# # with open("urldata.txt", "w") as f:
# #     f.write(data)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.llms import Bedrock
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain_community.chat_models import BedrockChat
from langchain_core.prompts import PromptTemplate
import boto3
import numpy as np
import streamlit as st 
import sqlite3
import os

# Connect to SQLite database (or create it if it doesn't exist)
conn = sqlite3.connect('myapp1.db',check_same_thread=False)
# Create a cursor object using the cursor method of the connection
cursor = conn.cursor()
# SQL statement to create a table
def create_table():
    create_table_sql = '''
    CREATE TABLE IF NOT EXISTS data (
        id VARCHAR PRIMARY KEY,
        large_text CHAR(6000)
    );
    '''
    cursor.execute(create_table_sql)
    conn_commit()
# Execute the SQL statement to create the table
# Commit the changes

def conn_commit():
    conn.commit()
# Close the connection
def conn_close():
    conn.close()
###########
def conn_insert(query, reply):
    insert_sql = '''
    INSERT INTO data (id, large_text) VALUES (?, ?);
    '''
    # Data to be inserted
    data_to_insert = (query, reply)
    # Execute the insert command for each row
    cursor.execute(insert_sql, data_to_insert)
    conn_commit()
    st.toast("Your results are saved", icon='😍')

# Commit the changes
# Querying data from the table
def conn_select(query):
    select_sql = 'SELECT id, large_text FROM data where id = ?;'
    data = query,
    # Execute the query
    cursor.execute(select_sql, data)
    results = cursor.fetchall()
    #print(results)
    for row in results:
        #print("ID:", row[0], "Large Text:", row[1])
        pass
    return results

# Close the connection

# Print all rows
###########
create_table()
#print("Table Created Successfully")


bedrock = boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1",
                                       client=bedrock)


#Importing the PDF Papers

def data_load(uploaded_file):
    #loader = PyPDFDirectoryLoader(r"F:\LLM\chatbots\ResearchpaperQnA\Papers")
    #print(uploaded_file)
    final_docs = []
    for i in uploaded_file:
        loader = PyPDFLoader(i)
        documents = loader.load()
        final_docs.extend(documents)
    #print(len(final_docs))
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 10000,
                                                chunk_overlap=1000)
    docs = text_splitter.split_documents(final_docs)

    return docs

#Store data in vector Store

def get_vector(docs):
    db = FAISS.from_documents(docs,
                              bedrock_embeddings)
    db.save_local("faiss_db")

def call_Claude():
    llm = BedrockChat(model_id = "anthropic.claude-3-haiku-20240307-v1:0", client=bedrock, 
                  model_kwargs = {"max_tokens":1000})
    return llm

def call_llama():
    llm = Bedrock(model_id = "meta.llama2-70b-chat-v1",
                  client=bedrock,
                  model_kwargs={"max_gen_len":512, "temperature":0.5, "top_p":0.9})
    return llm


# prompt_template = """
# Human: Use the context provided below and answer in only 250 words, Do not generate answer if 
# you don't know , simply say "Sorry Don't Know"
# <context>{context}</context>
# Question: {question}
# Assistant: """
prompt_template = """
Human: Answer Questions from given context , If you don't know , simply say "Sorry Don't Know"
<context>{context}</context>
Question: {question}
Assistant: """

prompt = PromptTemplate(template=prompt_template,input_variables=["context", "question"])

def get_response(llm, db, query):
    qa = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type="stuff",
        retriever = db.as_retriever(
            search_type="similarity",
            search_kwargs={"k":1}
        ),
        return_source_documents = True,
        chain_type_kwargs={"prompt":prompt}
        )
    #print(prompt)
    answer = qa({"query":query})
    ##print("Answer",answer)
    return answer['result']

def user_q(user_question):
    if user_question:
        table_answer = conn_select(user_question)
#        print("Answer", table_answer)
        if not table_answer:
            return False        
        else:
            #print(table_answer[0][1])
            st.write(table_answer[0][1])
            st.success("Hooray!!! Results Above were fetched without calling model or creating any Vector Embedding.")
            st.success("This way causes very low/ negligible Latency and saves Money!!!", icon="🔥")
            return True


def app():
    st.set_page_config("Research Paper Bot")
    #st.markdown("<br><br><br>", unsafe_allow_html=True)
    #st.header("Please Choose an Action")
    footer_html = """
        <style>
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: light black;
            color: orange;
            text-align: center;
            padding: 10px;
            font-size: 20px;
            display: flex;
            justify-content: space-between;
        }
        .footer .right .name {
            position: relative;
            font-size: 20px; /* Different font size for the name */
            font-weight: bold;
        }
        .footer .right .name::after {
            content: '';
            position: absolute;
            left: 0;
            bottom: -3px; /* Adjust the position to fit under the text */
            width: 100%;
            height: 2px; /* Thickness of the line */
            background: #555;
            box-shadow: 0 4px #555; /* Creates the second underline */
        }
        .footer .center .name {
            position: relative;
            font-size: 20px; /* Different font size for the name */
            font-weight: bold;
        }
        .footer .center .name::after {
            content: '';
            position: absolute;
            left: 0;
            bottom: -3px; /* Adjust the position to fit under the text */
            width: 100%;
            height: 2px; /* Thickness of the line */
            background: #555;
            box-shadow: 0 4px #555; /* Creates the second underline */
        }




        .footer .left {
            font-family: 'Monotype Corsiva', sans-serif;
            font-weight: bold;
            margin-left: 5px;
            font-size: 40px;
        }
        .footer .right {
            font-family: 'Monotype Corsiva', sans-serif;
            margin-right: 50px;
            font-weight: bold;
            font-size: 40px;

        }
        .footer .center {
        text-align: center;
        flex-grow: 1;
        padding:18px;
        }
        </style>
        <div class="footer">
            <div class="left">
                <p></p>
                <p></p>
                <p></p>
                <p>Dedicated to Amazon, Mayur, Praveen and Rahul</p>
            </div>
            <div class="right">
                <p>Made with ❤️ by  <span class="name">  Sanket Dodya </span></p>
                <p>Using Streamlit, Amazon Bedrock and LLM's </p>
            </div>
        </div>
    """
# Inject the HTML into the Streamlit app
    st.image("Amazon.jpg" )
    main_placeholder = st.empty()
    sub_placeholder = st.empty()
    action = ["I wish to update the Vector Store", "I wish to Query Chatbot"]
    option = st.radio(
    "What Action you wish to Perform",
    options=action,
    index = 1
    #format_func=lambda x: action[x]
    )

    index = action.index(option)
    if index == 0:
        var = False
        with st.sidebar:            
            st.title("Upload New Files or Url to create new Vector Store")
            #st.header("Please Choose an Action")
            action = ["Upload PDF", "Paste a URL"]
            option = st.radio(
            "What Action you wish to Perform",
            options=action,
            index = 0
            #format_func=lambda x: action[x]
            )
            index = action.index(option)
            if index == 0:
                uploaded_file = st.file_uploader("Choose files", accept_multiple_files=True, type=['pdf'])
                #print(type(uploaded_file))
                # for i in uploaded_file:
                #     print(i)
                if st.button("Start Processing"):
                    #st.markdown(footer_html, unsafe_allow_html=True)
# Show the spinner using markdown inside the placeholder
                    main_placeholder.markdown("""
                        <style>
                        .loader {
                            border: 16px solid #f3f3f3; /* Light grey */
                            border-top: 16px solid #3498db; /* Blue */
                            border-radius: 50%;
                            width: 60px;
                            height: 60px;
                            animation: spin 2s linear infinite;
                        }
                        @keyframes spin {
                            0% { transform: rotate(0deg); }
                            100% { transform: rotate(360deg); }
                        }
                        </style>
                        <div class="loader"></div>
                    """, unsafe_allow_html=True)
                    temp = []
                    count = 0
                    data = []
                    for i in uploaded_file:
                        with st.spinner("Processing  data ...✅✅✅"):
                            temp_file = f"./temp{count}.pdf"
                            with open(temp_file, "wb") as file:
                                file.write(i.getvalue())
                                file_name = i.name
                            temp.append(temp_file)
                        count += 1
                    with st.spinner("Processing..."):
                        sub_placeholder.text("Data Loading Started.....")
                        docs = data_load(temp)
                        sub_placeholder.text("Data Loading Completed...✅✅✅")
                    #print(type(docs))
                    #print(docs)
                        
                        sub_placeholder.text("Vector Creations Beginning.....")
                        get_vector(docs)
                        var = True
                        st.success("Vector Store Created and is Ready for Use")
                        sub_placeholder.empty()
                        main_placeholder.empty()
                    for i in temp:
                        os.remove(i)
        if var:
            st.success("Vector Store Created and is Ready for Use")
        st.markdown(footer_html, unsafe_allow_html=True)
    else:
        user_question = st.text_input("Ask a query please")
        col3, col4 = st.columns(2)
        with col3:
            if st.button("Call Claude 3 Haiku from Amazon Bedrock"):
                with st.spinner("Processing..."):
                    if user_q(user_question):
                        pass
                    else:
                        faiss_index = FAISS.load_local("faiss_db", bedrock_embeddings,allow_dangerous_deserialization=True)
                        llm = call_Claude()
                        reply = get_response(llm,faiss_index,user_question)
                        st.write(reply)
                        st.success("Done")
                        if 'Yes' not in st.session_state:
                            st.session_state.Yes = 0
                        if 'Nope' not in st.session_state:
                            st.session_state.Nope = 0
                        col1, col2 = st.columns(2)
                        with col1:
                        # Like button with thumbs up emoji
                            if st.button('👍 Would You Like to Have Same Answer the next time you ask similar query ?', on_click=conn_insert,args=(user_question, reply)):
                                #print("hiiiii")              
                                pass
                        with col2:
                        # Dislike button with thumbs down emoji
                            if st.button('👎 Nope'):
                                st.session_state.Nope += 1

                    # Display the counts
                    #st.write(f"Likes: {st.session_state['likes']}, Dislikes: {st.session_state['dislikes']}")
        with col4:
            if st.button("Call Llama Model from Amazon BedRock"):
                
                with st.spinner("Processing..."):
                    if user_q(user_question):
                        pass
                    else:
                        faiss_index = FAISS.load_local("faiss_db", bedrock_embeddings, allow_dangerous_deserialization=True)
                        llm = call_llama()
                        reply = get_response(llm,faiss_index,user_question)
                        st.write(reply)
                        st.success("Done")
                        # Session state to store the counts
                        if 'Yes' not in st.session_state:
                            st.session_state.Yes = 0
                        if 'Nope' not in st.session_state:
                            st.session_state.Nope = 0

                        col1, col2 = st.columns(2)
                        #print("Before ")
                        with col1:
                            # Like button with thumbs up emoji
                            if st.button('👍 Would You Like to Have Same Answer the next time you ask similar query ?', on_click=conn_insert,args=(user_question, reply)):
                                st.session_state.Yes += 1
                                st.write("Hiiiiiiiii")
                        with col2:
                            # Dislike button with thumbs down emoji
                            if st.button('👎 Nope'):
                                st.session_state.Nope += 1

                            # Display the counts
                            #st.write(f"Likes: {st.session_state['likes']}, Dislikes: {st.session_state['dislikes']}")
                        #st.markdown(footer_html, unsafe_allow_html=True)
            
        st.markdown(footer_html, unsafe_allow_html=True)
        #print("here")
        #print("User_question", user_question) 



if __name__ == "__main__":
    app()





