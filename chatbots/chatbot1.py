# Test

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
import transformers


inference_model = "bardsai/jaskier-7b-dpo-v5.6"

# Loads model
#def get_llm(model = "bardsai/jaskier-7b-dpo-v5.6"):
#    llm = AutoModelForCausalLM.from_pretrained(inference_model,
#                                             device_map="auto",
#                                             trust_remote_code=False,
#                                             revision="main")
#    return llm


def get_llm(model_path, tokenizer_path):
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Set up a pipeline for QA
    qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

    # Wrap the pipeline in a langchain LLM
    llm = HuggingFacePipeline(pipeline=qa_pipeline)

    return llm


def chatty_bot(llm, vectorstore):
    
    # set up question/answer scheme
    qa = ConversationalRetrievalChain.from_llm(
        llm, 
        vectorstore.as_retriever(), 
        return_source_documents = True)
    
    def user(user_message, history):
        print("user message", user_message)
        print("chat history:", history)

        # re format history as needs to be tuple
        chat_history_tuples = []
        for m in history:
            chat_history_tuples.append((m[0], m[1]))

        # Get result from QA chain
        response = qa({'question': user_message, 'chat_history': chat_history_tuples})   
    
        # Update history with the new Q&A pair
        history.append((user_message, response["answer"]))
        
        return gr.update(value=""), history
    
    with gr.Blocks() as chatty:
        chatbot = gr.Chatbot(elem_id="chatbot")
        msg = gr.Textbox()
        clear = gr.Button("Clear")

        msg.submit(user, inputs=[msg, chatbot], outputs=[msg, chatbot], queue=False)

        clear.click(lambda: None, None, chatbot, queue=False)

    chatty.launch()

if __name__ == "__main__":
    llm = get_llm(model_path = "inference_model",
        tokenizer_path = "infernce_model")
    vectorestore = 