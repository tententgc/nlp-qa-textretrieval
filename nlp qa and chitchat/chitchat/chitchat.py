import time, os
import pickle
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

class ChitChat():
    def __init__(self):
        ## Define Hyperparameter
        self.do_sample = True
        self.top_k = 50
        self.num_beams = 5               #Number of search path
        self.no_repeat_ngram_size = 4 
        self.early_stopping = True
        self.max_length = 200            #Maximum numbers of tokens to generate
        self.top_p = 0.95
        self.temperature = 1             #Randomness max:2
        self.num_return_sequences = 1  
        # # Load tokenizer and model
        # with open('chit-chat/tokenizer_th2eng_hel.pkl', 'rb') as f:
        #     tokenizer_th2eng_hel_loaded = pickle.load(f)
        # with open('chit-chat/tokenizer_eng2th_hel.pkl', 'rb') as f:
        #     tokenizer_eng2th_hel_loaded = pickle.load(f)

        # with open('chit-chat/model_th2eng_hel.pkl', 'rb') as f:
        #     model_th2eng_hel_loaded = pickle.load(f)
        # with open('chit-chat/model_eng2th_hel.pkl', 'rb') as f:
        #     model_eng2th_hel_loaded = pickle.load(f)
        
        # # Load tokenizer and model
        # with open('./chit-chat/tokenizer.pkl', 'rb') as f:
        #     tokenizer = pickle.load(f)

        # with open('./chit-chat/model.pkl', 'rb') as f:
        #     model= pickle.load(f)
        self.tokenizer_th2eng_hel_loaded  = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-th-en")

        self.model_th2eng_hel_loaded  = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-th-en")

        self.tokenizer_eng2th_hel_loaded  = AutoTokenizer.from_pretrained("Chayawat/opus-mt-en-mul-finetuned-en-to-th")

        self. model_eng2th_hel_loaded  = AutoModelForSeq2SeqLM.from_pretrained("Chayawat/opus-mt-en-mul-finetuned-en-to-th")
        
        tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
        
        model = AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-400M-distill")
        
        self.generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer,device=0)
        
    def th2eng_hel(self,th_text):
        input_ids = self.tokenizer_th2eng_hel_loaded.encode(th_text, return_tensors="pt", padding=True)

        outputs = self.model_th2eng_hel_loaded.generate(
            input_ids=input_ids,
            max_length=128,
            num_beams=2,
            early_stopping=True
        )

        output_text = self.tokenizer_th2eng_hel_loaded.decode(outputs[0], skip_special_tokens=True)
        return output_text

    def eng2th_hel(self,eng_text):
        input_ids = self.tokenizer_eng2th_hel_loaded.encode(eng_text, return_tensors="pt", padding=True)

        outputs = self.model_eng2th_hel_loaded.generate(
            input_ids=input_ids,
            max_length=128,
            num_beams=2,
            early_stopping=True
        )

        output_text = self.tokenizer_eng2th_hel_loaded.decode(outputs[0], skip_special_tokens=True)
        return output_text

    def generate(self,text):
        ## Translate Thai text to Eng text
        conver_in = self.th2eng_hel(text)                  #select translator -> vistec, fb, hel

        ##Inference text on model
        ans = self.generator(conver_in, do_sample=self.do_sample, top_k=self.top_k, num_beams=self.num_beams, no_repeat_ngram_size=self.no_repeat_ngram_size, 
            early_stopping=self.early_stopping, max_length= self.max_length, top_p=self.top_p, temperature=self.temperature, num_return_sequences=self.num_return_sequences)
        
        # output = []

        ## Translate Eng text to Thai text
        # for i in range(len(ans)):
        text_out = ans[0]
        conver_out = text_out['generated_text']
        result = self.eng2th_hel(conver_out)             #select translator -> vistec, fb, hel
        # output.append(result)

        return result, conver_in, conver_out

def main():
    chitchat= ChitChat()
    """
    INPUT HERE
    """
    text = input('You: ')
# -----------------------------------------------------------------------------
    start = time.time()
    
    
    """
    OUTPUT HERE
    """
    output, conver_in, conver_out = chitchat.generate(text)
# -----------------------------------------------------------------------------
    stop = time.time()
    print(f'You: {conver_in}')
    print(f'Bot:{conver_out}')
    print(f'Bot: {output}')
    print(f'Response Time: {stop-start:.2f} S', '\n')
    
if __name__ == '__main__':
    main()