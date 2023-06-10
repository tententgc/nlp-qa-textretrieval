# auto correct an entire sentence
# we will exlude the question out at the last stage since we will not recap our answer
from pythainlp.spell import correct # tokenize data and then perfor correction would be faster
import pythainlp
import deepcut
from pythainlp.tokenize import word_tokenize, newmm
from pythainlp.tokenize import word_tokenize as th_word_tokenize
from pythainlp.util import keywords, normalize
from tqdm import tqdm
import torch
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from mosestokenizer import MosesTokenizer, MosesDetokenizer
from fairseq.models.transformer import TransformerModel
from functools import partial
import re

model_name_or_path = "google/flan-t5-base"
tokenizer_name_or_path = "google/flan-t5-base"
checkpoint = "bigscience/mt0-large"

classifier = pipeline("zero-shot-classification",
                      model="joeddav/xlm-roberta-large-xnli",
                      device_map="auto",
                      torch_dtype=torch.float16,
                      # load_in_8bit=True
                      )

th2en_word2word = TransformerModel.from_pretrained(
                    model_name_or_path='/home/user/small_bot_test_chitchat/small_bot_test_chitchat/SCB_1M+TBASE_th-en_newmm-moses_130000-130000_v1.0/models/',
                    checkpoint_file='/home/user/small_bot_test_chitchat/small_bot_test_chitchat/SCB_1M+TBASE_th-en_newmm-moses_130000-130000_v1.0/models/checkpoint.pt',
                    data_name_or_path='/home/user/small_bot_test_chitchat/small_bot_test_chitchat/SCB_1M+TBASE_th-en_newmm-moses_130000-130000_v1.0/vocab/',
                    device_map="auto",
                    torch_dtype=torch.float16,
                    # load_in_8bit=True
)

en2th_word2bpe = TransformerModel.from_pretrained(
                    model_name_or_path='/home/user/small_bot_test_chitchat/small_bot_test_chitchat/SCB_1M+TBASE_en-th_moses-spm_130000-16000_v1.0/models/',
                    checkpoint_file='/home/user/small_bot_test_chitchat/small_bot_test_chitchat/SCB_1M+TBASE_en-th_moses-spm_130000-16000_v1.0/models/checkpoint.pt',
                    data_name_or_path='/home/user/small_bot_test_chitchat/small_bot_test_chitchat/SCB_1M+TBASE_en-th_moses-spm_130000-16000_v1.0/vocab/',
                    device_map="auto",
                    torch_dtype=torch.float16,
                    # load_in_8bit=True
)

model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, # mt0
                                              device_map="auto",
                                              torch_dtype=torch.float16, 
                                              # load_in_8bit=True
                                              )

en_word_tokenize = MosesTokenizer('en')

en_word_detokenize = MosesDetokenizer('en')

th_word_tokenize = partial(th_word_tokenize, keep_whitespace=False)

tokenizer = AutoTokenizer.from_pretrained(checkpoint) # mt0

def remove_stopwords(sentence):
  # remove uncessary word from Pantip
  stop_words = ['(', ')', '\\', '/', '-']
  for word in stop_words:
    if word in sentence:
      sentence = sentence.replace(word,"")
  return sentence

  # by using regex
  # clean_sentence = re.sub(r'[()\\/\']', '', sentence)
  # return clean_sentence

def spell_correction(input_text):
  correct_txt = []
  # token list must be tokenized
  token_list = word_tokenize(input_text, engine = "deepcut")
  for idx, word in enumerate(token_list):
    # if word in "Q:":
    #   # since we will include just only an answer part
    #   correct_txt = []
    # elif word in ["A:"," "]:
    #   correct_txt.append(word)
    # else:
    correct_txt.append(correct(word))

  # then, remove stopword
  ans = remove_stopwords("".join(correct_txt))
  
  return ans

def get_question_type(question_set):
  candidate_labels = ["คำถาม", "บอกเล่า"]
  #question_class = classifier(question_set, candidate_labels)

  if type(question_set) != list:
    process_question_set = []
    process_question_set.append(question_set)
    question_set = process_question_set

  q_type = []
  question_class = classifier(question_set, candidate_labels)
  for question_no in range(len(question_class)):
    qtype = question_class[question_no]['labels'][0] # where 0 is the most possible question type
    q_type.append(qtype)
  
  return q_type

  # else:
  #   # in case that user input just only one sentence each
  #   q_type = []
  #   question_class = classifier(question_set, candidate_labels)
  #   q_type.append(question_class['labels'][0])
  #   return q_type



# def translate_th_to_en(question):

#   if type(question) != list:
#     process_question = []
#     process_question.append(question)
#     question = process_question

#   translated_list = []
#   for q in tqdm(range(len(question))):
#     input_sentence = question[q]
#     tokenized_sentence = ' '.join(th_word_tokenize(input_sentence))

#     # translate and then detokenize the token
#     _hypothesis = th2en_word2word.to('cuda').translate(tokenized_sentence)
#     hypothesis = en_word_detokenize([_hypothesis])

#     translated_list.append(hypothesis)

#   return translated_list


def translate_en_to_th(generated_list):
  # this part is an answer for text2text generation model
  answer_gen = []

  # translate the english sentence bacl to thai sentence
  for t in tqdm(range(len(generated_list))):
    input_sentence = generated_list[t]
    tokenized_sentence = ' '.join(en_word_tokenize(input_sentence))

    hypothesis = en2th_word2bpe.to('cuda').translate(tokenized_sentence)

    hypothesis = hypothesis.replace(' ', '').replace('▁', ' ').strip()
    answer_gen.append(remove_stopwords(hypothesis))
  
  return answer_gen


# def generate_ans(translated_list, q_type):

#   # prompt format 1 -> Q: How about a trip this evening? Give the rationale before answering.
#   # prompt format 2 -> Answer the following question by reasoning step-by-step. How about a trip this evening?
#   # prompt format 3 -> Explain why with step-by-step reasoning

#   generated_list = []

#   for t in tqdm(range(len(translated_list))):

#     # with the predicted topic, then, we can guide the prompt into the model
#     if q_type[t] == "คำถาม":
#       input_text = f"Q: {translated_list[t]} Give the rationale before answering."
#     else:
#       input_text = f"Answer the following question by reasoning step-by-step. {translated_list[t]}"

#     # using MT0 instead
#     inputs = tokenizer.encode(input_text, return_tensors="pt").to("cuda")
#     outputs = model.generate(
#       inputs,
#       top_k=50, 
#       num_beams=5, 
#       no_repeat_ngram_size=2, 
#       early_stopping=False, 
#       top_p=2, 
#       temperature=1.0,
#       max_new_tokens=100,
#       min_new_tokens=50,
#       do_sample=True,
#       use_cache=True,
#     )

#     gen = tokenizer.decode(outputs[0], skip_special_tokens=True)

#     # input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to('cuda')

#     # # we can adjust the generator parameters here
#     # outputs = model.generate(
#     #       input_ids,
#     #       top_k=100, 
#     #       num_beams=40, 
#     #       no_repeat_ngram_size=2, 
#     #       early_stopping=True, 
#     #       #top_p=0.7, 
#     #       temperature=2.0,
#     #       max_new_tokens=200,
#     #       min_new_tokens=50,
#     #       do_sample=True,
#     #       use_cache=True,
#     #       #epsilon_cutoff=3e-4,
#     #     )

#     # gen = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     generated_list.append(gen)

#   return generated_list

def generate_ans(question, q_type=[]):

  generated_list = []
  question_list = []
  question_list.append(question)
  with torch.no_grad():
    for t in tqdm(range(len(question_list))):
      # if q_type[t] == "คำถามแบบอธิบาย":
      #   input_text = f"Answer this in English language : Q: {question[t]} A:"
      # else:
      #   input_text = f"{question[t]} Explain why with step-by-step reasoning in English."
      input_text = f"Answer this in English language : Q: {question_list[t]} A:"
      #input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

      inputs = tokenizer.encode(input_text, return_tensors="pt").to("cuda")
      outputs = model.generate(
          inputs,
          top_k=50, 
          num_beams=5, 
          no_repeat_ngram_size=2, 
          # early_stopping=False, 
          # top_p=2, 
          temperature=1.0,
          max_new_tokens=100,
          min_new_tokens=50,
          # do_sample=True,
          # use_cache=True,
      )

      gen = tokenizer.decode(outputs[0], skip_special_tokens=True)
      generated_list.append(gen)
      # print("this is " + question)

  return generated_list

def answering(question):

  # # get question type for guiding prompt
  # q_type = get_question_type(question)

  # # translate from thai lang to english
  # print("Translating from TH to EN ....")
  # translated_list = translate_th_to_en(question)

  # then, generate english ans according to prompt
  print("Generating EN Answer ....")
  # generated_list = generate_ans(translated_list, q_type)
  generated_list = generate_ans(question)

  # translate an english answer back to thai lang
  print("Translating from EN to TH ....")
  answer_gen = translate_en_to_th(generated_list)

  return answer_gen
