from transformers import pipeline

classifier = pipeline("sentiment-analysis", model="parinzee/xlm-chat-classify")

def classifier_data(text):
    return classifier(text)
    
    
if __name__ == "__main__":
    print(classifier_data("ทดสอบภาษาไทย"))