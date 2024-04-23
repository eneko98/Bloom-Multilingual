import os
import csv
import re
import html
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_metric

# Set the correct GPU to use
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
checkpoint_path = "Training/results/bloom-all/checkpoint-500"

tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, use_fast=True, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(checkpoint_path, trust_remote_code=True)

rouge_metric = load_metric("rouge")
bertscore_metric = load_metric("bertscore")

def clean_up_english(text):
    pattern = re.compile(
        r'^((?:.*?[\.;!\?]){1,2})'
        r'(?:\s+([^.;!?]*\[[^\]]+\][^.;!?]*[\.;!\?]))?'
    )
    
    match = pattern.match(text)
    if match:
        return ''.join(filter(None, match.groups())).strip()
    else:
        return text.strip()

def clean_up_spanish(text):
    text = text.replace("||", ", ")
    text = re.sub(r'\((.*?)\)', lambda m: f', {m.group(1)}', text)
    cleaned_text_parts = re.split(r'\[EOS.*?\]', text, maxsplit=1)
    text = cleaned_text_parts[0] if cleaned_text_parts else text
    text = re.split(r'\[Categoría:.*', text, maxsplit=1)[0]
    return text.strip(", ").strip()

def clean_up_basque(text):
    text = html.unescape(text)
    text = re.sub(r'\[ETCn.*?\]', '', text)
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'\[CAB_es\].*$', '', text, flags=re.MULTILINE)
    sentences = re.split(r'(?<=[.])\s+', text, maxsplit=2)
    if len(sentences) > 2:
        text = ' '.join(sentences[:2])
    else:
        text = ' '.join(sentences)
    return text.strip(",.:; ")

def generate_response(model, tokenizer, lang, word, pos, max_length=100, top_p=0.9, temperature=0.9, num_beams=3, do_sample=True):
    lang_tag = f"[LANG_{lang}]"
    prompt = f"[BOS] {lang_tag} {word} (POS: {pos}) <definition>"
    
    encoding = tokenizer.encode_plus(prompt, return_tensors='pt', padding=True, truncation=True)
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    
    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        do_sample=do_sample,
        top_k=50,
        top_p=top_p,
        temperature=temperature,
        num_beams=num_beams,
        no_repeat_ngram_size=2,
        early_stopping=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    text = tokenizer.decode(output[0], skip_special_tokens=False)
    if lang == 'en':
        text = clean_up_english(text)
    elif lang == 'es':
        text = clean_up_spanish(text)
    elif lang == 'eu':
        text = clean_up_basque(text)
    
    definition_start = text.find("<definition>") + len("<definition>")
    definition = text[definition_start:].split("[EOS]", 1)[0].strip()

    return definition

csv_file_path = "Evaluation/evaluation_results.csv"

with open(csv_file_path, mode='a', newline='', encoding='utf-8') as file:
    csv_writer = csv.writer(file)
    if file.tell() == 0:
        csv_writer.writerow(["Language", "Word", "POS", "Generated Definition", "Expected Definition", "ROUGE-L F1", "BERTScore F1"])
    
    while True:
        lang = input("Enter the language (en, es, eu) or 'quit' to stop: ").lower()
        if lang == 'quit':
            break
        word = input("Enter the word: ")
        pos = input("Enter the part of speech: ")
        generated_definition = generate_response(model, tokenizer, lang, word, pos)
        print(f"Generated Definition: {generated_definition}\n")

        expected_definition = input("Enter the expected definition for metric calculation: ")

        rouge_score = rouge_metric.compute(predictions=[generated_definition], references=[expected_definition])
        bertscore_result = bertscore_metric.compute(predictions=[generated_definition], references=[expected_definition], lang=lang)

        print(f"ROUGE-L F1: {rouge_score['rougeL'].mid.fmeasure}")
        print(f"BERTScore F1: {bertscore_result['f1'][0]}")
        print("\n---\n")

        csv_writer.writerow([
            lang,
            word,
            pos,
            generated_definition,
            expected_definition,
            rouge_score['rougeL'].mid.fmeasure,
            bertscore_result['f1'][0]
        ])
        file.flush()

print(f"Results saved to {csv_file_path}")
