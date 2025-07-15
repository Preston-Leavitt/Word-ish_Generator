import random
import csv
import os
import torch
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments,
)
from flask import Flask, render_template, url_for, request
CSV_ONE = "place.csv"
CSV_TWO = "relative.csv"
parser = argparse.ArgumentParser(description="Fake definition generator")

parser.add_argument("--output_dir", type=str, default="fake_def_model")
parser.add_argument("--model_dir", type=str, default="fake_def_model")
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir).to(device)
print(f"Loaded model from {args.model_dir} on {device}")
def by_place(placeNum,place):
    n = 0
    for i in place[placeNum]:
        n += int(i)
    num = random.randint(1, n)
    p = 1
    n = 0
    for i in place[placeNum]:
        n += int(i)
        if (n >= num): 
            break
        else:
            p += 1
    return place[0][p-1]

def letter_to_num(char):
    char.lower()
    if char == 'a':
        return 1
    elif char == 'b':
        return 2
    elif char == 'c':
        return 3
    elif char == 'd':
        return 4
    elif char == 'e':
        return 5
    elif char == 'f':
        return 6
    elif char == 'g':
        return 7
    elif char == 'h':
        return 8
    elif char == 'i':
        return 9
    elif char == 'j':
        return 10
    elif char == 'k':
        return 11
    elif char == 'l':
        return 12
    elif char == 'm':
        return 13
    elif char == 'n':
        return 14
    elif char == 'o':
        return 15
    elif char == 'p':
        return 16
    elif char == 'q':
        return 17
    elif char == 'r':
        return 18
    elif char == 's':
        return 19
    elif char == 't':
        return 20
    elif char == 'u':
        return 21
    elif char == 'v':
        return 22
    elif char == 'w':
        return 23
    elif char == 'x':
        return 24
    elif char == 'y':
        return 25
    elif char == 'z':
        return 26
    elif char == 'A':
        return 1
    elif char == 'B':
        return 2
    elif char == 'C':
        return 3
    elif char == 'D':
        return 4
    elif char == 'E':
        return 5
    elif char == 'F':
        return 6
    elif char == 'G':
        return 7
    elif char == 'H':
        return 8
    elif char == 'I':
        return 9
    elif char == 'J':
        return 10
    elif char == 'K':
        return 11
    elif char == 'L':
        return 12
    elif char == 'M':
        return 13
    elif char == 'N':
        return 14
    elif char == 'O':
        return 15
    elif char == 'P':
        return 16
    elif char == 'Q':
        return 17
    elif char == 'R':
        return 18
    elif char == 'S':
        return 19
    elif char == 'T':
        return 20
    elif char == 'U':
        return 21
    elif char == 'V':
        return 22
    elif char == 'W':
        return 23
    elif char == 'X':
        return 24
    elif char == 'Y':
        return 25
    elif char == 'Z':
        return 26
    else:
        return 0
    
def by_relative(placeNum,relative):
    n = 0
    for i in relative[placeNum]:
        n += int(i)
    num = random.randint(1, n)
    p = 1
    n = 0
    for i in relative[placeNum]:
        n += int(i)
        if (n >= num): 
            break
        else:
            p += 1
    return relative[0][p-1]

def interactive_generate(model, tokenizer, device,word):
    print("\nEnter words to generate fake definitions. Type 'exit' to quit.")
    FEW = """
    {word} refers to
    """

    prompt = FEW.format(word=word)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    outputs = model.generate(
        input_ids,
        max_length=60,
        do_sample=True,
        top_k=50,
        top_p=0.92,
        temperature=0.8,
        num_return_sequences=1,
    )
    print("  " + tokenizer.decode(outputs[0], skip_special_tokens=True) + "\n")
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ===== Flask Web App =====
app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return render_template('app.html', song_url=None, pic_url=None)
@app.route("/predict", methods=["POST"])
def guess():
    song_folder = os.path.join(app.static_folder, 'songs')
    songs = [f for f in os.listdir(song_folder) if f.endswith('.ogg')]
    random_song = random.choice(songs)
    song_url = url_for('static', filename=f'songs/{random_song}')
    pic_folder = os.path.join(app.static_folder, 'images')
    pics = [f for f in os.listdir(pic_folder) if f.endswith('.jpg')]
    random_pic = random.choice(pics)
    pic_url = url_for('static', filename=f'images/{random_pic}')
    print(pic_url)
    sLetter = request.form.get("sLetter", "")
    length = request.form.get("length", "")
    if (sLetter):
        
        print(sLetter)
    else:
        print("Nuh Uh")
    if length:
        print (length)
    else:
        print("Nope")
    with open(CSV_ONE, "r") as file:
        reader = csv.reader(file)
        place = [[str(cell) for cell in row] for row in reader]
    with open(CSV_TWO, "r") as file:
        reader = csv.reader(file)
        relative = [[str(cell) for cell in row] for row in reader]
        if length:
            max = int(length)
        else:
            max = random.randint(2,10)
        char_array = []
       
        for x in range(max):
            char1 = 'a'
            char2 = 'b'
            help = False
            while char1 != char2:
                if(len(place) > x+1):
                    char1 = by_place(x+1,place)
                else:
                    help = True
                if(x!=0):
                    char = char_array[x-1]
                    num = letter_to_num(char)
                    print(char)
                    print(num)
                    char2 = by_relative(num,relative)
                else:
                    char2 = char1
                if x==0 and sLetter:
                    char1 = sLetter
                    char2 = sLetter
                if help == True:
                    char1 = char2
            char_array.append(char1)
            print(char_array)
    reconstructed = ''.join(char_array)
    definition = interactive_generate(model, tokenizer, device,reconstructed)
    return render_template("app.html", word = reconstructed, song_url=song_url, pic_url=pic_url, definition = definition)






if __name__ == '__main__':
    app.run(debug=False, port=5000)