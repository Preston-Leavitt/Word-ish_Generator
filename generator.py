import random
import csv
from flask import Flask, request, render_template
CSV_ONE = "place.csv"
CSV_TWO = "relative.csv"

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



# ===== Flask Web App =====
app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return render_template("app.html", prediction=None)

@app.route("/predict", methods=["POST"])
def guess():
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
            max = random.randint(7,10)
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
                    char2 = by_relative(num,relative)
                else:
                    char2 = char1
                if x==0 and sLetter:
                    char1 = sLetter
                    char2 = sLetter
                if help == True:
                    char1 = char2
            char_array.append(char1)
    reconstructed = ''.join(char_array)
    return render_template("app.html", word = reconstructed)






if __name__ == '__main__':
    app.run(debug=False, port=5000)