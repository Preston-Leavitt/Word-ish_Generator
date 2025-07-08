
import csv
CSV_ONE = "place.csv"
CSV_TWO = "relative.csv"
FILENAME = "words_alpha.txt"
MAX_WORD_LENGTH = 20




def save_matrix_to_csv(matrix, filename):
    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(matrix)

def letter_to_num(char):
    if char == 'A':
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

def increment_cell(matrix, row_index, col_index):
    while len(matrix) <= row_index:
        matrix.append([0] * 26) 

    try:
        if (1 <= col_index <= 26):
            num = int(matrix[row_index][col_index - 1])
            num += 1
            matrix[row_index][col_index - 1] = num
    except:
        raise ValueError("Column index must be between 1 and 26")

def readFile():
    place = [
        ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]
    ]
    relative = [
        ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]

    ]
    with open(FILENAME, "r") as file:
        for line in file:
            for word in line.split():
                word = word.upper()
                print(word)
                chars = list(word)
                i = 0
                for char in chars:
                    if(i!=0):
                        Rnum1 = letter_to_num(char)
                        Rnum2 = letter_to_num(chars[i-1])
                        increment_cell(relative, Rnum2, Rnum1)

                    num = letter_to_num(char)              
                    increment_cell(place,i+1,num)
                    i = i + 1
    print(place)
    save_matrix_to_csv(place, CSV_ONE)
    save_matrix_to_csv(relative, CSV_TWO)






if __name__ == '__main__':
    readFile()