import os
import shutil


def convert(dir_in, dir_out, dir_move, rota=3):
    dir_in = os.fsdecode(dir_in)
    dir_out = os.fsdecode(dir_out)
    dir_move = os.fsdecode(dir_move)
    for file in os.listdir(dir_in):
        filename = os.fsdecode(file)
        if filename.endswith(".txt"):
            print(filename)
            f = open(dir_in + file, "r")
            text = f.read()
            ciphertext = caesar(text, rota)[0]
            f.close()
            #TODO move to converted
            #shutil.move(dir_in + file, dir_out + file)
            newfile = open(dir_out + str(rota) + "/" + filename, "w")
            newfile.write(ciphertext)
            newfile.close()
    return


def caesar(cleartext, i=3):
    ciphertext = ""
    for c in cleartext:
        ascii = ord(c)
        if 65 <= ascii <= 90:   # Großbuchstaben
            c = chr((ascii + i - 65) % 26 + 65)
        if 97 <= ascii <= 122:  # Kleinbuchstaben
            c = chr((ascii + i - 97) % 26 + 97)
        ciphertext += c
    return ciphertext, i


def uncaesar(ciphertext, i=3):
    return caesar(ciphertext, -i)


"""
print("Start")
print(caesar("abc! xyz. 123"))
print(uncaesar(caesar("abc! xyz. 123")))
print("ABC! XYZ. 789")
print(uncaesar("ABC! XYZ. 789"))
"""
convert("./in/", "./out/", "./converted/")
