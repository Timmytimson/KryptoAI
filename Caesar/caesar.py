import os

def convert(dir_in, dir_out):
    dir_in = os.fsdecode(dir_in)
    dir_out = os.fsdecode(dir_out)
    for file in os.listdir(dir_in):
        filename = os.fsdecode(file)
        if filename.endswith(".txt"):
            print(filename)
            #TODO fertig automatisieren
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
    return ciphertext


def uncaesar(ciphertext, i=3):
    return caesar(ciphertext, -i)


"""
print("Start")
print(caesar("abc! xyz. 123"))
print(uncaesar(caesar("abc! xyz. 123")))
print("ABC! XYZ. 789")
print(uncaesar("ABC! XYZ. 789"))
"""
convert("./in", "./out")