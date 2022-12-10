import os
import shutil


def test(dir_out, dir_move, rota=3):
    dir_out = os.fsdecode(dir_out)
    dir_move = os.fsdecode(dir_move)
    equal_strings = []
    unequal_strings = []

    for file in os.listdir(dir_move):
        if is_equal(file, dir_out, dir_move, rota):
            equal_strings.append(file)
        else:
            unequal_strings.append(file)

    print(str(len(equal_strings)) + " Dateien stimmen überein.\n")
    print(str(len(unequal_strings)) + " Fehler aufgetreten:")
    for name in unequal_strings:
        print(name)
    return


def is_equal(file, dir_out, dir_move, rota=3):
    filename = os.fsdecode(file)
    dir_out += "/" + str(rota) + "/"
    if filename.endswith(".txt"):
        f = open(dir_move + file, "r")
        text = f.read()
        cleartext = ""
        f.close()
        for f2 in os.listdir(dir_out):
            if os.fsdecode(f2) == ("3_"+filename):
                f2 = open(dir_out + f2, "r")
                cleartext = uncaesar(f2.read(), rota)[0]
                f2.close()
        if text == cleartext:
            return True
        else:
            return False


def convert(dir_in, dir_out, dir_move, rota=3):
    dir_in = os.fsdecode(dir_in)
    dir_out = os.fsdecode(dir_out)
    dir_move = os.fsdecode(dir_move)
    count = 0

    for file in os.listdir(dir_in):
        filename = os.fsdecode(file)
        if filename.endswith(".txt"):
            f = open(dir_in + file, "r")
            text = f.read()
            ciphertext = caesar(text, rota)[0]
            f.close()
            shutil.move(dir_in + file, dir_move + file)
            new_dir_out = dir_out + str(rota) + "/"
            if not(os.path.exists(new_dir_out)):
                os.mkdir(new_dir_out)
            newfile = open(new_dir_out + str(rota) + "_" + filename, "w")
            newfile.write(ciphertext)
            newfile.close()
            count += 1
    print("Verarbeitung vollständig. " + str(count) + " Dateien verschlüsselt.\n")
    return


def caesar(cleartext, i=3):
    ciphertext = ""
    for c in cleartext:
        ascii = ord(c)
        # Großbuchstaben
        if 65 <= ascii <= 90:
            c = chr((ascii + i - 65) % 26 + 65)
        # Kleinbuchstaben
        if 97 <= ascii <= 122:
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
test("./out/", "./converted/")
