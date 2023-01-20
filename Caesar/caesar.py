import os
import shutil
import random
import re


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
            splitted_text = split(text)
            count_of_words = len(splitted_text)
            for i in range(0, 26):
                new_dir_out = dir_out + str(i) + "/"
                if not(os.path.exists(new_dir_out)):
                    os.mkdir(new_dir_out)
                iterator = 0
                newfile = open(dir_out + str(i) + "_" + filename, "a")
                for word in splitted_text:
                    print("Rotate by " + str(i) + ", word " + str(iterator) + " out of " + str(count_of_words))
                    ciphertext = caesar(word, i)[0]
                    ciphertext += '\n'
                    iterator += 1
                    newfile.write(ciphertext)
                newfile.close()
            f.close()
            shutil.move(dir_in + file, dir_move + file)

            count += 1
    print("Verarbeitung vollständig. " + str(count) + " Dateien verschlüsselt.\n")
    return


def generate_training_and_test_data_at_random(dir_in, dir_test, dir_training):
    dir_test += "/"
    dir_training += "/"
    for file in os.listdir(dir_in):
        file_training = open(dir_training + os.fsdecode(file), 'a')
        file_testing = open(dir_test + os.fsdecode(file), 'a')
        for line in open(dir_in + '/' + file, 'r'):
            if random.randint(0, 1) == 0:
                file_training.write(line)
                print(line + " is training data")
            else:
                file_testing.write(line)
                print(line + " is test data")
        file_training.close()
        file_testing.close()
    return


def split(text):
    single_words = text.split()
    text = text.replace('!', '.')
    text = text.replace('?', '.')
    full_sentences = text.split('. ')
    text = text.replace(',', '.')
    text = text.replace(';', '.')
    text = text.replace('-', '.')
    text = text.replace(':', '.')
    partial_sentences = text.split('. ')

    for word in partial_sentences:
        single_words.append(word)
    for word in full_sentences:
        single_words.append(word)

    return remove_duplicates(single_words)


def remove_duplicates(list_of_words):
    return list(dict.fromkeys(list_of_words))


def mod(n, p):
    if n < 0:
        n = p - abs(n) % p

    return n % p


def caesar(cleartext, i=3):
    ciphertext = ""
    for c in cleartext:
        ascii = ord(c)
        # Großbuchstaben
        if 65 <= ascii <= 90:
            c = chr(mod((ascii + i - 65), 26) + 65)
        # Kleinbuchstaben
        if 97 <= ascii <= 122:
            c = chr(mod((ascii + i - 97), 26) + 97)
        ciphertext += c
    return ciphertext, i


def uncaesar(ciphertext, i=3):
    return caesar(ciphertext, -i)


"""str1 = caesar("abc! xyz. 123")[0]
print(str1)
print(uncaesar(str1)[0])
str2 = caesar("ABC! XYZ. 789")[0]
print(str2)
print(uncaesar(str2)[0])"""

convert("./in/", "./out/", "./converted/")
#test("./out/", "./converted/")
generate_training_and_test_data_at_random("./out", "./test data", "./training data")
