for i in range(1, 11):
    print(str(i))

i = 1
while (i <= 100):
    print(str(i))
    i += 2

for i in range(0, 100, 2):
    print(str(i))

alphabet = "abcdefghijklmnopqrstuvwxyz"
for letter in alphabet:
    print(letter)

sentence1 = "Without fame, he who spends his time on earth leaves only such a mark" \
            " upon the world as smoke does on air or foamon water."
sentence2 = "The quick brown fox jumps over the lazy dog."

print("".join(sorted(sentence1.lower())))
print("".join(sorted(sentence2.lower())))
