import sys

string = []
while True:
    input_data = sys.stdin.readline().rstrip()
    if input_data == '':
        break
    else:
        string.append(input_data)

print('-'*30)
print()
for line in string:
    print(line, end=' ')
print()
print()
print('-'*30)