# Built-in functions
text_length = len("Python syntax")
print(text_length)

user_input = input("Enter something: ")
print("You entered:", user_input)

# Conditional statements
if text_length > 10:
    print("Text length is greater than 10")
elif text_length == 10:
    print("Text length is exactly 10")
else:
    print("Text length is less than 10")

# Loops
for i in range(3):
    print("Iteration:", i)

while text_length > 0:
    print("Text length:", text_length)
    text_length -= 1