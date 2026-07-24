def longestpalindrome(s):
    a = ""
    for i in range(len(s)):
        for j in range(i, len(s)):
            word = s[i:j + 1]
            if word == word[::-1] and len(word) > len(a):
                a = word
    return a
text = input("Enter a string: ")
print(longestpalindrome(text))