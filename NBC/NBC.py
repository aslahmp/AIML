import pandas as pd

file_path=r'C:\Users\anuar\OneDrive\Desktop\nbc_data.xlsx'
df=pd.read_excel(file_path)

print("Dataset preview:")
print(df)

df["text"]=df["text"].str.replace(r'[^a-zA-z ]','',regex=True)
df["text"]=df["text"].str.replace('Subject','')
df["text"]=df["text"].str.lower()

print("Processed dataset:")
print(df)

messages=df['text']
labels=df['label']

spam_messages=messages[labels=='spam']
ham_messages=messages[labels=='not spam']

print("\nspam messages")
print(spam_messages)
print("\nham messages")
print(ham_messages)

spam_word_count={}
ham_word_count={}

for m in spam_messages:
    words = str(m).split()
    for word in words:
        if word in spam_word_count:
            spam_word_count[word] += 1
        else:
            spam_word_count[word] = 1

for m in ham_messages:
    words = str(m).split()
    for word in words:
        if word in ham_word_count:
            ham_word_count[word] += 1
        else:
            ham_word_count[word] = 1

print("\nspam words count")
print(spam_word_count)

print("\nham words count")
print(ham_word_count)

print("\nFROM THE GIVEN DATA SET WE CAN FIND THE FOLLOWING")

print("\npriori probability of spam mails")
priori_p_spam=len(spam_messages)/len(messages)
print(priori_p_spam)
print("\npriori probability of ham mails")
priori_p_ham=len(ham_messages)/len(messages)
print(priori_p_ham)

print("\nEnter the mail!")
mail=input()

words = mail.lower().split()
print(f"\nWords in the mail:{words}")

class_conditional_p_spam = 1
class_conditional_p_ham = 1

for word in words:
    p_word_given_spam = ((spam_word_count.get(word, 0) + 1) /
                         (spam_word_count.get(word, 0) + ham_word_count.get(word, 0) + 1))
    class_conditional_p_spam *= p_word_given_spam

    p_word_given_ham = ((ham_word_count.get(word, 0) + 1) /
                        (spam_word_count.get(word, 0) + ham_word_count.get(word, 0) + 1))
    class_conditional_p_ham *= p_word_given_ham

    k = spam_word_count.get(word, 0)
    g = ham_word_count.get(word, 0)
    print(f"\ncount of the word '{word}' in spam mails")
    print(k)
    print(f"count of the word '{word}' in ham mails")
    print(g)

    print(f"\nClass conditional probabilities of word:{word}")
    print(f"Probability(word/spam)={p_word_given_spam}")
    print(f"Probability(word/ham)={p_word_given_ham}")



posterior_p_spam_given_word = ((priori_p_spam*class_conditional_p_spam)/
                               (priori_p_spam*class_conditional_p_spam+priori_p_ham*class_conditional_p_ham))

posterior_p_ham_given_word = ((priori_p_ham * class_conditional_p_ham) /
                                   (priori_p_spam * class_conditional_p_spam + priori_p_ham * class_conditional_p_ham))

print("\nPosterior probability of this mail to be a spam is")
print(posterior_p_spam_given_word)
print("\nPosterior probability of this mail to be a ham is")
print(posterior_p_ham_given_word)

if posterior_p_spam_given_word > posterior_p_ham_given_word:
    print("\nHence the mail is a spam!")
else:
    print("\nHence the mail is not a spam!")