import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

if __name__ == '__main__':
    sentence = """At eight o'clock on Thursday morning Arthur didn't feel very good."""
    print(len(sentence))
    search = sentence.count("Thursday")/len(sentence) * 100
    print(search)
    tokens = nltk.word_tokenize(sentence)
    print(tokens)
    tagged = nltk.pos_tag(tokens)
    entities = nltk.chunk.ne_chunk(tagged)
    print(entities)
