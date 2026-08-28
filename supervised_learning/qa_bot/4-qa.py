#!/usr/bin/env python3
"""Defines a function that answers questions from multiple reference
texts using semantic search and question answering.
"""
semantic_search = __import__('3-semantic_search').semantic_search
find_answer = __import__('0-qa').question_answer


def question_answer(corpus_path):
    """Answers questions from multiple reference texts

    Args:
        corpus_path: the path to the corpus of reference documents
    """
    exit_words = ['exit', 'quit', 'goodbye', 'bye']

    while True:
        question = input('Q: ')

        if question.lower() in exit_words:
            print('A: Goodbye')
            break

        reference = semantic_search(corpus_path, question)
        answer = find_answer(question, reference)

        if answer is None:
            print('A: Sorry, I do not understand your question.')
        else:
            print('A: {}'.format(answer))
