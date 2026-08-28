#!/usr/bin/env python3
"""Script that takes in input from the user and prints a response,
exiting on a set of farewell keywords.
"""

if __name__ == '__main__':
    exit_words = ['exit', 'quit', 'goodbye', 'bye']

    while True:
        question = input('Q: ')

        if question.lower() in exit_words:
            print('A: Goodbye')
            break

        print('A:')
