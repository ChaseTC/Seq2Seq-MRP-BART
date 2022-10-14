from enum import Enum
import re
import logging

logger = logging.getLogger(__name__)

class TokenType(Enum):
    L_BRACKET = 0
    R_BRACKET = 1
    VARIABLE = 2
    ROLE = 3
    END = 4

    def __repr__(self):
        return self.name
    
    def __str__(self):
        return self.name

class Token():
    def __init__(self, type, value):
        self.type = type
        self.value = value
    
    def from_list(l):
        tokens = []
        prev = None
        for t in l:
            if t == '(':
                token = Token(TokenType.L_BRACKET, '(')
            elif t == ')':
                token = Token(TokenType.R_BRACKET, ')')
            elif t.startswith(':'):
                token = Token(TokenType.ROLE, t)
            elif re.match(r'^<\d+-\d+>[-\w+]+$', t):
                token = Token(TokenType.VARIABLE, t)
            else:
                if prev == TokenType.VARIABLE and re.match(r'^[-\w+]+$', t):
                    token = tokens.pop()
                    token.value += t
                else:
                    logger.warning("Unkown token %s", t)
                    continue
            prev = token.type
            tokens.append(token)
        tokens.append(Token(TokenType.END, None))
        return tokens
    
    def __str__(self):
        return self.value

    def __repr__(self):
        return self.value

class Parser():
    def __init__(self, p_string):
        self.p_string = self.preprocess(p_string)

    def preprocess(self, penman):
        penman = penman.replace('(', ' ( ').replace(')', ' ) ').replace('> ', '>').replace(' >', '>').replace('< ', '<')
        return re.sub(r"\s(?=\d+>)", '-', penman)
    
    def parse(self):
        new_string = []
        expected = [TokenType.L_BRACKET]
        tokens = Token.from_list(self.p_string.split())
        l_count = 0
        for i, t in enumerate(tokens):
            if t.type in expected:
                if t.type == TokenType.L_BRACKET:
                    l_count += 1
                    expected = [TokenType.VARIABLE]
                elif t.type == TokenType.R_BRACKET:
                    l_count -= 1
                    expected = []
                    if i < len(tokens) - 3:
                        expected = [TokenType.ROLE]
                    if l_count > 0:
                        expected.append(TokenType.R_BRACKET)
                    if i == len(tokens) - 2:
                        expected = [TokenType.END]
                elif t.type == TokenType.VARIABLE:
                    expected = []
                    if i < len(tokens) - 3:
                        expected.append(TokenType.ROLE)
                    expected.append(TokenType.R_BRACKET)
                elif t.type == TokenType.ROLE:
                    expected = [TokenType.L_BRACKET]
                if t.type != TokenType.END:
                    new_string.append(t.value)
            else:
                action = ""
                log_expected = expected
                if TokenType.L_BRACKET in expected and t.type == TokenType.VARIABLE:
                    l_count += 1
                    new_string.append('(')
                    new_string.append(t.value)
                    expected = []
                    if i < len(tokens) - 2:
                        expected.append(TokenType.ROLE)
                    expected.append(TokenType.R_BRACKET)
                    action = "Adding left bracket"
                    logger.warning('expected %s at %d / %d got %s. %s', ', '.join([str(e) for e in log_expected]), i, len(tokens)-1, str(t.type), action)
                elif TokenType.R_BRACKET in expected and t.type == TokenType.ROLE:
                    l_count -= 1
                    new_string.append(')')
                    expected = [TokenType.END]
                    action = "Replacing role with right bracket"
                    logger.warning('expected %s at %d / %d got %s. %s', ', '.join([str(e) for e in log_expected]), i, len(tokens)-1, str(t.type), action)
                elif t.type == TokenType.END:
                    v = new_string.pop()
                    while v != ')':
                        v = new_string.pop()
                    new_string.append(')')
                    action = "Reverting to last valid node"
                    logger.warning('expected %s at %d / %d got %s. %s', ', '.join([str(e) for e in log_expected]), i, len(tokens)-1, str(t.type), action)

                
        new_string.append(')'*l_count)

        return ' '.join(new_string).replace('( ', '(').replace(' )', ')')