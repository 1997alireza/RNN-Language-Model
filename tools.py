def read_data(path='data.txt'):
    data = open(path, 'r').read()
    chars = list(set(data))
    return data, chars
