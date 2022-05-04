def printReturn(func):
    def decorate(*args, **kwargs):
        data = func(*args, **kwargs)
        print(f'The return data from {func.__name__} is:')
        print(f'data: {data}')
        print('----------------------------------------------------------')
        return data
    return decorate

def funcCall(func):
    def decorate(*args, **kwargs):
        print(f'[Log] Function {func.__name__} is called...')
        data = func(*args, **kwargs)
        return data
    return decorate