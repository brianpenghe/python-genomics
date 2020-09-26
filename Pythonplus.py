import inspect

def CheckSource(Function2Check):
    print( "".join(inspect.getsourcelines(Function2Check)[0]))
