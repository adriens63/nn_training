from time import time






# *************** timers ****************

def timeit(method):

    def timed(*args, **kw):

        ts = time()
        result = method(*args, **kw)
        te = time()
        print('%r %2.2f sec' % \
            (method.__name__, te - ts)) # (method.__name__, args, kw, te-ts))
        print()
        return result

    return timed




class SpeedTest: 

    def __init__(self, testName=""): 

        self.funcName = testName 


    def __enter__(self): 

        print('Started: {}'.format(self.funcName)) 
        self.init_time = time() 
        return self 


    def __exit__(self, type, value, tb): 
        
        print('Finished: {} in: {:.4f} seconds'.format(self.funcName, time() - self.init_time)) 