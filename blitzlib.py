def isEmpty(alist):
    try:
        for a in alist:
            if not isEmpty(a):
                return False
    except:
        # we will reach here if alist is not a iterator/list
        return False
    return True


def tic():
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()


def toc():
    import time

    if 'startTime_for_tictoc' in globals():
        print('\x1b[6;30;47m' + "[Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.]" + '\x1b[0m')

    else:
        print('\x1b[6;30;47m' + '\x1b[0m' "[Toc: start time not set]" + '\x1b[0m')



def my_uniq(seq, idfun=None):
    # order preserving
    if idfun is None:
        def idfun(x): return x
    seen = {}
    result = []
    for item in seq:
        marker = idfun(item)
        if seen.has_key(marker): continue
        seen[marker] = 1
        result.append(item)
    return result