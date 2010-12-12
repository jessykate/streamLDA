def print_topics(lambda_, topn):
    ''' prints the top n most frequent words from each topic in lambda '''
    topics = lambda_.num_topics
    for k in xrange(topics):
        this_topic = lambda_._topics[k]
        if topn < len(this_topic):
            printlines = topn
        else:
            printlines = len(this_topic)
    
        print 'Topic %d' % k
        print '---------------------------'
        for word in this_topic.keys(): # this_topic.items() is pre-sorted
            if printlines > 0:
                print '%20s  \t---\t  %.4f' % (word, this_topic.freq(word))
                printlines -= 1
            else:
                break
        print
        
