user = "gizynski.dominik@gmail.com"
lpmn="any2txt|wcrft2|wsd"
nlprestURL = "http://ws.clarin-pl.eu/nlprest2/base/process/"

import urllib2
# from dataEventsWro import toDate
import sys
import numpy as np
import json
import xml.etree.ElementTree as ET

words = []

def lpmnProcess(text):
    array = lpmnProcessSingle(text)
    for word in array:
        if word not in words:
            words.append(word)
    return array
    # data = {'lpmn':lpmn,'user':user,'text':text}
    # doc = json.dumps(data)
    # resp = urllib2.urlopen(urllib2.Request(nlprestURL,doc,{'Content-Type': 'application/xml'})).read()
    # # print(resp)
    # result = ET.fromstring(resp)
    # for sentence in result.iter('sentence'):
    #     for tok in sentence.findall('tok'):
    #         lex = tok.find('lex')
    #         base = lex.find('base').text
    #         ctag = lex.find('ctag').text
    #         if ctag == 'interp' or ctag == 'conj' or 'prep:' in ctag:
    #             continue
    #         if base not in words:
    #             words.append(base)
    # return words

def lpmnProcessSingle(text):
    array = []
    data = {'lpmn':lpmn,'user':user,'text':text}
    doc = json.dumps(data)
    resp = urllib2.urlopen(urllib2.Request(nlprestURL,doc,{'Content-Type': 'application/xml'})).read()
    # print(resp)
    result = ET.fromstring(resp)
    for sentence in result.iter('sentence'):
        for tok in sentence.findall('tok'):
            lex = tok.find('lex')
            base = lex.find('base').text
            ctag = lex.find('ctag').text
            if ctag == 'interp' or ctag == 'conj' or 'prep:' in ctag or 'adj:' in ctag:
                continue
            if base not in array:
                array.append(base)
    return array

# print(lpmnProcess("Ala ma kota kotu"))

def saveNewInputsClarin():
    dtn = np.string_
    names = np.loadtxt("eventsWroFiltered.csv", delimiter=",", dtype=dtn, usecols=[2])
    print(names.shape)
    print(names[0])
    # print(lpmnProcess(names[9]))
    # sys.exit()
    k = 1
    for i in names:
        lpmnProcess(i)
        k+=1
        # if k == 40:
        #     break
        print(k, i)
    print(words.__len__())

    np.savetxt("newInputs.csv", np.asarray(words), fmt='%s', delimiter=",")

def expandEventsFilteredWithClarin():
    dtn = np.string_
    csv = np.loadtxt("eventsWroFiltered.csv", delimiter=",", dtype=dtn, usecols=[0,1,2])
    print(csv.shape)
    print(csv[0])
    samples = []
    sample = []
    k = 1
    for i in csv:
        firstDay = i[0]
        lastDay = i[1]
        event = i[2]
        wordsArray = lpmnProcess(event)
        wordsString = ".".join(wordsArray)
        sample.append(firstDay)
        sample.append(lastDay)
        sample.append(event.decode('utf-8'))
        sample.append(wordsString)
        if wordsString != "":
            samples.append(sample)
        sample = []

        print(k, event)
        k+=1
        # if k == 401:
        #     break

    print(len(samples))
    np.savetxt("eventsWroFilteredClarin.csv", np.asarray(samples), fmt='%s', delimiter=",")

    np.savetxt("newInputs0.csv", np.asarray(words), fmt='%s', delimiter=",")

def main():
    expandEventsFilteredWithClarin()
    # print(lpmnProcess("Ala posiada przystojnego kota w kotu Estudiantes Twin Peaks ENEMEF Borussia"))

if __name__ == '__main__':
    main()

# <?xml version="1.0" encoding="UTF-8"?>
# <!DOCTYPE chunkList SYSTEM "ccl.dtd">
# <chunkList>
#  <chunk id="ch1" type="p">
#   <sentence id="s1">
#    <tok>
#     <orth>Ala</orth>
#     <lex disamb="1"><base>Al</base><ctag>subst:sg:gen:m1</ctag></lex>
#     <lex disamb="1"><base>Alo</base><ctag>subst:sg:gen:m1</ctag></lex>
#     <prop key="sense:ukb:syns_id">44973</prop>
#     <prop key="sense:ukb:syns_rank">44973/3861.3948432140 19218/2808.2367828538</prop>
#     <prop key="sense:ukb:unitsstr">obliczeniowa_sztuczna_inteligencja.1(6:umy) AL.1(6:umy) obliczenia_inteligentne.1(6:umy)</prop>
#    </tok>
#    <tok>
#     <orth>ma</orth>
#     <lex disamb="1"><base>moj</base><ctag>adj:sg:nom:f:pos</ctag></lex>
#     <prop key="sense:ukb:syns_id">9724</prop>
#     <prop key="sense:ukb:syns_rank">9724/6836.2122703515</prop>
#     <prop key="sense:ukb:unitsstr">moj.1(42:jak)</prop>
#    </tok>
#    <tok>
#     <orth>kota</orth>
#     <lex disamb="1"><base>kot</base><ctag>subst:sg:gen:m1</ctag></lex>
#     <prop key="sense:ukb:syns_id">5168</prop>
#     <prop key="sense:ukb:syns_rank">5168/1153.7902152892 34154/1131.5803080437 406482/814.2826927054 405249/797.3057302360 227074/791.1602261788 227075/780.0571662630 75697/763.4500264867</prop>
#     <prop key="sense:ukb:unitsstr">kot.1(21:zw)</prop>
#    </tok>
#   </sentence>
#  </chunk>
# </chunkList>