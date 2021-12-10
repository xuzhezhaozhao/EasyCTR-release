import sys
import copy
import re

def chuli(datas,topnum,metatypes,metalines):
    #print(topnum)
    numkeys = 0
    newdatalines = []
    coun = 0
    for line in datas:
        tmpline = line.strip().split('\t')
        #realline = copy.deepcopy(tmpline)
        #print(tmpline)
        #print(tmpline)
        for i in range(2,len(tmpline)):
            if metatypes[i-2] == 'string_list':
                #print(metalines[i-2])
                tmplist = tmpline[i].strip().split(',')
                #print('Before chuli')
                #print(tmplist)
                if len(tmplist) == 1 and tmplist[0] == '':
                    #print('haha')
                    tmplist = ['-1']*topnum
                    tmpline[i] = '*'.join(tmplist)
                    #print('After chuli')
                    #print(tmplist)
                    continue

                if (len(tmplist)) >= topnum:
                    tmplist = tmplist[:topnum]
                    
                if bool(re.search('[a-zA-Z]',tmplist[0])):
                    for j in range(len(tmplist)):
                        if tmplist[j] not in keystonum:
                            numkeys += 1
                            keystonum[tmplist[j]] = numkeys
                            tmplist[j] = str(numkeys)
                        '''
                        if len(tmplist) < topnum:
                            for k in range(topnum-len(tmplist)):
                                tmplist.append('-1')
                            print('The %d, feature' %i)
                            print('not enough data')
                            print(tmplist)
                        '''
                        if len(tmplist) > topnum:
                            #print('ERror')
                            #print(tmplist)
                            sys.exit()
                if len(tmplist)<topnum:
                    for k in range(topnum-len(tmplist)):
                        tmplist.append('-1')
                    #print('The %d, feature' %i)
                    #print('not enough data')
                    
                #print('After Chuli')
                #print(tmplist)
                tmpline[i] = '*'.join(tmplist)
        #for debugging purposes
        #coun += 1
        #if coun == 5:
            #break
        #print(tmpline)
        newline = '\t'.join(tmpline)
        #print(len(newline))
        #print(newline)
        newdatalines.append(newline)
    return newdatalines

if __name__ == '__main__':

    datafile_tr = open('../data/data/train_080222.csv.clean','r')
    datafile_te = open('../data/data/test_080222.csv.clean','r')
    datalines_tr = datafile_tr.readlines()
    datalines_te = datafile_te.readlines()
    metafile = open('./data.meta','r')
    metalines = metafile.readlines()
    topnum = 3
    metatypes = {}
    keystonum = {}
    #numkeys = 0
    for i in range(len(metalines)):
        metatmp = metalines[i].strip().split(' ')
        mtype = metatmp[2]
        metatypes[i] = mtype

    # The commented code chunk below has been moved to the up.
    
    newdatalines_tr = chuli(datalines_tr,3,metatypes,metalines)
    print(len(newdatalines_tr))
    print('Train Data Passed')
    newdatalines_te = chuli(datalines_te,3,metatypes,metalines)
    print(len(newdatalines_te))
    print('Test Data Passed')
    newdatafile_tr = open('./../data/data/train_080222_withsl.csv','w')
    newdatafile_te = open('./../data/data/test_080222_withsl.csv','w')
    for line in newdatalines_tr:
        #print(len(line.strip().split('\t')))
        newdatafile_tr.writelines(line)
        newdatafile_tr.write('\n')

    for line in newdatalines_te:
        newdatafile_te.writelines(line)
        newdatafile_te.write('\n')


    # adjust meta according to the split of string lists
    is_strlist = 0
    for i in range(len(metalines)): 
        tmp = metalines[i].strip().split(' ')
        #print(tmp)
        mname = tmp[1]
        #print(mname)
        mtype = tmp[2]
        if mtype == 'string_list':
            is_strlist +=1
            metalines.pop(i)
            for j in range(1,topnum+1):
                tmpname = copy.deepcopy(mname)
                #print('tmpname in round %d'%j)
                #print(mname)
                iname = '%s_%d'%(mname,j)
                #print(iname)
                newtmp = tmp
                newtmp[1] = iname
                newtmp[2] = 'string'
                newline = ' '.join(newtmp)
                newline = newline+'\n'
                #print(newline)
                metalines.insert(i+j-1,newline)
        ## for debugging
        #if is_strlist == 2:
            #break
    meta_out = open('./data.meta.new','w')
    for line in metalines:
        meta_out.writelines(line)
    meta_out.close()
    metafile.close()

                 



