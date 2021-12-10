import sys


if __name__ == "__main__":
    #if len(sys.argv) != 3:
        #print('Invalid number of arguments')
        #sys.exit()

    #inp = sys.argv[1]
    #oup = sys.argv[2]

    f = open('./../data/feature/feature_importance_072509_withid.csv','r')
    out = open('./../data/feature/feature_importance_072509_wid_s.csv','w')
    f = open(inp,'r')
    out = open(oup,'w')
    count = 0
    for line in f:
        if count == 0:
            count += 1
            continue
        line = line.strip().split(',')
        print(line)
        line[1] = '{:.7f}'.format(float(line[1]))
        out.write(line[0]+',')
        out.write(line[1])
        out.write('\n')
    out.close()
    f.close()

