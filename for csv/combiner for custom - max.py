import os
import statistics

def log_csv(filename, values):
    line = ""
    for el in values:
        line = line + str(el) + ";"
    line = line[0:-1]
    with open(filename, "a") as f:
        f.write(line)
        f.close()

def write_to_file(outfile, averagetabs, finalvals, averageout):
    for i in range(len(averageout)):
        if averageout[i] == 1:
            finalvals[i] = str(max(averagetabs[i]))

    log_csv(outfile, finalvals)
    

directory = os.fsencode(".")
    
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".csv"): 
        print(filename)
        
        names = True
        keepnames = True
        averageout = [0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0] #for 27_
        #averageout = [0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0] #for 24_
        
        outfile = filename[:-4] + " merged.txt"
        file = open(filename, "r")
        first = True

        averagetabs = []
        finalvals = []
        sortval = ""
        for i in range(len(averageout)):
            averagetabs.append([])
            finalvals.append("X")

        for line in file:
            if names:
                names = False
                log_csv(outfile, [line])
                continue

            valsin = line.split(";")
            if not len(valsin) == len(averageout):
                print("LEN ERROR")
                print(valsin)
                print(averageout)
                print(len(valsin))
                print(len(averageout))
                exit(1)


            if not sortval == valsin[-1]:
                if first:
                    first = False
                    sortval = valsin[-1]                    
                else:
                    write_to_file(outfile, averagetabs, finalvals, averageout)
                    sortval = valsin[-1]
                    averagetabs = []
                    finalvals = []
                    for i in range(len(averageout)):
                        averagetabs.append([])
                        finalvals.append("X")
 

            for i in range(len(valsin)):
                if averageout[i] == 1:
                    averagetabs[i].append(float(valsin[i]))
                else:
                    finalvals[i] = str(valsin[i])

        write_to_file(outfile, averagetabs, finalvals, averageout)
        file.close()
        
