import os

dataset = "Digits"
output_file = "_full_output.txt"

directory = os.fsencode(".")
outfilename = dataset + output_file
outfile = open(outfilename, "w")
    
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith("_csv.txt"): 
        print(filename)
        file = open(filename, "r")
        start = True
        for line in file:
            if start:
                start = False
                continue
            #line = line.strip()
            try:
                values = line.split(";")
                if values[0] == dataset:
                    outfile.write(line)
                else:
                    break
            except:
                break
        file.close()
        
outfile.close()
