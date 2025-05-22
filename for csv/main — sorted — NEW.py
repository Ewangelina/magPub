import os
import statistics

def log_csv(filename, dataset_name, kan_grid_real, kan_grid, kan_degree, kan_lamb, kan_width, kan_params, kan_time, kan_test_acc, mlp_test_acc, mlp_time, mlp_params, mlp_layers, mlp_width, mlp_test_acc2, mlp2_time_difference, mlp_params2, no_layers2, width_layers2, delme):
    line = dataset_name + ";" + kan_grid_real + ";" + kan_grid + ";" + kan_degree + ";" + kan_lamb + ";" + kan_width + ";"  + kan_params + ";" + kan_time + ";" + kan_test_acc + ";" + mlp_test_acc + ";" + mlp_time + ";" + mlp_params + ";" + mlp_layers + ";" + mlp_width + ";" + mlp_test_acc2 + ";" + mlp2_time_difference + ";" + mlp_params2 + ";" + no_layers2 + ";" + width_layers2 + ";" + delme + '\n'
    with open(filename, "a") as f:
        f.write(line)
        f.close()

directory = os.fsencode(".")
    
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    #if filename.endswith(" sorted.csv"):
    if filename.endswith("mnist sorted.csv"): 
        print(filename)
        #outfile = filename[0:-11] + " presorted.txt"
        names = False
        outfile = filename[0:-9] + " final2.txt"
        file = open(filename, "r")
        prevdelme = ""
        first = True
        

        tab_kan_time = []
        tab_kan_test_acc = []
        tab_mlp_test_acc = []
        tab_mlp_time = []
        tab_mlp_test_acc2 = []
        tab_mlp2_time_difference = []

        dataset_name = ""
        prev_kan_grid_real = ""
        prev_kan_grid = ""
        prev_kan_degree = ""
        prev_kan_lamb = ""
        prev_kan_width = ""
        prev_kan_params = ""
        prev_kan_time = ""
        prev_kan_test_acc = ""
        prev_mlp_test_acc = ""
        prev_mlp_time = ""
        prev_mlp_params = ""
        prev_mlp_layers = ""
        prev_mlp_width = ""
        prev_mlp_test_acc2 = ""
        prev_mlp2_time_difference = ""
        prev_mlp_params2 = ""
        prev_no_layers2 = ""
        prev_width_layers2 = ""

        for line in file:
            if names:
                names = False
                continue
            dataset_name, kan_grid_real, kan_grid, kan_degree, kan_lamb, kan_width, kan_params, kan_time, kan_test_acc, mlp_test_acc, mlp_time, mlp_params, mlp_layers, mlp_width, mlp_test_acc2, mlp2_time_difference, mlp_params2, no_layers2, width_layers2, delme = line.split(";")
            tab_kan_time.append(float(kan_time))
            tab_kan_test_acc.append(float(kan_test_acc))
            tab_mlp_test_acc.append(float(mlp_test_acc))
            tab_mlp_time.append(float(mlp_time))
            tab_mlp_test_acc2.append(float(mlp_test_acc2))
            tab_mlp2_time_difference.append(float(mlp2_time_difference))

            prev_kan_grid = kan_grid
            prev_kan_grid_real = kan_grid_real
            prev_kan_degree = kan_degree
            prev_kan_lamb = kan_lamb
            prev_kan_width = kan_width
            prev_kan_params = kan_params
            prev_mlp_params = mlp_params
            prev_mlp_layers = mlp_layers
            prev_mlp_width = mlp_width
            prev_mlp_params2 = mlp_params2
            prev_no_layers2 = no_layers2
            prev_width_layers2 = width_layers2
            prev_delme = delme[0:-1]

            if prevdelme == delme:
                continue
            else:
                prevdelme = delme
                if first:
                    first = False
                    continue
                
                avg_kan_time = statistics.mean(tab_kan_time)
                avg_kan_test_acc = statistics.mean(tab_kan_test_acc)
                avg_mlp_test_acc = statistics.mean(tab_mlp_test_acc)
                avg_mlp_time = statistics.mean(tab_mlp_time)
                avg_mlp_test_acc2 = statistics.mean(tab_mlp_test_acc2)
                avg_mlp2_time_difference = statistics.mean(tab_mlp2_time_difference)
                log_csv(outfile, dataset_name, kan_grid_real, prev_kan_grid, prev_kan_degree, prev_kan_lamb, prev_kan_width, prev_kan_params, str(avg_kan_time), str(avg_kan_test_acc), str(avg_mlp_test_acc), str(avg_mlp_time), prev_mlp_params, prev_mlp_layers, prev_mlp_width, str(avg_mlp_test_acc2), str(avg_mlp2_time_difference), prev_mlp_params2, prev_no_layers2, prev_width_layers2, prev_delme)

                tab_kan_time = []
                tab_kan_test_acc = []
                tab_mlp_test_acc = []
                tab_mlp_time = []
                tab_mlp_test_acc2 = []
                tab_mlp2_time_difference = []
                tab_kan_time.append(float(kan_time))
                tab_kan_test_acc.append(float(kan_test_acc))
                tab_mlp_test_acc.append(float(mlp_test_acc))
                tab_mlp_time.append(float(mlp_time))
                tab_mlp_test_acc2.append(float(mlp_test_acc2))
                tab_mlp2_time_difference.append(float(mlp2_time_difference))
                prev_kan_grid = kan_grid
                prev_kan_degree = kan_degree
                prev_kan_lamb = kan_lamb
                prev_kan_width = kan_width
                prev_kan_params = kan_params
                prev_mlp_params = mlp_params
                prev_mlp_layers = mlp_layers
                prev_mlp_width = mlp_width
                prev_mlp_params2 = mlp_params2
                prev_no_layers2 = no_layers2
                prev_width_layers2 = width_layers2
                prev_delme = delme[0:-1]
                
            
        file.close()
        
