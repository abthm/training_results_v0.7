import csv
import argparse
import statistics

op_list=[]

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logfile",
                    default=None,
                    type=str,
                    required=True,
                    help="The input data dir. Should contain .hdf5 files  for the task.")
    parser.add_argument("--csvfile",
                    default=None,
                    type=str,
                    required=True,
                    help="The input data dir. Should contain .hdf5 files  for the task.")
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    print(args)
    with open(args.logfile,'r') as f:
    #with open("./test.log",'r') as f:
        for line in f:
            if 'step_loss' in line:
                string = line.split()
                for i in range(len(string)):
                    if 'step_loss' in string[i]:
                        print(string[i+1][:-1])
                        op_list.append(string[i+1][:-1])
    print(op_list)
    op_list_float = [float(x) for x in op_list]
    #print("avg",statistics.mean(op_list_float[2:]))
    with open(args.csvfile, mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quoting=csv.QUOTE_MINIMAL,
                                        lineterminator='\n')

        csv_writer.writerow(['step_loss_mi100']) 
        for items in op_list:
                csv_writer.writerow([items]) 

if __name__ == "__main__":
    main()
