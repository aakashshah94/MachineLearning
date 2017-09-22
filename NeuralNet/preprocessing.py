import sys
import math as np

class preprocessing:
    def __init__(self,data_path,output_path):
        self.data_path=data_path
        self.output_path=output_path
        self.num_records=0
    def readdata(self):
        test=open(self.data_path, 'r')
        testing=test.readline()
        test_list=[]
        rec = testing.strip()
        if (rec != " "):
            record = ' '.join(rec.split())
            test_list.append(record.split(','))
        if(len(test_list[0])>1):
            with open(self.data_path, 'r') as data:
                self.num_records = 0
                t_list = []
                for x in data.readlines():
                    record = x.strip()
                    record = ' '.join(record.split())
                    if (len(record.split())!=0):
                        t_list.append(record.split(','))
                        self.num_records = self.num_records + 1
        else:
            with open(self.data_path, 'r') as data:
                self.num_records = 0
                t_list = []
                for x in data.readlines():
                    record = x.strip()
                    record = ' '.join(record.split())
                    if(len(record.split())!=0):
                        t_list.append(record.split(" "))
                        self.num_records = self.num_records + 1
        self.num_records=self.num_records+1
        return t_list
    def checkfornull(self,t_list):
        n = len(t_list[0])
        t_column = [[0 for x in range(self.num_records)] for y in range(n)]
        for i in range(0, self.num_records - 1):
                for j in range(0, n):
                    if (t_list[i][j] == ' ' or t_list[i][j] is None):
                        t_list[i].remove
                        self.num_records=self.num_records-1
                    else:
                        t_column[j][i] = t_list[i][j].lstrip(' ')

        return t_column
    def standardize(self,t_list,t_column):
        n=len(t_list[0])

        sum = 0
        list = []
        for i in range(0, n):
            index = -1
            del list[:]
            if(t_column[i][0].isdigit()):

                sum = 0
                for j in range(0, self.num_records - 1):
                    sum = sum + float(t_column[i][j])
                mean = sum / (self.num_records - 1)
                sum = 0
                for j in range(0, self.num_records - 1):
                    sum = sum + (float(t_column[i][j]) - mean) * (float(t_column[i][j]) - mean)
                std = np.sqrt(sum / self.num_records)
                for j in range(0, self.num_records - 1):
                    t_column[i][j] = (float(t_column[i][j]) - mean) / std
            elif(t_column[i][0].replace('.', '', 1).isdigit()):
                sum = 0
                for j in range(0, self.num_records - 1):
                    sum = sum + float(t_column[i][j])
                mean = sum / (self.num_records - 1)
                sum = 0
                for j in range(0, self.num_records - 1):
                    sum = sum + (float(t_column[i][j]) - mean) * (float(t_column[i][j]) - mean)
                std = np.sqrt(sum / self.num_records)
                for j in range(0, self.num_records - 1):
                    t_column[i][j] = (float(t_column[i][j]) - mean) / std
            else:

                for j in range(0, self.num_records - 1):
                    if (t_column[i][j] not in list):
                        list.append(t_column[i][j])
                    index=list.index(t_column[i][j])
                    t_column[i][j] = index
        return t_column
    def writetofile(self,t_list,t_column):
        n=len(t_list[0])

        for i in range(0, n):
            for j in range(0, self.num_records - 1):
                t_list[j][i] = t_column[i][j]

        target = open(self.output_path, 'w')
        for i in range(0, self.num_records - 1):
            target.write(','.join(map(repr, t_list[i])))
            target.write("\n")


def main(args):
    data_path=args[1]
    output_path=args[2]
    pp=preprocessing(data_path,output_path)
    t_list=pp.readdata()
    t_colum=[]
    t_column=pp.checkfornull(t_list)
    t_colum=pp.standardize(t_list,t_column)
    pp.writetofile(t_list,t_column)
    print("file generated successfully.")
main(sys.argv)
