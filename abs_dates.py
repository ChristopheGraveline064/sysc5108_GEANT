import datetime
import os 
import csv
import sys

symbol_dst = os.path.join(os.getcwd(),'data.csv')
ctr = 0
dump_file = os.path.join(os.getcwd(),'newdata.csv')

if (os.path.basename(os.path.realpath(symbol_dst)).startswith("seconds_")):
    print ("Absolute times already generated.\n\tExiting...")
    sys.exit()

print ('Parsing XML data from:\n\t' + 'data.csv')
print ('Absolute path is:\n\t' + os.path.join(os.getcwd(),'data.csv'))
print ('Start parsing... ')

with open(symbol_dst) as f:

    first = f.readline().strip()
    r=csv.reader([first])
    list_r=list(r)
    date = list_r[0][0]
    time = list_r[0][1]
    dt = date+" "+time.strip()

    earliest = datetime.datetime.strptime(dt, "%Y-%m-%d %H:%M:%S")
    latest = datetime.datetime.strptime(dt, "%Y-%m-%d %H:%M:%S")

    for line in f:
        if(line.strip()!=''):
            r=csv.reader([line])
            list_r=list(r)
            date = list_r[0][0]
            time = list_r[0][1]
            dt = date+" "+time.strip()

            date = datetime.datetime.strptime(dt, "%Y-%m-%d %H:%M:%S")
            if date < earliest: earliest = date
            if date > latest: latest = date

print ("Earliest date: " + str(earliest))
print ("Latest date: " + str(latest))
print ("Delta seconds = " + str((latest-earliest).total_seconds()))

output = ''

print ("Regenerating data with absolute time in seconds... ")

with open(symbol_dst) as f:
    for line in f:
        if(line.strip()!=''):
            r=csv.reader([line])
            list_r=list(r)
            date = list_r[0][0]
            time = list_r[0][1]
            dt = date+" "+time.strip()
            date = datetime.datetime.strptime(dt, "%Y-%m-%d %H:%M:%S")
            delta = (date - earliest).total_seconds()
            output+=str(delta)+','+str(list_r[0][2])+','+str(list_r[0][3])+','+str(list_r[0][4])+"\n"
            

dump_filename = os.path.basename(os.path.realpath(symbol_dst)).replace("date_time_","seconds_")

print ("Data will be output into output file:\n\t " + dump_filename)

f=open(dump_filename, 'w')
print(output, file=f)

print ("Successfully output.")

symbol_src = os.path.join(os.getcwd(),dump_filename)
symbol_dst = os.path.join(os.getcwd(),'data.csv')

if os.path.exists (symbol_dst):
    print ("Removing old symbolic link...")
    os.remove(symbol_dst)
    print ("\tDone.")

os.symlink(symbol_src,symbol_dst)
print ("Linked the following: \n\t" + symbol_dst + "\n\tto:\n\t" + symbol_src)

def sortcsvfiles(inputfilename,outputfilename):
    with open(inputfilename,'rt') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        data = [row for row in reader if row] # ignore empty lines
        data.sort(key=lambda ro: (float(ro[0])))

    with open(outputfilename,'wt') as csvfile:
        writer=csv.writer(csvfile, lineterminator='\n')
        writer.writerow(header)
        writer.writerows(data)

# Order data.csv traffic data in-order of time
print("Ordering data by timestamp...")
sortcsvfiles('data.csv','data.csv')
print ("Done.")