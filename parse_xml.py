from distutils.command.config import dump_file
import os 
from datetime import datetime
from tarfile import NUL
import xml.etree.ElementTree as ET

dir_tgt = 'traffic-matrices'
day = NUL;
time = NUL;
src = NUL;
dst = NUL;
output = '';
num_files = 0;

print ('Parsing XML data from:\n\t' + dir_tgt)
print ('Absolute path is:\n\t' + os.path.join(os.getcwd(),dir_tgt))
print ('Start parsing... ')
for filename in os.listdir(os.path.join(os.getcwd(),dir_tgt)):

    with open(os.path.join(os.getcwd(),dir_tgt,filename),'r')as f:
        if(num_files%1000==0):
            print("\tParsing file " + str(num_files) +"...")
        num_files+=1

        xml_tgt = ET.parse(f)
        root = xml_tgt.getroot()

        for item in root.findall('./info/date'):
            day = item.text.split("T",1)[0]
            time = item.text.split("T",1)[1]

        for source in root.iter('src'):
            for destination in source.iter('dst'):
                src = str(list(source.attrib.values())[0])
                dst= str(list(destination.attrib.values())[0])
                kbps = destination.text
                output += day + ", " + time + ", " + str(src) + ", " + str(dst) + ", " + kbps + "\n"

print ("Parsed " + str(num_files) + " XML files.\n")

now=datetime.now()
dt_string = now.strftime("%d_%m_%Y_%H-%M-%S")
dump_filename = 'date_time_src_dst_kbps_GENTIME_' + dt_string + '.csv'

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
