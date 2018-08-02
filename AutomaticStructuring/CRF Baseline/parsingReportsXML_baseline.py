import xml.etree.cElementTree as Elem
import re

#recursive function to access all descendants of parents
def print_path_of_elems(out,EachReport, elem, elem_path=""):
    for child in elem:
        if not child.getchildren() and child.text:
            # leaf node with text => print
            #print "%s/%s, %s" % (elem_path, child.tag, child.text.strip('\t').strip().strip('\n'))
            EachReport.append((str(elem_path)+"/"+str(child.tag), child.text.strip('\t').strip().strip('\n')))
            if re.search(r'\S',child.tail):
                #print "%s, %s" % (elem_path, child.tail.strip('\t').strip().strip('\n'))
                EachReport.append((elem_path, child.tail.strip('\t').strip().strip('\n')))
        else:
            if re.search(r'\S',child.text):
                #print "%s/%s, %s" % (elem_path, child.tag, child.text.strip('\t').strip().strip('\n'))
                EachReport.append((str(elem_path)+"/"+str(child.tag), child.text.strip('\t').strip().strip('\n')))
            # node with child elements => recurse
            print_path_of_elems(out,EachReport, child, "%s/%s" % (elem_path, child.tag))
            if re.search(r'\S',child.tail):
                #print "%s, %s" % (elem_path, child.tail.strip('\t').strip().strip('\n'))
                EachReport.append((elem_path, child.tail.strip('\t').strip().strip('\n')))
        if child.tag=="report":
            if EachReport!=[]:
                out.write(str(EachReport)+'\n')
                EachReport=[]

#tree = Elem.parse('./../labeling/train_shuffled.xml')
#root=tree.getroot()
#out=open('parsed_labeledreports_fromXML_baseline.txt','w')
#EachReport=[]

#levels=[]
#for level1 in root:
#    level2name=[]
#    level1name=level1.tag
#    print "level1name",level1name

#print_path_of_elems(EachReport, root, root.tag)